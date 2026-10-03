"""One trial of a task (ARC-AGI-3 or polyomino packing): solo, solo_rules (solo + the
non-communication rules of A.2), or team@k.

Exit codes (nonzero means an infrastructure failure: the trial should not be scored):
  0  the trial ran and results are written (whatever the score)
  2  the model endpoint is unreachable or failing at startup
  3  harness/task preflight failed (pinned files missing or modified, isolation broken, port in
     use, the agent toolchain cannot reach the task's server)
  4  the trial ran but no model call ever succeeded (endpoint down during the run)
  5  stopped by a signal
  1  unexpected harness error
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import signal
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path

from . import prompts
from .agents import AgentLauncher
from .results import write_json, write_trajectories
from .services import Services
from .settings import TrialSettings
from .tasks import make_task
from .tasks.base import InfraError

log = logging.getLogger("ttc")
POLL_S = 2.0
SCHEMA_VERSION = 2


def _versions(copilot_version: str | None) -> dict:
    def v(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    out = {"harness": v("ttc"), "copilot_cli": copilot_version, "arc_agi": v("arc-agi"),
           "arcengine": v("arcengine"), "aiohttp": v("aiohttp"), "python": platform.python_version()}
    build_info = Path(os.environ.get("TTC_BUILD_INFO", "/app/build_info.json"))
    if build_info.exists():
        try:
            out["build"] = json.loads(build_info.read_text())
        except json.JSONDecodeError:
            pass
    return out


class Trial:
    def __init__(self, st: TrialSettings):
        self.st = st
        self.k = st.k
        self.launcher = AgentLauncher(AgentLauncher.should_drop(st.agent_users), st.agent_user_prefix,
                                      st.agent_group, st.k)
        self.t_start = time.time()
        self.deadline = self.t_start + st.max_wall_seconds - st.wall_margin_seconds
        self.task = make_task(st, self.launcher, Path("/nonexistent"))
        self.label = self.task.label
        self.work = Path(st.work_dir) / self.label
        self.task.work = self.work
        self.results = Path(st.results_dir)
        self.keys = [f"{self.label}.{i}" for i in range(st.k)]
        self.exit_codes: list[list[int | None]] = [[] for _ in range(st.k)]
        self.crashes = [0] * st.k
        self.ended_by = "all_agents_done"
        self.copilot_version: str | None = None
        self.preflight: dict = {}
        self.services: Services | None = None
        self.task_ready = False

    # ---- preflight ---------------------------------------------------------------------------------
    async def _check_model(self) -> None:
        assert self.services is not None
        body = json.dumps({"model": self.st.model_alias, "max_tokens": 1, "stream": False,
                           "messages": [{"role": "user", "content": "Reply with OK."}]}).encode()
        t0 = time.time()
        up, retries = await self.services.fetch(
            "POST", self.services.upstream_url("chat/completions"), body,
            {**self.services.upstream_headers(), "Content-Type": "application/json"})
        self.preflight["model"] = {"status": up.status, "retries": retries, "latency_s": round(time.time() - t0, 1),
                                   "error": up.error}
        if up.status == 599 or up.status >= 500:
            raise InfraError(2, f"model endpoint unreachable/failing: {up.error or up.status}")
        if up.status >= 400:  # reachable, but the probe was refused; record and continue
            log.warning("model preflight returned %s: %s", up.status, up.body[:300])

    async def _check_agent_path(self) -> None:
        env0 = self._agent_env(0)
        cwd = str(self.work / "task")

        async def run_as_agent0(argv: list[str]) -> tuple[int, str]:
            return await self.launcher.run(0, argv, env0, cwd)

        self.preflight.update(await self.task.preflight_agent(run_as_agent0))
        rc, out = await self.launcher.run(0, [self.st.copilot_binary, "--version"], env0, cwd, timeout=120)
        if rc != 0:
            raise InfraError(3, f"copilot CLI not runnable as agent (rc={rc}): {out[:300]}")
        self.copilot_version = out.strip().splitlines()[0] if out.strip() else None

    # ---- setup -----------------------------------------------------------------------------------------
    def _prepare_dirs(self) -> None:
        if self.work.exists():
            shutil.rmtree(self.work)
        for sub in ("task/bin", "task/shared", "task/scratch"):
            (self.work / sub).mkdir(parents=True)
        for i in range(self.k):
            a = self.work / "agents" / str(i)
            for sub in ("home", "copilot_home", "logs", "tmp"):
                (a / sub).mkdir(parents=True)
            (a / "stdout.jsonl").touch()
            self.launcher.chown(i, a)
            if self.launcher.drop:
                os.chmod(a, 0o700)
        os.chmod(self.work, 0o755)
        os.chmod(self.work / "agents", 0o755)
        if self.launcher.drop:  # every ancestor must be traversable by the agent users
            for d in [Path(self.st.work_dir).resolve(), *Path(self.st.work_dir).resolve().parents]:
                if not d.stat().st_mode & 0o001:
                    raise InfraError(3, f"{d} is not traversable by agent users (needs o+x)")

    def _prompt(self) -> str:
        task_root = str(self.work / "task")
        shared = f"{task_root}/shared"
        paths = prompts.SharedPaths(
            slots=f"{shared}/slots", findings=f"{shared}/findings.md",
            disconfirm=f"{shared}/disconfirmations.md", plateau=f"{shared}/score_log.txt",
            coordination=f"{shared}/coordination.md", base=f"{task_root}/scratch")
        task = self.task.task_prompt()
        (self.work / "task" / "AGENT.md").write_text(task)
        if self.st.mode == "team":
            extra = prompts.communication_prompt(self.k, paths)
        elif self.st.mode == "solo_rules":
            extra = prompts.solo_rules_prompt(paths)
        else:
            extra = ""
        prompt = task + ("\n\n" + extra if extra else "")
        (self.work / "prompt.md").write_text(prompt)
        return prompt

    def _agent_env(self, i: int) -> dict[str, str]:
        st = self.st
        a = self.work / "agents" / str(i)
        base_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        if not self.launcher.drop:  # development: keep the host PATH (node, copilot)
            base_path = os.environ.get("PATH", base_path)
        return {
            "PATH": f"{self.work / 'task' / 'bin'}:{base_path}",
            "HOME": str(a / "home"),
            "TMPDIR": str(a / "tmp"),
            "LANG": "C.UTF-8",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "COPILOT_PROVIDER_BASE_URL": f"http://{st.host}:{st.port}/llm/{self.keys[i]}",
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_API_KEY": "ttc-local-proxy",  # the real key never reaches agents
            "COPILOT_PROVIDER_WIRE_API": "completions",
            "COPILOT_MODEL": st.model_alias,
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": str(st.max_prompt_tokens),
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": str(st.max_output_tokens),
            "COPILOT_OFFLINE": "true",
            "COPILOT_AUTO_UPDATE": "false",
            "COPILOT_HOME": str(a / "copilot_home"),
            **self.task.agent_env(i, st.host, st.port),
        }

    def _copilot_argv(self, i: int, session_id: str, attempt: int, prompt: str) -> list[str]:
        st = self.st
        a = self.work / "agents" / str(i)
        argv = [st.copilot_binary]
        if attempt == 0:
            argv += ["--session-id", session_id, "-p", prompt]
        else:
            argv += ["--resume", session_id, "-p", st.continue_prompt]
        argv += ["--allow-all", "--no-ask-user", "--no-auto-update", "--no-custom-instructions",
                 "--disable-builtin-mcps", "--output-format", "json",
                 "--log-dir", str(a / "logs"), "--log-level", st.copilot_log_level,
                 "--usage-output-file", str(a / f"usage-{attempt:03d}.json")]
        if st.reasoning_effort:
            argv += ["--reasoning-effort", st.reasoning_effort]
        return argv + list(st.copilot_extra_args)

    # ---- callbacks from the proxy -------------------------------------------------------------------------
    def _index(self, key: str) -> int:
        return int(key.rsplit(".", 1)[1])

    def on_overflow(self, key: str) -> None:
        self.task.retire(self._index(key), "context_overflow")

    def on_model_failures(self, key: str, consecutive: int) -> None:
        if consecutive >= self.st.model_error_limit:
            self.task.retire(self._index(key), "model_errors")

    # ---- execution -------------------------------------------------------------------------------------------
    async def _agent_loop(self, i: int, prompt: str) -> None:
        s = self.task.sessions[i]
        env = self._agent_env(i)
        session_id = str(uuid.uuid4())
        stdout = self.work / "agents" / str(i) / "stdout.jsonl"
        attempt = 0
        while True:
            proc = await self.launcher.spawn(i, self._copilot_argv(i, session_id, attempt, prompt), env,
                                             str(self.work / "task"), stdout)
            killed = False
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), POLL_S)
                except asyncio.TimeoutError:
                    pass
                if s.status != "active" or time.time() >= self.deadline or self.task.stop_requested():
                    killed = True
                    await self.launcher.kill(proc)
            self.exit_codes[i].append(proc.returncode)
            if not killed and proc.returncode not in (0, None):
                self.crashes[i] += 1
            if s.status != "active":
                return
            if time.time() >= self.deadline:
                self.task.retire(i, "wall_clock")
                return
            if self.task.stop_requested():
                self.task.retire(i, "teammate_won")
                return
            if attempt >= self.st.max_relaunches:
                self.task.retire(i, "agent_exited")
                return
            attempt += 1
            log.info("%s agent %d exited (%s); relaunch %d", self.label, i, proc.returncode, attempt)
            await asyncio.sleep(min(60, 2 * attempt))

    async def run(self) -> int:
        st = self.st
        self.results.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        code = 0
        error: str | None = None
        try:
            self.task.preflight()
            self._prepare_dirs()
            self.task.setup()
            self.task_ready = True
            self.launcher.chown_group(self.work / "task")
            self.services = Services(st, self.work / "model_calls.jsonl", self.on_overflow, self.on_model_failures)
            try:
                await self.services.start(self.task.add_routes)
            except OSError as e:
                raise InfraError(3, f"cannot bind {st.host}:{st.port}: {e}")
            self.services.register(self.keys)
            if not st.skip_model_preflight:
                await self._check_model()
            prompt = self._prompt()
            self.launcher.chown_group(self.work / "task")
            await self._check_agent_path()
            self.task.mark_start()
            log.info("%s: %d agent(s) starting; deadline in %.0fs", self.label, self.k, self.deadline - time.time())
            await asyncio.gather(*(self._agent_loop(i, prompt) for i in range(self.k)))
            if any(s.end_reason == "wall_clock" for s in self.task.sessions):
                self.ended_by = "wall_clock"
            elif self.task.stop_requested():
                self.ended_by = "first_win"
            if sum(self.services.calls[k].ok for k in self.keys) == 0:
                code, error = 4, "no model call succeeded during the trial"
        except InfraError as e:
            code, error = e.code, str(e)
            log.error("%s", e)
        except asyncio.CancelledError:
            code, error, self.ended_by = 5, "stopped by signal", "signal"
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected harness error")
            code, error = 1, f"{type(e).__name__}: {e}"
        finally:
            await self._finalize(code, error)
        return code

    async def _finalize(self, code: int, error: str | None) -> None:
        if self.task_ready:
            reason = "signal" if code == 5 else "trial_ended"
            for i in range(self.k):
                self.task.retire(i, reason)
        await self.launcher.kill_all_owned()
        if self.services:
            await self.services.stop()
        result = self._result(code, error)
        try:
            if self.task_ready and hasattr(self.task, "write_artifacts"):
                self.task.write_artifacts(self.results)
            write_trajectories(self.work, self.results, self.k, self.task.trajectory_files() if self.task_ready else {})
        except Exception as e:  # noqa: BLE001
            result["trajectory_error"] = f"{type(e).__name__}: {e}"
        self.task.close()
        write_json(self.results / "result.json", result)
        if code == 0:
            result["verification"] = self._self_verify()
            write_json(self.results / "result.json", result)
        log.info("%s: wrote %s (exit %d)", self.label, self.results / "result.json", code)

    def _self_verify(self) -> dict:
        """Re-score the written results with the standalone verifier (ttc/verifiers). A mismatch is
        recorded, not fatal: the score stands and a post-run check decides what to do."""
        st = self.st
        remaining = self.t_start + st.max_wall_seconds - time.time()
        needed = 300 if st.task == "polyomino" else 60  # a re-judge vs. a replay of the action log
        if remaining < needed:
            return {"status": "skipped", "reason": f"only {remaining:.0f}s of wall clock left"}
        try:
            from .verifiers import verify_trial
            rep = verify_trial(self.results, environments_dir=st.environments_dir, frontiercs_dir=st.frontiercs_dir,
                               parallelism=st.judge_parallelism, compiler=st.judge_compiler,
                               extra_cflags=st.judge_extra_cflags, judge_user=st.judge_user)
            return {"status": "match" if rep["match"] else "mismatch", "mismatches": rep.get("mismatches", []),
                    "recomputed": rep.get("recomputed")}
        except Exception as e:  # noqa: BLE001
            log.exception("self-verification failed")
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    # ---- result.json ---------------------------------------------------------------------------------------------
    def _result(self, code: int, error: str | None) -> dict:
        st = self.st
        out: dict = {
            "schema_version": SCHEMA_VERSION,
            "trial": {"label": self.label, "task": st.task, "mode": st.mode, "k": st.k, "trial": st.trial,
                      **(self.task.identity() if self.task_ready else {"game": st.game})},
            "status": "ok" if code == 0 else "infra_error",
            "exit_code": code,
            "error": error,
            "started_at": self.t_start,
            "termination": {"ended_by": self.ended_by, "wall_seconds": round(time.time() - self.t_start, 1),
                            "max_wall_seconds": st.max_wall_seconds, "wall_margin_seconds": st.wall_margin_seconds},
            "preflight": self.preflight,
            "versions": _versions(self.copilot_version),
            "config": st.public_dict(),
        }
        if not self.task_ready:
            return out
        calls = self.services.calls if self.services else {}
        timeline = sorted(pt for k in self.keys if k in calls for pt in calls[k].timeline)
        outcome, task_agents, extra = self.task.outcome(timeline)
        agents = []
        for i, s in enumerate(self.task.sessions):
            c = calls.get(self.keys[i])
            agents.append({
                "agent": i,
                "termination_reason": s.end_reason or "trial_ended",
                **task_agents[i],
                "model_calls": c.summary() if c else None,
                "copilot_exit_codes": self.exit_codes[i], "copilot_crashes": self.crashes[i],
                "relaunches": max(0, len(self.exit_codes[i]) - 1),
            })
        total = {f: sum(calls[k].__dict__[f] for k in self.keys if k in calls)
                 for f in ("requests", "ok", "failed", "retries", "context_overflows", "truncated",
                           "prompt_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")}
        out["outcome"] = outcome
        out.update(extra)
        out["agents"] = agents
        out["totals"] = total
        out["degradation"] = {
            "model_call_failure_rate": total["failed"] / total["requests"] if total["requests"] else None,
            "model_call_retries": total["retries"],
            "context_overflows": total["context_overflows"],
            "agents_ended_by_model_errors": sum(a["termination_reason"] == "model_errors" for a in agents),
            "agents_ended_by_context_overflow": sum(a["termination_reason"] == "context_overflow" for a in agents),
            "copilot_crashes": sum(self.crashes),
        }
        out["sampling_params_sent"] = {str(i): calls[k].sampling_params for i, k in enumerate(self.keys) if k in calls}
        return out


def run(st: TrialSettings) -> int:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    trial = Trial(st)
    task = loop.create_task(trial.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return loop.run_until_complete(task)
    except asyncio.CancelledError:
        return 5
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit("use `run_trial` / `python -m ttc.cli trial`")
