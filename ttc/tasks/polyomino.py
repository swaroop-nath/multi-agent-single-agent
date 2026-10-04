"""Frontier-CS polyomino packing (problem 0). The verifier is a local re-implementation of the
Frontier-CS judge (ttc/frontiercs/judge.py) on the 70 pinned test cases. Agents submit as often
as they like through `submit` and see only aggregate feedback; a trial's score is its best
valid submission (Appendix A.4)."""

from __future__ import annotations

import asyncio
import grp
import json
import os
import pwd
import shutil
import stat
import subprocess
import threading
import time
from importlib.resources import files
from pathlib import Path

from aiohttp import web

from .. import prompts
from ..frontiercs.fetch import load_lock, problem_dir, verify
from ..frontiercs.judge import Judge
from .base import AgentState, InfraError, Task

TMP_DIRS = ("/tmp", "/var/tmp", "/dev/shm")


class PolyominoTask(Task):
    name = "polyomino"

    def __init__(self, settings, launcher, work: Path):
        super().__init__(settings, launcher, work)
        self._sessions = [AgentState(i) for i in range(settings.k)]
        self.by_token = {s.token: s for s in self._sessions}
        self.judge: Judge | None = None
        self.t0 = time.time()
        self.ledger: list[dict] = []
        self._ledger_lock = threading.Lock()
        self.best_so_far: list[dict] = []  # valid improvements: {t, score, agent, submission}
        self.agent_best: dict[int, dict] = {}
        self.queue_seconds = 0.0

    @property
    def label(self) -> str:
        return f"polyomino-{self.st.mode_tag}{self.st.k}-{self.st.trial:03d}"

    @property
    def sessions(self):
        return self._sessions

    def retire(self, i: int, reason: str) -> None:
        s = self._sessions[i]
        with s.lock:
            if s.status == "active":
                s.status, s.end_reason = "retired", reason

    def mark_start(self) -> None:
        self.t0 = time.time()

    # preflight / setup ------------------------------------------------------------------------------
    def preflight(self) -> None:
        root = Path(self.st.frontiercs_dir)
        problems = verify(root)
        if problems:
            raise InfraError(3, "Frontier-CS files do not match the lock: " + "; ".join(problems[:5]))
        if self.launcher.drop:
            m = root.stat()
            if m.st_uid != 0 or stat.S_IMODE(m.st_mode) & 0o077:
                raise InfraError(3, f"{root} must be root-owned with mode 0700 (agents could read test data)")
            try:
                pwd.getpwnam(self.st.judge_user)
            except KeyError:
                raise InfraError(3, f"judge user {self.st.judge_user} does not exist")
        try:
            subprocess.run([self.st.judge_compiler, "--version"], capture_output=True, timeout=30, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise InfraError(3, f"compiler {self.st.judge_compiler!r} not runnable: {e}")

    def setup(self) -> None:
        st = self.st
        run_as = None
        if self.launcher.drop:
            pw = pwd.getpwnam(st.judge_user)
            run_as = (pw.pw_uid, pw.pw_gid)
            if st.lock_tmp:
                self._lock_tmp()
        judge_work = self.work / "judge"
        judge_work.mkdir(parents=True, exist_ok=True)
        os.chmod(judge_work, 0o711)  # judge user may enter its own build dirs, list nothing
        self.judge = Judge(st.frontiercs_dir, judge_work, run_as, parallelism=st.judge_parallelism,
                           compiler=st.judge_compiler, extra_cflags=st.judge_extra_cflags,
                           wall_factor=st.judge_wall_factor)
        try:
            prebuilt = Path(st.prebuilt_checker or Path(st.frontiercs_dir) / "chk")  # built into the image
            self.judge.build_checker(prebuilt)
        except (RuntimeError, subprocess.SubprocessError) as e:
            raise InfraError(3, str(e))
        (self.work / "submissions").mkdir()
        os.chmod(self.work / "submissions", 0o700)
        task = self.work / "task"
        if st.judge_compiler != "g++":  # agents compile locally with the same compiler as the judge
            compiler = shutil.which(st.judge_compiler)
            if compiler:
                (task / "bin" / "g++").symlink_to(compiler)
        client = task / "bin" / "submit"
        client.write_text(files("ttc.tasks").joinpath("submit_client.py").read_text())
        client.chmod(0o755)
        statement = (problem_dir(st.frontiercs_dir) / "statement.txt").read_text()
        (task / "problem").mkdir()
        (task / "problem" / "statement.txt").write_text(statement)
        self.launcher.chown_group(task / "problem")
        self.statement = statement

    def _lock_tmp(self) -> None:
        """World-writable dirs would let the judged program leave file *names* for agents to
        read; restrict them to root and the agents' group (the judge user is not a member)."""
        gid = grp.getgrnam(self.st.agent_group).gr_gid
        for d in TMP_DIRS:
            if os.path.isdir(d):
                os.chown(d, 0, gid)
                os.chmod(d, 0o1775)

    # agent-facing -----------------------------------------------------------------------------------
    def add_routes(self, app: web.Application) -> None:
        app.router.add_route("GET", "/poly/{trial}/ping", self._ping)
        app.router.add_route("GET", "/poly/{trial}/best", self._best)
        app.router.add_route("POST", "/poly/{trial}/submit", self._submit)

    def _session_for(self, request: web.Request) -> AgentState:
        if request.match_info["trial"] != self.label:
            raise web.HTTPNotFound(text="unknown trial")
        s = self.by_token.get(request.headers.get("X-Agent-Token", ""))
        if s is None:
            raise web.HTTPForbidden(text="bad agent token")
        return s

    async def _ping(self, request):
        s = self._session_for(request)
        return web.json_response({"ok": True, "status": s.status})

    async def _best(self, request):
        s = self._session_for(request)
        b = self.agent_best.get(s.agent_index)
        return web.json_response({"submission": None, "your_best": b,
                                  "message": "no valid submission yet" if b is None else None})

    async def _submit(self, request):
        s = self._session_for(request)
        if s.status != "active":
            return web.json_response({"submission": None, "message": f"session ended ({s.end_reason})"})
        try:
            body = await request.json()
            source = str(body["source"])
        except (KeyError, TypeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='expected JSON {"source": "..."}')
        if len(source) > 1_000_000:
            return web.json_response({"submission": None, "message": "source larger than 1 MB"})
        with self._ledger_lock:
            sub_id = len(self.ledger) + 1
            self.ledger.append({"submission": sub_id, "agent": s.agent_index, "status": "queued"})
        (self.work / "submissions" / f"{sub_id:05d}.cpp").write_text(source)
        t_submit = time.time()
        result = await asyncio.get_running_loop().run_in_executor(
            None, self.judge.evaluate, source, self.work / "judge" / f"{sub_id:05d}")
        shutil.rmtree(self.work / "judge" / f"{sub_id:05d}", ignore_errors=True)
        t_done = time.time()
        per_case = result.pop("_per_case", None)
        entry = {"submission": sub_id, "agent": s.agent_index, "t_submit": round(t_submit - self.t0, 2),
                 "t_done": round(t_done - self.t0, 2), **result, "per_case": per_case}
        with self._ledger_lock:
            self.ledger[sub_id - 1] = entry
            if result.get("valid"):
                mine = self.agent_best.get(s.agent_index)
                if mine is None or result["score"] > mine["score"]:
                    self.agent_best[s.agent_index] = {"submission": sub_id, "score": result["score"]}
                top = self.best_so_far[-1]["score"] if self.best_so_far else -1.0
                if result["score"] > top:
                    self.best_so_far.append({"t": round(t_done - self.t0, 2), "score": result["score"],
                                             "agent": s.agent_index, "submission": sub_id})
        reply = {k: v for k, v in result.items()}
        reply["submission"] = sub_id
        reply["your_best"] = self.agent_best.get(s.agent_index)
        return web.json_response(reply)

    def task_prompt(self) -> str:
        team = self.st.k if self.st.team_prompt == "paper" else 1  # loose: no team paragraph
        return prompts.polyomino_task_prompt(self.statement, self.st.max_wall_seconds / 3600, team)

    def agent_env(self, i: int, host: str, port: int) -> dict[str, str]:
        return {"SUBMIT_URL": f"http://{host}:{port}/poly/{self.label}", "SUBMIT_TOKEN": self._sessions[i].token}

    async def preflight_agent(self, run_as_agent0) -> dict:
        out: dict = {}
        rc, text = await run_as_agent0(["submit", "--ping", "--json"])
        out["submit_ping_as_agent"] = rc
        if rc != 0 or '"ok": true' not in text:
            raise InfraError(3, f"agent cannot reach the scorer (rc={rc}): {text[:300]}")
        if self.launcher.drop:
            case = problem_dir(self.st.frontiercs_dir) / "testdata" / "1.in"
            rc, _ = await run_as_agent0(["cat", str(case)])
            if rc == 0:
                raise InfraError(3, "isolation broken: agent user can read hidden test data")
            out["test_data_hidden_from_agents"] = True
        return out

    # results -------------------------------------------------------------------------------------------
    def identity(self) -> dict:
        lock = load_lock()
        return {"problem": "frontier-cs/algorithmic/0 (polyomino packing)", "frontiercs_commit": lock["commit"]}

    def outcome(self, timeline):
        done = [e for e in self.ledger if e.get("status") in ("ok", "compile_error")]
        valid = [e for e in done if e.get("valid")]
        best = max(valid, key=lambda e: (e["score"], -e["submission"])) if valid else None
        best_any = max((e.get("score", 0.0) for e in done), default=0.0)
        t_best = (self.t0 + best["t_done"]) if best else None
        outcome = {
            "score": best["score"] if best else 0.0,     # best valid submission (paper, App. A.4)
            "max_score": 1.0,
            "normalized_score": best["score"] if best else 0.0,
            "solved": None,
            "best_submission": ({"submission": best["submission"], "agent": best["agent"],
                                 "t_seconds": best["t_done"], "verdicts": best["verdicts"]} if best else None),
            "best_score_any_submission": best_any,  # official-judge semantics: failed cases count as 0
            "submissions": len(done), "valid_submissions": len(valid),
            "compile_errors": sum(e.get("status") == "compile_error" for e in done),
            "time_to_best_seconds": best["t_done"] if best else None,
            "output_tokens_at_best": sum(t for ts, t in timeline if ts <= t_best) if t_best else None,
            "best_so_far": self.best_so_far,
        }
        agents = []
        for i in range(self.st.k):
            mine = [e for e in done if e["agent"] == i]
            agents.append({"submissions": len(mine), "valid_submissions": sum(bool(e.get("valid")) for e in mine),
                           "best_valid_score": (self.agent_best.get(i) or {}).get("score")})
        lock = load_lock()["problem"]
        extra = {"problem": {"cases": lock["cases"], "time_limit_s": lock["time_limit_s"],
                             "memory_mb": lock["memory_mb"], "compile": lock["compile"],
                             "judge_parallelism": self.st.judge_parallelism}}
        return outcome, agents, extra

    def write_artifacts(self, results: Path) -> None:
        with open(self.work / "ledger.jsonl", "w") as f:
            for e in self.ledger:
                f.write(json.dumps(e) + "\n")
        best = self.best_so_far[-1]["submission"] if self.best_so_far else None
        if best:
            shutil.copy(self.work / "submissions" / f"{best:05d}.cpp", results / "best_solution.cpp")

    def trajectory_files(self) -> dict[str, Path]:
        return {"submissions.jsonl.gz": self.work / "ledger.jsonl",
                "submission_sources.tar.gz": self.work / "submissions"}
