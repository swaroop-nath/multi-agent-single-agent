"""ARC-AGI-3 task: one local game session per agent; the verifier is the game engine itself
(level completion), which agents can query at any time through `arc info`."""

from __future__ import annotations

import asyncio
import json
import stat
import time
from importlib.resources import files
from pathlib import Path

from aiohttp import web

from .. import prompts
from ..analysis import rhae
from ..arc.download import game_dir, load_lock, verify
from ..arc.game import TrialGame, load_arc_game
from .base import InfraError, Task


class ArcTask(Task):
    name = "arc"

    def __init__(self, settings, launcher, work: Path):
        super().__init__(settings, launcher, work)
        self.game: TrialGame | None = None
        self.game_id: str | None = None

    @property
    def label(self) -> str:
        return f"{self.st.game}-{self.st.mode_tag}{self.st.k}-{self.st.trial:03d}"

    def preflight(self) -> None:
        st = self.st
        lock = load_lock(st.games_lock)["games"]
        if st.game not in lock:
            raise InfraError(3, f"unknown game {st.game!r}; known: {sorted(lock)}")
        problems = verify(st.environments_dir, [st.game], st.games_lock)
        if problems:
            raise InfraError(3, "game files do not match the lock: " + "; ".join(problems))
        self.game_id = lock[st.game]["game_id"]
        if self.launcher.drop:
            root = Path(st.environments_dir).stat()
            if root.st_uid != 0 or stat.S_IMODE(root.st_mode) & 0o077:
                raise InfraError(3, f"{st.environments_dir} must be root-owned with mode 0700")

    def setup(self) -> None:
        st = self.st
        _, baselines, make_env, action_from_id = load_arc_game(st.environments_dir, self.game_id, st.game_seed)
        self.game = TrialGame(
            self.label, self.game_id, baselines, st.k, make_env, action_from_id,
            multiplier=st.budget_multiplier, sync_barrier=st.sync_barrier and st.mode == "team",
            interval_fraction=st.barrier_interval_fraction, idle_timeout_s=st.idle_timeout_seconds,
            count_reset_as_action=st.count_reset_as_action, events_path=self.work / "events.jsonl")
        arc_bin = self.work / "task" / "bin" / "arc"
        arc_bin.write_text(files("ttc.arc").joinpath("arc_client.py").read_text())
        arc_bin.chmod(0o755)

    @property
    def sessions(self):
        return self.game.sessions

    def retire(self, i: int, reason: str) -> None:
        self.game.retire(self.game.sessions[i], reason)

    def stop_requested(self) -> bool:
        return self.st.stop_on_first_win and self.game.any_won()

    def mark_start(self) -> None:
        self.game.t0 = time.time()

    def progress(self, i: int) -> str:
        a = self.game.summary()["agents"][i]
        return f"level {a['max_level']} of {self.game.win_levels}, {a['total_actions']} actions"

    def close(self) -> None:
        if self.game:
            self.game.close()

    # agent-facing ------------------------------------------------------------------------------------
    def add_routes(self, app: web.Application) -> None:
        app.router.add_route("GET", "/arc/{trial}/info", self._info)
        app.router.add_route("GET", "/arc/{trial}/state", self._state)
        app.router.add_route("POST", "/arc/{trial}/act", self._act)

    def _session_for(self, request: web.Request):
        if request.match_info["trial"] != self.label or self.game is None:
            raise web.HTTPNotFound(text="unknown trial")
        s = self.game.by_token.get(request.headers.get("X-Agent-Token", ""))
        if s is None:
            raise web.HTTPForbidden(text="bad agent token")
        return s

    async def _info(self, request):
        return web.json_response(self.game.info(self._session_for(request)))

    async def _state(self, request):
        s = self._session_for(request)
        return web.json_response(self.game.state(s, all_frames=request.query.get("all_frames") == "1"))

    async def _act(self, request):
        s = self._session_for(request)
        try:
            body = await request.json()
            action = int(body["action"])
            x = int(body["x"]) if body.get("x") is not None else None
            y = int(body["y"]) if body.get("y") is not None else None
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text='expected JSON {"action": int, "x"?: int, "y"?: int}')
        all_frames = request.query.get("all_frames") == "1"
        out = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.game.act(s, action, x, y, all_frames))
        return web.json_response(out)

    def task_prompt(self) -> str:
        team = self.st.k if self.st.team_prompt == "paper" else 1  # loose/shared-file: no team paragraph
        return prompts.arc_task_prompt(self.st.game, self.game.win_levels, self.st.budget_multiplier, team)

    def agent_env(self, i: int, host: str, port: int) -> dict[str, str]:
        return {"ARC_SERVER_URL": f"http://{host}:{port}/arc/{self.label}",
                "ARC_AGENT_TOKEN": self.game.sessions[i].token}

    async def preflight_agent(self, run_as_agent0) -> dict:
        out: dict = {}
        rc, text = await run_as_agent0(["arc", "info"])
        out["arc_info_as_agent"] = rc
        if rc != 0 or "status=active" not in text:
            raise InfraError(3, f"agent cannot reach the game server (rc={rc}): {text[:300]}")
        if self.launcher.drop:
            src = next(game_dir(self.st.environments_dir, self.game_id).glob("*.py"))
            rc, _ = await run_as_agent0(["cat", str(src)])
            if rc == 0:
                raise InfraError(3, "isolation broken: agent user can read game source")
            out["game_source_hidden_from_agents"] = True
        return out

    # results ------------------------------------------------------------------------------------------
    def identity(self) -> dict:
        return {"game": self.st.game, "game_id": self.game_id, "game_seed": self.st.game_seed}

    def outcome(self, timeline):
        g = self.game
        summary = g.summary()
        agents, solve_t = [], None
        for a, s in zip(summary["agents"], g.sessions):
            if s.status == "finished" and s.level_log:
                t = g.t0 + s.level_log[-1]["t"]
                solve_t = t if solve_t is None else min(solve_t, t)
            agents.append({"max_level": a["max_level"], "total_actions": a["total_actions"],
                           "refused_actions": a["refused_actions"], "actions_per_level": a["actions_per_level"],
                           "level_log": a["level_log"], "rhae": rhae(a, summary["baselines"])})
        rhaes = [a["rhae"] for a in agents]
        outcome = {
            "score": summary["team_max_level"],          # levels cleared by the best agent
            "max_score": summary["win_levels"],
            "normalized_score": summary["team_max_level"] / summary["win_levels"] if summary["win_levels"] else 0.0,
            "solved": summary["solved"],
            "rhae_best_agent": max(rhaes), "rhae_mean_agent": sum(rhaes) / len(rhaes),
            "first_solve_seconds": round(solve_t - g.t0, 1) if solve_t else None,
            "output_tokens_at_first_solve": sum(t for ts, t in timeline if ts <= solve_t) if solve_t else None,
        }
        extra = {"game": {"win_levels": summary["win_levels"], "baseline_actions": summary["baselines"],
                          "level_budgets": summary["budgets"]}}
        return outcome, agents, extra

    def trajectory_files(self) -> dict[str, Path]:
        return {"game_events.jsonl.gz": self.work / "events.jsonl"}
