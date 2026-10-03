"""Per-trial ARC-AGI-3 game state: one game session per agent, per-level action budgets,
and the team synchronization barrier from Appendix A.3.

Budget rule: level L (0-based index = levels_completed when the action is taken) allows
ceil(multiplier * baseline[L]) charged actions. The counter is keyed by level index and is
cumulative, so a full game reset cannot be used to buy fresh budget for a level. An agent
that reaches the cap on a level without clearing it is terminated ("exhausted").

Barrier rule (teams only): with interval(L) = ceil(fraction * budget(L)), an agent's phase is
(L, used[L] // interval(L)). An agent whose phase is ahead of the slowest *live* teammate
(lexicographically: a higher level, or the same level with more intervals used) has its
game actions refused; refused actions are not executed and not charged. Finished, exhausted,
and retired agents are not live, and neither is an agent that has made no request for
`idle_timeout_s`.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
                5: "ACTION5", 6: "ACTION6", 7: "ACTION7"}
HEX = "0123456789abcdef"


class GameEnv(Protocol):
    """The subset of arc_agi.EnvironmentWrapper we use."""

    def reset(self) -> Any: ...
    def step(self, action: Any, data: dict | None = None) -> Any: ...


@dataclass
class Frame:
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    grids: list[list[str]]  # each grid: list of 64 hex strings

    def sha256(self) -> str:
        """Hash of the full observable state after an action (all frames, state, levels)."""
        h = hashlib.sha256(f"{self.state}|{self.levels_completed}|{self.available_actions}".encode())
        for g in self.grids:
            h.update(("\n".join(g) + "\n\n").encode())
        return h.hexdigest()

    @classmethod
    def from_raw(cls, raw: Any) -> "Frame":
        grids = []
        for g in raw.frame or []:
            grids.append(["".join(HEX[int(v)] if 0 <= int(v) < 16 else "?" for v in row) for row in g])
        state = raw.state.value if hasattr(raw.state, "value") else str(raw.state)
        return cls(state=state, levels_completed=int(raw.levels_completed),
                   win_levels=int(raw.win_levels), available_actions=list(raw.available_actions),
                   grids=grids)


@dataclass
class AgentSession:
    agent_index: int
    token: str
    env: GameEnv
    frame: Frame
    status: str = "active"  # active | finished | exhausted | retired
    used: dict[int, int] = field(default_factory=dict)  # level index -> charged actions
    total_actions: int = 0
    refused_actions: int = 0
    max_level: int = 0
    level_log: list[dict] = field(default_factory=list)  # first completion of each level
    last_seen: float = field(default_factory=time.time)
    end_reason: str | None = None  # finished | budget_exhausted | <retire code>
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def level(self) -> int:
        return self.frame.levels_completed


class TrialGame:
    def __init__(
        self,
        trial_id: str,
        game_id: str,
        baselines: list[int],
        k: int,
        make_env: Callable[[], GameEnv],
        action_from_id: Callable[[int], Any],
        *,
        multiplier: float = 5.0,
        sync_barrier: bool = True,
        interval_fraction: float = 0.5,
        idle_timeout_s: float = 1200.0,
        count_reset_as_action: bool = True,
        events_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.trial_id = trial_id
        self.game_id = game_id
        self.baselines = baselines
        self.multiplier = multiplier
        self.sync_barrier = sync_barrier and k > 1
        self.interval_fraction = interval_fraction
        self.idle_timeout_s = idle_timeout_s
        self.count_reset_as_action = count_reset_as_action
        self.action_from_id = action_from_id
        self.clock = clock
        self.t0 = clock()
        self._barrier_lock = threading.Lock()
        self._events = open(events_path, "a") if events_path else None
        self.sessions: list[AgentSession] = []
        for i in range(k):
            env = make_env()
            raw = env.reset()
            if raw is None:
                raise RuntimeError(f"failed to reset {game_id}")
            self.sessions.append(AgentSession(i, secrets.token_urlsafe(18), env, Frame.from_raw(raw),
                                              last_seen=clock()))
        self.by_token = {s.token: s for s in self.sessions}
        self.win_levels = self.sessions[0].frame.win_levels

    # ---- budgets and barrier -------------------------------------------------------------
    def budget(self, level: int) -> int:
        base = self.baselines[level] if level < len(self.baselines) else max(self.baselines or [100])
        return max(1, math.ceil(self.multiplier * base))

    def interval(self, level: int) -> int:
        return max(1, math.ceil(self.interval_fraction * self.budget(level)))

    def phase(self, s: AgentSession) -> tuple[int, int]:
        return (s.level, s.used.get(s.level, 0) // self.interval(s.level))

    def is_live(self, s: AgentSession, now: float) -> bool:
        return s.status == "active" and now - s.last_seen <= self.idle_timeout_s

    def blocked(self, s: AgentSession, now: float | None = None) -> bool:
        if not self.sync_barrier:
            return False
        now = self.clock() if now is None else now
        others = [p for p in self.sessions if p is not s and self.is_live(p, now)]
        if not others:
            return False
        return self.phase(s) > min(self.phase(p) for p in others)

    # ---- agent-facing operations -----------------------------------------------------------
    def info(self, s: AgentSession) -> dict:
        s.last_seen = self.clock()
        lvl = s.level
        used = s.used.get(lvl, 0)
        out = {
            "game": self.game_id,
            "status": s.status,
            "state": s.frame.state,
            "levels_completed": s.frame.levels_completed,
            "win_levels": s.frame.win_levels,
            "available_actions": s.frame.available_actions,
            "level_budget": self.budget(lvl) if s.status == "active" else None,
            "level_actions_used": used,
            "level_actions_left": max(0, self.budget(lvl) - used) if s.status == "active" else 0,
            "total_actions": s.total_actions,
        }
        if self.sync_barrier:
            out["waiting_for_teammates"] = s.status == "active" and self.blocked(s)
        if s.end_reason:
            out["end_reason"] = s.end_reason
        return out

    def state(self, s: AgentSession, all_frames: bool = False) -> dict:
        out = self.info(s)
        out["frames"] = s.frame.grids if all_frames else s.frame.grids[-1:]
        out["num_frames"] = len(s.frame.grids)
        return out

    def act(self, s: AgentSession, action_id: int, x: int | None = None, y: int | None = None,
            all_frames: bool = False) -> dict:
        with s.lock:
            now = self.clock()
            s.last_seen = now
            if s.status != "active":
                return {**self.info(s), "result": "ENDED",
                        "message": f"Your game session has ended ({s.end_reason or s.status})."}
            if action_id not in ACTION_NAMES:
                return {**self.info(s), "result": "INVALID", "message": f"Unknown action {action_id}."}
            if action_id != 0 and action_id not in s.frame.available_actions:
                return {**self.info(s), "result": "INVALID",
                        "message": f"Action {action_id} is not available; available: {s.frame.available_actions}."}
            if action_id == 6 and (x is None or y is None or not (0 <= x <= 63 and 0 <= y <= 63)):
                return {**self.info(s), "result": "INVALID", "message": "Action 6 needs x and y in 0..63."}

            with self._barrier_lock:
                if self.blocked(s, now):
                    s.refused_actions += 1
                    self._log("refused", s, action=action_id)
                    return {**self.info(s), "result": "WAITING",
                            "message": "WAITING: you are ahead of a teammate. This action was not "
                                       "executed and not charged. Keep working on notes/analysis and retry later."}
                level_before = s.level
                data = {"x": x, "y": y} if action_id == 6 else None
                raw = s.env.step(self.action_from_id(action_id), data)
                if raw is None:
                    return {**self.info(s), "result": "ERROR", "message": "The game engine rejected the action."}
                charged = action_id != 0 or self.count_reset_as_action
                if charged:
                    s.used[level_before] = s.used.get(level_before, 0) + 1
                    s.total_actions += 1
                s.frame = Frame.from_raw(raw)
                self._after_step(s, now)
                self._log("action", s, action=action_id, x=x, y=y, level_before=level_before,
                          frame_sha256=s.frame.sha256(),
                          charged=charged)
            out = self.state(s, all_frames)
            out["result"] = "OK"
            return out

    def _after_step(self, s: AgentSession, now: float) -> None:
        lvl = s.level
        if lvl > s.max_level:
            for done in range(s.max_level, lvl):
                s.level_log.append({"level": done + 1, "t": now - self.t0,
                                    "level_actions": s.used.get(done, 0), "total_actions": s.total_actions})
            s.max_level = lvl
        if s.frame.state == "WIN" or (s.frame.win_levels and lvl >= s.frame.win_levels):
            self._end(s, "finished", "finished")
        elif s.used.get(lvl, 0) >= self.budget(lvl):
            self._end(s, "exhausted", "budget_exhausted")

    def _end(self, s: AgentSession, status: str, reason: str) -> None:
        s.status = status
        s.end_reason = reason
        self._log("end", s, status=status, reason=reason)

    def retire(self, s: AgentSession, reason: str) -> None:
        """End an active session; `reason` is a termination code such as wall_clock."""
        with s.lock:
            if s.status == "active":
                self._end(s, "retired", reason)

    # ---- bookkeeping -------------------------------------------------------------------------
    def all_done(self) -> bool:
        return all(s.status != "active" for s in self.sessions)

    def any_won(self) -> bool:
        return any(s.status == "finished" for s in self.sessions)

    def summary(self) -> dict:
        return {
            "game": self.game_id,
            "win_levels": self.win_levels,
            "baselines": self.baselines,
            "budgets": [self.budget(i) for i in range(len(self.baselines))],
            "team_max_level": max(s.max_level for s in self.sessions),
            "solved": self.any_won(),
            "agents": [
                {"agent": s.agent_index, "status": s.status, "end_reason": s.end_reason,
                 "max_level": s.max_level, "total_actions": s.total_actions,
                 "refused_actions": s.refused_actions, "level_log": s.level_log,
                 "actions_per_level": {str(k): v for k, v in sorted(s.used.items())}}
                for s in self.sessions
            ],
        }

    def _log(self, kind: str, s: AgentSession, **kw: Any) -> None:
        if self._events:
            rec = {"t": round(self.clock() - self.t0, 3), "kind": kind, "agent": s.agent_index,
                   "level": s.level, "state": s.frame.state, **kw}
            self._events.write(json.dumps(rec) + "\n")
            self._events.flush()

    def close(self) -> None:
        if self._events:
            self._events.close()
            self._events = None


def toolkit_logger():
    """The arc-agi toolkit logs every game load at INFO unless given its own logger."""
    import logging
    lg = logging.getLogger("ttc.arc_toolkit")
    lg.setLevel(logging.WARNING)
    for name in ("arc_agi", "arc_agi.scorecard"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return lg


def load_arc_game(environments_dir: str, game: str, seed: int = 0):
    """Return (game_id, baselines, make_env, action_from_id) using the local game cache."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir,
                    logger=toolkit_logger())
    envs = arcade.get_environments()
    exact = [e for e in envs if e.game_id == game]
    infos = exact or [e for e in envs if e.game_id.split("-", 1)[0] == game.split("-", 1)[0]]
    if not infos:
        raise FileNotFoundError(f"game {game} not in {environments_dir}; run `ttc download` first")
    info = infos[0]
    baselines = list(info.baseline_actions or [])

    def make_env():
        env = arcade.make(info.game_id, seed=seed)
        if env is None:
            raise RuntimeError(f"could not load {info.game_id}")
        return env

    return info.game_id, baselines, make_env, GameAction.from_id
