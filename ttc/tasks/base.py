"""What a task provides to the generic trial runner (ttc/trial.py).

The runner owns everything task-independent: agents, prompts with the A.2 communication rules,
the model proxy, deadlines, relaunches, results. A task owns its environment and its verifier:
the files agents see, the local HTTP routes their tools call, when an agent is done, and how a
trial is scored.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web


class InfraError(Exception):
    """Preflight/harness failure: the trial must exit nonzero (not be scored)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class AgentState:
    """Generic per-agent lifecycle for tasks without their own session object."""

    agent_index: int
    token: str = field(default_factory=lambda: secrets.token_urlsafe(18))
    status: str = "active"  # active | finished | exhausted | retired
    end_reason: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class Task:
    name: str = ""
    # Shown in result.json and used for labels/manifests.

    def __init__(self, settings, launcher, work: Path):
        self.st = settings
        self.launcher = launcher
        self.work = work

    # identity -------------------------------------------------------------------------------------
    @property
    def label(self) -> str:
        raise NotImplementedError

    # lifecycle ------------------------------------------------------------------------------------
    def preflight(self) -> None:
        """Verify pinned files and isolation before anything starts (raise InfraError)."""

    def setup(self) -> None:
        """Create sessions and task-specific files under work/task (after the directories exist)."""

    @property
    def sessions(self) -> list:
        raise NotImplementedError

    def retire(self, i: int, reason: str) -> None:
        raise NotImplementedError

    def stop_requested(self) -> bool:
        """True when the whole trial should stop early (e.g. stop on first win)."""
        return False

    def mark_start(self) -> None:
        """Called right before agents launch; times are measured from here."""

    def progress(self, i: int) -> str:
        """A one-line summary of agent i's own progress, for the facilitator (ttc/facilitator.py)."""
        return ""

    def close(self) -> None:
        pass

    # agent-facing ---------------------------------------------------------------------------------------
    def add_routes(self, app: web.Application) -> None:
        raise NotImplementedError

    def task_prompt(self) -> str:
        raise NotImplementedError

    def agent_env(self, i: int, host: str, port: int) -> dict[str, str]:
        raise NotImplementedError

    async def preflight_agent(self, run_as_agent0) -> dict:
        """Run the agent toolchain once as agent 0 (raise InfraError on failure)."""
        return {}

    # results ----------------------------------------------------------------------------------------------
    def identity(self) -> dict:
        raise NotImplementedError

    def outcome(self, timeline: list[tuple[float, int]]) -> tuple[dict, list[dict], dict]:
        """(outcome, per-agent task fields, extra top-level sections) for result.json.
        `timeline` is the trial's (time, output_tokens) per model call, for tokens-at-X metrics."""
        raise NotImplementedError

    def trajectory_files(self) -> dict[str, Path]:
        """Extra files to gzip into results/trajectories: {name: source path}."""
        return {}
