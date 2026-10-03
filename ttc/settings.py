"""Settings for one trial, built from command-line arguments (`run_trial --help`).

Every task parameter is a command-line argument. The only environment inputs are the model
endpoint (OPENAI_BASE_URL or OPENAI_API_BASE) and its key (OPENAI_API_KEY).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field

MODES = ("solo", "solo_rules", "team")
TASKS = ("arc", "polyomino")
DEFAULT_ENVIRONMENTS_DIR = "/opt/arc_envs"
DEFAULT_FRONTIERCS_DIR = "/opt/frontiercs"


@dataclass
class TrialSettings:
    # task identity
    mode: str
    k: int
    trial: int
    task: str = "arc"
    game: str | None = None  # ARC only
    results_dir: str = "/tmp/results"
    max_wall_seconds: float = 12 * 3600
    wall_margin_seconds: float | None = None  # default: max(120 s, 3% of max_wall_seconds)
    work_dir: str = "/tmp/ttc_work"

    # model (endpoint and key come from the environment)
    model_alias: str = "ttc-model"
    context_window: int = 131072
    max_output_tokens: int = 16384
    max_prompt_tokens: int | None = None  # default: context_window - max_output_tokens
    reasoning_effort: str | None = None
    inject_max_tokens: bool = True  # Copilot never sends max_tokens; the proxy adds it
    base_url: str = field(default="", repr=False)
    api_key: str | None = field(default=None, repr=False)

    # model-call resilience
    model_retries: int = 6
    model_backoff_seconds: float = 5.0
    model_backoff_max_seconds: float = 120.0
    model_error_limit: int = 8  # consecutive failed calls before an agent is ended (model_errors)
    sse_keepalive_after_seconds: float = 20.0  # 0 disables keepalive comments to Copilot
    sse_keepalive_interval_seconds: float = 15.0
    upstream_timeout_seconds: float = 3600.0
    skip_model_preflight: bool = False

    # Copilot CLI
    copilot_binary: str = "copilot"
    copilot_extra_args: list[str] = field(default_factory=list)
    max_relaunches: int = 50
    continue_prompt: str = (
        "Your game session is still active and time remains. Re-read the task instructions "
        "(AGENT.md) and continue working. Check `arc info` for your current status."
    )
    copilot_log_level: str = "info"

    # agents' OS identity: "auto" drops to per-agent users when running as root
    agent_users: str = "auto"  # auto | none
    agent_user_prefix: str = "ttc-agent"
    agent_group: str = "ttc-agents"

    # ARC-AGI-3
    environments_dir: str = DEFAULT_ENVIRONMENTS_DIR
    games_lock: str | None = None  # default: the lock file shipped in the package
    game_seed: int = 0
    budget_multiplier: float = 5.0
    sync_barrier: bool = True
    barrier_interval_fraction: float = 0.5
    idle_timeout_seconds: float = 1200.0
    count_reset_as_action: bool = True
    stop_on_first_win: bool = False

    # polyomino packing (Frontier-CS problem 0)
    frontiercs_dir: str = DEFAULT_FRONTIERCS_DIR
    judge_user: str = "ttc-judge"
    judge_compiler: str = "g++"
    judge_extra_cflags: list[str] = field(default_factory=list)
    judge_parallelism: int = 2
    judge_wall_factor: float = 2.0
    prebuilt_checker: str | None = None
    lock_tmp: bool = True

    # local services
    host: str = "127.0.0.1"
    port: int = 8700

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.task == "arc" and not self.game:
            raise ValueError("ARC trials need --game")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == "team" and self.k < 2:
            raise ValueError("team trials need --k >= 2")
        if self.mode != "team" and self.k != 1:
            raise ValueError(f"{self.mode} trials are single-agent; use --k 1")
        if self.max_prompt_tokens is None:
            self.max_prompt_tokens = self.context_window - self.max_output_tokens
        if self.max_prompt_tokens + self.max_output_tokens > self.context_window:
            raise ValueError("max_prompt_tokens + max_output_tokens must fit in context_window")
        if self.wall_margin_seconds is None:
            self.wall_margin_seconds = max(120.0, 0.03 * self.max_wall_seconds)

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        d.pop("base_url")
        return d


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "on"):
        return True
    if s.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def add_trial_arguments(p: argparse.ArgumentParser) -> None:
    d = TrialSettings.__dataclass_fields__
    g = p.add_argument_group("task")
    g.add_argument("--task", choices=TASKS, default="arc")
    g.add_argument("--game", default=None, help="ARC-AGI-3 game id, e.g. lp85 (ARC only)")
    g.add_argument("--mode", required=True, choices=MODES)
    g.add_argument("--k", type=int, required=True, help="number of agents (1 for solo modes)")
    g.add_argument("--trial", type=int, required=True, help="trial index (labelling only)")
    g.add_argument("--results-dir", default=d["results_dir"].default)
    g.add_argument("--max-wall-seconds", type=float, default=d["max_wall_seconds"].default)
    g.add_argument("--wall-margin-seconds", type=float, default=None)
    g.add_argument("--work-dir", default=d["work_dir"].default)

    g = p.add_argument_group("model")
    g.add_argument("--model-alias", default=d["model_alias"].default)
    g.add_argument("--context-window", type=int, default=d["context_window"].default)
    g.add_argument("--max-output-tokens", type=int, default=d["max_output_tokens"].default)
    g.add_argument("--max-prompt-tokens", type=int, default=None)
    g.add_argument("--reasoning-effort", default=None,
                   choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    g.add_argument("--inject-max-tokens", type=_bool, default=True)
    g.add_argument("--model-retries", type=int, default=d["model_retries"].default)
    g.add_argument("--model-backoff-seconds", type=float, default=d["model_backoff_seconds"].default)
    g.add_argument("--model-backoff-max-seconds", type=float, default=d["model_backoff_max_seconds"].default)
    g.add_argument("--model-error-limit", type=int, default=d["model_error_limit"].default)
    g.add_argument("--sse-keepalive-after-seconds", type=float, default=d["sse_keepalive_after_seconds"].default)
    g.add_argument("--sse-keepalive-interval-seconds", type=float,
                   default=d["sse_keepalive_interval_seconds"].default)
    g.add_argument("--upstream-timeout-seconds", type=float, default=d["upstream_timeout_seconds"].default)
    g.add_argument("--skip-model-preflight", action="store_true")

    g = p.add_argument_group("copilot")
    g.add_argument("--copilot-binary", default=d["copilot_binary"].default)
    g.add_argument("--copilot-extra-arg", action="append", dest="copilot_extra_args", default=[],
                   help="extra Copilot CLI argument (repeatable)")
    g.add_argument("--max-relaunches", type=int, default=d["max_relaunches"].default)
    g.add_argument("--copilot-log-level", default=d["copilot_log_level"].default)
    g.add_argument("--agent-users", choices=["auto", "none"], default="auto")

    g = p.add_argument_group("arc")
    g.add_argument("--environments-dir", default=d["environments_dir"].default)
    g.add_argument("--games-lock", default=None)
    g.add_argument("--game-seed", type=int, default=0)
    g.add_argument("--budget-multiplier", type=float, default=d["budget_multiplier"].default)
    g.add_argument("--sync-barrier", type=_bool, default=True)
    g.add_argument("--barrier-interval-fraction", type=float, default=d["barrier_interval_fraction"].default)
    g.add_argument("--idle-timeout-seconds", type=float, default=d["idle_timeout_seconds"].default)
    g.add_argument("--count-reset-as-action", type=_bool, default=True)
    g.add_argument("--stop-on-first-win", type=_bool, default=False)

    g = p.add_argument_group("polyomino")
    g.add_argument("--frontiercs-dir", default=DEFAULT_FRONTIERCS_DIR)
    g.add_argument("--judge-user", default=d["judge_user"].default)
    g.add_argument("--judge-compiler", default=d["judge_compiler"].default)
    g.add_argument("--judge-extra-cflag", action="append", dest="judge_extra_cflags", default=[],
                   help="extra compiler flag for local testing only (repeatable)")
    g.add_argument("--judge-parallelism", type=int, default=d["judge_parallelism"].default,
                   help="test cases judged in parallel")
    g.add_argument("--judge-wall-factor", type=float, default=d["judge_wall_factor"].default)
    g.add_argument("--prebuilt-checker", default=None)
    g.add_argument("--lock-tmp", type=_bool, default=True)

    g = p.add_argument_group("services")
    g.add_argument("--port", type=int, default=d["port"].default, help="local port for game server + proxy")


def settings_from_args(args: argparse.Namespace, environ: dict[str, str] | None = None) -> TrialSettings:
    env = os.environ if environ is None else environ
    base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or ""
    if not base_url:
        raise SystemExit("OPENAI_BASE_URL (or OPENAI_API_BASE) is not set")
    names = {f for f in TrialSettings.__dataclass_fields__} - {"base_url", "api_key"}
    kwargs = {n: getattr(args, n) for n in names if hasattr(args, n)}
    return TrialSettings(**kwargs, base_url=base_url, api_key=env.get("OPENAI_API_KEY"))
