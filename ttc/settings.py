"""Settings for one trial, built from command-line arguments (`run_trial --help`).

Every task parameter is a command-line argument. The only environment inputs are credentials:
  --agent-cli claude-code --claude-auth subscription
                        CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`)
and, for Copilot or `--claude-auth api-key`, the model endpoint and its key:
  --provider openai     OPENAI_BASE_URL (or OPENAI_API_BASE) and OPENAI_API_KEY
  --provider anthropic  ANTHROPIC_API_KEY, and optionally ANTHROPIC_BASE_URL
                        (default https://api.anthropic.com)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field

MODES = ("solo", "solo_rules", "team")
TEAM_PROMPTS = ("paper", "loose", "shared-file")
PROVIDERS = ("openai", "anthropic")
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
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
    team_prompt: str = "paper"  # paper: Appendix A.2 protocol | loose: "Work as a team." only |
                                # shared-file: loose + one shared file the team decides how to use
    game: str | None = None  # ARC only
    results_dir: str = "/tmp/results"
    max_wall_seconds: float = 12 * 3600
    wall_margin_seconds: float | None = None  # default: max(120 s, 3% of max_wall_seconds)
    work_dir: str = "/tmp/ttc_work"

    # model (endpoint and key come from the environment)
    provider: str = "openai"  # openai (chat completions) | anthropic (Messages API)
    model_alias: str = "ttc-model"  # the model name sent upstream; for anthropic, a real model id
    context_window: int = 131072
    max_output_tokens: int = 16384
    max_prompt_tokens: int | None = None  # default: context_window - max_output_tokens
    reasoning_effort: str | None = None
    inject_max_tokens: bool = True  # openai: Copilot never sends max_tokens, so the proxy adds it
    # optional USD prices per million tokens, for a cost estimate in result.json
    price_input: float | None = None
    price_output: float | None = None
    price_cache_read: float | None = None
    price_cache_write: float | None = None
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
    log_model_responses: bool = True  # save every raw model response (incl. thinking)

    # agent CLI: copilot (the paper's harness) or claude-code
    agent_cli: str = "copilot"
    claude_binary: str = "claude"
    claude_auth: str = "subscription"  # subscription (CLAUDE_CODE_OAUTH_TOKEN) | api-key (via the proxy)
    claude_extra_args: list[str] = field(default_factory=list)
    limit_wait_seconds: float = 600.0  # subscription usage limit hit: wait this long, then resume
    extra_agent_env: list[str] = field(default_factory=list)  # KEY=VALUE pairs (testing aid)
    oauth_token: str | None = field(default=None, repr=False)

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
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        if self.agent_cli not in ("copilot", "claude-code"):
            raise ValueError("agent_cli must be copilot or claude-code")
        if self.agent_cli == "claude-code":
            if self.claude_auth not in ("subscription", "api-key"):
                raise ValueError("claude_auth must be subscription or api-key")
            if self.claude_auth == "api-key" and self.provider != "anthropic":
                raise ValueError("--agent-cli claude-code --claude-auth api-key needs --provider anthropic")
            if self.model_alias == "ttc-model":
                raise ValueError("Claude Code needs a real model id, e.g. --model-alias claude-sonnet-4-6")
        if self.provider == "anthropic" and self.model_alias == "ttc-model":
            raise ValueError("--provider anthropic needs a real model id, e.g. --model-alias claude-sonnet-4-6")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.task == "arc" and not self.game:
            raise ValueError("ARC trials need --game")
        if self.team_prompt not in TEAM_PROMPTS:
            raise ValueError(f"team_prompt must be one of {TEAM_PROMPTS}")
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

    @property
    def mode_tag(self) -> str:
        """Mode as used in labels: loose-protocol teams are labelled separately from paper teams."""
        if self.mode != "team" or self.team_prompt == "paper":
            return self.mode
        return {"loose": "teamloose", "shared-file": "teamfile"}[self.team_prompt]

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        d.pop("base_url")
        d.pop("oauth_token")
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
    g.add_argument("--team-prompt", choices=list(TEAM_PROMPTS), default="paper",
                   help="team mode only. paper: the Appendix A.2 communication protocol; loose: only "
                        "'You are one of N agents working on this same task at the same time. Work as a team.'; "
                        "shared-file: loose plus one shared file the team decides how to use")
    g.add_argument("--trial", type=int, required=True, help="trial index (labelling only)")
    g.add_argument("--results-dir", default=d["results_dir"].default)
    g.add_argument("--max-wall-seconds", type=float, default=d["max_wall_seconds"].default)
    g.add_argument("--wall-margin-seconds", type=float, default=None)
    g.add_argument("--work-dir", default=d["work_dir"].default)

    g = p.add_argument_group("model")
    g.add_argument("--provider", choices=PROVIDERS, default="openai",
                   help="openai: OpenAI-compatible chat completions; anthropic: Anthropic Messages API")
    g.add_argument("--model-alias", default=d["model_alias"].default,
                   help="model name sent upstream (anthropic: a model id such as claude-sonnet-4-6)")
    g.add_argument("--price-input", type=float, default=None, help="USD per million uncached input tokens")
    g.add_argument("--price-output", type=float, default=None, help="USD per million output tokens")
    g.add_argument("--price-cache-read", type=float, default=None, help="USD per million cache-read tokens")
    g.add_argument("--price-cache-write", type=float, default=None, help="USD per million cache-write tokens")
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
    g.add_argument("--log-model-responses", type=_bool, default=True,
                   help="save every raw model response, incl. thinking, to trajectories/")

    g = p.add_argument_group("agent CLI")
    g.add_argument("--agent-cli", choices=["copilot", "claude-code"], default="copilot",
                   help="the coding agent each agent runs: copilot (the paper's) or claude-code")
    g.add_argument("--claude-binary", default=d["claude_binary"].default)
    g.add_argument("--claude-auth", choices=["subscription", "api-key"], default="subscription",
                   help="subscription: CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`; "
                        "api-key: ANTHROPIC_API_KEY through the local proxy (needs --provider anthropic)")
    g.add_argument("--claude-extra-arg", action="append", dest="claude_extra_args", default=[])
    g.add_argument("--limit-wait-seconds", type=float, default=d["limit_wait_seconds"].default,
                   help="when a subscription usage limit is hit, wait this long before resuming")
    g.add_argument("--extra-agent-env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for every agent (repeatable; e.g. to point a CLI at a stub)")
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
    if getattr(args, "agent_cli", "copilot") == "claude-code" and getattr(args, "claude_auth", "") == "subscription":
        token = env.get("CLAUDE_CODE_OAUTH_TOKEN")
        if not token:
            raise SystemExit("CLAUDE_CODE_OAUTH_TOKEN is not set (create one with `claude setup-token`)")
        names = {f for f in TrialSettings.__dataclass_fields__} - {"base_url", "api_key", "oauth_token"}
        kwargs = {n: getattr(args, n) for n in names if hasattr(args, n)}
        return TrialSettings(**kwargs, base_url="", api_key=None, oauth_token=token)
    if getattr(args, "provider", "openai") == "anthropic":
        base_url = env.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
        api_key = env.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set")
    else:
        base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or ""
        api_key = env.get("OPENAI_API_KEY")
        if not base_url:
            raise SystemExit("OPENAI_BASE_URL (or OPENAI_API_BASE) is not set")
    names = {f for f in TrialSettings.__dataclass_fields__} - {"base_url", "api_key", "oauth_token"}
    kwargs = {n: getattr(args, n) for n in names if hasattr(args, n)}
    return TrialSettings(**kwargs, base_url=base_url, api_key=api_key)
