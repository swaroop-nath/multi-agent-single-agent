import json
from pathlib import Path
from types import SimpleNamespace

from ttc.agent_cli import CLAUDE_DISALLOWED_TOOLS, ClaudeCodeCLI


def _cli(**kw):
    st = SimpleNamespace(claude_auth="subscription", claude_binary="claude", model_alias="claude-sonnet-4-6",
                         reasoning_effort="max", continue_prompt="continue", claude_extra_args=[], oauth_token="t")
    for k, v in kw.items():
        setattr(st, k, v)
    return ClaudeCodeCLI(st)


def _stdout(tmp_path: Path, *results: dict) -> Path:
    p = tmp_path / "stdout.jsonl"
    p.write_text("".join(json.dumps({"type": "result", **r}) + "\n" for r in results))
    return p


def test_run_endings_are_classified(tmp_path):
    cli = _cli()
    assert cli.last_outcome(_stdout(tmp_path, {"is_error": False, "result": "done"})) is None
    assert cli.last_outcome(_stdout(tmp_path, {"is_error": True,
                                               "result": "You've hit your session limit · resets 3:45pm"})) == "limit"
    assert cli.last_outcome(_stdout(tmp_path, {"is_error": True, "result": "Prompt is too long"})) == "overflow"
    assert cli.last_outcome(_stdout(tmp_path, {"is_error": True, "result": "Not logged in · Please run /login"})) == "auth"
    assert cli.last_outcome(_stdout(tmp_path, {"is_error": True, "result": "API Error: 500"})) == "error"


def test_argv_isolates_and_resumes(tmp_path):
    cli = _cli()
    first = cli.argv(tmp_path, "sid", 0, "PROMPT")
    assert first[:3] == ["claude", "-p", "PROMPT"] and first[first.index("--session-id") + 1] == "sid"
    assert "--safe-mode" in first and "--dangerously-skip-permissions" in first
    assert first[first.index("--effort") + 1] == "max"
    denied = first[first.index("--disallowedTools") + 1].split(",")
    assert {"WebSearch", "WebFetch", "SendMessage", "Workflow", "RemoteTrigger"} <= set(denied) == set(CLAUDE_DISALLOWED_TOOLS)
    assert json.loads(first[first.index("--settings") + 1])["showThinkingSummaries"] is True
    again = cli.argv(tmp_path, "sid", 1, "PROMPT")
    assert again[2] == "continue" and again[again.index("--resume") + 1] == "sid"


def test_env_keeps_personal_setup_out(tmp_path):
    env = _cli().env(0, tmp_path, "http://proxy")
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude_config")
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "t" and "ANTHROPIC_BASE_URL" not in env
    via_proxy = _cli(claude_auth="api-key", oauth_token=None).env(0, tmp_path, "http://proxy")
    assert via_proxy["ANTHROPIC_BASE_URL"] == "http://proxy" and "CLAUDE_CODE_OAUTH_TOKEN" not in via_proxy


def test_usage_dedupes_messages_and_sums_cost(tmp_path):
    proj = tmp_path / "claude_config" / "projects" / "session"
    proj.mkdir(parents=True)
    u = {"input_tokens": 2, "output_tokens": 50, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 100}
    rows = [{"type": "assistant", "timestamp": "2026-10-04T00:00:01Z", "message": {"id": "m1", "usage": u}},
            {"type": "assistant", "timestamp": "2026-10-04T00:00:01Z", "message": {"id": "m1", "usage": u}},  # same message
            {"type": "assistant", "timestamp": "2026-10-04T00:00:05Z", "message": {"id": "m2", "usage": u}}]
    (proj / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = _cli().usage(tmp_path, _stdout(tmp_path, {"is_error": False, "total_cost_usd": 0.5},
                                         {"is_error": True, "total_cost_usd": 0.25}))
    assert out["requests"] == 2 and out["output_tokens"] == 100 and out["cached_tokens"] == 2000
    assert out["prompt_tokens"] == 2 * (2 + 1000 + 100) and out["cache_write_tokens"] == 200
    assert out["failed"] == 1 and out["reported_cost_usd"] == 0.75 and len(out["timeline"]) == 2


def test_loose_team_prompt_is_minimal():
    from ttc import prompts
    assert prompts.loose_team_prompt(3) == \
        "You are one of 3 agents working on this same task at the same time. Work as a team."
