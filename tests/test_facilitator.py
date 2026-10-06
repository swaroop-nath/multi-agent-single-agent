import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ttc.agent_cli import ClaudeCodeCLI
from ttc.facilitator import SlotTracker, parse_reminders
from ttc.settings import POLYOMINO_CONTINUE_PROMPT, TrialSettings
from ttc.trial import Trial

FCS = os.environ.get("TTC_FRONTIERCS_DIR", "/opt/frontiercs")


def _st(**kw):
    base = dict(mode="team", k=2, trial=0, task="polyomino", agent_cli="claude-code",
                model_alias="claude-sonnet-4-6", facilitator=True)
    return TrialSettings(**{**base, **kw})


def test_facilitator_needs_a_paper_team_on_claude_code():
    assert _st().mode_tag == "teamfac"
    for bad in (dict(team_prompt="loose"), dict(mode="solo", k=1), dict(agent_cli="copilot", model_alias="ttc-model")):
        with pytest.raises(ValueError):
            _st(**bad)


def test_polyomino_relaunch_prompt_is_not_about_arc():
    assert _st().continue_prompt == POLYOMINO_CONTINUE_PROMPT
    assert "arc info" in TrialSettings(mode="solo", k=1, trial=0, game="ls20").continue_prompt


def test_streaming_argv_sends_the_prompt_on_stdin(tmp_path):
    cli = ClaudeCodeCLI(_st())
    argv = cli.argv(tmp_path, "sid", 0, "the prompt", streaming=True)
    assert argv[:4] == ["claude", "-p", "--input-format", "stream-json"] and "the prompt" not in argv
    msg = json.loads(ClaudeCodeCLI.user_message("hi"))
    assert msg == {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}


def test_slots_are_inferred_from_scratch_paths(tmp_path):
    def call(cmd):
        return json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}) + "\n"
    a0, a1 = tmp_path / "a0.jsonl", tmp_path / "a1.jsonl"
    a0.write_text(call("cd /w/task/scratch/work-1 && g++ x.cpp") + call("ls /w/task/scratch/work-0"))
    a1.write_text("")
    tracker = SlotTracker([a0, a1])
    assert tracker.update() == [1, None]
    with open(a0, "a") as f:
        f.write(call("cat /w/task/scratch/work-1/notes"))
    with open(a1, "a") as f:
        f.write(call("mkdir -p /w/task/scratch/work-0"))
    assert tracker.update() == [1, 0]


def test_reminders_go_to_the_slot_owner():
    out = json.dumps({"structured_output": {"reminders": [{"slot": 1, "message": " a "}, {"slot": 0, "message": "b"},
                                                          {"slot": 7, "message": "x"}, {"slot": 0, "message": ""}]}})
    assert parse_reminders(out, 2, [1, 0]) == {0: "a", 1: "b"}
    # by agent index only while that agent's slot is unknown
    out = json.dumps({"result": json.dumps({"reminders": [{"agent": 1, "message": "c"}, {"agent": 0, "message": "d"},
                                                          {"agent": 5, "message": "e"}]})})
    assert parse_reminders("noise\n" + out, 2, [1, None]) == {1: "c"}
    assert parse_reminders("not json", 2, [0, 1]) == {}


def test_turn_end_is_detected_from_new_lines_only(tmp_path):
    out = tmp_path / "stdout.jsonl"
    out.write_text('{"type":"result","subtype":"success"}\n')
    assert Trial._new_result(out, out.stat().st_size) == (False, out.stat().st_size)
    with open(out, "a") as f:
        f.write('{"type": "assistant"}\n{"type": "result", "is_error": false}\n{"type": "partial')
    found, offset = Trial._new_result(out, 38)
    assert found and offset == out.stat().st_size - len('{"type": "partial')


@pytest.mark.skipif(not Path(FCS).is_dir() or not shutil.which("g++"), reason="Frontier-CS files or g++ missing")
def test_reminders_reach_running_agents(tmp_path):
    """A short facilitated team trial with a stand-in claude: reminders are generated each round,
    mention each agent's inferred slot, and arrive in the agents' running sessions."""
    from tests.test_judge import _cflags
    fake = tmp_path / "claude"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {Path(__file__).parent / 'fake_claude.py'} \"$@\"\n")
    fake.chmod(0o755)
    res, work = tmp_path / "results", tmp_path / "work"
    cmd = [sys.executable, "-m", "ttc.cli", "trial", "--task", "polyomino", "--mode", "team", "--k", "2", "--trial", "0",
           "--agent-cli", "claude-code", "--model-alias", "claude-sonnet-4-6", "--claude-binary", str(fake),
           "--facilitator", "--facilitator-first-seconds", "4", "--facilitator-interval-seconds", "6",
           "--facilitator-min-remaining-seconds", "1", "--max-wall-seconds", "20", "--wall-margin-seconds", "1",
           "--frontiercs-dir", FCS, "--results-dir", str(res), "--work-dir", str(work), "--port", "8791",
           "--price-input", "3", "--price-output", "15"]
    for flag in _cflags(tmp_path):
        cmd += [f"--judge-extra-cflag={flag}"]
    env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": "test-token"}
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180, cwd=Path(__file__).parent.parent)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads((res / "result.json").read_text())
    assert r["trial"]["facilitator"] and r["trial"]["label"] == "polyomino-teamfac2-000"
    fac = r["facilitator"]
    assert fac["rounds"] >= 2 and fac["reminders_sent"] == 2 * fac["rounds"] and fac["failed_rounds"] == 0
    rounds = [json.loads(line) for line in gzip.open(res / "trajectories" / "facilitator_rounds.jsonl.gz", "rt")]
    assert rounds[0]["reminders"] == {"0": "log your attempts (slot 1)", "1": "log your attempts (slot 0)"}
    assert all(d == "sent" for rd in rounds for d in rd["delivery"].values())
    label = work / "polyomino-teamfac2-000"
    for i, slot in ((0, 1), (1, 0)):
        got = [json.loads(line)["text"] for line in (label / "agents" / str(i) / "home" / "received.jsonl").open()]
        assert got[0].startswith("Polyomino") or "polyomino" in got[0].lower()  # the task prompt came first
        assert f"[Team reminder from the harness] log your attempts (slot {slot})" in got[1:]
