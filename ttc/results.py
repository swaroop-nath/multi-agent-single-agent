"""Writing a trial's results directory.

    <results_dir>/
      result.json                   identity, outcome/score, per-agent termination + tokens, versions
      trajectories/
        prompt.md                   the exact prompt every agent received
        agent-<i>.events.jsonl.gz   Copilot's event stream (non-ephemeral events)
        model_calls.jsonl.gz        every model call: agent, status, retries, latency, tokens
        game_events.jsonl.gz        ARC: every game action, refusal and session end
        submissions.jsonl.gz        polyomino: every submission with per-case verdicts and ratios
        submission_sources.tar.gz   polyomino: every submitted source file
        shared_workspace.tar.gz     the agents' shared notes, slots and scratch dirs
        copilot_logs.tar.gz         Copilot process logs
      best_solution.cpp             polyomino: the best valid submission
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from pathlib import Path

MAX_WORKSPACE_FILE_BYTES = 5 * 1024 * 1024
TERMINATION_CODES = ("finished", "budget_exhausted", "wall_clock", "context_overflow", "model_errors",
                     "agent_exited", "teammate_won", "trial_ended", "signal")


def gzip_file(src: Path, dst: Path, keep=lambda line: True) -> None:
    if not src.exists():
        return
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        for line in f:
            if keep(line):
                g.write(line)


def _non_ephemeral(line: bytes) -> bool:
    try:
        return not json.loads(line).get("ephemeral", False)
    except (json.JSONDecodeError, AttributeError):
        return True  # keep non-JSON output (e.g. CLI error messages)


def tar_dir(src: Path, dst: Path, skip_large: bool = True) -> None:
    if not src.exists():
        return

    def filt(info: tarfile.TarInfo):
        if skip_large and info.isfile() and info.size > MAX_WORKSPACE_FILE_BYTES:
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(dst, "w:gz") as t:
        t.add(src, arcname=src.name, filter=filt)


def write_trajectories(work: Path, results: Path, k: int, extra: dict[str, Path] | None = None) -> None:
    traj = results / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    if (work / "prompt.md").exists():
        shutil.copy(work / "prompt.md", traj / "prompt.md")
    for i in range(k):
        gzip_file(work / "agents" / str(i) / "stdout.jsonl", traj / f"agent-{i}.events.jsonl.gz",
                  keep=_non_ephemeral)
    gzip_file(work / "model_calls.jsonl", traj / "model_calls.jsonl.gz")
    for name, src in (extra or {}).items():
        if src.is_dir():
            tar_dir(src, traj / name)
        else:
            gzip_file(src, traj / name)
    tar_dir(work / "task", traj / "shared_workspace.tar.gz")
    logs = work / "copilot_logs"
    logs.mkdir(exist_ok=True)
    for i in range(k):
        src = work / "agents" / str(i) / "logs"
        if src.exists():
            shutil.copytree(src, logs / f"agent-{i}", dirs_exist_ok=True)
    tar_dir(logs, traj / "copilot_logs.tar.gz", skip_large=False)


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)
