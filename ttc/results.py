"""Writing a trial's results directory.

    <results_dir>/
      result.json                     identity, outcome/score, per-agent termination + tokens, versions
      trajectories/
        prompt.md                     the exact prompt every agent received
        agent-<i>.events.jsonl.gz     Copilot's event stream: messages, the model's reasoning
                                      (assistant.reasoning), tool calls and their full results
        agent-<i>-private.tar.gz      the agent's private state: Copilot session store (full
                                      session history), home and temp dirs, Copilot logs
        model_responses.jsonl.gz      every raw model response the proxy relayed (incl. thinking)
        model_calls.jsonl.gz          every model call: agent, status, retries, latency, tokens
        game_events.jsonl.gz          ARC: every game action, refusal and session end
        submissions.jsonl.gz          polyomino: every submission with per-case verdicts and ratios
        submission_sources.tar.gz     polyomino: every submitted source file
        shared_workspace.tar.gz       the shared folder (notes, slots, score log, coordination) and
                                      every agent's scratch/work-<slot> dir
      best_solution.cpp               polyomino: the best valid submission
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from pathlib import Path

MAX_WORKSPACE_FILE_BYTES = 5 * 1024 * 1024
MAX_PRIVATE_FILE_BYTES = 200 * 1024 * 1024
# Copilot events that carry no information beyond what other events already hold
NOISE_EVENTS = {"session.background_tasks_changed", "tool.execution_partial_result"}
# program caches (e.g. Copilot unpacks its ~100 MB runtime into ~/.cache), not agent output
CACHE_DIRS = ("/Library/Caches/", "/.cache/", "/.npm/", "/node_modules/")
TERMINATION_CODES = ("finished", "budget_exhausted", "wall_clock", "context_overflow", "model_errors",
                     "agent_exited", "teammate_won", "trial_ended", "signal")


def gzip_file(src: Path, dst: Path, keep=lambda line: True) -> None:
    if not src.exists():
        return
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        for line in f:
            if keep(line):
                g.write(line)


def _keep_event(line: bytes) -> bool:
    """Keep every meaningful event, including ephemeral ones such as assistant.reasoning (the
    model's thinking); drop only token-by-token deltas and bookkeeping noise."""
    try:
        t = json.loads(line).get("type", "")
    except (json.JSONDecodeError, AttributeError):
        return True  # keep non-JSON output (e.g. CLI error messages)
    return not (t.endswith("_delta") or t in NOISE_EVENTS)


def tar_dir(src: Path | list[tuple[Path, str]], dst: Path, max_file_bytes: int | None = MAX_WORKSPACE_FILE_BYTES) -> None:
    sources = [(src, src.name)] if isinstance(src, Path) else src
    sources = [(p, name) for p, name in sources if p.exists()]
    if not sources:
        return

    def filt(info: tarfile.TarInfo):
        if max_file_bytes and info.isfile() and info.size > max_file_bytes:
            return None
        if info.name.endswith(".lock") or any(c in "/" + info.name + "/" for c in CACHE_DIRS):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(dst, "w:gz") as t:
        for p, name in sources:
            t.add(p, arcname=name, filter=filt)


def write_trajectories(work: Path, results: Path, k: int, extra: dict[str, Path] | None = None) -> None:
    traj = results / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    if (work / "prompt.md").exists():
        shutil.copy(work / "prompt.md", traj / "prompt.md")
    for i in range(k):
        a = work / "agents" / str(i)
        gzip_file(a / "stdout.jsonl", traj / f"agent-{i}.events.jsonl.gz", keep=_keep_event)
        tar_dir([(a / sub, f"agent-{i}/{sub}") for sub in ("copilot_home", "home", "tmp", "logs")],
                traj / f"agent-{i}-private.tar.gz", max_file_bytes=MAX_PRIVATE_FILE_BYTES)
    gzip_file(work / "model_calls.jsonl", traj / "model_calls.jsonl.gz")
    gzip_file(work / "model_io" / "responses.jsonl", traj / "model_responses.jsonl.gz")
    for name, src in (extra or {}).items():
        if src.is_dir():
            tar_dir(src, traj / name)
        else:
            gzip_file(src, traj / name)
    tar_dir(work / "task", traj / "shared_workspace.tar.gz")


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)
