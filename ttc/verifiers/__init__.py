"""Standalone verifiers: re-score finished trials, or score raw submissions, with the same
upstream verifiers the trials use and without trusting the harness's own result.json.

  ARC-AGI-3:  replay recorded actions through the pinned game in the official engine (arc.py)
  Polyomino:  re-judge a C++ solution with the Frontier-CS judge re-implementation (polyomino.py)
"""

from __future__ import annotations

import json
from pathlib import Path


def verify_trial(results_dir: str | Path, *, environments_dir: str, frontiercs_dir: str, **kw) -> dict:
    r = json.loads((Path(results_dir) / "result.json").read_text())
    if r.get("status") != "ok":
        return {"label": r.get("trial", {}).get("label"), "match": False,
                "error": f"trial status {r.get('status')}: {r.get('error')}"}
    task = r["trial"].get("task", "arc")
    if task == "arc":
        from .arc import verify_trial as v
        return v(results_dir, environments_dir)
    from .polyomino import verify_trial as v
    return v(results_dir, frontiercs_dir, **kw)
