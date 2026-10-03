"""Polyomino packing verifier: score any C++17 solution with the Frontier-CS judge
re-implementation (ttc/frontiercs/judge.py) on the 70 pinned test cases, or re-score a finished
trial by re-judging its best_solution.cpp and comparing with the recorded score.

Scores of solutions that budget themselves by wall-clock time vary slightly from run to run
(also under the official judge), so trial comparisons use a tolerance.
"""

from __future__ import annotations

import json
import os
import pwd
import tempfile
from pathlib import Path

from ..frontiercs.fetch import verify
from ..frontiercs.judge import Judge


def _judge_identity(user: str) -> tuple[int, int] | None:
    """Untrusted code never runs as root: as root, judge as the unprivileged judge user."""
    if os.geteuid() != 0:
        return None
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        raise SystemExit(f"refusing to run untrusted code as root: user {user} does not exist")
    return pw.pw_uid, pw.pw_gid


def score_solution(source: str, frontiercs_dir: str, *, parallelism: int = 2, compiler: str = "g++",
                   extra_cflags: list[str] | None = None, checker: str | None = None,
                   per_case: bool = False, judge_user: str = "ttc-judge") -> dict:
    problems = verify(frontiercs_dir)
    if problems:
        return {"status": "error", "error": "Frontier-CS files do not match the lock: " + "; ".join(problems[:5])}
    run_as = _judge_identity(judge_user)
    with tempfile.TemporaryDirectory(prefix="ttc-judge-") as tmp:
        os.chmod(tmp, 0o711)  # the judge user must reach its build dir
        judge = Judge(frontiercs_dir, Path(tmp), run_as=run_as, parallelism=parallelism, compiler=compiler,
                      extra_cflags=extra_cflags)
        prebuilt = Path(checker) if checker else Path(frontiercs_dir) / "chk"
        judge.build_checker(prebuilt)
        result = judge.evaluate(source, Path(tmp) / "submission")
    cases = result.pop("_per_case", None)
    if per_case:
        result["per_case"] = cases
    return result


def verify_trial(results_dir: str | Path, frontiercs_dir: str, *, tolerance: float = 0.005, **judge_kw) -> dict:
    results_dir = Path(results_dir)
    r = json.loads((results_dir / "result.json").read_text())
    label = r["trial"]["label"]
    recorded = r["outcome"]["score"]
    best = results_dir / "best_solution.cpp"
    if not best.exists():
        ok = recorded == 0.0 and r["outcome"]["valid_submissions"] == 0
        return {"task": "polyomino", "label": label, "match": ok, "recorded": {"score": recorded},
                "recomputed": {"score": 0.0},
                "mismatches": [] if ok else ["recorded a valid score but best_solution.cpp is missing"]}
    rescored = score_solution(best.read_text(), frontiercs_dir, **judge_kw)
    mismatches = []
    if rescored.get("status") != "ok":
        mismatches.append(f"re-judge failed: {rescored.get('error') or rescored.get('message')}")
    else:
        if not rescored["valid"]:
            mismatches.append(f"best solution is invalid on re-judge: {rescored['verdicts']}")
        if abs(rescored["score"] - recorded) > tolerance:
            mismatches.append(f"score re-judged {rescored['score']:.6f} vs recorded {recorded:.6f} "
                              f"(tolerance {tolerance})")
    return {"task": "polyomino", "label": label, "match": not mismatches,
            "recorded": {"score": recorded, "best_submission": r["outcome"]["best_submission"]},
            "recomputed": {k: rescored.get(k) for k in ("score", "valid", "verdicts", "max_cpu_s")},
            "tolerance": tolerance, "mismatches": mismatches}
