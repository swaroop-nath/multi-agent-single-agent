"""Judge tests. They need the pinned Frontier-CS files (`ttc fetch-frontiercs --dir DIR`) and a
C++ compiler; point TTC_FRONTIERCS_DIR at the files to run them."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ttc.frontiercs.fetch import problem_dir, verify
from ttc.frontiercs.judge import Judge

ROOT = os.environ.get("TTC_FRONTIERCS_DIR", "/opt/frontiercs")
FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(not Path(ROOT).is_dir() or verify(ROOT) or not shutil.which("g++"),
                                reason="pinned Frontier-CS files or g++ not available")

SHIM = "#include <algorithm>\n#include <cmath>\n#include <cstdio>\n#include <cstring>\n#include <climits>\n" \
       "#include <iostream>\n#include <limits>\n#include <map>\n#include <numeric>\n#include <random>\n" \
       "#include <set>\n#include <string>\n#include <unordered_map>\n#include <unordered_set>\n" \
       "#include <utility>\n#include <vector>\n#include <chrono>\n#include <functional>\n#include <queue>\n"


def _cflags(tmp_path: Path) -> list[str]:
    probe = tmp_path / "probe.cpp"
    probe.write_text("#include <bits/stdc++.h>\nint main(){}\n")
    if subprocess.run(["g++", "-fsyntax-only", str(probe)], capture_output=True).returncode == 0:
        return []
    (tmp_path / "shim" / "bits").mkdir(parents=True)  # Apple clang: no bits/stdc++.h
    (tmp_path / "shim" / "bits" / "stdc++.h").write_text(SHIM)
    return ["-I", str(tmp_path / "shim")]


@pytest.fixture(scope="module")
def judge(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("judge")
    j = Judge(ROOT, tmp / "work", run_as=None, parallelism=4, extra_cflags=_cflags(tmp))
    j.build_checker()
    return j


def test_reference_solution_scores_like_the_human_best(judge, tmp_path):
    src = (problem_dir(ROOT) / "examples" / "reference.cpp").read_text()
    r = judge.evaluate(src, tmp_path / "ref")
    assert r["valid"] and r["verdicts"] == {"OK": 70}
    assert 0.88 < r["score"] < 0.90  # ~0.891; time-budgeted, so it varies slightly with load


def test_simple_valid_and_invalid(judge, tmp_path):
    r = judge.evaluate((FIXTURES / "simple_pack.cpp").read_text(), tmp_path / "simple")
    assert r["valid"] and 0.2 < r["score"] < 0.4
    r = judge.evaluate("int main(){return 0;}", tmp_path / "empty")
    assert not r["valid"] and r["score"] == 0 and r["verdicts"] == {"WA": 70}
    r = judge.evaluate("int main(){ nope }", tmp_path / "ce")
    assert r["status"] == "compile_error" and "error" in r["message"]


def test_program_cannot_write_files(judge, tmp_path):
    r = judge.evaluate((FIXTURES / "leak_attempt.cpp").read_text(), tmp_path / "leak")
    assert r["verdicts"] == {"RE": 70}
    assert not any(p.stat().st_size for p in (tmp_path / "leak").rglob("ttc_leak.txt"))
