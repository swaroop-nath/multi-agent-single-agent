"""Local re-implementation of the Frontier-CS algorithmic judge for problem 0 (polyomino packing).

Matches the official judge (algorithmic/judge/src at the pinned commit):
  * compile:  g++ <src> -O2 -pipe -std=gnu++17 -o <bin>         (checker: also -I <testlib dir>)
  * per case: CPU limit = time (2 s), wall limit = 2 x time, memory = stack = 256 MB
  * checker:  the unmodified upstream chk.cc (testlib); a case scores only if the checker exits
              0 (ok) or 7 (points), and its score is the "Ratio: x" in the checker's message
  * score:    mean over all cases of the per-case ratio (failed cases contribute 0)
The paper (and the problem statement) additionally treat a submission with any failed case as
invalid; `valid` reports that, and a trial keeps its highest valid submission.

Untrusted code (the agent's program) is compiled and run as an unprivileged judge user, with
the test input on stdin, output captured through a pipe, and RLIMIT_FSIZE=0 so it cannot write
files (and so cannot copy hidden test inputs anywhere the agents could read them).
"""

from __future__ import annotations

import math
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .fetch import case_ids, load_lock, problem_dir, testlib_dir

TESTLIB_OK, TESTLIB_POINTS = 0, 7
RATIO_RE = re.compile(r"Ratio: ([\d.]+)")
OUTPUT_CAP = 64 * 1024 * 1024
COMPILE_CPU_S, COMPILE_WALL_S = 30, 60


@dataclass
class RunResult:
    verdict: str  # OK | TLE | MLE | RE | OLE
    cpu_s: float
    wall_s: float
    max_rss_mb: float
    exit_code: int | None
    stdout: bytes


def _maxrss_mb(ru) -> float:
    return ru.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru.ru_maxrss / 1024


def _limits(cpu_s: float, mem_bytes: int | None, fsize: int, nproc: int | None):
    def apply():
        os.setsid()
        hard_cpu = int(math.ceil(cpu_s)) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if mem_bytes is not None:
            for lim in (resource.RLIMIT_STACK, resource.RLIMIT_AS):
                try:
                    resource.setrlimit(lim, (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    pass  # macOS does not enforce RLIMIT_AS; peak RSS is still checked
        if nproc is not None:
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            except (ValueError, OSError):
                pass
    return apply


def clean_env(tmpdir: Path) -> dict[str, str]:
    """Untrusted processes never inherit the harness environment (it holds the API key)."""
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "TMPDIR": str(tmpdir), "LANG": "C", "HOME": str(tmpdir)}


def run_limited(argv: list[str], stdin_path: Path | None, cwd: Path, cpu_s: float, wall_s: float,
                mem_mb: int | None, fsize: int = 0, nproc: int | None = 128,
                as_factor: float = 4.0, merge_stderr: bool = False) -> RunResult:
    """Run with limits; CPU and peak memory come from wait4's rusage for this child only."""
    mem_bytes = int(mem_mb * as_factor * 1024 * 1024) if mem_mb else None
    stdin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
                                cwd=cwd, env=clean_env(cwd), preexec_fn=_limits(cpu_s, mem_bytes, fsize, nproc),
                                close_fds=True)
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()
    chunks, size = [], [0]

    def reader():
        while True:
            b = proc.stdout.read(1 << 16)
            if not b:
                break
            if size[0] < OUTPUT_CAP:
                chunks.append(b)
            size[0] += len(b)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    timer = threading.Timer(wall_s, lambda: _killpg(proc.pid))
    timer.start()
    try:
        _, status, ru = os.wait4(proc.pid, 0)
    finally:
        timer.cancel()
    proc.returncode = os.waitstatus_to_exitcode(status)
    _killpg(proc.pid)  # stray children
    t.join(5)
    wall = time.monotonic() - t0
    cpu = ru.ru_utime + ru.ru_stime
    rss = _maxrss_mb(ru)
    if cpu > cpu_s or wall > wall_s:
        verdict = "TLE"
    elif mem_mb and rss > mem_mb:
        verdict = "MLE"
    elif size[0] > OUTPUT_CAP:
        verdict = "OLE"
    elif proc.returncode != 0:
        verdict = "RE"
    else:
        verdict = "OK"
    return RunResult(verdict, round(cpu, 3), round(wall, 3), round(rss, 1), proc.returncode, b"".join(chunks))


def _killpg(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class Judge:
    def __init__(self, root: str | Path, work: Path, run_as: tuple[int, int] | None, parallelism: int = 2,
                 compiler: str = "g++", extra_cflags: list[str] | None = None, wall_factor: float = 2.0):
        lock = load_lock()["problem"]
        self.root = Path(root)
        self.prob = problem_dir(root)
        self.work = work
        self.run_as = run_as
        self.parallelism = parallelism
        self.compiler = compiler
        self.extra = list(extra_cflags or [])
        self.time_s = float(lock["time_limit_s"])
        self.mem_mb = int(lock["memory_mb"])
        self.wall_s = wall_factor * self.time_s
        self.cases = case_ids()
        self.checker: Path | None = None
        self._lock = threading.Lock()  # one submission at a time, cases in parallel
        self.work.mkdir(parents=True, exist_ok=True)

    def _as_judge(self, argv: list[str]) -> list[str]:
        if self.run_as is None:
            return argv
        uid, gid = self.run_as
        return ["setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", "--", *argv]

    def build_checker(self, prebuilt: Path | None = None) -> Path:
        if prebuilt and prebuilt.exists():
            self.checker = prebuilt
            return prebuilt
        out = self.work / "chk"
        cmd = [self.compiler, str(self.prob / "chk.cc"), "-O2", "-pipe", "-std=gnu++17",
               "-I", str(testlib_dir(self.root)), *self.extra, "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"checker compile failed: {r.stderr[-2000:]}")
        self.checker = out
        return out

    def _check(self, case: int, output: bytes, tmp: Path) -> tuple[float, str]:
        out_path = tmp / f"{case}.out"
        out_path.write_bytes(output)
        r = subprocess.run([str(self.checker), str(self.prob / f"testdata/{case}.in"), str(out_path),
                            str(self.prob / f"testdata/{case}.ans")], capture_output=True, text=True, timeout=120)
        out_path.unlink(missing_ok=True)
        msg = (r.stderr or "") + (r.stdout or "")
        if r.returncode not in (TESTLIB_OK, TESTLIB_POINTS):
            return 0.0, "WA"
        m = RATIO_RE.search(msg)
        return (float(m.group(1)) if m else 1.0), "OK"

    def evaluate(self, source: str, sub_dir: Path) -> dict:
        """Blocking; call from a worker thread. Never exposes per-case data beyond verdict counts."""
        with self._lock:
            return self._evaluate(source, sub_dir)

    def _evaluate(self, source: str, sub_dir: Path) -> dict:
        assert self.checker is not None
        t0 = time.monotonic()
        build = sub_dir / "build"
        build.mkdir(parents=True, exist_ok=True)
        os.chmod(sub_dir, 0o711)  # judge user can reach build/, nobody else can list
        (build / "solution.cpp").write_text(source)
        if self.run_as:
            os.chown(build, *self.run_as)
            os.chown(build / "solution.cpp", *self.run_as)
            os.chmod(build, 0o700)
        cc = run_limited(self._as_judge([self.compiler, "solution.cpp", "-O2", "-pipe", "-std=gnu++17",
                                         *self.extra, "-o", "solution"]),
                         None, build, COMPILE_CPU_S, COMPILE_WALL_S, None, fsize=512 * 1024 * 1024,
                         nproc=None, merge_stderr=True)
        if cc.verdict != "OK" or not (build / "solution").exists():
            msg = cc.stdout.decode(errors="replace") if cc.verdict == "RE" else f"compiler {cc.verdict}"
            return {"status": "compile_error", "score": 0.0, "valid": False, "message": msg[-4000:],
                    "judge_seconds": round(time.monotonic() - t0, 1)}

        tmp = sub_dir / "check"
        tmp.mkdir(exist_ok=True)
        os.chmod(tmp, 0o700)  # program outputs, read only by the (trusted) checker
        binary = str(build / "solution")

        def one(case: int) -> dict:
            r = run_limited(self._as_judge([binary]), self.prob / f"testdata/{case}.in", build,
                            self.time_s, self.wall_s, self.mem_mb, fsize=0)
            if r.verdict != "OK":
                return {"case": case, "verdict": r.verdict, "ratio": 0.0, "cpu_s": r.cpu_s, "rss_mb": r.max_rss_mb}
            ratio, v = self._check(case, r.stdout, tmp)
            return {"case": case, "verdict": v, "ratio": ratio, "cpu_s": r.cpu_s, "rss_mb": r.max_rss_mb}

        with ThreadPoolExecutor(self.parallelism) as ex:
            results = list(ex.map(one, self.cases))
        verdicts = Counter(r["verdict"] for r in results)
        score = sum(r["ratio"] for r in results) / len(self.cases)
        return {
            "status": "ok",
            "score": round(score, 9),                # mean per-case ratio in [0, 1]; official judge = 100 x this
            "valid": verdicts.get("OK", 0) == len(self.cases),
            "cases": len(self.cases),
            "verdicts": dict(verdicts),
            "max_cpu_s": max(r["cpu_s"] for r in results),
            "max_rss_mb": max(r["rss_mb"] for r in results),
            "judge_seconds": round(time.monotonic() - t0, 1),
            "_per_case": results,                    # kept in the harness ledger only, never sent to agents
        }
