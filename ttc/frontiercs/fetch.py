"""Pinned Frontier-CS files for the polyomino packing task (problem 0).

`polyomino.lock.json` lists every file needed (statement, checker, testlib.h, the 70 test
cases, the human-best reference solution, the license) at a fixed upstream commit, with
SHA-256 hashes. The image build fetches them (`ttc fetch-frontiercs`); every trial re-verifies.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from importlib.resources import files
from pathlib import Path

DEFAULT_DIR = "/opt/frontiercs"
PROBLEM = "algorithmic/problems/0"


def load_lock() -> dict:
    return json.loads(files("ttc.frontiercs").joinpath("polyomino.lock.json").read_text())


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # present via requests; python.org macOS builds ship without CA certs
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(dest: str | Path) -> int:
    lock = load_lock()
    base = f"https://raw.githubusercontent.com/FrontierCS/Frontier-CS/{lock['commit']}/"
    dest = Path(dest)
    for rel, sha in lock["files"].items():
        out = dest / rel
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha:
            continue
        with urllib.request.urlopen(base + rel, timeout=60, context=_ssl_context()) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{rel}: sha256 {got} != locked {sha}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return len(lock["files"])


def verify(root: str | Path) -> list[str]:
    root = Path(root)
    problems = []
    for rel, sha in load_lock()["files"].items():
        p = root / rel
        if not p.exists():
            problems.append(f"missing {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            problems.append(f"modified {rel}")
    return problems


def problem_dir(root: str | Path) -> Path:
    return Path(root) / PROBLEM


def testlib_dir(root: str | Path) -> Path:
    return Path(root) / "algorithmic/judge/include"


def case_ids() -> list[int]:
    return list(range(1, load_lock()["problem"]["cases"] + 1))
