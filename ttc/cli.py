"""ttc command line.

  ttc trial     --game G --mode M --k K --trial T [...]   run one trial (what /app/run_trial calls)
  ttc analyze   RESULTS_DIR [...] --out DIR                 offline best@k / team@k / RHAE / tokens
  ttc manifest  NAME [--out FILE]                           task list (arc_pilot, arc_full,
                                                            polyomino_pilot, polyomino_full)
  ttc download  [--dir DIR] [--games ...]                   fetch the pinned ARC games (image build)
  ttc verify    [--dir DIR]                                 check cached ARC games against the lock
  ttc fetch-frontiercs [--dir DIR]                          fetch pinned polyomino files (image build)
  ttc verify-frontiercs [--dir DIR]
  ttc lock-games --dir DIR                                  maintainers: re-pin the public games
  ttc verify-trial RESULTS_DIR [...]                        re-score finished trial(s) independently
  ttc trace     RESULTS_DIR [--agent I]                     readable timeline: thinking, messages, tools
  ttc score-polyomino FILE.cpp                              judge one polyomino solution
  ttc score-arc --game G --actions FILE.jsonl               replay an ARC action sequence
  ttc sweep     --manifest FILE --results-root DIR [...]    development: run many trials locally
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .settings import DEFAULT_ENVIRONMENTS_DIR, DEFAULT_FRONTIERCS_DIR, add_trial_arguments, settings_from_args


def _sweep(args: argparse.Namespace, passthrough: list[str]) -> int:
    tasks = json.loads(Path(args.manifest).read_text())["tasks"]
    if args.only:
        tasks = [t for t in tasks if t.get("game", t.get("task")) in args.only]
    root = Path(args.results_root)
    sem = asyncio.Semaphore(args.parallel)
    failures = 0

    async def one(idx: int, t: dict) -> None:
        nonlocal failures
        task = t.get("task", "arc")
        label = f"{t.get('game', task)}-{t['mode']}{t['k']}-{t['trial']:03d}"
        out = root / label
        if (out / "result.json").exists() and json.loads((out / "result.json").read_text()).get("status") == "ok":
            return
        async with sem:
            cmd = [sys.executable, "-m", "ttc.cli", "trial", "--task", task, "--mode", t["mode"],
                   "--k", str(t["k"]), "--trial", str(t["trial"]), "--results-dir", str(out),
                   "--port", str(args.base_port + idx % 1000)]
            if t.get("game"):
                cmd += ["--game", t["game"]]
            if t.get("max_wall_seconds") and "--max-wall-seconds" not in passthrough:
                cmd += ["--max-wall-seconds", str(t["max_wall_seconds"])]
            cmd += passthrough
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "harness.log", "wb") as logf:
                p = await asyncio.create_subprocess_exec(*cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT)
                rc = await p.wait()
            failures += rc != 0
            print(f"{label}: exit {rc}", flush=True)

    async def main():
        await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))

    asyncio.run(main())
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:  # sweep: everything after -- goes to each trial
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(prog="ttc")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_trial_arguments(sub.add_parser("trial", help="run one trial"))
    an = sub.add_parser("analyze")
    an.add_argument("paths", nargs="+", type=Path)
    an.add_argument("--out", type=Path, default=Path("analysis"))
    mf = sub.add_parser("manifest")
    from .manifest import NAMES
    mf.add_argument("name", choices=NAMES)
    mf.add_argument("--out", type=Path)
    for name in ("download", "verify", "lock-games"):
        sp = sub.add_parser(name)
        sp.add_argument("--dir", default=DEFAULT_ENVIRONMENTS_DIR)
        sp.add_argument("--games", nargs="*")
    for name in ("fetch-frontiercs", "verify-frontiercs"):
        sp = sub.add_parser(name)
        sp.add_argument("--dir", default=DEFAULT_FRONTIERCS_DIR)
    vt = sub.add_parser("verify-trial", help="re-score finished trials with the upstream verifiers")
    vt.add_argument("paths", nargs="+", type=Path, help="results dirs (searched for result.json)")
    vt.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    vt.add_argument("--frontiercs-dir", default=DEFAULT_FRONTIERCS_DIR)
    vt.add_argument("--tolerance", type=float, default=0.005, help="polyomino score tolerance")
    vt.add_argument("--judge-parallelism", type=int, default=2)
    vt.add_argument("--judge-extra-cflag", action="append", dest="judge_extra_cflags", default=[])
    vt.add_argument("--out", type=Path, help="write the report as JSON")
    sp = sub.add_parser("score-polyomino", help="judge a C++17 solution on the 70 pinned cases")
    sp.add_argument("file", type=Path)
    sp.add_argument("--frontiercs-dir", default=DEFAULT_FRONTIERCS_DIR)
    sp.add_argument("--judge-parallelism", type=int, default=2)
    sp.add_argument("--judge-extra-cflag", action="append", dest="judge_extra_cflags", default=[])
    sp.add_argument("--per-case", action="store_true")
    tr = sub.add_parser("trace", help="print agents' thinking, messages and tool calls as a timeline")
    tr.add_argument("results_dir", type=Path)
    tr.add_argument("--agent", type=int, default=None)
    tr.add_argument("--max-chars", type=int, default=2000, help="clip long entries (0 = no limit)")
    tr.add_argument("--no-tool-output", action="store_true")
    sa = sub.add_parser("score-arc", help="replay an action sequence through a pinned ARC game")
    sa.add_argument("--game", required=True)
    sa.add_argument("--actions", required=True, type=Path,
                    help='JSON lines: {"action": 0-7, "x"?: int, "y"?: int}')
    sa.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    sa.add_argument("--budget-multiplier", type=float, default=5.0)
    sw = sub.add_parser("sweep", help="development: run manifest tasks locally (trial args after --)")
    sw.add_argument("--manifest", required=True)
    sw.add_argument("--results-root", required=True)
    sw.add_argument("--parallel", type=int, default=2)
    sw.add_argument("--base-port", type=int, default=8700)
    sw.add_argument("--only", nargs="*", help="game ids (ARC) or task names")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    for noisy in ("arc_agi", "arcengine", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "trial":
        from .trial import run
        return run(settings_from_args(args))
    if args.cmd == "analyze":
        from .analysis import analyze
        analyze(args.paths, args.out)
        return 0
    if args.cmd == "manifest":
        from .manifest import manifest
        text = json.dumps(manifest(args.name), indent=1) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.cmd == "download":
        from .arc.download import download_locked
        print("cached:", download_locked(args.dir, games=args.games))
        return 0
    if args.cmd == "verify":
        from .arc.download import load_lock, verify
        problems = verify(args.dir, args.games or sorted(load_lock()["games"]))
        print("\n".join(problems) or "all games match the lock")
        return 1 if problems else 0
    if args.cmd == "lock-games":
        from .arc.download import default_lock_path, write_lock
        lock = write_lock(args.dir, default_lock_path())
        print(f"locked {len(lock['games'])} games -> {default_lock_path()}")
        return 0
    if args.cmd == "fetch-frontiercs":
        from .frontiercs.fetch import fetch, verify
        n = fetch(args.dir)
        problems = verify(args.dir)
        print(f"fetched {n} files" if not problems else "\n".join(problems))
        return 1 if problems else 0
    if args.cmd == "verify-frontiercs":
        from .frontiercs.fetch import verify
        problems = verify(args.dir)
        print("\n".join(problems) or "all Frontier-CS files match the lock")
        return 1 if problems else 0
    if args.cmd == "verify-trial":
        from .verifiers import verify_trial
        reports = []
        for root in args.paths:
            for p in ([root] if root.name == "result.json" else sorted(root.rglob("result.json"))):
                rep = verify_trial(p.parent, environments_dir=args.environments_dir,
                                   frontiercs_dir=args.frontiercs_dir, tolerance=args.tolerance,
                                   parallelism=args.judge_parallelism, extra_cflags=args.judge_extra_cflags)
                reports.append(rep)
                status = "MATCH" if rep["match"] else "MISMATCH"
                print(f"{status:8} {rep.get('label')}: " + ("; ".join(rep.get("mismatches", [])) or rep.get("error", "")
                                                           or json.dumps(rep.get("recomputed"))))
        if args.out:
            args.out.write_text(json.dumps(reports, indent=2) + "\n")
        return 0 if reports and all(r["match"] for r in reports) else 1
    if args.cmd == "trace":
        from .traces import render
        print(render(args.results_dir, args.agent, args.max_chars, not args.no_tool_output))
        return 0
    if args.cmd == "score-polyomino":
        from .verifiers.polyomino import score_solution
        r = score_solution(args.file.read_text(), args.frontiercs_dir, parallelism=args.judge_parallelism,
                           extra_cflags=args.judge_extra_cflags, per_case=args.per_case)
        print(json.dumps(r, indent=2))
        return 0 if r.get("status") in ("ok", "compile_error") else 1
    if args.cmd == "score-arc":
        from .arc.download import load_lock
        from .verifiers.arc import score_actions
        actions = [json.loads(line) for line in args.actions.read_text().splitlines() if line.strip()]
        game_id = load_lock()["games"][args.game]["game_id"]
        r = score_actions(args.environments_dir, game_id, actions, multiplier=args.budget_multiplier)
        print(json.dumps({"game_id": game_id, **r}, indent=2))
        return 0 if "error" not in r else 1
    if args.cmd == "sweep":
        return _sweep(args, passthrough)
    return 2


if __name__ == "__main__":
    sys.exit(main())
