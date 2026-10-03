#!/usr/bin/env python3
"""`arc`: the agent's only interface to its ARC-AGI-3 game session (stdlib only).

Reads ARC_SERVER_URL and ARC_AGENT_TOKEN from the environment.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def call(method, path, payload=None):
    url = os.environ["ARC_SERVER_URL"].rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Agent-Token", os.environ["ARC_AGENT_TOKEN"])
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"arc: server error {e.code}: {body}")
    except urllib.error.URLError as e:
        sys.exit(f"arc: cannot reach game server: {e.reason}")


def render(out, show_grid):
    lines = []
    if out.get("result") and out["result"] != "OK":
        lines.append(f"[{out['result']}] {out.get('message', '')}".rstrip())
    lines.append(
        f"status={out['status']} state={out['state']} "
        f"levels_completed={out['levels_completed']}/{out['win_levels']} "
        f"available_actions={out['available_actions']}"
    )
    if out.get("level_budget") is not None:
        lines.append(
            f"level_actions_used={out['level_actions_used']}/{out['level_budget']} "
            f"(left {out['level_actions_left']}) total_actions={out['total_actions']}"
        )
    if out.get("waiting_for_teammates"):
        lines.append("waiting_for_teammates=true (game actions are paused; they will not be charged)")
    if out.get("end_reason"):
        lines.append(f"end_reason={out['end_reason']}")
    frames = out.get("frames") or []
    if show_grid and frames:
        n = out.get("num_frames", len(frames))
        for i, grid in enumerate(frames):
            label = f"frame {n - len(frames) + i + 1}/{n}" if n > 1 else "frame"
            width = len(grid[0]) if grid else 0
            lines.append(f"--- {label} ({len(grid)}x{width}, x across, y down) ---")
            lines.append("    " + "".join(str(c // 10) if c % 10 == 0 else " " for c in range(width)))
            lines.append("    " + "".join(str(c % 10) for c in range(width)))
            for y, row in enumerate(grid):
                lines.append(f"{y:>3} {row}")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(prog="arc", description="Play your ARC-AGI-3 game session.")
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    p.add_argument("--all-frames", action="store_true", help="include every animation frame")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="status without the frame (free)")
    sub.add_parser("state", help="current frame (free)")
    a = sub.add_parser("act", help="take simple action n (1-5, 7)")
    a.add_argument("n", help="action number, e.g. 3 or ACTION3")
    c = sub.add_parser("click", help="take action 6 at column x, row y")
    c.add_argument("x", type=int)
    c.add_argument("y", type=int)
    sub.add_parser("reset", help="take RESET (charged)")
    # also accept the global flags after the subcommand: `arc act 3 --json`
    for sp in sub.choices.values():
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        sp.add_argument("--all-frames", action="store_true", default=argparse.SUPPRESS)
    args = p.parse_args()

    qs = "?all_frames=1" if args.all_frames else ""
    if args.cmd == "info":
        out = call("GET", "/info")
    elif args.cmd == "state":
        out = call("GET", "/state" + qs)
    elif args.cmd == "reset":
        out = call("POST", "/act" + qs, {"action": 0})
    elif args.cmd == "click":
        out = call("POST", "/act" + qs, {"action": 6, "x": args.x, "y": args.y})
    else:
        n = args.n.upper().removeprefix("ACTION")
        if not n.isdigit() or int(n) == 6:
            sys.exit("arc: use `arc act <1-5|7>`, or `arc click <x> <y>` for action 6")
        out = call("POST", "/act" + qs, {"action": int(n)})

    if args.json:
        print(json.dumps(out))
    else:
        render(out, show_grid=args.cmd != "info")


if __name__ == "__main__":
    main()
