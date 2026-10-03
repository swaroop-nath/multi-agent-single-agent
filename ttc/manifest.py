"""Task manifests: one entry per trial. Each entry becomes one command via the manifest's
`command` template (fields come from the entry)."""

from __future__ import annotations

from .arc.download import load_lock

ARC_COMMAND = ("/app/run_trial --task arc --game {game} --mode {mode} --k {k} --trial {trial} "
               "--results-dir /tmp/results --max-wall-seconds {max_wall_seconds}")
POLY_COMMAND = ("/app/run_trial --task polyomino --mode {mode} --k {k} --trial {trial} "
                "--results-dir /tmp/results --max-wall-seconds {max_wall_seconds}")

# ARC (Section 2.1): 64 single-agent trials per game; 20 trials per (game, team size).
ARC_FULL = {"solo": 64, "solo_rules": 64, "team": {3: 20, 5: 20}}
# The four games where communication helped in the paper, plus two where it did not.
ARC_PILOT_GAMES = ["lp85", "ft09", "sb26", "ar25", "ls20", "vc33"]
ARC_PILOT = {"solo": 16, "solo_rules": 16, "team": {3: 8, 5: 8}}
ARC_WALL = 12 * 3600

# Polyomino (Section 3.2): 3 h runs with 60 solo + 20 team@3 trials; 72 h runs with
# 12 solo + 2 team@4 trials.
POLY_FULL = [({"solo": 60, "solo_rules": 60, "team": {3: 20}}, 3 * 3600),
             ({"solo": 12, "solo_rules": 12, "team": {4: 2}}, 72 * 3600)]
POLY_PILOT = [({"solo": 12, "solo_rules": 12, "team": {3: 4}}, 3 * 3600)]

# Table 4: ARC used 4 CPU / 8 GiB per container and polyomino 2 CPU / 6 GiB, whatever the team
# size; Copilot needs roughly 0.3-0.6 GiB RSS per agent. Polyomino here also runs the scorer
# inside the container (the paper used a separate scorer service), so it gets 2 extra CPUs.
ARC_RESOURCES = {"cpus": 4, "memory_gib": 8}
POLY_RESOURCES = {"cpus": 4, "memory_gib": 8}


def _entries(counts: dict, wall: int, **fixed) -> list[dict]:
    out = []
    for mode in ("solo", "solo_rules"):
        out += [{**fixed, "mode": mode, "k": 1, "trial": t, "max_wall_seconds": wall}
                for t in range(counts.get(mode, 0))]
    for k, n in sorted(counts.get("team", {}).items()):
        out += [{**fixed, "mode": "team", "k": k, "trial": t, "max_wall_seconds": wall} for t in range(n)]
    return out


def manifest(name: str) -> dict:
    if name in ("arc_full", "arc_pilot"):
        games = sorted(load_lock()["games"]) if name == "arc_full" else ARC_PILOT_GAMES
        counts = ARC_FULL if name == "arc_full" else ARC_PILOT
        tasks = [e for g in games for e in _entries(counts, ARC_WALL, task="arc", game=g)]
        command, resources = ARC_COMMAND, ARC_RESOURCES
    elif name in ("polyomino_full", "polyomino_pilot"):
        plan = POLY_FULL if name == "polyomino_full" else POLY_PILOT
        tasks = []
        for counts, wall in plan:
            # 72 h trials get their own index range so labels never collide with 3 h ones
            offset = 0 if wall == 3 * 3600 else 1000
            tasks += [{**e, "trial": e["trial"] + offset}
                      for e in _entries(counts, wall, task="polyomino")]
        command, resources = POLY_COMMAND, POLY_RESOURCES
    else:
        raise ValueError(name)
    return {"name": name, "command": command, "suggested_outer_timeout": "max_wall_seconds + 15 minutes",
            "resources": resources, "num_tasks": len(tasks), "tasks": tasks}


NAMES = ("arc_pilot", "arc_full", "polyomino_pilot", "polyomino_full")
