"""ARC-AGI-3 verifier: re-score a trial by replaying every agent's recorded actions through a
fresh copy of the pinned game in the official engine.

It is independent of the live game server: it reads only the action log
(trajectories/game_events.jsonl.gz), the pinned game files, and the budget rules, and it
recomputes levels cleared, per-level action counts, win/exhaustion, RHAE and the trial score.
Every recorded action carries a hash of the frame the live server saw after it, so the replay
also confirms, step by step, that it reproduces exactly what the agents saw.

It can also score any action sequence directly (`score_actions`).
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from ..analysis import rhae
from ..arc.download import load_lock, verify
from ..arc.game import Frame, load_arc_game


def budget(baselines: list[int], level: int, multiplier: float) -> int:
    base = baselines[level] if level < len(baselines) else max(baselines or [100])
    return max(1, math.ceil(multiplier * base))


def score_actions(environments_dir: str, game_id: str, actions: list[dict], *, multiplier: float = 5.0,
                  count_reset_as_action: bool = True, seed: int = 0) -> dict:
    """Replay one agent's actions ({action, x?, y?, frame_sha256?}) from a fresh game.

    Returns levels cleared, the first-completion log, per-level charged actions, how the session
    ended, and any divergence from the recorded frame hashes.
    """
    _, baselines, make_env, action_from_id = load_arc_game(environments_dir, game_id, seed)
    env = make_env()
    frame = Frame.from_raw(env.reset())
    win_levels = frame.win_levels
    used: dict[int, int] = defaultdict(int)
    total = 0
    max_level = 0
    level_log: list[dict] = []
    status = "active"
    divergence = None
    for idx, a in enumerate(actions):
        if status != "active":
            return {"error": f"action {idx} was taken after the session ended ({status})"}
        aid = int(a["action"])
        level_before = frame.levels_completed
        data = {"x": a["x"], "y": a["y"]} if aid == 6 else None
        raw = env.step(action_from_id(aid), data)
        if raw is None:
            return {"error": f"engine rejected action {idx} ({aid})"}
        frame = Frame.from_raw(raw)
        if aid != 0 or count_reset_as_action:
            used[level_before] += 1
            total += 1
        if frame.levels_completed > max_level:
            for done in range(max_level, frame.levels_completed):
                level_log.append({"level": done + 1, "level_actions": used[done], "action_index": idx})
            max_level = frame.levels_completed
        if divergence is None and a.get("frame_sha256") and frame.sha256() != a["frame_sha256"]:
            divergence = idx
        if frame.state == "WIN" or (win_levels and frame.levels_completed >= win_levels):
            status = "finished"
        elif used[frame.levels_completed] >= budget(baselines, frame.levels_completed, multiplier):
            status = "budget_exhausted"
    agent = {"max_level": max_level, "level_log": level_log}
    return {"max_level": max_level, "win_levels": win_levels, "solved": status == "finished",
            "status": status, "total_actions": total,
            "actions_per_level": {str(k): v for k, v in sorted(used.items()) if v},
            "level_log": level_log, "rhae": rhae(agent, baselines),
            "frames_match": divergence is None, "first_divergent_action": divergence}


def verify_trial(results_dir: str | Path, environments_dir: str) -> dict:
    results_dir = Path(results_dir)
    r = json.loads((results_dir / "result.json").read_text())
    t = r["trial"]
    game, game_id = t["game"], t["game_id"]
    cfg = r["config"]
    problems = verify(environments_dir, [game])
    if problems:
        return {"task": "arc", "label": t["label"], "match": False, "error": "; ".join(problems)}
    if load_lock()["games"][game]["game_id"] != game_id:
        return {"task": "arc", "label": t["label"], "match": False,
                "error": f"trial used {game_id}, lock pins {load_lock()['games'][game]['game_id']}"}

    by_agent: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(results_dir / "trajectories" / "game_events.jsonl.gz", "rt") as f:
        for line in f:
            ev = json.loads(line)
            if ev["kind"] == "action":
                by_agent[ev["agent"]].append(ev)

    mismatches: list[str] = []
    agents = []
    for rec in r["agents"]:
        i = rec["agent"]
        rep = score_actions(environments_dir, game_id, by_agent.get(i, []), multiplier=cfg["budget_multiplier"],
                            count_reset_as_action=cfg["count_reset_as_action"], seed=t.get("game_seed", 0))
        if "error" in rep:
            mismatches.append(f"agent {i}: {rep['error']}")
            agents.append({"agent": i, **rep})
            continue
        for key in ("max_level", "total_actions", "actions_per_level"):
            if rep[key] != rec.get(key):
                mismatches.append(f"agent {i}: {key} replay={rep[key]} recorded={rec.get(key)}")
        if not math.isclose(rep["rhae"], rec.get("rhae", -1), abs_tol=1e-9):
            mismatches.append(f"agent {i}: rhae replay={rep['rhae']} recorded={rec.get('rhae')}")
        if rep["status"] in ("finished", "budget_exhausted") and rep["status"] != rec["termination_reason"]:
            mismatches.append(f"agent {i}: replay ends {rep['status']}, recorded {rec['termination_reason']}")
        if not rep["frames_match"]:
            mismatches.append(f"agent {i}: frames diverge from the recording at action {rep['first_divergent_action']}")
        agents.append({"agent": i, **{k: rep[k] for k in ("max_level", "solved", "status", "total_actions",
                                                           "rhae", "frames_match")}})
    ok_agents = [a for a in agents if "error" not in a]
    recomputed = {
        "score": max((a["max_level"] for a in ok_agents), default=0),
        "solved": any(a["solved"] for a in ok_agents),
        "rhae_best_agent": max((a["rhae"] for a in ok_agents), default=0.0),
    }
    recorded = {k: r["outcome"][k] for k in ("score", "solved", "rhae_best_agent")}
    for k in recomputed:
        same = (math.isclose(recomputed[k], recorded[k], abs_tol=1e-9) if k == "rhae_best_agent"
                else recomputed[k] == recorded[k])
        if not same:
            mismatches.append(f"outcome {k}: replay={recomputed[k]} recorded={recorded[k]}")
    return {"task": "arc", "label": t["label"], "game_id": game_id, "match": not mismatches,
            "recomputed": recomputed, "recorded": recorded, "agents": agents, "mismatches": mismatches}
