"""Verifier tests against the real pinned games / Frontier-CS files.
Set TTC_ARC_ENVS (from `ttc download --dir DIR`) and TTC_FRONTIERCS_DIR to run them."""

import json
import os
import random
import shutil
from pathlib import Path

import pytest

from ttc.arc.download import load_lock, verify as verify_games
from ttc.arc.game import TrialGame, load_arc_game
from ttc.verifiers.arc import score_actions

ARC_ENVS = os.environ.get("TTC_ARC_ENVS", "/opt/arc_envs")
GAME = "ls20"
needs_arc = pytest.mark.skipif(not Path(ARC_ENVS).is_dir() or bool(verify_games(ARC_ENVS, [GAME])),
                               reason="pinned ARC games not available (set TTC_ARC_ENVS)")


def _play(tmp_path, n_actions, multiplier=5.0, seed=0):
    game_id = load_lock()["games"][GAME]["game_id"]
    _, baselines, make_env, action_from_id = load_arc_game(ARC_ENVS, game_id)
    events = tmp_path / "events.jsonl"
    g = TrialGame("t", game_id, baselines, 1, make_env, action_from_id, multiplier=multiplier,
                  events_path=events)
    s = g.sessions[0]
    rng = random.Random(seed)
    for _ in range(n_actions):
        if s.status != "active":
            break
        choice = rng.choice(s.frame.available_actions + [0])
        g.act(s, choice, rng.randrange(64), rng.randrange(64)) if choice == 6 else g.act(s, choice)
    g.close()
    actions = [e for e in map(json.loads, events.read_text().splitlines()) if e["kind"] == "action"]
    return game_id, g.summary()["agents"][0], actions


@needs_arc
def test_replay_matches_live_server(tmp_path):
    game_id, live, actions = _play(tmp_path, 80)
    rep = score_actions(ARC_ENVS, game_id, actions)
    assert rep["frames_match"], rep["first_divergent_action"]
    assert rep["max_level"] == live["max_level"]
    assert rep["total_actions"] == live["total_actions"]
    assert rep["actions_per_level"] == live["actions_per_level"]


@needs_arc
def test_replay_detects_a_tampered_frame(tmp_path):
    game_id, _, actions = _play(tmp_path, 20)
    actions[7]["frame_sha256"] = "0" * 64
    rep = score_actions(ARC_ENVS, game_id, actions)
    assert not rep["frames_match"] and rep["first_divergent_action"] == 7


@needs_arc
def test_replay_reproduces_budget_exhaustion(tmp_path):
    game_id, live, actions = _play(tmp_path, 50, multiplier=0.1)  # level 1 budget: ceil(0.1 * 22) = 3
    assert live["status"] == "exhausted" and live["total_actions"] == 3
    rep = score_actions(ARC_ENVS, game_id, actions, multiplier=0.1)
    assert rep["status"] == "budget_exhausted" and rep["total_actions"] == 3
    assert "error" in score_actions(ARC_ENVS, game_id, actions + actions[:1], multiplier=0.1)


FCS = os.environ.get("TTC_FRONTIERCS_DIR", "/opt/frontiercs")
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(not Path(FCS).is_dir() or not shutil.which("g++"), reason="Frontier-CS files or g++ missing")
def test_polyomino_trial_verification(tmp_path):
    from tests.test_judge import _cflags
    from ttc.verifiers.polyomino import verify_trial
    res = tmp_path / "results"
    res.mkdir()
    shutil.copy(FIXTURES / "simple_pack.cpp", res / "best_solution.cpp")
    result = {"trial": {"label": "polyomino-solo1-000", "task": "polyomino"}, "status": "ok",
              "outcome": {"score": 0.278105176, "valid_submissions": 1, "best_submission": {"submission": 1}}}
    (res / "result.json").write_text(json.dumps(result))
    flags = _cflags(tmp_path)
    assert verify_trial(res, FCS, extra_cflags=flags)["match"]
    result["outcome"]["score"] = 0.5
    (res / "result.json").write_text(json.dumps(result))
    rep = verify_trial(res, FCS, extra_cflags=flags)
    assert not rep["match"] and "re-judged" in rep["mismatches"][0]
