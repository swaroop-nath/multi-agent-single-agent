import itertools
import math
from statistics import mean

from ttc.analysis import Analysis, best_at_k, expected_max_at_k, rhae


def test_best_at_k_matches_enumeration():
    pool = [1, 0, 0, 1, 0, 0, 0]
    for k in range(1, len(pool) + 1):
        brute = mean(any(c) for c in itertools.combinations(pool, k))
        assert math.isclose(best_at_k(len(pool), sum(pool), k), brute)


def test_expected_max_matches_enumeration():
    vals = [0, 3, 1, 1, 5, 2]
    for k in range(1, len(vals) + 1):
        brute = mean(max(c) for c in itertools.combinations(vals, k))
        assert math.isclose(expected_max_at_k(vals, k), brute)


def test_rhae_eq_1_2():
    baselines = [10, 10, 10]
    # level 1 at the baseline (score 1), level 2 at twice the baseline (0.25), level 3 not done
    agent = {"max_level": 2, "level_log": [{"level": 1, "level_actions": 10}, {"level": 2, "level_actions": 20}]}
    expected = min((1 + 2) / 6, (1 * 1 + 2 * 0.25) / 6)
    assert math.isclose(rhae(agent, baselines), expected)
    fast = {"max_level": 1, "level_log": [{"level": 1, "level_actions": 1}]}
    assert math.isclose(rhae(fast, baselines), 1 / 6)  # capped by fraction completed


def _r(mode, k, game, max_level, win=3):
    return {"label": f"{game}-{mode}{k}-{max_level}", "mode": mode, "k": k, "game": game,
            "win_levels": win, "max_level": max_level, "solved": max_level >= win,
            "rhae_agents": [0.0] * k, "tokens_total": 0, "tokens_at_solve": None}


def test_depth_table_and_matching_pool():
    results = [_r("solo", 1, "g", lvl) for lvl in (0, 1, 3, 1, 2, 0, 0, 1)]
    results += [_r("team", 2, "g", lvl) for lvl in (3, 2)]
    a = Analysis(results)
    rows = {r["levels_from_target"]: r for r in a.depth_table(2)}
    assert math.isclose(rows[0]["team@2"], 0.5)
    assert math.isclose(rows[0]["best@2"], best_at_k(8, 1, 2))
    n = a.matching_pool_size(2)
    assert best_at_k(8, 1, n) >= 0.5 > best_at_k(8, 1, n - 1)


def test_loose_teams_are_a_separate_config():
    results = [_r("solo", 1, "g", lvl) for lvl in (0, 1, 3, 1)]
    results += [_r("team", 2, "g", 3), {**_r("team_loose", 2, "g", 0)}]
    a = Analysis(results)
    assert ("team", 2) in a.configs and ("team_loose", 2) in a.configs
    a.team_mode = "team_loose"
    assert a.team_rate(2, "g", 0) == 0.0
    a.team_mode = "team"
    assert a.team_rate(2, "g", 0) == 1.0
