from types import SimpleNamespace

import numpy as np

from ttc.arc.game import TrialGame


class FakeEnv:
    """Level L is cleared by taking ACTION1 `need[L]` times in a row; ACTION2 does nothing."""

    def __init__(self, need):
        self.need = need
        self.level = 0
        self.streak = 0

    def _frame(self):
        state = "WIN" if self.level >= len(self.need) else "NOT_FINISHED"
        return SimpleNamespace(frame=[np.full((64, 64), self.level)], state=SimpleNamespace(value=state),
                               levels_completed=self.level, win_levels=len(self.need),
                               available_actions=[1, 2, 6])

    def reset(self):
        return self._frame()

    def step(self, action, data=None):
        if action == 1:
            self.streak += 1
            if self.streak >= self.need[self.level]:
                self.level += 1
                self.streak = 0
        elif action == 0:
            self.streak = 0
        return self._frame()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(k, need=(2, 2, 2), baselines=(2, 2, 2), **kw):
    clock = kw.pop("clock", Clock())
    g = TrialGame("t", "fake", list(baselines), k, lambda: FakeEnv(list(need)), lambda i: i,
                  clock=clock, **kw)
    return g, clock


def test_budget_and_exhaustion():
    g, _ = make(1, need=(100,), baselines=(2,), multiplier=2.0)  # budget 4
    s = g.sessions[0]
    assert g.budget(0) == 4
    for _ in range(3):
        assert g.act(s, 2)["result"] == "OK"
    out = g.act(s, 2)
    assert out["result"] == "OK" and s.status == "exhausted"
    assert g.act(s, 1)["result"] == "ENDED"
    assert s.total_actions == 4


def test_level_log_and_win():
    g, clock = make(1, need=(2, 1), baselines=(2, 2))
    s = g.sessions[0]
    g.act(s, 2)
    g.act(s, 1)
    clock.t += 5
    g.act(s, 1)
    assert s.max_level == 1 and s.level_log[0]["level_actions"] == 3 and s.level_log[0]["t"] == 5
    g.act(s, 1)
    assert s.status == "finished" and g.any_won()
    assert g.summary()["solved"] and g.summary()["team_max_level"] == 2


def test_unavailable_and_click_validation():
    g, _ = make(1)
    s = g.sessions[0]
    assert g.act(s, 3)["result"] == "INVALID"
    assert g.act(s, 6)["result"] == "INVALID"
    assert g.act(s, 6, 70, 1)["result"] == "INVALID"
    assert g.act(s, 6, 3, 4)["result"] == "OK"
    assert s.total_actions == 1


def test_barrier_pauses_leader_until_peer_catches_up():
    # budget 10, interval 5
    g, _ = make(2, need=(100, 100), baselines=(2, 2), multiplier=5.0)
    a, b = g.sessions
    for _ in range(5):
        assert g.act(a, 2)["result"] == "OK"
    out = g.act(a, 2)  # a is now in phase (0, 1), b in (0, 0)
    assert out["result"] == "WAITING" and a.used[0] == 5 and a.refused_actions == 1
    for _ in range(5):
        assert g.act(b, 2)["result"] == "OK"
    assert g.act(a, 2)["result"] == "OK"


def test_barrier_level_ahead_waits_and_exhausted_peer_releases():
    g, _ = make(2, need=(1, 100), baselines=(2, 2), multiplier=1.0)  # budget 2, interval 1
    a, b = g.sessions
    assert g.act(a, 1)["result"] == "OK" and a.level == 1
    assert g.act(a, 2)["result"] == "WAITING"
    assert g.act(b, 2)["result"] == "OK"  # b at (0,1)
    assert g.act(b, 2)["result"] == "OK" and b.status == "exhausted"
    assert g.act(a, 2)["result"] == "OK"  # b no longer live


def test_idle_peer_stops_blocking():
    g, clock = make(2, need=(100,), baselines=(2,), multiplier=1.0, idle_timeout_s=60)
    a, b = g.sessions
    g.act(a, 2)
    assert g.act(a, 2)["result"] == "WAITING"
    clock.t += 61
    assert g.act(a, 2)["result"] == "OK"


def test_no_barrier_for_solo_or_disabled():
    g, _ = make(2, need=(100,), baselines=(2,), multiplier=1.0, sync_barrier=False)
    a, _ = g.sessions
    assert g.act(a, 2)["result"] == "OK"
    assert g.act(a, 2)["result"] == "OK"


def test_reset_charging_toggle():
    g, _ = make(1, count_reset_as_action=False)
    s = g.sessions[0]
    g.act(s, 0)
    assert s.total_actions == 0
    g2, _ = make(1)
    g2.act(g2.sessions[0], 0)
    assert g2.sessions[0].total_actions == 1
