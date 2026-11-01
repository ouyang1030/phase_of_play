"""phase.prep: missing ball coordinates in play are measured, short gaps interpolated (review item 4)."""
import numpy as np
import pytest

from phase import prep
from synth import fused_csv

FAST = dict(min_break_s=1.0, min_period_s=2.0, min_gk_s=1.0)
T = 1000  # 25 Hz


def moving_ball():
    t = np.arange(T, dtype=float)
    return np.stack([t * 0.04 - 20, np.full(T, 3.0), np.full(T, 0.2)], 1)  # 1 m/s along x


def load(tmp_path, gaps, dead=()):
    ball = moving_ball()
    for a, b in gaps:
        ball[a:b] = np.nan
    status = np.full(T, "alive", object)
    for a, b in dead:
        status[a:b] = "dead"
    return prep.load(fused_csv(tmp_path / "f.csv", T=T, ball=ball, status=status), **FAST), moving_ball()


def test_short_gap_in_play_is_interpolated(tmp_path):
    d, truth = load(tmp_path, gaps=[(100, 112)])  # 12 frames = 0.48 s
    assert d["ball_valid"].all()
    assert np.allclose(d["ball"], truth[::2], atol=1e-4)
    assert np.allclose(d["ball_v"][60:440, 0], 1.0, atol=1e-3)  # no velocity spike at the gap
    assert d["ball_missing_play"] == pytest.approx(12 / T)
    assert d["ball_gap_max_s"] == pytest.approx(0.48)
    assert d["ball_interp_play"] == pytest.approx(12 / T)


def test_long_gap_and_gaps_touching_a_stoppage_stay_missing(tmp_path):
    d, truth = load(tmp_path, gaps=[(100, 114), (300, 310), (600, 610)], dead=[(305, 400)])
    v = d["ball_valid"]  # 12.5 Hz
    assert not v[50:57].any() and v[57:150].all()  # 14 frames = 0.56 s: not filled
    assert not v[150:155].any()  # gap runs into the stoppage
    assert v[300:305].all() and np.allclose(d["ball"][300:305], truth[600:610:2], atol=1e-4)
    assert (d["ball"][~v] == 0).all()  # as before: missing ball -> 0 in the arrays
    assert d["ball_missing_play"] == pytest.approx((14 + 5 + 10) / (T - 95))
    assert d["ball_gap_max_s"] == pytest.approx(14 / 25)
    assert d["ball_interp_play"] == pytest.approx(10 / (T - 95))


def test_no_gap(tmp_path):
    d, truth = load(tmp_path, gaps=[])
    assert d["ball_valid"].all() and d["ball_missing_play"] == 0 and d["ball_gap_max_s"] == 0
    assert np.allclose(d["ball"], truth[::2], atol=1e-4)


# ---------------------------------------------------------------- provider holds (repeated ball position)
def fast_ball(v=10.0):
    t = np.arange(T, dtype=float)
    return np.stack([t * v / 25 - 50, np.full(T, 3.0), np.full(T, 0.2)], 1)


def hold(ball, a, b):
    ball = ball.copy()
    ball[a:b] = ball[a - 1]  # the provider repeats the last position on [a, b)
    return ball


def test_ball_holds_only_where_the_ball_was_moving():
    alive = np.ones(T, bool)
    assert np.flatnonzero(prep.ball_holds(hold(fast_ball(), 200, 203), alive)).tolist() == [200, 201, 202]
    slow = np.round(fast_ball(0.25), 2)  # 1 cm per frame at 0.25 m/s, rounded to cm: exact repeats
    slow[::3] = np.roll(slow, 1, 0)[::3]
    assert not prep.ball_holds(slow, alive).any()
    rest_then_kick = fast_ball()
    rest_then_kick[:300] = rest_then_kick[300]  # ball on the spot, kicked at frame 300
    assert not prep.ball_holds(rest_then_kick, alive).any()
    dead = alive.copy()
    dead[190:210] = False
    assert not prep.ball_holds(hold(fast_ball(), 200, 203), dead).any()


def test_short_hold_is_interpolated(tmp_path):
    truth = fast_ball()
    path = fused_csv(tmp_path / "f.csv", T=T, ball=hold(truth, 200, 204))
    d = prep.load(path, **FAST)
    assert np.allclose(d["ball"], truth[::2], atol=1e-4) and d["ball_valid"].all()
    assert d["ball_held_play"] == pytest.approx(4 / T) and d["ball_held_max_s"] == pytest.approx(4 / 25)
    assert d["ball_missing_play"] == 0 and d["ball_interp_play"] == 0  # holds are counted on their own
    assert np.allclose(d["ball_v"][105:395, 0], 10.0, atol=1e-3)


def test_long_hold_is_kept_but_not_valid(tmp_path):
    truth = fast_ball()
    held = hold(truth, 200, 220)  # 0.8 s
    d = prep.load(fused_csv(tmp_path / "f.csv", T=T, ball=held), **FAST)
    assert np.allclose(d["ball"], held[::2], atol=1e-4)  # values kept (not zeroed)
    v = d["ball_valid"]
    assert not v[100:110].any() and v[:100].all() and v[110:].all()
    assert d["ball_held_max_s"] == pytest.approx(0.8)
