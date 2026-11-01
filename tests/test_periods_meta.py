"""Periods from meta.json (kick-off and last frame of each period) instead of an unfilled period_id column."""
import contextlib
import warnings

import numpy as np
import pytest

from phase import features as Fe
from phase import meta as M
from phase import prep
from synth import fused_csv, meta_json

FAST = dict(min_break_s=1.0, min_period_s=2.0, min_gk_s=1.0)


@contextlib.contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------- phase.meta.periods
def test_meta_periods(tmp_path):
    m = M.load(meta_json(tmp_path / "m.json", period_start={1: 0, 2: 400}, period_end_frame={"1": 299, "2": 699}))
    assert M.periods(m) == {1: (0, 299), 2: (400, 699)}
    assert M.periods(M.load(meta_json(tmp_path / "m2.json"))) is None  # no period_end_frame
    m = M.load(meta_json(tmp_path / "m3.json", period_start={1: 0, 2: 400}, period_end_frame={"1": 299}))
    with pytest.raises(ValueError, match="only a start"):
        M.periods(m)


# ---------------------------------------------------------------- detect_periods(periods=...)
def test_meta_periods_take_precedence():
    n_on = np.full(1000, 22)
    n_on[300:350] = 0  # players leave at half-time, but line up again 50 frames before the kick-off at 400
    n_on[500:503] = 0  # dropout inside period 2
    col = np.r_[np.ones(300), np.zeros(100), np.full(600, 2)]  # a filled column that disagrees is ignored
    with no_warnings():
        p = prep.detect_periods(n_on, col, hz=25, min_break_s=1, min_period_s=2, periods={1: (0, 299), 2: (400, 989)})
    assert (p[:300] == 1).all() and (p[300:400] == 0).all() and (p[400:500] == 2).all()
    assert (p[500:503] == 0).all() and (p[503:990] == 2).all() and (p[990:] == 0).all()
    breaks = prep.detect_periods(n_on, None, hz=25, min_break_s=1, min_period_s=2)
    assert (breaks[350:400] == 2).all()  # what the player-free break rule gives: period 2 from the line-up


@pytest.mark.parametrize("periods", [{1: (0, 500), 2: (400, 900)}, {1: (0, 299), 2: (400, 1000)}, {1: (300, 200)}])
def test_bad_meta_periods_raise(periods):
    with pytest.raises(ValueError, match="period"):
        prep.detect_periods(np.full(1000, 22), periods=periods)


def test_meta_periods_without_players_warn():
    n_on = np.full(1000, 22)
    n_on[400:] = 0
    with pytest.warns(UserWarning, match="period 2"):
        prep.detect_periods(n_on, periods={1: (0, 399), 2: (400, 999)})


# ---------------------------------------------------------------- prep.load
def test_load_with_meta_periods(tmp_path):
    # home defends -x until frame 400, the second half kicks off at 450; nobody on the pitch on [300, 350)
    path = fused_csv(tmp_path / "f.csv", T=800, empty=[(300, 350)], flip_from=350, period_col=np.zeros(800))
    with no_warnings():  # the unfilled period_id column is not even looked at
        d = prep.load(path, periods={1: (0, 299), 2: (450, 799)}, **FAST)
    per = d["period"]  # 12.5 Hz
    assert (per[:150] == 1).all() and (per[150:225] == 0).all() and (per[225:] == 2).all()
    assert d["attack_sign"][1] == 1 and d["attack_sign"][2] == -1
    with pytest.warns(UserWarning, match="not filled"):
        old = prep.load(path, **FAST)
    assert (old["period"][175:225] == 2).all()  # player-free break rule: the line-up already counts as period 2


# ---------------------------------------------------------------- features: direction outside a period
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_frames_outside_a_period_take_the_direction_of_the_nearest_period(tmp_path):
    # half-time: nobody on [300, 400); teams line up at swapped ends on [400, 450); second-half kick-off at 450
    path = fused_csv(tmp_path / "f.csv", T=800, empty=[(300, 400)], flip_from=400)
    d = prep.load(path, periods={1: (0, 299), 2: (450, 799)}, **FAST)
    assert (d["period"][200:225] == 0).all() and d["attack_sign"][2] == -1
    x = Fe.build(d, np.zeros(400))[0]["node_pos"][:, :11, 0]  # home players, rotated so home attacks +x
    assert np.allclose(x[200:225], x[300])  # line-up oriented like period 2 (before: like period 1, mirrored)
    assert np.allclose(x[300], x[100])
