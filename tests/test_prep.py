"""phase.prep: stable player slots (review item 1) and period detection (item 2)."""
import contextlib
import warnings

import numpy as np
import pytest

from phase import prep
from synth import fused_csv

FAST = dict(min_break_s=1.0, min_period_s=2.0)  # synthetic matches are seconds long


def load(tmp_path, prep_kw=FAST, **kw):
    return prep.load(fused_csv(tmp_path / "fused.csv", **kw), **prep_kw)


def slot_of(d, team, col, frame):
    return int(np.flatnonzero(d["slot_col"][frame, team] == col)[0])


@contextlib.contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------- slots
def test_dropout_keeps_slot_and_does_not_shift_others(tmp_path):
    d = load(tmp_path, T=400, drop={(0, 3): [(100, 200)]})  # 25 Hz frames -> 50..99 at 12.5 Hz
    s = slot_of(d, 0, 3, 0)
    held = d["slot_col"][:, 0, s]
    assert (held[:50] == 3).all() and (held[50:100] == -1).all() and (held[100:] == 3).all()
    assert np.isnan(d["pos"][50:100, 0, s]).all()
    others = [k for k in range(11) if k != s]
    assert (d["slot_col"][:, 0, others] == d["slot_col"][0, 0, others]).all()  # nobody moves into the gap
    assert np.allclose(d["pos"][75, 0, others], d["pos"][0, 0, others])


def test_substitution_reuses_slot_and_keeps_goalkeeper(tmp_path):
    # column 11 comes on at 25 Hz frame 200, column 5 leaves at 204 (both present for 4 frames)
    with no_warnings():
        d = load(tmp_path, T=400, n_cols=12, extra={(0, 11): (200, 400), (1, 11): (0, 0)}, drop={(0, 5): [(204, 400)]})
    s = slot_of(d, 0, 5, 0)
    assert (d["slot_col"][110:, 0, s] == 11).all()
    assert ((d["slot_col"] >= 0).sum(-1) == 11).all()  # always 11 players per team
    assert (d["is_gk"].sum(-1) == 1).all()  # goalkeeper never lost
    assert (d["slot_col"][:, 0][d["is_gk"][:, 0]] == 0).all()  # and it is column 0


def test_more_than_eleven_players_warns_and_keeps_incumbents(tmp_path):
    with pytest.warns(UserWarning, match="more than 11 players"):
        d = load(tmp_path, T=400, n_cols=12, extra={(0, 11): (0, 400), (1, 11): (0, 0)})
    assert not (d["slot_col"][:, 0] == 11).any()
    assert (d["is_gk"][:, 0].sum(-1) == 1).all()


def test_slot_names_are_stored(tmp_path):
    d = load(tmp_path, T=100)
    assert d["player_cols"][0, 0] == "home_home_1" and d["player_cols"][1, 10] == "away_away_11"


# ---------------------------------------------------------------- periods
def test_no_break_does_not_crash(tmp_path):
    with pytest.warns(UserWarning, match="period 1"):
        d = load(tmp_path, prep_kw={}, T=400)  # default thresholds: 16 s is not a period
    assert (d["period"] == 1).all()
    assert d["attack_sign"][1] == 1


def test_two_halves_and_direction(tmp_path):
    d = load(tmp_path, T=1000, empty=[(450, 550)], flip_from=550)
    p = d["period"]
    assert (p[:225] == 1).all() and (p[225:275] == 0).all() and (p[275:] == 2).all()
    assert d["attack_sign"][1] == 1 and d["attack_sign"][2] == -1
    gk = d["is_gk"][:, 0]
    assert (gk.sum(1)[p > 0] == 1).all() and (gk.sum(1)[p == 0] == 0).all()


def test_short_dropout_is_not_a_break(tmp_path):
    d = load(tmp_path, T=1000, empty=[(100, 110), (450, 550)], flip_from=550)  # 0.4 s glitch in the first half
    assert set(np.unique(d["period"])) == {0, 1, 2}
    assert (d["period"][50:55] == 0).all() and (d["period"][:50] == 1).all() and (d["period"][56:225] == 1).all()


def test_extra_time_has_four_periods(tmp_path):
    d = load(tmp_path, T=2000, empty=[(400, 500), (900, 1000), (1400, 1500)])
    assert set(np.unique(d["period"])) == {0, 1, 2, 3, 4}
    assert len(d["attack_sign"]) == 5


def test_provider_period_column_is_used(tmp_path):
    per = np.r_[np.ones(300), np.full(300, 2)]  # no player-free break in the data at all
    d = load(tmp_path, T=600, flip_from=300, period_col=per)
    assert (d["period"][:150] == 1).all() and (d["period"][150:] == 2).all()
    assert d["attack_sign"][1] == 1 and d["attack_sign"][2] == -1


def test_home_attack_sign_outside_periods_uses_nearest_period():
    d = dict(period=np.array([0, 1, 1, 0, 0, 2, 2, 0]), attack_sign=np.array([0.0, 1.0, -1.0]))
    assert prep.home_attack_sign(d, np.arange(8)).tolist() == [1, 1, 1, 1, -1, -1, -1, -1]
    assert float(prep.home_attack_sign(d, 5)) == -1
