"""phase.prep goalkeepers: roster or nearest-to-own-goal rule, several per half allowed (review item 3).

The player columns of the fused CSV are '<side>_<side>_<shirt number>': for Ajax-PSV the numbers equal the
Opta F7 shirt numbers of every player who played (GK 26 and 32)."""
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

from phase import meta as M
from phase import prep
from synth import fused_csv, meta_json

FAST = dict(min_break_s=1.0, min_period_s=2.0)
T = 4000  # 25 Hz = 160 s, one period
SUB = 2000  # home goalkeeper column 0 replaced by column 11 (a second goalkeeper) at 25 Hz frame 2000


@contextlib.contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def gk_sub_csv(tmp_path, **kw):
    return fused_csv(tmp_path / "f.csv", T=T, n_cols=12, spot={(0, 11): 50.0},
                     extra={(0, 11): (SUB, T), (1, 11): (0, 0)}, drop={(0, 0): [(SUB + 4, T)]}, **kw)


def gk_column(d, team):
    """Player column of the goalkeeper per 12.5 Hz frame (-1 = none, -2 = several)."""
    gk = d["is_gk"][:, team]
    col = np.where(gk, d["slot_col"][:, team], -1).max(1)
    return np.where(gk.sum(1) > 1, -2, col)


def test_goalkeeper_substituted_during_the_half(tmp_path):
    with no_warnings():
        d = prep.load(gk_sub_csv(tmp_path), **FAST)
    col = gk_column(d, 0)
    assert (col[:SUB // 2] == 0).all() and (col[SUB // 2 + 2:] == 11).all()
    assert (d["is_gk"].sum(-1) == 1).all()  # never zero or two (overlap frames: the newcomer has no slot yet)
    assert d["player_is_gk"][0].tolist() == [True] + [False] * 10 + [True]
    assert d["player_is_gk"][1].sum() == 1


def test_roster_gives_the_same_goalkeepers(tmp_path):
    names = [f"home_home_{j + 1}" for j in range(12)] + [f"away_away_{j + 1}" for j in range(12)]
    roster = {n: n in ("home_home_1", "home_home_12", "away_away_1") for n in names}
    with no_warnings():
        d = prep.load(gk_sub_csv(tmp_path), roster=roster, **FAST)
    assert np.array_equal(gk_column(d, 0), gk_column(prep.load(gk_sub_csv(tmp_path), **FAST), 0))


def test_roster_overrides_the_geometry(tmp_path):
    roster = {f"{s}_{s}_{j + 1}": False for s in ("home", "away") for j in range(11)}
    roster["home_home_5"] = roster["away_away_1"] = True  # say the roster names an outfield player
    d = prep.load(fused_csv(tmp_path / "f.csv", T=T), roster=roster, **FAST)
    assert (gk_column(d, 0) == 4).all() and (gk_column(d, 1) == 0).all()


def test_short_stint_nearest_to_goal_is_not_a_goalkeeper(tmp_path):
    # column 11 stands in goal for 20 s while column 0 is off: < min_gk_s (60 s)
    p = fused_csv(tmp_path / "f.csv", T=T, n_cols=12, spot={(0, 11): 50.0},
                  extra={(0, 11): (1000, 1500), (1, 11): (0, 0)}, drop={(0, 0): [(1000, 1500)]})
    with pytest.warns(UserWarning, match="home: 12.5% of frames"):
        d = prep.load(p, **FAST)
    assert d["player_is_gk"][0].tolist() == [True] + [False] * 11
    assert (gk_column(d, 0)[500:750] == -1).all()


def test_warning_when_a_team_has_no_goalkeeper(tmp_path):
    roster = {f"{s}_{s}_{j + 1}": (s, j) == ("home", 0) for s in ("home", "away") for j in range(11)}
    with pytest.warns(UserWarning, match=r"away: 100.0% of frames .* exactly one goalkeeper"):
        prep.load(fused_csv(tmp_path / "f.csv", T=T), roster=roster, **FAST)


def test_dead_ball_frames_are_not_checked(tmp_path):
    # no goalkeeper column while the ball is dead (e.g. walk-on before a kick-off): no warning
    p = fused_csv(tmp_path / "f.csv", T=T, drop={(0, 0): [(0, 400)]})
    df = pd.read_csv(p)
    df.loc[:399, "ball_status"] = "dead"
    df.to_csv(p, index=False)
    with no_warnings():
        d = prep.load(p, **FAST)
    assert d["is_gk"][:200, 0].sum() == 0 and (d["is_gk"][200:, 0].sum(1) == 1).all()


def test_roster_missing_columns_warn(tmp_path):
    roster = {"home_home_1": True, "away_away_1": True}
    with pytest.warns(UserWarning, match="no roster entry"):
        d = prep.load(fused_csv(tmp_path / "f.csv", T=T), roster=roster, **FAST)
    assert (d["is_gk"].sum(-1) == 1).all()


def test_roster_from_meta(tmp_path):
    players = [dict(column="home_home_26", team="home", shirt_num=26, name="x", position="Goalkeeper", is_gk=True),
               dict(column="home_home_4", team="home", shirt_num=4, name="y", position="Defender", is_gk=False)]
    m = M.load(meta_json(tmp_path / "meta.json", players=players))
    assert M.roster(m) == {"home_home_26": True, "home_home_4": False}
    with pytest.raises(ValueError, match="no players"):
        M.roster(M.load(meta_json(tmp_path / "m2.json")))
