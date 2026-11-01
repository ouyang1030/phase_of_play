"""phase.evaluate.timer_baseline: one timer for heuristic and event turnovers (review item 10)."""
import numpy as np
import pandas as pd
import pytest

from phase import evaluate as E
from phase import events as Ev
from synth import HZ

HOME, AWAY = 0, 1


# ---------------------------------------------------------------- the two implementations before the merge
def old_timer_baseline(poss_home, tau_s, hz):
    p = np.nan_to_num(poss_home, nan=0.0)
    pred = np.where(p == 0, 0, 2)
    chg = np.flatnonzero(np.diff(p) != 0) + 1
    n = int(round(tau_s * hz))
    for t in chg:
        pred[t:t + n] = 1 if p[t] == 0 else 3
    return pred


def old_timer_from_turnovers(poss_home, to, tau_s, hz, live_only=False):
    if to is None:
        return old_timer_baseline(poss_home, tau_s, hz)
    p = np.nan_to_num(poss_home, nan=0.0)
    pred = np.where(p == 0, 0, 2)
    n = int(round(tau_s * hz))
    t0s = to.idx.to_numpy()
    nxt = np.r_[t0s[1:], len(p)]
    for (t0, w, dead), t1 in zip(to[["idx", "team", "dead"]].itertuples(index=False), nxt):
        if live_only and dead:
            continue
        pred[t0:min(t0 + n, t1)] = 1 if w == 0 else 3
    return pred


# ---------------------------------------------------------------- synthetic possession
def random_possession(T=3000, seed=0):
    """Possession runs of 0.2-30 s, home first (no NaN)."""
    rng = np.random.default_rng(seed)
    p, t, team = np.zeros(T), 0, HOME
    while t < T:
        n = int(rng.uniform(0.2, 30) * HZ)
        p[t:t + n] = team
        t, team = t + n, 1 - team
    return p


def event_match():
    """Event table: home plays, away wins the ball in play (frame 100), the ball goes out (300), home throws in
    (330, dead-ball turnover at the stoppage frame 300), away intercepts (500), away fouls (600), home free kick
    (640, dead-ball turnover at 600)."""
    rows = [(0, HOME, 1), (50, HOME, 1), (100, AWAY, 8), (150, AWAY, 1), (300, AWAY, 5), (330, HOME, 1),
            (400, HOME, 1), (500, AWAY, 8), (520, AWAY, 1), (600, AWAY, 4), (640, HOME, 1), (700, HOME, 1)]
    e = pd.DataFrame([dict(idx=i, team=t, type=ty, ok=True, x=0.0, y=0.0, period_id=1, original_event="e",
                           event_id=k) for k, (i, t, ty) in enumerate(rows)])
    poss, to = Ev.possession(e, 900)
    return poss, to


# ---------------------------------------------------------------- regression: identical to the old code
@pytest.mark.parametrize("tau", [1.0, 4.5, 9.34, 20.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heuristic_path_matches_old_timer_frame_by_frame(tau, seed):
    p = random_possession(seed=seed)
    assert np.array_equal(E.timer_baseline(p, tau, HZ), old_timer_baseline(p, tau, HZ))


@pytest.mark.parametrize("tau", [1.0, 6.0, 20.0])
def test_turnover_path_matches_old_timer_frame_by_frame(tau):
    poss, to = event_match()
    assert to.dead.tolist() == [False, True, False, True]
    for live in (False, True):
        new = E.timer_baseline(poss, tau, HZ, turnovers=to, live_only=live)
        assert np.array_equal(new, old_timer_from_turnovers(poss, to, tau, HZ, live))
    # event possession changes exactly at the turnovers: both paths agree
    assert np.array_equal(E.timer_baseline(poss, tau, HZ), E.timer_baseline(poss, tau, HZ, turnovers=to))


def test_macro_f1_matches_evaluate():
    rng = np.random.default_rng(3)
    gt = rng.integers(-1, 4, 2000)
    for pr in (rng.integers(0, 4, 2000), np.zeros(2000, int), E.timer_baseline(random_possession(2000), 3, HZ)):
        assert E.macro_f1(gt, pr) == pytest.approx(E.evaluate(gt, pr, HZ)["macro_f1"], abs=1e-12)


# ---------------------------------------------------------------- new behaviour
def test_leading_nan_is_back_filled():
    p = np.r_[np.full(10, np.nan), np.ones(40), np.zeros(50)]  # away has the first touch at frame 10
    new = E.timer_baseline(p, 2.0, HZ)
    assert (new[:50] == 2).all()  # est_defense before and after the first touch, no transition at the first touch
    assert (new[50:75] == 1).all() and (new[75:] == 0).all()
    old = old_timer_baseline(p, 2.0, HZ)
    assert (old[:10] == 0).all() and (old[10:35] == 3).all()  # old: home before the first touch, then att_to_def


def test_alive_cuts_the_window_at_the_next_stoppage():
    p = np.r_[np.zeros(100), np.ones(200)]
    alive = np.ones(300, bool)
    alive[130:160] = False
    pred = E.timer_baseline(p, 10.0, HZ, alive=alive)  # 125 frames
    assert (pred[100:130] == 3).all() and (pred[130:] == 2).all()
    assert (E.timer_baseline(p, 10.0, HZ)[100:225] == 3).all()  # without alive: full window


def test_turnover_at_a_stoppage_keeps_its_window_through_it():
    poss, to = event_match()
    alive = np.ones(900, bool)
    alive[300:330] = False  # ball out at 300, home throw-in at 330
    alive[600:640] = False  # foul at 600, away restart at 640
    alive[360:370] = False  # a later stoppage cuts the home transition
    pred = E.timer_baseline(poss, 20.0, HZ, alive=alive, turnovers=to)
    assert (pred[300:360] == 1).all() and (pred[360:500] == 0).all()
    assert (E.timer_baseline(poss, 20.0, HZ, alive=alive, turnovers=to, live_only=True)[300:500] == 0).all()
    # the in-play turnover at 500 is cut by the foul at 600; the dead-ball turnover at 600 runs through the restart
    assert (pred[500:600] == 3).all() and (pred[600:850] == 1).all() and (pred[850:] == 0).all()


def test_next_turnover_cuts_the_window():
    poss, to = event_match()
    pred = E.timer_baseline(poss, 30.0, HZ, turnovers=to)
    assert (pred[100:300] == 3).all() and (pred[300:500] == 1).all() and (pred[500:600] == 3).all()


def test_live_only_needs_turnovers():
    with pytest.raises(ValueError, match="turnover table"):
        E.timer_baseline(np.zeros(10), 1.0, HZ, live_only=True)
