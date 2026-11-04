"""phase.episodes: labels = possession + one transition length per possession (contribution 1)."""
import numpy as np
import pandas as pd
import pytest

from phase import evaluate as E
from phase import episodes as P


def poss_runs(*runs):
    """[(team, n), ...] -> possession array."""
    return np.concatenate([np.full(n, float(t)) for t, n in runs])


def test_fill_possession_keeps_previous_possessor_in_gaps():
    assert E.fill_possession(np.array([np.nan, 0, np.nan, np.nan, 1, np.nan])).tolist() == [0, 0, 0, 0, 1, 1]


def test_label_possession():
    out = P.label_possession(np.array([-1, 0, 1, 2, 3]))
    assert np.isnan(out[0]) and out[1:].tolist() == [0, 0, 1, 1]


def test_windows_end_at_next_turnover_dead_ball_max_and_period():
    poss = poss_runs((0, 10), (1, 10), (0, 10))
    alive, period = np.ones(30, bool), np.ones(30, int)
    eps = P.episodes(poss, alive, period, hz=1.0, max_s=5)
    assert eps[["t0", "s0", "t1", "team"]].values.tolist() == [[10, 10, 15, 1], [20, 20, 25, 0]]
    assert not eps.dead.any()
    alive[12:14] = False  # stoppage inside the first window
    alive[18:22] = False  # second turnover during a stoppage: its clock starts at the restart
    period[24:] = 2
    eps = P.episodes(poss, alive, period, hz=1.0, max_s=5)
    assert eps[["t0", "s0", "t1"]].values.tolist() == [[10, 10, 12], [20, 22, 24]]
    assert eps.dead.tolist() == [False, True]


def test_turnover_table_and_empty_windows():
    poss = poss_runs((0, 10), (1, 10), (0, 10))
    alive = np.ones(30, bool)
    alive[9:21] = False  # both turnovers inside one stoppage: the first window is empty
    to = pd.DataFrame(dict(idx=[10, 20], team=[1, 0], dead=[True, True], how=["out", "other"]))
    eps = P.episodes(poss, alive, np.ones(30, int), hz=1.0, max_s=30, turnovers=to)
    assert eps[["s0", "t1"]].values.tolist() == [[21, 21], [21, 30]]
    assert eps.how.tolist() == ["out", "other"] and P.lengths(eps).tolist() == [0, 9]


@pytest.mark.parametrize("y,lo,hi,kind,lag", [
    ([3, 3, 3, 2, 2], 3, 3, "exact", 0),
    ([2, 2, 2, 2, 2], 0, 0, "exact", 0),  # no transition labelled
    ([3, 3, -1, -1, 2], 2, 4, "interval", 0),  # unlabelled gap between the transition and the established class
    ([3, 3, 3, 3, 3], 5, 5, "exact", 0),  # transition through the whole window (overflow)
    ([0, 0, 3, 3, 2], 4, 4, "exact", 2),  # annotator still shows the old possession for 2 frames
    ([-1, -1, -1, -1, -1], 0, 5, "none", 0),
])
def test_observe(y, lo, hi, kind, lag):
    label = np.r_[np.zeros(5, int), y]
    eps = pd.DataFrame(dict(t0=[5], s0=[5], t1=[10], team=[1], dead=[False], how=["other"]))
    o = P.observe(label, eps).iloc[0]
    assert (o.lo, o.hi, o.kind, o.lag) == (lo, hi, kind, lag)


def test_observe_censors_outside_the_training_region():
    label = np.r_[np.zeros(5, int), 3, 3, 3, 3, 2]
    eps = pd.DataFrame(dict(t0=[5], s0=[5], t1=[10], team=[1], dead=[False], how=["other"]))
    allowed = np.ones(10, bool)
    allowed[7:] = False
    o = P.observe(label, eps, allowed).iloc[0]
    assert (o.lo, o.hi, o.kind) == (2, 5, "interval")  # E >= 2, possibly the whole window


def random_match(rng, T=400):
    runs, n = [], 0
    while n < T:
        k = int(rng.integers(5, 40))
        runs.append((len(runs) % 2, k))
        n += k
    poss = poss_runs(*runs)[:T]
    alive = np.ones(T, bool)
    for s in rng.integers(0, T - 10, 4):
        alive[s:s + int(rng.integers(1, 6))] = False
    return poss, alive


@pytest.mark.parametrize("seed", range(5))
def test_reconstruct_and_oracle_reproduce_grammar_conforming_labels(seed):
    rng = np.random.default_rng(seed)
    poss, alive = random_match(rng)
    eps = P.episodes(poss, alive, np.ones(len(poss), int), hz=1.0, max_s=25)
    E_true = rng.integers(0, P.lengths(eps) + 1)
    label = P.reconstruct(poss, eps, E_true)
    label[~alive] = -1
    label[rng.random(len(label)) < 0.1] = -1  # unlabelled frames
    pred = P.reconstruct(poss, eps, P.oracle_lengths(label, poss, eps))
    m = label >= 0
    assert (pred[m] == label[m]).all()
    obs = P.observe(label, eps)
    ok = obs.kind != "none"
    assert ((obs.lo <= E_true) & (E_true <= obs.hi))[ok].all()  # the observation always contains the truth


@pytest.mark.parametrize("seed", range(5))
def test_scorer_equals_frame_macro_f1(seed):
    rng = np.random.default_rng(seed)
    poss, alive = random_match(rng)
    eps = P.episodes(poss, alive, np.ones(len(poss), int), hz=1.0, max_s=25)
    gt = rng.integers(-1, 4, len(poss))
    sc = P.EpisodeScorer(gt, poss, eps)
    for _ in range(5):
        Ev = rng.integers(0, P.lengths(eps) + 3)  # beyond W is clipped like reconstruct does
        assert sc.macro_f1(Ev) == pytest.approx(E.macro_f1(gt, P.reconstruct(poss, eps, Ev)), abs=1e-12)


def test_constant_length_is_the_timer():
    rng = np.random.default_rng(0)
    poss, _ = random_match(rng)
    alive = np.ones(len(poss), bool)
    alive[100:103] = False  # a stoppage inside a possession cuts its window, as timer_baseline(alive=...) does
    poss[100:103] = poss[99]
    eps = P.episodes(poss, alive, np.ones(len(poss), int), hz=12.5, max_s=30)
    for tau in (0.5, 2.0, 6.0):
        timer = E.timer_baseline(poss, tau, 12.5, alive)
        assert (P.reconstruct(poss, eps, np.full(len(eps), int(round(tau * 12.5)))) == timer).all()
    gt = E.timer_baseline(poss, 2.0, 12.5, alive)
    tau, f1 = P.tune_constant(P.EpisodeScorer(gt, poss, eps), 12.5)
    assert tau == 2.0 and f1 == pytest.approx(1.0)


def test_cv_folds():
    label = np.r_[np.zeros(100, int), np.full(20, -1)]
    L0, L1, folds = P.cv_folds(label, 4, 5)
    assert (L0, L1) == (0, 100) and [(a, b) for a, b, _ in folds] == [(0, 25), (25, 50), (50, 75), (75, 100)]
    m = folds[1][2]
    assert not m[20:55].any() and m[:20].all() and m[55:100].all() and not m[100:].any()
