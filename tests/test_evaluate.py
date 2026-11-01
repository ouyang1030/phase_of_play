"""phase.evaluate: boundary matching, typed boundary F1, per-run sufficient statistics, bootstrap (review item 11)."""
import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score, f1_score

from phase import evaluate as E
from synth import HZ


def brute_force_matches(g, p, tol):
    if not len(g) or not len(p):
        return 0
    ok = (np.abs(np.asarray(g)[:, None] - np.asarray(p)[None]) <= tol).astype(int)
    r, c = linear_sum_assignment(-ok)
    return int(ok[r, c].sum())


def old_boundary_tp(gt_b, pr_b, tol):  # the matching before the fix: predictions in the given order, nearest free GT
    used = np.zeros(len(gt_b), bool)
    tp = 0
    for p in pr_b:
        if len(gt_b) == 0:
            break
        d = np.abs(np.asarray(gt_b) - p).astype(float)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j], tp = True, tp + 1
    return tp


def old_segmental_f1(gt, pr, runs, iou_thr):
    tp = fp = fn = 0
    for a, b in runs:
        g = gt[a:b].copy()
        p = np.where(g >= 0, pr[a:b], -1)
        gs, ps = E.segments(g), E.segments(p)
        hit = np.zeros(len(gs), bool)
        for s, e, c in ps:
            best, bj = 0.0, -1
            for j, (s2, e2, c2) in enumerate(gs):
                if c2 == c:
                    iou = max(0, min(e, e2) - max(s, s2)) / (max(e, e2) - min(s, s2))
                    if iou > best:
                        best, bj = iou, j
            if best >= iou_thr and not hit[bj]:
                tp, hit[bj] = tp + 1, True
            else:
                fp += 1
        fn += int((~hit).sum())
    p_, r_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * p_ * r_ / max(p_ + r_, 1e-9)


def labels(T=3000, seed=0, p_unlabeled=0.1, mean_run=40):
    """Random label sequence with runs of 1-2*mean_run frames and unlabeled stretches."""
    rng = np.random.default_rng(seed)
    y, t = np.full(T, -1), 0
    while t < T:
        n = int(rng.integers(1, 2 * mean_run))
        y[t:t + n] = -1 if rng.random() < p_unlabeled else rng.integers(0, 4)
        t += n
    return y


def noisy(y, seed, p_flip=0.05, shift=3):
    rng = np.random.default_rng(seed)
    p = np.roll(np.where(y >= 0, y, 0), shift)
    flip = rng.random(len(y)) < p_flip
    p[flip] = rng.integers(0, 4, flip.sum())
    return p


# ---------------------------------------------------------------- boundary matching
def test_matching_is_maximal_and_order_independent():
    rng = np.random.default_rng(0)
    for _ in range(300):
        g = rng.choice(60, rng.integers(0, 8), replace=False)
        p = rng.choice(60, rng.integers(0, 8), replace=False)
        tol = int(rng.integers(0, 6))
        best = brute_force_matches(g, p, tol)
        assert E.match_boundaries(g, p, tol) == best
        assert E.match_boundaries(g[::-1], rng.permutation(p), tol) == best
        assert old_boundary_tp(g, p, tol) <= best


def test_old_greedy_matching_lost_pairs():
    gt, pr = np.array([10, 13]), np.array([12, 15])  # 12 grabs the nearer 13, 15 is then left without a partner
    assert old_boundary_tp(gt, pr, 3) == 1
    assert E.match_boundaries(gt, pr, 3) == 2
    assert E.boundary_f1(gt, pr, 3) == (1.0, 1.0, 1.0)
    assert E.boundary_f1([], [], 3) == (1.0, 1.0, 1.0)


def test_typed_boundaries_need_the_same_class_change():
    gt = np.r_[np.zeros(20), np.ones(20), np.full(20, 2)].astype(int)  # 0->1 at 20, 1->2 at 40
    pr = np.r_[np.zeros(21), np.full(19, 3), np.full(20, 2)].astype(int)  # 0->3 at 21, 3->2 at 40
    r = E.evaluate(gt, pr, HZ)
    assert r["bnd_f1@0.5s"] == 1.0
    assert r["bnd_typed_f1@0.5s"] == 0.0
    r = E.evaluate(gt, np.r_[np.zeros(22), np.ones(18), np.full(20, 2)].astype(int), HZ)
    assert r["bnd_typed_f1@0.5s"] == 1.0  # same types, 2 frames late


def test_boundaries_are_matched_inside_their_run_only():
    gt = np.r_[np.zeros(30), np.full(20, -1), np.ones(30)].astype(int)  # two runs (gap 1.6 s), no boundary
    pr = np.r_[np.zeros(28), np.ones(52)].astype(int)  # boundary at 28 inside run 1, 22 frames from run 2's start
    r = E.evaluate(gt, pr, HZ)
    assert r["n_gt_boundaries"] == 0 and r["n_pred_boundaries"] == 1
    assert r["bnd_f1@3s"] == 0.0


# ---------------------------------------------------------------- evaluate from sufficient statistics
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frame_and_segment_metrics_unchanged(seed):
    gt = labels(seed=seed)
    pr = noisy(gt, seed)
    r = E.evaluate(gt, pr, HZ)
    m = gt >= 0
    y, p = gt[m], pr[m]
    per = f1_score(y, p, labels=range(4), average=None, zero_division=0)
    assert r["acc"] == pytest.approx(float((y == p).mean()), abs=1e-12)
    assert r["macro_f1"] == pytest.approx(float(per.mean()), abs=1e-12)
    assert r["f1_per_class"] == pytest.approx(per.tolist(), abs=1e-12)
    assert r["kappa"] == pytest.approx(cohen_kappa_score(y, p), abs=1e-12)
    tr = f1_score(np.isin(y, E.TRANSITIONS), np.isin(p, E.TRANSITIONS), zero_division=0)
    assert r["transition_f1"] == pytest.approx(tr, abs=1e-12)
    runs = E.eval_runs(gt, int(HZ))
    for thr in E.IOU_THRS:
        assert r[f"seg_f1@{thr}"] == pytest.approx(old_segmental_f1(gt, pr, runs, thr), abs=1e-12)
    assert r["n_gt_boundaries"] == len(E.boundaries(gt, runs))


def test_resampled_statistics_equal_evaluate_on_the_resampled_match():
    """A bootstrap replicate with weights (2, 0, 1, ...) is evaluate() on a match made of those runs."""
    gt, pr = labels(seed=4), noisy(labels(seed=4), 4)
    runs = E.eval_runs(gt, int(HZ))
    st = E.run_stats(gt, pr, HZ, runs)
    w = np.random.default_rng(1).multinomial(len(runs), np.full(len(runs), 1 / len(runs))).astype(float)
    res = E.metrics({k: np.tensordot(w, v, axes=(0, 0)) for k, v in st.items()})
    gap = np.full(int(3 * HZ), -1)  # wider than the run bridge and the largest tolerance
    parts_g, parts_p = [], []
    for (a, b), n in zip(runs, w.astype(int)):
        for _ in range(n):
            parts_g += [gt[a:b], gap]
            parts_p += [pr[a:b], np.zeros_like(gap)]
    ref = E.evaluate(np.concatenate(parts_g), np.concatenate(parts_p), HZ)
    for k in E.SCALAR_METRICS:
        assert float(res[k]) == pytest.approx(ref[k], abs=1e-12), k


def test_bootstrap_intervals_and_paired_differences():
    gt = labels(T=6000, seed=5)
    good, bad = noisy(gt, 1, p_flip=0.02, shift=1), noisy(gt, 2, p_flip=0.4, shift=6)
    out = E.bootstrap(gt, {"good": good, "bad": bad, "good_copy": good.copy()}, HZ, n_boot=300, seed=0)
    assert out["n_units"] == len(E.eval_runs(gt, int(HZ))) and out["n_boot"] == 300
    for name, pr in (("good", good), ("bad", bad)):
        ref = E.evaluate(gt, pr, HZ)
        for m in E.SCALAR_METRICS:
            c = out["ci"][name][m]
            assert c["point"] == pytest.approx(ref[m], abs=1e-12)
            assert c["lo"] <= c["point"] <= c["hi"], (name, m)
        assert len(out["ci"][name]["f1_per_class"]) == 4
    d = out["diff"]["good - bad"]
    assert d["macro_f1"]["diff"] > 0 and d["macro_f1"]["lo"] > 0 and d["macro_f1"]["p_le_0"] == 0.0
    same = out["diff"]["good - good_copy"]["macro_f1"]
    assert same["diff"] == 0 and same["lo"] == same["hi"] == 0 and same["p_le_0"] == 1.0
    assert set(out["diff"]) == {f"{a} - {b}" for a, b in itertools.combinations(["good", "bad", "good_copy"], 2)}


def test_bootstrap_is_reproducible():
    gt = labels(seed=6)
    a = E.bootstrap(gt, {"x": noisy(gt, 0)}, HZ, n_boot=50, seed=3)
    b = E.bootstrap(gt, {"x": noisy(gt, 0)}, HZ, n_boot=50, seed=3)
    assert a["ci"] == b["ci"]
