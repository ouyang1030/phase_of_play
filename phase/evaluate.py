"""Decoding (mode filter, HSMM-lite), timer baseline, and evaluation metrics (STUDY1_PLAN §8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score

K = 4
TRANSITIONS = (1, 3)


# ---------------------------------------------------------------- baselines / decoding
def fill_possession(poss_home: np.ndarray) -> np.ndarray:
    """Possession per frame without NaN: frames before the first touch take the first possessor (back-fill);
    any later gap keeps the previous possessor; no possession at all -> home (0)."""
    return pd.Series(np.asarray(poss_home, float)).bfill().ffill().fillna(0.0).to_numpy()


def established(poss_home: np.ndarray) -> np.ndarray:
    """Established class of the possessor: 0 (est_attack) where home has the ball, 2 (est_defense) otherwise."""
    return np.where(fill_possession(poss_home) == 0, 0, 2)


def timer_baseline(poss_home: np.ndarray, tau_s: float, hz: float, alive: np.ndarray | None = None,
                   turnovers: pd.DataFrame | None = None, live_only: bool = False) -> np.ndarray:
    """L_fix(tau): a turnover opens a transition of the new possessor for tau s, otherwise established of possessor.

    Turnovers are the changes of `poss_home` (NaN before the first touch is back-filled, so the first touch is no
    turnover) or, if given, the rows of an event turnover table (columns idx, team, dead; phase.events.possession).
    The window of a turnover at t0 is [t0, min(t0 + n, next turnover, next dead ball)), n = round(tau_s * hz).
    The dead-ball cut applies only with `alive`: it is the first frame after t0 where the ball goes from alive to
    dead, so a turnover at a stoppage (table column dead) keeps its window through that stoppage.
    live_only (needs `turnovers`): turnovers across a dead ball open no transition (the next turnover still cuts
    the window before them).
    """
    p = fill_possession(poss_home)
    T = len(p)
    pred = np.where(p == 0, 0, 2)
    n = int(round(tau_s * hz))
    if turnovers is None:
        if live_only:
            raise ValueError("live_only needs the event turnover table (dead-ball flag)")
        t0s = np.flatnonzero(np.diff(p) != 0) + 1
        team, dead = p[t0s].astype(int), np.zeros(len(t0s), bool)
    elif len(turnovers):
        t0s = turnovers["idx"].to_numpy(int)
        team, dead = turnovers["team"].to_numpy(int), turnovers["dead"].to_numpy(bool)
    else:
        return pred
    t1s = np.r_[t0s[1:], T]
    if alive is not None:
        a = np.asarray(alive, bool)
        onset = np.flatnonzero(a[:-1] & ~a[1:]) + 1  # first dead frame of each stoppage
        k = np.searchsorted(onset, t0s, side="right")
        t1s = np.minimum(t1s, np.r_[onset, T][k])
    for t0, t1, w, dd in zip(t0s, t1s, team, dead):
        if live_only and dd:
            continue
        pred[t0:min(t0 + n, t1)] = 1 if w == 0 else 3
    return pred


def mode_filter(pred: np.ndarray, win: int) -> np.ndarray:
    """Rolling majority vote (Bassek 2023 / Li & Link 2026 style post-processing)."""
    oh = np.eye(K)[pred]
    c = np.cumsum(np.vstack([np.zeros((1, K)), oh]), 0)
    h = win // 2
    idx = np.arange(len(pred))
    lo, hi = np.clip(idx - h, 0, len(pred)), np.clip(idx + h + 1, 0, len(pred))
    return np.argmax(c[hi] - c[lo], 1)


def fit_hsmm(labels: np.ndarray, hz: float, min_frac: float = 0.5, eps: float = 1e-3):
    """Min-duration per class and transition matrix from training labels (labeled runs only)."""
    segs = segments(labels)
    dur = {c: [b - a for a, b, cc in segs if cc == c] for c in range(K)}
    min_d = np.array([max(1, int(min_frac * np.percentile(dur[c], 10))) if dur[c] else int(hz) for c in range(K)])
    A = np.full((K, K), eps)
    for (a0, b0, c0), (a1, b1, c1) in zip(segs[:-1], segs[1:]):
        if a1 - b0 <= hz and c0 != c1:  # adjacent within 1 s
            A[c0, c1] += 1
    np.fill_diagonal(A, 0)
    A /= A.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        logA = np.log(A)  # diagonal -> -inf (self-transitions live inside the chain)
    return dict(min_d=min_d, logA=logA)


def hsmm_decode(logp: np.ndarray, params: dict, switch_penalty: float = 2.0) -> np.ndarray:
    """Viterbi over an expanded chain (each class = min_d[c] sub-states, last one self-loops).

    Enforces a minimum segment length and learned class-to-class transitions, i.e. an
    explicit-duration (semi-Markov) prior with a geometric tail beyond the minimum.
    """
    T = len(logp)
    min_d, logA = params["min_d"], params["logA"]
    offs = np.r_[0, np.cumsum(min_d)]
    S = offs[-1]
    cls_of = np.repeat(np.arange(K), min_d)
    first, last = offs[:-1], offs[1:] - 1

    delta = np.full(S, -np.inf)
    delta[first] = logp[0]
    back = np.zeros((T, S), np.int32)
    for t in range(1, T):
        new = np.full(S, -np.inf)
        bp = np.zeros(S, np.int32)
        # advance along each chain
        chain = np.ones(S, bool)
        chain[first] = False
        new[chain] = delta[np.flatnonzero(chain) - 1]
        bp[chain] = np.flatnonzero(chain) - 1
        # self-loop on last sub-state
        stay = delta[last]
        better = stay > new[last]
        new[last] = np.where(better, stay, new[last])
        bp[last] = np.where(better, last, bp[last])
        # enter class c' from the last sub-state of c
        cand = delta[last][:, None] + logA - switch_penalty  # (from, to)
        src = np.argmax(cand, 0)
        enter = cand[src, np.arange(K)]
        better = enter > new[first]
        new[first] = np.where(better, enter, new[first])
        bp[first] = np.where(better, last[src], bp[first])
        delta = new + logp[t][cls_of]
        back[t] = bp
    s = int(np.argmax(delta))
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = cls_of[s]
        s = back[t, s]
    return path


def possession_decode(logp: np.ndarray, poss_home: np.ndarray, hz: float,
                      pre_s: float = 0.0, lag_s: float = 0.0) -> np.ndarray:
    """Viterbi under the possession structure of the labels (STUDY1_PLAN §16.1).

    Inside a possession of team w the labels run [transition_w]* [established_w]* (one change
    point, E = 0 allowed). Classes of the other team are only allowed within [t0 - pre_s, t0 + lag_s]
    of a turnover t0 (annotator switching before / after the event); est_w -> trans_w only there too.
    """
    T = len(logp)
    team = np.array([0, 0, 1, 1])
    is_trans = np.array([False, True, False, True])
    p = poss_home.copy()
    known = ~np.isnan(p)
    t0s = np.flatnonzero(known[1:] & known[:-1] & (p[1:] != p[:-1])) + 1
    win = np.zeros(T, bool)
    pre, lag = int(round(pre_s * hz)), int(round(lag_s * hz))
    for t0 in t0s:
        win[max(0, t0 - pre):min(T, t0 + lag + 1)] = True
    win[t0s] = True
    same = team[:, None] == team[None, :]
    base = np.eye(4, dtype=bool) | (same & is_trans[:, None] & ~is_trans[None, :])  # stay, trans_w -> est_w
    free = np.ones((4, 4), bool)
    neg = -1e9
    delta = logp[0] + np.where(known[0] & (team != p[0]), neg, 0.0) * (not win[0])
    back = np.zeros((T, 4), np.int8)
    for t in range(1, T):
        M = free if (win[t] or not known[t]) else base
        allowed = np.ones(4, bool) if (win[t] or not known[t]) else team == p[t]
        cand = delta[:, None] + np.where(M, 0.0, neg)
        back[t] = np.argmax(cand, 0)
        delta = cand[back[t], np.arange(4)] + logp[t] + np.where(allowed, 0.0, neg)
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def fit_durations(labels: list[np.ndarray], hz: float, classes=TRANSITIONS) -> dict:
    """Log-normal transition durations (s) per class, right-censored MLE.

    A run is observed if the next frame carries another label; it is censored if the next frame
    is unlabeled (stoppage, masked region) or the end of the array.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    out = {}
    for k in classes:
        obs, cen = [], []
        for y in labels:
            ch = np.flatnonzero(np.diff(np.r_[-9, y, -9]) != 0)
            for a, b in zip(ch[:-1], ch[1:]):
                if y[a] != k:
                    continue
                (obs if b < len(y) and y[b] >= 0 else cen).append((b - a) / hz)
        lo, lc = np.log(obs), np.log(cen)

        def nll(th):
            mu, s = th[0], np.exp(th[1])
            return -(norm.logpdf(lo, mu, s).sum() - lo.sum() + norm.logsf(lc, mu, s).sum())

        r = minimize(nll, x0=[np.log(8.0), np.log(0.5)], method="Nelder-Mead")
        out[k] = dict(mu=float(r.x[0]), sigma=float(np.exp(r.x[1])), n_obs=len(obs), n_cen=len(cen))
    return out


def edhsmm_decode(logp: np.ndarray, alive: np.ndarray, dur: dict, hz: float,
                  switch_penalty: float = 2.0, hard: bool = False, max_s: float = 30.0) -> np.ndarray:
    """Explicit-duration decoding (STUDY1_PLAN §7): transitions are segments with a log-normal
    duration pmf (`fit_durations`), established classes self-loop. Decoded per alive run; a
    transition cut by a stoppage scores with the survival function (censoring).
    hard=True forbids est_attack <-> est_defense inside a run (only via a transition or a stoppage)."""
    from scipy.stats import norm

    Dm = int(round(max_s * hz))
    d_s = np.arange(1, Dm + 1) / hz
    logP, logS = {}, {}
    for k, p in dur.items():
        cdf = lambda x: norm.cdf((np.log(np.maximum(x, 1e-9)) - p["mu"]) / p["sigma"])
        pm = cdf(d_s + 0.5 / hz) - cdf(d_s - 0.5 / hz)
        logP[k] = np.log(pm / pm.sum() + 1e-12)
        logS[k] = np.log(1 - cdf(d_s - 0.5 / hz) + 1e-12)
    path = logp.argmax(1)
    ch = np.flatnonzero(np.diff(np.r_[0, alive.astype(int), 0]))
    for a, b in zip(ch[::2], ch[1::2]):
        path[a:b] = _ed_viterbi(logp[a:b], logP, logS, switch_penalty, hard, Dm)
    return path


def _ed_viterbi(lp, logP, logS, lam, hard, Dm):
    n = len(lp)
    est, trans = (0, 2), (1, 3)
    cs = np.vstack([np.zeros(K), np.cumsum(lp, 0)])
    Ev = np.full((n, K), -np.inf)
    Tv = np.full((n, K), -np.inf)
    entry = np.full((n, K), -np.inf)
    bE = np.full((n, K), -1, np.int8)
    bEn = np.full((n, K), -1, np.int8)
    bT = np.zeros((n, K), np.int32)
    for t in range(n):
        if t == 0:
            entry[0] = 0.0
        else:
            prev = np.array([Ev[t - 1, 0], Tv[t - 1, 1], Ev[t - 1, 2], Tv[t - 1, 3]]) - lam
            for x in range(K):
                cand = prev.copy()
                cand[x] = -np.inf  # same class: stay (est) / no split (trans)
                if hard and x in est:
                    cand[2 - x] = -np.inf
                j = int(np.argmax(cand))
                entry[t, x], bEn[t, x] = cand[j], j
        for c in est:
            if t == 0:
                Ev[t, c], bE[t, c] = lp[t, c], -1
            elif Ev[t - 1, c] >= entry[t, c]:
                Ev[t, c], bE[t, c] = Ev[t - 1, c] + lp[t, c], c
            else:
                Ev[t, c], bE[t, c] = entry[t, c] + lp[t, c], bEn[t, c]
        dmax = min(t + 1, Dm)
        ss = t - np.arange(dmax)  # segment starts for d = 1..dmax
        for k in trans:
            sc = entry[ss, k] + cs[t + 1, k] - cs[ss, k] + logP[k][:dmax]
            j = int(np.argmax(sc))
            Tv[t, k], bT[t, k] = sc[j], j + 1
    # final state; a transition still running at the end of the run is censored
    t = n - 1
    dmax = min(n, Dm)
    ss = t - np.arange(dmax)
    fin = {c: (Ev[t, c], None) for c in est}
    for k in trans:
        sc = entry[ss, k] + cs[n, k] - cs[ss, k] + logS[k][:dmax]
        j = int(np.argmax(sc))
        fin[k] = (sc[j], j + 1)
    x = max(fin, key=lambda c: fin[c][0])
    path = np.zeros(n, int)
    first = True
    while t >= 0 and x >= 0:
        if x in est:
            path[t] = x
            x, t = int(bE[t, x]), t - 1
        else:
            dd = fin[x][1] if first else int(bT[t, x])
            s = t - dd + 1
            path[s:t + 1] = x
            x, t = int(bEn[s, x]), s - 1
        first = False
    return path


def boundary_refine(probs: np.ndarray, bnd: np.ndarray, thr: float = 0.5,
                    alive: np.ndarray | None = None) -> np.ndarray:
    """ASRF refinement (Ishikawa et al., WACV 2021, §3.4): cut at boundary-probability local maxima
    above thr, then majority vote of the frame-wise argmax inside each segment.
    With `alive`, ball dead/alive changes are added as cuts (segments never span a stoppage)."""
    peak = np.r_[False, (bnd[1:-1] > bnd[:-2]) & (bnd[1:-1] >= bnd[2:]), False] & (bnd > thr)
    if alive is not None:
        peak[1:] |= alive[1:] != alive[:-1]
    cuts = np.r_[0, np.flatnonzero(peak), len(bnd)]
    pred = probs.argmax(1)
    out = np.empty_like(pred)
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            out[a:b] = np.bincount(pred[a:b], minlength=K).argmax()
    return out


# ---------------------------------------------------------------- metrics
def macro_f1(gt: np.ndarray, pr: np.ndarray) -> float:
    """Macro F1 over the K classes on frames gt >= 0 (= evaluate()['macro_f1'], without the boundary metrics)."""
    m = gt >= 0
    C = np.bincount(gt[m].astype(int) * K + pr[m].astype(int), minlength=K * K).reshape(K, K)
    tp = np.diag(C).astype(float)
    den = 2 * tp + (C.sum(0) - tp) + (C.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def segments(y: np.ndarray, max_gap: int = 0):
    """Contiguous runs of the same label (>= 0): list of (start, end, class)."""
    out = []
    t, T = 0, len(y)
    while t < T:
        if y[t] < 0:
            t += 1
            continue
        s = t
        while t < T and y[t] == y[s]:
            t += 1
        out.append((s, t, int(y[s])))
    return out


def eval_runs(y: np.ndarray, max_gap: int):
    """Stretches of annotated play (unlabeled gaps <= max_gap are bridged)."""
    lab = np.flatnonzero(y >= 0)
    if len(lab) == 0:
        return []
    cuts = np.flatnonzero(np.diff(lab) > max_gap + 1)
    starts = np.r_[lab[0], lab[cuts + 1]]
    ends = np.r_[lab[cuts], lab[-1]] + 1
    return list(zip(starts, ends))


def typed_boundaries(y: np.ndarray, runs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label changes inside each run, unlabeled frames skipped: (frame, class before, class after)."""
    pos, frm, to = [], [], []
    for a, b in runs:
        seg = y[a:b]
        idx = np.flatnonzero(seg >= 0)
        v = seg[idx]
        ch = np.flatnonzero(v[1:] != v[:-1]) + 1
        pos += list(a + idx[ch])
        frm += list(v[ch - 1])
        to += list(v[ch])
    return np.array(pos, int), np.array(frm, int), np.array(to, int)


def boundaries(y: np.ndarray, runs) -> np.ndarray:
    return typed_boundaries(y, runs)[0]


def match_boundaries(gt_b, pr_b, tol: int) -> int:
    """Largest number of (gt, pred) pairs with |gt - pred| <= tol, each boundary in at most one pair.

    Two pointers over both sorted lists: a boundary that cannot reach the other list's current one cannot reach
    any later one either, and pairing the two earliest reachable ones never loses a pair. Independent of the
    order in which the boundaries are given."""
    g, p = np.sort(np.asarray(gt_b)), np.sort(np.asarray(pr_b))
    i = j = tp = 0
    while i < len(g) and j < len(p):
        if abs(int(g[i]) - int(p[j])) <= tol:
            tp, i, j = tp + 1, i + 1, j + 1
        elif p[j] < g[i]:
            j += 1
        else:
            i += 1
    return tp


def match_typed_boundaries(gt_tb, pr_tb, tol: int) -> int:
    """As match_boundaries, but a pair also needs the same (class before, class after)."""
    (gp, gf, gt), (pp, pf, pt) = gt_tb, pr_tb
    tp = 0
    for f, t in set(zip(gf.tolist(), gt.tolist())) & set(zip(pf.tolist(), pt.tolist())):
        tp += match_boundaries(gp[(gf == f) & (gt == t)], pp[(pf == f) & (pt == t)], tol)
    return tp


def _prf(tp, n_pred, n_gt):
    """Precision, recall, F1 from counts (arrays allowed); no boundaries at all on either side -> 1."""
    tp, n_pred, n_gt = (np.asarray(x, float) for x in (tp, n_pred, n_gt))
    prec, rec = tp / np.maximum(n_pred, 1), tp / np.maximum(n_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    empty = (n_pred == 0) & (n_gt == 0)
    return np.where(empty, 1.0, prec), np.where(empty, 1.0, rec), np.where(empty, 1.0, f1)


def boundary_f1(gt_b, pr_b, tol: int):
    """(precision, recall, F1) of predicted boundaries within +-tol frames of a GT boundary (maximum matching)."""
    return tuple(float(x) for x in _prf(match_boundaries(gt_b, pr_b, tol), len(pr_b), len(gt_b)))


def _segment_counts(g: np.ndarray, p: np.ndarray, iou_thrs) -> np.ndarray:
    """MS-TCN F1@IoU counts (tp, fp, fn) per threshold for one stretch; p is already masked to g >= 0."""
    gs, ps = segments(g), segments(p)
    out = np.zeros((len(iou_thrs), 3), int)
    for k, thr in enumerate(iou_thrs):
        hit = np.zeros(len(gs), bool)
        tp = fp = 0
        for s, e, c in ps:
            best, bj = 0.0, -1
            for j, (s2, e2, c2) in enumerate(gs):
                if c2 != c:
                    continue
                iou = max(0, min(e, e2) - max(s, s2)) / (max(e, e2) - min(s, s2))
                if iou > best:
                    best, bj = iou, j
            if best >= thr and not hit[bj]:
                tp += 1
                hit[bj] = True
            else:
                fp += 1
        out[k] = tp, fp, int((~hit).sum())
    return out


def segmental_f1(gt, pr, runs, iou_thr):
    """MS-TCN style F1@IoU over segments inside annotated stretches."""
    c = sum((_segment_counts(gt[a:b], np.where(gt[a:b] >= 0, pr[a:b], -1), (iou_thr,))[0] for a, b in runs),
            np.zeros(3, int))
    return float(_prf(c[0], c[0] + c[1], c[0] + c[2])[2])


# ---------------------------------------------------------------- evaluation from sufficient statistics
TOLS_S = (0.5, 1, 2, 3)  # boundary tolerances (s)
IOU_THRS = (0.1, 0.25, 0.5)  # segmental F1 thresholds


def run_stats(gt: np.ndarray, pr: np.ndarray, hz: float, runs=None) -> dict:
    """Sufficient statistics per eval run (annotated stretch, unlabeled gaps <= 1 s bridged), one row per run:
    cm (R, K, K) confusion matrix (gt x pred); bnd_n_gt / bnd_n_pred (R,); bnd_tp / bnd_typed_tp (R, len(TOLS_S));
    seg (R, len(IOU_THRS), 3) tp / fp / fn. Boundaries are matched inside a run only.
    metrics() of the column sums is evaluate(); bootstrap() re-weights the rows."""
    if runs is None:
        runs = eval_runs(gt, max_gap=int(hz))
    tols = [int(round(t * hz)) for t in TOLS_S]
    R = len(runs)
    st = dict(cm=np.zeros((R, K, K)), bnd_n_gt=np.zeros(R), bnd_n_pred=np.zeros(R),
              bnd_tp=np.zeros((R, len(tols))), bnd_typed_tp=np.zeros((R, len(tols))),
              seg=np.zeros((R, len(IOU_THRS), 3)))
    for r, (a, b) in enumerate(runs):
        g = gt[a:b]
        p = np.where(g >= 0, pr[a:b], -1)
        m = g >= 0
        st["cm"][r] = np.bincount(g[m].astype(int) * K + p[m].astype(int), minlength=K * K).reshape(K, K)
        gb, pb = typed_boundaries(g, [(0, b - a)]), typed_boundaries(p, [(0, b - a)])
        st["bnd_n_gt"][r], st["bnd_n_pred"][r] = len(gb[0]), len(pb[0])
        for k, tol in enumerate(tols):
            st["bnd_tp"][r, k] = match_boundaries(gb[0], pb[0], tol)
            st["bnd_typed_tp"][r, k] = match_typed_boundaries(gb, pb, tol)
        st["seg"][r] = _segment_counts(g, p, IOU_THRS)
    return st


def metrics(s: dict) -> dict:
    """Metrics from summed sufficient statistics (run_stats summed over runs). Leading axes of the arrays are kept
    (e.g. bootstrap replicates): every value is then an array over them."""
    cm = np.asarray(s["cm"], float)
    n = cm.sum((-2, -1))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1c = np.where(rows + cols > 0, 2 * tp / np.maximum(rows + cols, 1e-12), 0.0)
        acc = tp.sum(-1) / n
        pe = (rows * cols).sum(-1) / n ** 2
        kappa = (acc - pe) / (1 - pe)
    tr, est = list(TRANSITIONS), [c for c in range(K) if c not in TRANSITIONS]
    ttp = cm[..., tr, :][..., tr].sum((-2, -1))
    tfp = cm[..., est, :][..., tr].sum((-2, -1))
    tfn = cm[..., tr, :][..., est].sum((-2, -1))
    res = dict(n_frames=n, acc=acc, macro_f1=f1c.mean(-1), kappa=kappa, f1_per_class=f1c,
               transition_f1=np.where(2 * ttp + tfp + tfn > 0, 2 * ttp / np.maximum(2 * ttp + tfp + tfn, 1e-12), 0.0),
               n_gt_boundaries=np.asarray(s["bnd_n_gt"]), n_pred_boundaries=np.asarray(s["bnd_n_pred"]))
    for k, tol_s in enumerate(TOLS_S):
        res[f"bnd_f1@{tol_s}s"] = _prf(s["bnd_tp"][..., k], s["bnd_n_pred"], s["bnd_n_gt"])[2]
        res[f"bnd_typed_f1@{tol_s}s"] = _prf(s["bnd_typed_tp"][..., k], s["bnd_n_pred"], s["bnd_n_gt"])[2]
    for k, thr in enumerate(IOU_THRS):
        c = np.asarray(s["seg"])[..., k, :]
        res[f"seg_f1@{thr}"] = _prf(c[..., 0], c[..., 0] + c[..., 1], c[..., 0] + c[..., 2])[2]
    return res


SCALAR_METRICS = ("acc", "macro_f1", "kappa", "transition_f1", *(f"bnd_f1@{t}s" for t in TOLS_S),
                  *(f"bnd_typed_f1@{t}s" for t in TOLS_S), *(f"seg_f1@{t}" for t in IOU_THRS))


def _plain(res: dict) -> dict:
    out = {}
    for k, v in res.items():
        v = np.asarray(v)
        out[k] = [float(x) for x in v] if v.ndim else (int(v) if k.startswith("n_") else float(v))
    return out


def evaluate(gt: np.ndarray, pr: np.ndarray, hz: float, mask: np.ndarray | None = None) -> dict:
    """All metrics on frames where gt >= 0 (and mask): frame (acc, macro / per-class / transition F1, kappa),
    boundary F1 at TOLS_S (bnd_f1: position only; bnd_typed_f1: also the same class before and after) and
    segmental F1 at IOU_THRS. Boundaries and segments are matched inside each eval run."""
    gt = np.asarray(gt).copy()
    if mask is not None:
        gt[~mask] = -1
    st = run_stats(gt, np.asarray(pr), hz)
    return _plain(metrics({k: v.sum(0) for k, v in st.items()}))


def bootstrap(gt: np.ndarray, preds: dict, hz: float, mask: np.ndarray | None = None, n_boot: int = 2000,
              seed: int = 0, level: float = 0.95) -> dict:
    """Cluster bootstrap over eval runs (annotated stretches), paired across methods.

    Sufficient statistics are computed once per run and method (run_stats); each replicate draws multinomial
    run weights (R draws with replacement out of R runs) and sums the weighted statistics, the same weights
    for every method. Returns per method the percentile interval of each metric (`ci`), and per pair of
    methods "a - b" the point difference, its interval and P(diff <= 0) over replicates (`diff`).
    """
    gt = np.asarray(gt).copy()
    if mask is not None:
        gt[~mask] = -1
    runs = eval_runs(gt, max_gap=int(hz))
    R = len(runs)
    stats = {name: run_stats(gt, np.asarray(p), hz, runs) for name, p in preds.items()}
    W = np.random.default_rng(seed).multinomial(R, np.full(R, 1.0 / max(R, 1)), size=n_boot).astype(float)
    point = {name: metrics({k: v.sum(0) for k, v in st.items()}) for name, st in stats.items()}
    boot = {name: metrics({k: np.tensordot(W, v, axes=(1, 0)) for k, v in st.items()}) for name, st in stats.items()}
    q = 100 * np.array([(1 - level) / 2, (1 + level) / 2])

    def interval(x):
        x = np.asarray(x, float)
        with np.errstate(invalid="ignore"):
            return np.nanpercentile(x, q, axis=0).T.tolist() if np.isfinite(x).any() else [np.nan, np.nan]

    ci = {}
    for name, b in boot.items():
        ci[name] = {m: dict(point=float(point[name][m]), lo=interval(b[m])[0], hi=interval(b[m])[1])
                    for m in SCALAR_METRICS}
        ci[name]["f1_per_class"] = [dict(point=float(point[name]["f1_per_class"][c]), lo=lo, hi=hi)
                                    for c, (lo, hi) in enumerate(interval(b["f1_per_class"]))]
    diff = {}
    names = list(preds)
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            diff[f"{a} - {b_}"] = {}
            for m in SCALAR_METRICS:
                dd = np.asarray(boot[a][m], float) - np.asarray(boot[b_][m], float)
                ok = np.isfinite(dd)
                lo, hi = interval(dd)
                diff[f"{a} - {b_}"][m] = dict(diff=float(point[a][m] - point[b_][m]), lo=lo, hi=hi,
                                              p_le_0=float(np.mean(dd[ok] <= 0)) if ok.any() else float("nan"))
    return dict(unit="eval run (annotated stretch, unlabeled gaps <= 1 s bridged)", n_units=R, n_boot=n_boot,
                level=level, seed=seed, ci=ci, diff=diff)
