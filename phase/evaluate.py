"""Decoding (mode filter, HSMM-lite), timer baseline, and evaluation metrics (STUDY1_PLAN §8)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

K = 4
TRANSITIONS = (1, 3)


# ---------------------------------------------------------------- baselines / decoding
def timer_baseline(poss_home: np.ndarray, tau_s: float, hz: float) -> np.ndarray:
    """L_fix(tau): turnover + fixed window -> transition, otherwise established of possessor."""
    p = np.nan_to_num(poss_home, nan=0.0)  # 0 = home in possession
    pred = np.where(p == 0, 0, 2)
    chg = np.flatnonzero(np.diff(p) != 0) + 1
    n = int(round(tau_s * hz))
    for t in chg:
        pred[t:t + n] = 1 if p[t] == 0 else 3
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


def boundaries(y: np.ndarray, runs) -> np.ndarray:
    out = []
    for a, b in runs:
        seg = y[a:b]
        idx = np.flatnonzero(seg >= 0)
        v = seg[idx]
        ch = np.flatnonzero(v[1:] != v[:-1]) + 1
        out += list(a + idx[ch])
    return np.array(out, int)


def boundary_f1(gt_b, pr_b, tol: int):
    if len(gt_b) == 0 and len(pr_b) == 0:
        return 1.0, 1.0, 1.0
    used = np.zeros(len(gt_b), bool)
    tp = 0
    for p in pr_b:
        if len(gt_b) == 0:
            break
        d = np.abs(gt_b - p).astype(float)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    prec = tp / max(len(pr_b), 1)
    rec = tp / max(len(gt_b), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def segmental_f1(gt, pr, runs, iou_thr):
    """MS-TCN style F1@IoU over segments inside annotated stretches."""
    tp = fp = fn = 0
    for a, b in runs:
        g = gt[a:b].copy()
        p = np.where(g >= 0, pr[a:b], -1)
        gs, ps = segments(g), segments(p)
        hit = np.zeros(len(gs), bool)
        for s, e, c in ps:
            best, bj = 0.0, -1
            for j, (s2, e2, c2) in enumerate(gs):
                if c2 != c:
                    continue
                inter = max(0, min(e, e2) - max(s, s2))
                union = max(e, e2) - min(s, s2)
                iou = inter / union
                if iou > best:
                    best, bj = iou, j
            if best >= iou_thr and not hit[bj]:
                tp += 1
                hit[bj] = True
            else:
                fp += 1
        fn += int((~hit).sum())
    p_ = tp / max(tp + fp, 1)
    r_ = tp / max(tp + fn, 1)
    return 2 * p_ * r_ / max(p_ + r_, 1e-9)


def evaluate(gt: np.ndarray, pr: np.ndarray, hz: float, mask: np.ndarray | None = None) -> dict:
    """All metrics on frames where gt >= 0 (and mask)."""
    gt = gt.copy()
    if mask is not None:
        gt[~mask] = -1
    m = gt >= 0
    y, p = gt[m], pr[m]
    per = f1_score(y, p, labels=range(K), average=None, zero_division=0)
    runs = eval_runs(gt, max_gap=int(hz))
    gb = boundaries(gt, runs)
    pr_masked = np.where(gt >= 0, pr, -1)
    pb = boundaries(pr_masked, runs)
    res = dict(
        n_frames=int(m.sum()),
        acc=float((y == p).mean()),
        macro_f1=float(per.mean()),
        kappa=float(cohen_kappa_score(y, p)),
        f1_per_class=[float(x) for x in per],
        transition_f1=float(f1_score(np.isin(y, TRANSITIONS), np.isin(p, TRANSITIONS), zero_division=0)),
        n_gt_boundaries=int(len(gb)),
        n_pred_boundaries=int(len(pb)),
    )
    for tol_s in (0.5, 1, 2, 3):
        res[f"bnd_f1@{tol_s}s"] = float(boundary_f1(gb, pb, int(round(tol_s * hz)))[2])
    for thr in (0.1, 0.25, 0.5):
        res[f"seg_f1@{thr}"] = float(segmental_f1(gt, pr, runs, thr))
    return res
