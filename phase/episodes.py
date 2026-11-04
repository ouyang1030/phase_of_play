"""Four-moment labels = possession sequence + one transition length per possession (codebook v4.1 §2.4, §2.5).

Inside a possession of team w the reference-team (home perspective) labels run [transition_w]* [established_w]*:
a transition starts only at a gain of possession and ends once, or not at all before the possession ends.
Given the possession, a label sequence is therefore fixed by one number per possession, the transition length
E >= 0 (E = 0: no transition labelled).

Episode = one turnover, columns
  t0    turnover frame (tracking index at d['hz'])
  s0    first frame with the ball in play at or after t0: the restart after a dead-ball turnover, else t0
  t1    end of the window [s0, t1): the next turnover, the next dead ball, a period change or s0 + max_s
  team  new possessor w (0 home / 1 away); dead: the possession changed across a dead ball; how: win group
E counts frames from s0 on the support {0, ..., W - 1, W}, W = t1 - s0; E = W means the transition fills the
whole window (it is cut by the dead ball / next turnover, i.e. right-censored).

A label sequence says E is in [lo, hi] (`observe`): an established_w frame at k gives E <= k, a transition_w
frame at k gives E >= k + 1; unlabelled frames, frames outside the training region and the other team's classes
(annotator switching possession later than the event, §2.4) say nothing. lo == hi is an exact observation,
(0, W) none at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E

K = 4
TRANS = np.array([1, 3])  # transition class of possessor w in home-perspective labels: home def_to_att, away att_to_def
EST = np.array([0, 2])  # established class of possessor w: est_attack, est_defense
COLUMNS = ["t0", "s0", "t1", "team", "dead", "how"]


def label_possession(label: np.ndarray) -> np.ndarray:
    """Possession implied by home-perspective labels: 0 (est_attack, def_to_att), 1 (est_defense, att_to_def),
    NaN on unlabelled frames. Oracle input for the error decomposition only (it is the test labels)."""
    lab = np.asarray(label)
    return np.where(lab < 0, np.nan, np.where(np.isin(lab, (0, 1)), 0.0, 1.0))


def _dead_between(alive: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Any dead-ball frame in [lo, hi) (vectorised over pairs)."""
    c = np.r_[0, np.cumsum(~alive)]
    return c[np.clip(hi, 0, len(alive))] - c[np.clip(lo, 0, len(alive))] > 0


def episodes(poss_home: np.ndarray, alive: np.ndarray, period: np.ndarray, hz: float, max_s: float = 30.0,
             turnovers: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per turnover (see module doc).

    turnovers: event turnover table (phase.events.possession: idx, team, dead, how); otherwise the turnovers are
    the changes of `poss_home` with gaps filled by the previous possessor (E.fill_possession), dead = a dead-ball
    frame lies between the last frame of known possession before the change and the change. A window whose
    restart falls in another period than its turnover (or after the next turnover) is empty (t1 = s0)."""
    alive = np.asarray(alive, bool)
    period = np.asarray(period).astype(int)
    p = E.fill_possession(poss_home)
    T = len(p)
    if turnovers is not None:
        if not len(turnovers):
            return pd.DataFrame({c: np.array([], int) for c in COLUMNS})
        t0 = turnovers["idx"].to_numpy(int)
        team = turnovers["team"].to_numpy(int)
        dead = turnovers["dead"].to_numpy(bool)
        how = turnovers["how"].astype(str).to_numpy() if "how" in turnovers else np.full(len(t0), "other")
    else:
        t0 = np.flatnonzero(np.diff(p) != 0) + 1
        team = p[t0].astype(int)
        known = np.flatnonzero(~np.isnan(np.asarray(poss_home, float)))
        j = np.searchsorted(known, t0) - 1
        prev = np.where(j >= 0, known[np.clip(j, 0, max(len(known) - 1, 0))] if len(known) else 0, 0)
        dead = _dead_between(alive, prev + 1, t0) | ~alive[np.clip(t0 - 1, 0, T - 1)]
        how = np.full(len(t0), "other")
    live = np.flatnonzero(alive)
    j = np.searchsorted(live, t0)
    s0 = np.where(j < len(live), live[np.clip(j, 0, max(len(live) - 1, 0))] if len(live) else T, T)
    nxt = np.r_[t0[1:], T]
    t1 = np.minimum(nxt, s0 + int(round(max_s * hz)))
    dead_frames = np.flatnonzero(~alive)
    k = np.searchsorted(dead_frames, s0)
    t1 = np.minimum(t1, np.where(k < len(dead_frames), dead_frames[np.clip(k, 0, max(len(dead_frames) - 1, 0))]
                                 if len(dead_frames) else T, T))
    for i in range(len(t0)):
        if s0[i] >= T or period[s0[i]] != period[min(t0[i], T - 1)] or s0[i] >= nxt[i]:
            t1[i] = s0[i] = min(s0[i], T)
            continue
        pc = np.flatnonzero(period[s0[i]:t1[i]] != period[s0[i]])
        if len(pc):
            t1[i] = s0[i] + pc[0]
    t1 = np.maximum(t1, s0)
    return pd.DataFrame(dict(t0=t0, s0=s0, t1=t1, team=team, dead=dead, how=how))


def lengths(eps: pd.DataFrame) -> np.ndarray:
    return (eps.t1 - eps.s0).to_numpy(int)


def observe(label: np.ndarray, eps: pd.DataFrame, allowed: np.ndarray | None = None) -> pd.DataFrame:
    """What a label sequence says about E per episode: columns lo, hi (E in [lo, hi], hi == W includes the
    overflow), kind ('exact' lo == hi, 'interval', 'none' = (0, W)), lag (leading frames of the other team's
    classes). Frames with allowed False count as unlabelled (training-region censoring)."""
    label = np.asarray(label)
    rows = []
    for s0, t1, w in zip(eps.s0.to_numpy(), eps.t1.to_numpy(), eps.team.to_numpy()):
        W = t1 - s0
        y = label[s0:t1].astype(int)
        if allowed is not None:
            y = np.where(allowed[s0:t1], y, -1)
        est = np.flatnonzero(y == EST[w])
        hi = int(est[0]) if len(est) else W
        tr = np.flatnonzero(y[:hi] == TRANS[w])
        lo = int(tr[-1]) + 1 if len(tr) else 0
        other = np.flatnonzero((y >= 0) & (y != EST[w]) & (y != TRANS[w]))
        own = np.flatnonzero((y == EST[w]) | (y == TRANS[w]))
        lag = int(own[0]) if len(own) and len(other) and other[0] < own[0] else 0
        kind = "none" if (lo, hi) == (0, W) else ("exact" if lo == hi else "interval")
        rows.append((lo, hi, W, kind, lag))
    return pd.DataFrame(rows, columns=["lo", "hi", "W", "kind", "lag"])


def reconstruct(poss_home: np.ndarray, eps: pd.DataFrame, E_frames) -> np.ndarray:
    """Labels from possession + one transition length per episode: established of the possessor everywhere,
    transition of w on [s0, s0 + E) inside each window. E.timer_baseline is the special case E = tau * hz."""
    pred = E.established(poss_home)
    for s0, t1, w, e in zip(eps.s0.to_numpy(), eps.t1.to_numpy(), eps.team.to_numpy(), np.asarray(E_frames, int)):
        pred[s0:min(s0 + max(e, 0), t1)] = TRANS[w]
    return pred


class EpisodeScorer:
    """Exact frame metrics of reconstruct(poss, eps, E) against gt for any vector E in O(#episodes).

    Outside the windows the prediction is the established class of the possessor whatever E is; inside a window
    E moves the frames [s0, s0 + E) from established_w to transition_w. The confusion matrix is the one of the
    all-established prediction plus, per episode, the class counts of gt on [s0, s0 + E) moved between the two
    columns (cumulative counts per window)."""

    def __init__(self, gt: np.ndarray, poss_home: np.ndarray, eps: pd.DataFrame):
        gt = np.asarray(gt).astype(int)
        base = E.established(poss_home)
        m = gt >= 0
        self.C0 = np.bincount(gt[m] * K + base[m], minlength=K * K).reshape(K, K).astype(float)
        self.W = lengths(eps)
        Wmax = int(self.W.max()) if len(self.W) else 0
        self.cum = np.zeros((len(eps), Wmax + 1, K))
        for i, (s0, t1) in enumerate(zip(eps.s0.to_numpy(), eps.t1.to_numpy())):
            y = gt[s0:t1]
            oh = np.zeros((len(y), K))
            ok = y >= 0
            oh[np.flatnonzero(ok), y[ok]] = 1
            c = np.r_[np.zeros((1, K)), np.cumsum(oh, 0)]
            self.cum[i, :len(c)] = c
            self.cum[i, len(c):] = c[-1]
        team = eps.team.to_numpy(int)
        self.est, self.trans = EST[team], TRANS[team]

    def confusion(self, E_frames) -> np.ndarray:
        e = np.clip(np.asarray(E_frames, int), 0, self.W)
        n = self.cum[np.arange(len(e)), e]  # (n_ep, K) gt class counts on [s0, s0 + E)
        C = self.C0.copy()
        for col in range(K):
            C[:, col] += n[self.trans == col].sum(0) - n[self.est == col].sum(0)
        return C

    def macro_f1(self, E_frames) -> float:
        C = self.confusion(E_frames)
        tp = np.diag(C)
        den = 2 * tp + (C.sum(0) - tp) + (C.sum(1) - tp)
        return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))

    def accuracy_curves(self) -> list[np.ndarray]:
        """Per episode the number of window frames reconstruct gets right for each E = 0..W."""
        out = []
        for i, W in enumerate(self.W):
            c = self.cum[i, :W + 1]
            out.append(c[:, self.trans[i]] + (c[W, self.est[i]] - c[:, self.est[i]]))
        return out


def oracle_lengths(gt: np.ndarray, poss_home: np.ndarray, eps: pd.DataFrame) -> np.ndarray:
    """Best single change point per episode: the E that maximises the frames of the window labelled correctly
    (smallest E on ties). With the labels' own possession this reproduces every grammar-conforming label."""
    return np.array([int(np.argmax(a)) for a in EpisodeScorer(gt, poss_home, eps).accuracy_curves()], int)


def tune_constant(scorer: EpisodeScorer, hz: float, taus=np.arange(0, 20.5, 0.5)) -> tuple[float, float]:
    """Timer in the episode algebra: the same E = tau * hz for every episode, tau maximising macro-F1."""
    best = max(((scorer.macro_f1(np.full(len(scorer.W), int(round(t * hz)))), float(t)) for t in taus))
    return best[1], best[0]


def cv_folds(label: np.ndarray, folds: int, buffer_frames: int):
    """Blocked time CV of phase.train / phase.hazard: [L0, L1) = up to the last labelled frame, K contiguous
    blocks; fold f trains on everything but its block +- the buffer. Returns L0, L1, [(a, b, train_mask)]."""
    label = np.asarray(label)
    lab = np.flatnonzero(label >= 0)
    L0, L1 = 0, int(lab[-1]) + 1
    edges = np.linspace(L0, L1, folds + 1).astype(int)
    out = []
    for f in range(folds):
        a, b = int(edges[f]), int(edges[f + 1])
        m = np.zeros(len(label), bool)
        m[L0:L1] = True
        m[max(L0, a - buffer_frames):min(L1, b + buffer_frames)] = False
        out.append((a, b, m))
    return L0, L1, out
