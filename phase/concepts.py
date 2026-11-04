"""Codebook v4.1 §2.5 transition-end criteria as tracking quantities, and the rule baseline (contribution 2).

A transition of the new possessor w ends once the new defending team is organised: C1 its deepest line is formed,
C2 no player is recovering, C3 its lines are connected, C4 there is no uncovered direct route to its goal; all four
for about 3 s, the boundary on the first frame of that window. Per frame, in the view of w (w attacks +x, the
defending team's goal is at x = +52.5; outfield players only):
  c1_goalside     defenders goal-side of the ball (x > ball x)                               C1, more = formed
  c1_back_spread  x range of the four deepest defenders (m)                                  C1, less = one line
  c2_recovering   defenders running toward their own goal faster than V_REC (m/s)            C2, 0 = nobody
  c3_length       x range of the defending team (m)                                          C3, less = connected
  c3_line_gap     mean gap between its three lines (1D k-means, as features.team_block)     C3
  c4_uncovered    attackers ahead of the ball with no defender goal-side within LANE_M of y C4, 0 = covered
  c4_cone         defenders inside the triangle ball - goal posts (+ CONE_MARGIN_M)          C4, 0 = route open
The rule thresholds one quantity per criterion (c1_goalside >= th1, c2_recovering <= th2, c3_length <= th3,
c4_uncovered <= th4) and ends the transition at the first frame from which all four hold for hold_s (or until
the window ends). Thresholds and hold_s: grid search on the training episodes, macro-F1 (EpisodeScorer).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import episodes as P
from .features import HALF_L, _fill_team, _line_structure
from .prep import home_attack_sign

V_REC = 2.5
LANE_M = 5.0
POST_Y = 3.66
CONE_MARGIN_M = 1.0
NAMES = ["c1_goalside", "c1_back_spread", "c2_recovering", "c3_length", "c3_line_gap", "c4_uncovered", "c4_cone"]
CRITERIA = ["C1", "C2", "C3", "C4"]
RULE_GRID = dict(th1=np.arange(3, 10), th2=np.array([0, 1, 2]), th3=np.arange(25.0, 52.5, 2.5),
                 th4=np.array([0, 1, 2]), hold_s=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def _view(d: dict, w: int):
    """Positions / velocities / goalkeeper flags / ball of team w (index 0) and the defending team (index 1),
    rotated so that w attacks +x (frames outside a period: direction of the nearest period)."""
    s = home_attack_sign(d, np.arange(len(d["pos"]))).astype(float) * (1 if w == 0 else -1)
    P_ = d["pos"][:, [w, 1 - w]] * s[:, None, None, None]
    V = d["vel"][:, [w, 1 - w]] * s[:, None, None, None]
    return P_, V, d["is_gk"][:, [w, 1 - w]].astype(bool), d["ball"][:, :2] * s[:, None]


def compute(d: dict) -> dict:
    """{w: (T, len(NAMES)) float32} concept quantities in the view of each team w (see module doc)."""
    out = {}
    with np.errstate(invalid="ignore"):
        for w in (0, 1):
            P_, V, G, B = _view(d, w)
            T = len(P_)
            valid = ~np.isnan(P_[..., 0])
            att, dfd = valid[:, 0] & ~G[:, 0], valid[:, 1] & ~G[:, 1]
            ax, ay, dx, dy = P_[:, 0, :, 0], P_[:, 0, :, 1], P_[:, 1, :, 0], P_[:, 1, :, 1]
            bx, by = B[:, 0], B[:, 1]

            goalside = (dfd & (dx > bx[:, None])).sum(1)
            xs = -np.sort(-np.where(dfd, dx, -np.inf), 1)[:, :4]  # deepest four, largest x first
            n4 = np.minimum(dfd.sum(1), 4)
            spread = np.where(n4 >= 2, xs[:, 0] - xs[np.arange(T), np.maximum(n4 - 1, 0)], 0.0)
            recovering = (dfd & (V[:, 1, :, 0] > V_REC)).sum(1)
            dmax, dmin = np.where(dfd, dx, -np.inf).max(1), np.where(dfd, dx, np.inf).min(1)
            length = np.where(dfd.sum(1) >= 2, dmax - dmin, 0.0)
            # three lines of the defending team in its own orientation (own goal at -x), as features.team_block
            own = -_fill_team(P_[:, 1])[..., 0]
            order = np.argsort(G[:, 1], axis=1, kind="stable")[:, :10]
            _, m3 = _line_structure(own[np.arange(T)[:, None], order], 3)
            line_gap = (m3[:, 2] - m3[:, 0]) / 2
            ahead = att & (ax > bx[:, None])
            cover = dfd[:, None, :] & (dx[:, None, :] > ax[:, :, None]) & (np.abs(dy[:, None, :] - ay[:, :, None]) <= LANE_M)
            uncovered = (ahead & ~cover.any(-1)).sum(1)
            frac = (dx - bx[:, None]) / np.maximum(HALF_L - bx[:, None], 1e-6)
            lo = by[:, None] + (-POST_Y - CONE_MARGIN_M - by[:, None]) * frac
            hi = by[:, None] + (POST_Y + CONE_MARGIN_M - by[:, None]) * frac
            cone = (dfd & (dx > bx[:, None]) & (dx <= HALF_L) & (dy >= lo) & (dy <= hi)).sum(1)
            out[w] = np.nan_to_num(np.stack([goalside, spread, recovering, length, line_gap, uncovered, cone], 1)
                                   ).astype(np.float32)
    return out


class Rule:
    """Rule baseline on the concatenated window frames of a set of episodes (vectorised over episodes)."""

    def __init__(self, conc: dict, eps: pd.DataFrame, hz: float):
        self.hz = hz
        self.W = P.lengths(eps)
        self.offs = np.r_[0, np.cumsum(self.W)]
        N = int(self.offs[-1])
        self.ep_id = np.repeat(np.arange(len(eps)), self.W)
        self.k = np.arange(N) - self.offs[self.ep_id] if N else np.zeros(0, int)
        self.end = self.offs[self.ep_id + 1] if N else np.zeros(0, int)
        frames = np.concatenate([np.arange(s, t) for s, t in zip(eps.s0, eps.t1)] or [np.zeros(0, int)]).astype(int)
        team = np.repeat(eps.team.to_numpy(int), self.W)
        X = np.zeros((N, len(NAMES)), np.float32)
        for w in (0, 1):
            m = team == w
            X[m] = conc[w][frames[m]]
        self.X = X

    def criteria(self, th1, th2, th3, th4) -> np.ndarray:
        """(N, 4) bool: C1..C4 hold on each window frame."""
        X = self.X
        return np.stack([X[:, 0] >= th1, X[:, 2] <= th2, X[:, 3] <= th3, X[:, 5] <= th4], 1)

    def _run(self, ok: np.ndarray) -> np.ndarray:
        """Consecutive frames from each frame on (inside its episode) on which ok holds."""
        N = len(ok)
        idx = np.arange(N)
        bad = np.flatnonzero(~ok)
        if len(bad):
            j = np.searchsorted(bad, idx)  # first frame >= i where the criteria fail
            nxt = np.where(j < len(bad), bad[np.minimum(j, len(bad) - 1)], N)
        else:
            nxt = np.full(N, N)
        return np.minimum(nxt, self.end) - idx

    def _first(self, ok: np.ndarray, hold: int, run: np.ndarray | None = None) -> np.ndarray:
        """Per episode the first k from which ok holds for `hold` frames (or to the window end); W if never."""
        E = self.W.copy()
        if len(ok) == 0:
            return E
        run = self._run(ok) if run is None else run
        cond = ok & (run >= np.minimum(hold, self.end - np.arange(len(ok))))
        np.minimum.at(E, self.ep_id[cond], self.k[cond])
        return E

    def lengths(self, th1, th2, th3, th4, hold_s) -> np.ndarray:
        return self._first(self.criteria(th1, th2, th3, th4).all(1), int(round(hold_s * self.hz)))

    def fit(self, scorer: P.EpisodeScorer, grid: dict = RULE_GRID) -> tuple[dict, float]:
        """Grid search of (th1..th4, hold_s) maximising scorer.macro_f1 (the scorer is built on the same episodes,
        usually with the training labels only)."""
        X = self.X
        c1 = {t: X[:, 0] >= t for t in grid["th1"]}
        c2 = {t: X[:, 2] <= t for t in grid["th2"]}
        c3 = {t: X[:, 3] <= t for t in grid["th3"]}
        c4 = {t: X[:, 5] <= t for t in grid["th4"]}
        best = (-1.0, None)
        for t1, t2, t3, t4 in itertools.product(grid["th1"], grid["th2"], grid["th3"], grid["th4"]):
            ok = c1[t1] & c2[t2] & c3[t3] & c4[t4]
            run = self._run(ok) if len(ok) else None
            for h in grid["hold_s"]:
                f1 = scorer.macro_f1(self._first(ok, int(round(h * self.hz)), run))
                if f1 > best[0]:
                    best = (f1, dict(th1=int(t1), th2=int(t2), th3=float(t3), th4=int(t4), hold_s=float(h)))
        return best[1], best[0]

    def explain(self, params: dict, E_frames) -> list[str]:
        """Per episode the criterion that was satisfied last at the decoded end ('C1'..'C4'), 'at_start' if all
        held from the first frame, 'never' if the transition fills the window."""
        crit = self.criteria(params["th1"], params["th2"], params["th3"], params["th4"])
        out = []
        for i, e in enumerate(np.asarray(E_frames, int)):
            if e >= self.W[i]:
                out.append("never")
                continue
            c = crit[self.offs[i]:self.offs[i] + e + 1]  # frames 0..e of the window
            start = [int(np.flatnonzero(~c[:, j])[-1]) + 1 if (~c[:, j]).any() else 0 for j in range(4)]
            out.append("at_start" if max(start) == 0 else CRITERIA[int(np.argmax(start))])
        return out
