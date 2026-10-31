"""Per-view features: node features (L1), organization block (L3), possession heuristic.

A *view* fixes the reference team (0 = home, 1 = away). Coordinates are rotated so the
reference team always attacks +x; labels are mirrored for the away view
(est_attack <-> est_defense, def_to_att <-> att_to_def).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

HALF_L, HALF_W = 52.5, 34.0
MIRROR = np.array([2, 3, 0, 1])  # class permutation between the two views

NODE_FEATS = [
    "x", "y", "vx", "vy", "speed", "vx_2s", "vy_2s", "d_ball", "d_own_goal", "d_opp_goal",
    "behind_ball", "d_centroid", "is_ref", "is_opp", "is_gk", "is_ball", "team_poss", "ball_z",
]


# ---------------------------------------------------------------- helpers
def _roll(a: np.ndarray, win: int) -> np.ndarray:
    flat = a.reshape(a.shape[0], -1)
    out = pd.DataFrame(flat).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return out.reshape(a.shape)


def _fill_team(xy: np.ndarray) -> np.ndarray:
    """Impute missing players with the team centroid (rare; <1% of frames)."""
    c = np.nanmean(xy, axis=1, keepdims=True)
    c = np.nan_to_num(c)
    return np.where(np.isnan(xy), c, xy)


def _line_structure(x: np.ndarray, k: int):
    """Optimal 1D k-means on sorted outfield depth + silhouette score.

    x: (T, n) outfield x in own-attack orientation (own goal at -x).
    Returns silhouette (T,), cluster means sorted deep->high (T, k).
    """
    T, n = x.shape
    xs = np.sort(x, axis=1)
    cs = np.concatenate([np.zeros((T, 1)), np.cumsum(xs, 1)], 1)
    cs2 = np.concatenate([np.zeros((T, 1)), np.cumsum(xs ** 2, 1)], 1)

    def sse(a, b):  # segment [a, b)
        s, s2, m = cs[:, b] - cs[:, a], cs2[:, b] - cs2[:, a], b - a
        return s2 - s ** 2 / m

    import itertools
    splits = list(itertools.combinations(range(1, n), k - 1))
    costs = np.stack([sum(sse(a, b) for a, b in zip((0,) + s, s + (n,))) for s in splits], 1)
    best = np.array(splits)[np.argmin(costs, 1)]  # (T, k-1)

    lab = (np.arange(n)[None, :, None] >= best[:, None, :]).sum(-1)  # (T, n) cluster id
    onehot = np.eye(k)[lab]  # (T, n, k)
    cnt = onehot.sum(1)  # (T, k)
    D = np.abs(xs[:, :, None] - xs[:, None, :])
    sum_d = D @ onehot  # (T, n, k)
    own_cnt = (onehot * cnt[:, None, :]).sum(-1)
    a = (sum_d * onehot).sum(-1) / np.maximum(own_cnt - 1, 1)
    mean_other = np.where(onehot > 0, np.inf, sum_d / np.maximum(cnt[:, None, :], 1))
    b = mean_other.min(-1)
    s = np.where(own_cnt > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-6), 0.0)
    means = (xs[:, :, None] * onehot).sum(1) / np.maximum(cnt, 1)
    return s.mean(1), means


def _hull_area(xy: np.ndarray) -> np.ndarray:
    out = np.zeros(len(xy))
    for t in range(len(xy)):
        try:
            out[t] = ConvexHull(xy[t]).volume
        except Exception:
            out[t] = 0.0
    return out


# ---------------------------------------------------------------- possession
def possession(d: dict, ctrl_dist: float = 1.5, min_run_s: float = 1.0) -> np.ndarray:
    """Heuristic possession from tracking: 0 = home, 1 = away (NaN before first touch).

    Nearest player within `ctrl_dist` m of a low ball controls it; possession is carried
    forward and runs shorter than `min_run_s` (deflections, contested touches) are merged.
    """
    hz = float(d["hz"])
    pos, ball = d["pos"], d["ball"]
    dist = np.linalg.norm(pos - ball[:, None, None, :2], axis=-1)  # (T, 2, 11)
    dmin = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=2)  # (T, 2)
    team = np.argmin(dmin, 1).astype(float)
    ctrl = (dmin.min(1) < ctrl_dist) & (ball[:, 2] < 1.5) & d["alive"]
    s = pd.Series(np.where(ctrl, team, np.nan)).ffill()
    p = s.to_numpy().copy()
    # merge short runs into the preceding run
    min_run = int(min_run_s * hz)
    for _ in range(3):
        v = np.nan_to_num(p, nan=-1)
        chg = np.flatnonzero(np.diff(np.r_[-9, v, -9]))
        for a, b in zip(chg[:-1], chg[1:]):
            if b - a < min_run and a > 0 and v[a] >= 0:
                p[a:b] = p[a - 1]
    return p


# ---------------------------------------------------------------- team block
def team_block(xy: np.ndarray, v: np.ndarray, gk: np.ndarray, sign: np.ndarray) -> dict:
    """Organization features of one team, in its own attacking orientation (sign)."""
    xy = _fill_team(xy) * sign[:, None, None]
    v = v * sign[:, None, None]
    T = len(xy)
    order = np.argsort(gk, axis=1, kind="stable")[:, :10]  # 10 outfield first
    rows = np.arange(T)[:, None]
    out_xy, out_v = xy[rows, order], v[rows, order]

    c = out_xy.mean(1)
    r = out_xy - c[:, None]
    sp = np.linalg.norm(out_v, axis=-1)
    pol = np.linalg.norm(out_v.sum(1), axis=-1) / np.maximum(sp.sum(1), 1e-6)
    cross = r[..., 0] * out_v[..., 1] - r[..., 1] * out_v[..., 0]
    rot = np.abs((cross / np.maximum(np.linalg.norm(r, axis=-1) * sp, 1e-6)).mean(1))

    sil3, m3 = _line_structure(out_xy[..., 0], 3)
    sil4, _ = _line_structure(out_xy[..., 0], 4)
    sil2, _ = _line_structure(out_xy[..., 0], 2)
    return dict(
        cx=c[:, 0], cy=c[:, 1],
        length=np.ptp(out_xy[..., 0], 1), width=np.ptp(out_xy[..., 1], 1),
        hull=_hull_area(out_xy), stretch=np.linalg.norm(r, axis=-1).mean(1),
        polarization=pol, rotation=rot, speed_mean=sp.mean(1), speed_std=sp.std(1),
        vx_mean=out_v[..., 0].mean(1),
        sil2=sil2, sil3=sil3, sil4=sil4,
        line_def=m3[:, 0], line_gap=(m3[:, 2] - m3[:, 0]) / 2,
    )


# ---------------------------------------------------------------- main entry
def mark_pairs(pos: np.ndarray, is_gk: np.ndarray) -> np.ndarray:
    """Per frame 1:1 assignment of home to away outfield players (Hungarian on distance).
    Returns (T, 11): away slot marked by each home slot, -1 for goalkeeper / missing."""
    from scipy.optimize import linear_sum_assignment

    T = len(pos)
    out = np.full((T, 11), -1, np.int64)
    ok = ~np.isnan(pos[..., 0]) & ~is_gk
    for t in range(T):
        h, a = np.flatnonzero(ok[t, 0]), np.flatnonzero(ok[t, 1])
        if len(h) and len(a):
            D = np.linalg.norm(pos[t, 0, h][:, None] - pos[t, 1, a][None], axis=-1)
            r, c = linear_sum_assignment(D)
            out[t, h[r]] = a[c]
    return out


def build(d: dict, poss: np.ndarray, mark: bool = False) -> dict:
    """Features for both views. Returns dict view -> {node, node_pos, node_mask, glob, ...}."""
    hz = float(d["hz"])
    T = len(d["label"])
    pairs = mark_pairs(d["pos"], d["is_gk"]) if mark else None
    per = d["period"].astype(int)
    home_sign = d["attack_sign"][per]  # +1 if home attacks +x
    home_sign = np.where(per == 0, 1.0, home_sign).astype(np.float32)
    team_sign = np.stack([home_sign, -home_sign], 1)  # (T, 2)

    blocks = [team_block(d["pos"][:, t], d["vel"][:, t], d["is_gk"][:, t], team_sign[:, t])
              for t in range(2)]
    vel2s = _roll(d["vel"], int(2 * hz))
    alive = d["alive"].astype(np.float32)

    views = {}
    for ref in (0, 1):
        opp = 1 - ref
        s = team_sign[:, ref]  # rotate so ref attacks +x
        P = d["pos"][:, [ref, opp]] * s[:, None, None, None]
        V = d["vel"][:, [ref, opp]] * s[:, None, None, None]
        V2 = vel2s[:, [ref, opp]] * s[:, None, None, None]
        G = d["is_gk"][:, [ref, opp]]
        B = d["ball"][:, :2] * s[:, None]
        BV = d["ball_v"] * s[:, None]
        mask = ~np.isnan(P[..., 0])  # (T, 2, 11)
        Pz = np.nan_to_num(P)

        poss_ref = np.where(np.isnan(poss), 0.5, (poss == ref).astype(float))
        d_ball = np.linalg.norm(Pz - B[:, None, None], axis=-1)
        goal = np.array([[-HALF_L, 0.0], [HALF_L, 0.0]])  # own goal of ref, of opp
        d_own = np.linalg.norm(Pz - goal[None, :, None], axis=-1)
        d_opp = np.linalg.norm(Pz - goal[::-1][None, :, None], axis=-1)
        behind = np.stack([Pz[:, 0, :, 0] < B[:, None, 0], Pz[:, 1, :, 0] > B[:, None, 0]], 1)
        cen = np.nanmean(np.where(mask[..., None], P, np.nan), axis=2, keepdims=True)
        d_cen = np.linalg.norm(Pz - np.nan_to_num(cen), axis=-1)
        sp = np.linalg.norm(V, axis=-1)
        tp = np.stack([poss_ref, 1 - poss_ref], 1)[:, :, None].repeat(11, 2)

        pl = np.stack([
            Pz[..., 0] / HALF_L, Pz[..., 1] / HALF_W, V[..., 0] / 8, V[..., 1] / 8, sp / 8,
            V2[..., 0] / 8, V2[..., 1] / 8, d_ball / 50, d_own / 100, d_opp / 100,
            behind.astype(float), d_cen / 30,
            np.broadcast_to(np.array([1.0, 0.0])[None, :, None], mask.shape),
            np.broadcast_to(np.array([0.0, 1.0])[None, :, None], mask.shape),
            G.astype(float), np.zeros(mask.shape), tp, np.zeros(mask.shape),
        ], -1).reshape(T, 22, -1)
        bs = np.linalg.norm(BV, axis=-1)
        ball_node = np.zeros((T, 1, len(NODE_FEATS)))
        ball_node[:, 0, [0, 1, 2, 3, 4]] = np.stack(
            [B[:, 0] / HALF_L, B[:, 1] / HALF_W, BV[:, 0] / 20, BV[:, 1] / 20, bs / 20], 1)
        ball_node[:, 0, 15] = 1.0
        ball_node[:, 0, 17] = d["ball"][:, 2] / 3
        node = np.concatenate([pl * mask.reshape(T, 22, 1), ball_node], 1)
        node_pos = np.concatenate([Pz.reshape(T, 22, 2), B[:, None]], 1)
        node_mask = np.concatenate([mask.reshape(T, 22), np.ones((T, 1), bool)], 1)

        # ---- global organization block (L3), multi-scale (L4)
        br, bo = blocks[ref], blocks[opp]
        g = {f"ref_{k}": v for k, v in br.items()}
        g.update({f"opp_{k}": v for k, v in bo.items()})
        # inter-team relations (ref orientation: opp centroid x = -opp own-orientation cx)
        g["cen_dx"] = br["cx"] + bo["cx"]
        g["cen_dy"] = br["cy"] + bo["cy"]
        dr = np.where(mask[:, 0], d_ball[:, 0], np.inf)
        do = np.where(mask[:, 1], d_ball[:, 1], np.inf)
        for rad in (5, 10, 20):
            g[f"ball_ndiff_{rad}"] = (dr < rad).sum(1) - (do < rad).sum(1)
        g["ball_dmin_ref"] = np.minimum(dr.min(1), 60)
        g["ball_dmin_opp"] = np.minimum(do.min(1), 60)
        # pressure: mean nearest-opponent distance for each team
        pr, po = np.nan_to_num(P[:, 0], nan=1e3), np.nan_to_num(P[:, 1], nan=1e3)
        D = np.linalg.norm(pr[:, :, None] - po[:, None], axis=-1)
        g["press_on_ref"] = np.minimum(D.min(2), 60).mean(1)
        g["press_on_opp"] = np.minimum(D.min(1), 60).mean(1)
        glob = np.stack(list(g.values()), 1).astype(np.float32)
        glob_names = list(g.keys())
        glob = np.concatenate([glob, _roll(glob, int(2 * hz)), _roll(glob, int(5 * hz))], 1)
        glob_names = glob_names + [n + "@2s" for n in glob_names] + [n + "@5s" for n in glob_names]

        # ---- context (ball / possession / timer), kept separate for ablation
        chg = np.r_[False, np.diff(np.nan_to_num(poss, nan=-1)) != 0]
        since = np.zeros(T)
        for t in range(1, T):
            since[t] = 0 if chg[t] else since[t - 1] + 1 / hz
        ctx = np.stack([
            B[:, 0] / HALF_L, B[:, 1] / HALF_W, d["ball"][:, 2] / 3, bs / 20, alive,
            poss_ref, np.minimum(since, 30) / 30,
        ], 1).astype(np.float32)

        lab = d["label"].astype(int)
        if ref == 1:
            lab = np.where(lab >= 0, MIRROR[np.clip(lab, 0, 3)], -1)
        views[ref] = dict(
            node=node.astype(np.float32), node_pos=node_pos.astype(np.float32),
            node_mask=node_mask, glob=np.nan_to_num(glob), glob_names=glob_names,
            ctx=ctx, ctx_names=["ball_x", "ball_y", "ball_z", "ball_speed", "alive",
                                "poss_ref", "time_since_turnover"],
            poss_ref=poss_ref.astype(np.float32), label=lab,
        )
        if pairs is not None:  # node index of the marking partner (nodes: 11 ref, 11 opp, ball)
            m = np.full((T, 23), -1, np.int64)
            ti, hs = np.nonzero(pairs >= 0)
            aw = pairs[ti, hs]
            h_node, a_node = (hs, 11 + aw) if ref == 0 else (11 + hs, aw)
            m[ti, h_node] = a_node
            m[ti, a_node] = h_node
            views[ref]["mark"] = m
    return views
