"""Load a fused tracking+annotation CSV into compact arrays (12.5 Hz).

Output npz keys
  pos    (T, 2, 11, 2)  player xy per team slot [0=home, 1=away], NaN if empty
  vel    (T, 2, 11, 2)  velocity m/s (0.2 s centred diff, smoothed)
  is_gk  (T, 2, 11)     bool
  ball   (T, 3)         ball xyz
  ball_v (T, 2)
  alive  (T,)           ball in play
  period (T,)           1/2, 0 = no players (break)
  attack_sign (3,)      per period: +1 if HOME attacks +x
  label  (T,)           4-moment label, home perspective, -1 = unlabeled
  sub    (T,)           subcategory id, -1 = unlabeled
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SRC_HZ = 25
STEP = 2  # 25 Hz -> 12.5 Hz
HZ = SRC_HZ / STEP

# 4 moments, home (reference team) perspective
CLASSES = ["est_attack", "def_to_att", "est_defense", "att_to_def"]
CATEGORY_MAP = {
    "Attacking": 0,
    "Defending Transition": 1,  # ball won -> counterattack / delayed attack
    "Defending": 2,
    "Attacking Transition": 3,  # ball lost -> counterpressing / delay & recovery
}
SUBCATS = [
    "Build up", "Progression", "Creation",
    "Counterattack", "Delayed Attack",
    "High Block", "Mid Block", "Low Block",
    "Counterpressing", "Delay and recovery",
]


def _first(val: str) -> str:
    """'Attacking; Attacking' -> 'Attacking' (overlapping annotation clips)."""
    return val.split(";")[0].strip()


def _player_cols(columns, team):
    xs = [c for c in columns if re.fullmatch(rf"{team}_{team}_\d+_X", c)]
    return xs, [c[:-1] + "Y" for c in xs]


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average along axis 0, NaN-aware."""
    flat = a.reshape(a.shape[0], -1)
    out = pd.DataFrame(flat).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return out.reshape(a.shape)


def load(csv_path: str | Path) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    T0 = len(df)

    team_xy, team_ids = [], []
    for team in ("home", "away"):
        cx, cy = _player_cols(df.columns, team)
        team_ids.append(cx)
        team_xy.append(np.stack([df[cx].to_numpy(float), df[cy].to_numpy(float)], -1))  # (T, P, 2)

    # periods: the longest run with no players at all is the half-time break
    n_on = sum((~np.isnan(xy[..., 0])).sum(1) for xy in team_xy)
    empty = n_on == 0
    runs = np.flatnonzero(np.diff(np.r_[0, empty.astype(int), 0]))
    starts, ends = runs[::2], runs[1::2]
    k = np.argmax(ends - starts)
    period = np.where(np.arange(T0) < starts[k], 1, 2)
    period[empty] = 0

    # direction of play per period from team centroids
    attack_sign = np.zeros(3)
    for p in (1, 2):
        m = period == p
        hx = np.nanmean(team_xy[0][m, :, 0])
        ax = np.nanmean(team_xy[1][m, :, 0])
        attack_sign[p] = 1.0 if hx < ax else -1.0

    # goalkeepers: player column with the most extreme mean x toward own goal
    gk_col = np.zeros((2, 3), int)
    for t in range(2):
        for p in (1, 2):
            m = period == p
            own = -attack_sign[p] if t == 0 else attack_sign[p]
            mean_x = np.nanmean(team_xy[t][m, :, 0], 0)
            mean_x = np.where(np.isnan(mean_x), 0, mean_x)
            gk_col[t, p] = np.argmax(mean_x * own)

    # velocity at source rate, then smooth 0.2 s
    vel_src = []
    for xy in team_xy:
        v = np.gradient(xy, axis=0) * SRC_HZ
        vel_src.append(_smooth(v, 5))

    # pack to 11 slots per team (on-pitch players, stable column order)
    idx = np.arange(0, T0, STEP)
    T = len(idx)
    pos = np.full((T, 2, 11, 2), np.nan)
    vel = np.full((T, 2, 11, 2), np.nan)
    is_gk = np.zeros((T, 2, 11), bool)
    for t in range(2):
        xy = team_xy[t][idx]
        v = vel_src[t][idx]
        present = ~np.isnan(xy[..., 0])
        order = np.argsort(~present, axis=1, kind="stable")[:, :11]  # present columns first
        rows = np.arange(T)[:, None]
        pos[:, t] = xy[rows, order]
        vel[:, t] = v[rows, order]
        ok = present[rows, order]
        pos[:, t][~ok] = np.nan
        vel[:, t][~ok] = np.nan
        per = period[idx]
        is_gk[:, t] = (order == gk_col[t, per][:, None]) & ok & (per[:, None] > 0)

    ball = df[["Ball_X", "Ball_Y", "Ball_Z"]].to_numpy(float)
    ball_v = _smooth(np.gradient(ball[:, :2], axis=0) * SRC_HZ, 5)

    cat = df["Annotation_Category"].fillna("").map(_first)
    sub = df["Annotation_Subcategory"].fillna("").map(_first)
    label = cat.map(CATEGORY_MAP).fillna(-1).astype(int).to_numpy()
    sub_id = sub.map({s: i for i, s in enumerate(SUBCATS)}).fillna(-1).astype(int).to_numpy()

    return dict(
        pos=pos.astype(np.float32),
        vel=np.nan_to_num(vel).astype(np.float32),
        is_gk=is_gk,
        ball=np.nan_to_num(ball[idx]).astype(np.float32),
        ball_v=np.nan_to_num(ball_v[idx]).astype(np.float32),
        alive=(df["ball_status"].to_numpy()[idx] == "alive"),
        period=period[idx].astype(np.int8),
        attack_sign=attack_sign.astype(np.float32),
        label=label[idx].astype(np.int8),
        sub=sub_id[idx].astype(np.int8),
        hz=np.float32(HZ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="artifacts/match.npz")
    a = ap.parse_args()
    d = load(a.csv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    lab = d["label"]
    print(f"T={len(lab)} @ {HZ} Hz | labeled {np.mean(lab >= 0):.1%} | "
          f"attack_sign={d['attack_sign'][1:]} | class counts "
          + str({CLASSES[c]: int((lab == c).sum()) for c in range(4)}))
