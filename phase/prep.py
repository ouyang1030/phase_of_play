"""Load a fused tracking+annotation CSV into compact arrays (12.5 Hz).

Output npz keys
  pos    (T, 2, 11, 2)  player xy per team slot [0=home, 1=away], NaN if empty
  vel    (T, 2, 11, 2)  velocity m/s (0.2 s centred diff, smoothed)
  is_gk  (T, 2, 11)     bool
  slot_col (T, 2, 11)   player column held by each slot (index into player_cols[team]), -1 = empty
  player_cols (2, C)    player column names per team ('home_home_7'), '' = padding
  ball   (T, 3)         ball xyz
  ball_v (T, 2)
  alive  (T,)           ball in play
  period (T,)           1, 2, ... (extra time 3, 4, ...), 0 = no players (break)
  attack_sign (P+1,)    per period: +1 if HOME attacks +x (index 0 unused, length >= 3)
  label  (T,)           4-moment label, home perspective, -1 = unlabeled
  sub    (T,)           subcategory id, -1 = unlabeled
"""
from __future__ import annotations

import argparse
import re
import warnings
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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each True run."""
    ch = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(ch[::2], ch[1::2]))


def detect_periods(n_on: np.ndarray, period_col=None, hz: float = SRC_HZ,
                   min_break_s: float = 60.0, min_period_s: float = 300.0) -> np.ndarray:
    """Period id per frame (1, 2, ... ; 0 = no players on the pitch).

    With `period_col` (e.g. databallpy's `period_id`) that column is used as is (values <= 0 -> 0), unless it
    has no period at all (an unfilled column, e.g. all 0): then it is ignored with a warning.
    Otherwise periods are the stretches with players on the pitch, separated by player-free runs of at
    least `min_break_s` (shorter player-free runs are tracking dropouts inside a period); stretches shorter
    than `min_period_s` are not periods. Handles any number of periods (extra time, penalties).
    """
    empty = n_on == 0
    T = len(empty)
    if period_col is not None:
        p = pd.to_numeric(pd.Series(np.asarray(period_col)), errors="coerce").fillna(0).to_numpy()
        p = np.where(p > 0, p, 0).astype(int)
        if (p > 0).any():
            p[empty] = 0
            return p
        warnings.warn("period column has no period > 0 (not filled): periods from player-free breaks instead")
    play = ~empty
    for a, b in _runs(empty):
        if a > 0 and b < T and b - a < min_break_s * hz:
            play[a:b] = True
    period = np.zeros(T, int)
    blocks = [(a, b) for a, b in _runs(play) if b - a >= min_period_s * hz]
    if not blocks:
        warnings.warn(f"no stretch with players of >= {min_period_s} s: all frames with players are period 1")
        period[play] = 1
    for k, (a, b) in enumerate(blocks, 1):
        period[a:b] = k
    if len(blocks) > 5:
        warnings.warn(f"{len(blocks)} periods detected (expected 2, 4 or 5); check min_break_s / min_period_s")
    period[empty] = 0
    return period


def assign_slots(present: np.ndarray, n_slots: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Map player columns to `n_slots` slots so that a slot keeps the same player while he is on the pitch.

    present: (T, C) bool. Returns col_of (T, n_slots) column index per slot (-1 = empty this frame) and
    overflow (C,) frames on which a present column found no free slot (more than `n_slots` present).
    A player who drops out keeps his slot (empty meanwhile); a column without a slot takes a slot whose
    player is absent: never-used slots first, then the one absent the longest. With more than `n_slots`
    columns present (e.g. both players during a substitution) the players already holding slots keep them.
    """
    T, C = present.shape
    col_of = np.full((T, n_slots), -1, np.int64)
    slot_col = np.full(n_slots, -1)
    last_seen = np.full(n_slots, -1)  # last frame the slot's player was present; -1 = never used
    col_slot = np.full(C, -1)
    overflow = np.zeros(C, int)
    chg = np.r_[0, 1 + np.flatnonzero((present[1:] != present[:-1]).any(1)), T] if T else np.r_[0, 0]
    for a, b in zip(chg[:-1], chg[1:]):
        P = present[a]
        here = np.zeros(n_slots, bool)
        held = slot_col >= 0
        here[held] = P[slot_col[held]]
        for c in np.flatnonzero(P & (col_slot < 0)):
            free = np.flatnonzero(~here)
            if not len(free):
                overflow[c] += b - a
                continue
            s = free[np.lexsort((free, last_seen[free]))[0]]
            if slot_col[s] >= 0:
                col_slot[slot_col[s]] = -1
            slot_col[s], col_slot[c], here[s] = c, s, True
        last_seen[here] = b - 1
        col_of[a:b] = np.where(here, slot_col, -1)
    return col_of, overflow


def home_attack_sign(d: dict, idx) -> np.ndarray:
    """+1 where the home team attacks +x at tracking index idx (d['hz'] rate), from d['period'] / d['attack_sign'].

    Frames outside a period (period 0) take the period of the nearest frame inside one."""
    per = np.asarray(d["period"]).astype(int)
    idx = np.clip(np.asarray(idx, int), 0, len(per) - 1)
    ok = np.flatnonzero(per > 0)
    p = per[idx]
    if len(ok) and (p == 0).any():
        j = np.clip(np.searchsorted(ok, idx), 1, len(ok) - 1) if len(ok) > 1 else np.zeros_like(idx)
        lo, hi = ok[np.maximum(j - 1, 0)], ok[j]
        near = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
        p = np.where(p == 0, per[near], p)
    sign = np.asarray(d["attack_sign"], float)[p]
    return np.where(sign == 0, 1.0, sign)


def load(csv_path: str | Path, min_break_s: float = 60.0, min_period_s: float = 300.0) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    T0 = len(df)

    team_xy, team_cols = [], []
    for team in ("home", "away"):
        cx, cy = _player_cols(df.columns, team)
        team_cols.append(cx)
        team_xy.append(np.stack([df[cx].to_numpy(float), df[cy].to_numpy(float)], -1))  # (T, P, 2)

    # periods: provider column if present, else player-free breaks
    n_on = sum((~np.isnan(xy[..., 0])).sum(1) for xy in team_xy)
    pcol = next((c for c in df.columns if c.strip().rstrip(";").lower() in ("period_id", "period")), None)
    period = detect_periods(n_on, None if pcol is None else df[pcol].to_numpy(), SRC_HZ, min_break_s, min_period_s)
    n_per = int(period.max())

    # direction of play per period from team centroids
    attack_sign = np.zeros(max(3, n_per + 1))
    for p in range(1, n_per + 1):
        m = period == p
        hx = np.nanmean(team_xy[0][m, :, 0]) if m.any() and np.isfinite(team_xy[0][m, :, 0]).any() else np.nan
        ax = np.nanmean(team_xy[1][m, :, 0]) if m.any() and np.isfinite(team_xy[1][m, :, 0]).any() else np.nan
        attack_sign[p] = -1.0 if hx > ax else 1.0  # NaN (no data) -> +1

    # goalkeepers: player column with the most extreme mean x toward own goal
    gk_col = np.full((2, n_per + 1), -1, int)
    for t in range(2):
        for p in range(1, n_per + 1):
            x = team_xy[t][period == p, :, 0]
            if not np.isfinite(x).any():
                continue
            own = -attack_sign[p] if t == 0 else attack_sign[p]
            cnt = np.isfinite(x).sum(0)
            mean_x = np.where(cnt > 0, np.nansum(x, 0) / np.maximum(cnt, 1), 0.0)
            gk_col[t, p] = np.argmax(mean_x * own)

    # velocity at source rate per player column, then smooth 0.2 s
    vel_src = []
    for xy in team_xy:
        v = np.gradient(xy, axis=0) * SRC_HZ
        vel_src.append(_smooth(v, 5))

    # pack to 11 slots per team; a slot keeps its player while he is on the pitch (see assign_slots)
    idx = np.arange(0, T0, STEP)
    T = len(idx)
    per = period[idx]
    rows = np.arange(T)[:, None]
    pos = np.full((T, 2, 11, 2), np.nan)
    vel = np.full((T, 2, 11, 2), np.nan)
    is_gk = np.zeros((T, 2, 11), bool)
    slot_col = np.full((T, 2, 11), -1, np.int16)
    for t, team in enumerate(("home", "away")):
        xy, v = team_xy[t][idx], vel_src[t][idx]
        present = ~np.isnan(xy[..., 0])
        col_of, overflow = assign_slots(present)
        ok = col_of >= 0
        c = np.clip(col_of, 0, None)
        pos[:, t] = np.where(ok[..., None], xy[rows, c], np.nan)
        vel[:, t] = np.where(ok[..., None], v[rows, c], np.nan)
        is_gk[:, t] = ok & (col_of == gk_col[t, per][:, None]) & (per[:, None] > 0)
        slot_col[:, t] = col_of
        long = overflow > HZ  # short overlaps (substitutions) are expected
        if long.any():
            warnings.warn(f"{team}: more than 11 players on the pitch; dropped "
                          + ", ".join(f"{team_cols[t][i][:-2]} ({overflow[i] / HZ:.1f} s)" for i in np.flatnonzero(long)))

    ball = df[["Ball_X", "Ball_Y", "Ball_Z"]].to_numpy(float)
    ball_v = _smooth(np.gradient(ball[:, :2], axis=0) * SRC_HZ, 5)

    cat = df["Annotation_Category"].fillna("").map(_first)
    sub = df["Annotation_Subcategory"].fillna("").map(_first)
    label = cat.map(CATEGORY_MAP).fillna(-1).astype(int).to_numpy()
    sub_id = sub.map({s: i for i, s in enumerate(SUBCATS)}).fillna(-1).astype(int).to_numpy()

    width = max(len(c) for c in team_cols) if any(team_cols) else 1
    player_cols = np.array([[c[:-2] for c in cs] + [""] * (width - len(cs)) for cs in team_cols])

    return dict(
        pos=pos.astype(np.float32),
        vel=np.nan_to_num(vel).astype(np.float32),
        is_gk=is_gk,
        slot_col=slot_col,
        player_cols=player_cols,
        ball=np.nan_to_num(ball[idx]).astype(np.float32),
        ball_v=np.nan_to_num(ball_v[idx]).astype(np.float32),
        alive=(df["ball_status"].to_numpy()[idx] == "alive"),
        period=per.astype(np.int8),
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
