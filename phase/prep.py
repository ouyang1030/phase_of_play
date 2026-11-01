"""Load a fused tracking+annotation CSV into compact arrays (12.5 Hz).

Output npz keys
  pos    (T, 2, 11, 2)  player xy per team slot [0=home, 1=away], NaN if empty
  vel    (T, 2, 11, 2)  velocity m/s (0.2 s centred diff, smoothed)
  is_gk  (T, 2, 11)     bool (from the roster, else goalkeeper_columns)
  player_is_gk (2, C)   goalkeeper flag per player column (aligned with player_cols)
  slot_col (T, 2, 11)   player column held by each slot (index into player_cols[team]), -1 = empty
  player_cols (2, C)    player column names per team ('home_home_7'), '' = padding
  ball   (T, 3)         ball xyz (gaps <= 0.5 s in play interpolated, other missing frames 0)
  ball_v (T, 2)
  ball_valid (T,)       ball xy known (measured or interpolated)
  ball_missing_play, ball_gap_max_s, ball_interp_play   scalars: share of frames in play (ball alive) without
                        ball xy, longest such run (s), share of frames in play interpolated
  alive  (T,)           ball in play
  period (T,)           1, 2, ... (extra time 3, 4, ...), 0 = outside a period (break) or no players
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


PLAYER_COL = re.compile(r"(home|away)_(home|away)_(\d+)_([xy])", re.IGNORECASE)


def canonical_column(name: str) -> str:
    """Fused-CSV column name as phase expects it: trailing ';' / blanks removed (a provider export ends its header
    with 'Annotation_Notes;;;;') and player columns as 'home_home_26_X' whatever their case ('home_Home_26_x')."""
    name = str(name).strip().rstrip(";").strip()
    m = PLAYER_COL.fullmatch(name)
    if m and m.group(1).lower() == m.group(2).lower():
        team = m.group(1).lower()
        return f"{team}_{team}_{m.group(3)}_{m.group(4).upper()}"
    return name


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
                   min_break_s: float = 60.0, min_period_s: float = 300.0, periods: dict | None = None) -> np.ndarray:
    """Period id per frame (1, 2, ... ; 0 = outside a period or no players on the pitch).

    With `periods` ({period: (kick-off frame, last frame)}, inclusive, from meta.json: phase.meta.periods) those
    frames are used; this takes precedence over everything else. With `period_col` (e.g. databallpy's `period_id`) that column is used as is (values <= 0 -> 0), unless it
    has no period at all (an unfilled column, e.g. all 0): then it is ignored with a warning.
    Otherwise periods are the stretches with players on the pitch, separated by player-free runs of at
    least `min_break_s` (shorter player-free runs are tracking dropouts inside a period); stretches shorter
    than `min_period_s` are not periods. Handles any number of periods (extra time, penalties).
    """
    empty = n_on == 0
    T = len(empty)
    if periods:
        p = np.zeros(T, int)
        prev_end = -1
        for k, (a, b) in sorted(periods.items(), key=lambda kv: kv[1][0]):
            if not (prev_end < a <= b < T):
                raise ValueError(f"period {k}: frames {a}..{b} overlap the previous period or lie outside the "
                                 f"{T} frames of the CSV (meta.json from another export?)")
            if np.mean(~empty[a:b + 1]) < 0.5:
                warnings.warn(f"period {k} (frames {a}..{b}): players on the pitch on less than half of its frames")
            p[a:b + 1], prev_end = k, b
        p[empty] = 0
        return p
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


def fill_ball_gaps(ball: np.ndarray, alive: np.ndarray, max_gap_s: float = 0.5, hz: float = SRC_HZ):
    """Linear interpolation of ball gaps (no x or y) of at most max_gap_s that lie inside play (alive on every
    frame, known ball on both sides). Returns the filled copy and the interpolated-frame mask."""
    ball = ball.copy()
    miss = ~np.isfinite(ball[:, :2]).all(1)
    filled = np.zeros(len(ball), bool)
    for a, b in _runs(miss):
        if a > 0 and b < len(ball) and b - a <= max_gap_s * hz and alive[a:b].all():
            w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
            ball[a:b] = (1 - w)[:, None] * ball[a - 1] + w[:, None] * ball[b]
            filled[a:b] = True
    return ball, filled


def ball_gap_stats(ball: np.ndarray, alive: np.ndarray, filled: np.ndarray, hz: float = SRC_HZ) -> dict:
    """Missing ball xy while the ball is in play (before interpolation): share of frames, longest run (s)."""
    miss = ~np.isfinite(ball[:, :2]).all(1) & alive
    n = max(int(alive.sum()), 1)
    longest = max((b - a for a, b in _runs(miss)), default=0)
    return dict(ball_missing_play=float(miss.sum() / n), ball_gap_max_s=float(longest / hz),
                ball_interp_play=float((filled & alive).sum() / n))


def goalkeeper_columns(xy: np.ndarray, period: np.ndarray, attack_sign: np.ndarray, team: int, hz: float = SRC_HZ,
                       min_s: float = 60.0, half_length: float = 52.5) -> np.ndarray:
    """Goalkeeper flag per player column of one team, without a roster.

    xy (T, C, 2) at `hz`. A column is a goalkeeper if, over the frames it is on the pitch inside a period, it is the
    team's player nearest to its own goal (centre of the goal line, from the period's direction of play) on more
    than half of them, and it is on the pitch for at least `min_s`. Several columns can qualify (a goalkeeper
    substituted during a half)."""
    T, C = xy.shape[:2]
    home_sign = np.asarray(attack_sign, float)[np.asarray(period, int)]  # 0 outside periods
    own_x = -half_length * home_sign * (1 if team == 0 else -1)
    on = np.isfinite(xy[..., 0]) & (home_sign != 0)[:, None]
    dist = np.where(on, np.hypot(xy[..., 0] - own_x[:, None], xy[..., 1]), np.inf)
    has = on.any(1)
    nearest = np.bincount(np.argmin(dist[has], 1), minlength=C) if has.any() else np.zeros(C, int)
    n_on = on.sum(0)
    return (nearest > 0.5 * n_on) & (n_on >= min_s * hz)


def load(csv_path: str | Path, min_break_s: float = 60.0, min_period_s: float = 300.0, roster: dict | None = None,
         min_gk_s: float = 60.0, max_gk_bad: float = 0.01, pitch_length: float = 105.0,
         periods: dict | None = None) -> dict:
    """Fused CSV -> arrays (module docstring).

    periods: {period: (kick-off frame, last frame)} of the CSV rows (meta.json, phase.meta.periods); without it
    periods come from a filled provider period column, else from player-free breaks (detect_periods).

    roster: player column name ('home_home_26') -> is goalkeeper (meta.json players, phase.meta.roster); columns
    missing from it are not goalkeepers (warning). Without a roster goalkeepers come from goalkeeper_columns.
    Warns if, on frames with the ball in play, a team has no or several goalkeepers on more than max_gk_bad of them.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    new = [canonical_column(c) for c in df.columns]
    dup = sorted({c for c in new if new.count(c) > 1})
    if dup:
        raise ValueError(f"{csv_path}: columns {dup} appear more than once after normalising case / ';'")
    df.columns = new
    T0 = len(df)

    team_xy, team_cols = [], []
    for team in ("home", "away"):
        cx, cy = _player_cols(df.columns, team)
        team_cols.append(cx)
        team_xy.append(np.stack([df[cx].to_numpy(float), df[cy].to_numpy(float)], -1))  # (T, P, 2)

    # periods: meta.json if given, else a filled provider column, else player-free breaks
    n_on = sum((~np.isnan(xy[..., 0])).sum(1) for xy in team_xy)
    pcol = next((c for c in df.columns if c.lower() in ("period_id", "period")), None)
    period = detect_periods(n_on, None if pcol is None or periods else df[pcol].to_numpy(), SRC_HZ, min_break_s,
                            min_period_s, periods)
    n_per = int(period.max())

    # direction of play per period from team centroids
    attack_sign = np.zeros(max(3, n_per + 1))
    for p in range(1, n_per + 1):
        m = period == p
        hx = np.nanmean(team_xy[0][m, :, 0]) if m.any() and np.isfinite(team_xy[0][m, :, 0]).any() else np.nan
        ax = np.nanmean(team_xy[1][m, :, 0]) if m.any() and np.isfinite(team_xy[1][m, :, 0]).any() else np.nan
        attack_sign[p] = -1.0 if hx > ax else 1.0  # NaN (no data) -> +1

    # goalkeepers per player column: roster if given, else nearest to own goal most of the time (goalkeeper_columns)
    gk_flag = []
    for t, team in enumerate(("home", "away")):
        names = [c[:-2] for c in team_cols[t]]
        if roster is not None:
            missing = [n for n in names if n not in roster]
            if missing:
                warnings.warn(f"{team}: no roster entry for {missing}: not goalkeepers")
            gk_flag.append(np.array([bool(roster.get(n, False)) for n in names], bool))
        else:
            gk_flag.append(goalkeeper_columns(team_xy[t], period, attack_sign, t, SRC_HZ, min_gk_s, pitch_length / 2))

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
        is_gk[:, t] = ok & gk_flag[t][c] & (per[:, None] > 0) if len(gk_flag[t]) else False
        slot_col[:, t] = col_of
        long = overflow > HZ  # short overlaps (substitutions) are expected
        if long.any():
            warnings.warn(f"{team}: more than 11 players on the pitch; dropped "
                          + ", ".join(f"{team_cols[t][i][:-2]} ({overflow[i] / HZ:.1f} s)" for i in np.flatnonzero(long)))

    alive = df["ball_status"].to_numpy()[idx] == "alive"
    play = alive & (per > 0)
    for t, team in enumerate(("home", "away")):
        bad = float(np.mean(is_gk[play, t].sum(1) != 1)) if play.any() else 0.0
        if bad > max_gk_bad:
            gks = [team_cols[t][i][:-2] for i in np.flatnonzero(gk_flag[t])]
            warnings.warn(f"{team}: {bad:.1%} of frames with the ball in play do not have exactly one goalkeeper on the "
                          f"pitch (goalkeeper columns {gks}{'' if roster is not None else ', no roster'})")

    ball_raw = df[["Ball_X", "Ball_Y", "Ball_Z"]].to_numpy(float)
    alive_src = df["ball_status"].to_numpy() == "alive"
    ball, filled = fill_ball_gaps(ball_raw, alive_src)
    ball_qc = ball_gap_stats(ball_raw, alive_src, filled)
    ball_v = _smooth(np.gradient(ball[:, :2], axis=0) * SRC_HZ, 5)

    cat = df["Annotation_Category"].fillna("").map(_first)
    sub = df["Annotation_Subcategory"].fillna("").map(_first)
    label = cat.map(CATEGORY_MAP).fillna(-1).astype(int).to_numpy()
    sub_id = sub.map({s: i for i, s in enumerate(SUBCATS)}).fillna(-1).astype(int).to_numpy()

    width = max(len(c) for c in team_cols) if any(team_cols) else 1
    player_cols = np.array([[c[:-2] for c in cs] + [""] * (width - len(cs)) for cs in team_cols])
    player_is_gk = np.array([np.r_[f, np.zeros(width - len(f), bool)] for f in gk_flag], bool)

    return dict(
        pos=pos.astype(np.float32),
        vel=np.nan_to_num(vel).astype(np.float32),
        is_gk=is_gk,
        slot_col=slot_col,
        player_cols=player_cols,
        player_is_gk=player_is_gk,
        ball=np.nan_to_num(ball[idx]).astype(np.float32),
        ball_v=np.nan_to_num(ball_v[idx]).astype(np.float32),
        ball_valid=np.isfinite(ball[idx, :2]).all(1),
        **{k: np.float32(v) for k, v in ball_qc.items()},
        alive=alive,
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
    ap.add_argument("--meta", help="meta.json: periods (kick-off / last frames), goalkeepers from its players "
                                   "(roster) and the pitch length")
    a = ap.parse_args()
    kw = {}
    if a.meta:
        from . import meta as M
        m = M.load(a.meta)
        kw = dict(roster=M.roster(m) if m.get("players") else None, pitch_length=float(m["pitch"]["length"]),
                  periods=M.periods(m))
    d = load(a.csv, **kw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    lab = d["label"]
    print(f"ball in play: {d['ball_missing_play']:.2%} of frames without ball xy, longest gap "
          f"{d['ball_gap_max_s']:.1f} s, {d['ball_interp_play']:.2%} interpolated (gaps <= 0.5 s)")
    print("periods (12.5 Hz frames):", {int(p): (int(np.flatnonzero(d["period"] == p)[0]),
                                                   int(np.flatnonzero(d["period"] == p)[-1]))
                                         for p in np.unique(d["period"]) if p > 0})
    print(f"T={len(lab)} @ {HZ} Hz | labeled {np.mean(lab >= 0):.1%} | "
          f"attack_sign={d['attack_sign'][1:]} | class counts "
          + str({CLASSES[c]: int((lab == c).sum()) for c in range(4)}))
