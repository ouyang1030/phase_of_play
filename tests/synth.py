"""Synthetic fused CSVs, tracking dicts and event tables for the unit tests (no licensed data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HZ = 12.5  # tracking dict rate (phase.prep output)


def fused_csv(path, T: int, n_cols: int = 11, empty=(), flip_from=None, drop=None, extra=None, period_col=None):
    """Write a fused 25 Hz CSV.

    Player column j of each team stands at a fixed spot (x = side * (5 + 3 j), y = j - 5), column 0 is the
    goalkeeper (x = side * 50). Home defends -x (side -1) until frame `flip_from`, then the teams swap ends.
    empty: frame ranges with no players (breaks). drop: {(team, col): [(a, b), ...]} frames a column is NaN.
    extra: {(team, col): (a, b)} columns that are present only on [a, b) (substitutes), NaN elsewhere.
    """
    drop, extra = drop or {}, extra or {}
    cols = {}
    flip = np.ones(T)
    if flip_from is not None:
        flip[flip_from:] = -1
    for t, team in enumerate(("home", "away")):
        side = (-1 if t == 0 else 1) * flip
        for j in range(n_cols):
            x = side * (50.0 if j == 0 else 5.0 + 3 * j)
            y = np.full(T, j - 5.0)
            x = x.copy()
            if (t, j) in extra:
                a, b = extra[(t, j)]
                m = np.ones(T, bool)
                m[a:b] = False
                x[m], y[m] = np.nan, np.nan
            for a, b in drop.get((t, j), []):
                x[a:b], y[a:b] = np.nan, np.nan
            for a, b in empty:
                x[a:b], y[a:b] = np.nan, np.nan
            cols[f"{team}_{team}_{j + 1}_X"], cols[f"{team}_{team}_{j + 1}_Y"] = x, y
    df = pd.DataFrame(cols)
    df["Ball_X"], df["Ball_Y"], df["Ball_Z"] = 0.0, 0.0, 0.0
    df["ball_status"] = "alive"
    df["Annotation_Category"] = "Attacking"
    df["Annotation_Subcategory"] = "Build up"
    if period_col is not None:
        df["period_id"] = period_col
    df.to_csv(path, index=False)
    return path


def tracking(T: int, periods, attack_sign=(0.0, 1.0, -1.0), ball=(0.0, 30.0)):
    """12.5 Hz tracking dict: period array from [(period, a, b), ...]; home players near (-10 * sign, y)."""
    period = np.zeros(T, np.int8)
    for p, a, b in periods:
        period[a:b] = p
    sign = np.array(attack_sign, np.float32)[period]
    sign = np.where(sign == 0, 1, sign)
    pos = np.zeros((T, 2, 11, 2), np.float32)
    for t, s0 in ((0, -1), (1, 1)):
        pos[:, t, :, 0] = (s0 * sign)[:, None] * (10 + np.arange(11))[None]
        pos[:, t, :, 1] = np.arange(11) - 5
    b = np.zeros((T, 3), np.float32)
    b[:, 0], b[:, 1] = ball
    return dict(pos=pos, ball=b, period=period, attack_sign=np.array(attack_sign, np.float32),
                label=np.full(T, -1, np.int8), alive=np.ones(T, bool), hz=np.float32(HZ))


def put_ball_at(d, idx, team, slot=0):
    d["ball"][idx, :2] = d["pos"][idx, team, slot]


def events_csv(path, rows, t0="2024-01-01T12:00:00Z"):
    """rows: dicts with period, type, team_id, sec (wall clock from t0), frame (25 Hz tracking_frame or -999),
    x, y, ok. Writes the databallpy-style event_data.csv columns used by phase.events.load."""
    base = pd.Timestamp(t0)
    out = []
    for i, r in enumerate(rows):
        out.append(dict(event_id=i, period_id=r["period"], event_type_id=r["type"], team_id=r["team_id"],
                        is_successful=r.get("ok", True), start_x=r.get("x", 0.0), start_y=r.get("y", 0.0),
                        datetime=(base + pd.Timedelta(seconds=r["sec"])).isoformat(),
                        tracking_frame=r.get("frame", -999), original_event=r.get("name", "pass")))
    pd.DataFrame(out).to_csv(path, index=False)
    return path
