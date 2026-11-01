"""Synthetic fused CSVs, tracking dicts and event tables for the unit tests (no licensed data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

HZ = 12.5  # tracking dict rate (phase.prep output)


def fused_csv(path, T: int, n_cols: int = 11, empty=(), flip_from=None, drop=None, extra=None, period_col=None,
              spot=None, ball=None, status=None):
    """Write a fused 25 Hz CSV.

    Player column j of each team stands at a fixed spot (x = side * (5 + 3 j), y = j - 5), column 0 is the
    goalkeeper (x = side * 50). Home defends -x (side -1) until frame `flip_from`, then the teams swap ends.
    empty: frame ranges with no players (breaks). drop: {(team, col): [(a, b), ...]} frames a column is NaN.
    extra: {(team, col): (a, b)} columns that are present only on [a, b) (substitutes), NaN elsewhere.
    spot: {(team, col): |x|} overrides the distance from the halfway line (50 = in goal).
    ball: (T, 3) ball xyz (default: at the centre spot). status: (T,) ball_status (default: all "alive").
    """
    drop, extra, spot = drop or {}, extra or {}, spot or {}
    cols = {}
    flip = np.ones(T)
    if flip_from is not None:
        flip[flip_from:] = -1
    for t, team in enumerate(("home", "away")):
        side = (-1 if t == 0 else 1) * flip
        for j in range(n_cols):
            x = side * spot.get((t, j), 50.0 if j == 0 else 5.0 + 3 * j)
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
    df["Ball_X"], df["Ball_Y"], df["Ball_Z"] = (0.0, 0.0, 0.0) if ball is None else np.asarray(ball, float).T
    df["ball_status"] = "alive" if status is None else np.asarray(status)
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


def meta_json(path, home=("Ajax", 10), away=("PSV", 20), period_start=None, **kw):
    """meta.json as written by notebooks/preprocessing.ipynb (phase.meta)."""
    import json
    m = dict(match_id="m1", home_team=dict(name=home[0], id=home[1]), away_team=dict(name=away[0], id=away[1]),
             period_start_frame={str(k): v for k, v in (period_start or {1: 0, 2: 400}).items()},
             pitch=dict(length=105.0, width=68.0), frame_rate=25.0, **kw)
    path.write_text(json.dumps(m))
    return path


def adj_dir(path, T25=800, sub=True):
    """Codebook-v1 adjudication dir (as codebook_v1.convert + make_pack write it) for a 32 s match at 25 Hz.

    Coder A: IP [0, 300), TD [300, 400), OP [400, 800); coder B: IP [0, 320), TD [320, 450), OP [450, 800);
    the ball is dead on [600, 650). Disagreements [300, 320) and [400, 450) are the clips c000 and c001.
    Subcategories (sub=False: no subA / subB columns, as before item #9): A IP Build up [0, 150) then
    Progression, B Build up throughout IP; both TD Counterpressing; A OP Mid Block, B OP Mid Block before the dead
    ball and Low Block after it."""
    import json
    A = np.r_[np.zeros(300), np.full(100, 3), np.full(T25 - 400, 2)].astype(int)
    B = np.r_[np.zeros(320), np.full(130, 3), np.full(T25 - 450, 2)].astype(int)
    alive = np.ones(T25, bool)
    alive[600:650] = False
    A[~alive] = B[~alive] = 5
    final = np.where(A == B, A, -9)
    src = np.where(A == B, "agree", "pending").astype(object)
    src[~alive] = "inactive"
    path.mkdir(parents=True, exist_ok=True)
    cols = {"Frame": np.arange(T25), "alive": alive.astype(int), "A": A, "B": B, "final": final, "src": src}
    if sub:
        sub_of = {0: 0, 3: 8, 2: 6}  # Build up, Counterpressing, Mid Block
        sA = np.array([sub_of.get(x, -1) for x in A])
        sB = np.array([sub_of.get(x, -1) for x in B])
        sA[150:300] = 1  # Progression
        sB[650:] = np.where(B[650:] == 2, 7, sB[650:])  # Low Block
        cols.update(subA=sA, subB=sB)
    pd.DataFrame(cols).to_csv(path / "labels_v1_25hz.csv", index=False)
    pd.DataFrame([dict(clip_id="c000", start=300, end=319, dur_s=0.8, type="BOUNDARY", A="TD@300", B="IP@300"),
                  dict(clip_id="c001", start=400, end=449, dur_s=2.0, type="BOUNDARY", A="OP@400", B="TD@400")]
                 ).to_csv(path / "clips.csv", index=False)
    pd.DataFrame([dict(onset=0, end=600, kind="kickoff", team=0, x_att=0.0, zone="middle", set_piece=False),
                  dict(onset=650, end=T25, kind="throw_in", team=1, x_att=0.0, zone="middle", set_piece=False)]
                 ).to_csv(path / "restarts.csv", index=False)
    (path / "key_DO_NOT_SHARE.json").write_text(json.dumps({"R1": "A", "R2": "B", "seed": 0}))
    return path
