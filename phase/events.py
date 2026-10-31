"""Event-based possession and turnovers (Opta events synced to tracking, databallpy CSV export).

Input: `event_data.csv` with columns event_type_id, team_id, is_successful, start_x/start_y
(same pitch frame as the tracking), datetime, tracking_frame (25 Hz fused-CSV frame, -999 if
databallpy could not sync the event).

Frames: synced events keep their `tracking_frame`; the rest get wall-clock time + the per-period
median clock offset of the synced events, then are snapped (+-1 s) to the frame where a player of
the event team is closest to the ball.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Opta type ids that mean "this team has / takes the ball"
CONTROL = {
    1: "pass", 2: "offside pass", 3: "take on", 8: "interception", 11: "claim",
    13: "miss", 14: "post", 15: "attempt saved", 16: "goal", 49: "ball recovery",
    52: "keeper pick-up", 54: "smother",
}
TACKLE, DISPOSSESSED, BALL_TOUCH, GOAL = 7, 50, 61, 16
DEAD_BALL = {4, 5, 6}  # foul, out, corner awarded -> next possession starts from a restart
WIN_GROUP = {  # how the ball was won (context feature of a turnover)
    8: "intercept", 49: "recovery", 7: "tackle", 52: "keeper", 11: "keeper", 54: "keeper",
}


def load(csv_path: str, d: dict, src_hz: float = 25.0, snap_s: float = 1.0) -> pd.DataFrame:
    """Events -> DataFrame[idx (tracking index at d['hz']), team (0 home / 1 away), type, ok, x, y]."""
    e = pd.read_csv(csv_path)
    e = e[(e.period_id > 0) & (e.event_type_id != 43)].copy()  # drop pre-match and deleted events
    step = int(round(src_hz / float(d["hz"])))
    T = len(d["label"])

    # frame: synced where available, else clock + per-period offset
    t = pd.to_datetime(e.datetime, utc=True).map(lambda x: x.timestamp()).to_numpy()
    f_clock = (t - t.min()) * src_hz
    synced = e.tracking_frame.to_numpy() >= 0
    frame = e.tracking_frame.to_numpy().astype(float)
    for p in e.period_id.unique():
        m = e.period_id.to_numpy() == p
        off = np.median(frame[m & synced] - f_clock[m & synced])
        frame[m & ~synced] = f_clock[m & ~synced] + off
    e["idx"] = np.clip(np.round(frame / step), 0, T - 1).astype(int)
    e["synced"] = synced

    # which event team_id is the tracking home team: nearest team to the ball at synced events
    near = _nearest_team(d, e.idx.to_numpy())
    ids = e.team_id.unique()
    votes = {i: np.nanmean(near[(e.team_id == i).to_numpy() & synced] == 0) for i in ids}
    home_id = max(votes, key=votes.get)
    e["team"] = np.where(e.team_id == home_id, 0, 1)

    # snap unsynced on-ball events to the closest ball contact of their team
    w = int(round(snap_s * float(d["hz"])))
    dist = _team_ball_dist(d)  # (T, 2)
    onball = e.event_type_id.isin(list(CONTROL) + [TACKLE, DISPOSSESSED, BALL_TOUCH, 12]).to_numpy()
    for i in np.flatnonzero(~synced & onball):
        c, tm = e.idx.iat[i], e.team.iat[i]
        lo, hi = max(0, c - w), min(T, c + w + 1)
        j = lo + int(np.argmin(dist[lo:hi, tm]))
        if dist[j, tm] < 3.0:
            e.iat[i, e.columns.get_loc("idx")] = j

    e = e.sort_values(["idx", "event_id"]).reset_index(drop=True)
    return e.rename(columns={"event_type_id": "type", "is_successful": "ok", "start_x": "x", "start_y": "y"})[
        ["idx", "team", "type", "ok", "x", "y", "period_id", "synced", "original_event"]]


def _team_ball_dist(d: dict) -> np.ndarray:
    dist = np.linalg.norm(d["pos"] - d["ball"][:, None, None, :2], axis=-1)
    return np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=2)


def _nearest_team(d: dict, idx: np.ndarray) -> np.ndarray:
    return np.argmin(_team_ball_dist(d)[idx], 1).astype(float)


def _controller(row) -> int | None:
    """Team that has the ball after this event, or None if the event says nothing about it."""
    ty, ok = row.type, row.ok
    if ty in CONTROL:
        return row.team
    if ty == TACKLE:  # Opta tackle outcome 1 = tackler's team won the ball
        return row.team if ok is True or ok == "True" else None
    if ty == DISPOSSESSED:
        return 1 - row.team
    return None


def possession(e: pd.DataFrame, T: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Possession per frame (0 home / 1 away, NaN before the first touch) + turnover table.

    Turnover = first controlling event of the other team. Columns of the turnover table:
    idx, team (new possessor), how (win group), x, y (event location), dead (possession
    changed across a dead ball: foul/out/corner/goal or a period change; idx is then the
    stoppage frame, since the restart team owns the ball from there on).
    """
    poss = np.full(T, np.nan)
    cur, last, period, dead_at = None, 0, None, 0  # dead_at: frame the ball went dead, None if live
    rows = []
    for r in e.itertuples(index=False):
        if r.period_id != period:
            period, dead_at = r.period_id, r.idx
        elif r.type in DEAD_BALL and dead_at is None:
            dead_at = r.idx
        c = _controller(r)
        if c is None:
            continue
        if cur is not None:
            # across a dead ball the restart team owns the ball from the stoppage on
            split = dead_at if dead_at is not None and dead_at > last else r.idx
            poss[last:split] = cur
            poss[split:r.idx] = c
            if c != cur:
                rows.append(dict(idx=int(split), team=int(c), how=WIN_GROUP.get(r.type, "other"),
                                 x=float(r.x), y=float(r.y), dead=dead_at is not None,
                                 event=r.original_event))
        cur, last, dead_at = c, r.idx, (r.idx + 1 if r.type == GOAL else None)
    if cur is not None:
        poss[last:] = cur
    return poss, pd.DataFrame(rows)
