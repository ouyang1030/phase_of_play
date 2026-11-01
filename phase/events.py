"""Event-based possession and turnovers (Opta events synced to tracking, databallpy CSV export).

Input: `event_data.csv` with columns event_type_id, team_id, is_successful, start_x/start_y
(same pitch frame as the tracking), datetime, tracking_frame (25 Hz fused-CSV frame, -999 if
databallpy could not sync the event).

Frames: synced events keep their `tracking_frame`; the rest get wall-clock time + the per-period
median clock offset of the synced events (or, for a period without synced events, the offset that puts
its first event on the period's first tracking frame), then are snapped (+-1 s) to the frame where a
player of the event team is closest to the ball. See `load` for the sync checks.
"""
from __future__ import annotations

import warnings

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


def load(csv_path: str, d: dict, src_hz: float = 25.0, snap_s: float = 1.0,
         max_ball_dist_m: float = 3.0, max_period_mismatch: float = 0.05) -> pd.DataFrame:
    """Events -> DataFrame[idx (tracking index at d['hz']), team (0 home / 1 away), type, ok, x, y].

    Checks (raise on errors that would silently corrupt possession, warn on doubtful sync):
    - `tracking_frame` must be the 0-based row index of the fused 25 Hz CSV: a synced frame past the end of
      the tracking raises (provider frame ids such as TRACAB's would otherwise all be clipped to the last frame);
    - a period without synced events is placed by aligning its first event (kick-off) with the first tracking
      frame of that period (warning); no synced events at all raises;
    - unsynced events that fall outside the tracking or have no valid time are dropped (warning), not clipped;
    - warns if, at synced on-ball events, the ball is far from every player (median > max_ball_dist_m) or the
      tracking period disagrees with the event period on more than max_period_mismatch of synced events.
    A summary is returned in `df.attrs["sync_report"]`.
    """
    e = pd.read_csv(csv_path)
    e = e[(e.period_id > 0) & (e.event_type_id != 43)].reset_index(drop=True)  # drop pre-match and deleted events
    step = int(round(src_hz / float(d["hz"])))
    T = len(d["label"])
    n_src = T * step  # 25 Hz frames covered by the tracking arrays
    per_trk = np.asarray(d["period"]).astype(int)

    tf = pd.to_numeric(e.tracking_frame, errors="coerce").to_numpy(float)
    synced = np.isfinite(tf) & (tf >= 0)
    if not synced.any():
        raise ValueError("no event has a tracking_frame >= 0: run the databallpy synchronisation first")
    beyond = synced & (tf >= n_src)
    if beyond.any():
        raise ValueError(f"{int(beyond.sum())} synced events have tracking_frame >= {n_src} (tracking length at "
                         f"{src_hz:g} Hz, max seen {int(tf[synced].max())}): tracking_frame must be the 0-based row "
                         "index of the fused CSV, not the provider frame id")

    # frame: synced where available, else clock + per-period offset
    t = pd.to_datetime(e.datetime, utc=True, errors="coerce").map(lambda x: x.timestamp()).to_numpy(float)
    f_clock = (t - np.nanmin(t)) * src_hz
    frame = np.where(synced, tf, np.nan)
    per_ev = e.period_id.to_numpy().astype(int)
    fallback = []
    for p in np.unique(per_ev):
        m = per_ev == p
        ms = m & synced & np.isfinite(f_clock)
        if ms.any():
            off = np.median(frame[ms] - f_clock[ms])
        else:
            start = np.flatnonzero(per_trk == p)
            if not len(start) or not np.isfinite(f_clock[m]).any():
                raise ValueError(f"period {p}: no synced events and no tracking frames / event times to align it")
            off = start[0] * step - np.nanmin(f_clock[m])
            fallback.append(int(p))
            warnings.warn(f"period {p}: no synced events; aligned its first event with tracking frame "
                          f"{start[0] * step} (period start)")
        frame[m & ~synced] = f_clock[m & ~synced] + off
    keep = np.isfinite(frame) & (frame >= 0) & (frame < n_src)
    if (~keep).any():
        warnings.warn(f"dropped {int((~keep).sum())} unsynced events outside the tracking or without a valid time")
    e, frame, synced = e[keep].copy(), frame[keep], synced[keep]
    e["idx"] = np.minimum(np.round(frame / step), T - 1).astype(int)
    e["synced"] = synced

    # which event team_id is the tracking home team: nearest team to the ball at synced events
    near = _nearest_team(d, e.idx.to_numpy())
    ids = e.team_id.unique()
    votes = {i: np.nanmean(near[(e.team_id == i).to_numpy() & synced] == 0) for i in ids}
    home_id = max(votes, key=lambda i: -np.inf if np.isnan(votes[i]) else votes[i])
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

    # sync sanity on synced events
    so = synced & onball
    ball_d = float(np.median(dist[e.idx.to_numpy()[so]].min(1))) if so.any() else float("nan")
    if ball_d > max_ball_dist_m:
        warnings.warn(f"at synced on-ball events the ball is {ball_d:.1f} m from the nearest player (median): "
                      "tracking_frame is probably offset from the fused CSV rows")
    mism = float(np.mean(per_trk[e.idx.to_numpy()[synced]] != e.period_id.to_numpy()[synced])) if synced.any() else 0.0
    if mism > max_period_mismatch:
        warnings.warn(f"{mism:.1%} of synced events fall in another tracking period than their event period")

    e = e.sort_values(["idx", "event_id"]).reset_index(drop=True)
    out = e.rename(columns={"event_type_id": "type", "is_successful": "ok", "start_x": "x", "start_y": "y"})[
        ["idx", "team", "type", "ok", "x", "y", "period_id", "synced", "original_event"]]
    out.attrs["sync_report"] = dict(n_events=int(len(out)), n_synced=int(synced.sum()), n_dropped=int((~keep).sum()),
                                    fallback_periods=fallback, median_ball_dist_m=round(ball_d, 2),
                                    period_mismatch=round(mism, 4))
    return out


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
