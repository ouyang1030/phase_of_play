"""phase.events.load: event frames and sync checks (review item 6)."""
import numpy as np
import pytest

from phase import events as Ev
from synth import events_csv, put_ball_at, tracking

HOME, AWAY = 10, 20


def match():
    d = tracking(400, [(1, 0, 180), (2, 220, 400)], attack_sign=(0, 1, -1))
    put_ball_at(d, 50, team=0)
    put_ball_at(d, 100, team=1)
    return d


P1_SYNCED = [dict(period=1, type=1, team_id=HOME, sec=4, frame=100),
             dict(period=1, type=1, team_id=AWAY, sec=8, frame=200)]


def test_synced_events_keep_their_frame_and_teams(tmp_path):
    e = Ev.load(str(events_csv(tmp_path / "e.csv", P1_SYNCED)), match())
    assert e.idx.tolist() == [50, 100] and e.team.tolist() == [0, 1]
    assert e.attrs["sync_report"]["n_synced"] == 2


def test_period_without_synced_events_is_aligned_to_its_start(tmp_path):
    # wall-clock half-time (88 s) does not match the tracking break: a global offset would put period 2
    # past the end of the tracking; the old per-period median of nothing gave NaN -> index 0
    rows = P1_SYNCED + [dict(period=2, type=1, team_id=HOME, sec=100),
                        dict(period=2, type=1, team_id=AWAY, sec=104)]
    with pytest.warns(UserWarning, match="period 2: no synced events"):
        e = Ev.load(str(events_csv(tmp_path / "e.csv", rows)), match())
    assert e.idx.tolist() == [50, 100, 220, 270]
    assert e.attrs["sync_report"]["fallback_periods"] == [2]


def test_provider_frame_ids_raise(tmp_path):
    rows = [dict(r, frame=r["frame"] + 1_000_000) for r in P1_SYNCED]
    with pytest.raises(ValueError, match="row index"):
        Ev.load(str(events_csv(tmp_path / "e.csv", rows)), match())


def test_no_synced_events_raise(tmp_path):
    rows = [dict(r, frame=-999) for r in P1_SYNCED]
    with pytest.raises(ValueError, match="synchronisation"):
        Ev.load(str(events_csv(tmp_path / "e.csv", rows)), match())


def test_unsynced_events_past_the_tracking_are_dropped_not_clipped(tmp_path):
    rows = P1_SYNCED + [dict(period=1, type=1, team_id=HOME, sec=500)]  # 12,000 frames after kick-off
    with pytest.warns(UserWarning, match="dropped 1"):
        e = Ev.load(str(events_csv(tmp_path / "e.csv", rows)), match())
    assert len(e) == 2 and (e.idx < 399).all()


def test_offset_sync_warns(tmp_path):
    d = match()
    d["ball"][:, :2] = (0.0, 30.0)  # ball far from every player at the synced on-ball events
    with pytest.warns(UserWarning) as rec:
        Ev.load(str(events_csv(tmp_path / "e.csv", P1_SYNCED)), d)
    msgs = [str(w.message) for w in rec]
    assert any("offset" in m for m in msgs)
    assert any("mapping" in m and "doubtful" in m for m in msgs)  # nearest team says nothing either
