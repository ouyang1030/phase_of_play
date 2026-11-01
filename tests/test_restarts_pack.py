"""tools/adj: restart zone and SET_PIECE rule end follow the direction of play of each period (review item 5)."""
from collections import namedtuple

import numpy as np
import pandas as pd

import make_pack
import restarts
from synth import events_csv, put_ball_at, tracking

R = namedtuple("R", "onset end team")


def test_free_kick_zone_in_second_half(tmp_path):
    d = tracking(400, [(1, 0, 200), (2, 200, 400)], attack_sign=(0, 1, -1))  # home attacks -x after the break
    alive25 = np.ones(800, bool)
    alive25[500:560] = False  # stoppage, restart at 25 Hz frame 560 (12.5 Hz index 280)
    d["ball"][280, 0] = -40.0  # 40 m into the half home attacks in period 2
    put_ball_at(d, 245, team=1)
    put_ball_at(d, 285, team=0)
    ev = events_csv(tmp_path / "ev.csv", [
        dict(period=2, type=4, team_id=20, sec=19.6, frame=490, name="foul"),
        dict(period=2, type=1, team_id=10, sec=22.8, frame=570, name="pass")])
    rs = restarts.detect(alive25, d, str(ev))
    fk = rs.iloc[1]
    assert fk.kind == "free_kick" and fk.team == 0
    assert fk.x_att == 40.0 and fk.zone == "final"


def test_sp_rule_end_uses_period_direction():
    d = tracking(400, [(1, 0, 200), (2, 200, 400)], attack_sign=(0, 1, -1))
    # period 2: home attacks -x. A home pass at x=-30 stays in the attacking half; x=+10 is its own half.
    e = pd.DataFrame(dict(idx=[260, 270], team=[0, 0], type=[1, 1], x=[-30.0, 10.0]))
    assert make_pack.sp_rule_end(R(onset=500, end=700, team=0), e, d, 2) == 540
    # period 1, mirrored: home attacks +x
    e1 = pd.DataFrame(dict(idx=[60, 70], team=[0, 0], type=[1, 1], x=[30.0, -10.0]))
    assert make_pack.sp_rule_end(R(onset=100, end=300, team=0), e1, d, 2) == 140


def test_clock_handles_extra_time():
    start = {1: 0, 2: 100_000, 3: 200_000}
    assert make_pack.clock(100_000 + 25 * 60, start) == (2, "46:00")
    assert make_pack.clock(200_000, start) == (3, "90:00")


def test_clock_in_a_break_counts_down_to_the_next_kickoff():
    start, end = {1: 0, 2: 95369}, {1: 69023, 2: 168828}
    assert make_pack.clock(94246, start, end) == (2, "-00:44")  # ball 'alive' during half-time
    assert make_pack.clock(95369 + 25 * 77, start, end) == (2, "46:17")
    assert make_pack.clock(69000, start, end) == (1, "46:00")
    assert make_pack.clock(94246, start) == (1, "62:49")  # without period ends
