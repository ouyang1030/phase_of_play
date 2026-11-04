"""phase.concepts: §2.5 criteria as tracking quantities and the rule baseline (contribution 2)."""
import numpy as np
import pandas as pd
import pytest

from phase import concepts as C
from phase import episodes as P


def frame():
    """One frame, home attacks +x: away defends the goal at x = +52.5, ball at (20, 0)."""
    pos = np.full((2, 11, 2), np.nan)
    pos[1, 0] = (50, 0)  # away goalkeeper
    pos[1, 1:5] = [(40, -15), (41, -5), (42, 5), (44, 15)]  # back four: spread 4 m
    pos[1, 5:8] = [(30, -10), (31, 0), (29, 10)]  # (31, 0) stands in the ball - goal cone
    pos[1, 8:11] = [(15, -5), (10, 0), (5, 5)]  # three defenders not goal-side of the ball
    pos[0, 0] = (-50, 0)
    pos[0, 1] = (35, 25)  # ahead of the ball, nobody goal-side within 5 m of y = 25 -> uncovered
    pos[0, 2] = (38, 4)  # covered by (40, -5)? no: |-5 - 4| = 9; by (42, 5): yes
    pos[0, 3:11] = [(x, y) for x, y in zip(range(0, 16, 2), range(-20, 20, 5))]  # behind the ball
    vel = np.zeros((2, 11, 2))
    vel[1, 8] = (4.0, 0.0)  # running back toward its own goal
    gk = np.zeros((2, 11), bool)
    gk[:, 0] = True
    return pos, vel, gk


def match():
    pos, vel, gk = frame()
    d = dict(pos=np.stack([pos, -pos]), vel=np.stack([vel, -vel]), is_gk=np.stack([gk, gk]),
             ball=np.array([[20.0, 0, 0], [-20.0, 0, 0]]), period=np.array([1, 2]),
             attack_sign=np.array([0.0, 1.0, -1.0]))
    return d  # frame 1 = frame 0 mirrored in the second half (home attacks -x)


def test_concepts_of_one_frame_and_orientation():
    c = C.compute(match())[0]
    q = dict(zip(C.NAMES, c[0]))
    assert (q["c1_goalside"], q["c1_back_spread"], q["c2_recovering"]) == (7, 4, 1)
    assert (q["c3_length"], q["c4_uncovered"], q["c4_cone"]) == (39, 1, 1)
    assert q["c3_line_gap"] > 0
    assert np.allclose(c[0], c[1])  # same situation after the teams swap ends


def test_concepts_of_the_other_view_have_the_other_team_defending():
    c = C.compute(match())[1][0]
    q = dict(zip(C.NAMES, c))
    assert q["c2_recovering"] == 0  # home defenders do not move
    assert q["c1_goalside"] == 8  # the eight home outfield players at x in [0, 14], between the ball and their goal


def rule_case(hz=1.0):
    """Two episodes of team 0 at hz = 1: criteria hold briefly at k = 4, 5, then from k = 10 on; the second never."""
    T = 100
    conc = {w: np.tile(np.array([2, 0, 0, 30, 5, 0, 1], np.float32), (T, 1)) for w in (0, 1)}
    conc[0][14:16, 0] = 8  # blip
    conc[0][20:40, 0] = 8  # C1 from frame 20 (k = 10)
    conc[0][18:40, 3] = 30  # C3 already holds
    eps = pd.DataFrame(dict(t0=[10, 50], s0=[10, 50], t1=[40, 70], team=[0, 0], dead=[False, False], how=["other"] * 2))
    return conc, eps


def test_rule_lengths_hold_and_never():
    conc, eps = rule_case()
    r = C.Rule(conc, eps, hz=1.0)
    assert r.lengths(8, 0, 30.0, 0, 0.0).tolist() == [4, 20]  # first frame with all four; never -> W = 20
    assert r.lengths(8, 0, 30.0, 0, 3.0).tolist() == [10, 20]  # the 2-frame blip does not hold for 3 s
    conc[0][36:40, 0] = 2
    conc[0][37:40, 0] = 8  # criteria hold again only 3 frames before the window ends at 40 (k = 27..29)
    conc[0][20:36, 0] = 2
    assert C.Rule(conc, eps, 1.0).lengths(8, 0, 30.0, 0, 5.0).tolist() == [27, 20]  # holds to the window end


def test_rule_explains_the_last_criterion():
    conc, eps = rule_case()
    conc[0][:, 3] = 45.0
    conc[0][22:40, 3] = 30.0  # C3 (lines connected) is the last to hold, from k = 12
    r = C.Rule(conc, eps, 1.0)
    p = dict(th1=8, th2=0, th3=30.0, th4=0, hold_s=3.0)
    E_ = r.lengths(**p)
    assert E_.tolist() == [12, 20] and r.explain(p, E_) == ["C3", "never"]
    conc[0][:, :] = [8, 0, 0, 30, 5, 0, 1]
    assert C.Rule(conc, eps, 1.0).explain(p, [0, 20]) == ["at_start", "never"]


def test_rule_fit_recovers_its_own_labels():
    conc, eps = rule_case()
    poss = np.zeros(100)
    r = C.Rule(conc, eps, 1.0)
    truth = dict(th1=8, th2=0, th3=30.0, th4=0, hold_s=3.0)
    label = P.reconstruct(poss, eps, r.lengths(**truth))
    grid = dict(th1=np.array([5, 8]), th2=np.array([0, 1]), th3=np.array([30.0, 50.0]), th4=np.array([0]),
                hold_s=np.array([0.0, 3.0]))
    sc = P.EpisodeScorer(label, poss, eps)
    params, f1 = r.fit(sc, grid)
    assert (r.lengths(**params) == r.lengths(**truth)).all()
    assert f1 == pytest.approx(sc.macro_f1(r.lengths(**truth)))  # 0.5: two of the four classes occur
