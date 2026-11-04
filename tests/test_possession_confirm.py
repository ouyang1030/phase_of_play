"""phase.events.possession(confirm=...): gains confirmed by the second controlled action (codebook v4.1 §2.4)."""
import numpy as np
import pandas as pd
import pytest

from phase import events as Ev
from phase import hazard

A, B = 0, 1
PASS, TAKE_ON, TACKLE, INTERCEPT, RECOVERY, FOUL, OUT, GOAL, BALL_TOUCH, DISPOSSESSED = 1, 3, 7, 8, 49, 4, 5, 16, 61, 50
T = 1000


# ---------------------------------------------------------------- the implementation before confirm=
def old_possession(e, T):
    poss = np.full(T, np.nan)
    cur, last, period, dead_at = None, 0, None, 0
    rows = []
    for r in e.itertuples(index=False):
        if r.period_id != period:
            period, dead_at = r.period_id, r.idx
        elif r.type in Ev.DEAD_BALL and dead_at is None:
            dead_at = r.idx
        c = Ev._controller(r)
        if c is None:
            continue
        if cur is not None:
            split = dead_at if dead_at is not None and dead_at > last else r.idx
            poss[last:split] = cur
            poss[split:r.idx] = c
            if c != cur:
                rows.append(dict(idx=int(split), team=int(c), how=Ev.WIN_GROUP.get(r.type, "other"),
                                 x=float(r.x), y=float(r.y), dead=dead_at is not None, event=r.original_event))
        cur, last, dead_at = c, r.idx, (r.idx + 1 if r.type == Ev.GOAL else None)
    if cur is not None:
        poss[last:] = cur
    return poss, pd.DataFrame(rows)


def events(*rows, period=None):
    """rows: (frame, team, type[, ok]) -> event table as phase.events.load returns it."""
    out = []
    for k, r in enumerate(rows):
        idx, team, ty = r[:3]
        out.append(dict(idx=idx, team=team, type=ty, ok=r[3] if len(r) > 3 else True, x=float(k), y=0.0,
                        period_id=1 if period is None else period[k], original_event=f"e{k}"))
    return pd.DataFrame(out)


def random_events(seed, n=150):
    rng = np.random.default_rng(seed)
    types = [PASS, PASS, PASS, TAKE_ON, TACKLE, INTERCEPT, RECOVERY, BALL_TOUCH, DISPOSSESSED, FOUL, OUT, 6, GOAL, 13]
    idx = np.sort(rng.choice(np.arange(1, T - 1), n, replace=False))
    team = np.where(rng.random(n) < 0.8, 0, 1)
    team = np.abs(np.cumsum(rng.random(n) < 0.15) % 2 - team)  # possession-like runs with noise
    return events(*[(int(i), int(t), int(rng.choice(types)), bool(rng.random() < 0.7)) for i, t in zip(idx, team)],
                  period=(idx >= T // 2).astype(int) + 1)


# ---------------------------------------------------------------- confirm="first": the old behaviour
@pytest.mark.parametrize("seed", range(8))
def test_first_is_the_old_implementation(seed):
    e = random_events(seed)
    p_old, to_old = old_possession(e, T)
    p, to = Ev.possession(e, T)
    p1, to1 = Ev.possession(e, T, confirm="first")
    assert np.array_equal(p, p_old, equal_nan=True) and np.array_equal(p1, p_old, equal_nan=True)
    pd.testing.assert_frame_equal(to, to_old)
    pd.testing.assert_frame_equal(to1, to_old)
    assert to.attrs["unconfirmed"] == 0


def test_unknown_confirm_raises():
    with pytest.raises(ValueError, match="confirm"):
        Ev.possession(events((0, A, PASS)), T, confirm="third")


# ---------------------------------------------------------------- confirm="second"
def test_loose_ball_is_not_a_change():
    e = events((0, A, PASS), (100, B, INTERCEPT), (110, A, RECOVERY), (150, A, PASS))
    p, to = Ev.possession(e, 200, confirm="second")
    assert (p == A).all() and len(to) == 0 and to.attrs["unconfirmed"] == 1
    p1, to1 = Ev.possession(e, 200)
    assert (p1[100:110] == B).all() and to1.idx.tolist() == [100, 110]


def test_confirmed_gain_is_placed_at_the_first_action():
    e = events((0, A, PASS), (100, B, INTERCEPT), (150, B, PASS), (300, B, PASS))
    p, to = Ev.possession(e, 400, confirm="second")
    p1, to1 = Ev.possession(e, 400)
    assert np.array_equal(p, p1) and (p[100:] == B).all()
    pd.testing.assert_frame_equal(to, to1)
    assert to.idx.tolist() == [100] and to.attrs["unconfirmed"] == 0


def test_successful_pass_is_confirmed_by_its_reception():
    # B's first action is a completed pass; the receiver is then tackled by A, who keeps the ball
    e = events((0, A, PASS), (100, B, PASS, True), (120, A, TACKLE, True), (160, A, PASS))
    p, to = Ev.possession(e, 200, confirm="second")
    assert to.idx.tolist() == [100, 120] and (p[100:120] == B).all()
    e = events((0, A, PASS), (100, B, PASS, False), (120, A, RECOVERY), (160, A, PASS))  # misplaced pass
    p, to = Ev.possession(e, 200, confirm="second")
    assert len(to) == 0 and (p == A).all()


def test_dead_ball_and_period_changes_need_no_confirmation():
    # A shoots wide (out at 300); B's goal kick is a misplaced pass that A recovers: B still had the restart
    e = events((0, A, PASS), (250, A, 13), (300, A, OUT), (330, B, PASS, False), (350, A, RECOVERY),
               (400, A, PASS), (600, B, PASS), (620, A, INTERCEPT), (700, A, PASS),
               period=[1, 1, 1, 1, 1, 1, 2, 2, 2])
    p, to = Ev.possession(e, 800, confirm="second")
    assert to[["idx", "team", "dead"]].values.tolist() == [[300, B, True], [350, A, False], [600, B, True],
                                                            [620, A, False]]
    assert to.attrs["unconfirmed"] == 0


def test_first_possession_needs_no_confirmation():
    e = events((10, B, INTERCEPT), (40, A, RECOVERY), (90, A, PASS))
    p, to = Ev.possession(e, 100, confirm="second")
    assert np.isnan(p[:10]).all() and (p[10:40] == B).all() and to.idx.tolist() == [40]


@pytest.mark.parametrize("seed", range(8))
def test_second_keeps_a_subset_of_the_first_turnovers(seed):
    e = random_events(seed)
    p1, to1 = Ev.possession(e, T)
    p2, to2 = Ev.possession(e, T, confirm="second")
    key = ["idx", "team", "how", "dead", "event"]
    first = set(map(tuple, to1[key].values.tolist()))
    assert set(map(tuple, to2[key].values.tolist())) <= first
    assert to2.attrs["unconfirmed"] > 0 and len(to2) < len(to1)
    assert np.array_equal(np.isnan(p1), np.isnan(p2))


# ---------------------------------------------------------------- CLI
@pytest.mark.parametrize("mod, io", [("train", ["--npz", "m.npz", "--out", "o"]),
                                     ("hazard", ["--npz", "m.npz", "--out", "o", "--events", "e.csv"])])
def test_cli_poss_confirm(mod, io):
    mod = pytest.importorskip(f"phase.{mod}")  # phase.train needs torch; the other tests here do not
    assert vars(mod.parser().parse_args(io))["poss_confirm"] == "first"
    assert vars(mod.parser().parse_args(io + ["--poss_confirm", "second"]))["poss_confirm"] == "second"
    with pytest.raises(SystemExit):
        mod.parser().parse_args(io + ["--poss_confirm", "third"])
