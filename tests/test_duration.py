"""phase.duration: targets, decoding, metrics and the end-to-end CV run (contribution 2 / main model)."""
import json

import numpy as np
import pandas as pd
import pytest

from phase import duration as Du
from phase import episodes as P
from phase import prep
from synth import fused_csv

HZ = 12.5


def obs_table(rows):
    return pd.DataFrame(rows, columns=["lo", "hi", "W", "kind", "lag"])


def test_targets():
    A = obs_table([(3, 3, 10, "exact", 0), (0, 10, 10, "none", 0), (2, 5, 10, "interval", 0)])
    B = obs_table([(5, 5, 10, "exact", 0), (4, 4, 10, "exact", 0), (0, 10, 10, "none", 0)])
    tgt, ok = Du.targets({"label": A}, "label")
    assert tgt["mode"] == "range" and tgt["lo"].tolist() == [3, 0, 2] and ok.tolist() == [True, False, True]
    tgt, ok = Du.targets({"A": A, "B": B}, "interval")
    assert list(zip(tgt["lo"], tgt["hi"])) == [(3, 5), (4, 4), (2, 5)] and ok.all()  # hull of the usable coders
    tgt, ok = Du.targets({"A": A, "B": B}, "noise")
    assert tgt["lo"].shape == (3, 2) and tgt["use"].tolist() == [[True, True], [False, True], [True, False]]
    with pytest.raises(ValueError):
        Du.targets({"A": A, "B": B}, "label")


def test_quantiles_frame_probs_and_scores():
    p = np.array([0.1, 0.2, 0.3, 0.4])  # W = 3: the last entry is "fills the window"
    assert [Du.quantile_length(p, q) for q in (0.05, 0.3, 0.31, 0.6, 1.0)] == [0, 1, 2, 2, 3]
    poss = np.r_[np.zeros(5), np.ones(5)]
    eps = pd.DataFrame(dict(t0=[5], s0=[5], t1=[8], team=[1], dead=[False], how=["other"]))
    fp = Du.frame_probs(poss, eps, {0: p}, 10)
    assert np.allclose(fp.sum(1), 1) and np.allclose(fp[5:8, 3], [0.9, 0.7, 0.4]) and (fp[8:, 2] == 1).all()
    assert (fp[:5, 0] == 1).all()
    assert Du.crps(np.eye(4)[2], 2, HZ) == 0 and Du.crps(np.eye(4)[0], 2, HZ) == pytest.approx(2 / HZ)


def test_duration_metrics():
    o = obs_table([(4, 4, 10, "exact", 0), (0, 0, 10, "exact", 0), (2, 6, 10, "interval", 0), (10, 10, 10, "exact", 0)])
    m = Du.duration_metrics(np.array([4 + 25, 0, 3, 10]), o, HZ)  # interval and overflow rows are not scored
    assert m["n_exact"] == 2 and m["mae_s"] == pytest.approx(1.0) and m["zero_acc"] == 1.0 and m["within_1s"] == 0.5
    m = Du.duration_metrics(np.array([4, -1, 3, 10]), o, HZ, probs=[np.eye(11)[4], None, None, None])
    assert m["n_exact"] == 1 and m["crps_s"] == 0 and m["cover80"] == 1.0


def labels_for(T, rng):
    """Home-perspective labels following the possession grammar, possessions of 8-16 s."""
    runs, n = [], 0
    while n < T:
        k = int(rng.integers(100, 200))
        runs.append(np.full(k, float(len(runs) % 2)))
        n += k
    poss = np.concatenate(runs)[:T]
    eps = P.episodes(poss, np.ones(T, bool), np.ones(T, int), HZ, 30)
    E_true = np.minimum(rng.integers(0, 60, len(eps)), P.lengths(eps))
    return P.reconstruct(poss, eps, E_true), poss, eps, E_true


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dur")
    d = prep.load(fused_csv(tmp / "f.csv", T=4000), min_break_s=1, min_period_s=2, min_gk_s=1)
    T = len(d["label"])
    label, poss, eps, E_true = labels_for(T, np.random.default_rng(0))
    d["label"] = label.astype(np.int8)
    d["alive"] = np.ones(T, bool)
    B = P.reconstruct(poss, eps, np.minimum(E_true + 6, P.lengths(eps)))  # coder B ends transitions 0.5 s later
    np.savez(tmp / "m.npz", **d)
    np.savez(tmp / "A.npz", label=label.astype(np.int8))
    np.savez(tmp / "B.npz", label=B.astype(np.int8))
    return tmp


def run(tmp, *args):
    cfg = vars(Du.parser().parse_args(["--npz", str(tmp / "m.npz"), "--out", str(tmp / "out"), "--poss", "labels",
                                       "--folds", "2", "--n_boot", "20", *args]))
    with pytest.warns(UserWarning, match="oracle"):
        return Du.run(cfg), tmp / "out"


def test_baselines_without_torch(match):
    s, out = run(match, "--no-model")
    assert set(s["results"]) == {"timer", "rule"} and s["config"]["model"] is False
    assert s["results"]["timer"]["macro_f1"] > 0.5
    o = np.load(out / "oof.npz")
    assert str(o["method"]) == "rule" and len(o["pred"]) == len(o["gt"]) and "probs" not in o.files
    tab = pd.read_csv(out / "episodes.csv")
    assert {"E_timer_s", "E_rule_s", "rule_reason", "obs_label_lo_s"} <= set(tab.columns)
    assert json.loads((out / "results.json").read_text())["n_episodes"] == s["n_episodes"]


@pytest.mark.parametrize("target", ["label", "interval", "noise"])
def test_model_runs_end_to_end(match, target):
    pytest.importorskip("torch")
    s, out = run(match, "--target", target, "--coders", str(match / "A.npz"), str(match / "B.npz"),
                 "--epochs", "3", "--ensemble", "1", "--d", "8", "--h", "4")
    assert set(s["results"]) == {"timer", "rule", "model"}
    assert set(s["durations"]["model"]) == {"label", "A", "B", "A|B interval"}
    o = np.load(out / "oof.npz")
    assert str(o["method"]) == "model" and np.allclose(o["probs"].sum(1), 1)
    assert all("q" in f and f["model"]["n_train"] > 0 for f in s["folds"])
    if target == "noise":
        assert len(s["folds"][0]["model"]["coder_offset_s"][0]) == 2
    tab = pd.read_csv(out / "episodes.csv")
    assert tab.model_median_s.notna().any() and (tab.model_p_fill.dropna() <= 1).all()


def test_needs_two_coders_for_interval(match):
    cfg = vars(Du.parser().parse_args(["--npz", str(match / "m.npz"), "--out", str(match / "x"), "--poss", "labels",
                                       "--target", "interval"]))
    with pytest.raises(ValueError, match="two coders"):
        Du.run(cfg)


def organisation_match(seed=0, T=4000):
    """Tracking where the transition end is visible: after each turnover the new defending team stands ahead of the
    ball and runs back (vx 4 m/s toward its goal) until the planted end E, then forms a compact block goal-side."""
    rng = np.random.default_rng(seed)
    label, poss, eps, E_true = labels_for(T, rng)
    pos = np.zeros((T, 2, 11, 2))
    vel = np.zeros((T, 2, 11, 2))
    trans = np.zeros(T, bool)
    for s0, t1, e in zip(eps.s0, eps.t1, E_true):
        trans[s0:s0 + e] = True
    for t in range(T):
        w = int(poss[t])
        sign = 1.0 if w == 0 else -1.0  # raw coordinates: home attacks +x
        att = np.c_[rng.normal(-5, 8, 11), rng.normal(0, 15, 11)]
        att[0] = (-50, 0)
        if trans[t]:  # disorganised: ahead of the ball, spread out, running back
            dfd = np.c_[rng.normal(-12, 6, 11), rng.normal(0, 20, 11)]
            vel[t, 1 - w, 1:, 0] = 4.0 * sign
        else:  # organised block between the ball and the goal
            dfd = np.c_[rng.normal(20, 4, 11), rng.normal(0, 8, 11)]
        dfd[0] = (50, 0)
        pos[t, w], pos[t, 1 - w] = att * sign, dfd * sign
    gk = np.zeros((T, 2, 11), bool)
    gk[:, :, 0] = True
    d = dict(pos=pos.astype(np.float32), vel=vel.astype(np.float32), is_gk=gk,
             slot_col=np.tile(np.arange(11), (T, 2, 1)).astype(np.int16),
             ball=np.zeros((T, 3), np.float32), ball_v=np.zeros((T, 2), np.float32), alive=np.ones(T, bool),
             period=np.ones(T, np.int8), attack_sign=np.array([0, 1, 1], np.float32), label=label.astype(np.int8),
             sub=np.full(T, -1, np.int8), hz=np.float32(HZ))
    return d


def test_model_and_rule_find_the_organisation_signal(tmp_path):
    pytest.importorskip("torch")
    np.savez(tmp_path / "m.npz", **organisation_match())
    cfg = vars(Du.parser().parse_args(["--npz", str(tmp_path / "m.npz"), "--out", str(tmp_path / "out"),
                                       "--poss", "labels", "--folds", "2", "--n_boot", "20", "--epochs", "25",
                                       "--ensemble", "1", "--features", "concepts,ball,time,episode"]))
    with pytest.warns(UserWarning, match="oracle"):
        s = Du.run(cfg)
    mae = {k: s["durations"][k]["label"]["mae_s"] for k in ("timer", "rule", "model")}
    f1 = {k: s["results"][k]["macro_f1"] for k in ("timer", "rule", "model")}
    assert mae["model"] < 0.5 * mae["timer"] and mae["rule"] < 0.5 * mae["timer"], mae
    assert f1["model"] > f1["timer"] and f1["rule"] > f1["timer"], f1


def test_event_possession_path_uses_the_turnover_table(match, monkeypatch):
    """--poss events: windows from the event turnover table (dead-ball flag, win group), no labels involved."""
    d = np.load(match / "m.npz")
    label = d["label"].astype(int)
    poss = P.label_possession(label)
    poss = pd.Series(poss).ffill().bfill().to_numpy()
    t0 = np.flatnonzero(np.diff(poss) != 0) + 1
    to = pd.DataFrame(dict(idx=t0, team=poss[t0].astype(int), dead=np.arange(len(t0)) % 3 == 0,
                           how=np.where(np.arange(len(t0)) % 2, "tackle", "intercept"), x=0.0, y=0.0))
    monkeypatch.setattr(Du, "possession", lambda cfg, d, label: (poss, to))
    cfg = vars(Du.parser().parse_args(["--npz", str(match / "m.npz"), "--out", str(match / "ev"), "--events", "x.csv",
                                       "--folds", "2", "--n_boot", "20", "--no-model"]))
    s = Du.run(cfg)
    tab = pd.read_csv(match / "ev" / "episodes.csv")
    assert s["n_episodes"] == len(to) and set(tab.how) == {"tackle", "intercept"} and tab.dead.any()
    assert s["results"]["timer"]["macro_f1"] > 0.5  # static players here: the rule has no signal to use
