"""meta.json: team ids / names, kick-off frames and file prefix instead of hard-coded Ajax-PSV values (items 7, 12)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

import apply_decisions
import eval_ceiling
import make_pack
from phase import events as Ev
from phase import meta as M
from synth import adj_dir, events_csv, meta_json, put_ball_at, tracking

HOME, AWAY = 10, 20


# ---------------------------------------------------------------- phase.meta
def test_load_teams_prefix(tmp_path):
    m = M.load(meta_json(tmp_path / "meta.json", period_start={1: 0, 2: 95369}))
    assert M.teams(m) == ("Ajax", "PSV")
    assert m["period_start_frame"] == {1: 0, 2: 95369}
    assert M.prefix(m) == "ajax_psv_full"
    m = M.load(meta_json(tmp_path / "m2.json", home=("Go Ahead Eagles", 1), away=("N.E.C.", 2)))
    assert M.prefix(m) == "go_ahead_eagles_n_e_c_full"
    assert M.prefix(M.load(meta_json(tmp_path / "m3.json", prefix="custom"))) == "custom"


def test_load_rejects_incomplete_meta(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps(dict(match_id="x", home_team=dict(name="A", id=1))))
    with pytest.raises(ValueError, match="missing keys"):
        M.load(tmp_path / "bad.json")


# ---------------------------------------------------------------- events.load(home_team_id)
def match():
    d = tracking(400, [(1, 0, 180), (2, 220, 400)], attack_sign=(0, 1, -1))
    put_ball_at(d, 50, team=0)
    put_ball_at(d, 100, team=1)
    return d


SYNCED = [dict(period=1, type=1, team_id=HOME, sec=4, frame=100),
          dict(period=1, type=1, team_id=AWAY, sec=8, frame=200)]


def test_home_team_id_that_agrees_with_the_vote(tmp_path):
    e = Ev.load(str(events_csv(tmp_path / "e.csv", SYNCED)), match(), home_team_id=HOME)
    assert e.team.tolist() == [0, 1]
    rep = e.attrs["sync_report"]
    assert rep["home_vote"] == 1.0 and rep["home_team_id"] == str(HOME)
    assert Ev.load(str(events_csv(tmp_path / "e.csv", SYNCED)), match(), home_team_id=str(HOME)).team.tolist() == [0, 1]


def test_home_team_id_against_the_vote_raises(tmp_path):
    with pytest.raises(ValueError, match="swapped"):
        Ev.load(str(events_csv(tmp_path / "e.csv", SYNCED)), match(), home_team_id=AWAY)
    with pytest.raises(ValueError, match="not a team_id"):
        Ev.load(str(events_csv(tmp_path / "e.csv", SYNCED)), match(), home_team_id=99)


def test_low_home_vote_warns(tmp_path):
    d = match()
    rows = SYNCED + [dict(period=1, type=1, team_id=AWAY, sec=2, frame=100)]  # away event where home has the ball
    with pytest.warns(UserWarning, match="67% of synced events"):
        e = Ev.load(str(events_csv(tmp_path / "e.csv", rows)), d)
    assert e.attrs["sync_report"]["home_vote"] == pytest.approx(0.667, abs=1e-3)


# ---------------------------------------------------------------- make_pack / apply_decisions
@pytest.fixture
def adj_match(tmp_path):
    adj = adj_dir(tmp_path / "adj")
    d = tracking(400, [(1, 0, 170), (2, 175, 400)], attack_sign=(0, 1, -1))
    put_ball_at(d, 50, team=0)
    put_ball_at(d, 210, team=1)
    d["sub"] = np.full(400, -1, np.int8)
    d["alive"] = pd.read_csv(adj / "labels_v1_25hz.csv").alive.to_numpy()[::2].astype(bool)
    np.savez_compressed(tmp_path / "trk.npz", **d)
    ev = events_csv(tmp_path / "ev.csv", [dict(period=1, type=1, team_id=HOME, sec=4, frame=100, name="pass"),
                                          dict(period=2, type=1, team_id=AWAY, sec=17, frame=420, name="pass")])
    meta = meta_json(tmp_path / "meta.json", home=("Home FC", HOME), away=("Away FC", AWAY),
                     period_start={1: 0, 2: 350})  # kick-off of period 2 at 25 Hz frame 350, not 650 (npz)
    return adj, tmp_path / "trk.npz", ev, meta


def test_make_pack_takes_teams_and_kickoffs_from_meta(adj_match, tmp_path, monkeypatch):
    adj, npz, ev, meta = adj_match
    pack = tmp_path / "pack"
    monkeypatch.setattr(sys, "argv", ["make_pack.py", str(adj), str(npz), str(ev), str(pack), "--meta", str(meta)])
    make_pack.main()
    clips = pd.read_csv(pack / "clips.csv").set_index("clip_id")
    assert clips.loc["c001", "period"] == 2 and clips.loc["c001", "clock_start"] == "45:02"  # (400 - 350) / 25 s
    assert "Away FC pass" in clips.loc["c001", "events"]
    readme = (pack / "README.md").read_text()
    assert "Home FC vs Away FC (m1)" in readme and "reference team: Home FC" in readme


def test_make_pack_needs_meta(adj_match, tmp_path, monkeypatch):
    adj, npz, ev, _ = adj_match
    monkeypatch.setattr(sys, "argv", ["make_pack.py", str(adj), str(npz), str(ev), str(tmp_path / "pack")])
    with pytest.raises(SystemExit):
        make_pack.main()


def write_decisions(path):
    pd.DataFrame([dict(clip_id="c000", decision="R1", other_segments="", reason_code="C1", confidence=3, note=""),
                  dict(clip_id="c001", decision="OTHER", other_segments="TD@400; OP@430", reason_code="C2",
                       confidence=2, note="")]).to_csv(path, index=False)
    return path


def test_apply_decisions_prefix_from_meta(adj_match, tmp_path):
    adj, npz, _, meta = adj_match
    out = tmp_path / "final"
    apply_decisions.main(adj, write_decisions(tmp_path / "dec.csv"), npz, out, meta)
    for k in ("adj", "A", "B", "soft"):
        assert (out / f"home_fc_away_fc_full_{k}.npz").exists()
    L = pd.read_csv(out / "labels_final_25hz.csv")
    assert (L.final[300:320] == 3).all() and (L.final[400:430] == 3).all() and (L.final[430:450] == 2).all()


# ---------------------------------------------------------------- eval_ceiling
@pytest.fixture
def ceiling_dir(tmp_path):
    rng = np.random.default_rng(0)
    T = 600
    y = np.repeat(rng.integers(0, 4, T // 30), 30)
    A, B = y.copy(), y.copy()
    B[rng.random(T) < 0.1] = 2
    A[100:130] = B[300:320] = -1
    for k, lab in (("adj", y), ("A", A), ("B", B)):
        np.savez_compressed(tmp_path / f"pre_{k}.npz", label=lab.astype(np.int8), hz=np.float32(12.5))
    L1 = 500  # predictions cover the labelled stretch only
    (tmp_path / "hazard").mkdir()
    np.savez_compressed(tmp_path / "hazard" / "preds.npz", gt=y[:L1], poss_events=np.zeros(L1),
                        timer_taustar=y[:L1], hazard_time=np.roll(y, 5)[:L1])
    (tmp_path / "deep").mkdir()
    np.savez_compressed(tmp_path / "deep" / "oof.npz", gt=y[:L1], probs=np.eye(4)[y[:L1]], hsmm=y[:L1])
    return tmp_path, y, A, B


def test_eval_ceiling_optional_sources_and_hz_from_npz(ceiling_dir):
    D, y, A, B = ceiling_dir
    rows, ci = eval_ceiling.main([str(D), "--prefix", "pre", "--n_boot", "20"])
    assert set(rows) == {"HUMAN  B vs A  (both coded)", "HUMAN  A vs B  (both coded)"}  # nothing else asked for
    assert set(ci) == {"A (both coded)", "B (both coded)"}
    rows, ci = eval_ceiling.main([str(D), "--prefix", "pre", "--hazard", str(D / "hazard"), "--deep", str(D / "deep"),
                                  "--n_boot", "20", "--out", "x"])
    assert rows[f"{'timer_taustar':24s} vs ADJ (adj frames)"]["macro_f1"] == 1.0
    assert rows[f"{'deep_hsmm':24s} vs ADJ (adj frames)"]["n_frames"] == 500
    assert "hazard_time" in ci["ADJ (adj frames)"]["ci"] and "HUMAN B" in ci["A (both coded)"]["ci"]
    saved = json.loads((D / "x.json").read_text())
    assert set(saved) == {"results", "ci"}


def test_eval_ceiling_hz25_v1_codes_are_unlabelled(ceiling_dir):
    D, y, A, B = ceiling_dir
    y25, A25, B25 = (np.repeat(x, 2) for x in (y, A, B))
    A25[:40], B25[40:80], y25[80:100] = 4, 5, -9  # EX-SP, EX-DEAD, pending
    pd.DataFrame({"Frame": np.arange(len(y25)), "final": y25, "A": A25, "B": B25}).to_csv(
        D / "labels_final_25hz.csv", index=False)
    rows, _ = eval_ceiling.main([str(D), "--prefix", "pre", "--deep", str(D / "deep"), "--hz25", "--n_boot", "20"])
    both = (A25[:1000] >= 0) & (A25[:1000] <= 3) & (B25[:1000] >= 0) & (B25[:1000] <= 3)
    assert rows["HUMAN  B vs A  (both coded)"]["n_frames"] == both.sum()
    assert rows[f"{'deep_hsmm':24s} vs ADJ (adj frames)"]["n_frames"] == 1000 - 20  # pending frames dropped
    assert (D / "eval_ceiling_25hz.json").exists()


def test_eval_ceiling_hz25_v0(ceiling_dir):
    D, y, A, B = ceiling_dir
    pd.DataFrame({"Frame": np.arange(2 * len(y)), "label_adj": np.repeat(y, 2), "label_A": np.repeat(A, 2),
                  "label_B": np.repeat(B, 2)}).to_csv(D / "labels_25hz.csv", index=False)
    rows, _ = eval_ceiling.main([str(D), "--prefix", "pre", "--deep", str(D / "deep"), "--hz25", "--n_boot", "20"])
    assert rows[f"{'deep_hsmm':24s} vs ADJ (adj frames)"]["macro_f1"] == 1.0
    assert rows[f"{'deep_hsmm':24s} vs ADJ (adj frames)"]["n_frames"] == 1000


# ---------------------------------------------------------------- CLI: no hard-coded artifact paths
def test_hazard_cli_requires_npz_out_and_reads_no_deep_oof_by_default():
    from phase import hazard
    with pytest.raises(SystemExit):
        hazard.parser().parse_args(["--events", "e.csv"])
    cfg = vars(hazard.parser().parse_args(["--events", "e.csv", "--npz", "m.npz", "--out", "o"]))
    assert cfg["deep_oof"] is None
