"""Subcategories through codebook v1 and adjudication (review item 9)."""
import json

import numpy as np
import pandas as pd
import pytest

import apply_decisions
import codebook_v1 as C
from synth import adj_dir, meta_json, tracking

U, SP, IN = C.UNCODED, C.SP, C.INACTIVE
BUILD, PROG, COUNTER, MID, LOW, PRESS = 0, 1, 3, 6, 7, 8  # phase.prep.SUBCATS


def strings(*parts):
    """[(text, n), ...] -> pd.Series of n copies of each text."""
    return pd.Series([t for t, n in parts for _ in range(n)], dtype=object)


# ---------------------------------------------------------------- codebook_v1: clean / to_v1 / consistent_sub
def test_sub_category_table_matches_subcats():
    assert C.SUB_CATEGORY.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    assert [C.SUBCATS[i] for i in (BUILD, COUNTER, MID, PRESS)] == ["Build up", "Counterattack", "Mid Block",
                                                                     "Counterpressing"]


def test_clean_reads_subcategories_splits_overlaps_and_bridges():
    cat = strings(("Attacking", 10), ("", 3), ("Attacking", 5), ("Attacking; Defending", 10), ("Defending", 4))
    sub = strings(("Build up", 10), ("", 3), ("Progression", 5), ("Creation; Low Block", 10), ("Nonsense", 4))
    lab, n_overlap, sc = C.clean(cat, sub)
    assert n_overlap == 10
    assert lab.tolist() == [0] * 23 + [2] * 9
    assert sc[:10].tolist() == [BUILD] * 10
    assert sc[10:13].tolist() == [BUILD] * 3  # 3-frame gap bridged with the label and subcategory before it
    assert sc[13:18].tolist() == [PROG] * 5
    assert sc[18:23].tolist() == [2] * 5 and sc[23:28].tolist() == [LOW] * 5  # overlap split at its midpoint
    assert sc[28:].tolist() == [-1] * 4  # unknown name


def test_clean_without_subcategories_is_unchanged():
    cat = strings(("Attacking", 10), ("", 3), ("Defending", 5))
    lab, n, sc = C.clean(cat)
    assert lab.tolist() == [0] * 13 + [2] * 5 and n == 0 and (sc == -1).all()


def restarts(*rows):
    return pd.DataFrame([dict(onset=s, end=e, kind=k, zone="middle", x_att=0.0) for s, e, k in rows])


def test_to_v1_copies_subcategories_with_every_fill():
    T = 200
    alive = np.ones(T, bool)
    alive[100:120] = False
    lab = np.full(T, U)
    sub = np.full(T, -1)
    lab[0:40], sub[0:40] = 0, BUILD
    lab[50:90], sub[50:90] = 3, PRESS  # gap [40, 50) between two labels -> split at its midpoint
    # [90, 100): label stopped before the ball went dead -> dead_lag
    lab[130:160], sub[130:160] = 2, MID  # [120, 130) after a throw-in -> restart_lag
    # [160, 170): blank after a corner -> SET_PIECE
    lab[170:200], sub[170:200] = 1, COUNTER
    rs = restarts((120, 160, "throw_in"), (160, 200, "corner"))
    lab[160:170] = U
    v1, src, s1 = C.to_v1(lab, alive, rs, sub)
    assert s1[40:45].tolist() == [BUILD] * 5 and s1[45:50].tolist() == [PRESS] * 5 and set(src[40:50]) == {"gap_mid"}
    assert (v1[90:100] == 3).all() and (s1[90:100] == PRESS).all() and set(src[90:100]) == {"dead_lag"}
    assert (v1[100:120] == IN).all() and (s1[100:120] == -1).all()
    assert (v1[120:130] == 2).all() and (s1[120:130] == MID).all() and set(src[120:130]) == {"restart_lag"}
    assert (v1[160:170] == SP).all() and (s1[160:170] == -1).all()
    v1_old, src_old, s_none = C.to_v1(lab, alive, rs)  # without subcategories: labels unchanged, all -1
    assert np.array_equal(v1_old, v1) and np.array_equal(src_old, src) and (s_none == -1).all()


def test_consistent_sub_drops_subcategories_of_another_moment():
    lab = np.array([0, 0, 1, 2, 3, SP, IN, U])
    sub = np.array([BUILD, MID, COUNTER, LOW, PRESS, BUILD, MID, PROG])
    out, n = C.consistent_sub(lab, sub)
    assert out.tolist() == [BUILD, -1, COUNTER, LOW, PRESS, -1, -1, -1] and n == 4


def test_kappa_is_none_when_undefined():
    x = np.array([6, 6, 6])
    assert C.kappa(x, x, [5, 6, 7]) == dict(n=3, kappa=None)  # one code throughout: chance agreement 1
    assert C.kappa(x, x, [0, 1]) == dict(n=0, kappa=None)
    assert C.kappa(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), [0, 1]) == dict(n=4, kappa=1.0)


def test_irr_subcategory():
    a = np.array([0, 0, 0, 0, 2, 2, 2, 2, 3])
    b = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0])
    sa = np.array([0, 0, 1, 1, 5, 6, 6, 7, 8])
    sb = np.array([0, 0, 1, 1, 6, 6, 6, 7, 0])
    alive = np.ones(len(a), bool)
    irr = C.irr_subcategory(a, b, sa, sb, alive)
    assert irr["alive, frames where both coded a subcategory"]["n"] == 9
    assert irr["alive, both coded the same moment and a subcategory"]["n"] == 8  # TD vs IP left out
    by = irr["by moment (both coded it)"]
    assert by["IP"] == dict(n=4, kappa=1.0) and by["OP"]["n"] == 4 and by["OP"]["kappa"] < 1
    assert by["TA"] == dict(n=0, kappa=None)


def fused(path, cat, sub, alive):
    pd.DataFrame({"ball_status": np.where(alive, "alive", "dead"), "Annotation_Category": cat,
                  "Annotation_Subcategory": sub}).to_csv(path, index=False)
    return path


def test_convert_writes_sub_columns_and_sub_kappa(tmp_path, monkeypatch):
    T = 100
    alive = np.ones(T, bool)
    alive[60:70] = False
    cat = strings(("Attacking", 30), ("Attacking Transition", 30), ("Inactive", 10), ("Defending", 30))
    sub_a = strings(("Build up", 30), ("Counterpressing", 30), ("", 10), ("Mid Block", 30))
    sub_b = strings(("Build up", 15), ("Progression", 15), ("Counterpressing", 30), ("", 10), ("Counterattack", 30))
    monkeypatch.setattr(C.R, "detect", lambda alive, d, ev: restarts((70, 100, "throw_in")))
    np.savez(tmp_path / "trk.npz", hz=np.float32(12.5))
    C.convert(fused(tmp_path / "a.csv", cat, sub_a, alive), fused(tmp_path / "b.csv", cat, sub_b, alive), "ev.csv",
              tmp_path / "trk.npz", tmp_path / "out")
    L = pd.read_csv(tmp_path / "out" / "labels_v1_25hz.csv")
    assert L.subA.tolist() == [BUILD] * 30 + [PRESS] * 30 + [-1] * 10 + [MID] * 30
    assert L.subB.tolist() == [BUILD] * 15 + [PROG] * 15 + [PRESS] * 30 + [-1] * 10 + [-1] * 30  # TA sub on OP
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["conversion"]["B"]["subcategory_of_another_moment_frames"] == 30
    assert summary["conversion"]["B"]["moment_without_subcategory_s"] == round(30 / 25, 1)
    irr = summary["irr_subcategory_pre_adjudication"]
    assert irr["alive, frames where both coded a subcategory"]["n"] == 60
    assert irr["by moment (both coded it)"]["TD"] == dict(n=30, kappa=None)  # both used Counterpressing only


# ---------------------------------------------------------------- apply_decisions
@pytest.fixture
def adj_match(tmp_path):
    d = tracking(400, [(1, 0, 400)])
    d["sub"] = np.full(400, 9, np.int8)  # tracking npz subcategories must not leak into the label npz files
    np.savez_compressed(tmp_path / "trk.npz", **d)
    pd.DataFrame([dict(clip_id="c000", decision="R1", other_segments="", reason_code="C1", confidence=3, note=""),
                  dict(clip_id="c001", decision="OTHER", other_segments="TD@400; OP@430", reason_code="C2",
                       confidence=2, note="")]).to_csv(tmp_path / "dec.csv", index=False)
    return tmp_path, meta_json(tmp_path / "meta.json")


def run(tmp_path, meta, adj):
    out = tmp_path / "final"
    apply_decisions.main(adj, tmp_path / "dec.csv", tmp_path / "trk.npz", out, meta)
    return out, pd.read_csv(out / "labels_final_25hz.csv")


def test_final_sub_only_where_both_coders_agree_with_the_final_label(adj_match):
    tmp_path, meta = adj_match
    out, L = run(tmp_path, meta, adj_dir(tmp_path / "adj"))
    sub = L["sub"].to_numpy()
    expect = np.full(800, -1)
    expect[0:150] = BUILD  # [150, 300): A Progression, B Build up
    # [300, 320): final TD (R1 = A) but B coded IP;  [400, 450): final differs from one coder
    expect[320:400] = PRESS
    expect[450:600] = MID  # [600, 650) dead; [650, 800): A Mid Block, B Low Block
    assert np.array_equal(sub, expect)
    assert (L.sub_name[0:150] == "Build up").all() and L.sub_name.isna()[150:300].all()

    adj, a = np.load(out / "ajax_psv_full_adj.npz"), np.load(out / "ajax_psv_full_A.npz")
    assert np.array_equal(adj["sub"], expect[::2].astype(np.int8))
    assert np.array_equal(np.load(out / "ajax_psv_full_soft.npz")["sub"], expect[::2].astype(np.int8))
    sub_a = pd.read_csv(tmp_path / "adj" / "labels_v1_25hz.csv").subA.to_numpy()[::2]
    assert np.array_equal(a["sub"], np.where(a["label"] >= 0, sub_a, -1))
    for k in ("adj", "A", "B", "soft"):  # every subcategory belongs to its frame's moment
        z = np.load(out / f"ajax_psv_full_{k}.npz")
        s = z["sub"]
        assert (C.SUB_CATEGORY[s[s >= 0]] == z["label"][s >= 0]).all()
    assert json.loads((out / "qc.json").read_text())["checks"]["moment_with_subcategory"] == round(380 / 750, 3)


def test_labels_and_moments_unchanged_by_subcategories(adj_match):
    tmp_path, meta = adj_match
    with pytest.warns(UserWarning, match="no subA / subB"):
        out0, L0 = run(tmp_path, meta, adj_dir(tmp_path / "adj0", sub=False))
    (tmp_path / "final").rename(tmp_path / "final0")
    out1, L1 = run(tmp_path, meta, adj_dir(tmp_path / "adj1"))
    assert (L0["sub"] == -1).all()
    for c in ("final", "src", "clip_id", "A", "B"):
        assert L0[c].fillna("").tolist() == L1[c].fillna("").tolist()
    for k in ("adj", "A", "B", "soft"):
        z0, z1 = np.load(tmp_path / "final0" / f"ajax_psv_full_{k}.npz"), np.load(out1 / f"ajax_psv_full_{k}.npz")
        assert set(z0.files) == set(z1.files)
        assert all(np.array_equal(z0[f], z1[f]) for f in z0.files if f != "sub")
        assert (z0["sub"] == -1).all()


def test_subcategory_of_another_moment_raises(adj_match):
    tmp_path, meta = adj_match
    adj = adj_dir(tmp_path / "adj")
    L = pd.read_csv(adj / "labels_v1_25hz.csv")
    L.loc[10, "subA"] = MID  # Mid Block on an IP frame
    L.to_csv(adj / "labels_v1_25hz.csv", index=False)
    with pytest.raises(AssertionError, match="subA: .* first at 10"):
        run(tmp_path, meta, adj)


def test_check_sub():
    C_ = apply_decisions.check_sub
    C_(np.array([0, 1, 2, 3, 4, 5, -1]), np.array([2, 4, 5, 9, -1, -1, -1]), "ok")
    for lab, sub in (([4], [0]), ([2], [8]), ([-1], [0]), ([3], [10])):
        with pytest.raises(AssertionError):
            C_(np.array(lab), np.array(sub), "bad")
