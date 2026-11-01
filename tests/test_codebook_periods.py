"""codebook_v1 --meta: frames outside a period are dead ball even if ball_status says alive (half-time blip)."""
import json

import numpy as np
import pandas as pd
import pytest

import codebook_v1 as C
from synth import meta_json

T = 200


def fused(path, alive, cat="Attacking"):
    pd.DataFrame({"ball_status": np.where(alive, "alive", "dead"), "Annotation_Category": np.where(alive, cat, ""),
                  "Annotation_Subcategory": np.where(alive, "Build up", "")}).to_csv(path, index=False)
    return path


@pytest.fixture
def match(tmp_path, monkeypatch):
    alive = np.ones(T, bool)
    alive[80:120] = False  # half-time break 80..119 ...
    alive[100:105] = True  # ... with a 5-frame "alive" blip in it, which neither coder labelled
    a = fused(tmp_path / "a.csv", alive)
    b = fused(tmp_path / "b.csv", alive)
    for p in (a, b):  # coders did not label the blip
        df = pd.read_csv(p)
        df.loc[100:104, ["Annotation_Category", "Annotation_Subcategory"]] = ""
        df.to_csv(p, index=False)
    monkeypatch.setattr(C.R, "detect", lambda alive, d, ev: pd.DataFrame(
        [dict(onset=s, end=e, kind="kickoff", zone="middle", x_att=0.0, team=0)
         for s, e in C.runs(alive)]))
    np.savez(tmp_path / "trk.npz", hz=np.float32(12.5))
    meta = meta_json(tmp_path / "meta.json", period_start={1: 0, 2: 120}, period_end_frame={"1": 79, "2": T - 1})
    return tmp_path, a, b, meta


def test_without_meta_the_blip_becomes_a_clip(match):
    tmp_path, a, b, _ = match
    C.main(["convert", str(a), str(b), "ev.csv", str(tmp_path / "trk.npz"), str(tmp_path / "out")])
    clips = pd.read_csv(tmp_path / "out" / "clips.csv")
    assert clips[["start", "end", "type"]].values.tolist() == [[100, 104, "UNC"]]


def test_meta_masks_alive_frames_outside_periods(match):
    tmp_path, a, b, meta = match
    C.main(["convert", str(a), str(b), "ev.csv", str(tmp_path / "trk.npz"), str(tmp_path / "out"), "--meta", str(meta)])
    assert pd.read_csv(tmp_path / "out" / "clips.csv").empty
    L = pd.read_csv(tmp_path / "out" / "labels_v1_25hz.csv")
    assert (L.alive[100:105] == 0).all() and (L.final[100:105] == C.INACTIVE).all()
    assert (L.final[:80] == 0).all() and (L.final[120:] == 0).all()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["alive_outside_periods_masked"] == 5 and summary["clips"] == 0


def test_validate_with_meta(match):
    tmp_path, a, _, meta = match
    assert C.validate(a, "ev.csv", tmp_path / "trk.npz")["blanks_over_1s"] == []  # 5 frames: a short blank
    assert C.validate(a, "ev.csv", tmp_path / "trk.npz")["warn_blanks_up_to_1s"] == 1
    assert C.validate(a, "ev.csv", tmp_path / "trk.npz", meta)["warn_blanks_up_to_1s"] == 0


def test_meta_without_period_end_warns(match):
    tmp_path, *_ = match
    with pytest.warns(UserWarning, match="period_end_frame"):
        assert C.in_periods(10, meta_json(tmp_path / "m2.json")).all()
