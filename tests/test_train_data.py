"""phase.train.Data: auxiliary possession target from the annotators' labels (review item 8)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase import features as Fe  # noqa: E402
from phase import prep  # noqa: E402
from phase import train  # noqa: E402
from phase.model import loss_fn  # noqa: E402
from synth import fused_csv  # noqa: E402

T = 100  # 12.5 Hz frames
CFG = dict(timer_feat=True, org_block=True, bnd_sigma_s=0.5, soft=False, poss_target="label", poss_input=True)


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    d = prep.load(fused_csv(tmp_path_factory.mktemp("m") / "f.csv", T=2 * T), min_break_s=1, min_period_s=2)
    label = np.full(T, -1)
    label[10:40], label[40:50], label[50:80], label[80:90] = 0, 3, 2, 1  # home, away, away, home
    d["label"] = label.astype(np.int8)
    d["alive"] = np.ones(T, bool)
    d["alive"][45:55] = False
    poss = np.zeros(T)  # input possession disagrees with the labels on 40..80 (home throughout)
    poss[:5] = np.nan
    soft = np.zeros((T, 4), np.float32)
    soft[10:30, 0] = 1.0
    soft[30:40, [0, 3]] = 0.5  # coders split: home attack vs home lost the ball
    d.update(label_soft=soft, label_A=label.astype(np.int8), label_B=label.astype(np.int8))
    return d, Fe.build(d, poss)


def batch(d, views, train_mask=None, **cfg):
    data = train.Data(views, d, {**CFG, **cfg})
    tm = torch.ones(T, dtype=torch.bool) if train_mask is None else torch.from_numpy(train_mask)
    g = torch.zeros(data.glob_dim)
    return data, data.batch([0], T, g, torch.ones_like(g), tm, "cpu")


def test_label_target_is_annotator_possession(match):
    d, views = match
    _, b = batch(d, views)
    lab = d["label"].astype(int)
    home_has = np.isin(lab, (0, 1)).astype(np.float32)
    assert torch.equal(b["poss_tgt"][0], torch.from_numpy(home_has))  # home view
    assert torch.equal(b["poss_tgt"][1][lab >= 0], torch.from_numpy(1 - home_has[lab >= 0]))  # away view
    assert torch.equal(b["poss_mask"][0], torch.from_numpy((lab >= 0).astype(np.float32)))  # dead frames included


def test_label_target_mask_is_training_frames_with_a_label(match):
    d, views = match
    tm = np.ones(T, bool)
    tm[20:60] = False  # test block + buffer
    _, b = batch(d, views, train_mask=tm)
    expect = ((d["label"] >= 0) & tm).astype(np.float32)
    assert torch.equal(b["poss_mask"][0], torch.from_numpy(expect))
    assert torch.equal(b["poss_mask"][1], torch.from_numpy(expect))


def test_soft_target_is_q0_plus_q1(match):
    d, views = match
    _, b = batch(d, views, soft=True)
    q = d["label_soft"]
    assert torch.allclose(b["poss_tgt"][0], torch.from_numpy(q[:, 0] + q[:, 1]))
    assert torch.allclose(b["poss_tgt"][1], torch.from_numpy(q[:, 2] + q[:, 3]))  # mirrored view
    assert b["poss_tgt"][0][35] == 0.5
    assert torch.equal(b["poss_mask"][0], torch.from_numpy((q.sum(1) > 0).astype(np.float32)))


def test_input_target_is_the_old_behaviour(match):
    d, views = match
    _, b = batch(d, views, poss_target="input")
    assert torch.equal(b["poss_tgt"][0], torch.from_numpy(views[0]["poss_ref"]))
    assert torch.equal(b["poss_tgt"][1], torch.from_numpy(views[1]["poss_ref"]))
    assert torch.equal(b["poss_mask"][0], torch.from_numpy(d["alive"].astype(np.float32)))
    with pytest.raises(ValueError, match="poss_target"):
        batch(d, views, poss_target="events")


def test_no_poss_input_drops_poss_ref_from_the_context(match):
    d, views = match
    full, _ = batch(d, views)
    abl, b = batch(d, views, poss_input=False)
    assert "poss_ref" in full.ctx_names and "poss_ref" not in abl.ctx_names
    assert abl.glob_dim == full.glob_dim - 1
    assert "time_since_turnover" in abl.ctx_names


def test_loss_uses_the_target_on_masked_frames_only(match):
    d, views = match
    _, b = batch(d, views)
    n = b["label"].shape
    out = dict(cls=torch.zeros(*n, 4), bnd=torch.zeros(n), poss=torch.zeros(n))
    cfg = dict(gamma=1.0, lam_bnd=0.0, lam_smooth=0.0, lam_poss=1.0, lam_sym=0.0)
    _, parts = loss_fn(out, b, cfg, torch.ones(4))
    assert parts["poss"] == pytest.approx(np.log(2), rel=1e-6)  # logit 0 on every target
    out["poss"] = torch.where(b["poss_mask"] > 0, torch.zeros(n), torch.full(n, 50.0))  # confident on unmasked frames
    assert loss_fn(out, b, cfg, torch.ones(4))[1]["poss"] == pytest.approx(np.log(2), rel=1e-6)


IO = ["--npz", "m.npz", "--out", "run"]


def test_cli_defaults_and_flags_go_into_the_config():
    cfg = vars(train.parser().parse_args(IO))
    assert cfg["poss_target"] == "label" and cfg["poss_input"] is True
    cfg = vars(train.parser().parse_args(IO + ["--poss_target", "input", "--no-poss-input"]))
    assert cfg["poss_target"] == "input" and cfg["poss_input"] is False


@pytest.mark.parametrize("args", [[], ["--npz", "m.npz"], ["--out", "run"]])
def test_npz_and_out_are_required(args):
    with pytest.raises(SystemExit):
        train.parser().parse_args(args)
