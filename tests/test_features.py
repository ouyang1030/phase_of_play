"""phase.features: 2 s velocities are smoothed per player, not per slot (review item 1)."""
import numpy as np

from phase import features as Fe
from phase import prep
from synth import fused_csv


def test_rolling_never_mixes_two_players_in_one_slot():
    T = 20
    v = np.zeros((T, 2, 11, 2))
    ids = np.full((T, 2, 11), -1)
    v[:10, 0, 0], ids[:10, 0, 0] = 5.0, 3  # player 3 leaves ...
    v[10:, 0, 0], ids[10:, 0, 0] = -5.0, 7  # ... player 7 takes the slot
    out = Fe._roll_per_player(v, ids, win=25)
    assert np.allclose(out[:10, 0, 0], 5) and np.allclose(out[10:, 0, 0], -5)
    assert not np.allclose(Fe._roll(v, 25)[9, 0, 0], 5)  # the old per-slot rolling mixes them


def test_rolling_ignores_frames_where_the_player_is_off():
    v = np.zeros((10, 2, 11, 2))
    ids = np.full((10, 2, 11), -1)
    v[:4, 0, 1], ids[:4, 0, 1] = 2.0, 4
    v[6:, 0, 1], ids[6:, 0, 1] = 2.0, 4  # dropout 4..5 (vel 0 there in the npz)
    out = Fe._roll_per_player(v, ids, win=25)
    assert np.allclose(out[:4, 0, 1], 2) and np.allclose(out[6:, 0, 1], 2) and np.allclose(out[4:6, 0, 1], 0)


def test_build_runs_on_prep_output_with_gaps(tmp_path):
    p = fused_csv(tmp_path / "f.csv", T=200, n_cols=12, extra={(0, 11): (100, 200), (1, 11): (0, 0)},
                  drop={(0, 5): [(100, 200)], (1, 2): [(40, 60)]})
    d = prep.load(p, min_break_s=1, min_period_s=2, min_gk_s=1)
    views = Fe.build(d, np.zeros(len(d["label"])))
    for v in views.values():
        assert v["node"].shape == (100, 23, len(Fe.NODE_FEATS))
        assert np.isfinite(v["node"]).all() and np.isfinite(v["glob"]).all()
    assert views[0]["node_mask"][20:30, 11:22].sum(1).tolist() == [10] * 10  # away column 2 off for 0.8 s
