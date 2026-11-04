"""tools/adj/superhuman.py: L1 / L2 / L3 comparisons of a model with human coders."""
import json

import numpy as np
import pytest

import superhuman as SH
from phase import episodes as P

HZ = 12.5


@pytest.fixture(scope="module")
def adj(tmp_path_factory):
    """Coders A, B, C = a true length per possession + independent +-15 frame noise; runs: 'good' (+-1 frame),
    'copyA' (coder A itself). About 120 possessions and 80 short stoppages (the bootstrap units of the frame
    metrics are the stretches between stoppages)."""
    tmp = tmp_path_factory.mktemp("sh")
    rng = np.random.default_rng(0)
    T = 20000
    runs, n = [], 0
    while n < T:
        k = int(rng.integers(100, 220))
        runs.append(np.full(k, float(len(runs) % 2)))
        n += k
    poss = np.concatenate(runs)[:T]
    alive, period = np.ones(T, bool), np.ones(T, np.int8)
    for s0 in rng.choice(np.arange(50, T - 50), 80, replace=False):
        alive[s0:s0 + 12] = False
    eps = P.episodes(poss, alive, period, HZ, 30)
    W = P.lengths(eps)
    E_true = rng.integers(15, 70, len(eps))

    def coder(noise):
        y = P.reconstruct(poss, eps, np.clip(E_true + rng.integers(-noise, noise + 1, len(eps)), 0, W))
        return np.where(alive, y, -1)

    A, B, C, good = coder(15), coder(15), coder(15), coder(1)
    ADJ = np.where(A == B, A, -1)
    np.savez(tmp / "m_adj.npz", label=ADJ.astype(np.int8), alive=alive, period=period, hz=np.float32(HZ))
    for k, y in (("A", A), ("B", B)):
        np.savez(tmp / f"m_{k}.npz", label=y.astype(np.int8))
    np.savez(tmp / "C.npz", label=C.astype(np.int8))
    good = np.where(good >= 0, good, 0)  # models label every frame
    for name, y in (("good", good), ("copyA", np.where(A >= 0, A, 0))):
        (tmp / name).mkdir()
        np.savez(tmp / name / "oof.npz", pred=y, gt=ADJ)
    return tmp


def test_levels_and_verdicts(adj):
    out = SH.main([str(adj), "--prefix", "m", f"good={adj / 'good'}", f"copyA={adj / 'copyA'}",
                   "--l2", f"A={adj / 'copyA'}", f"B={adj / 'good'}", "--third", str(adj / "C.npz"),
                   "--n_boot", "300"])
    g = out["L1"]["good"]
    for comp in ("vs B (competitor A)", "vs A (competitor B)"):
        assert g[comp]["verdict"] == {"macro_f1": "exceeds", "kappa": "exceeds", "length_mae": "exceeds"}
        assert g[comp]["lengths"]["mae_model_s"] < g[comp]["lengths"]["mae_human_s"]
    same = out["L1"]["copyA"]["vs B (competitor A)"]  # the model is coder A: no difference to coder A
    assert same["frames"]["macro_f1"]["diff"] == 0 and same["verdict"]["macro_f1"] == "no difference"
    lo, hi = g["vs B (competitor A)"]["frames"]["macro_f1"]["lo"], g["vs B (competitor A)"]["frames"]["macro_f1"]["hi"]
    assert hi - lo > 1e-4  # a real interval: many stretches between stoppages
    assert same["verdict"]["length_mae"] == "no difference"
    assert out["L2"]["trained on A"]["vs B (competitor A)"]["verdict"]["macro_f1"] == "no difference"
    assert out["L2"]["trained on B"]["vs A (competitor B)"]["verdict"]["macro_f1"] == "exceeds"
    assert out["L3"]["good"]["vs C (competitor A)"]["verdict"]["length_mae"] == "exceeds"
    assert out["episodes_from"] == "adjudicated labels" and out["n_episodes"] > 100
    assert json.loads((adj / "superhuman.json").read_text())["L1"]["good"]["vs A (competitor B)"]["n_frames"] > 0


def test_l2_needs_a_known_coder(adj):
    with pytest.raises(ValueError, match="A or B"):
        SH.main([str(adj), "--prefix", "m", f"good={adj / 'good'}", "--l2", f"X={adj / 'good'}", "--n_boot", "10"])


def test_eval_ceiling_reads_duration_runs(adj):
    import eval_ceiling as EC
    preds = EC.load_preds(EC.parse_args([str(adj), "--prefix", "m", f"good={adj / 'good'}"]))
    assert list(preds) == ["good"]  # pred of a phase.duration run, no decoder suffix
