"""phase.decompose: 2 x 2 error decomposition (RQ1)."""
import json

import numpy as np

from phase import decompose as D
from phase import episodes as P

HZ = 12.5


def match(seed=0, T=3000):
    rng = np.random.default_rng(seed)
    runs, n = [], 0
    while n < T:
        k = int(rng.integers(60, 300))
        runs.append(np.full(k, float(len(runs) % 2)))
        n += k
    poss = np.concatenate(runs)[:T]
    alive = np.ones(T, bool)
    period = np.ones(T, int)
    eps = P.episodes(poss, alive, period, HZ, 30)
    E_true = np.minimum(rng.integers(0, 80, len(eps)), P.lengths(eps))
    label = P.reconstruct(poss, eps, E_true)
    return poss, alive, period, label


def test_cells_separate_possession_and_length_errors():
    poss, alive, period, label = match()
    lag = np.r_[np.full(5, poss[0]), poss[:-5]]  # the "event" possession switches 5 frames late
    sources = {"labels": (P.label_possession(label), P.episodes(P.label_possession(label), alive, period, HZ, 30)),
               "events": (lag, P.episodes(lag, alive, period, HZ, 30))}
    L0, L1, preds, taus = D.cells(label, sources, HZ, folds=3, buffer_frames=int(5 * HZ))
    from phase import evaluate as E
    res = {k: E.evaluate(label[L0:L1], p, HZ) for k, p in preds.items()}
    assert res["labels | oracle"]["macro_f1"] == 1.0  # grammar-conforming labels: nothing lost
    assert res["events | oracle"]["macro_f1"] < 1.0  # possession errors alone
    assert res["labels | timer"]["macro_f1"] < 1.0  # length errors alone (lengths vary, the timer does not)
    dec = D.decomposition(res, list(sources))
    assert dec["grammar_loss"] == 0.0 and dec["possession_loss"] > 0 and dec["length_loss_timer"] > 0
    assert len(taus["labels"]) == 3


def test_cli_without_events(tmp_path):
    poss, alive, period, label = match(1)
    npz = tmp_path / "m.npz"
    np.savez(npz, label=label.astype(np.int8), alive=alive, period=period.astype(np.int8), hz=np.float32(HZ))
    s = D.run(vars(D.parser().parse_args(["--npz", str(npz), "--out", str(tmp_path / "o"), "--n_boot", "50",
                                          "--folds", "3"])))
    assert set(s["results"]) == {"labels | oracle", "labels | timer"}
    assert s["results"]["labels | oracle"]["macro_f1"] == 1.0
    assert "labels | oracle - labels | timer" in s["ci"]["diff"]
    assert json.loads((tmp_path / "o" / "decompose.json").read_text())["decomposition"]["grammar_loss"] == 0.0
