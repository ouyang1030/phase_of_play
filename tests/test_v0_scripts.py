"""v0 adjudication scripts take the npz prefix (--prefix) instead of a hard-coded ajax_psv_full."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "home_away_full"


def run(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "adj" / script), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


@pytest.fixture
def adj(tmp_path):
    rng = np.random.default_rng(0)
    T = 400
    y = np.repeat(rng.integers(0, 4, T // 40), 40)
    A, B = y.copy(), y.copy()
    B[rng.random(T) < 0.05] = 2
    A[50:60] = -1
    for k, lab in (("adj", y), ("A", A), ("B", B)):
        np.savez_compressed(tmp_path / f"{PREFIX}_{k}.npz", label=lab.astype(np.int8), hz=np.float32(12.5),
                            alive=np.ones(T, bool))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    probs = np.full((T, 4), 0.05, np.float32)
    probs[np.arange(T), y] = 0.85
    bnd = np.zeros(T, np.float32)
    bnd[np.flatnonzero(np.diff(y)) + 1] = 0.9
    np.savez_compressed(run_dir / "oof.npz", gt=y, probs=probs, bnd=bnd, hsmm=y, alive=np.ones(T, bool),
                        poss_home=np.zeros(T), hz=np.float32(12.5))
    (run_dir / "results.json").write_text(json.dumps(dict(config=dict(buffer_s=2, switch_penalty=2.0), folds=[
        dict(fold=0, block=[0, 200]), dict(fold=1, block=[200, 400])])))
    return tmp_path, run_dir, y, A, B


@pytest.mark.parametrize("script, args", [("build_soft.py", ["D"]), ("check_s7.py", ["D"]),
                                          ("bnd_diag.py", ["D", "R"]), ("decode_ed.py", ["D", "R"]),
                                          ("build_labels.py", ["a.csv", "b.csv", "out"])])
def test_prefix_is_required(script, args):
    r = run(script, *args)
    assert r.returncode == 2 and "--prefix" in r.stderr


def test_build_soft_uses_the_prefix(adj):
    D, _, y, A, B = adj
    r = run("build_soft.py", D, "--prefix", PREFIX)
    assert r.returncode == 0, r.stderr
    z = np.load(D / f"{PREFIX}_soft.npz")
    both = (A >= 0) & (B >= 0)
    assert np.allclose(z["label_soft"][both].sum(1), 1) and (z["label_soft"][~both] == 0).all()
    assert np.array_equal(z["label_A"], np.where(both, A, -1))


def test_bnd_diag_uses_the_prefix(adj):
    D, run_dir, *_ = adj
    r = run("bnd_diag.py", D, run_dir, "--prefix", PREFIX)
    assert r.returncode == 0, r.stderr
    assert run("bnd_diag.py", D, run_dir, "--prefix", "ajax_psv_full").returncode != 0  # no such files here


def test_decode_ed_uses_the_prefix(adj):
    D, run_dir, *_ = adj
    r = run("decode_ed.py", D, run_dir, "--prefix", PREFIX)
    assert r.returncode == 0, r.stderr
    assert {"edhsmm", "edhsmm_hard"} <= set(np.load(run_dir / "decoded.npz").files)
