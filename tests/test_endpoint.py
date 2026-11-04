"""phase.endpoint: likelihoods against brute force, model output, and learning a planted signal."""
import numpy as np
import pytest
from scipy.stats import norm

torch = pytest.importorskip("torch")

from phase import endpoint as EP  # noqa: E402

CFG = dict(d=16, h=8, dropout=0.0, lr=1e-2, weight_decay=0.0, epochs=60, batch=8, ensemble=1)


def random_logp(rng, lengths, L):
    logp = np.full((len(lengths), L + 1), -np.inf)
    for i, W in enumerate(lengths):
        z = rng.normal(size=W + 1)
        p = np.exp(z) / np.exp(z).sum()
        logp[i, :W], logp[i, L] = np.log(p[:W]), np.log(p[W])
    return logp


def probs_row(logp, W, L):
    return np.r_[np.exp(logp[:W]), np.exp(logp[L])]  # support 0..W (W = fills the window)


def test_range_loglik_brute_force():
    rng = np.random.default_rng(0)
    lengths, L = np.array([5, 9, 3]), 9
    logp = random_logp(rng, lengths, L)
    lo, hi = np.array([2, 0, 3]), np.array([4, 9, 3])  # interval, censored to the end, overflow only
    got = EP.range_loglik(torch.tensor(logp), torch.tensor(lengths), torch.tensor(lo), torch.tensor(hi)).numpy()
    want = [np.log(probs_row(logp[i], W, L)[lo[i]:hi[i] + 1].sum()) for i, W in enumerate(lengths)]
    assert np.allclose(got, want)


def test_noise_loglik_brute_force():
    rng = np.random.default_rng(1)
    lengths, L = np.array([6, 10]), 10
    logp = random_logp(rng, lengths, L)
    lo = np.array([[2.0, 0.0], [4.0, 7.0]])
    hi = np.array([[2.0, 6.0], [4.0, 10.0]])  # exact | open to the left (lo = 0) and right (hi = W)
    use = np.array([[True, True], [True, False]])
    b, s, eps = np.array([1.5, -1.5]), np.array([2.0, 3.0]), 0.01
    got = EP.noise_loglik(torch.tensor(logp), torch.tensor(lengths), torch.tensor(lo), torch.tensor(hi),
                          torch.tensor(use), torch.tensor(b), torch.tensor(s), eps).numpy()
    for i, W in enumerate(lengths):
        p = probs_row(logp[i], W, L)
        tot = 0.0
        for k in range(W + 1):
            q = 1.0
            for h in range(2):
                if not use[i, h]:
                    continue
                up = 1.0 if hi[i, h] >= W else norm.cdf((hi[i, h] + 0.5 - k - b[h]) / s[h])
                dn = 0.0 if lo[i, h] <= 0 else norm.cdf((lo[i, h] - 0.5 - k - b[h]) / s[h])
                q *= (1 - eps) * (up - dn) + eps * (hi[i, h] - lo[i, h] + 1) / (W + 1)
            tot += p[k] * q
        assert got[i] == pytest.approx(np.log(tot), rel=1e-9)


def test_model_output_is_a_distribution_over_the_window():
    m = EP.EndPointer(4, 8, 4, 0.0)
    x = torch.randn(3, 7, 4)
    lengths = torch.tensor([7, 3, 1])
    logp = m(x, lengths)
    assert logp.shape == (3, 8)
    p = logp.exp()
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-6)
    assert (p[1, 3:7] == 0).all() and (p[2, 1:7] == 0).all()  # padded frames
    assert EP.Noise(2, 12.5).b().sum().abs() < 1e-6


def planted(rng, n=60):
    """Episodes whose feature 0 switches from 0 to 1 at the end of the transition e (frames)."""
    X, e = [], []
    for _ in range(n):
        W = int(rng.integers(15, 40))
        ei = int(rng.integers(0, W))
        x = rng.normal(scale=0.3, size=(W, 3)).astype(np.float32)
        x[ei:, 0] += 1.0
        X.append(x)
        e.append(ei)
    return X, np.array(e)


def test_fit_predict_learns_a_planted_end():
    rng = np.random.default_rng(0)
    X, e = planted(rng)
    tgt = dict(mode="range", lo=e, hi=e)
    tr, te = np.arange(40), np.arange(40, 60)
    probs, info = EP.fit_predict(X, tgt, tr, te, CFG, hz=12.5, seed=0)
    med = np.array([int(np.searchsorted(np.cumsum(p), 0.5)) for p in probs])
    err = np.abs(med - e[te])
    assert np.mean(err <= 1) >= 0.75 and np.median(err) <= 1
    assert info["n_params"] > 0 and len(info["losses"]) == 1


def test_noise_model_learns_coder_offsets():
    rng = np.random.default_rng(1)
    X, e = planted(rng, 60)
    eA = np.minimum(e + 3, [len(x) for x in X])  # coder A ends 3 frames later, coder B 3 frames earlier
    eB = np.maximum(e - 3, 0)
    tgt = dict(mode="noise", lo=np.c_[eA, eB].astype(float), hi=np.c_[eA, eB].astype(float),
               use=np.ones((60, 2), bool))
    probs, info = EP.fit_predict(X, tgt, np.arange(60), np.arange(60), dict(CFG, epochs=60), hz=1.0, seed=0)
    bA, bB = info["coder_offset_s"][0]
    assert bA > 1.5 and bB < -1.5  # offsets in s at hz = 1: about +3 / -3 frames
