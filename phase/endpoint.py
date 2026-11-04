"""Torch part of the end-point model (phase.duration): encoder, likelihoods, training. Imported only when the
neural model is trained, so phase.duration --no-model and the baselines run without torch.

EndPointer: per-frame MLP (F -> d -> d) -> one-layer bidirectional GRU (h per direction) -> one logit per window
frame ("the transition ends here") and one logit "the transition fills the window"; a softmax over the W + 1
outcomes is p(E = k). About 27k parameters with the defaults (F ~ 60, d = 64, h = 32).

Likelihoods (per episode):
  range_loglik  log P(E in [lo, hi]); hi >= W includes the overflow outcome (targets label / interval)
  noise_loglik  log sum_k p(E* = k) prod_h q_h([lo_h, hi_h] | k), q_h = (1 - eps) P(lo_h <= k + b_h + e <= hi_h)
                + eps (hi_h - lo_h + 1) / (W + 1), e ~ N(0, s_h^2) discretised to frames; the open ends lo_h = 0 and
                hi_h = W integrate to -inf / +inf. Offsets b_h sum to zero (E* = the average coder), s_h >= 1 frame.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EndPointer(nn.Module):
    def __init__(self, f_in: int, d: int = 64, h: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU())
        self.rnn = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.step = nn.Linear(2 * h, 1)
        self.over = nn.Linear(2 * h, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x (B, L, F) zero-padded, lengths (B,) >= 1 -> log p (B, L + 1): column k < length = the transition ends
        at window frame k, column L = it fills the window; padded frames get -inf."""
        B, L, _ = x.shape
        z = self.enc(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        out = self.drop(out)
        s = self.step(out).squeeze(-1)
        valid = torch.arange(L, device=x.device)[None] < lengths.to(x.device)[:, None]
        s = s.masked_fill(~valid, float("-inf"))
        last = out[torch.arange(B, device=x.device), lengths.to(x.device) - 1]
        return torch.log_softmax(torch.cat([s, self.over(last)], 1), 1)


class Noise(nn.Module):
    """Coder offsets b_h (frames, summing to zero) and scales s_h (frames) of the annotator-noise likelihood."""

    def __init__(self, n_coders: int, init_s: float):
        super().__init__()
        self.raw_b = nn.Parameter(torch.zeros(n_coders))
        self.raw_s = nn.Parameter(torch.full((n_coders,), float(np.log(np.expm1(max(init_s - 1.0, 1e-3))))))

    def b(self) -> torch.Tensor:
        return self.raw_b - self.raw_b.mean() if len(self.raw_b) > 1 else self.raw_b * 0

    def s(self) -> torch.Tensor:
        return F.softplus(self.raw_s) + 1.0


def range_loglik(logp: torch.Tensor, lengths: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    L = logp.shape[1] - 1
    k = torch.arange(L, device=logp.device)[None]
    steps = (k >= lo[:, None]) & (k <= hi[:, None])
    mask = torch.cat([steps, (hi >= lengths)[:, None]], 1)
    return torch.logsumexp(logp.masked_fill(~mask, float("-inf")), 1)


def noise_loglik(logp, lengths, lo, hi, use, b, s, eps: float = 0.01) -> torch.Tensor:
    """lo, hi (B, H) float, use (B, H) bool, b, s (H,)."""
    B, L1 = logp.shape
    L = L1 - 1
    Wf = lengths.to(logp.dtype)[:, None]
    k = torch.arange(L1, device=logp.device, dtype=logp.dtype)[None].repeat(B, 1)
    k[:, L] = Wf[:, 0]  # the overflow outcome sits at k = W
    total = logp
    for h in range(lo.shape[1]):
        mu = k + b[h]
        lo_h, hi_h = lo[:, h:h + 1], hi[:, h:h + 1]
        up = torch.where(hi_h >= Wf, torch.ones_like(mu), torch.special.ndtr((hi_h + 0.5 - mu) / s[h]))
        dn = torch.where(lo_h <= 0, torch.zeros_like(mu), torch.special.ndtr((lo_h - 0.5 - mu) / s[h]))
        q = (1 - eps) * (up - dn).clamp(min=0.0) + eps * (hi_h - lo_h + 1) / (Wf + 1)
        total = total + torch.where(use[:, h:h + 1], torch.log(q), torch.zeros_like(q))
    return torch.logsumexp(total, 1)


def _batch(X, idx, mu, sd):
    lens = [len(X[i]) for i in idx]
    xb = np.zeros((len(idx), max(lens), X[idx[0]].shape[1]), np.float32)
    for j, i in enumerate(idx):
        xb[j, :lens[j]] = (X[i] - mu) / sd
    return torch.from_numpy(xb), torch.tensor(lens, dtype=torch.long)


def _loglik(logp, lens, tgt, idx, noise):
    if tgt["mode"] == "range":
        lo = torch.as_tensor(tgt["lo"][idx], dtype=torch.long)
        hi = torch.as_tensor(tgt["hi"][idx], dtype=torch.long)
        return range_loglik(logp, lens, lo, hi)
    lo = torch.as_tensor(tgt["lo"][idx], dtype=logp.dtype)
    hi = torch.as_tensor(tgt["hi"][idx], dtype=logp.dtype)
    use = torch.as_tensor(tgt["use"][idx], dtype=torch.bool)
    return noise_loglik(logp, lens, lo, hi, use, noise.b(), noise.s())


def fit_predict(X: list, tgt: dict, train_idx, pred_idx, cfg: dict, hz: float, seed: int = 0):
    """Train cfg['ensemble'] models on the episodes train_idx and return, for pred_idx, the ensemble-mean
    p(E = k) (arrays of length W_i + 1) and an info dict (losses, coder offsets / scales in s).

    X: per episode (W_i, F) float32 features; tgt: dict(mode='range', lo, hi) with (n,) arrays, or
    dict(mode='noise', lo, hi, use) with (n, H) arrays (phase.duration.targets)."""
    train_idx, pred_idx = np.asarray(train_idx), np.asarray(pred_idx)
    Xtr = np.concatenate([X[i] for i in train_idx])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
    mu = mu.astype(np.float32)
    probs = [np.zeros(len(X[i]) + 1) for i in pred_idx]
    info = dict(n_train=int(len(train_idx)), losses=[], coder_offset_s=[], coder_scale_s=[])
    bs = int(cfg["batch"])
    for member in range(int(cfg["ensemble"])):
        torch.manual_seed(seed + member)
        rng = np.random.default_rng(seed + member)
        model = EndPointer(X[train_idx[0]].shape[1], cfg["d"], cfg["h"], cfg["dropout"])
        noise = Noise(tgt["lo"].shape[1], init_s=hz) if tgt["mode"] == "noise" else None
        groups = [dict(params=model.parameters(), weight_decay=cfg["weight_decay"])]
        if noise is not None:
            groups.append(dict(params=noise.parameters(), weight_decay=0.0))
        opt = torch.optim.AdamW(groups, lr=cfg["lr"])
        for _ in range(int(cfg["epochs"])):
            model.train()
            perm = rng.permutation(train_idx)
            tot = 0.0
            for j in range(0, len(perm), bs):
                idx = perm[j:j + bs]
                xb, lens = _batch(X, idx, mu, sd)
                loss = -_loglik(model(xb, lens), lens, tgt, idx, noise).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
        info["losses"].append(tot / len(perm))
        if noise is not None:
            info["coder_offset_s"].append((noise.b().detach().numpy() / hz).round(3).tolist())
            info["coder_scale_s"].append((noise.s().detach().numpy() / hz).round(3).tolist())
        model.eval()
        with torch.no_grad():
            for j in range(0, len(pred_idx), 64):
                idx = pred_idx[j:j + 64]
                xb, lens = _batch(X, idx, mu, sd)
                p = model(xb, lens).exp().numpy()
                for r, i in enumerate(idx):
                    W = len(X[i])
                    probs[j + r] += np.r_[p[r, :W], p[r, -1]] / cfg["ensemble"]
    info["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return probs, info
