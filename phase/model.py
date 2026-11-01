"""Proposed model (STUDY1_PLAN §7): heterogeneous graph attention -> temporal encoder -> heads.

Nodes per frame: 11 ref players, 11 opp players, 1 ball (23).
Edge types (dense masks, rebuilt per frame from positions):
  intra_ref  ref <-> ref           (own-team structure)
  intra_opp  opp <-> opp           (opponent structure, e.g. defensive organization)
  press      ref <-> k nearest opp (pressure / marking relations, both directions)
  ball       players <-> ball
  mark       ref <-> opp 1:1 Hungarian outfield assignment (optional, features.mark_pairs)

Class head: flat 4-way softmax, or hierarchical p(class) = p(possession) * p(organization | possession)
with the possession factor tied to the auxiliary (event) possession head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4
MIRROR = torch.tensor([2, 3, 0, 1])
EDGE_TYPES = ("intra_ref", "intra_opp", "press", "ball")


def edge_masks(pos: torch.Tensor, valid: torch.Tensor, k_press: int = 3, mark: torch.Tensor | None = None) -> torch.Tensor:
    """pos (N, 23, 2), valid (N, 23), mark (N, 23) partner index or -1 -> bool masks (N, R, 23, 23)."""
    N = pos.shape[0]
    dev = pos.device
    team = torch.cat([torch.zeros(11), torch.ones(11), torch.full((1,), 2)]).to(dev)
    same = team[:, None] == team[None, :]
    eye = torch.eye(23, dtype=torch.bool, device=dev)
    vv = valid[:, :, None] & valid[:, None, :]

    intra_ref = (same & (team[:, None] == 0) & ~eye)[None] & vv
    intra_opp = (same & (team[:, None] == 1) & ~eye)[None] & vv

    D = torch.cdist(pos, pos)
    cross = (team[:, None] != team[None, :]) & (team[:, None] < 2) & (team[None, :] < 2)
    Dm = D.masked_fill(~(cross[None] & vv), float("inf"))
    kth = Dm.topk(k_press, dim=-1, largest=False).values[..., -1:]
    knn = (Dm <= kth) & torch.isfinite(Dm)
    press = knn | knn.transpose(1, 2)

    is_ball = team == 2
    ball = ((is_ball[:, None] ^ is_ball[None, :]))[None] & vv
    masks = [intra_ref, intra_opp, press, ball]
    if mark is not None:
        m = torch.zeros(N, 23, 23, device=dev)
        m.scatter_(2, mark.clamp(min=0).unsqueeze(-1), (mark >= 0).float().unsqueeze(-1))
        masks.append((m > 0) & vv)
    return torch.stack(masks, 1), D


class HeteroAttnLayer(nn.Module):
    """GAT-style multi-head attention with separate parameters per edge type + distance bias."""

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1, R: int = len(EDGE_TYPES)):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(R)])
        self.dist_scale = nn.Parameter(torch.full((R, heads), 0.5))
        self.out = nn.Linear(R * d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, masks, dist):
        N, V, d = h.shape
        outs = []
        for r, proj in enumerate(self.qkv):
            q, k, v = proj(h).view(N, V, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
            logit = q @ k.transpose(-1, -2) / self.dk ** 0.5
            logit = logit - F.softplus(self.dist_scale[r])[None, :, None, None] * dist[:, None] / 10
            m = masks[:, r][:, None]
            logit = logit.masked_fill(~m, float("-inf"))
            att = torch.softmax(logit, -1).nan_to_num(0.0)  # rows with no neighbours -> 0
            outs.append((att @ v).transpose(1, 2).reshape(N, V, d))
        h = self.ln1(h + self.drop(self.out(torch.cat(outs, -1))))
        return self.ln2(h + self.drop(self.ff(h)))


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h, valid):
        s = self.score(h).squeeze(-1).masked_fill(~valid, float("-inf"))
        w = torch.softmax(s, -1).nan_to_num(0.0)
        return (w.unsqueeze(-1) * h).sum(1)


class SpatialEncoder(nn.Module):
    def __init__(self, node_dim, glob_dim, d=64, layers=2, heads=4, use_graph=True, dropout=0.1, use_mark=False):
        super().__init__()
        self.use_graph = use_graph
        self.embed = nn.Sequential(nn.Linear(node_dim, d), nn.GELU(), nn.Linear(d, d))
        R = len(EDGE_TYPES) + int(use_mark)
        self.layers = nn.ModuleList([HeteroAttnLayer(d, heads, dropout, R) for _ in range(layers)])
        self.pool_ref, self.pool_opp = AttnPool(d), AttnPool(d)
        in_dim = (4 * d if use_graph else 0) + glob_dim
        self.mix = nn.Sequential(nn.Linear(in_dim, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 2 * d))
        self.out_dim = 2 * d

    def forward(self, node, pos, valid, glob, mark=None):
        parts = []
        if self.use_graph:
            B, T, V, _ = node.shape
            h = self.embed(node.reshape(B * T, V, -1))
            p, vd = pos.reshape(B * T, V, 2), valid.reshape(B * T, V)
            masks, dist = edge_masks(p, vd, mark=None if mark is None else mark.reshape(B * T, V))
            for layer in self.layers:
                h = layer(h, masks, dist)
            z_ref = self.pool_ref(h[:, :11], vd[:, :11])
            z_opp = self.pool_opp(h[:, 11:22], vd[:, 11:22])
            # the difference term encodes inter-team organizational asymmetry
            parts += [z_ref, z_opp, z_ref - z_opp, h[:, 22]]
            parts = [x.reshape(B, T, -1) for x in parts]
        parts.append(glob)
        return self.mix(torch.cat(parts, -1))


class DilatedResidual(nn.Module):
    def __init__(self, d, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.proj(F.relu(self.conv(x))))


class TemporalEncoder(nn.Module):
    def __init__(self, d_in, kind="bigru", d=128, dropout=0.2, tcn_layers=10):
        super().__init__()
        self.kind = kind
        if kind == "bigru":
            self.rnn = nn.GRU(d_in, d, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
            self.out_dim = 2 * d
        elif kind == "tcn":  # MS-TCN single stage, non-causal
            self.inp = nn.Conv1d(d_in, 2 * d, 1)
            self.blocks = nn.Sequential(*[DilatedResidual(2 * d, 2 ** i, dropout) for i in range(tcn_layers)])
            self.out_dim = 2 * d
        elif kind == "none":
            self.out_dim = d_in
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "bigru":
            return self.rnn(x)[0]
        if self.kind == "tcn":
            return self.blocks(self.inp(x.transpose(1, 2))).transpose(1, 2)
        return x


class PhaseModel(nn.Module):
    def __init__(self, node_dim, glob_dim, d=64, temporal="bigru", use_graph=True, dropout=0.2,
                 head="flat", use_mark=False):
        super().__init__()
        self.spatial = SpatialEncoder(node_dim, glob_dim, d=d, use_graph=use_graph, dropout=dropout, use_mark=use_mark)
        self.temporal = TemporalEncoder(self.spatial.out_dim, temporal, d=2 * d, dropout=dropout)
        D = self.temporal.out_dim
        self.head = head
        self.cls = nn.Linear(D, N_CLASSES if head == "flat" else 2)  # hier: org logits | ref poss, | opp poss
        self.bnd = nn.Linear(D, 1)
        self.poss = nn.Linear(D, 1)

    def forward(self, node, pos, valid, glob, mark=None):
        z = self.temporal(self.spatial(node, pos, valid, glob, mark))
        u = self.poss(z).squeeze(-1)
        if self.head == "flat":
            cls = self.cls(z)
        else:  # log p over [est_att, def_to_att, est_def, att_to_def] = log p(poss) + log p(org | poss)
            o = self.cls(z)
            lpu, lnu = F.logsigmoid(u), F.logsigmoid(-u)
            cls = torch.stack([lpu + F.logsigmoid(-o[..., 0]), lpu + F.logsigmoid(o[..., 0]),
                               lnu + F.logsigmoid(-o[..., 1]), lnu + F.logsigmoid(o[..., 1])], -1)
        return dict(cls=cls, bnd=self.bnd(z).squeeze(-1), poss=u)


def loss_fn(out, batch, cfg, class_w):
    """out/batch tensors are (2B, T, ...): first B = home view, last B = mirrored away view."""
    y, m = batch["label"], batch["label"] >= 0
    logits = out["cls"]
    logp = F.log_softmax(logits, -1)

    # focal CE with class weights, masked to labeled frames
    if "soft" in batch:  # soft targets (annotator vote shares), STUDY1_PLAN §4.5
        q = batch["soft"]
        m = q.sum(-1) > 0
        focal = -(q * (1 - logp.exp()) ** cfg["gamma"] * logp * class_w).sum(-1)
    else:
        yc = y.clamp(min=0)
        lp = logp.gather(-1, yc.unsqueeze(-1)).squeeze(-1)
        focal = -((1 - lp.exp()) ** cfg["gamma"]) * lp * class_w[yc]
    l_cls = (focal * m).sum() / m.sum().clamp(min=1)

    # boundary: Gaussian-smoothed targets
    bm = batch["bnd_mask"]
    l_bnd = (F.binary_cross_entropy_with_logits(out["bnd"], batch["bnd"], reduction="none",
                                                pos_weight=torch.tensor(5.0, device=y.device)) * bm).sum() / bm.sum().clamp(min=1)

    # MS-TCN truncated smoothing loss
    dl = (logp[:, 1:] - logp[:, :-1].detach()) ** 2
    l_smooth = dl.clamp(max=16).mean()

    # auxiliary possession: annotator possession on labelled frames, or the input possession on alive frames
    # (train.Data, cfg poss_target)
    am = batch["poss_mask"]
    l_poss = (F.binary_cross_entropy_with_logits(out["poss"], batch["poss_tgt"], reduction="none") * am).sum() / am.sum().clamp(min=1)

    # view symmetry: p(home view) ~ mirror(p(away view))
    B = logits.shape[0] // 2
    pa = logp[:B]
    pb = logp[B:][..., MIRROR.to(y.device)]
    l_sym = F.kl_div(pb, pa, log_target=True, reduction="batchmean") / logits.shape[1]

    total = l_cls + cfg["lam_bnd"] * l_bnd + cfg["lam_smooth"] * l_smooth + cfg["lam_poss"] * l_poss + cfg["lam_sym"] * l_sym
    return total, dict(cls=l_cls.item(), bnd=l_bnd.item(), smooth=l_smooth.item(), poss=l_poss.item(), sym=l_sym.item())
