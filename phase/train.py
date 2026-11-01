"""Blocked cross-validation on one partially annotated match (smoke test of the full pipeline).

The labeled stretch is cut into K contiguous time blocks; each fold trains on the others
(minus a buffer around the test block) and predicts the test block. Out-of-fold predictions
are pooled and compared against the timer baseline L_fix(tau).

NOTE: one match -> no match-level GroupKFold is possible. Numbers are a sanity check, not a result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import events as Ev
from . import features as Fe
from .model import MIRROR, PhaseModel, loss_fn


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def boundary_target(label: np.ndarray, sigma: float) -> np.ndarray:
    idx = np.flatnonzero((label[1:] != label[:-1]) & (label[1:] >= 0) & (label[:-1] >= 0)) + 1
    t = np.arange(len(label))
    tgt = np.zeros(len(label), np.float32)
    for b in idx:
        lo, hi = max(0, b - int(4 * sigma)), min(len(label), b + int(4 * sigma) + 1)
        tgt[lo:hi] = np.maximum(tgt[lo:hi], np.exp(-((t[lo:hi] - b) ** 2) / (2 * sigma ** 2)))
    return tgt


class Data:
    """Holds both views as tensors on CPU; builds window batches.

    Auxiliary possession target (cfg poss_target): "label" = the annotators' possession, i.e. the frame label is
    one of the reference team's classes (est_attack, def_to_att; soft labels: q0 + q1), on labelled training
    frames; "input" = the possession fed to the model (poss_ref: events or tracking heuristic) on alive training
    frames (the behaviour before 2026-11). cfg poss_input=False drops poss_ref from the context features."""

    def __init__(self, views, d, cfg):
        self.T = len(d["label"])
        self.poss_target = cfg.get("poss_target", "label")
        if self.poss_target not in ("label", "input"):
            raise ValueError(f"poss_target must be 'label' or 'input', got {self.poss_target!r}")
        use = [i for i, n in enumerate(views[0]["ctx_names"])
               if (cfg["timer_feat"] or n != "time_since_turnover") and (cfg.get("poss_input", True) or n != "poss_ref")]
        self.ctx_names = [views[0]["ctx_names"][i] for i in use]
        self.soft = cfg.get("soft", False)
        self.v = {}
        for r in (0, 1):
            v = views[r]
            g = v["glob"] if cfg["org_block"] else v["glob"][:, :0]
            g = np.concatenate([g, v["ctx"][:, use]], 1)
            if self.poss_target == "input":
                tgt, has = v["poss_ref"], d["alive"].astype(bool)
            elif self.soft:
                q = d["label_soft"] if r == 0 else d["label_soft"][:, Fe.MIRROR]
                tgt, has = q[:, 0] + q[:, 1], q.sum(1) > 0
            else:
                tgt, has = np.isin(v["label"], (0, 1)), v["label"] >= 0
            self.v[r] = dict(
                node=torch.from_numpy(v["node"]), pos=torch.from_numpy(v["node_pos"]),
                valid=torch.from_numpy(v["node_mask"]), glob=torch.from_numpy(g.astype(np.float32)),
                label=torch.from_numpy(v["label"].astype(np.int64)),
                poss_tgt=torch.from_numpy(np.asarray(tgt, np.float32)),
                poss_mask=torch.from_numpy(np.asarray(has, np.float32)),
                bnd=torch.from_numpy(boundary_target(v["label"], cfg["bnd_sigma_s"] * float(d["hz"]))),
            )
        if self.soft:  # npz keys label_soft (T, 4) and label_A / label_B (home view, -1 = not coded)
            sig = cfg["bnd_sigma_s"] * float(d["hz"])
            bnd = 0.5 * (boundary_target(d["label_A"], sig) + boundary_target(d["label_B"], sig))
            for r in (0, 1):
                q = d["label_soft"] if r == 0 else d["label_soft"][:, Fe.MIRROR]
                self.v[r]["soft"] = torch.from_numpy(q.astype(np.float32))
                self.v[r]["bnd"] = torch.from_numpy(bnd.astype(np.float32))
        self.mark = "mark" in views[0]
        if self.mark:
            for r in (0, 1):
                self.v[r]["mark"] = torch.from_numpy(views[r]["mark"])
        self.glob_dim = self.v[0]["glob"].shape[1]
        self.node_dim = self.v[0]["node"].shape[2]

    def normalize(self, train_idx):
        g = torch.cat([self.v[r]["glob"][train_idx] for r in (0, 1)])
        mu, sd = g.mean(0), g.std(0).clamp(min=1e-4)
        return mu, sd

    def batch(self, starts, W, mu, sd, train_mask, device):
        out = {}
        keys = ("node", "pos", "valid", "glob", "label", "poss_tgt", "poss_mask", "bnd") \
            + (("soft",) if self.soft else ()) + (("mark",) if self.mark else ())
        for key in keys:
            xs = [self.v[r][key][s:s + W] for r in (0, 1) for s in starts]
            out[key] = torch.stack(xs)
        out["glob"] = (out["glob"] - mu) / sd
        tm = torch.stack([train_mask[s:s + W] for _ in (0, 1) for s in starts])
        out["label"] = torch.where(tm, out["label"], torch.full_like(out["label"], -1))
        if self.soft:
            out["soft"] = out["soft"] * tm[..., None]
            out["bnd_mask"] = (out["soft"].sum(-1) > 0).float()
        else:
            out["bnd_mask"] = ((out["label"] >= 0)).float()
        out["poss_mask"] = out["poss_mask"] * tm.float()  # training frames only
        return {k: v.to(device) for k, v in out.items()}


@torch.no_grad()
def predict(model, data, mu, sd, lo, hi, W, device):
    """Sliding-window inference over [lo, hi), both views, overlap-averaged probs."""
    model.eval()
    T = hi - lo
    probs = np.zeros((T, 4))
    bnd = np.zeros(T)
    cnt = np.zeros(T)
    step = W // 2
    starts = list(range(lo, max(lo + 1, hi - W + 1), step))
    if starts[-1] + W < hi:
        starts.append(max(lo, hi - W))
    for s in starts:
        e = min(s + W, data.T)
        b = data.batch([s], e - s, mu, sd, torch.ones(data.T, dtype=torch.bool), device)
        out = model(b["node"], b["pos"], b["valid"], b["glob"], b.get("mark"))
        p = torch.softmax(out["cls"], -1).cpu().numpy()
        pb = torch.sigmoid(out["bnd"]).cpu().numpy()
        p_sym = 0.5 * (p[0] + p[1][:, MIRROR.numpy()])  # test-time view symmetry
        w = np.hanning(e - s + 2)[1:-1] + 1e-3
        sl = slice(s - lo, e - lo)
        n = min(e, hi) - s
        probs[s - lo:s - lo + n] += p_sym[:n] * w[:n, None]
        bnd[s - lo:s - lo + n] += 0.5 * (pb[0] + pb[1])[:n] * w[:n]
        cnt[s - lo:s - lo + n] += w[:n]
    return probs / cnt[:, None], bnd / cnt


def run(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = get_device()
    d = dict(np.load(cfg["npz"]))
    hz = float(d["hz"])
    if cfg["events"]:  # Opta events synced to tracking (phase.events)
        poss, turnovers = Ev.possession(Ev.load(cfg["events"], d), len(d["label"]))
    else:
        poss, turnovers = Fe.possession(d, min_run_s=cfg["poss_min_run_s"]), None
    views = Fe.build(d, poss, mark=cfg["mark"])
    data = Data(views, d, cfg)
    label = views[0]["label"]
    poss_home = poss  # 0 = home

    lab_idx = np.flatnonzero(label >= 0)
    L0, L1 = 0, int(lab_idx[-1]) + 1
    W = int(cfg["window_s"] * hz)
    buf = int(cfg["buffer_s"] * hz)
    edges = np.linspace(L0, L1, cfg["folds"] + 1).astype(int)

    # timer tau*: tau, cut at the next dead ball, and (event turnovers only) skip dead-ball turnovers, tuned per fold
    alive = d["alive"].astype(bool)
    timer_grid = [(float(tau), cut, live) for tau in np.arange(1, 20.5, 0.5) for cut in (False, True)
                  for live in ((False, True) if turnovers is not None else (False,))]

    def timer(tau, cut, live):
        return E.timer_baseline(poss_home, tau, hz, alive if cut else None, turnovers, live)[L0:L1]

    oof = np.zeros((L1 - L0, 4))
    oof_bnd = np.zeros(L1 - L0)
    pred_timer_star = np.zeros(L1 - L0, int)
    pred_hsmm = np.zeros(L1 - L0, int)
    fold_info = []
    t_start = time.time()
    for f in range(cfg["folds"]):
        a, b = edges[f], edges[f + 1]
        train_mask = np.zeros(data.T, bool)
        train_mask[L0:L1] = True
        train_mask[max(L0, a - buf):min(L1, b + buf)] = False
        tm_t = torch.from_numpy(train_mask)

        # windows fully inside the training region
        ok = np.convolve(train_mask.astype(int), np.ones(W, int), "valid") == W
        cand = np.flatnonzero(ok)
        stride = int(cfg["stride_s"] * hz)
        starts_all = cand[::stride]
        # keep windows with enough labeled frames
        coded = (d["label_soft"].sum(1) > 0) if cfg["soft"] else label >= 0
        lab_frac = np.array([coded[s:s + W].mean() for s in starts_all])
        starts_all = starts_all[lab_frac > 0.1]

        train_idx = torch.from_numpy(np.flatnonzero(train_mask))
        mu, sd = data.normalize(train_idx)

        if cfg["soft"]:
            freq = d["label_soft"][train_mask].sum(0) + 1
        else:
            tr_lab = label[train_mask]
            freq = np.bincount(tr_lab[tr_lab >= 0], minlength=4) + 1
        class_w = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
        class_w = (class_w / class_w.mean()).to(device)

        model = PhaseModel(data.node_dim, data.glob_dim, d=cfg["d"], temporal=cfg["temporal"],
                           use_graph=cfg["graph"], dropout=cfg["dropout"], head=cfg["head"],
                           use_mark=cfg["mark"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
        steps = cfg["epochs"] * int(np.ceil(len(starts_all) / cfg["batch"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1))
        for ep in range(cfg["epochs"]):
            model.train()
            # random jitter of window starts
            jit = np.random.randint(0, stride, len(starts_all))
            st = np.clip(starts_all + jit, 0, None)
            st = st[[ok[s] if s < len(ok) else False for s in st]]
            np.random.shuffle(st)
            logs = []
            for i in range(0, len(st), cfg["batch"]):
                b_ = data.batch(st[i:i + cfg["batch"]], W, mu, sd, tm_t, device)
                out = model(b_["node"], b_["pos"], b_["valid"], b_["glob"], b_.get("mark"))
                loss, parts = loss_fn(out, b_, cfg, class_w)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                logs.append(loss.item())
            if (ep + 1) % max(1, cfg["epochs"] // 5) == 0:
                print(f"  fold {f} ep {ep + 1}/{cfg['epochs']} loss {np.mean(logs):.3f} "
                      f"({time.time() - t_start:.0f}s)", flush=True)

        # --- predict test block with context
        ctx = W // 2
        lo, hi = max(0, a - ctx), min(data.T, b + ctx)
        probs, bnd = predict(model, data, mu, sd, lo, hi, W, device)
        oof[a - L0:b - L0] = probs[a - lo:b - lo]
        oof_bnd[a - L0:b - L0] = bnd[a - lo:b - lo]

        # --- per-fold tuned baselines / decoders (fit on training labels only)
        tr_label = np.where(train_mask, label, -1)[L0:L1]
        best, best_f1 = None, -1
        for setting in timer_grid:
            f1 = E.macro_f1(tr_label, timer(*setting))
            if f1 > best_f1:
                best, best_f1 = setting, f1
        pred_timer_star[a - L0:b - L0] = timer(*best)[a - L0:b - L0]
        hp = E.fit_hsmm(tr_label, hz)
        seg_logp = np.log(probs[a - lo:b - lo] + 1e-6)
        pred_hsmm[a - L0:b - L0] = E.hsmm_decode(seg_logp, hp, cfg["switch_penalty"])
        fold_info.append(dict(fold=f, block=[int(a), int(b)], tau_star=best[0], timer_cut_at_dead_ball=best[1],
                              timer_live_only=best[2], hsmm_min_d_s=(hp["min_d"] / hz).round(2).tolist(),
                              n_train_windows=int(len(starts_all))))
        print(f"fold {f}: tau*={best[0]}s (cut at dead ball {best[1]}, live only {best[2]})  "
              f"hsmm min_d={fold_info[-1]['hsmm_min_d_s']}", flush=True)

    gt = label[L0:L1]
    pred_arg = oof.argmax(1)
    preds = {"timer_tau*": pred_timer_star, "model_argmax": pred_arg,
             "model_modefilter5s": E.mode_filter(pred_arg, int(5 * hz)), "model_hsmm": pred_hsmm}
    for tau in (3, 5, 6, 9.34):
        preds[f"timer_{tau}s"] = E.timer_baseline(poss_home, tau, hz)[L0:L1]
    results = {k: E.evaluate(gt, v, hz) for k, v in preds.items()}
    ci = E.bootstrap(gt, preds, hz, n_boot=cfg["n_boot"], seed=cfg["seed"])
    # possession heuristic quality vs labels (est_attack/def_to_att = home in possession)
    ph = np.nan_to_num(poss_home[L0:L1], nan=0)
    m = gt >= 0
    poss_agree = float(((ph[m] == 0) == np.isin(gt[m], (0, 1))).mean())

    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "oof.npz", probs=oof, bnd=oof_bnd, gt=gt, hsmm=pred_hsmm,
                        timer_star=pred_timer_star, poss_home=poss_home[L0:L1], alive=d["alive"][L0:L1], hz=hz)
    summary = dict(config=cfg, folds=fold_info, possession_label_agreement=poss_agree,
                   elapsed_s=time.time() - t_start, results=results, ci=ci)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_table(summary):
    cols = ["macro_f1", "kappa", "transition_f1", "bnd_f1@1s", "bnd_f1@2s", "seg_f1@0.25", "seg_f1@0.5", "n_pred_boundaries"]
    print(f"\n{'method':22s}" + "".join(f"{c:>18s}" for c in cols))
    for k, r in summary["results"].items():
        print(f"{k:22s}" + "".join(f"{r[c]:18.3f}" if isinstance(r[c], float) else f"{r[c]:18d}" for c in cols))
    r = next(iter(summary["results"].values()))
    print(f"GT boundaries: {r['n_gt_boundaries']} | labeled frames: {r['n_frames']} | "
          f"possession ({'events' if summary['config'].get('events') else 'heuristic'}) agrees with labels on {summary['possession_label_agreement']:.1%} of frames")
    print("per-class F1 [est_att, def->att, est_def, att->def]:")
    for k, r in summary["results"].items():
        print(f"  {k:22s} {np.round(r['f1_per_class'], 3).tolist()}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/ajax_psv.npz")
    ap.add_argument("--out", default="artifacts/run_default")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--window_s", type=float, default=60)
    ap.add_argument("--stride_s", type=float, default=5)
    ap.add_argument("--buffer_s", type=float, default=15)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--temporal", default="bigru", choices=["bigru", "tcn", "none"])
    ap.add_argument("--no-graph", dest="graph", action="store_false")
    ap.add_argument("--no-org", dest="org_block", action="store_false")
    ap.add_argument("--no-timer-feat", dest="timer_feat", action="store_false")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam_bnd", type=float, default=0.5)
    ap.add_argument("--lam_smooth", type=float, default=0.15)
    ap.add_argument("--lam_poss", type=float, default=0.1)
    ap.add_argument("--lam_sym", type=float, default=0.1)
    ap.add_argument("--bnd_sigma_s", type=float, default=0.5)
    ap.add_argument("--switch_penalty", type=float, default=2.0)
    ap.add_argument("--poss_min_run_s", type=float, default=1.0)
    ap.add_argument("--events", default=None, help="event_data.csv -> event-based possession")
    ap.add_argument("--soft", action="store_true", help="train on npz label_soft (annotator vote shares)")
    ap.add_argument("--head", default="flat", choices=["flat", "hier"], help="hier: p(possession) x p(organization | possession)")
    ap.add_argument("--poss_target", default="label", choices=["label", "input"],
                    help="auxiliary possession target: annotator possession from the labels (default) or the input "
                         "possession (events / heuristic; runs before 2026-11)")
    ap.add_argument("--no-poss-input", dest="poss_input", action="store_false",
                    help="ablation: drop poss_ref from the context features")
    ap.add_argument("--mark", action="store_true", help="add 1:1 marking edges (Hungarian outfield assignment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=2000, help="bootstrap replicates over eval runs (results.json ci)")
    return ap


if __name__ == "__main__":
    s = run(vars(parser().parse_args()))
    print_table(s)
