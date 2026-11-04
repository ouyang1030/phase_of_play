"""Possession-structured end-point model (main model) and its baselines on the same episodes.

Every turnover opens an episode (phase.episodes); the model reads the frames of its window [s0, t1) in the view of
the new possessor and outputs p(E = k) for k = 0 .. W - 1 (the transition ends at window frame k) and k = W (it
fills the window). Labels follow by construction (phase.episodes.reconstruct): no HSMM, mode filter or other
post-processing. Training target (--target):
  label     the npz labels: P(E in [lo, hi]) with [lo, hi] = phase.episodes.observe (adjudicated labels; with a
            single coder's npz this is the L2 run "trained on one coder only")
  interval  two coders: P(E in [min lo, max hi]), their disagreement taken as an interval, not as an error
  noise     two coders as noisy observations of a latent length E* (phase.endpoint.noise_loglik): coder offsets
            and scales are learnt with the model; predictions are E* (no coder offset)
Decoding: the q-quantile of p(E), q tuned on the training frames (macro-F1) like tau for the timer, or the median
(--decode median; the median minimises the expected number of wrongly labelled window frames).
Baselines on the same episodes: timer (one tau for every episode) and rule (phase.concepts), tuned per fold.
Blocked CV as phase.train / phase.hazard (phase.episodes.cv_folds).

  python -m phase.duration --npz NPZ --events EVENTS --out DIR [--meta META] [--poss_confirm second]
         [--target label|interval|noise] [--coders A.npz B.npz] [--features org,concepts,ball,time,episode]
         [--no-model] [--ensemble 5] [--epochs 80] ...
Writes DIR/oof.npz (gt, pred, probs, timer, rule, poss_home, alive, hz), DIR/episodes.csv, DIR/results.json.
--poss labels reads the possession off the labels (oracle input, only for phase.decompose).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import concepts as C
from . import episodes as P
from . import evaluate as E
from . import events as Ev
from . import features as Fe
from . import meta as M

HOW = ["intercept", "recovery", "tackle", "keeper", "other"]
GROUPS = ("org", "concepts", "ball", "time", "episode")
QUANTILES = np.round(np.arange(0.2, 0.81, 0.05), 2)
BALL = ("ball_x", "ball_y", "ball_z", "ball_speed")


# ---------------------------------------------------------------- features
def frame_features(views: dict, conc: dict, groups=GROUPS) -> tuple[dict, list]:
    """Per view w the (T, F) frame features: org = unsmoothed organisation block of phase.features (ref = w, the
    attacking team; opp = the defending team), concepts = phase.concepts, ball = ball x, y, z, speed in w's view."""
    n_raw = len(views[0]["glob_names"]) // 3
    cols = [views[0]["ctx_names"].index(n) for n in BALL]
    names = ((views[0]["glob_names"][:n_raw] if "org" in groups else []) + (C.NAMES if "concepts" in groups else [])
             + (list(BALL) if "ball" in groups else []))
    out = {}
    for w in (0, 1):
        parts = ([views[w]["glob"][:, :n_raw]] if "org" in groups else []) + ([conc[w]] if "concepts" in groups else []) \
            + ([views[w]["ctx"][:, cols]] if "ball" in groups else [])
        T = len(views[w]["ctx"])
        out[w] = np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((T, 0), np.float32)
    return out, names


def episode_inputs(ff: dict, ball_xy: dict, eps: pd.DataFrame, hz: float, max_s: float, groups=GROUPS):
    """Per episode the (W, F) model input: frame features of the possessor's view on [s0, t1), time since s0
    (fraction of max_s and log1p of the seconds) and episode constants (dead-ball turnover, how the ball was won,
    ball x, y at s0 in w's view). Returns the list and the names of the added columns."""
    extra = (["t_frac", "t_log"] if "time" in groups else []) + \
        (["dead", *[f"how_{h}" for h in HOW], "ball_x0", "ball_y0"] if "episode" in groups else [])
    X = []
    for s0, t1, w, dead, how in zip(eps.s0, eps.t1, eps.team, eps.dead, eps.how):
        W = int(t1 - s0)
        parts = [ff[w][s0:t1]]
        if "time" in groups:
            k = np.arange(W) / hz
            parts += [(k / max_s)[:, None], np.log1p(k)[:, None]]
        if "episode" in groups:
            const = np.r_[float(dead), np.eye(len(HOW))[HOW.index(how) if how in HOW else len(HOW) - 1],
                          ball_xy[w][min(s0, len(ball_xy[w]) - 1)]]
            parts.append(np.repeat(const[None], W, 0))
        X.append(np.concatenate(parts, 1).astype(np.float32) if W else np.zeros((0, sum(p.shape[1] for p in parts)), np.float32))
    return X, extra


# ---------------------------------------------------------------- targets, decoding, metrics
def targets(obs: dict, mode: str) -> tuple[dict, np.ndarray]:
    """Training targets from observe() tables {name: DataFrame(lo, hi, kind)} -> (tgt for phase.endpoint,
    usable (n,) bool). label: the only table; interval: hull of the coders' intervals; noise: one column per coder."""
    names = list(obs)
    lo = np.stack([obs[k].lo.to_numpy(int) for k in names], 1)
    hi = np.stack([obs[k].hi.to_numpy(int) for k in names], 1)
    use = np.stack([(obs[k].kind != "none").to_numpy() for k in names], 1)
    if mode == "label":
        if len(names) != 1:
            raise ValueError("target label uses exactly one label source")
        return dict(mode="range", lo=lo[:, 0], hi=hi[:, 0]), use[:, 0]
    if mode == "interval":
        usable = use.any(1)
        lo_c = np.where(use, lo, np.iinfo(np.int64).max).min(1)
        hi_c = np.where(use, hi, -1).max(1)
        return dict(mode="range", lo=np.where(usable, lo_c, 0), hi=np.where(usable, hi_c, 0)), usable
    if mode == "noise":
        return dict(mode="noise", lo=lo.astype(float), hi=hi.astype(float), use=use), use.any(1)
    raise ValueError(f"target must be label, interval or noise, got {mode!r}")


def quantile_length(p: np.ndarray, q: float) -> int:
    """Smallest k with P(E <= k) >= q (k = W: the transition fills the window)."""
    return int(min(np.searchsorted(np.cumsum(p), q - 1e-12), len(p) - 1))


def lengths_from(probs: dict, n: int, q: float) -> np.ndarray:
    out = np.zeros(n, int)
    for i, p in probs.items():
        out[i] = quantile_length(p, q)
    return out


def frame_probs(poss_home, eps, probs: dict, T: int) -> np.ndarray:
    """(T, 4) class probabilities: established of the possessor outside the windows, inside a window of w
    P(transition_w at frame s0 + k) = P(E > k)."""
    est = E.established(poss_home)
    out = np.eye(4)[est]
    for i, p in probs.items():
        s0, t1, w = int(eps.s0.iat[i]), int(eps.t1.iat[i]), int(eps.team.iat[i])
        tr = 1 - np.cumsum(p)[:t1 - s0]
        out[s0:t1] = 0
        out[s0:t1, P.TRANS[w]] = tr
        out[s0:t1, P.EST[w]] = 1 - tr
    return out


def crps(p: np.ndarray, e: int, hz: float) -> float:
    """Continuous ranked probability score of p(E) for the observed length e, in s."""
    return float(np.sum((np.cumsum(p) - (np.arange(len(p)) >= e)) ** 2) / hz)


def duration_metrics(E_pred: np.ndarray, obs: pd.DataFrame, hz: float, probs: list | None = None) -> dict:
    """Against the exact observations of one label source (lo == hi < W; episodes with E_pred >= 0): MAE, median
    absolute error and share within 1 s (s), zero-length agreement; with probs also CRPS (s) and the share inside
    the central 80% interval."""
    ex = ((obs.lo == obs.hi) & (obs.lo < obs.W)).to_numpy() & (np.asarray(E_pred) >= 0)
    e_obs, e_pr = obs.lo.to_numpy()[ex], np.asarray(E_pred)[ex]
    err = np.abs(e_pr - e_obs) / hz
    out = dict(n_exact=int(ex.sum()), mae_s=float(err.mean()) if len(err) else float("nan"),
               median_ae_s=float(np.median(err)) if len(err) else float("nan"),
               within_1s=float((err <= 1).mean()) if len(err) else float("nan"),
               zero_acc=float(((e_pr == 0) == (e_obs == 0)).mean()) if len(err) else float("nan"))
    if probs is not None:
        sel = [(p, e) for p, e, k in zip(probs, obs.lo.to_numpy(), ex) if k and p is not None]
        if sel:
            out["crps_s"] = float(np.mean([crps(p, e, hz) for p, e in sel]))
            out["cover80"] = float(np.mean([quantile_length(p, 0.1) <= e <= quantile_length(p, 0.9) for p, e in sel]))
    return out


def coder_interval_metrics(E_pred, oa, ob, hz, probs=None, tol_s=0.5) -> dict:
    """Episodes where both coders give an exact length: share of predictions inside [min, max] +- tol_s; with
    probs the mean predicted mass on [min, max] (the coders' interval)."""
    ex = ((oa.lo == oa.hi) & (ob.lo == ob.hi) & (oa.lo < oa.W) & (ob.lo < ob.W)).to_numpy() & (np.asarray(E_pred) >= 0)
    a, b = oa.lo.to_numpy()[ex], ob.lo.to_numpy()[ex]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    e = np.asarray(E_pred)[ex]
    tol = tol_s * hz
    out = dict(n=int(ex.sum()), coder_gap_mae_s=float(np.mean(hi - lo) / hz) if ex.any() else float("nan"),
               inside=float(np.mean((e >= lo - tol) & (e <= hi + tol))) if ex.any() else float("nan"))
    if probs is not None:
        sel = [(p, l_, h_) for p, k, l_, h_ in zip(probs, ex, np.minimum(oa.lo, ob.lo), np.maximum(oa.hi, ob.hi))
               if k and p is not None]
        if sel:
            out["mass_on_interval"] = float(np.mean([np.sum(p[l_:h_ + 1]) for p, l_, h_ in sel]))
    return out


# ---------------------------------------------------------------- data sources
def label_sources(cfg: dict, d: dict) -> dict:
    """Coders' label arrays: --coders [name=]path (each npz 'label'), else the npz keys label_A / label_B."""
    T = len(d["label"])
    out = {}
    for i, spec in enumerate(cfg.get("coders") or []):
        name, path = spec.split("=", 1) if "=" in spec else ("AB"[i] if i < 2 else f"C{i}", spec)
        y = np.load(path)["label"].astype(int)
        if len(y) != T:
            raise ValueError(f"{path}: {len(y)} frames, the npz has {T}")
        out[name] = y
    if not out and "label_A" in d and "label_B" in d:
        out = {"A": d["label_A"].astype(int), "B": d["label_B"].astype(int)}
    return out


def possession(cfg, d, label):
    if cfg["poss"] == "labels":
        warnings.warn("--poss labels: possession read off the labels (oracle input, for phase.decompose only)")
        return P.label_possession(label), None
    if not cfg.get("events"):
        raise ValueError("--poss events needs --events")
    home_id = M.load(cfg["meta"])["home_team"]["id"] if cfg.get("meta") else None
    ev = Ev.load(cfg["events"], d, home_team_id=home_id)
    return Ev.possession(ev, len(label), confirm=cfg["poss_confirm"])


# ---------------------------------------------------------------- run
def run(cfg: dict) -> dict:
    t_start = time.time()
    d = dict(np.load(cfg["npz"]))
    hz = float(d["hz"])
    label = d["label"].astype(int)
    T = len(label)
    alive = d["alive"].astype(bool)
    groups = tuple(g for g in cfg["features"].split(",") if g)
    if set(groups) - set(GROUPS):
        raise ValueError(f"unknown feature groups {sorted(set(groups) - set(GROUPS))}; choose from {GROUPS}")
    coders = label_sources(cfg, d)
    if cfg["target"] in ("interval", "noise") and len(coders) < 2:
        raise ValueError(f"target {cfg['target']} needs two coders (--coders A.npz B.npz or npz label_A / label_B)")
    train_sources = {"label": label} if cfg["target"] == "label" else coders

    poss, to = possession(cfg, d, label)
    eps = P.episodes(poss, alive, d["period"], hz, cfg["max_s"], turnovers=to)
    Wn = P.lengths(eps)
    s0c = np.minimum(eps.s0.to_numpy(), T - 1)
    views = Fe.build(d, poss)
    conc = C.compute(d)
    rule = C.Rule(conc, eps, hz)
    X = names = None
    if cfg["model"]:
        ff, fnames = frame_features(views, conc, groups)
        X, extra = episode_inputs(ff, {w: views[w]["ctx"][:, :2] for w in (0, 1)}, eps, hz, cfg["max_s"], groups)
        names = fnames + extra
        if not names:
            raise ValueError("no input features: --features is empty")

    L0, L1, folds = P.cv_folds(label, cfg["folds"], int(cfg["buffer_s"] * hz))
    methods = ["timer", "rule"] + (["model"] if cfg["model"] else [])
    preds = {k: np.zeros(L1 - L0, int) for k in methods}
    E_by = {k: np.full(len(eps), -1) for k in methods}
    reason = np.full(len(eps), "", object)
    probs_frames = np.zeros((L1 - L0, 4)) if cfg["model"] else None
    p_ep = {}
    fold_info = []
    for f, (a, b, m) in enumerate(folds):
        sc = P.EpisodeScorer(np.where(m, label, -1), poss, eps)
        tau, _ = P.tune_constant(sc, hz)
        rp, _ = rule.fit(sc)
        lens = {"timer": np.full(len(eps), int(round(tau * hz))), "rule": rule.lengths(**rp)}
        info = dict(fold=f, block=[int(a), int(b)], tau_s=tau, rule=rp)
        if cfg["model"]:
            from . import endpoint
            tgt, usable = targets({n: P.observe(y, eps, allowed=m) for n, y in train_sources.items()}, cfg["target"])
            train_idx = np.flatnonzero(usable & m[s0c] & (Wn > 0))
            pred_idx = np.flatnonzero(Wn > 0)
            probs, minfo = endpoint.fit_predict(X, tgt, train_idx, pred_idx, cfg, hz, seed=cfg["seed"] + 100 * f)
            pmap = dict(zip(pred_idx.tolist(), probs))
            q = 0.5 if cfg["decode"] == "median" else \
                max(QUANTILES, key=lambda qq: (sc.macro_f1(lengths_from(pmap, len(eps), qq)), -abs(qq - 0.5)))
            lens["model"] = lengths_from(pmap, len(eps), float(q))
            info.update(q=float(q), model=minfo)
            probs_frames[a - L0:b - L0] = frame_probs(poss, eps, pmap, T)[a:b]
        in_block = ((eps.s0 >= a) & (eps.s0 < b)).to_numpy()
        for k, Ek in lens.items():
            preds[k][a - L0:b - L0] = P.reconstruct(poss, eps, Ek)[a:b]
            E_by[k][in_block] = Ek[in_block]
        reason[in_block] = np.array(rule.explain(rp, lens["rule"]), object)[in_block]
        if cfg["model"]:
            p_ep.update({i: pmap[i] for i in np.flatnonzero(in_block) if i in pmap})
        info["n_train_episodes"] = int(len(train_idx)) if cfg["model"] else None
        fold_info.append(info)
        print(f"fold {f}: tau={tau}s rule={rp}" + (f" q={info['q']} train episodes {info['n_train_episodes']}"
                                                   if cfg["model"] else ""), flush=True)

    gt = label[L0:L1]
    results = {k: E.evaluate(gt, v, hz) for k, v in preds.items()}
    ci = E.bootstrap(gt, preds, hz, n_boot=cfg["n_boot"], seed=cfg["seed"])
    obs_full = {n: P.observe(y, eps) for n, y in {"label": label, **coders}.items()}
    ep_probs = [p_ep.get(i) for i in range(len(eps))]
    durations = {k: {n: duration_metrics(E_by[k], o, hz, ep_probs if k == "model" else None)
                     for n, o in obs_full.items()} for k in methods}
    if len(coders) >= 2:
        ca, cb = list(coders)[:2]
        for k in methods:
            durations[k][f"{ca}|{cb} interval"] = coder_interval_metrics(E_by[k], obs_full[ca], obs_full[cb], hz,
                                                                         ep_probs if k == "model" else None)
    main = "model" if cfg["model"] else "rule"
    m_lab = gt >= 0
    agree = float(((E.fill_possession(poss)[L0:L1][m_lab] == 0) == np.isin(gt[m_lab], (0, 1))).mean())

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "oof.npz", gt=gt, pred=preds[main], method=main, timer=preds["timer"], rule=preds["rule"],
                        poss_home=poss[L0:L1], alive=alive[L0:L1], hz=hz,
                        **({"probs": probs_frames} if cfg["model"] else {}))
    tab = episode_table(eps, E_by, reason, ep_probs, obs_full, hz, fold_of(eps, folds), (eps.s0 < L1).to_numpy())
    tab.to_csv(out / "episodes.csv", index=False)
    summary = dict(config=cfg, n_episodes=int(len(eps)), n_empty_windows=int((Wn == 0).sum()),
                   features=names, possession_label_agreement=agree, folds=fold_info, results=results, ci=ci,
                   durations=durations, elapsed_s=time.time() - t_start)
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=_json))
    return summary


def fold_of(eps, folds) -> np.ndarray:
    f = np.full(len(eps), -1)
    for i, (a, b, _) in enumerate(folds):
        f[((eps.s0 >= a) & (eps.s0 < b)).to_numpy()] = i
    return f


def episode_table(eps, E_by, reason, ep_probs, obs, hz, fold, keep) -> pd.DataFrame:
    """One row per episode of the evaluated stretch: window, decoded lengths (s), rule reason, model quantiles,
    and each label source's observation [lo, hi] (s)."""
    tab = eps.assign(fold=fold, W_s=P.lengths(eps) / hz)
    for k, Ek in E_by.items():
        tab[f"E_{k}_s"] = np.where(Ek >= 0, Ek / hz, np.nan)
    tab["rule_reason"] = reason
    if "model" in E_by:
        for q, name in ((0.1, "q10"), (0.5, "median"), (0.9, "q90")):
            tab[f"model_{name}_s"] = [quantile_length(p, q) / hz if p is not None else np.nan for p in ep_probs]
        tab["model_p_zero"] = [p[0] if p is not None and len(p) > 1 else np.nan for p in ep_probs]
        tab["model_p_fill"] = [p[-1] if p is not None else np.nan for p in ep_probs]
    for n, o in obs.items():
        tab[f"obs_{n}_lo_s"], tab[f"obs_{n}_hi_s"], tab[f"obs_{n}_kind"] = o.lo / hz, o.hi / hz, o.kind
    return tab[keep].reset_index(drop=True)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def print_table(s):
    cols = ["macro_f1", "kappa", "transition_f1", "bnd_f1@1s", "bnd_typed_f1@2s", "seg_f1@0.5"]
    print(f"\n{'method':12s}" + "".join(f"{c:>18s}" for c in cols))
    for k, r in s["results"].items():
        print(f"{k:12s}" + "".join(f"{r[c]:18.3f}" for c in cols))
    print("transition length vs each label source (exact observations): MAE s / within 1 s")
    for k, by in s["durations"].items():
        print(f"  {k:10s} " + " | ".join(f"{n}: {v.get('mae_s', v.get('inside', float('nan'))):.2f}"
                                        f"/{v.get('within_1s', float('nan')):.2f}" for n, v in by.items()))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="phase.prep / apply_decisions npz (label = evaluation labels)")
    ap.add_argument("--events", default=None, help="event_data.csv (possession and turnovers)")
    ap.add_argument("--meta", default=None, help="meta.json (home team_id for the event mapping)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poss", default="events", choices=["events", "labels"])
    ap.add_argument("--poss_confirm", default="first", choices=["first", "second"])
    ap.add_argument("--target", default="label", choices=["label", "interval", "noise"])
    ap.add_argument("--coders", nargs="*", default=None, help="[name=]npz of each coder (label key), A then B")
    ap.add_argument("--features", default=",".join(GROUPS), help=f"comma list of {GROUPS}")
    ap.add_argument("--no-model", dest="model", action="store_false", help="baselines only (no torch needed)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buffer_s", type=float, default=15.0)
    ap.add_argument("--max_s", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--decode", default="tuned", choices=["tuned", "median"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=2000)
    return ap


if __name__ == "__main__":
    print_table(run(vars(parser().parse_args())))
