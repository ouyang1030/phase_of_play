"""Turnover-anchored discrete-time hazard model for transition duration (STUDY1_PLAN §15).

Each event turnover opens an episode for the new possessor w. At steps k = 0, 1, ... (dt = 0.4 s)
the model gives h_k = P(transition ends at k | not ended before), from features observed up to
t0 + k*dt only. The transition is decoded online as ending at the first k where the cumulative
end probability 1 - prod_{j<=k}(1 - h_j) reaches q (q tuned on training frames, as tau is for the
timer); a fixed timer is the special case of a hazard that depends on k alone.

Ground truth per episode (home-perspective labels): the first labeled frame of w's
established class ends the transition (E = 0 if the annotator labels no transition at all);
unlabeled frames, the next turnover, a period change or leaving the training region censor it.

Blocked CV identical to `train.py`, so the numbers sit next to `artifacts/run_full`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import evaluate as E
from . import events as Ev
from . import features as Fe

DT_S = 0.4
TIME_EDGES = np.array([0, 0.4, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30.01])
HOW = ["intercept", "recovery", "tackle", "keeper", "other"]
# compact organization set (defending = opp in w's view, attacking = ref)
ORG_CORE = [
    "opp_sil3", "opp_line_def", "opp_line_gap", "opp_length", "opp_width", "opp_hull",
    "opp_polarization", "opp_speed_mean", "opp_vx_mean", "ref_speed_mean", "ref_polarization",
    "ref_vx_mean", "cen_dx", "press_on_ref", "ball_ndiff_10", "ball_dmin_opp",
]
QUANTILES = np.round(np.arange(0.2, 0.81, 0.05), 2)
TAUS = [float(t) for t in np.arange(1, 20.5, 0.5)]
FEATURE_SETS = {
    "hazard_time": ("time",),
    "hazard_ctx": ("time", "ctx"),
    "hazard_org_core": ("time", "ctx", "org_core"),
    "hazard_org_all": ("time", "ctx", "org_all"),
}


def _trailing(a: np.ndarray, win: int) -> np.ndarray:
    return pd.DataFrame(a).rolling(win, min_periods=1).mean().to_numpy()


class Featurizer:
    """Row features for (episode, step k), causal: frame t = t0 + k*step and earlier only."""

    def __init__(self, d, views, turnovers):
        self.hz = float(d["hz"])
        self.step = int(round(DT_S * self.hz))
        n_raw = len(views[0]["glob_names"]) // 3  # first third = unsmoothed
        self.names = views[0]["glob_names"][:n_raw]
        self.core = [self.names.index(n) for n in ORG_CORE]
        self.raw = {w: views[w]["glob"][:, :n_raw] for w in (0, 1)}
        self.trail = {w: _trailing(self.raw[w], int(2 * self.hz)) for w in (0, 1)}
        ctx_keep = [i for i, n in enumerate(views[0]["ctx_names"]) if n not in ("poss_ref", "time_since_turnover")]
        self.ctx = {w: views[w]["ctx"][:, ctx_keep] for w in (0, 1)}
        per = d["period"].astype(int)
        self.home_sign = np.where(per == 0, 1.0, d["attack_sign"][per])
        self.to = turnovers

    def episode_ctx(self, i):
        r = self.to.iloc[i]
        s = self.home_sign[r.idx] * (1 if r.team == 0 else -1)
        how = np.eye(len(HOW))[HOW.index(r.how)]
        return np.r_[float(r.dead), how, r.x * s / Fe.HALF_L, abs(r.y) / Fe.HALF_W]

    def rows(self, i, ks, groups):
        r = self.to.iloc[i]
        w, t0 = int(r.team), int(r.idx)
        t = t0 + ks * self.step
        s = ks * DT_S
        parts = []
        if "time" in groups:
            parts += [np.eye(len(TIME_EDGES) - 1)[np.clip(np.searchsorted(TIME_EDGES, s, "right") - 1, 0, len(TIME_EDGES) - 2)],
                      np.log1p(s)[:, None]]
        if "ctx" in groups:
            parts += [np.repeat(self.episode_ctx(i)[None], len(ks), 0), self.ctx[w][t]]
        for g in ("org_core", "org_all"):
            if g in groups:
                cols = self.core if g == "org_core" else slice(None)
                raw, tr = self.raw[w][:, cols], self.trail[w][:, cols]
                parts += [raw[t], tr[t], raw[t] - raw[t0]]
        return np.concatenate(parts, 1)


def episode_targets(to, label, period, hz, allowed, max_s=30.0, grace_s=2.0):
    """Per turnover: (event step E or None, n_rows at risk). None n_rows = unusable."""
    step = int(round(DT_S * hz))
    T = len(label)
    nxt = np.r_[to.idx.to_numpy()[1:], T]
    out = []
    for i, r in enumerate(to.itertuples(index=False)):
        t0, w = int(r.idx), int(r.team)
        trans_c, est_c = (1, 0) if w == 0 else (3, 2)
        t_end = min(int(nxt[i]), t0 + int(max_s * hz), T)
        per_change = np.flatnonzero(period[t0:t_end] != period[t0])
        if len(per_change):
            t_end = t0 + int(per_change[0])
        ev, n, seen = None, 0, False
        for k in range(int((t_end - t0) / step) + 1):
            t = t0 + k * step
            if t >= t_end or not allowed[t]:
                break
            l = label[t]
            if l == est_c:
                ev = k if seen else 0
                n = ev + 1
                break
            if l == trans_c:
                seen, n = True, k + 1
                continue
            if l >= 0 and not seen and k * DT_S < grace_s:  # annotator switches possession a bit later
                continue
            break  # unlabeled / contradicting label -> censored here
        out.append((ev, n if (seen or ev is not None) else 0))
    return out


def fit(feat, targets, train_eps, groups, C):
    X, y = [], []
    for i in train_eps:
        ev, n = targets[i]
        if n == 0:
            continue
        ks = np.arange(n)
        X.append(feat.rows(i, ks, groups))
        y.append((ks == ev).astype(int) if ev is not None else np.zeros(n, int))
    X, y = np.concatenate(X), np.concatenate(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return (sc, clf), dict(n_rows=int(len(y)), n_events=int(y.sum()))


def end_cdf(model, feat, eps, groups, max_steps):
    """Per episode: cumulative P(transition over by step k), causal in k."""
    sc, clf = model
    out = {}
    for i in eps:
        ks = np.arange(max_steps[i] + 1)
        h = clf.predict_proba(sc.transform(feat.rows(i, ks, groups)))[:, 1]
        out[i] = 1 - np.cumprod(1 - h)
    return out


def durations(cdf, q):
    """Decoded length in steps: first k with P(over by k) >= q."""
    out = {}
    for i, F in cdf.items():
        hit = np.flatnonzero(F >= q)
        out[i] = int(hit[0]) if len(hit) else len(F)
    return out


def decode(poss_home, to, dur_steps, step, t_end):
    """Frame labels: established of possessor, transition for dur_steps after each turnover."""
    pred = E.established(poss_home)
    for i, n in dur_steps.items():
        t0 = int(to.idx.iat[i])
        pred[t0:min(t0 + n * step, t_end[i])] = 1 if to.team.iat[i] == 0 else 3
    return pred


def run(cfg):
    d = dict(np.load(cfg["npz"]))
    hz = float(d["hz"])
    T = len(d["label"])
    ev = Ev.load(cfg["events"], d)
    poss, to = Ev.possession(ev, T, confirm=cfg.get("poss_confirm", "first"))
    by_confirm = {c: Ev.possession(ev, T, confirm=c) for c in ("first", "second")}  # reported, not used
    poss_heur = Fe.possession(d)
    views = Fe.build(d, poss)
    label = views[0]["label"]
    period = d["period"]
    feat = Featurizer(d, views, to)
    step = feat.step

    lab_idx = np.flatnonzero(label >= 0)
    L0, L1 = 0, int(lab_idx[-1]) + 1
    buf = int(cfg["buffer_s"] * hz)
    edges = np.linspace(L0, L1, cfg["folds"] + 1).astype(int)
    t0s = to.idx.to_numpy()
    nxt = np.r_[t0s[1:], T]
    t_end = np.minimum(nxt, t0s + int(cfg["max_s"] * hz))
    for i, t0 in enumerate(t0s):
        pc = np.flatnonzero(period[t0:t_end[i]] != period[t0])
        if len(pc):
            t_end[i] = t0 + pc[0]
    max_steps = np.maximum((t_end - t0s) // step, 0)

    alive = d["alive"].astype(bool)
    timers = {  # name -> (possession, turnover table, grid of (tau, cut at dead ball, live only))
        "timer_tau*_heur": (poss_heur, None, [(tau, False, False) for tau in TAUS]),
        "timer_tau*": (poss, to, [(tau, False, False) for tau in TAUS]),
        "timer_tau*_live": (poss, to, [(tau, False, True) for tau in TAUS]),
        "timer_tau*_grid": (poss, to, [(tau, cut, live) for tau in TAUS for cut in (False, True) for live in (False, True)]),
    }
    preds = {k: np.zeros(L1 - L0, int) for k in [*timers, *FEATURE_SETS]}
    dur_log = {k: [] for k in FEATURE_SETS}
    fold_info = []
    for f in range(cfg["folds"]):
        a, b = edges[f], edges[f + 1]
        train_mask = np.zeros(T, bool)
        train_mask[L0:L1] = True
        train_mask[max(L0, a - buf):min(L1, b + buf)] = False
        tr_label = np.where(train_mask, label, -1)[L0:L1]

        # strongest timer: tau (and whether a stoppage ends the window / dead-ball turnovers open one) tuned on train
        info = dict(fold=f, block=[int(a), int(b)])
        for name, (p, tt, grid) in timers.items():
            best = (-1, None)
            for tau, cut, live in grid:
                tp = E.timer_baseline(p, tau, hz, alive if cut else None, tt, live)[L0:L1]
                f1 = E.macro_f1(tr_label, tp)
                if f1 > best[0]:
                    best = (f1, (tau, cut, live))
            tau, cut, live = best[1]
            info[name] = tau if name != "timer_tau*_grid" else dict(tau=tau, cut_at_dead_ball=cut, live_only=live)
            preds[name][a - L0:b - L0] = E.timer_baseline(p, tau, hz, alive if cut else None, tt, live)[L0:L1][a - L0:b - L0]

        targets = episode_targets(to, label, period, hz, train_mask, cfg["max_s"])
        train_eps = [i for i in range(len(to)) if train_mask[t0s[i]] and targets[i][1] > 0]
        test_eps = [i for i in range(len(to)) if a - int(cfg["max_s"] * hz) <= t0s[i] < b]
        test_mask = np.zeros(T, bool)
        test_mask[a:b] = True
        test_targets = episode_targets(to, label, period, hz, test_mask, cfg["max_s"])
        region_eps = [i for i in range(len(to)) if train_mask[t0s[i]]]
        for name, groups in FEATURE_SETS.items():
            model, st = fit(feat, targets, train_eps, groups, cfg["C"])
            # decision quantile tuned on training frames, like tau for the timer
            cdf_tr = end_cdf(model, feat, region_eps, groups, max_steps)
            best = (-1, 0.5)
            for q in QUANTILES:
                tp = decode(poss, to, durations(cdf_tr, q), step, t_end)[L0:L1]
                f1 = E.evaluate(tr_label, tp, hz)["macro_f1"]
                if f1 > best[0]:
                    best = (f1, float(q))
            dur = durations(end_cdf(model, feat, test_eps, groups, max_steps), best[1])
            pr = decode(poss, to, dur, step, t_end)[L0:L1]
            preds[name][a - L0:b - L0] = pr[a - L0:b - L0]
            info[name] = dict(st, q=best[1])
            for i in test_eps:
                e_, n_ = test_targets[i]
                if e_ is not None and a <= t0s[i] < b:
                    dur_log[name].append((e_ * DT_S, dur[i] * DT_S))
        info["n_train_episodes"] = len(train_eps)
        fold_info.append(info)
        print(f"fold {f}: tau*={info['timer_tau*']}s (live {info['timer_tau*_live']}s, heur {info['timer_tau*_heur']}s, "
              f"grid {info['timer_tau*_grid']}) "
              f"| train episodes {len(train_eps)}, rows {info['hazard_org_core']['n_rows']}, "
              f"ends {info['hazard_org_core']['n_events']} | q " + str({k: info[k]['q'] for k in FEATURE_SETS}), flush=True)

    gt = label[L0:L1]
    results = {k: E.evaluate(gt, v, hz) for k, v in preds.items()}
    if cfg.get("deep_oof"):
        o = np.load(cfg["deep_oof"])
        if len(o["gt"]) != len(gt) or not np.array_equal(o["gt"], gt):
            raise ValueError(f"{cfg['deep_oof']}: labels differ from {cfg['npz']}")
        results["deep_hsmm"] = E.evaluate(gt, o["hsmm"], hz)
        preds["deep_hsmm"] = o["hsmm"]
    # atypical slice: frames the best event timer gets wrong (§6 D.3)
    hard = (preds["timer_tau*"] != gt) & (gt >= 0)
    for k in results:
        results[k]["acc_on_timer_errors"] = float((preds[k][hard] == gt[hard]).mean()) if hard.any() else float("nan")
    dur_stats = {k: dict(n=len(v), mae_s=float(np.mean([abs(p - g) for g, p in v])) if v else float("nan"),
                         zero_acc=float(np.mean([(g == 0) == (p == 0) for g, p in v])) if v else float("nan"))
                 for k, v in dur_log.items()}
    m = gt >= 0
    agree = {n: float(((np.nan_to_num(p[L0:L1], nan=0)[m] == 0) == np.isin(gt[m], (0, 1))).mean())
             for n, p in (("heuristic", poss_heur), ("events", poss),
                          *((f"events_confirm_{c}", pc) for c, (pc, _) in by_confirm.items()))}
    ci = E.bootstrap(gt, {k: preds[k] for k in results}, hz, n_boot=cfg["n_boot"])
    summary = dict(config=cfg, folds=fold_info, possession_label_agreement=agree,
                   n_turnovers=int(len(to)), n_turnovers_dead=int(to.dead.sum()),
                   turnovers_by_confirm={c: dict(n=int(len(t)), unconfirmed_events=int(t.attrs["unconfirmed"]))
                                         for c, (_, t) in by_confirm.items()},
                   durations=dur_stats, results=results, ci=ci)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "preds.npz", gt=gt, poss_events=poss[L0:L1], poss_heur=poss_heur[L0:L1],
                        **{k.replace("*", "star").replace(" ", "_"): v for k, v in preds.items()})
    to.to_csv(out / "turnovers.csv", index=False)
    return summary


def print_table(s):
    cols = ["macro_f1", "kappa", "transition_f1", "bnd_f1@1s", "bnd_f1@2s", "seg_f1@0.5", "acc_on_timer_errors"]
    print(f"\n{'method':34s}" + "".join(f"{c:>20s}" for c in cols))
    for k, r in s["results"].items():
        print(f"{k:34s}" + "".join(f"{r[c]:20.3f}" for c in cols))
    print("per-class F1 [est_att, def->att, est_def, att->def]:")
    for k, r in s["results"].items():
        print(f"  {k:32s} {np.round(r['f1_per_class'], 3).tolist()}")
    print("possession agreement with labels:", {k: round(v, 3) for k, v in s["possession_label_agreement"].items()},
          f"| turnovers {s['n_turnovers']} ({s['n_turnovers_dead']} dead-ball)")
    print("duration on uncensored test episodes:", json.dumps(s["durations"]))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="phase.prep / apply_decisions npz")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--deep_oof", default=None, help="phase.train oof.npz on the same labels: adds its HSMM row")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buffer_s", type=float, default=15)
    ap.add_argument("--max_s", type=float, default=30)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--n_boot", type=int, default=2000, help="bootstrap replicates over eval runs (results.json ci)")
    ap.add_argument("--poss_confirm", default="first", choices=["first", "second"],
                    help="event possession: a gain counts at the first controlling event, or only once the gaining "
                         "team's second controlled action confirms it (codebook §2.4); both are reported")
    return ap


if __name__ == "__main__":
    print_table(run(vars(parser().parse_args())))
