"""RQ1: how much of the four-moment labels is decided by possession? 2 x 2 error decomposition.

  python -m phase.decompose --npz NPZ --events EVENTS --out DIR [--meta META] [--poss_confirm second]
                            [name=RUN_DIR ...]

Labels = possession + one transition length per possession (phase.episodes). Cells:
  possession  labels: the possession the labels imply (oracle) | events: phase.events, as every model uses it
  lengths     oracle: best single change point per episode, read off the labels |
              timer: one tau for every episode, tuned per fold on the training frames (blocked CV) |
              name=RUN_DIR: out-of-fold lengths of a phase.duration run (its config 'poss' picks the row)
(labels, oracle) misses only labels that break the possession grammar; (events, oracle) adds the possession
errors; (labels, timer) adds only the length errors; (events, timer) is the full timer system. Differences are
paired run-level bootstrap intervals (E.bootstrap). Writes DIR/decompose.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import episodes as P
from . import evaluate as E
from . import events as Ev
from . import meta as M

COLS = ["macro_f1", "kappa", "transition_f1", "bnd_f1@1s", "bnd_typed_f1@2s", "acc"]


def possession_sources(d, label, events, hz, max_s, confirm="first", home_team_id=None):
    """{'labels': (poss, eps), 'events': (poss, eps)} over the whole match."""
    alive, period = d["alive"].astype(bool), d["period"]
    poss_l = P.label_possession(label)
    out = {"labels": (poss_l, P.episodes(poss_l, alive, period, hz, max_s))}
    if events is not None:
        ev = Ev.load(events, d, home_team_id=home_team_id)
        poss_e, to = Ev.possession(ev, len(label), confirm=confirm)
        out["events"] = (poss_e, P.episodes(poss_e, alive, period, hz, max_s, turnovers=to))
    return out


def cells(label, sources, hz, folds, buffer_frames):
    """Out-of-fold predictions per (possession, lengths) cell, over [L0, L1)."""
    L0, L1, cv = P.cv_folds(label, folds, buffer_frames)
    preds, taus = {}, {}
    for pname, (poss, eps) in sources.items():
        preds[f"{pname} | oracle"] = P.reconstruct(poss, eps, P.oracle_lengths(label, poss, eps))[L0:L1]
        timer = np.zeros(L1 - L0, int)
        taus[pname] = []
        for a, b, m in cv:
            tau, _ = P.tune_constant(P.EpisodeScorer(np.where(m, label, -1), poss, eps), hz)
            taus[pname].append(tau)
            timer[a - L0:b - L0] = P.reconstruct(poss, eps, np.full(len(eps), int(round(tau * hz))))[a:b]
        preds[f"{pname} | timer"] = timer
    return L0, L1, preds, taus


def run(cfg):
    d = dict(np.load(cfg["npz"]))
    hz = float(d["hz"])
    label = d["label"].astype(int)
    home_id = M.load(cfg["meta"])["home_team"]["id"] if cfg.get("meta") else None
    sources = possession_sources(d, label, cfg["events"], hz, cfg["max_s"], cfg["poss_confirm"], home_id)
    L0, L1, preds, taus = cells(label, sources, hz, cfg["folds"], int(cfg["buffer_s"] * hz))
    for extra in cfg.get("runs") or []:
        name, rd = extra.split("=", 1)
        rd = Path(rd)
        o = np.load(rd / "oof.npz")
        poss = json.loads((rd / "results.json").read_text())["config"].get("poss", "events")
        if len(o["pred"]) != L1 - L0:
            raise ValueError(f"{rd}: oof covers {len(o['pred'])} frames, the labels {L1 - L0}")
        preds[f"{poss} | {name}"] = o["pred"]
    gt = label[L0:L1]
    results = {k: E.evaluate(gt, p, hz) for k, p in preds.items()}
    ci = E.bootstrap(gt, preds, hz, n_boot=cfg["n_boot"])
    grammar = {}
    for pname, (poss, eps) in sources.items():
        obs = P.observe(label, eps)
        grammar[pname] = dict(episodes=int(len(eps)), empty_windows=int((P.lengths(eps) == 0).sum()),
                              no_information=int((obs.kind == "none").sum()),
                              exact=int((obs.kind == "exact").sum()), interval=int((obs.kind == "interval").sum()),
                              mean_lag_s=float(obs.lag.mean() / hz) if len(obs) else float("nan"))
    summary = dict(config=cfg, tau_per_fold=taus, episodes=grammar, results=results, ci=ci,
                   decomposition=decomposition(results, list(sources)))
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "decompose.json").write_text(json.dumps(summary, indent=1))
    return summary


def decomposition(results, sources, metric="macro_f1"):
    """Losses relative to (labels, oracle): possession = events|oracle, lengths = labels|timer, both = events|timer."""
    r = {k: v[metric] for k, v in results.items()}
    if "labels | oracle" not in r:
        return {}
    ref = r["labels | oracle"]
    out = {"metric": metric, "labels | oracle": ref, "grammar_loss": 1.0 - ref}
    if "events | oracle" in r:
        out["possession_loss"] = ref - r["events | oracle"]
    out["length_loss_timer"] = ref - r["labels | timer"]
    if "events | timer" in r:
        out["total_loss_timer"] = ref - r["events | timer"]
    return out


def print_table(s):
    print(f"{'possession | lengths':28s}" + "".join(f"{c:>16s}" for c in COLS))
    for k, r in s["results"].items():
        print(f"{k:28s}" + "".join(f"{r[c]:16.3f}" for c in COLS))
    print("decomposition:", json.dumps(s["decomposition"]))
    print("episodes:", json.dumps(s["episodes"]))


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", metavar="name=RUN_DIR", help="phase.duration runs (oof.npz pred)")
    ap.add_argument("--npz", required=True, help="labels: phase.prep / apply_decisions npz")
    ap.add_argument("--events", default=None, help="event_data.csv (without it only the labels row)")
    ap.add_argument("--meta", default=None, help="meta.json (home team_id for the event mapping)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poss_confirm", default="first", choices=["first", "second"])
    ap.add_argument("--max_s", type=float, default=30.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--buffer_s", type=float, default=15.0)
    ap.add_argument("--n_boot", type=int, default=2000)
    return ap


if __name__ == "__main__":
    print_table(run(vars(parser().parse_args())))
