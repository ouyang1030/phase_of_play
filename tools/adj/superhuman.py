"""Is a model better than a human coder? Protocol L1 / L2 / L3, on frames and on transition lengths.

  python tools/adj/superhuman.py ADJ_DIR --prefix PREFIX [name=RUN_DIR ...] [--l2 A=RUN_DIR B=RUN_DIR]
         [--third C.npz] [--events EVENTS --meta META --poss_confirm second] [--n_boot N] [--out NAME]

ADJ_DIR holds <prefix>_{adj,A,B}.npz (12.5 Hz). A run is a directory with oof.npz (pred of phase.duration, else
hsmm of phase.train), out of fold from frame 0 over its npz's labelled stretch; runs of different length (e.g. on
<prefix>_A.npz and <prefix>_B.npz) are compared on the common stretch.
L1  better than one coder: on frames both coders labelled, the model against B vs coder A against B, and the
    model against A vs coder B against A. The adjudicated labels are never the reference: where A and B agree
    they are A and B, so a coder would be right there by construction.
L2  better than its teacher (--l2): a run trained on coder A only (phase.duration --npz <prefix>_A.npz) against B
    vs A against B, and the run trained on B against A vs B against A. A model that only averages the two
    coders cannot pass L2: it never saw the reference coder.
L3  panel level (--third: an independent coder C, 12.5 Hz npz): the model against C vs A against C and B vs C,
    on frames all three labelled.
Frames: paired run-level bootstrap (phase.evaluate.bootstrap); the human row is the competing coder.
Transition lengths: per episode (event possession with --events, else the possession of the adjudicated labels)
where the competitor, the reference and the model each give an exact length (phase.episodes.observe; the model's
length is read off its predicted labels): |E_model - E_ref| vs |E_human - E_ref|, paired episode bootstrap.
Verdict per comparison and metric: 'exceeds' if the 95% interval of (model - human) for macro-F1 / kappa, or of
(human error - model error) for the length MAE, lies above 0; 'below' if it lies below 0; else 'no difference'.
Writes ADJ_DIR/<NAME>.json (default superhuman).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_ceiling import load_labels  # noqa: E402
from phase import episodes as P  # noqa: E402
from phase import evaluate as E  # noqa: E402
from phase import events as Ev  # noqa: E402
from phase import meta as M  # noqa: E402

FRAME_METRICS = ('macro_f1', 'kappa', 'transition_f1', 'bnd_f1@1s', 'bnd_typed_f1@2s')
VERDICT_METRICS = ('macro_f1', 'kappa')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('adj_dir', type=Path)
    ap.add_argument('runs', nargs='*', metavar='name=RUN_DIR', help='models trained on both coders / adjudicated')
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--l2', nargs='*', default=[], metavar='CODER=RUN_DIR', help='runs trained on one coder only')
    ap.add_argument('--third', type=Path, help='npz of an independent third coder (label, 12.5 Hz)')
    ap.add_argument('--events', help='event_data.csv: episodes from event possession (default: adjudicated labels)')
    ap.add_argument('--meta', help='meta.json (home team_id for the events)')
    ap.add_argument('--poss_confirm', default='first', choices=['first', 'second'])
    ap.add_argument('--max_s', type=float, default=30.0)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='superhuman')
    return ap.parse_args(argv)


def load_run(rd: Path) -> np.ndarray:
    o = np.load(Path(rd) / 'oof.npz')
    return o['pred'] if 'pred' in o.files else o['hsmm']


def episodes_for(args, adj_npz: dict, adj_label: np.ndarray, hz: float):
    alive, period = adj_npz['alive'].astype(bool), adj_npz['period']
    if args.events:
        home_id = M.load(args.meta)['home_team']['id'] if args.meta else None
        ev = Ev.load(args.events, adj_npz, home_team_id=home_id)
        poss, to = Ev.possession(ev, len(adj_label), confirm=args.poss_confirm)
        return P.episodes(poss, alive, period, hz, args.max_s, turnovers=to), 'events'
    poss = P.label_possession(adj_label)
    return P.episodes(poss, alive, period, hz, args.max_s), 'adjudicated labels'


def exact(obs) -> np.ndarray:
    return ((obs.lo == obs.hi) & (obs.lo < obs.W)).to_numpy()


def length_comparison(eps, model, human, ref, hz, n_boot, seed) -> dict:
    """Paired episode bootstrap of (human error - model error) in s, on episodes where all three are exact."""
    om, oh, orf = (P.observe(y, eps) for y in (model, human, ref))
    ok = exact(om) & exact(oh) & exact(orf)
    em = np.abs(om.lo.to_numpy() - orf.lo.to_numpy())[ok] / hz
    eh = np.abs(oh.lo.to_numpy() - orf.lo.to_numpy())[ok] / hz
    out = dict(n_episodes=int(ok.sum()))
    if not ok.any():
        return out
    idx = np.random.default_rng(seed).integers(0, len(em), (n_boot, len(em)))
    diff = eh[idx].mean(1) - em[idx].mean(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    out.update(mae_model_s=float(em.mean()), mae_human_s=float(eh.mean()), diff_s=float(eh.mean() - em.mean()),
               lo=float(lo), hi=float(hi), p_le_0=float(np.mean(diff <= 0)))
    return out


def verdict(lo, hi) -> str:
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return 'n/a'
    return 'exceeds' if lo > 0 else ('below' if hi < 0 else 'no difference')


def compare(name, model, human_name, human, ref, mask, eps, hz, n_boot, seed) -> dict:
    """The model and a competing human coder against one reference coder, on frames where mask holds."""
    gt = np.where(mask, ref, -1)
    bs = E.bootstrap(gt, {name: model, f'HUMAN {human_name}': human}, hz, n_boot=n_boot, seed=seed)
    d = bs['diff'][f'{name} - HUMAN {human_name}']
    frames = {m: dict(model=bs['ci'][name][m]['point'], human=bs['ci'][f'HUMAN {human_name}'][m]['point'],
                      diff=d[m]['diff'], lo=d[m]['lo'], hi=d[m]['hi'], p_le_0=d[m]['p_le_0']) for m in FRAME_METRICS}
    full = lambda y: np.where(mask, y, -1)  # noqa: E731  (lengths from the same frames)
    lengths = length_comparison(eps, model, full(human), full(ref), hz, n_boot, seed)
    v = {m: verdict(frames[m]['lo'], frames[m]['hi']) for m in VERDICT_METRICS}
    v['length_mae'] = verdict(lengths.get('lo'), lengths.get('hi'))
    return dict(n_frames=int((gt >= 0).sum()), frames=frames, lengths=lengths, verdict=v)


def main(argv=None):
    args = parse_args(argv)
    D = args.adj_dir
    lab, hz = load_labels(D, args.prefix)
    adj_npz = dict(np.load(D / f'{args.prefix}_adj.npz'))
    runs = {n: load_run(rd) for n, rd in (s.split('=', 1) for s in args.runs)}
    l2 = {c: load_run(rd) for c, rd in (s.split('=', 1) for s in args.l2)}
    lens = {len(p) for p in [*runs.values(), *l2.values()]}
    if not lens:
        raise ValueError('give at least one run')
    # every out-of-fold array starts at frame 0; runs on <prefix>_A / _B npz end at that coder's last labelled
    # frame, so compare on the common stretch
    L = min(lens)
    runs = {k: p[:L] for k, p in runs.items()}
    l2 = {k: p[:L] for k, p in l2.items()}
    A, B, ADJ = (lab[k][:L] for k in ('A', 'B', 'adj'))
    eps, ep_source = episodes_for(args, adj_npz, lab['adj'], hz)
    eps = eps[(eps.t1 <= L).to_numpy()].reset_index(drop=True)
    both = (A >= 0) & (B >= 0)
    out = dict(config=vars(args) | dict(adj_dir=str(D), third=str(args.third) if args.third else None),
               episodes_from=ep_source, n_episodes=int(len(eps)), n_frames_compared=int(L), human_ceiling=dict(
                   B_vs_A=E.evaluate(np.where(both, A, -1), B, hz)['macro_f1'],
                   A_vs_B=E.evaluate(np.where(both, B, -1), A, hz)['macro_f1']), L1={}, L2={}, L3={})
    for name, pr in runs.items():
        out['L1'][name] = {'vs B (competitor A)': compare(name, pr, 'A', A, B, both, eps, hz, args.n_boot, args.seed),
                           'vs A (competitor B)': compare(name, pr, 'B', B, A, both, eps, hz, args.n_boot, args.seed)}
    other = {'A': ('B', B), 'B': ('A', A)}
    for coder, pr in l2.items():
        if coder not in other:
            raise ValueError(f'--l2 coder must be A or B, got {coder!r}')
        ref_name, ref = other[coder]
        teacher = A if coder == 'A' else B
        out['L2'][f'trained on {coder}'] = {f'vs {ref_name} (competitor {coder})': compare(
            f'model_{coder}', pr, coder, teacher, ref, both, eps, hz, args.n_boot, args.seed)}
    if args.third:
        Cl = np.load(args.third)['label'].astype(int)[:L]
        all3 = both & (Cl >= 0)
        for name, pr in runs.items():
            out['L3'][name] = {f'vs C (competitor {h})': compare(name, pr, h, y, Cl, all3, eps, hz, args.n_boot,
                                                                 args.seed) for h, y in (('A', A), ('B', B))}
    report(out)
    (D / f'{args.out}.json').write_text(json.dumps(out, indent=1, default=float))
    print('wrote', D / f'{args.out}.json')
    return out


def report(out):
    print(f"human ceiling (macro-F1 on both-coded frames): B vs A {out['human_ceiling']['B_vs_A']:.3f}, "
          f"A vs B {out['human_ceiling']['A_vs_B']:.3f} | episodes from {out['episodes_from']} ({out['n_episodes']})")
    for level in ('L1', 'L2', 'L3'):
        for name, comps in out[level].items():
            for comp, r in comps.items():
                f, ln = r['frames']['macro_f1'], r['lengths']
                print(f"{level} {name:14s} {comp:22s} macro-F1 model {f['model']:.3f} human {f['human']:.3f} "
                      f"diff {f['diff']:+.3f} [{f['lo']:+.3f}, {f['hi']:+.3f}] | length MAE model "
                      f"{ln.get('mae_model_s', float('nan')):.2f}s human {ln.get('mae_human_s', float('nan')):.2f}s "
                      f"(n {ln['n_episodes']}) | {r['verdict']}")


if __name__ == '__main__':
    main()
