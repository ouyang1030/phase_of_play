"""Evaluate predictions against adjudicated / raw A / raw B labels on shared frames + human ceiling (A vs B).

  python tools/adj/eval_ceiling.py ADJ_DIR --prefix PREFIX [--hazard DIR] [--deep DIR] [name=RUN_DIR ...]
                                   [--hz25] [--out NAME] [--n_boot N]

ADJ_DIR holds <prefix>_{adj,A,B}.npz (12.5 Hz labels; hz is read from the adj npz). Predictions (all optional):
--hazard DIR/preds.npz (timer_* and hazard_* rows of phase.hazard), --deep DIR/oof.npz (argmax, hsmm),
name=RUN_DIR (RUN_DIR/oof.npz: pred of a phase.duration run, else hsmm + RUN_DIR/decoded.npz). All prediction
arrays must have the same length.
--hz25 evaluates at 25 Hz (each prediction covers frames 2i and 2i+1, STUDY1_PLAN §7.3) against
labels_final_25hz.csv (codebook v1, apply_decisions.py: final / A / B; 4 = EX-SP, 5 = EX-DEAD, -1 = UNCODED are
unlabelled) or, if absent, labels_25hz.csv (v0, build_labels.py: label_adj / label_A / label_B).
Writes ADJ_DIR/<NAME>[_25hz].json = {"results": ..., "ci": ...}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E  # noqa: E402

COLS = ['n_frames', 'macro_f1', 'kappa', 'transition_f1', 'bnd_f1@1s', 'bnd_f1@2s', 'bnd_typed_f1@2s', 'seg_f1@0.5']
LABELS_25HZ = {'labels_final_25hz.csv': ('final', 'A', 'B'),  # v1 codes, moments 0-3 only
               'labels_25hz.csv': ('label_adj', 'label_A', 'label_B')}  # v0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('adj_dir', type=Path)
    ap.add_argument('runs', nargs='*', metavar='name=RUN_DIR', help='model runs (oof.npz hsmm + decoded.npz)')
    ap.add_argument('--prefix', required=True, help='npz prefix, e.g. ajax_psv_full (phase.meta.prefix)')
    ap.add_argument('--hazard', type=Path, help='phase.hazard output dir (preds.npz)')
    ap.add_argument('--deep', type=Path, help='phase.train output dir (oof.npz)')
    ap.add_argument('--hz25', action='store_true', help='evaluate at 25 Hz against the 25 Hz label CSV')
    ap.add_argument('--out', help='output name (default: <last run name>_eval, or eval_ceiling)')
    ap.add_argument('--n_boot', type=int, default=2000)
    return ap.parse_args(argv)


def load_labels(adj_dir: Path, prefix: str):
    """12.5 Hz labels {adj, A, B} and hz."""
    lab = {k: np.load(adj_dir / f'{prefix}_{k}.npz')['label'].astype(int) for k in ('adj', 'A', 'B')}
    return lab, float(np.load(adj_dir / f'{prefix}_adj.npz')['hz'])


def load_labels_25hz(adj_dir: Path):
    """25 Hz labels (adj, A, B) from the v1 or v0 CSV; everything but the four moments -> -1."""
    for name, cols in LABELS_25HZ.items():
        if (adj_dir / name).exists():
            L = pd.read_csv(adj_dir / name, usecols=list(cols))
            return tuple(np.where(np.isin(x, (0, 1, 2, 3)), x, -1) for x in (L[c].to_numpy(int) for c in cols))
    raise FileNotFoundError(f'{adj_dir}: neither {" nor ".join(LABELS_25HZ)}')


def load_preds(args) -> dict:
    preds, src = {}, {}
    if args.hazard:
        hp = np.load(args.hazard / 'preds.npz')
        for k in hp.files:
            if k.startswith(('timer_', 'hazard_')):
                preds[k], src[k] = hp[k], args.hazard
    if args.deep:
        o = np.load(args.deep / 'oof.npz')
        preds['deep_argmax'], preds['deep_hsmm'] = o['probs'].argmax(1), o['hsmm']
        src['deep_argmax'] = src['deep_hsmm'] = args.deep
    for extra in args.runs:
        name, rd = extra.split('=', 1)
        rd = Path(rd)
        o = np.load(rd / 'oof.npz')
        if 'pred' in o.files:  # phase.duration run: labels by construction, no decoder name
            preds[name], src[name] = o['pred'], rd
        else:
            preds[f'{name}_hsmm'], src[f'{name}_hsmm'] = o['hsmm'], rd
        if (rd / 'decoded.npz').exists():
            dz = np.load(rd / 'decoded.npz')
            for k in dz.files:
                preds[f'{name}_{k}'], src[f'{name}_{k}'] = dz[k], rd
    lens = {len(p) for p in preds.values()}
    if len(lens) > 1:
        raise ValueError('predictions differ in length: ' + ', '.join(f'{k} ({src[k]}): {len(p)}' for k, p in preds.items()))
    return preds


def evaluate_all(A, B, ADJ, preds, hz, n_boot=2000):
    both = (A >= 0) & (B >= 0)
    rows = {}

    def add(name, gt, pr, mask):
        rows[name] = E.evaluate(np.where(mask, gt, -1), np.where(pr >= 0, pr, 0), hz)

    add('HUMAN  B vs A  (both coded)', A, B, both)
    add('HUMAN  A vs B  (both coded)', B, A, both)
    for k, p in preds.items():
        add(f'{k:24s} vs ADJ (adj frames)', ADJ, p, ADJ >= 0)
        add(f'{k:24s} vs A   (both coded)', A, p, both)
        add(f'{k:24s} vs B   (both coded)', B, p, both)
    # bootstrap over eval runs of each reference, paired across methods (human = the other coder)
    pr0 = {k: np.where(p >= 0, p, 0) for k, p in preds.items()}
    ci = {'A (both coded)': E.bootstrap(np.where(both, A, -1), {'HUMAN B': B, **pr0}, hz, n_boot=n_boot),
          'B (both coded)': E.bootstrap(np.where(both, B, -1), {'HUMAN A': A, **pr0}, hz, n_boot=n_boot)}
    if preds:
        ci['ADJ (adj frames)'] = E.bootstrap(np.where(ADJ >= 0, ADJ, -1), pr0, hz, n_boot=n_boot)
    return rows, ci


def report(rows, A, B, preds):
    print(f"{'':52s}" + ''.join(f'{c:>16s}' for c in COLS))
    for k, r in rows.items():
        print(f'{k:52s}' + ''.join(f'{r[c]:16.3f}' if isinstance(r[c], float) else f'{r[c]:16d}' for c in COLS))
    print('\nper-class F1 [est_att, def->att, est_def, att->def]')
    for k, r in rows.items():
        if 'vs A ' in k or 'HUMAN' in k or 'ADJ' in k:
            print(f'  {k:48s} {np.round(r["f1_per_class"], 3).tolist()}')
    both = (A >= 0) & (B >= 0)
    dis = both & (A != B)  # disputed frames: how the model does where the coders disagree
    print(f'\ndisputed frames (both coded, A!=B): {dis.sum()} = {dis.sum() / max(both.sum(), 1):.1%} of both-coded')
    for k, p in preds.items():
        print(f'  {k:20s} agrees with A {np.mean(p[dis] == A[dis]):.2f} | with B {np.mean(p[dis] == B[dis]):.2f} | '
              f'neither {np.mean((p[dis] != A[dis]) & (p[dis] != B[dis])):.2f}')


def main(argv=None):
    args = parse_args(argv)
    D = args.adj_dir
    lab, hz = load_labels(D, args.prefix)
    preds = load_preds(args)
    L1 = len(next(iter(preds.values()))) if preds else len(lab['adj'])
    if L1 > len(lab['adj']):
        raise ValueError(f'predictions ({L1}) are longer than the labels ({len(lab["adj"])})')
    A, B, ADJ = (lab[k][:L1] for k in ('A', 'B', 'adj'))
    if args.hz25:  # report at 25 Hz: each 12.5 Hz prediction covers frames 2i and 2i+1 (STUDY1_PLAN §7.3)
        step = int(round(25.0 / hz))
        ADJ, A, B = (x[:step * L1] for x in load_labels_25hz(D))
        preds = {k: np.repeat(p, step)[:len(A)] for k, p in preds.items()}
        hz = 25.0
    rows, ci = evaluate_all(A, B, ADJ, preds, hz, args.n_boot)
    report(rows, A, B, preds)
    name = args.out or (args.runs[-1].split('=')[0] + '_eval' if args.runs else 'eval_ceiling')
    out = D / (name + ('_25hz' if args.hz25 else '') + '.json')
    out.write_text(json.dumps(dict(results=rows, ci=ci), indent=1))
    print('wrote', out)
    return rows, ci


if __name__ == '__main__':
    main()
