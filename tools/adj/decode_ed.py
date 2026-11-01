"""Explicit-duration HSMM on OOF probs: per fold, durations fitted on training-region A/B labels."""
import argparse, sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('adj_dir', type=Path); ap.add_argument('run', type=Path)
ap.add_argument('--prefix', required=True, help='npz prefix: <prefix>_{adj,A,B}.npz, e.g. ajax_psv_full (phase.meta.prefix)')
args = ap.parse_args()
D, run = args.adj_dir, args.run
o = np.load(run / 'oof.npz'); hz = float(o['hz']); L1 = len(o['gt'])
cfg = json.loads((run / 'results.json').read_text())
buf = int(cfg['config']['buffer_s'] * hz)
A = np.load(D / f'{args.prefix}_A.npz')['label'].astype(int)[:L1]
B = np.load(D / f'{args.prefix}_B.npz')['label'].astype(int)[:L1]
logp = np.log(o['probs'] + 1e-6); alive = o['alive'].astype(bool)
out = {k: o['probs'].argmax(1) for k in ('edhsmm', 'edhsmm_hard')}
t0 = time.time()
for f in cfg['folds']:
    a, b = f['block']
    tr = np.ones(L1, bool); tr[max(0, a - buf):min(L1, b + buf)] = False
    dur = E.fit_durations([np.where(tr, A, -1), np.where(tr, B, -1)], hz)
    print(f"fold {f['fold']}: " + ', '.join(f"{E.K and k}: median {np.exp(p['mu']):.1f}s sd(log) {p['sigma']:.2f} (obs {p['n_obs']}, cens {p['n_cen']})" for k, p in dur.items()))
    for name, hard in (('edhsmm', False), ('edhsmm_hard', True)):
        out[name][a:b] = E.edhsmm_decode(logp[a:b], alive[a:b], dur, hz, cfg['config']['switch_penalty'], hard)
prev = dict(np.load(run / 'decoded.npz')) if (run / 'decoded.npz').exists() else {}
prev.update(out)
np.savez_compressed(run / 'decoded.npz', **prev)
print(f'decoded in {time.time() - t0:.0f}s; label changes', {k: int((np.diff(v) != 0).sum()) for k, v in out.items()}, 'hsmm', int((np.diff(o['hsmm']) != 0).sum()))
