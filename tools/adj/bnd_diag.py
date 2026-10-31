"""Boundary head as a detector: peaks vs GT boundaries of A (both-coded frames), ASRF-style threshold sweep for diagnosis only."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E
D, run = Path(sys.argv[1]), Path(sys.argv[2])
o = np.load(run / 'oof.npz'); hz = float(o['hz']); L1 = len(o['gt'])
A = np.load(D / 'ajax_psv_full_A.npz')['label'].astype(int)[:L1]
B = np.load(D / 'ajax_psv_full_B.npz')['label'].astype(int)[:L1]
both = (A >= 0) & (B >= 0)
gA = np.where(both, A, -1)
runs = E.eval_runs(gA, max_gap=int(hz))
gb = E.boundaries(gA, runs)
inside = np.zeros(L1, bool)
for a, b in runs: inside[a:b] = True
bnd = o['bnd']
pk = np.r_[False, (bnd[1:-1] > bnd[:-2]) & (bnd[1:-1] >= bnd[2:]), False] & inside
hsmm_b = E.boundaries(np.where(both, o['hsmm'], -1), runs)
for tol in (1, 2):
    t = int(tol * hz)
    print(f'±{tol}s  HSMM label changes: P/R/F1 =', np.round(E.boundary_f1(gb, hsmm_b, t), 3).tolist(), f'(n={len(hsmm_b)})')
    for thr in (0.5, 0.3, 0.1):
        pb = np.flatnonzero(pk & (bnd > thr))
        print(f'±{tol}s  bnd peaks >{thr}: P/R/F1 =', np.round(E.boundary_f1(gb, pb, t), 3).tolist(), f'(n={len(pb)})')
print('GT boundaries (A, both-coded runs):', len(gb))
# boundary prob at GT boundaries
mx = [bnd[max(0, g - int(hz)):g + int(hz) + 1].max() for g in gb]
print('max bnd within ±1 s of GT boundary: quantiles', np.quantile(mx, [.1, .25, .5, .75]).round(3).tolist())
