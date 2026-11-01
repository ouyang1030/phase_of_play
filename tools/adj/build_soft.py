"""adj npz + soft targets: frames coded by both A and B -> mean one-hot; anything else -> zeros (masked)."""
import argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('adj_dir', type=Path)
ap.add_argument('--prefix', required=True, help='npz prefix: <prefix>_{adj,A,B}.npz, e.g. ajax_psv_full (phase.meta.prefix)')
args = ap.parse_args()
D = args.adj_dir
d = dict(np.load(D / f'{args.prefix}_adj.npz'))
A = np.load(D / f'{args.prefix}_A.npz')['label'].astype(int)
B = np.load(D / f'{args.prefix}_B.npz')['label'].astype(int)
both = (A >= 0) & (B >= 0)
soft = np.zeros((len(A), 4), np.float32)
for L in (A, B):
    soft[both, L[both]] += 0.5
d['label_soft'] = soft
d['label_A'] = np.where(both, A, -1).astype(np.int8)
d['label_B'] = np.where(both, B, -1).astype(np.int8)
np.savez_compressed(D / f'{args.prefix}_soft.npz', **d)
adj = d['label'].astype(int)
print('soft-coded frames', int(both.sum()), '| adj frames', int((adj >= 0).sum()),
      '| class mass soft', soft.sum(0).round(0).tolist(), '| adj counts', np.bincount(adj[adj >= 0], minlength=4).tolist())
