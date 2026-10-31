"""adj npz + soft targets: frames coded by both A and B -> mean one-hot; anything else -> zeros (masked)."""
import sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1])
d = dict(np.load(D / 'ajax_psv_full_adj.npz'))
A = np.load(D / 'ajax_psv_full_A.npz')['label'].astype(int)
B = np.load(D / 'ajax_psv_full_B.npz')['label'].astype(int)
both = (A >= 0) & (B >= 0)
soft = np.zeros((len(A), 4), np.float32)
for L in (A, B):
    soft[both, L[both]] += 0.5
d['label_soft'] = soft
d['label_A'] = np.where(both, A, -1).astype(np.int8)
d['label_B'] = np.where(both, B, -1).astype(np.int8)
np.savez_compressed(D / 'ajax_psv_full_soft.npz', **d)
adj = d['label'].astype(int)
print('soft-coded frames', int(both.sum()), '| adj frames', int((adj >= 0).sum()),
      '| class mass soft', soft.sum(0).round(0).tolist(), '| adj counts', np.bincount(adj[adj >= 0], minlength=4).tolist())
