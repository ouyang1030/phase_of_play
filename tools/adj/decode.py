"""Possession-constrained decoding of deep OOF probabilities (two pre-registered settings)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E
run = Path(sys.argv[1])
o = np.load(run / 'oof.npz')
logp = np.log(o['probs'] + 1e-6)
hz = float(o['hz'])
out = {'possdec_strict': E.possession_decode(logp, o['poss_home'], hz, 0.0, 0.0),
       'possdec_lag': E.possession_decode(logp, o['poss_home'], hz, 1.0, 5.0),
       'asrf': E.boundary_refine(o['probs'], o['bnd'], 0.5),
       'asrf_alive': E.boundary_refine(o['probs'], o['bnd'], 0.5, o['alive'].astype(bool))}
b = o['bnd']
pk = np.r_[False, (b[1:-1] > b[:-2]) & (b[1:-1] >= b[2:]), False]
print('bnd quantiles', np.quantile(b, [.5, .9, .99, 1]).round(3).tolist(), '| peaks >0.5:', int((pk & (b > .5)).sum()),
      '| peaks >0.3:', int((pk & (b > .3)).sum()), '| GT bnd (adj, raw changes):', int((np.diff(o['gt'][o['gt'] >= 0]) != 0).sum()))
np.savez_compressed(run / 'decoded.npz', **out)
print({k: int((np.diff(v) != 0).sum()) for k, v in out.items()}, 'argmax changes', int((np.diff(o['probs'].argmax(1)) != 0).sum()))
