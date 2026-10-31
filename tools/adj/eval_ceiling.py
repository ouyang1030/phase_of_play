"""Evaluate predictions against adjudicated / raw A / raw B labels on shared frames + human ceiling (A vs B).

  python tools/adj/eval_ceiling.py ADJ_DIR [name=run_dir ...] [--hz25]
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E

HZ25 = '--hz25' in sys.argv  # report at 25 Hz: each 12.5 Hz prediction covers frames 2i and 2i+1 (STUDY1_PLAN §7.3)
sys.argv = [a for a in sys.argv if a != '--hz25']
D = Path(sys.argv[1])
lab = {k: np.load(D / f'ajax_psv_full_{k}.npz')['label'].astype(int) for k in ('adj', 'A', 'B')}
hz = 12.5
hp = np.load(D / 'hazard' / 'preds.npz')
L1 = len(hp['gt'])
preds = {k: hp[k] for k in ('timer_taustar', 'timer_taustar_live', 'hazard_time', 'hazard_org_all')}
deep = D / 'deep' / 'oof.npz'
if deep.exists():
    o = np.load(deep)
    assert len(o['gt']) == L1
    preds['deep_argmax'] = o['probs'].argmax(1)
    preds['deep_hsmm'] = o['hsmm']
for extra in sys.argv[2:]:  # name=run_dir : adds <run_dir>/oof.npz hsmm + decoded.npz
    name, rd = extra.split('=')
    rd = Path(rd)
    o = np.load(rd / 'oof.npz'); assert len(o['gt']) == L1
    preds[f'{name}_hsmm'] = o['hsmm']
    if (rd / 'decoded.npz').exists():
        dz = np.load(rd / 'decoded.npz')
        for k in dz.files:
            preds[f'{name}_{k}'] = dz[k]
A, B, ADJ = (lab[k][:L1] for k in ('A', 'B', 'adj'))
if HZ25:
    import pandas as pd
    L25 = pd.read_csv(D / 'labels_25hz.csv')
    A, B, ADJ = (L25[c].to_numpy()[:2 * L1] for c in ('label_A', 'label_B', 'label_adj'))
    preds = {k: np.repeat(p, 2)[:len(A)] for k, p in preds.items()}
    hz = 25.0
both = (A >= 0) & (B >= 0)
cols = ['n_frames', 'macro_f1', 'kappa', 'transition_f1', 'bnd_f1@1s', 'bnd_f1@2s', 'seg_f1@0.5']
rows = {}
def add(name, gt, pr, mask):
    rows[name] = E.evaluate(np.where(mask, gt, -1), np.where(pr >= 0, pr, 0), hz)
add('HUMAN  B vs A  (both coded)', A, B, both)
add('HUMAN  A vs B  (both coded)', B, A, both)
for k, p in preds.items():
    add(f'{k:24s} vs ADJ (adj frames)', ADJ, p, ADJ >= 0)
    add(f'{k:24s} vs A   (both coded)', A, p, both)
    add(f'{k:24s} vs B   (both coded)', B, p, both)
print(f"{'':52s}" + ''.join(f'{c:>14s}' for c in cols))
for k, r in rows.items():
    print(f'{k:52s}' + ''.join(f'{r[c]:14.3f}' if isinstance(r[c], float) else f'{r[c]:14d}' for c in cols))
print('\nper-class F1 [est_att, def->att, est_def, att->def]')
for k, r in rows.items():
    if 'vs A ' in k or 'HUMAN' in k or 'ADJ' in k:
        print(f'  {k:48s} {np.round(r["f1_per_class"], 3).tolist()}')
# disputed frames: how the model does where the coders disagree
dis = both & (A != B)
print(f'\ndisputed frames (both coded, A!=B): {dis.sum()} = {dis.sum()/both.sum():.1%} of both-coded')
for k, p in preds.items():
    print(f'  {k:20s} agrees with A {np.mean(p[dis]==A[dis]):.2f} | with B {np.mean(p[dis]==B[dis]):.2f} | neither {np.mean((p[dis]!=A[dis])&(p[dis]!=B[dis])):.2f}')
name = sys.argv[-1].split('=')[0] + '_eval' if len(sys.argv) > 2 else 'eval_ceiling'
(D / (name + ('_25hz' if HZ25 else '') + '.json')).write_text(json.dumps(rows, indent=1))
