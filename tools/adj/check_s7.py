"""Data checks for STUDY1_PLAN §7 assumptions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import evaluate as E
D = Path(sys.argv[1]); FPS = 25
L = pd.read_csv(D / 'labels_25hz.csv')
d = np.load(D / 'ajax_psv_full_adj.npz')
alive25 = np.repeat(d['alive'], 2)[:len(L)]
names = ['ATT', 'D>A', 'DEF', 'A>D']

print('== (i) class switches: direct (adjacent labelled frames) vs across stoppage')
for c in ('label_A', 'label_B'):
    y = L[c].to_numpy()
    lab = np.flatnonzero(y >= 0)
    prev, nxt = lab[:-1], lab[1:]
    sw = y[prev] != y[nxt]
    gap = nxt - prev - 1
    dead_between = np.array([(~alive25[a + 1:b]).any() if b > a + 1 else False for a, b in zip(prev[sw], nxt[sw])])
    pairs = pd.Series([f'{names[y[a]]}→{names[y[b]]}' for a, b in zip(prev[sw], nxt[sw])])
    direct = (gap[sw] <= FPS) & ~dead_between
    tab = pd.DataFrame({'pair': pairs, 'direct': direct}).groupby('pair').direct.agg(direct='sum', total='size')
    tab['via_stoppage_or_gap'] = tab.total - tab.direct
    print(c); print(tab.sort_values('total', ascending=False).to_string())

print('\n== (ii) durations (runs inside alive play, both ends observed vs censored by stoppage/NONE)')
for c in ('label_A', 'label_B'):
    y = L[c].to_numpy()
    ch = np.flatnonzero(np.diff(np.r_[-9, y, -9]) != 0)
    for k in (1, 3):
        rows = []
        for a, b in zip(ch[:-1], ch[1:]):
            if y[a] != k: continue
            obs = a > 0 and b < len(y) and y[a - 1] >= 0 and y[b] >= 0
            rows.append(((b - a) / FPS, obs))
        r = pd.DataFrame(rows, columns=['dur', 'obs'])
        o = r[r.obs].dur
        print(f'{c} {names[k]}: n={len(r)} observed={len(o)} median={o.median():.1f}s IQR=({o.quantile(.25):.1f},{o.quantile(.75):.1f}) p90={o.quantile(.9):.1f} max={o.max():.1f} | log-dur sd={np.log(o).std():.2f}')

print('\n== (iii) mark (1:1 Hungarian outfield assignment) vs press (kNN k=3, both directions)')
pos, gk, alive = d['pos'], d['is_gk'], d['alive']
idx = np.flatnonzero(alive)[::25]
inside = tot = 0; nn_same = 0
for t in idx:
    P0, P1 = pos[t, 0], pos[t, 1]
    v0, v1 = ~np.isnan(P0[:, 0]), ~np.isnan(P1[:, 0])
    if v0.sum() < 11 or v1.sum() < 11: continue
    Dm = np.linalg.norm(P0[:, None] - P1[None], axis=-1)
    kth0 = np.sort(Dm, 1)[:, 2:3]; kth1 = np.sort(Dm, 0)[2:3, :]
    press = (Dm <= kth0) | (Dm <= kth1)
    o0, o1 = np.flatnonzero(~gk[t, 0]), np.flatnonzero(~gk[t, 1])
    r, c = linear_sum_assignment(Dm[np.ix_(o0, o1)])
    inside += press[o0[r], o1[c]].sum(); tot += len(r)
    nn_same += (Dm[o0[r]].argmin(1) == o1[c]).sum()
print(f'frames {len(idx)}: mark edges inside press set {inside / tot:.1%}; mark partner = nearest opponent {nn_same / tot:.1%}')

print('\n== (iv) evaluation at 12.5 Hz vs 25 Hz (soft model HSMM, vs A, both-coded)')
o = np.load(D / 'deep_soft' / 'oof.npz'); L1 = len(o['gt'])
A12 = np.load(D / 'ajax_psv_full_A.npz')['label'].astype(int)[:L1]
B12 = np.load(D / 'ajax_psv_full_B.npz')['label'].astype(int)[:L1]
m12 = (A12 >= 0) & (B12 >= 0)
r12 = E.evaluate(np.where(m12, A12, -1), o['hsmm'], 12.5)
A25, B25 = L.label_A.to_numpy()[:2 * L1], L.label_B.to_numpy()[:2 * L1]
m25 = (A25 >= 0) & (B25 >= 0)
r25 = E.evaluate(np.where(m25, A25, -1), np.repeat(o['hsmm'], 2)[:len(A25)], 25.0)
for k in ('macro_f1', 'kappa', 'bnd_f1@0.5s', 'bnd_f1@1s', 'bnd_f1@2s', 'seg_f1@0.5'):
    print(f'  {k:12s} 12.5 Hz {r12[k]:.3f} | 25 Hz {r25[k]:.3f}')
