"""Provisional adjudicated labels (automatic part of the plan only) + cleaned raw A/B, as phase npz files.

Rules: dead ball -> masked; agreement -> label; disagreement run <= 1 s between agreed sides -> midpoint
(blips <= 1 s with equal sides -> side label); everything else -> masked (pending human adjudication).
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import prep

A_CSV, B_CSV, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
FPS, TAU, GAP = 25, 25, 5
U = -2  # not coded (dead ball or alive NONE)

def read_cats(p):
    df = pd.read_csv(p, low_memory=False, dtype=str, usecols=lambda c: c.rstrip(';') in ('Annotation_Category', 'ball_status'))
    df.columns = [c.rstrip(';') for c in df.columns]
    return df.Annotation_Category, df.ball_status

def clean(cat):
    s = cat.fillna('').str.strip()
    v = s.to_numpy(object)
    multi = np.array([';' in x for x in v])
    ch = np.flatnonzero(np.diff(np.r_[0, multi.astype(int), 0]))
    for a, b in zip(ch[::2], ch[1::2]):  # overlap "X; Y" -> split at midpoint
        first, second = [x.strip() for x in v[a].split(';')[:2]]
        mid = (a + b) // 2
        v[a:mid], v[mid:b] = first, second
    lab = pd.Series(v).map(prep.CATEGORY_MAP).fillna(U).astype(int).to_numpy().copy()
    i = 0
    while i < len(lab):  # bridge micro gaps <= GAP frames
        if lab[i] == U:
            j = i
            while j < len(lab) and lab[j] == U:
                j += 1
            if 0 < i and j < len(lab) and j - i <= GAP and lab[i - 1] >= 0:
                lab[i:j] = lab[i - 1]
            i = j
        else:
            i += 1
    return lab, int(multi.sum())

ca_raw, status = read_cats(A_CSV)
cb_raw, _ = read_cats(B_CSV)
dead = status.ne('alive').to_numpy()
ca, nA = clean(ca_raw); cb, nB = clean(cb_raw)
ca[dead] = U; cb[dead] = U

final = np.where(ca == cb, ca, -9)  # -9 = disagreement
src = np.where(ca == cb, np.where(ca >= 0, 'agree', 'uncoded'), 'pending').astype(object)
dis = final == -9
ch = np.flatnonzero(np.diff(np.r_[0, dis.astype(int), 0]))
clips = []
for s, e in zip(ch[::2], ch[1::2]):
    left = final[s - 1] if s > 0 else None
    right = final[e] if e < len(final) else None
    ok_sides = left is not None and right is not None
    within = ok_sides and set(ca[s:e]) <= {left, right} and set(cb[s:e]) <= {left, right}
    if e - s <= TAU and within:
        mid = (s + e) // 2
        final[s:mid], final[mid:e] = left, right
        src[s:e] = 'auto_mid'
    else:
        final[s:e] = U
        clips.append(dict(start=int(s), end=int(e - 1), dur_s=round((e - s) / FPS, 2),
                          A=sorted({int(x) for x in ca[s:e]}), B=sorted({int(x) for x in cb[s:e]})))
final[final == U] = -1

counts = pd.Series(src[~dead]).value_counts().to_dict()
alive_n = int((~dead).sum())
summary = dict(alive_frames=alive_n, sources_alive=counts,
               pending_clips=len(clips), pending_s=round(sum(c['dur_s'] for c in clips), 1),
               overlap_frames_split=dict(A=nA, B=nB),
               class_counts={prep.CLASSES[k]: int((final == k).sum()) for k in range(4)})
print(json.dumps(summary, indent=1))
(OUT / 'adj_summary.json').write_text(json.dumps(summary, indent=1))
pd.DataFrame(clips).to_csv(OUT / 'pending_clips.csv', index=False)
pd.DataFrame({'Frame': np.arange(len(final)), 'label_adj': final, 'src': src,
              'label_A': np.where(ca >= 0, ca, -1), 'label_B': np.where(cb >= 0, cb, -1)}).to_csv(OUT / 'labels_25hz.csv', index=False)

d = prep.load(A_CSV)  # tracking base = file A
idx = np.arange(0, len(final), prep.STEP)
for name, lab in (('adj', final), ('A', np.where(ca >= 0, ca, -1)), ('B', np.where(cb >= 0, cb, -1))):
    dd = dict(d)
    dd['label'] = lab[idx].astype(np.int8)
    dd['sub'] = np.where(dd['label'] >= 0, d['sub'], -1).astype(np.int8)
    np.savez_compressed(OUT / f'ajax_psv_full_{name}.npz', **dd)
    print(name, 'labeled @12.5Hz', int((dd['label'] >= 0).sum()), 'last', int(np.flatnonzero(dd['label'] >= 0)[-1]))
