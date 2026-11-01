"""Apply the third adjudicator's decisions -> final labels, intervals, training npz, QC report (STUDY1_PLAN §4.7).

  python tools/adj/apply_decisions.py ADJ_DIR DECISIONS_CSV TRACKING_NPZ OUT_DIR META_JSON

ADJ_DIR holds labels_v1_25hz.csv, clips.csv and key_DO_NOT_SHARE.json (from codebook_v1.py / make_pack.py).
TRACKING_NPZ is a 12.5 Hz phase npz whose tracking arrays are reused (labels are replaced).
META_JSON (phase.meta) gives the prefix of the output npz files: <prefix>_{adj,A,B,soft}.npz.
Subcategories (subA / subB of labels_v1_25hz.csv): the final subcategory is kept only where the final label is a
moment that both coders gave the frame and both chose the same subcategory, else -1. The npz 'sub' arrays come
from these (adj / soft: final, A: subA, B: subB), not from the tracking npz.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook_v1 as C  # noqa: E402
from phase import meta as M  # noqa: E402  (codebook_v1 puts the repo root on sys.path)

CODE = {v: k for k, v in C.NAMES.items()}
REASONS = {'C1', 'C2', 'C3', 'C4', 'P', 'X', 'O'}  # codebook v4.1: §2.5 criteria, §2.4 possession, §2.2.3 EX


def parse_other(text, s, e):
    """'TD@51200; OP@51390' -> codes for frames s..e (inclusive)."""
    out = np.full(e - s + 1, C.PENDING)
    parts = [p.strip() for p in str(text).split(';') if p.strip()]
    marks = []
    for p in parts:
        name, f = p.rsplit('@', 1)
        if name.strip() not in CODE or CODE[name.strip()] not in (0, 1, 2, 3, C.SP):
            raise ValueError(f'unknown label {name!r}')
        marks.append((int(f), CODE[name.strip()]))
    if not marks or marks[0][0] != s:
        raise ValueError(f'first segment must start at start_frame {s}')
    fr = [m[0] for m in marks]
    if fr != sorted(fr) or len(set(fr)) != len(fr) or fr[-1] > e:
        raise ValueError('segment frames must increase and stay inside the clip')
    for (f, c), nxt in zip(marks, fr[1:] + [e + 1]):
        out[f - s:nxt - s] = c
    return out


def check_sub(label, sub, what):
    """Every subcategory must belong to its frame's moment (SUBCATS 0-2 -> 0, 3-4 -> 1, 5-7 -> 2, 8-9 -> 3)."""
    has = sub >= 0
    bad = has & ((label < 0) | (label > 3) | (C.SUB_CATEGORY[np.clip(sub, 0, len(C.SUB_CATEGORY) - 1)] != label))
    if bad.any() or (sub >= len(C.SUB_CATEGORY)).any():
        i = np.flatnonzero(bad | (sub >= len(C.SUB_CATEGORY)))
        raise AssertionError(f'{what}: subcategory does not belong to the label on {len(i)} frames, '
                             f'first at {i[0]} (label {label[i[0]]}, sub {sub[i[0]]})')


def final_sub(final, a, b, sub_a, sub_b):
    """Final subcategory: the coders' common one where the final moment equals both coders' moment, else -1."""
    keep = (final >= 0) & (final <= 3) & (final == a) & (final == b) & (sub_a == sub_b) & (sub_a >= 0)
    return np.where(keep, sub_a, -1)


def read_subs(L):
    """subA / subB of labels_v1_25hz.csv; -1 (with a warning) if they are missing (codebook_v1 before item #9)."""
    if {'subA', 'subB'} <= set(L.columns):
        return L.subA.to_numpy(int), L.subB.to_numpy(int)
    warnings.warn('labels_v1_25hz.csv has no subA / subB (re-run codebook_v1.py convert): all subcategories -1')
    return np.full(len(L), -1), np.full(len(L), -1)


def main(adj_dir, decisions_csv, npz, out_dir, meta_json):
    prefix = M.prefix(M.load(meta_json))
    adj, out = Path(adj_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(adj / 'labels_v1_25hz.csv')
    clips = pd.read_csv(adj / 'clips.csv').set_index('clip_id')
    key = json.loads((adj / 'key_DO_NOT_SHARE.json').read_text())
    dec = pd.read_csv(decisions_csv, dtype=str).fillna('')
    dec['decision'] = dec.decision.str.strip().str.upper()
    dec['reason_code'] = dec.reason_code.str.strip().str.upper()

    final, src = L.final.to_numpy().copy(), L.src.to_numpy(object).copy()
    clip_of = np.full(len(final), '', object)
    lab = {'A': L.A.to_numpy(), 'B': L.B.to_numpy()}
    sub_a, sub_b = read_subs(L)
    check_sub(lab['A'], sub_a, 'subA')
    check_sub(lab['B'], sub_b, 'subB')
    errors = []
    missing = sorted(set(clips.index) - set(dec.clip_id))
    if missing:
        errors.append(f'no decision for {len(missing)} clips: {missing[:10]}')
    for r in dec.itertuples():
        if r.clip_id not in clips.index:
            errors.append(f'{r.clip_id}: unknown clip')
            continue
        c = clips.loc[r.clip_id]
        s, e = int(c.start), int(c.end)
        try:
            if r.decision in ('R1', 'R2'):
                seg = lab[key[r.decision]][s:e + 1]
                if (seg == C.UNCODED).any():
                    raise ValueError(f'{r.decision} has UNCODED frames here; use OTHER')
            elif r.decision == 'OTHER':
                seg = parse_other(r.other_segments, s, e)
            else:
                raise ValueError(f'decision must be R1, R2 or OTHER, got {r.decision!r}')
            if r.reason_code not in REASONS:
                raise ValueError(f'reason_code must be one of {sorted(REASONS)}')
            if r.reason_code == 'O' and not r.note.strip():
                raise ValueError('reason_code O needs a note')
        except ValueError as ex:
            errors.append(f'{r.clip_id}: {ex}')
            continue
        final[s:e + 1], src[s:e + 1], clip_of[s:e + 1] = seg, 'adjudicated', r.clip_id
    if errors:
        print('NOT APPLIED — fix decisions first:\n  ' + '\n  '.join(errors))
        sys.exit(1)

    sub = final_sub(final, lab['A'], lab['B'], sub_a, sub_b)
    check_sub(final, sub, 'final')
    alive = L.alive.to_numpy().astype(bool)
    short = [(int(s), int(e - s)) for c_ in (0, 1, 2, 3) for s, e in C.runs(final == c_)
             if e - s < C.MIN_SEG and C.INACTIVE not in (final[max(s - 1, 0)], final[min(e, len(final) - 1)])]
    q = np.r_[np.flatnonzero(np.diff(final) != 0) + 1]
    est_switch = sum(1 for i in q if {final[i - 1], final[i]} == {0, 2})
    checks = dict(pending_left=int((final == C.PENDING).sum()),
                  uncoded_alive=int(((final == C.UNCODED) & alive).sum()),
                  inactive_mismatch=int(((final == C.INACTIVE) != ~alive).sum()),
                  short_segments=short,
                  direct_est_switches=est_switch,  # codebook question, reported not enforced
                  moment_with_subcategory=round(float((sub >= 0).sum() / max(((final >= 0) & (final <= 3)).sum(), 1)), 3))

    # adjudicator vs coders (blinded names resolved only here)
    d_ = dec.set_index('clip_id')
    picked = d_.decision.map({'R1': key['R1'], 'R2': key['R2']}).fillna('OTHER')
    share = picked.value_counts().to_dict()
    ab = share.get('A', 0) + share.get('B', 0)
    qc = dict(decisions=share,
              bias_flag=bool(ab and max(share.get('A', 0), share.get('B', 0)) / ab > 0.7),
              by_type=pd.crosstab(clips.loc[d_.index, 'type'], picked).to_dict(),
              reason_codes=d_.reason_code.value_counts().to_dict(),
              confidence=d_.confidence.value_counts().to_dict(),
              checks=checks)
    print(json.dumps(qc, indent=1, ensure_ascii=False))
    (out / 'qc.json').write_text(json.dumps(qc, indent=1, ensure_ascii=False))

    pd.DataFrame({'Frame': L.Frame, 'final': final, 'final_name': [C.NAMES.get(int(x), '?') for x in final],
                  'src': src, 'clip_id': clip_of, 'A': L.A, 'B': L.B, 'sub': sub,
                  'sub_name': [C.SUBCATS[x] if x >= 0 else '' for x in sub], 'subA': sub_a, 'subB': sub_b}
                 ).to_csv(out / 'labels_final_25hz.csv', index=False)
    ch = np.r_[0, np.flatnonzero(np.diff(final) != 0) + 1, len(final)]
    iv = [dict(seg_id=i, start_frame=int(a), end_frame=int(b - 1), category=C.NAMES[int(final[a])],
               src='/'.join(sorted(set(src[a:b]))), clip_id='/'.join(sorted(set(clip_of[a:b]) - {''})))
          for i, (a, b) in enumerate(zip(ch[:-1], ch[1:]))]
    pd.DataFrame(iv).to_csv(out / 'intervals.csv', index=False)

    # training npz (12.5 Hz): 4 moments, everything else masked; soft = A/B vote shares where both coded a moment
    d = dict(np.load(npz))
    step = int(round(C.FPS / float(d['hz'])))
    idx = np.arange(0, len(final), step)[:len(d['label'])]
    moment = lambda x: np.where((x >= 0) & (x <= 3), x, -1)  # noqa: E731
    A12, B12, F12 = (moment(x[idx]) for x in (lab['A'], lab['B'], final))
    sA12, sB12, sF12 = (np.where(x >= 0, s_[idx], -1) for x, s_ in ((A12, sub_a), (B12, sub_b), (F12, sub)))
    for name, x, s_ in (('adj', F12, sF12), ('A', A12, sA12), ('B', B12, sB12)):
        check_sub(x, s_, f'{name} npz')
        dd = dict(d)
        dd['label'] = x.astype(np.int8)
        dd['sub'] = s_.astype(np.int8)
        np.savez_compressed(out / f'{prefix}_{name}.npz', **dd)
    both = (A12 >= 0) & (B12 >= 0)
    soft = np.zeros((len(F12), 4), np.float32)
    for x in (A12, B12):
        soft[both, x[both]] += 0.5
    dd = dict(d)
    dd.update(label=F12.astype(np.int8), sub=sF12.astype(np.int8), label_soft=soft,
              label_A=np.where(both, A12, -1).astype(np.int8), label_B=np.where(both, B12, -1).astype(np.int8))
    np.savez_compressed(out / f'{prefix}_soft.npz', **dd)
    print('wrote', out)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:6])
