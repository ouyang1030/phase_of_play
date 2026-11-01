"""Codebook v1: convert the two Ajax–PSV fused CSVs, list clips for adjudication, validate exports.

Codes (25 Hz): 0 Attacking, 1 Defending Transition (ball won), 2 Defending, 3 Attacking Transition (ball lost),
4 SET_PIECE, 5 INACTIVE (ball_status == dead), -1 UNCODED (alive, no label).
SET_PIECE is never annotated: coders leave the timeline empty after a corner or a free kick <= 40 m from goal
until open play starts, and that leading blank is coded SET_PIECE (STUDY1_PLAN §4.1). Any other alive frame
must carry a moment label.

Conversion of the already-coded match (STUDY1_PLAN §4.7), per coder:
  clean (strip, split "X; Y" overlaps at the midpoint, bridge gaps <= 5 frames) -> dead frames = INACTIVE
  -> gap from the onset of a set-piece restart (corner, free kick <= 40 m from goal) = SET_PIECE
  -> gap <= 1 s after any other restart = the following label (reaction lag)
  -> gap <= 1 s between two labels = split at its midpoint (button-switch gap)
  -> gap <= 1 s right before a dead ball = the preceding label -> other alive gaps = UNCODED.
Merge: agreement -> label; disagreement <= 1 s between agreed sides -> midpoint; everything else, plus
alive frames neither coder labelled -> clip for the blinded third adjudicator.
Subcategories (Annotation_Subcategory, index into phase.prep.SUBCATS) follow every step of their coder's moment
label (subA / subB in labels_v1_25hz.csv); one that belongs to another moment than the frame's label is dropped (-1).

  python tools/adj/codebook_v1.py convert A.csv B.csv events.csv tracking.npz OUT_DIR
  python tools/adj/codebook_v1.py validate export.csv events.csv tracking.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import restarts as R  # noqa: E402
from phase.prep import SUBCATS  # noqa: E402

FPS, TAU, GAP, LAG = 25, 25, 5, 25
SP, INACTIVE, UNCODED, PENDING = 4, 5, -1, -9
# codebook v4.1 names: IP / TA / OP / TD; EX split into set-piece play (EX-SP) and dead ball (EX-DEAD)
NAMES = {0: 'IP', 1: 'TA', 2: 'OP', 3: 'TD', SP: 'EX-SP', INACTIVE: 'EX-DEAD', UNCODED: 'UNCODED'}
CATEGORY = {'Attacking': 0, 'Defending Transition': 1, 'Defending': 2, 'Attacking Transition': 3,
            'Inactive': INACTIVE}
SP_FK_X = 12.5  # free kicks with the ball >= 12.5 m into the opponent half (<= 40 m from the goal line)
MIN_SEG = int(0.5 * FPS)
SUB_ID = {name: i for i, name in enumerate(SUBCATS)}
SUB_CATEGORY = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])  # moment of each subcategory (0-2 -> 0, ..., 8-9 -> 3)
assert len(SUB_CATEGORY) == len(SUBCATS)


def read_cats(path):
    """(Annotation_Category, Annotation_Subcategory, alive) of one coder's fused CSV; a missing subcategory
    column gives empty strings."""
    df = pd.read_csv(path, low_memory=False, dtype=str,
                     usecols=lambda c: c.rstrip(';') in ('Annotation_Category', 'Annotation_Subcategory', 'ball_status'))
    df.columns = [c.rstrip(';') for c in df.columns]
    sub = df.Annotation_Subcategory if 'Annotation_Subcategory' in df else pd.Series([''] * len(df))
    return df.Annotation_Category, sub, df.ball_status.eq('alive').to_numpy()


def runs(mask):
    ch = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(ch[::2], ch[1::2]))


def split_overlaps(text):
    """Strings with overlapping clips "X; Y" -> X before the run's midpoint, Y after; (values, overlap mask)."""
    v = text.fillna('').str.strip().to_numpy(object).copy()
    multi = np.array([';' in x for x in v], bool)
    for a, b in runs(multi):
        first, second = [x.strip() for x in v[a].split(';')[:2]]
        mid = (a + b) // 2
        v[a:mid], v[mid:b] = first, second
    return v, multi


def clean(cat, sub=None):
    """Raw category (and subcategory) strings -> codes with UNCODED gaps.
    Returns (codes, n overlap frames, subcategory codes: index into SUBCATS, -1 = none / unknown)."""
    v, multi = split_overlaps(cat)
    lab = pd.Series(v).map(CATEGORY).fillna(UNCODED).astype(int).to_numpy().copy()
    sc = np.full(len(lab), -1)
    if sub is not None:
        sc = pd.Series(split_overlaps(sub)[0]).map(SUB_ID).fillna(-1).astype(int).to_numpy().copy()
    for a, b in runs(lab == UNCODED):  # bridge micro gaps
        if a > 0 and b < len(lab) and b - a <= GAP and lab[a - 1] >= 0:
            lab[a:b], sc[a:b] = lab[a - 1], sc[a - 1]
    return lab, int(multi.sum()), sc


def consistent_sub(lab, sub):
    """Subcategories that belong to the frame's moment label; others -> -1. Returns (sub, n frames dropped)."""
    ok = (sub >= 0) & (lab >= 0) & (lab <= 3) & (SUB_CATEGORY[np.clip(sub, 0, len(SUBCATS) - 1)] == lab)
    return np.where(ok, sub, -1), int(((sub >= 0) & ~ok).sum())


def is_set_piece(r):
    return r.kind == 'corner' or (r.kind == 'free_kick' and r.zone != '' and r.x_att >= SP_FK_X)


def to_v1(lab, alive, rs, sub=None):
    """Codes after clean() -> codebook v1 (module docstring). The subcategory codes (if given) are copied along
    with every filled label. Returns (codes, source per frame, subcategory codes)."""
    lab = lab.copy()
    sub = np.full(len(lab), -1) if sub is None else sub.copy()
    lab[~alive], sub[~alive] = INACTIVE, -1
    src = np.full(len(lab), '', object)
    for r in rs.itertuples():
        s, t = r.onset, r.end
        j = np.flatnonzero(lab[s:t] != UNCODED)
        first = s + (j[0] if len(j) else t - s)
        if first == s:
            continue
        if is_set_piece(r):
            lab[s:first], sub[s:first], src[s:first] = SP, -1, 'sp_gap'
        elif first - s <= LAG and first < t:
            lab[s:first], sub[s:first], src[s:first] = lab[first], sub[first], 'restart_lag'
    for s, e in runs(lab == UNCODED):
        if not (0 < s and e < len(lab) and e - s <= TAU and 0 <= lab[s - 1] <= SP):
            continue
        if lab[e] == INACTIVE:  # label stopped just before the ball went dead
            lab[s:e], sub[s:e], src[s:e] = lab[s - 1], sub[s - 1], 'dead_lag'
        elif 0 <= lab[e] <= SP:  # button-switch gap inside play: split at the midpoint
            mid = (s + e) // 2
            lab[s:mid], lab[mid:e], src[s:e] = lab[s - 1], lab[e], 'gap_mid'
            sub[s:mid], sub[mid:e] = sub[s - 1], sub[e]
    return lab, src, sub


def merge(a, b, alive):
    final = np.where(a == b, a, PENDING)
    src = np.where(a == b, 'agree', 'pending').astype(object)
    src[~alive] = 'inactive'
    for s, e in runs(final == PENDING):
        left = final[s - 1] if s > 0 else None
        right = final[e] if e < len(final) else None
        ok = left is not None and right is not None and left >= 0 and right >= 0
        if ok and e - s <= TAU and set(a[s:e]) <= {left, right} and set(b[s:e]) <= {left, right}:
            mid = (s + e) // 2
            final[s:mid], final[mid:e] = left, right
            src[s:e] = 'auto_mid'
    both_unc = alive & (a == UNCODED) & (b == UNCODED)
    final[both_unc], src[both_unc] = PENDING, 'pending'
    return final, src


def collapse(x):
    return [int(v) for i, v in enumerate(x) if i == 0 or v != x[i - 1]]


def clip_type(a, b, s, e):
    """Compare the two label sequences over the clip plus one agreed frame on each side."""
    lo, hi = max(s - 1, 0), min(e + 1, len(a))
    qa, qb = collapse(a[lo:hi]), collapse(b[lo:hi])
    if UNCODED in qa + qb:
        return 'UNC'
    if qa == qb:
        return 'BOUNDARY'
    if SP in qa + qb:
        return 'SP'
    if {1, 3} & set(qa) != {1, 3} & set(qb):
        return 'EXIST'
    return 'CLASS'


def seq(lab, s, e):
    """Label sequence inside [s, e) as 'NAME@frame; ...' (25 Hz frame numbers)."""
    ch = np.r_[s, s + 1 + np.flatnonzero(np.diff(lab[s:e]) != 0)]
    return '; '.join(f'{NAMES[int(lab[c])]}@{c}' for c in ch)


def kappa(x, y, cats):
    """Cohen's kappa on the frames where both codes are in cats; None if undefined (no frames, or chance
    agreement 1: both coders used one and the same code)."""
    m = np.isin(x, cats) & np.isin(y, cats)
    x, y = x[m], y[m]
    if not m.any():
        return dict(n=0, kappa=None)
    po = np.mean(x == y)
    pe = sum(np.mean(x == c) * np.mean(y == c) for c in cats)
    return dict(n=int(m.sum()), kappa=round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None)


def irr_subcategory(a, b, suba, subb, alive):
    """Subcategory kappa: over alive frames where both coded one (a moment disagreement counts as a subcategory
    disagreement), over frames where both coded the same moment, and within each moment."""
    same = alive & (a == b) & (a >= 0) & (a <= 3)
    subs = list(range(len(SUBCATS)))
    return {'alive, frames where both coded a subcategory': kappa(suba[alive], subb[alive], subs),
            'alive, both coded the same moment and a subcategory': kappa(suba[same], subb[same], subs),
            'by moment (both coded it)': {NAMES[k]: kappa(suba[same & (a == k)], subb[same & (a == k)],
                                                          list(np.flatnonzero(SUB_CATEGORY == k)))
                                           for k in (0, 1, 2, 3)}}


def convert(a_csv, b_csv, events_csv, npz, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ca, sca, alive = read_cats(a_csv)
    cb, scb, alive_b = read_cats(b_csv)
    assert (alive == alive_b).all(), 'ball_status differs between the two files'
    d = dict(np.load(npz))
    rs = R.detect(alive, d, events_csv)
    rs['set_piece'] = [is_set_piece(r) for r in rs.itertuples()]
    a0, nA, suba0 = clean(ca, sca)
    b0, nB, subb0 = clean(cb, scb)
    a, sa, suba = to_v1(a0, alive, rs, suba0)
    b, sb, subb = to_v1(b0, alive, rs, subb0)
    suba, xA = consistent_sub(a, suba)
    subb, xB = consistent_sub(b, subb)
    final, src = merge(a, b, alive)

    clips = []
    for s, e in runs(final == PENDING):
        clips.append(dict(start=int(s), end=int(e - 1), dur_s=round((e - s) / FPS, 2),
                          type=clip_type(a, b, s, e), A=seq(a, s, e), B=seq(b, s, e)))
    clips = pd.DataFrame(clips)
    clips.insert(0, 'clip_id', [f'c{i:03d}' for i in range(len(clips))])

    irr = {'alive, UNCODED as a category (report this)': kappa(a[alive], b[alive], [0, 1, 2, 3, SP, UNCODED]),
           'alive, frames coded by both, moments + SP': kappa(a[alive], b[alive], [0, 1, 2, 3, SP]),
           'alive, frames coded by both, moments only': kappa(a[alive], b[alive], [0, 1, 2, 3]),
           'all frames incl. INACTIVE (inflated, do not report)': kappa(a, b, [0, 1, 2, 3, SP, INACTIVE])}
    irr_sub = irr_subcategory(a, b, suba, subb, alive)
    summary = dict(
        frames=len(final), alive=int(alive.sum()),
        restarts=rs.kind.value_counts().to_dict(), set_piece_restarts=int(rs.set_piece.sum()),
        conversion={c: dict(sp_gap_s=round(float((s_ == 'sp_gap').sum()) / FPS, 1),
                            restart_lag_s=round(float((s_ == 'restart_lag').sum()) / FPS, 1),
                            gap_mid_s=round(float((s_ == 'gap_mid').sum()) / FPS, 1),
                            dead_lag_s=round(float((s_ == 'dead_lag').sum()) / FPS, 1),
                            uncoded_alive_s=round(float(((l == UNCODED) & alive).sum()) / FPS, 1),
                            overlap_frames_split=n,
                            moment_without_subcategory_s=round(float((alive & (l >= 0) & (l <= 3) & (sc < 0)).sum()) / FPS, 1),
                            subcategory_of_another_moment_frames=x)
                    for c, s_, l, n, sc, x in (('A', sa, a, nA, suba, xA), ('B', sb, b, nB, subb, xB))},
        sources_alive={k: int(v) for k, v in pd.Series(src[alive]).value_counts().items()},
        clips=len(clips), clips_s=round(float(clips.dur_s.sum()), 1) if len(clips) else 0,
        clip_types=clips.type.value_counts().to_dict() if len(clips) else {},
        irr_pre_adjudication=irr, irr_subcategory_pre_adjudication=irr_sub)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    rs.to_csv(out / 'restarts.csv', index=False)
    clips.to_csv(out / 'clips.csv', index=False)
    pd.DataFrame({'Frame': np.arange(len(final)), 'alive': alive.astype(int), 'A': a, 'B': b,
                  'final': final, 'src': src, 'subA': suba, 'subB': subb}).to_csv(out / 'labels_v1_25hz.csv', index=False)


def validate(export_csv, events_csv, npz):
    """Checks for one coder's export under codebook v1. Errors: overlaps, unknown categories, moment labels or
    'Inactive' on the wrong side of ball_status, missing subcategories, segments < 0.5 s (unless cut by a dead
    ball), alive blanks > 1 s other than the leading blank after a set-piece restart. Alive blanks <= 1 s are
    listed as warnings (the conversion fixes them)."""
    df = pd.read_csv(export_csv, low_memory=False, dtype=str)
    df.columns = [c.rstrip(';') for c in df.columns]
    cat = df.Annotation_Category.fillna('').str.strip()
    alive = df.ball_status.eq('alive').to_numpy()
    lab = cat.map(CATEGORY).fillna(UNCODED).astype(int).to_numpy()
    sub = df.get('Annotation_Subcategory', pd.Series([''] * len(df))).fillna('').str.strip()
    rs = R.detect(alive, dict(np.load(npz)), events_csv)
    sp_onsets = {r.onset for r in rs.itertuples() if is_set_piece(r)}
    seg = np.where(alive, lab, INACTIVE)
    blanks = [(int(s), round((e - s) / FPS, 2)) for s, e in runs(alive & (lab == UNCODED)) if s not in sp_onsets]
    v = dict(
        overlap_frames=int(cat.str.contains(';').sum()),
        unknown_category=sorted(set(cat[(lab == UNCODED) & (cat != '') & ~cat.str.contains(';')])),
        moment_on_dead_frames=int((~alive & (lab >= 0) & (lab <= 3)).sum()),
        inactive_on_alive_frames=int((alive & (lab == INACTIVE)).sum()),
        missing_subcategory=int(((lab >= 0) & (lab <= 3) & (sub == '')).sum()),
        short_segments=[(int(s), int(e - s)) for s, e in (r for c in (0, 1, 2, 3) for r in runs(seg == c))
                        if e - s < MIN_SEG and INACTIVE not in (seg[max(s - 1, 0)], seg[min(e, len(seg) - 1)])],
        blanks_over_1s=[b for b in blanks if b[1] > TAU / FPS],
        warn_blanks_up_to_1s=len([b for b in blanks if b[1] <= TAU / FPS]),
        set_piece_blanks=dict(restarts=len(sp_onsets),
                              left_blank=int(sum(alive[s] and lab[s] == UNCODED for s in sp_onsets))),
    )
    v['ok'] = not (v['overlap_frames'] or v['unknown_category'] or v['moment_on_dead_frames']
                   or v['inactive_on_alive_frames'] or v['missing_subcategory'] or v['short_segments']
                   or v['blanks_over_1s'])
    return v


if __name__ == '__main__':
    if sys.argv[1] == 'convert':
        convert(*sys.argv[2:7])
    elif sys.argv[1] == 'validate':
        print(json.dumps(validate(*sys.argv[2:5]), indent=1, ensure_ascii=False))
