"""Blinded adjudication pack for the clips listed by codebook_v1.py (STUDY1_PLAN §4.7).

Coders are renamed R1 / R2 at random (one mapping per match); the key is written next to, not inside, the pack.
Per clip: label timelines of R1 and R2 with events (PNG) and an optional 2D animation (MP4), +-5 s of context.
No model features are shown. For clips that touch a set piece, the end frame suggested by the SET_PIECE rule is
added as a reference.

  python tools/adj/make_pack.py ADJ_DIR NPZ EVENTS_CSV PACK_DIR --meta META_JSON [--seed N] [--anim]

Team names, the event team_id of the home team and the kick-off frames for the match clock come from meta.json
(phase.meta; tracking team 0 = home = the labels' reference team).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import events as Ev  # noqa: E402
from phase import meta as M  # noqa: E402
from phase.prep import home_attack_sign  # noqa: E402
import codebook_v1 as C  # noqa: E402

FPS, CTX = 25, 5 * 25
COLORS = {0: '#1b7837', 1: '#a6dba0', 2: '#762a83', 3: '#f1a340', C.SP: '#4393c3', C.INACTIVE: '#bdbdbd',
          C.UNCODED: '#ffffff'}
PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}


def clock(frame, period_start, period_end=None):
    """(period, 'mm:ss' match clock) of a 25 Hz frame; period_start / period_end map period -> kick-off / last frame.
    A frame after the end of its period and before the next kick-off (a break) is given as '-mm:ss' before that
    kick-off."""
    started = [p for p, f in period_start.items() if frame >= f]
    p = max(started) if started else min(period_start)
    later = [q for q in period_start if q > p]
    if period_end and p in period_end and frame > period_end[p] and later:
        q = min(later)
        t = (period_start[q] - frame) / FPS
        return q, f'-{int(t // 60):02d}:{int(t % 60):02d}'
    t = max(frame - period_start[p], 0) / FPS + 60 * PERIOD_OFFSET_MIN.get(p, 0)
    return p, f'{int(t // 60):02d}:{int(t % 60):02d}'


def sp_rule_end(r, e, d, step):
    """SET_PIECE end by the codebook rule (§4.2), first of: the defending team completes its first pass (saves,
    claims, clearances and recoveries alone do not end it); the attacking team plays the ball into its own half;
    the ball goes dead. The restart itself is skipped."""
    s, t, team = r.onset, r.end, r.team
    if team < 0:
        return t
    ev = e[(e.idx * step > s + step) & (e.idx * step < t)]
    for row in ev.itertuples():
        if row.team != team and row.type in (1, 2):  # pass / offside pass
            return row.idx * step
        sign = float(home_attack_sign(d, row.idx)) * (1 if team == 0 else -1)  # attacking team plays toward +x*sign
        if row.team == team and row.type in Ev.CONTROL and row.x * sign < 0:
            return row.idx * step
    return t


def timeline_png(path, clip, lab1, lab2, alive, ev, lo, hi, sp_end, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12, 3.2))
    tt = lambda f: (f - clip.start) / FPS  # noqa: E731
    for y, lab in ((2, lab1), (1, lab2)):
        x = lab[lo:hi]
        ch = np.r_[0, 1 + np.flatnonzero(np.diff(x) != 0), len(x)]
        for a, b in zip(ch[:-1], ch[1:]):
            ax.barh(y, (b - a) / FPS, left=tt(lo + a), height=0.6, color=COLORS[int(x[a])], edgecolor='k', lw=0.3,
                    hatch='//' if x[a] == C.UNCODED else None)
    dead = ~alive[lo:hi]
    for a, b in C.runs(dead):
        ax.axvspan(tt(lo + a), tt(lo + b), color='#eeeeee', zorder=0)
    ax.axvspan(tt(clip.start), tt(clip.end + 1), color='yellow', alpha=0.25, zorder=0)
    for row in ev.itertuples():
        f = row.f
        ax.plot([tt(f)] * 2, [0.2, 0.6], color='#2166ac' if row.team == 0 else '#b2182b', lw=1)
        ax.text(tt(f), 0.15, row.original_event, rotation=60, fontsize=6, ha='right', va='top')
    if sp_end is not None:
        ax.axvline(tt(sp_end), color=COLORS[C.SP], ls='--', lw=1.2)
        ax.text(tt(sp_end), 2.45, 'SP rule end', color=COLORS[C.SP], fontsize=7)
    ax.set_yticks([2, 1, 0.4], ['R1', 'R2', 'events'])
    ax.set_ylim(-0.9, 2.6)
    ax.set_xlim(tt(lo), tt(hi))
    ax.set_xlabel('s from clip start (yellow = disputed interval, grey = dead ball)')
    ax.set_title(title, fontsize=9)
    ax.legend(handles=[Patch(color=COLORS[k], label=C.NAMES[k]) for k in (0, 1, 2, 3)] +
              [Patch(color=COLORS[C.SP], label='EX-SP (left empty after set piece)')] +
              [Patch(facecolor='w', edgecolor='k', hatch='//', label='UNCODED')],
              ncol=6, fontsize=7, loc='upper right', bbox_to_anchor=(1, -0.32))
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def anim_mp4(path, d, lab1, lab2, lo, hi, clip, title, step, teams):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig = plt.figure(figsize=(8, 6.2))
    ax = fig.add_axes([0.03, 0.2, 0.94, 0.72])
    tl = fig.add_axes([0.08, 0.04, 0.88, 0.1])
    ax.set_xlim(-55, 55); ax.set_ylim(-36, 36); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Rectangle((-52.5, -34), 105, 68, fill=False))
    ax.plot([0, 0], [-34, 34], 'k', lw=0.8)
    for sx in (-1, 1):
        ax.add_patch(plt.Rectangle((sx * 52.5 - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32, fill=False, lw=0.8))
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, lw=0.8))
    direction = ax.text(-52, 35, '', fontsize=8)
    home = ax.scatter([], [], s=60, c='#2166ac', edgecolors='k', zorder=3)
    away = ax.scatter([], [], s=60, c='#b2182b', edgecolors='k', zorder=3)
    ball = ax.scatter([], [], s=25, c='k', zorder=4)
    head = ax.set_title('', fontsize=9)
    for y, lab in ((1, lab1), (0, lab2)):
        x = lab[lo:hi]
        ch = np.r_[0, 1 + np.flatnonzero(np.diff(x) != 0), len(x)]
        for a, b in zip(ch[:-1], ch[1:]):
            tl.barh(y, b - a, left=a, color=COLORS[int(x[a])], height=0.8, hatch='//' if x[a] == C.UNCODED else None)
    tl.axvspan(clip.start - lo, clip.end + 1 - lo, color='yellow', alpha=0.3, zorder=0)
    tl.set_yticks([1, 0], ['R1', 'R2']); tl.set_xticks([]); tl.set_xlim(0, hi - lo)
    cur = tl.axvline(0, color='k')
    writer = FFMpegWriter(fps=FPS / step)
    with writer.saving(fig, str(path), dpi=80):
        for f in range(lo - lo % step, hi, step):
            i = f // step
            home.set_offsets(np.nan_to_num(d['pos'][i, 0], nan=-99))
            away.set_offsets(np.nan_to_num(d['pos'][i, 1], nan=-99))
            ball.set_offsets(d['ball'][i, :2][None])
            cur.set_xdata([f - lo])
            head.set_text(f'{title}   frame {f}' + ('   [dead ball]' if not d['alive'][i] else ''))
            direction.set_text(f"{teams[0]} (blue) attacks {'→' if home_attack_sign(d, i) > 0 else '←'}")
            writer.grab_frame()
    plt.close(fig)


README = """# Adjudication pack — {match}

You are the third (adjudicating) coder. Two coders, **R1** and **R2**, labelled this match independently;
their identities are hidden. Each row of `clips.csv` is an interval where they disagree, or where neither
labelled a stretch of live play. Decide only **inside the yellow interval**; everything outside it is
already agreed and must not change.

## What to look at
- The match video at the given frame / clock (frames are 25 Hz frame numbers of the fused CSV, as in the
  annotation tool), from `view_from` to `view_to` (5 s of context on each side). A negative clock is the time
  before that period's kick-off (e.g. ball tracked as alive during half-time).
- `timelines/<clip_id>.png`: R1 and R2 labels, dead-ball periods (grey) and on-ball events.
- `anim/<clip_id>.mp4` (if present): 2D animation from tracking with both timelines and a cursor.
- Do **not** use any computed organisation metrics (they are model inputs).

## Labels (reference team: {home}; codebook: Phase definition v4.1)
IP = In Possession · TA = Transition to Attack · OP = Out of Possession · TD = Transition to Defence
· EX-DEAD = dead ball (fixed, from ball_status) · EX-SP = set-piece play.

**Set pieces were not labelled.** After a corner or a free kick ≤ 40 m from goal both coders left the timeline
empty until they judged open play had begun; that empty stretch is shown as EX-SP. Open play resumes at the
first of: the defending team completes its first pass (a save, catch, clearance or recovery alone does not
count; second balls and repeated crosses are still part of the set piece); the attacking team plays the ball
back into its own half; the ball goes dead. The PNG marks the frame this rule gives from event data
("SP rule end"); it is a reference only.

Apply the codebook as written, in particular:
- possession changes only when confirmed (§2.4: second controlled action; loose ball keeps the previous label);
  a transition starts at `gain_frame`;
- a transition ends when all four criteria of §2.5 hold for ≈ 3 s, and the boundary goes on the **first frame**
  of that window; nothing else ends it except a further possession change or EX;
- a zero-length transition (§2.5) is allowed: IP → OP directly if the new defending team is already organized.

## How to record a decision (`decisions.csv`, one row per clip)
| column | values |
|---|---|
| decision | `R1`, `R2`, or `OTHER` |
| other_segments | only for OTHER: the labels inside the yellow interval as `LABEL@frame; LABEL@frame`, first frame = `start_frame`, e.g. `TD@51200; OP@51390` |
| reason_code | C1–C4: the §2.5 criterion that decided the transition end (1 deepest line formed · 2 no player recovering · 3 lines connected · 4 no uncovered direct route) · P: possession (§2.4) · X: set piece / EX · O: other (explain in note) |
| confidence | 1 (unsure) – 3 (certain) |
| note | free text |

Pick R1 or R2 only if that coder's labels inside the interval are right as they are (including where each
boundary falls, within about 1 s). If a coder left part of the interval unlabelled (UNCODED), that coder
cannot be picked; use OTHER.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('adj_dir'); ap.add_argument('npz'); ap.add_argument('events'); ap.add_argument('pack')
    ap.add_argument('--meta', required=True, help='meta.json of the match (notebooks/preprocessing.ipynb)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--anim', action='store_true')
    a = ap.parse_args()
    adj, pack = Path(a.adj_dir), Path(a.pack)
    meta = M.load(a.meta)
    if float(meta['frame_rate']) != FPS:
        sys.exit(f"{a.meta}: frame_rate {meta['frame_rate']} Hz, but the pack works on the {FPS} Hz fused CSV")
    teams = M.teams(meta)
    (pack / 'timelines').mkdir(parents=True, exist_ok=True)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(10 ** 6)
    flip = random.Random(seed).random() < 0.5
    key = {'R1': 'B' if flip else 'A', 'R2': 'A' if flip else 'B', 'seed': seed}
    key_path = adj / 'key_DO_NOT_SHARE.json'
    if key_path.exists():  # keep the mapping stable if the pack is rebuilt
        key = json.loads(key_path.read_text())
    key_path.write_text(json.dumps(key, indent=1))

    L = pd.read_csv(adj / 'labels_v1_25hz.csv')
    clips = pd.read_csv(adj / 'clips.csv')
    rs = pd.read_csv(adj / 'restarts.csv')
    d = dict(np.load(a.npz))
    step = int(round(FPS / float(d['hz'])))
    e = Ev.load(a.events, d, home_team_id=meta['home_team']['id'])
    lab = {'A': L.A.to_numpy(), 'B': L.B.to_numpy()}
    r1, r2 = lab[key['R1']], lab[key['R2']]
    alive = L.alive.to_numpy().astype(bool)
    period_start = meta['period_start_frame']  # kick-off frames (25 Hz rows of the fused CSV)
    period_end = meta.get('period_end_frame')  # optional: frames in a break get '-mm:ss' before the next kick-off

    sp = rs[rs.set_piece].copy()
    sp['rule_end'] = [sp_rule_end(r, e, d, step) for r in sp.itertuples()]
    rows = []
    for c in clips.itertuples():
        lo, hi = max(c.start - CTX, 0), min(c.end + 1 + CTX, len(L))
        p, t0 = clock(c.start, period_start, period_end)
        _, t1 = clock(c.end, period_start, period_end)
        near = sp[(sp.onset <= c.end + CTX) & (sp.end >= c.start - CTX)]
        sp_end = int(near.rule_end.iat[0]) if len(near) else None
        ev = e[(e.idx * step >= lo) & (e.idx * step < hi)].assign(f=lambda x: x.idx * step)
        rows.append(dict(
            clip_id=c.clip_id, type=c.type, start_frame=c.start, end_frame=c.end, dur_s=c.dur_s,
            period=p, clock_start=t0, clock_end=t1, view_from=lo, view_to=hi - 1,
            R1=C.seq(r1, c.start, c.end + 1), R2=C.seq(r2, c.start, c.end + 1),
            sp_rule_end='' if sp_end is None else sp_end,
            events='; '.join(f'{clock(f, period_start, period_end)[1]} {teams[t]} {n}' for f, t, n in
                             zip(ev.f, ev.team, ev.original_event))))
        title = f'{c.clip_id}  [{c.type}]  period {p}  {t0}–{t1}  frames {c.start}–{c.end}'
        timeline_png(pack / 'timelines' / f'{c.clip_id}.png', c, r1, r2, alive, ev, lo, hi, sp_end, title)
        if a.anim:
            (pack / 'anim').mkdir(exist_ok=True)
            anim_mp4(pack / 'anim' / f'{c.clip_id}.mp4', d, r1, r2, lo, hi, c, title, step, teams)
    out = pd.DataFrame(rows)
    out.to_csv(pack / 'clips.csv', index=False)
    pd.DataFrame({'clip_id': out.clip_id, 'decision': '', 'other_segments': '', 'reason_code': '',
                  'confidence': '', 'note': ''}).to_csv(pack / 'decisions.csv', index=False)
    (pack / 'README.md').write_text(README.format(match=f'{teams[0]} vs {teams[1]} ({meta["match_id"]})', home=teams[0]))

    # how well the SET_PIECE rule reproduces the coders' own set-piece ends (both coders, all set pieces)
    diffs = []
    for r in sp.itertuples():
        for x in lab.values():
            j = np.flatnonzero(x[r.onset:r.end] != C.SP)
            if j.size and j[0] > 0:
                diffs.append((r.rule_end - (r.onset + j[0])) / FPS)
    diffs = np.array(diffs)
    summary = dict(clips=len(out), types=out.type.value_counts().to_dict(), set_pieces=len(sp),
                   sp_rule_vs_coders_s=dict(n=len(diffs), median=round(float(np.median(diffs)), 2),
                                            median_abs=round(float(np.median(np.abs(diffs))), 2),
                                            within_2s=round(float(np.mean(np.abs(diffs) <= 2)), 2)) if len(diffs) else {})
    sp.to_csv(adj / 'set_pieces_rule_end.csv', index=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
