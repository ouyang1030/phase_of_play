"""Restarts (dead -> alive onsets) at 25 Hz with type, taking team and zone, from ball_status + synced events."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase import events as Ev
from phase.prep import home_attack_sign

FPS = 25
CONTROL = set(Ev.CONTROL) | {Ev.TACKLE, Ev.BALL_TOUCH}
THIRD = 52.5 / 3  # x beyond +-17.5 m from the halfway line = final third


def ball_status(csv_path: str) -> np.ndarray:
    s = pd.read_csv(csv_path, usecols=lambda c: c.rstrip(';') == 'ball_status', dtype=str).iloc[:, 0]
    return s.eq('alive').to_numpy()


def detect(alive25: np.ndarray, d: dict, events_csv: str) -> pd.DataFrame:
    """One row per restart: onset/end frame (25 Hz), type, team (0 home / 1 away), x_att (m toward the opponent goal)."""
    step = int(round(FPS / float(d['hz'])))
    e = Ev.load(events_csv, d)
    ef = e.idx.to_numpy() * step
    ch = np.flatnonzero(np.diff(np.r_[0, alive25.astype(int), 0]))
    runs = list(zip(ch[::2], ch[1::2]))
    rows = []
    for k, (s, t) in enumerate(runs):
        prev_end = runs[k - 1][1] if k else 0
        win = e[(ef >= prev_end - 2 * FPS) & (ef <= s + FPS)]
        types = set(win.type)
        after = e[(ef >= s - FPS // 2) & (ef < min(t, s + 5 * FPS)) & e.type.isin(CONTROL)]
        team = int(after.team.iat[0]) if len(after) else -1
        i = min(s // step, len(d['ball']) - 1)
        bx = float(d['ball'][i, 0]) * float(home_attack_sign(d, i))  # home attacks +x in this frame
        x_att = bx if team == 0 else -bx
        if k == 0 or d['period'][s // step] != d['period'][max(prev_end - 1, 0) // step] or Ev.GOAL in types:
            kind = 'kickoff'
        elif 6 in types:
            kind = 'corner'
        elif types & {4, 55, 2}:
            kind = 'free_kick'
        elif 5 in types:
            kind = 'goal_kick' if abs(bx) > 47 else 'throw_in'
        else:
            kind = 'other'
        rows.append(dict(onset=int(s), end=int(t), kind=kind, team=team, x_att=round(x_att, 1),
                         zone=('final' if x_att > THIRD else 'middle' if x_att > -THIRD else 'own') if team >= 0 else ''))
    return pd.DataFrame(rows)
