# phase — Study 1 pipeline (v0, one match)

Four-moment segmentation (est. attack / def→att transition / est. defense / att→def transition)
from tracking data, trained on manual labels. Implements STUDY1_PLAN §6–§8 at prototype scale.

```bash
.venv/bin/python -m phase.prep  <fused.csv> --out artifacts/<match>.npz [--meta meta.json]  # 25 Hz -> 12.5 Hz arrays
.venv/bin/python -m phase.train --npz artifacts/<match>.npz --out artifacts/run_x
.venv/bin/python -m phase.plot  artifacts/run_x                            # timeline.png
# event-based possession (recommended) + turnover-anchored hazard baselines
.venv/bin/python -m phase.train  --npz artifacts/<match>.npz --events <event_data.csv> --out artifacts/run_x
.venv/bin/python -m phase.hazard --npz artifacts/<match>.npz --events <event_data.csv> --out artifacts/hazard
```

| file | content |
|---|---|
| `prep.py` | CSV → slots (11/team; a slot keeps its player while on the pitch, `slot_col` / `player_cols`), velocities, GK (roster from `meta.json` with `--meta`, else the player nearest to his own goal on > 50% of his frames and on ≥ 60 s; several per half allowed), periods (kick-off / last frame from `meta.json` with `--meta`, else a filled provider `period_id` column, else player-free breaks ≥ 60 s; any number of periods), play direction per period, label mapping |
| `features.py` | L1 node features, L3 Organization Block (polarization, rotation, line silhouette k=2/3/4, hull, stretch, pressure, ball-area numbers), L4 multi-scale (raw/2 s/5 s), possession heuristic, two mirrored views |
| `model.py` | heterogeneous graph attention (intra_ref / intra_opp / press-kNN / ball) → BiGRU or MS-TCN → class / boundary / possession heads; focal CE + boundary + smoothing + possession + view-symmetry loss |
| `evaluate.py` | timer baseline L_fix(τ), mode filter, HSMM-lite Viterbi, frame / boundary / segmental metrics |
| `train.py` | blocked time CV on the labeled stretch, per-fold τ* and HSMM fit, pooled out-of-fold evaluation |
| `events.py` | Opta events (databallpy `event_data.csv`) → frames (synced / clock + offset + snap), sync checks (`attrs['sync_report']`), event possession, turnover table (how, where, dead ball) |
| `hazard.py` | turnover-anchored discrete-time hazard for transition duration (causal features, censoring); timer τ* on heuristic vs event possession; same blocked CV as `train.py` (PLAN §15) |

Ablation flags: `--no-graph`, `--no-org`, `--no-timer-feat`, `--temporal {bigru,tcn,none}`, `--no-poss-input`
(drops `poss_ref` from the context features only: the node feature `team_poss` and, unless `--no-timer-feat`,
`time_since_turnover` still come from the input possession).

Auxiliary possession target `--poss_target {label,input}` (default `label`): the possession head learns the
annotators' possession (label in est_attack / def_to_att of the reference team; soft labels: q0 + q1) on labelled
training frames. `input` is the old target: the input possession (events or tracking heuristic) on alive frames.

Event possession `--poss_confirm {first,second}` (default `first`, train with `--events` and hazard): `second`
follows codebook v4.1 §2.4 and counts an open-play gain only once the gaining team's next controlling event (or the
reception of its completed pass) confirms it; otherwise the touch is a loose ball and possession stays. Changes
across a dead ball need no confirmation. Both runs report the label agreement of both definitions
(`possession_label_agreement` in hazard, `possession_label_agreement_by_confirm` in train).

## Label mapping (reference team = home)
| CSV `Annotation_Category` | class |
|---|---|
| Attacking | 0 est_attack |
| Defending Transition (Counterattack, Delayed Attack) | 1 def_to_att |
| Defending | 2 est_defense |
| Attacking Transition (Counterpressing, Delay and recovery) | 3 att_to_def |

## Changes that need a re-run of `phase.prep`
- 2026-11: player slots are stable (before, every slot after a missing player shifted by one for the length of
  the gap, and 2 s velocities were averaged across different players). Old npz files have no `slot_col`;
  `features.build` still accepts them but warns. Re-create the npz files and re-run models before comparing.
- 2026-11: with `--meta`, periods are the provider's kick-off / last frames. The player-free break rule counts the
  minute after the half-time whistle and the line-up before the second-half kick-off as play (Ajax–PSV: 1848
  frames at 12.5 Hz, 30 of them "alive"). Frames outside a period now take the play direction of the nearest
  period in `features` / `hazard` (before: home attacks +x).

## Changes that alter training results
- 2026-11: the auxiliary possession target is the annotators' possession (`--poss_target label`). All earlier
  runs (`artifacts/run_full`, `artifacts/adj_v0/deep*`) used the input possession, i.e. `--poss_target input`;
  re-run with `--poss_target input` to reproduce them.

## Known limitations (v0)
- **One match, blocked CV inside it** → same teams in train and test; results are a smoke test, not evidence.
- The fused CSV's event columns are empty; events come from the separate databallpy export
  (`event_data.csv`, `--events`). Without it, possession falls back to the tracking heuristic
  (nearest player < 1.5 m, runs < 1 s merged), which agrees with labels on 92.3% of frames vs 95.4%
  for events and makes the timer baseline much weaker (macro-F1 0.54 vs 0.69, PLAN §15.2).
  Compare models only under the same possession source.
- HSMM is a min-duration chain approximation (geometric tail), not a full learned duration distribution.
- Not yet implemented from §6: Procrustes formation deviation, role-return cost, control-surface fragmentation, `mark` edges.
