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
| `episodes.py` | labels = possession + one transition length per possession: episode windows, `observe` (the interval a label sequence gives for the length), `reconstruct`, exact frame macro-F1 for any length vector (`EpisodeScorer`), best single change point, `cv_folds` |
| `decompose.py` | RQ1: 2 × 2 error decomposition, possession (labels / events) × lengths (oracle / timer / `phase.duration` runs) |
| `concepts.py` | codebook §2.5 criteria C1–C4 as tracking quantities per frame and view; rule baseline (one threshold per criterion + hold time, grid search) with the criterion satisfied last as explanation |
| `duration.py` | main model: per turnover the distribution of the transition length (targets `label` / `interval` / `noise`), quantile decoding, timer and rule baselines on the same episodes, transition-length metrics; `--no-model` without torch |
| `endpoint.py` | torch part of `duration.py`: MLP → 1-layer BiGRU → one logit per window frame + "fills the window" (~25k parameters), range and coder-noise likelihoods |

Ablation flags: `--no-graph`, `--no-org`, `--no-timer-feat`, `--temporal {bigru,tcn,none}`, `--no-poss-input`
(removes every input derived from the input possession: `poss_ref` and `time_since_turnover` from the context
features and the node feature `team_poss`; before 2026-11 it removed `poss_ref` only).

Auxiliary possession target `--poss_target {label,input}` (default `label`): the possession head learns the
annotators' possession (label in est_attack / def_to_att of the reference team; soft labels: q0 + q1) on labelled
training frames. `input` is the old target: the input possession (events or tracking heuristic) on alive frames.

Event possession `--poss_confirm {first,second}` (default `first`, train with `--events` and hazard): `second`
follows codebook v4.1 §2.4 and counts an open-play gain only once the gaining team's next controlling event (or the
reception of its completed pass) confirms it; otherwise the touch is a loose ball and possession stays. Changes
across a dead ball need no confirmation. Both runs report the label agreement of both definitions
(`possession_label_agreement` in hazard, `possession_label_agreement_by_confirm` in train).

## Research line: possession + transition length (2026-11)

Within a possession of team w the labels are [transition_w]* [established_w]*, so given the possession a match is
one number per turnover: the transition length E (`episodes.py`). The main model predicts p(E) per turnover from
the window of frames after it (`duration.py`); labels follow by construction, without post-processing.

```bash
# RQ1: how much of the labels is decided by possession? (labels row = oracle possession)
.venv/bin/python -m phase.decompose --npz ADJ/ajax_psv_full_adj.npz --events event_data.csv --meta meta.json \
    --poss_confirm second --out artifacts/decompose [model=artifacts/dur_noise model=artifacts/dur_poss_labels]
# main model + timer + rule on the same episodes; --target noise / interval use both coders (label_A / label_B of
# the soft npz, or --coders A.npz B.npz); --target label uses the npz labels
.venv/bin/python -m phase.duration --npz ADJ/ajax_psv_full_soft.npz --events event_data.csv --meta meta.json \
    --poss_confirm second --target noise --out artifacts/dur_noise
.venv/bin/python -m phase.duration ... --no-model                       # timer + rule only, no torch
.venv/bin/python -m phase.duration ... --features concepts,ball,time,episode   # feature-group ablation
# L2 runs: trained on one coder only
.venv/bin/python -m phase.duration --npz ADJ/ajax_psv_full_A.npz --events event_data.csv --meta meta.json \
    --poss_confirm second --out artifacts/dur_A
.venv/bin/python -m phase.duration --npz ADJ/ajax_psv_full_B.npz --events event_data.csv --meta meta.json \
    --poss_confirm second --out artifacts/dur_B
# oracle possession, only for the labels row of the decomposition
.venv/bin/python -m phase.duration --npz ADJ/ajax_psv_full_adj.npz --poss labels --out artifacts/dur_poss_labels
```

The model's inputs are the unsmoothed organisation block (`features.py`; ref = the new possessor), the C1–C4
quantities, the ball, the time since the window start and episode constants (dead-ball turnover, how the ball was
won, ball position at the start); ablate them with `--features`. The window starts at the first live frame at or
after the turnover (the restart after a dead-ball turnover) and ends at the next turnover, the next dead ball, a
period change or `--max_s` (30 s). `--decode tuned` (default) picks the quantile of p(E) on the training frames
(macro-F1), as τ is tuned for the timer; `--decode median` minimises the expected number of wrong window frames.
Outputs: `oof.npz` (`pred`, `probs`, `timer`, `rule`), `episodes.csv` (per turnover: lengths of all methods,
model quantiles, rule reason, each label source's observation), `results.json` (frame metrics, paired run
bootstrap, transition-length MAE / CRPS / 80% coverage against the labels and each coder, and the coders' interval).
Better than a human coder: `tools/adj/superhuman.py` (L1 / L2 / L3, see `tools/adj/README.md`).

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
- 2026-11: ball positions that the provider held while the ball was moving (exact repeat after a step > 2 m/s;
  `ball_held_play`, `ball_held_max_s`) are interpolated like gaps (≤ 0.5 s). Exact repeats of a ball at rest or
  rolling slowly (1 cm resolution) are real and kept. Ajax–PSV: 15 single frames.

## Changes that alter training results
- 2026-11: `evaluate.fill_possession` forward-fills before back-filling, as documented (gaps keep the previous
  possessor). Event and heuristic possession have NaN only before the first touch, so no earlier result changes.
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
