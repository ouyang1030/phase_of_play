# phase — Study 1 pipeline (v0, one match)

Four-moment segmentation (est. attack / def→att transition / est. defense / att→def transition)
from tracking data, trained on manual labels. Implements STUDY1_PLAN §6–§8 at prototype scale.

```bash
.venv/bin/python -m phase.prep  <fused.csv> --out artifacts/<match>.npz   # 25 Hz -> 12.5 Hz arrays
.venv/bin/python -m phase.train --npz artifacts/<match>.npz --out artifacts/run_x
.venv/bin/python -m phase.plot  artifacts/run_x                            # timeline.png
# event-based possession (recommended) + turnover-anchored hazard baselines
.venv/bin/python -m phase.train  --npz artifacts/<match>.npz --events <event_data.csv> --out artifacts/run_x
.venv/bin/python -m phase.hazard --npz artifacts/<match>.npz --events <event_data.csv> --out artifacts/hazard
```

| file | content |
|---|---|
| `prep.py` | CSV → slots (11/team), velocities, GK, periods, play direction, label mapping |
| `features.py` | L1 node features, L3 Organization Block (polarization, rotation, line silhouette k=2/3/4, hull, stretch, pressure, ball-area numbers), L4 multi-scale (raw/2 s/5 s), possession heuristic, two mirrored views |
| `model.py` | heterogeneous graph attention (intra_ref / intra_opp / press-kNN / ball) → BiGRU or MS-TCN → class / boundary / possession heads; focal CE + boundary + smoothing + possession + view-symmetry loss |
| `evaluate.py` | timer baseline L_fix(τ), mode filter, HSMM-lite Viterbi, frame / boundary / segmental metrics |
| `train.py` | blocked time CV on the labeled stretch, per-fold τ* and HSMM fit, pooled out-of-fold evaluation |
| `events.py` | Opta events (databallpy `event_data.csv`) → frames (synced / clock + offset + snap), event possession, turnover table (how, where, dead ball) |
| `hazard.py` | turnover-anchored discrete-time hazard for transition duration (causal features, censoring); timer τ* on heuristic vs event possession; same blocked CV as `train.py` (PLAN §15) |

Ablation flags: `--no-graph`, `--no-org`, `--no-timer-feat`, `--temporal {bigru,tcn,none}`.

## Label mapping (reference team = home)
| CSV `Annotation_Category` | class |
|---|---|
| Attacking | 0 est_attack |
| Defending Transition (Counterattack, Delayed Attack) | 1 def_to_att |
| Defending | 2 est_defense |
| Attacking Transition (Counterpressing, Delay and recovery) | 3 att_to_def |

## Known limitations (v0)
- **One match, blocked CV inside it** → same teams in train and test; results are a smoke test, not evidence.
- The fused CSV's event columns are empty; events come from the separate databallpy export
  (`event_data.csv`, `--events`). Without it, possession falls back to the tracking heuristic
  (nearest player < 1.5 m, runs < 1 s merged), which agrees with labels on 92.3% of frames vs 95.4%
  for events and makes the timer baseline much weaker (macro-F1 0.54 vs 0.69, PLAN §15.2).
  Compare models only under the same possession source.
- HSMM is a min-duration chain approximation (geometric tail), not a full learned duration distribution.
- Not yet implemented from §6: Procrustes formation deviation, role-return cost, control-surface fragmentation, `mark` edges.
