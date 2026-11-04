# phase_of_play

Segmenting football matches into phases of play (established attack, defence-to-attack transition,
established defence, attack-to-defence transition) from tracking and event data.

## Layout

| path | content |
|---|---|
| `phase/` | the pipeline: preprocessing, features, graph + temporal model, training with blocked CV, evaluation, turnover hazard baseline; the possession + transition-length line (episodes, error decomposition, C1–C4 rule baseline, end-point model). See [`phase/README.md`](phase/README.md). |
| `tools/adj/` | double annotation → adjudication pack → final labels, plus decoding, evaluation against the human ceiling and the better-than-a-human protocol (L1 / L2 / L3). See [`tools/adj/README.md`](tools/adj/README.md). |
| `tools/parse_events_to_df.py` | match event XML → roster and event CSVs |
| `notebooks/preprocessing.ipynb` | tracking–event synchronisation with databallpy |
| `tests/` | unit tests on synthetic matches (`tests/synth.py` builds fused CSVs, tracking dicts, event tables) |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest          # synthetic data only, no licensed match data needed (torch not required)
```

## Data

Match data (tracking, events, annotations) is licensed and **not included**. Put it under `data/`, `opta/`
or `比赛数据/`; outputs go to `artifacts/`. All of these are in `.gitignore`.
