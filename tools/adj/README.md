# tools/adj — double annotation → adjudicated labels → evaluation

Protocol: STUDY1_PLAN §4.1, §4.2, §4.5, §4.7. Model side: §7.

| step | command | output |
|---|---|---|
| 1. convert to codebook v1, list clips | `python tools/adj/codebook_v1.py convert A.csv B.csv events.csv tracking.npz artifacts/adj_v1` | `labels_v1_25hz.csv`, `clips.csv`, `restarts.csv`, `summary.json` (incl. pre-adjudication IRR) |
| 2. blinded pack for the third adjudicator | `python tools/adj/make_pack.py artifacts/adj_v1 tracking.npz events.csv artifacts/adj_v1/pack --anim` | `pack/` (send this folder only), `key_DO_NOT_SHARE.json` (keep), `set_pieces_rule_end.csv` |
| 3. apply decisions | `python tools/adj/apply_decisions.py artifacts/adj_v1 pack/decisions.csv tracking.npz artifacts/adj_v1/final` | `labels_final_25hz.csv`, `intervals.csv`, `qc.json`, `ajax_psv_full_{adj,A,B,soft}.npz` |
| validate a new export | `python tools/adj/codebook_v1.py validate export.csv events.csv tracking.npz` | violations; blanks are allowed only right after a corner / free kick ≤ 40 m (SET_PIECE is never annotated) |

Evaluation of model runs (v0 labels; switch the directory once step 3 is done):

```
python tools/adj/decode.py RUN_DIR            # possession-constrained decoding + ASRF -> RUN_DIR/decoded.npz
python tools/adj/decode_ed.py ADJ_DIR RUN_DIR # explicit-duration HSMM -> RUN_DIR/decoded.npz
python tools/adj/eval_ceiling.py ADJ_DIR name=RUN_DIR ... [--hz25]   # vs ADJ / A / B + human ceiling
```

`build_labels.py` / `build_soft.py` built the provisional v0 labels in `artifacts/adj_v0` (dead ball and gaps masked,
no SET_PIECE); `check_s7.py` / `bnd_diag.py` are the one-off checks behind §7. These scripts were recovered from the
session transcript on 2026-10-26 after the temporary directory was cleared.
