# tools/adj — double annotation → adjudicated labels → evaluation

Protocol: STUDY1_PLAN §4.1, §4.2, §4.5, §4.7. Model side: §7.

| step | command | output |
|---|---|---|
| 1. convert to codebook v1, list clips | `python tools/adj/codebook_v1.py convert A.csv B.csv events.csv tracking.npz artifacts/adj_v1 --meta meta.json` | `labels_v1_25hz.csv` (incl. subcategories `subA`/`subB`), `clips.csv`, `restarts.csv`, `summary.json` (incl. pre-adjudication IRR of moments and subcategories) |
| 2. blinded pack for the third adjudicator | `python tools/adj/make_pack.py artifacts/adj_v1 tracking.npz events.csv artifacts/adj_v1/pack --meta meta.json --anim` | `pack/` (send this folder only), `key_DO_NOT_SHARE.json` (keep), `set_pieces_rule_end.csv` |
| 3. apply decisions | `python tools/adj/apply_decisions.py artifacts/adj_v1 pack/decisions.csv tracking.npz artifacts/adj_v1/final meta.json` | `labels_final_25hz.csv` (`sub` = subcategory kept only where the final moment is both coders' moment and their subcategories agree, else -1), `intervals.csv`, `qc.json`, `<prefix>_{adj,A,B,soft}.npz` (prefix from meta, e.g. `ajax_psv_full`; `sub` from these labels) |
| validate a new export | `python tools/adj/codebook_v1.py validate export.csv events.csv tracking.npz --meta meta.json` | violations; blanks are allowed only right after a corner / free kick ≤ 40 m (SET_PIECE is never annotated) |

`meta.json` (match id, team names and event team ids, kick-off frames, pitch, frame rate) is written by
`notebooks/preprocessing.ipynb` from the databallpy Game; see `phase/meta.py`.

Evaluation of model runs (v0 labels; switch the directory once step 3 is done):

```
python tools/adj/decode.py RUN_DIR            # possession-constrained decoding + ASRF -> RUN_DIR/decoded.npz
python tools/adj/decode_ed.py ADJ_DIR RUN_DIR # explicit-duration HSMM -> RUN_DIR/decoded.npz
# vs ADJ / A / B + human ceiling, with bootstrap intervals; --hz25 reads labels_final_25hz.csv (v1) or labels_25hz.csv (v0)
python tools/adj/eval_ceiling.py ADJ_DIR --prefix ajax_psv_full [--hazard ADJ_DIR/hazard] [--deep ADJ_DIR/deep] [name=RUN_DIR ...] [--hz25]
```

`build_labels.py` / `build_soft.py` built the provisional v0 labels in `artifacts/adj_v0` (dead ball and gaps masked,
no SET_PIECE); `check_s7.py` / `bnd_diag.py` are the one-off checks behind §7. These scripts were recovered from the
session transcript on 2026-10-26 after the temporary directory was cleared.
