# Reproducing the resubmitted paper (IEEE Access Access-2026-52475)

"Feature Scaling in Arithmetic-Combination Intrusion Detectors: Empirical Limits of Affine Normalization"

This file describes the **second resubmission** (release v2.1.0). It supersedes the v2.0.1
version of this file and the three-seed instructions in `REPRODUCE_paper1_scaling.md`
(release v1.0.0), which is kept for reference.

## Environment
Python 3.11.15, NumPy 1.26.4, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.17.1, TensorFlow 2.16.2 (CPU).
Exact package versions: `requirements-lock.txt` (pip) and `environment-lock.yml` (conda).
All batch files activate the conda environment `dcfnet-ds` before running.

## Protocol
| Condition | Seeds | Script |
|---|---|---|
| Interaction model (K = 4), 47 features: min–max, signed-log, signed-log + min–max, no scaling | 1–10 | `run_1a.bat` |
| Interaction model, 47 features: z-score and the five additional transforms | 1–3 | `run_1a.bat`, v1.0.0 scripts |
| Interaction model, NF-UNSW-NB15-v3 with 42 features (nominal codes and bitmasks dropped) and 191 features (one-hot and per-bit encoding): min–max, signed-log, signed-log + min–max | 1–10 | `run_1e.bat`, then `run_1g.bat` (see *Corrections*) |
| Same, z-score | 1–3 | v1.0.0 scripts |
| 42 features, fixed budget of 40 epochs, early stopping off (Table IX, "42 feat., 40 ep.") | 1–3 | `src/config_old.py` (EPOCHS = 40, EARLY_STOP_PATIENCE = 999); results in `outputs/tables/budget40/` |
| HistGradientBoosting, random forest (full training partition) | 1–3 | `run_1b.bat` |
| Logistic regression (lbfgs, max 10,000 iterations; deterministic) | one run (seed 1) | `run_1c.bat` |
| Feature-level attribution | 1–3 | `run_1d.bat` |
| Resolution versus distribution shape (no retraining) | 1–3 | `run_1f.bat` |

All interaction-model runs except the 40-epoch budget arm use early stopping on validation loss
(patience 20, ReduceLROnPlateau, restore best weights) with a ceiling of 500 epochs
(`src/config.py`). Runs that reach the ceiling are reported in Section V-I.

Splits: CICIoT2023 stratified 70/30 (seed 1); NF-UNSW-NB15-v3 chronological 70/30 on
`FLOW_START_MILLISECONDS` (stable sort; boundary 1424237460039). A fixed stratified 20 % of each
training partition (seed 1) is the validation set in every run.

On the machine and environment above, retraining reproduces the per-seed PR-AUC values to the
fourth decimal (checked for the nine 42-feature runs of seeds 1–3, `run_1g.bat`).

## Tables and figures
| Output | Script |
|---|---|
| Tables II, III (47 features), V; paired statistics (paired t, Holm within dataset over the two primary contrasts, exact sign-flip permutation, d_z) | `python -m src.p1_table1_v1`, `python -m src.p1_table4_v1` (uses `src/p2_paired_stats_v1.py`) |
| Table III, 42- and 191-feature rows (ten seeds; Holm within encoding over the two primary contrasts) | `python -m src.p1_encoding10_stats_v1 --seeds 10` |
| Table IX (three-seed means, four configurations) | `outputs/tables/seeds_nfunsw_{full_scaler,catdrop_,catenc_}*.csv`, `outputs/tables/budget40/` |
| Section VI-A and Table X, resolution versus skewness, kurtosis and raw range (partial Spearman) | `python -m src.p1_mechanism_compare_v2` (after `src/p1_mechanism_v1.py`, `src/p1_mechanism_compare_v1.py`) |
| Figs. 2 and 5 | `python -m src.p1_figures_resub_v1` |
| Supplementary per-seed workbook (S1) | `python -m src.p1_supplementary_v1` |
| Remaining tables and figures | unchanged v1.0.0 scripts |

## Verification without retraining
The per-seed result files used in the paper are in `outputs/tables/`.

```
python -m src.p1_verify_resub_v2
```
recomputes the values of Tables II, III (including the encoding rows), V, IX and X and the
span excluding no scaling, and prints PASS/FAIL against the printed values (exit code 0 when
all pass). `src/p1_verify_resub_v1.py` is the v2.0.1 check and covers Tables II, III and V only.

`MANIFEST_sha256.txt` lists the SHA-256 of every file in the release, so the GitHub release,
the Zenodo archive and the supplementary ZIP can be checked for identity.

## Corrections to v2.0.1
- The files `outputs/tables/seeds_nfunsw_catdrop_*.csv` and `history_nfunsw_catdrop_*_m4*.csv`
  released in v2.0.1 held the 40-epoch budget runs, not the early-stopping runs printed in the
  "42 feat." column of the paper. The 40-epoch runs had been trained later under the same file
  names. In v2.1.0 the 40-epoch files are in `outputs/tables/budget40/`, and the 42-feature
  early-stopping runs (seeds 1–10) are under the original names. `src/p1_catdrop_protocol_v1.py`
  performed the separation and confirmed that the retrained seeds 1–3 equal the printed values.
- `outputs/tables/seeds_nfunsw_catdrop_standard.csv` (z-score) is unchanged: every z-score run
  stops before 40 epochs, so both protocols give the same values.
