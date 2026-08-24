# Reproducing paper 1 — feature scaling

> **Feature Scaling Determines What Arithmetic-Combination Intrusion Detectors
> Can Learn: The Limits of Affine Normalization.**
> Eunmok Yang, Seokjin Choi, Changho Seo. Submitted to *IEEE Access*.

This file is self-contained. Nothing in it depends on the companion paper.

The experiment compares five input maps under an otherwise fixed protocol: the
detector is held at K = 4, the datasets are used at full scale with their
natural class distribution, and only the scaling changes.

| Key | Scaler |
| --- | --- |
| `minmax` | min–max to [0, 1] |
| `standard` | z-score |
| `signedlog` | `sign(x)·log(1+|x|)`, no fitted parameters |
| `signedlog_minmax` | signed-log followed by min–max |
| `none` | no scaling |

---

## Datasets

Neither dataset is redistributed here. Download each from its source and place
the CSV in `data/source/`.

| Dataset | File expected | Source |
| --- | --- | --- |
| CICIoT2023 | `data/source/CICIoT2023.csv` | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| NF-UNSW-NB15-v3 | `data/source/NF-UNSW-NB15-v3.csv` | NetFlow v3 datasets, University of Queensland |

Both are used at full scale: 1,176,851 flows and 43 features on CICIoT2023,
2,242,931 flows and 47 features on NF-UNSW-NB15-v3, after the identifier
columns are dropped.

## Environment

TensorFlow 2.16.2 with Keras 3.15.1, CPU only. TensorFlow has shipped no
native Windows GPU build since 2.10, so the GPUs in this machine are not used;
at this model size the CPU run is not the bottleneck in any case. The runs
behind the paper were made on an Intel Core i9-7940X with 34 GB of memory
under Windows 10; one training run takes roughly 15 to 20 minutes.

```bash
conda create -n dcfnet-ds python=3.11
conda activate dcfnet-ds
pip install -r requirements.txt
python -m src.check_env
```

`check_env.py` reports the versions it finds and stops if one is missing.

`src/config.py` holds every path and hyperparameter. Nothing else needs editing.

## Tags

Every experiment writes into shared tables keyed by a tag, and `src/sweep.py`
merges rows on (dataset, value, variant, tag) with `keep="last"`, so a run
under an existing tag replaces the rows already there. Keep the tags below as
they are.

This paper writes under three tag families: `oob_{scaler}` for the per-scaler
preprocessing and baselines, `full_scaler{scaler}` for the scaler sweep, and
`oob_{scaler}_degree{K}` for the degree sweep. The companion paper writes
`full` and `full_degree{K}`. The two sets do not overlap, and the degree sweeps
of the two papers coexist in one file, `outputs/tables/sweep_degree.csv`, only
because their base tags differ.

---

## Commands

Run in order. Each stage writes into `outputs/` and later stages read what
earlier ones produced.

### 1. Data preparation

```bash
python -m src.prepare_data --dataset ciciot2023 --max-per-class 0
python -m src.prepare_data --dataset nfunsw     --max-per-class 0
```

`--max-per-class 0` keeps the original distribution; the paper does not cap
classes.

### 2. Preprocessing and baselines, once per scaler

Five scalers × two datasets. CICIoT2023 has no usable time column and uses a
random split; NF-UNSW-NB15-v3 uses a temporal one.

```bash
# CICIoT2023, shown for one scaler; repeat for standard, signedlog,
# signedlog_minmax and none
python -m src.preprocess --dataset ciciot2023 --tag oob_minmax --full --scaler minmax --split random
python -m src.baselines  --dataset ciciot2023 --tag oob_minmax

# NF-UNSW-NB15-v3, likewise
python -m src.preprocess --dataset nfunsw --tag oob_minmax --full --scaler minmax --split temporal
python -m src.baselines  --dataset nfunsw --tag oob_minmax
```

`preprocess` writes `data/processed/{dataset}_oob_{scaler}_meta.json`, which
carries the per-degree rms, the dynamic range and the range-exceedance counts
the paper reports. `baselines` writes
`outputs/tables/metrics_baseline_{dataset}_oob_{scaler}.csv`.

### 3. Scaler sweep

```bash
python -m src.sweep --kind scaler \
       --values minmax standard signedlog signedlog_minmax none \
       --variants m4 --seeds 3 --base-tag full
```

Writes `outputs/tables/sweep_scaler.csv`, one row per (dataset, scaler) with
the macro PR-AUC mean and standard deviation over three seeds, and preprocesses
each condition under the tag `full_scaler{scaler}` on the way. The paper reports
the five scalers listed above; the sweep accepts others, and rows for scalers
the paper does not discuss may be present in the file.

### 4. Degree dependence under two scalers

```bash
python -m src.sweep --kind degree --values 1 2 3 4 --variants m4 --seeds 3 --base-tag oob_minmax
python -m src.sweep --kind degree --values 1 2 3 4 --variants m4 --seeds 3 --base-tag oob_signedlog
```

Appends to `outputs/tables/sweep_degree.csv` under the tags
`oob_minmax_degree{K}` and `oob_signedlog_degree{K}`. This is the only stage
that touches a table the companion paper also writes to, which is why the base
tags differ from that paper's `full`.

### 5. Figures and tables

```bash
python -m src.p1_figures_v1              # all six figures and all five tables
python -m src.p1_figures_v1 --fig 3      # one figure only; tables are skipped
```

---

## Where each output comes from

`src/p1_figures_v1.py` produces every figure and table of the paper. Figures go
to `outputs/figures/paper1_scaler/`, tables to `outputs/tables/paper1_scaler/`.

| Output | Reads | Shows |
| --- | --- | --- |
| `p1_fig1_rms_curve.png` | `*_oob_{scaler}_meta.json` | rms of the degree-*k* term of the raw input under each scaler |
| `p1_fig2_prauc_bar.png` | `sweep_scaler.csv` | macro PR-AUC by scaler |
| `p1_fig3_cond_vs_perf.png` | both of the above | conditioning against performance |
| `p1_fig4_resolution_vs_perf.png` | both of the above | within-feature resolution against performance |
| `p1_fig5_architecture_sensitivity.png` | `metrics_baseline_*`, `sweep_scaler.csv` | scaler sensitivity of the trees, the linear model and the interaction model |
| `p1_fig6_degree_dependence.png` | `sweep_degree.csv`, tags `oob_minmax*` and `oob_signedlog*` | the scaler gap as a function of degree |
| `p1_table1.csv` | as figure 1 | dataset and protocol summary |
| `p1_table2.csv` | as figure 3 | scaler diagnostics against performance |
| `p1_table3.csv` | `*_oob_{scaler}_meta.json` | range exceedance at evaluation time |
| `p1_table4.csv` | as figure 5 | architecture sensitivity |
| `p1_table5.csv` | `sweep_scaler.csv` | Welch tests between scalers |

The figure numbers above are those of the paper.

---

## Two things worth knowing

**The validation split does not depend on the seed.** `train.py` draws it with
a fixed random state, so the seed changes weight initialization, batch ordering
and the non-determinism of parallel CPU reduction, but not the data partition.
Reported standard deviations therefore understate the variability that
resampling the split would add.

**Figure 1 is a statement about the input, not the model.** The curves are the
rms of the degree-*k* term computed on the raw features. The detector
normalizes each degree internally, so its own activations do not reach these
values; the curves show what the term sizes would be without that
normalization.
