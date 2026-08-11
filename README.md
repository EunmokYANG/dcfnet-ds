# DCFNet-DS — exact polynomial-degree decomposition for network intrusion detection

Code and result tables for

> **Polynomial-Degree Decomposition of a Network Intrusion Detector: Exactness
> and the Accuracy–Auditability Trade-off**
> Eunmok Yang, Jiyong Park, Changho Seo.
> Department of Convergence Science, Kongju National University.
> Submitted to *IEEE Access*. ⟨DOI once published⟩

The detector separates each class logit into an exact sum of homogeneous
polynomial terms, one per degree, so that

```
logit_c(x) = phi_1,c(x) + ... + phi_K,c(x) + b_c
```

holds as an identity rather than as a post-hoc approximation. This repository
contains everything needed to reproduce the tables and figures of the paper.

---

## 1. What is and is not here

| | |
| --- | --- |
| Included | preprocessing, model, training, evaluation and analysis scripts; the per-seed and aggregated result tables under `outputs/tables/`; the paper-2 tables and figures under `outputs/paper2_degree/` |
| Not included | the datasets, which their authors distribute (§2); the trained checkpoints, several hundred files that no analysis needs and that §4 regenerates |

---

## 2. Datasets

Neither dataset is redistributed here. Download each from its source and place
the CSV in `data/source/`.

| Dataset | File expected | Source |
| --- | --- | --- |
| CICIoT2023 | `data/source/CICIoT2023.csv` | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| NF-UNSW-NB15-v3 | `data/source/NF-UNSW-NB15-v3.csv` | NetFlow v3 datasets, University of Queensland |

The paper uses both at full scale with their natural class distribution:
1,176,851 flows and 43 features on CICIoT2023, and 2,242,931 flows and 47
features on NF-UNSW-NB15-v3 after the identifier columns are dropped.

---

## 3. Environment

TensorFlow 2.16.2 with Keras 3.15.1, CPU only. The runs behind the paper were
made on an Intel Core i9-7940X with 34 GB of memory under Windows 10; one
training run takes roughly 15 to 20 minutes and stops after 68 to 103 epochs.

```bash
conda env create -f environment.yml
conda activate dcfnet-ds
python -m src.check_env
```

`src/config.py` holds every path and hyperparameter. Nothing else needs editing.

---

## 4. Reproducing the paper

Run the stages in order. Each writes into `outputs/` and later stages read
what earlier ones produced.

### 4.1 Data preparation and preprocessing

```bash
python -m src.prepare_data --dataset ciciot2023 --max-per-class 0
python -m src.prepare_data --dataset nfunsw     --max-per-class 0

python -m src.preprocess --dataset ciciot2023 --tag full --full --scaler signedlog --split random
python -m src.preprocess --dataset nfunsw     --tag full --full --scaler signedlog --split temporal
```

`--max-per-class 0` keeps the original distribution; the paper does not cap
classes. The tag `full` is the base every later command refers to.

### 4.2 Training and evaluation

```bash
# variant comparison, five shared seeds
python -m src.multiseed --dataset ciciot2023 --tag full --variants m4 m4g m4m nn_mlp --seeds 5
python -m src.multiseed --dataset nfunsw     --tag full --variants m4 m4g m4m nn_mlp --seeds 5

# tree and linear baselines
python -m src.baselines --dataset ciciot2023 --tag full
python -m src.baselines --dataset nfunsw     --tag full

# degree sweep, three seeds
python -m src.sweep --kind degree --values 2 3 4 5 6 --variants m4 --seeds 3 --base-tag full
```

`multiseed` merges into `outputs/tables/seeds_{dataset}_full.csv`, so running
one variant later does not delete the rows of another. Add `--reuse` to
evaluate existing checkpoints instead of retraining.

### 4.3 Analysis

```bash
# degree contributions, identity and homogeneity checks
python -m src.degree_analysis --dataset ciciot2023 --variant m4  --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4  --tag full --max-degree 4
python -m src.degree_analysis --dataset ciciot2023 --variant m4g --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4g --tag full --max-degree 4
python -m src.degree_analysis --dataset ciciot2023 --variant m4m --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4m --tag full --max-degree 4

# paired significance tests
python -m src.p2_paired_stats_v1 --dataset ciciot2023 --tag full
python -m src.p2_paired_stats_v1 --dataset nfunsw     --tag full

# closed-form monomial coefficients
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/ciciot2023_full_m4.keras \
       --K 4 --features data/processed/ciciot2023_full_meta.json --topn 8
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/nfunsw_full_m4.keras \
       --K 4 --features data/processed/nfunsw_full_meta.json --topn 8 \
       --out outputs/paper2_degree/tables/p2_table6_coeff_attribution_nfunsw.csv

# arity within each degree
python -m src.p2_arity_v1

# perturbation test
python -m src.perturbation --dataset ciciot2023 --tag full --variant m4 --n-feat 3 --n-control 30
python -m src.perturbation --dataset nfunsw     --tag full --variant m4 --n-feat 3 --n-control 30

# figures
python -m src.p2_figures_v2 --base-tag full
```

---

## 5. Where each table and figure comes from

| Item | Produced by | Output |
| --- | --- | --- |
| Table I — baseline settings | `src/config.py`, `src/baselines.py` | read from the source |
| Table II — exactness | `degree_analysis --variant m4` | `purity_{ds}_full_m4.csv`, identity residual printed |
| Table III — gated variant | `degree_analysis --variant m4g` | `purity_{ds}_full_m4g.csv` |
| Table IV — degree sweep | `sweep --kind degree` | `sweep_degree.csv` |
| Table V — detection performance | `multiseed`, `baselines` | `summary_{ds}_full.csv`, `metrics_baseline_{ds}_full.csv` |
| Table VI — ablation significance | `p2_paired_stats_v1` | `p2_paired_{ds}_full.csv` |
| Table VII — non-decomposable share | `degree_analysis --variant m4m` | `degree_contrib_{ds}_full_m4m.csv` |
| Table VIII — largest coefficients | `p2_coeff_attribution_v1` | `p2_table6_coeff_attribution*.csv` |
| Table IX — arity by degree | `p2_arity_v1` | `p2_table9_arity.csv` |
| Figure 1 — architecture | drawn, not computed | `figures/Figure1_architecture.svg` |
| Figure 2 — experimental pipeline | drawn, not computed | `figures/Figure2_pipeline.svg` |
| Figure 3 — degree sweep | `p2_figures_v2 --only 1` | `p2_fig1_k_sweep.pdf` |
| Figure 4 — accuracy vs decomposability | `p2_figures_v2 --only 2` | `p2_fig2_tradeoff.pdf` |
| Figure 5 — arity by degree | `p2_arity_v1` | `p2_fig5_arity.pdf` |
| Figure 6 — class × degree heat map | `p2_figures_v2 --only 4` | `p2_fig4_class_degree_heatmap.pdf` |

Figure numbering follows the paper; the script filenames keep the numbering
they had while the figures were being produced.

---

## 6. Parameter counts

```bash
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/ciciot2023_full_m4.keras --summary
```

Reports trainable, non-trainable and total counts. The closed form in the
paper is `(K-1)D^2 + KCD + C + 2K`, the last term being the running
root-mean-square and step counter each DegreeScale layer keeps.

---

## 7. Two things worth knowing before rerunning

**Seed counts differ by experiment.** The degree sweep uses three seeds and
the variant comparison uses five. They are separate runs, which is why their
K = 4 entries differ slightly; neither difference exceeds one standard
deviation. Table IV and Table V of the paper state this.

**The validation split does not depend on the seed.** `train.py` draws it with
a fixed random state, so the seed changes weight initialization, batch
ordering and the non-determinism of parallel CPU reduction, but not the data
partition. Reported standard deviations therefore understate the variability
that resampling the split would add. Section IV-D of the paper says so.

---

## 8. Repository layout

```
src/                    all code
  config.py             paths, datasets, hyperparameters, variant definitions
  prepare_data.py       source CSV -> raw CSV
  preprocess.py         scaling, splitting, identifier removal -> npz
  model.py              DCFNet-DS and the baselines
  train.py              one run
  multiseed.py          repeated runs, merged into seeds_*.csv
  sweep.py              condition sweeps (degree, scaler, split, amplify)
  baselines.py          logistic regression, random forest, HistGradientBoosting
  evaluate.py           metrics
  degree_analysis.py    per-degree contributions, identity and homogeneity
  perturbation.py       selectivity test for the recovered combinations
  compare.py            pairwise tests over the seed table
  p1_figures_v1.py      figures and tables of the companion scaling paper
  p2_figures_v2.py      figures of this paper
  p2_coeff_attribution_v1.py   closed-form monomial coefficients
  p2_arity_v1.py        interaction arity within each degree
  p2_paired_stats_v1.py paired significance tests
  p2_verify_numbers_v1.py      seed counts and single comparisons
data/                   not tracked; see §2
outputs/
  tables/               shared experiment tables
  paper1_scaler/        companion paper deliverables
  paper2_degree/        this paper's tables and figures
  models/               not tracked; regenerate with §4.2
```

---

## 9. Citing

If you use this code, please cite the archived release:

> E. Yang, C. Seo, "DCFNet-DS: exact polynomial-degree
> decomposition for network intrusion detection," v1.0.0, Zenodo, 2026.
> https://doi.org/10.5281/zenodo.21883692

The accompanying paper citation will be added once the DOI is assigned.

## 10. License

MIT. See `LICENSE`.
