"""Is the change in resolution a better predictor of the change in reliance than
the change in distribution shape?

IEEE Access Access-2026-52475, second-round comments R1-4, R2-2.

p1_mechanism_compare_v1.py reports Spearman(d log occupancy, d share) between
min-max and signed-log. Signed-log changes more than occupancy: it also lowers
skewness and tail weight, and how much it does so depends on how wide the raw
range is. A positive rho is therefore also expected if the model simply
relies more on features whose shape became less extreme. This script adds
those alternatives and asks whether occupancy keeps its association once they
are held fixed. No model is retrained or re-evaluated.

Per feature (training partition, continuous features kept by v1):
  dlog_occ      log occupancy_b - log occupancy_a              (from v1)
  dskew         skew_b - skew_a
  dlogkurt      log10 kurt_b - log10 kurt_a   (Pearson kurtosis, normal = 3)
  log_range     log10(max - min) of the unscaled feature   (no-scaling npz)

  occupancy = (Q3 - Q1) / (max - min) as in p1_mechanism_v1.py; skewness and
  kurtosis are invariant to the affine min-max map, so their "a" values are
  those of the unscaled feature.

Statistics, for each attribution indicator and each seed, and once more on
seed-mean feature values:
  Spearman rho of every predictor with d share
  partial Spearman of dlog_occ given each alternative, and given all three
  partial Spearman of each alternative given dlog_occ
  Spearman of dlog_occ with each alternative (collinearity)
Partial Spearman: Pearson correlation of rank residuals after least-squares
regression on the ranks of the controls; p from t with n - 2 - k df.

Reads   outputs/tables/mech_compare_features_{ds}_{a}_vs_{b}[_nocat].csv  (v1)
        data/processed/{ds}_full_scaler{a|b|none}.npz and _meta.json   (x_train only)
Writes  outputs/tables/mech_shape_{ds}.csv                     per-feature shape statistics
        outputs/tables/mech_compare_v2_{ds}_{a}_vs_{b}[_nocat].csv

Run     python -m src.p1_mechanism_compare_v2 --dataset ciciot2023
        python -m src.p1_mechanism_compare_v2 --dataset nfunsw
        python -m src.p1_mechanism_compare_v2 --dataset nfunsw --exclude-categorical
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

INDICATORS = ["perm_drop", "grad_x_input", "deg1_contrib"]
ALTS = ["dskew", "dlogkurt", "log_range"]


def shape_stats(dataset, scalers):
    """Skewness, Pearson kurtosis and range per feature for each scaler (x_train).

    Cached in outputs/tables/mech_shape_{dataset}.csv; the cache is used only
    if it holds every requested scaler.
    """
    cache = C.TABLE_DIR / f"mech_shape_{dataset}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if set(scalers) <= set(df.scaler.unique()):
            print(f"  [cache] {cache.name}")
            return df
    rows = []
    for sc in scalers:
        stem = f"{dataset}_full_scaler{sc}"
        meta = json.loads((C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
        names = meta["feature_names"]
        print(f"  [read] {stem}.npz x_train", flush=True)
        x = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)["x_train"]
        if x.shape[1] != len(names):
            raise SystemExit(f"[stop] {stem}: {x.shape[1]} columns, {len(names)} names")
        for j, f in enumerate(names):
            col = x[:, j].astype("float64")
            col = col[np.isfinite(col)]
            lo, hi = float(col.min()), float(col.max())
            sd = col.std()
            rows.append({
                "dataset": dataset, "scaler": sc, "feature": f,
                "skew": float(stats.skew(col)) if sd > 0 else np.nan,
                "kurt": float(stats.kurtosis(col, fisher=False)) if sd > 0 else np.nan,
                "min": lo, "max": hi,
            })
        del x
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f"  [save] {cache.name}")
    return df


def _ranks(v):
    return stats.rankdata(v)


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after removing the ranks of controls."""
    rx, ry = _ranks(x), _ranks(y)
    n = len(rx)
    if controls:
        Z = np.column_stack([np.ones(n)] + [_ranks(c) for c in controls])
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    k = len(controls)
    dof = n - 2 - k
    if dof < 1 or rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    t = r * np.sqrt(dof / max(1e-15, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), dof))
    return r, p


def analyse(k, label, out):
    """k: one row per feature with predictors and dshare_* columns."""
    preds = ["dlog_occ"] + ALTS
    for ind in INDICATORS:
        y = f"dshare_{ind}"
        ok = np.isfinite(k[[y] + preds].to_numpy(float)).all(axis=1)
        d = k[ok]
        n = len(d)

        def add(pred, ctrl, r, p):
            out.append({"seed": label, "indicator": ind, "predictor": pred,
                        "control": "+".join(ctrl) if ctrl else "-",
                        "n_features": n, "rho": r, "p": p})
        for pr in preds:                                   # zero-order
            r, p = partial_spearman(d[pr].to_numpy(), d[y].to_numpy(), [])
            add(pr, [], r, p)
        for alt in ALTS:                                   # occupancy | one alternative
            r, p = partial_spearman(d["dlog_occ"].to_numpy(), d[y].to_numpy(),
                                    [d[alt].to_numpy()])
            add("dlog_occ", [alt], r, p)
        r, p = partial_spearman(d["dlog_occ"].to_numpy(), d[y].to_numpy(),
                                [d[c].to_numpy() for c in ALTS])
        add("dlog_occ", ALTS, r, p)                        # occupancy | all three
        for alt in ALTS:                                   # alternative | occupancy
            r, p = partial_spearman(d[alt].to_numpy(), d[y].to_numpy(),
                                    [d["dlog_occ"].to_numpy()])
            add(alt, ["dlog_occ"], r, p)
    for alt in ALTS:                                       # collinearity
        ok = np.isfinite(k[["dlog_occ", alt]].to_numpy(float)).all(axis=1)
        r, p = stats.spearmanr(k.loc[ok, "dlog_occ"], k.loc[ok, alt])
        out.append({"seed": label, "indicator": "-", "predictor": "dlog_occ",
                    "control": f"vs {alt}", "n_features": int(ok.sum()),
                    "rho": float(r), "p": float(p)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--a", default="minmax")
    ap.add_argument("--b", default="signedlog")
    ap.add_argument("--exclude-categorical", action="store_true")
    a = ap.parse_args()

    tag = f"{a.dataset}_{a.a}_vs_{a.b}" + ("_nocat" if a.exclude_categorical else "")
    src = C.TABLE_DIR / f"mech_compare_features_{tag}.csv"
    if not src.exists():
        raise SystemExit(f"[stop] {src.name} missing. Run src.p1_mechanism_compare_v1 first.")
    feats = pd.read_csv(src)

    sh = shape_stats(a.dataset, [a.a, a.b, "none"])
    A = sh[sh.scaler == a.a].set_index("feature")
    B = sh[sh.scaler == a.b].set_index("feature")
    N = sh[sh.scaler == "none"].set_index("feature")
    per = pd.DataFrame({
        "dskew": B["skew"] - A["skew"],
        "dlogkurt": np.log10(B["kurt"]) - np.log10(A["kurt"]),
        "log_range": np.log10((N["max"] - N["min"]).where(lambda s: s > 0)),
    })
    miss = sorted(set(feats.feature) - set(per.index))
    if miss:
        raise SystemExit(f"[stop] features without shape statistics: {miss}")
    feats = feats.merge(per, left_on="feature", right_index=True, how="left")

    out = []
    for s, k in feats.groupby("seed"):
        analyse(k, int(s), out)
    cols = ["dlog_occ"] + ALTS + [f"dshare_{c}" for c in INDICATORS]
    analyse(feats.groupby("feature")[cols].mean().reset_index(), "mean", out)
    res = pd.DataFrame(out)
    dest = C.TABLE_DIR / f"mech_compare_v2_{tag}.csv"
    res.to_csv(dest, index=False)

    seeds = sorted(int(s) for s in feats.seed.unique())
    print(f"\n{'=' * 82}\n  {tag}   seeds {seeds}   features "
          f"{feats.groupby('seed').size().iloc[0]}\n{'=' * 82}")
    per_seed = res[res.seed != "mean"]
    mean_row = res[res.seed == "mean"]
    for ind in INDICATORS:
        print(f"\n  d share({ind})")
        print(f"  {'predictor':12s}{'control':28s}{'rho per seed (mean, min..max)':>32s}"
              f"{'seed-mean rho':>15s}{'p':>9s}")
        q = per_seed[per_seed.indicator == ind]
        for (pr, ct), g in q.groupby(["predictor", "control"], sort=False):
            m = mean_row[(mean_row.indicator == ind) & (mean_row.predictor == pr)
                         & (mean_row.control == ct)].iloc[0]
            print(f"  {pr:12s}{ct:28s}{g.rho.mean():>+12.3f} "
                  f"({g.rho.min():+.3f}..{g.rho.max():+.3f})"
                  f"{m.rho:>+15.3f}{m.p:>9.4f}")
    col = mean_row[mean_row.indicator == "-"]
    print("\n  collinearity, seed-mean features: "
          + ", ".join(f"{r.control[3:]} rho={r.rho:+.3f}" for _, r in col.iterrows()))
    print(f"\n  [save] {dest.name}\n")


if __name__ == "__main__":
    main()
