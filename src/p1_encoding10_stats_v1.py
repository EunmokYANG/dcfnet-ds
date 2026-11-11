"""Feature-encoding check on NF-UNSW-NB15-v3 with ten paired seeds.

IEEE Access Access-2026-52475, second-round comments R1-3, R2-3, R2-4.

The 42-feature (nominal codes and bitmasks dropped) and 191-feature (one-hot
and per-bit encoding) rows of Table III were based on three seeds. This script
recomputes them from the per-seed files after run_1e.bat has added seeds 4-10,
with the same statistics as the 47-feature rows (src.p1_table4_v1.paired_gain):

  primary contrasts (Holm within each encoding, prespecified):
      Signed-log - Min-Max,  SLog+MinMax - Min-Max
  diagnostic contrast (raw p):
      SLog+MinMax - Signed-log

The 47-feature rows are read from the main ten-seed files so the three
encodings are reported in one table under one protocol.

Reads    outputs/tables/seeds_nfunsw_catdrop_{scaler}.csv      42 features
         outputs/tables/seeds_nfunsw_catenc_{scaler}.csv       191 features
         outputs/tables/seeds_nfunsw_full_scaler{scaler}.csv   47 features
Writes   outputs/tables/p1_encoding10_stats.csv
         outputs/tables/p1_encoding10_cells.csv

Run      python -m src.p1_encoding10_stats_v1
         python -m src.p1_encoding10_stats_v1 --seeds 10   (stop if any cell has fewer)
"""

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src.p1_table4_v1 import paired_gain

METRIC = "PR-AUC(macro)"
SCALERS = ["minmax", "signedlog", "signedlog_minmax", "none"]
ENCODINGS = [  # (label, file stem prefix, n_features)
    ("47-feature", "nfunsw_full_scaler", 47),
    ("42-feature", "nfunsw_catdrop_", 42),
    ("191-feature", "nfunsw_catenc_", 191),
]
KEEP = ["Signed-log - Min-Max", "SLog+MinMax - Min-Max", "SLog+MinMax - Signed-log"]


def load_raw(variant):
    rows = []
    for label, prefix, nf in ENCODINGS:
        for sc in SCALERS:
            path = C.TABLE_DIR / f"seeds_{prefix}{sc}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "Variant" in df.columns:
                df = df[df.Variant == variant]
            for _, r in df.iterrows():
                rows.append({"dataset": label, "model": variant, "scaler": sc,
                             "seed": int(r["seed"]), "score": float(r[METRIC]),
                             "n_features": nf})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--seeds", type=int, default=None,
                    help="required seeds per cell; stop if a primary cell has fewer")
    a = ap.parse_args()

    raw = load_raw(a.variant)
    if raw.empty:
        raise SystemExit("[stop] no seeds_nfunsw_cat*.csv found in outputs/tables")

    cells = (raw.groupby(["dataset", "scaler"])["score"]
             .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="size")
             .reset_index())
    cells["seeds"] = [sorted(raw[(raw.dataset == d) & (raw.scaler == s)].seed.tolist())
                      for d, s in zip(cells.dataset, cells.scaler)]
    enc_order = {lab: i for i, (lab, _, _) in enumerate(ENCODINGS)}
    cells = cells.sort_values(["dataset", "scaler"],
                              key=lambda c: c.map(enc_order) if c.name == "dataset"
                              else c.map({s: i for i, s in enumerate(SCALERS)}))
    cells = cells.reset_index(drop=True)
    cells.to_csv(C.TABLE_DIR / "p1_encoding10_cells.csv", index=False)

    print(f"\n{'=' * 78}\n  Cells ({METRIC}, model {a.variant})\n{'=' * 78}")
    for _, r in cells.iterrows():
        print(f"  {r.dataset:12s} {r.scaler:17s} mean {r['mean']:.4f}  sd {r.sd:.4f}"
              f"  n {r.n:2d}  seeds {r.seeds}")

    if a.seeds:
        prim = cells[cells.scaler.isin(["minmax", "signedlog", "signedlog_minmax"])]
        short = prim[prim.n < a.seeds]
        if len(short):
            print("\n[stop] cells with fewer than", a.seeds, "seeds:")
            print(short[["dataset", "scaler", "n", "seeds"]].to_string(index=False))
            raise SystemExit(1)

    st = paired_gain(raw[raw.scaler != "none"], a.variant)
    st = st[st.contrast.isin(KEEP)].copy()
    order = {lab: i for i, (lab, _, _) in enumerate(ENCODINGS)}
    st["_o"] = st.dataset.map(order)
    st["_c"] = st.contrast.map({c: i for i, c in enumerate(KEEP)})
    st = st.sort_values(["_o", "_c"]).drop(columns=["_o", "_c"])
    st = st.rename(columns={"dataset": "encoding"})
    st.to_csv(C.TABLE_DIR / "p1_encoding10_stats.csv", index=False)

    print(f"\n{'=' * 78}\n  Paired contrasts (pp). * = Holm within encoding over the two "
          f"primary contrasts\n{'=' * 78}")
    print(f"  {'encoding':12s}{'contrast':26s}{'n':>4s}{'pos':>7s}{'d pp':>7s}"
          f"{'95% CI':>16s}{'t':>7s}{'d_z':>6s}{'p':>11s}{'p_perm':>10s}")
    for _, r in st.iterrows():
        p = (f"{r.p_holm:.2e}*" if r.role == "primary" and not np.isnan(r.p_holm)
             else f"{r.p_raw:.2e} ")
        print(f"  {r.encoding:12s}{r.contrast:26s}{r.n_seeds:>4d}"
              f"{str(r.n_pos) + '/' + str(r.n_seeds):>7s}{r.delta_pp:>7.2f}"
              f"{f'[{r.ci_lo_pp:.2f}, {r.ci_hi_pp:.2f}]':>16s}{r.t:>7.2f}{r.d_z:>6.2f}"
              f"{p:>11s}{r.p_perm_exact:>10.5f}")
    floor = st.p_perm_floor.min()
    print(f"\n  Smallest attainable exact sign-flip p at this n: {floor:.5f}")
    print("  [save] p1_encoding10_stats.csv, p1_encoding10_cells.csv\n")


if __name__ == "__main__":
    main()
