"""재투고 보충자료 S1: 모든 표의 시드별 원자료를 한 파일로 모은다.

IEEE Access Access-2026-40899 (R4-1 시드별 값, R1-5 / R3-7 재현성).

읽는 파일 (이미 저장된 것만 모은다. 재계산하지 않는다)
  outputs/tables/seeds_{ds}_full_scaler*.csv          m4 시드별 (Table II, III)
  outputs/tables/metrics_baseline_{ds}_full_scaler*_s*.csv, seeds_baseline_*.csv
                                                      대조 모델 시드별 (Table V)
  outputs/tables/p1_summary_gain.csv                  Table III
  outputs/tables/p1_table4_span.csv, p1_table4_cells.csv   Table V
  outputs/tables/p1_table1_perf.csv                   Table II
  outputs/tables/mech_compare_*.csv                   §VI 메커니즘 점검

산출물
  outputs/tables/Supplementary_S1_per_seed.xlsx   (openpyxl 이 있으면)
  outputs/tables/supplementary_S1/*.csv           (항상)

실행:
  python -m src.p1_supplementary_v1
"""

import glob
import os

import pandas as pd

from src import config as C
from src.p1_table4_v1 import load_proposed, load_baselines, NAME, DS_LABEL

STATIC = [
    ("T2_perf_by_scaler", "p1_table1_perf.csv"),
    ("T3_paired_summary", "p1_summary_gain.csv"),
    ("T5_cells", "p1_table4_cells.csv"),
    ("T5_span", "p1_table4_span.csv"),
]


def main():
    sheets = {}

    prop = load_proposed("m4")
    base = load_baselines()
    seeds = pd.concat([prop, base], ignore_index=True)
    seeds["dataset"] = seeds.dataset.map(DS_LABEL).fillna(seeds.dataset)
    seeds["scaler"] = seeds.scaler.map(NAME).fillna(seeds.scaler)
    seeds = seeds.sort_values(["dataset", "model", "scaler", "seed"])
    keep = [c for c in ["dataset", "model", "scaler", "seed", "score", "converged", "n_iter"]
            if c in seeds.columns]
    sheets["S1_per_seed_PRAUC"] = seeds[keep].rename(columns={"score": "PR-AUC(macro)"})

    for name, fn in STATIC:
        path = C.TABLE_DIR / fn
        if path.exists():
            sheets[name] = pd.read_csv(path)
        else:
            print(f"  [skip] {fn} 없음")

    for path in sorted(glob.glob(str(C.TABLE_DIR / "mech_compare_*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        short = stem.replace("mech_compare_", "M_").replace("_minmax_vs_signedlog", "")
        sheets[short[:31]] = pd.read_csv(path)

    out_dir = C.TABLE_DIR / "supplementary_S1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in sheets.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    xlsx = C.TABLE_DIR / "Supplementary_S1_per_seed.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name[:31], index=False)
        print(f"  [save] {xlsx}")
    except ImportError:
        print("  [info] openpyxl 이 없어 xlsx 는 건너뜀 (pip install openpyxl)")

    print(f"  [save] {out_dir}  ({len(sheets)} 개 시트)")
    for name, df in sheets.items():
        print(f"    {name:28s} {len(df):6d} rows")


if __name__ == "__main__":
    main()
