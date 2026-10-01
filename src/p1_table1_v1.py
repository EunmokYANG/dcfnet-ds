"""수정본 Table II (V5 Table I) 교체: 스케일러별 PR-AUC / F1 / MCC 평균 +- 표준편차.

읽는 파일: outputs/tables/seeds_{ds}_full_scaler*.csv
산출물  : outputs/tables/p1_table1_perf.csv
실행    : python -m src.p1_table1_v1
"""

import argparse
import glob
import re

import pandas as pd

from src import config as C
from src.p1_table4_v1 import NAME, ORDER, DS_LABEL, scaler_of


def find_cols(cols):
    """PR-AUC, F1, MCC 열 이름을 찾는다. macro 가 붙은 것을 우선한다."""
    found = []
    for key in ["pr-auc", "f1", "mcc"]:
        hit = [c for c in cols if key in c.lower()]
        hit = sorted(hit, key=lambda c: ("macro" not in c.lower(), len(c)))
        if hit:
            found.append(hit[0])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m4")
    a = ap.parse_args()

    rows, used = [], set()
    for ds in C.DATASETS:
        for path in glob.glob(str(C.TABLE_DIR / f"seeds_{ds}_full_scaler*.csv")):
            fname = path.replace("\\", "/").split("/")[-1]
            m = re.match(rf"^seeds_{ds}_(.+)\.csv$", fname)
            sc = scaler_of(m.group(1)) if m else None
            if sc not in NAME:
                continue
            df = pd.read_csv(path)
            if "Variant" in df.columns:
                df = df[df.Variant == a.variant]
            if df.empty:
                continue
            cols = find_cols(df.columns)
            used.update(cols)
            rec = {"dataset": ds, "scaler": sc,
                   "n_seeds": int(df["seed"].nunique()) if "seed" in df.columns else len(df)}
            for c in cols:
                rec[f"{c}|mean"] = float(df[c].mean())
                rec[f"{c}|sd"] = float(df[c].std(ddof=1))
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("[중단] seeds 파일이 없습니다.")
    out["k"] = out.scaler.map({s: i for i, s in enumerate(ORDER)})
    out = out.sort_values(["dataset", "k"]).drop(columns="k")
    out.to_csv(C.TABLE_DIR / "p1_table1_perf.csv", index=False)

    print(f"\n  사용한 열: {sorted(used)}")
    for ds in C.DATASETS:
        sub = out[out.dataset == ds]
        if sub.empty:
            continue
        print(f"\n  {DS_LABEL.get(ds, ds)}")
        for r in sub.itertuples(index=False):
            rd = r._asdict()
            line = f"    {NAME[rd['scaler']]:14s} n={rd['n_seeds']:2d}"
            for c in sorted(used):
                mk, sk = f"{c}|mean", f"{c}|sd"
                if mk in out.columns and pd.notna(sub.loc[sub.scaler == rd['scaler'], mk]).all():
                    mv = sub.loc[sub.scaler == rd['scaler'], mk].iloc[0]
                    sv = sub.loc[sub.scaler == rd['scaler'], sk].iloc[0]
                    line += f"   {c}: {mv:.4f} +- {sv:.4f}"
            print(line)
    print("\n  [save] p1_table1_perf.csv")


if __name__ == "__main__":
    main()