"""스케일러 조건 간 통계 비교 (1a 실험용). 재학습 없음.

compare.py 는 한 태그 안의 모델 변형끼리만 비교한다.
이 스크립트는 같은 변형(기본 m4)을 스케일러 태그별 seeds_*.csv 에서 모아
조건 쌍마다 Welch t-검정, Cohen d, Bonferroni 보정을 수행한다.

실행:
  python -m src.compare_scalers --dataset nfunsw
  python -m src.compare_scalers --dataset ciciot2023
  python -m src.compare_scalers --dataset nfunsw --metrics "PR-AUC(macro)" MCC
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.compare import cohens_d

DEFAULT_SCALERS = ["minmax", "signedlog", "signedlog_minmax", "none"]
DEFAULT_METRICS = ["PR-AUC(macro)", "F1-score", "MCC", "Accuracy"]


def load_condition(dataset, prefix, scaler, variant):
    path = C.TABLE_DIR / f"seeds_{dataset}_{prefix}{scaler}.csv"
    if not path.exists():
        print(f"  [skip] {path.name} 없음")
        return None
    df = pd.read_csv(path)
    df = df[df["Variant"] == variant].copy()
    if df.empty:
        print(f"  [skip] {path.name} 에 {variant} 행 없음")
        return None
    df["Scaler"] = scaler
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--prefix", default="full_scaler",
                    help="태그 앞부분. 태그 = prefix + 스케일러 이름")
    ap.add_argument("--scalers", nargs="+", default=DEFAULT_SCALERS)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    parts = [load_condition(a.dataset, a.prefix, s, a.variant) for s in a.scalers]
    parts = [p for p in parts if p is not None]
    if len(parts) < 2:
        print("  비교할 조건이 2개 미만이다.")
        return
    df = pd.concat(parts, ignore_index=True)
    order = [s for s in a.scalers if s in set(df["Scaler"])]

    # ---------------- 요약표 (평균 +- 표준편차) ----------------
    g = df.groupby("Scaler")
    summ = pd.DataFrame({"n_seeds": g.size()})
    for m in a.metrics:
        summ[f"{m}_mean"] = g[m].mean()
        summ[f"{m}_std"] = g[m].std()
    summ = summ.loc[order].reset_index()
    stem = f"{a.dataset}_{a.variant}"
    summ.to_csv(C.TABLE_DIR / f"scalers_summary_{stem}.csv", index=False)

    print(f"\n{'=' * 78}\n  {a.dataset} / {a.variant}   스케일러 조건 요약\n{'=' * 78}")
    head = f"  {'Scaler':<18}{'n':>4}" + "".join(f"{m:>20}" for m in a.metrics)
    print(head)
    for _, r in summ.iterrows():
        cells = "".join(f"{r[f'{m}_mean']:>11.4f} ±{r[f'{m}_std']:<7.4f}" for m in a.metrics)
        print(f"  {r['Scaler']:<18}{int(r['n_seeds']):>4}{cells}")

    # ---------------- 쌍별 검정 (지표마다 별도 family) ----------------
    rows = []
    for m in a.metrics:
        print(f"\n{'=' * 78}\n  쌍별 Welch t-검정   지표: {m}\n{'=' * 78}")
        print(f"  {'비교':<36}{'차이':>9}{'t':>9}{'p_raw':>11}{'Cohen d':>10}")
        print("  " + "-" * 74)
        block = []
        for s1, s2 in combinations(order, 2):
            x = df.loc[df.Scaler == s1, m].values
            y = df.loc[df.Scaler == s2, m].values
            t, p = stats.ttest_ind(x, y, equal_var=False)
            d = cohens_d(x, y)
            print(f"  {s1 + ' vs ' + s2:<36}{x.mean() - y.mean():>9.4f}"
                  f"{t:>9.2f}{p:>11.2e}{d:>10.2f}")
            block.append({"metric": m, "A": s1, "B": s2,
                          "mean_A": x.mean(), "mean_B": y.mean(),
                          "diff": x.mean() - y.mean(), "n_A": len(x), "n_B": len(y),
                          "t": t, "p_raw": p, "cohens_d": d})
        n_pairs = len(block)
        for b in block:
            b["family_size"] = n_pairs
            b["p_bonferroni"] = min(1.0, b["p_raw"] * n_pairs)
            b["significant_bonferroni"] = b["p_bonferroni"] < a.alpha
            win = b["A"] if b["diff"] > 0 else b["B"]
            b["verdict"] = f"{win} 우세" if b["significant_bonferroni"] else "구분 불가"
        print(f"  Bonferroni (family={n_pairs}) 기준 p < {a.alpha / n_pairs:.5f}")
        for b in block:
            print(f"    {b['A'] + ' vs ' + b['B']:<34} p_bonf={b['p_bonferroni']:.2e}  {b['verdict']}")
        rows += block

    out = pd.DataFrame(rows)
    out.to_csv(C.TABLE_DIR / f"compare_scalers_{stem}.csv", index=False)
    print(f"\n  [save] scalers_summary_{stem}.csv, compare_scalers_{stem}.csv\n")


if __name__ == "__main__":
    main()