"""다중 시드 결과의 통계적 비교.

multiseed 가 저장한 seeds_{stem}.csv 를 읽어 모델 쌍별로 Welch t-검정을
수행한다. 재학습이 필요 없다.

단순히 "모델 간 최대 차이 vs 최대 표준편차" 로 판정하면 가장 불안정한
모델의 분산이 전체 판정을 좌우해 잘못된 결론이 나온다. 쌍별 검정이 옳다.

실행:  python -m src.compare --dataset ciciot2023 --tag full
       python -m src.compare --dataset ciciot2023 --tag full --metric F1-score
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="PR-AUC(macro)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}" if a.tag else a.dataset
    path = C.TABLE_DIR / f"seeds_{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 먼저 src.multiseed 를 실행하세요.")
    df = pd.read_csv(path)
    m = a.metric

    print(f"\n{'=' * 74}\n  {stem}   지표: {m}\n{'=' * 74}")
    summ = df.groupby("Variant")[m].agg(["mean", "std", "count"])
    summ = summ.sort_values("mean", ascending=False)
    print(f"\n  {'모델':<10}{'평균':>10}{'표준편차':>12}{'시드수':>8}")
    for v, r in summ.iterrows():
        print(f"  {v:<10}{r['mean']:>10.4f}{r['std']:>12.4f}"
              f"{int(r['count']):>8d}")

    print(f"\n{'=' * 74}\n  쌍별 Welch t-검정 (alpha={a.alpha})\n{'=' * 74}")
    print(f"  {'비교':<22}{'차이':>9}{'t':>8}{'p_raw':>10}"
          f"{'Cohen d':>10}  판정")
    print("  " + "-" * 70)

    rows = []
    order = list(summ.index)
    for v1, v2 in combinations(order, 2):
        x = df.loc[df.Variant == v1, m].values
        y = df.loc[df.Variant == v2, m].values
        t, p = stats.ttest_ind(x, y, equal_var=False)
        d = cohens_d(x, y)
        sig = p < a.alpha
        mark = f"{v1} 우세" if sig and t > 0 else (
            f"{v2} 우세" if sig else "구분 불가")
        print(f"  {v1 + ' vs ' + v2:<22}{x.mean() - y.mean():>9.4f}"
              f"{t:>8.2f}{p:>10.4f}{d:>10.2f}  {mark}")
        rows.append({"A": v1, "B": v2, "diff": x.mean() - y.mean(),
                     "n_A": len(x), "n_B": len(y),
                     "t": t, "p_raw": p, "cohens_d": d,
                     "significant_raw": sig, "winner_raw": mark})

    if not rows:
        print(f"\n  비교할 쌍이 없다. {path.name} 에 변형이 "
              f"{len(summ)}개뿐이다: {list(summ.index)}")
        print("  변형이 둘 이상인 파일을 지정할 것 "
              "(예: --tag full). K 스윕 파일에는 m4 하나만 들어 있다.\n")
        return

    out = pd.DataFrame(rows)
    # The printed Bonferroni note was not reaching the CSV, so a reader of
    # the file saw only the uncorrected verdict. Both are stored now, and
    # the corrected one is the column the paper must quote.
    n_pairs = len(out)
    out["family_size"] = n_pairs
    out["p_bonferroni"] = (out["p_raw"] * n_pairs).clip(upper=1.0)
    out["significant_bonferroni"] = out["p_bonferroni"] < a.alpha
    out.to_csv(C.TABLE_DIR / f"compare_{stem}.csv", index=False)

    print(f"\n  위 표의 p 는 보정 전 값이다. 쌍이 {n_pairs}개이므로 "
          f"Bonferroni 기준은 p < {a.alpha / n_pairs:.5f} 이다.")
    strict = out[out["significant_bonferroni"]]
    if len(strict):
        print("  Bonferroni 보정 후에도 유의한 쌍:")
        for _, r in strict.iterrows():
            print(f"    {r['A']} vs {r['B']}  "
                  f"p_raw={r['p_raw']:.6f}  "
                  f"p_bonf={r['p_bonferroni']:.6f}  "
                  f"d={r['cohens_d']:.2f}  {r['winner_raw']}")
    else:
        print("  Bonferroni 보정 후 유의한 쌍 없음.")
    print(f"  논문 표 IV 는 p_bonferroni 열과 family_size={n_pairs} 를 "
          f"함께 명시할 것.")

    print(f"\n  [save] compare_{stem}.csv\n")


if __name__ == "__main__":
    main()
