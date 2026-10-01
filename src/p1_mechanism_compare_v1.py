"""1d 교차 스케일러 비교 - 해상도가 오른 피처일수록 모델이 더 쓰게 되는가.

IEEE Access Access-2026-40899 심사 대응 (R4 Comment 2).

p1_mechanism_v1.py 는 스케일러 하나 안에서 occupancy 와 기여도의 순위
상관을 잰다. 그러나 피처마다 원래 정보량이 다르므로 한 조건 안의 상관만으로는
"해상도가 원인" 이라고 말하기 어렵다. 이 스크립트는 같은 피처, 같은 시드를
min-max 와 signed-log 사이에서 짝지어 변화량끼리 비교한다.

    예측: signed-log 로 바꿀 때 occupancy 가 많이 오른 피처일수록
          모델 안에서 차지하는 기여 비중(share)도 많이 오른다.
          -> Spearman(dlog occupancy, d share) > 0

기여 비중은 모델마다 합이 1 이 되도록 정규화한다. 로짓 크기나 기준 성능이
스케일러마다 달라도 "모델이 어느 피처에 의존하는가" 만 비교하기 위해서다.
perm_drop 의 음수(셔플해도 성능이 오른 경우)는 0 으로 자른다.

읽는 파일 (p1_mechanism_v1.py 산출물, 재계산 없음)
  outputs/tables/mech_features_{ds}_full_scaler{A}_{variant}_s{seed}.csv
  outputs/tables/mech_features_{ds}_full_scaler{B}_{variant}_s{seed}.csv

산출물
  outputs/tables/mech_compare_{ds}_{A}_vs_{B}.csv          시드별 결과
  outputs/tables/mech_compare_features_{ds}_{A}_vs_{B}.csv 피처별 변화량

실행:
  python -m src.p1_mechanism_compare_v1 --dataset ciciot2023
  python -m src.p1_mechanism_compare_v1 --dataset nfunsw --exclude-categorical
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config as C

INDICATORS = ["grad_x_input", "perm_drop", "deg1_contrib"]


def load_condition(dataset, scaler, variant):
    pat = str(C.TABLE_DIR / f"mech_features_{dataset}_full_scaler{scaler}_{variant}_s*.csv")
    out = {}
    for path in glob.glob(pat):
        fn = path.replace("\\", "/").split("/")[-1]
        m = re.match(rf"^mech_features_{dataset}_full_scaler{scaler}_{variant}_s(\d+)\.csv$", fn)
        if m:
            out[int(m.group(1))] = pd.read_csv(path)
    return out


def shares(df, col):
    v = df[col].to_numpy(dtype="float64")
    v = np.where(np.isfinite(v), np.clip(v, 0.0, None), np.nan)
    s = np.nansum(v)
    return v / s if s > 0 else np.full_like(v, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--a", default="minmax", help="기준 스케일러")
    ap.add_argument("--b", default="signedlog", help="비교 스케일러")
    ap.add_argument("--exclude-categorical", action="store_true",
                    help="config 의 nominal_cols / bitmask_cols 를 제외 (R4 Comment 3)")
    a = ap.parse_args()

    A = load_condition(a.dataset, a.a, a.variant)
    B = load_condition(a.dataset, a.b, a.variant)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit(f"[중단] 두 조건에 공통 시드가 없습니다: {a.a}={sorted(A)}, {a.b}={sorted(B)}")

    spec = C.DATASETS[a.dataset]
    cat = set(spec.get("nominal_cols", [])) | set(spec.get("bitmask_cols", []))

    rows, feat_rows = [], []
    for s in seeds:
        fa, fb = A[s], B[s]
        m = fa.merge(fb, on="feature", suffixes=("_a", "_b"))
        if len(m) != len(fa) or len(m) != len(fb):
            print(f"  [주의] seed {s}: 피처 이름이 두 조건에서 일치하지 않음 "
                  f"({len(fa)}/{len(fb)} -> 공통 {len(m)})")
        for col in INDICATORS:
            m[f"share_{col}_a"] = shares(m.rename(columns={f"{col}_a": col}), col)
            m[f"share_{col}_b"] = shares(m.rename(columns={f"{col}_b": col}), col)
        # 연속형만 (이진 피처는 occupancy 정의가 무의미), 필요 시 범주형 제외
        keep = (m["n_unique_a"] > 2) & (m["occupancy_a"] > 0) & (m["occupancy_b"] > 0)
        if a.exclude_categorical:
            keep &= ~m["feature"].isin(cat)
        k = m[keep].copy()
        k["dlog_occ"] = np.log(k["occupancy_b"]) - np.log(k["occupancy_a"])
        row = {"dataset": a.dataset, "a": a.a, "b": a.b, "seed": s,
               "n_features": int(len(k)),
               "exclude_categorical": a.exclude_categorical}
        for col in INDICATORS:
            d = k[f"share_{col}_b"] - k[f"share_{col}_a"]
            k[f"dshare_{col}"] = d
            ok = np.isfinite(d) & np.isfinite(k["dlog_occ"])
            if ok.sum() >= 4 and np.nanstd(d[ok]) > 0:
                r, p = spearmanr(k.loc[ok, "dlog_occ"], d[ok])
                row[f"rho_{col}"], row[f"p_{col}"] = float(r), float(p)
            else:
                row[f"rho_{col}"], row[f"p_{col}"] = np.nan, np.nan
        rows.append(row)
        k["seed"] = s
        feat_rows.append(k[["seed", "feature", "occupancy_a", "occupancy_b", "dlog_occ"]
                           + [f"share_{c}_{x}" for c in INDICATORS for x in "ab"]
                           + [f"dshare_{c}" for c in INDICATORS]])

    res = pd.DataFrame(rows)
    feats = pd.concat(feat_rows, ignore_index=True)
    tag = f"{a.dataset}_{a.a}_vs_{a.b}" + ("_nocat" if a.exclude_categorical else "")
    res.to_csv(C.TABLE_DIR / f"mech_compare_{tag}.csv", index=False)
    feats.to_csv(C.TABLE_DIR / f"mech_compare_features_{tag}.csv", index=False)

    # 시드 평균 변화량으로 한 번 더 (피처 단위 집계)
    mean_feat = feats.groupby("feature")[["dlog_occ"] + [f"dshare_{c}" for c in INDICATORS]].mean()

    print(f"\n{'=' * 74}\n  {a.dataset}: {a.a} -> {a.b}   공통 시드 {seeds}"
          f"{'   (범주형 제외)' if a.exclude_categorical else ''}\n{'=' * 74}")
    print("  예측: occupancy 가 많이 오른 피처일수록 기여 비중도 많이 오른다 (rho > 0)\n")
    print(f"  {'지표':16s}{'시드별 rho 평균 ± sd':>24s}{'rho>0 시드':>12s}{'피처평균 rho':>14s}{'p':>10s}")
    for col in INDICATORS:
        r = res[f"rho_{col}"].dropna()
        mf = mean_feat[["dlog_occ", f"dshare_{col}"]].dropna()
        if len(mf) >= 4:
            rr, pp = spearmanr(mf["dlog_occ"], mf[f"dshare_{col}"])
        else:
            rr, pp = np.nan, np.nan
        print(f"  {col:16s}{r.mean():>+14.3f} ± {r.std(ddof=1) if len(r) > 1 else 0:.3f}"
              f"{f'{int((r > 0).sum())}/{len(r)}':>12s}{rr:>+14.3f}{pp:>10.4f}")
    print(f"\n  피처 수 {int(res['n_features'].iloc[0])} (연속형"
          f"{', 범주형 제외' if a.exclude_categorical else ''})")
    print("  상위 occupancy 증가 피처 5개 (시드 평균):")
    top = mean_feat.sort_values("dlog_occ", ascending=False).head(5)
    for f, r in top.iterrows():
        print(f"    {f:32s} dlog_occ={r['dlog_occ']:+.2f}  "
              f"dshare(gxi)={r['dshare_grad_x_input']:+.4f}  dshare(perm)={r['dshare_perm_drop']:+.4f}")
    print(f"\n  [save] mech_compare_{tag}.csv, mech_compare_features_{tag}.csv")


if __name__ == "__main__":
    main()