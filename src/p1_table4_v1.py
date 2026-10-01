"""Table IV 교체본과 결과 요약표를 만든다.

IEEE Access Access-2026-40899 심사 대응 (R3 Comment 2, R2 Comment 2/7).

기존 Table IV 는 베이스라인만 단일 시드 + 400,000행 부분표본이라
제안 모델의 span 과 같은 noise floor 위에서 비교할 수 없었다. 이 스크립트는
전체 학습 파티션에서 재실행한 결과만 모아 모델별 스케일러 span 을 계산하고,
시드 간 표준편차를 함께 실어 span/noise 비를 보고한다.

읽는 파일 (재계산하지 않고 이미 저장된 것만 모은다)
  outputs/tables/seeds_{ds}_{tag}.csv              m4 다중 시드
  outputs/tables/metrics_baseline_{ds}_{tag}_s*.csv  베이스라인 (전체 데이터)

옛 프로토콜의 metrics_baseline_{ds}_{tag}.csv (시드 접미사 없음) 는
400,000행 단일 시드 결과이므로 섞이지 않도록 일부러 제외한다.

산출물
  outputs/tables/p1_table4_span.csv     모델 x 데이터셋 span
  outputs/tables/p1_table4_cells.csv    모델 x 스케일러 셀 값
  outputs/tables/p1_summary_gain.csv    R2 요구 요약표 (개선폭, CI, p, n)

실행:
  python -m src.p1_table4_v1
  python -m src.p1_table4_v1 --variant m4
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.p2_paired_stats_v1 import paired_permutation_p

METRIC = "PR-AUC(macro)"

# 표기 순서와 이름. tag 의 full_scaler 접두어를 뗀 값이 키다.
SCALERS = [
    ("minmax", "Min-Max"),
    ("standard", "Z-score"),
    ("signedlog", "Signed-log"),
    ("signedlog_minmax", "SLog+MinMax"),
    ("none", "No scaling"),
]
NAME = dict(SCALERS)
ORDER = [s for s, _ in SCALERS]
DS_LABEL = {"ciciot2023": "CICIoT2023", "nfunsw": "NF-UNSW-NB15-v3"}


def scaler_of(tag):
    """full_scalerminmax -> minmax. 접두어가 없으면 None."""
    m = re.match(r"^full_scaler(.+)$", tag)
    return m.group(1) if m else None


def load_proposed(variant):
    """multiseed 가 남긴 seeds_*.csv 에서 제안 모델 행을 모은다."""
    rows = []
    for ds in C.DATASETS:
        for path in glob.glob(str(C.TABLE_DIR / f"seeds_{ds}_full_scaler*.csv")):
            tag = re.match(rf"^seeds_{ds}_(.+)\.csv$",
                           path.split("\\")[-1].split("/")[-1])
            if not tag:
                continue
            sc = scaler_of(tag.group(1))
            if sc not in NAME:
                continue
            df = pd.read_csv(path)
            if "Variant" not in df.columns or METRIC not in df.columns:
                continue
            df = df[df.Variant == variant]
            # PR-AUC(macro) 는 itertuples 에서 이름이 뭉개지므로 iterrows 를 쓴다.
            for _, r in df.iterrows():
                rows.append({"dataset": ds, "model": variant, "scaler": sc,
                             "seed": int(r["seed"]), "score": float(r[METRIC]),
                             "converged": None})
    return pd.DataFrame(rows)


def _baseline_row(ds, sc, seed, r, cols):
    return {
        "dataset": ds, "model": str(r["Variant"]), "scaler": sc,
        "seed": int(seed), "score": float(r[METRIC]),
        "converged": (None if "converged" not in cols or pd.isna(r["converged"])
                      else bool(r["converged"])),
        "n_iter": (None if "n_iter" not in cols or pd.isna(r["n_iter"])
                   else int(r["n_iter"])),
    }


def load_baselines():
    """전체 데이터 재실행분만 모은다. 두 곳을 함께 읽는다.

      metrics_baseline_{ds}_{tag}_s{seed}.csv   run() 이 시드별로 남긴 것
      seeds_baseline_{ds}_{tag}.csv             run_multi() 가 병합해 둔 것

    run() 은 그 호출에 포함된 모델만 담아 같은 (stem, seed) 파일을
    덮어쓴다. 모델을 나눠 실행하면 앞 모델의 행이 사라지므로
    (예: logreg 단독 실행이 hgb 의 시드 1 행을 지운다), 병합 로직이 있는
    seeds_baseline 쪽도 함께 읽고 중복을 제거해 복구한다.

    시드 접미어가 없는 metrics_baseline_{ds}_{tag}.csv 는 옛 프로토콜
    (400,000행 단일 시드) 이므로 일부러 제외한다.
    """
    rows = []
    for ds in C.DATASETS:
        # (1) 시드별 파일
        for path in glob.glob(
                str(C.TABLE_DIR / f"metrics_baseline_{ds}_full_scaler*_s*.csv")):
            fn = path.replace("\\", "/").split("/")[-1]
            m = re.match(rf"^metrics_baseline_{ds}_(.+)_s(\d+)\.csv$", fn)
            if not m:
                continue
            sc = scaler_of(m.group(1))
            if sc not in NAME:
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                rows.append(_baseline_row(ds, sc, m.group(2), r, df.columns))

        # (2) run_multi 가 병합해 둔 파일
        for path in glob.glob(
                str(C.TABLE_DIR / f"seeds_baseline_{ds}_full_scaler*.csv")):
            fn = path.replace("\\", "/").split("/")[-1]
            m = re.match(rf"^seeds_baseline_{ds}_(.+)\.csv$", fn)
            if not m:
                continue
            sc = scaler_of(m.group(1))
            if sc not in NAME:
                continue
            df = pd.read_csv(path)
            if "seed" not in df.columns:
                continue
            for _, r in df.iterrows():
                rows.append(_baseline_row(ds, sc, r["seed"], r, df.columns))

    df = pd.DataFrame(rows)
    if len(df):
        # 같은 실행이 두 파일에 모두 남으므로 중복을 정리한다.
        df = df.drop_duplicates(subset=["dataset", "model", "scaler", "seed"],
                                keep="first").reset_index(drop=True)
    return df


def cells_and_spans(raw):
    """(dataset, model, scaler) 평균표와 모델별 span 을 만든다."""
    g = raw.groupby(["dataset", "model", "scaler"]).agg(
        mean=("score", "mean"), sd=("score", "std"),
        n=("score", "size")).reset_index()
    g["sd"] = g["sd"].fillna(0.0)

    spans = []
    for (ds, mdl), sub in g.groupby(["dataset", "model"]):
        s = sub.set_index("scaler")
        best = s["mean"].idxmax()
        worst = s["mean"].idxmin()
        span = (s.loc[best, "mean"] - s.loc[worst, "mean"]) * 100
        noise = s["sd"].mean() * 100
        nc = raw[(raw.dataset == ds) & (raw.model == mdl)]
        n_notconv = int((nc["converged"] == False).sum()) \
            if "converged" in nc.columns else 0
        spans.append({
            "dataset": ds, "model": mdl, "n_scalers": len(s),
            "n_seeds_min": int(s["n"].min()), "n_seeds_max": int(s["n"].max()),
            "span_pp": round(span, 2),
            "noise_pp": round(noise, 2),
            "span_over_noise": (round(span / noise, 2) if noise > 0 else np.nan),
            "best_scaler": NAME[best], "best": round(s.loc[best, "mean"], 4),
            "worst_scaler": NAME[worst], "worst": round(s.loc[worst, "mean"], 4),
            "n_not_converged": n_notconv,
        })
    return g, pd.DataFrame(spans)


def paired_gain(raw, variant, ref="minmax"):
    """제안 모델의 시드 대응 개선폭. 95% CI, d_z, 양수 시드 수, Holm 보정 p 를 낸다.

    V5 Table II 의 사전 지정을 그대로 따른다.
      주 대비 (데이터셋별 2건, 데이터셋 안에서 Holm):
        signed-log - min-max, SLog+MinMax - min-max
      진단 대비 (보정 없음, raw p):
        SLog+MinMax - signed-log, min-max - z-score, z-score - no scaling
    """
    pairs = [("signedlog", ref, "primary"),
             ("signedlog_minmax", ref, "primary"),
             ("signedlog_minmax", "signedlog", "diagnostic"),
             (ref, "standard", "diagnostic"),
             ("standard", "none", "diagnostic")]
    sub = raw[raw.model == variant]
    out = []
    for ds in sorted(sub.dataset.unique()):
        d = sub[sub.dataset == ds]
        for sc, rf, role in pairs:
            base = d[d.scaler == rf].set_index("seed")["score"]
            cur = d[d.scaler == sc].set_index("seed")["score"]
            common = sorted(set(base.index) & set(cur.index))
            if len(common) < 2:
                continue
            diff = (cur.loc[common] - base.loc[common]).to_numpy() * 100
            n = len(diff)
            m = diff.mean()
            sd = diff.std(ddof=1)
            se = sd / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            t, p = stats.ttest_rel(cur.loc[common], base.loc[common])
            p_perm, p_floor = paired_permutation_p(diff)
            out.append({
                "dataset": ds, "model": variant,
                "contrast": f"{NAME[sc]} - {NAME[rf]}",
                "role": role,
                "n_seeds": n,
                "n_pos": int((diff > 0).sum()),
                "baseline": round(float(base.loc[common].mean()), 4),
                "proposed": round(float(cur.loc[common].mean()), 4),
                "delta_pp": round(float(m), 2),
                "delta_rel_pct": round(float(
                    m / (base.loc[common].mean() * 100) * 100), 1),
                "sd_diff_pp": round(float(sd), 2),
                "ci_lo_pp": round(float(m - tcrit * se), 2),
                "ci_hi_pp": round(float(m + tcrit * se), 2),
                "t": round(float(t), 2),
                "d_z": round(float(m / sd), 2) if sd > 0 else np.nan,
                "p_raw": float(p),
                "p_perm_exact": round(float(p_perm), 5),
                "p_perm_floor": round(float(p_floor), 5),
            })
    df = pd.DataFrame(out)
    if len(df):
        df["p_holm"] = np.nan
        for ds in df.dataset.unique():
            prim = df[(df.dataset == ds) & (df.role == "primary")]
            k = len(prim)
            if not k:
                continue
            idx = prim.sort_values("p_raw").index
            adj, run = [], 0.0
            for i, ix in enumerate(idx):
                run = max(run, (k - i) * df.loc[ix, "p_raw"])
                adj.append(min(run, 1.0))
            df.loc[idx, "p_holm"] = adj
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="m4", help="제안 모델 변형 이름")
    a = ap.parse_args()

    prop = load_proposed(a.variant)
    base = load_baselines()
    raw = pd.concat([prop, base], ignore_index=True)
    if raw.empty:
        raise SystemExit("[중단] 모을 결과 파일이 없습니다.")

    cells, spans = cells_and_spans(raw)
    gains = paired_gain(raw, a.variant)

    cells["scaler_name"] = cells.scaler.map(NAME)
    cells["k"] = cells.scaler.map({s: i for i, s in enumerate(ORDER)})
    cells = cells.sort_values(["dataset", "model", "k"]).drop(columns="k")
    cells.to_csv(C.TABLE_DIR / "p1_table4_cells.csv", index=False)
    spans.to_csv(C.TABLE_DIR / "p1_table4_span.csv", index=False)
    if len(gains):
        gains.to_csv(C.TABLE_DIR / "p1_summary_gain.csv", index=False)

    print(f"\n{'=' * 78}\n  Table IV 교체본 - 모델별 스케일러 span\n{'=' * 78}")
    print(f"{'dataset':16s}{'model':9s}{'span pp':>9s}{'noise pp':>10s}"
          f"{'span/noise':>12s}{'seeds':>7s}{'best':>16s}{'worst':>16s}")
    for ds in C.DATASETS:
        for r in spans[spans.dataset == ds].sort_values(
                "span_pp", ascending=False).itertuples():
            sd = (f"{r.n_seeds_min}" if r.n_seeds_min == r.n_seeds_max
                  else f"{r.n_seeds_min}-{r.n_seeds_max}")
            print(f"{DS_LABEL.get(ds, ds):16s}{r.model:9s}{r.span_pp:9.2f}"
                  f"{r.noise_pp:10.2f}{r.span_over_noise:12.2f}{sd:>7s}"
                  f"{r.best_scaler:>16s}{r.worst_scaler:>16s}")
        print()

    print(f"{'=' * 78}\n  셀 값 (평균 +- 표준편차, 시드 수)\n{'=' * 78}")
    for ds in C.DATASETS:
        sub = cells[cells.dataset == ds]
        if sub.empty:
            continue
        models = list(dict.fromkeys(sub.model))
        print(f"\n  {DS_LABEL.get(ds, ds)}")
        print("    " + f"{'scaler':14s}" + "".join(f"{m:>22s}" for m in models))
        for sc in ORDER:
            line = f"    {NAME[sc]:14s}"
            for m in models:
                c = sub[(sub.model == m) & (sub.scaler == sc)]
                line += (f"{c['mean'].iloc[0]:.4f}+-{c['sd'].iloc[0]:.4f}"
                         f"({int(c['n'].iloc[0])})").rjust(22) if len(c) \
                    else "-".rjust(22)
            print(line)

    if len(gains):
        print(f"\n{'=' * 78}\n  결과 요약표 - 개선폭과 유의성 (R2 Comment 2/7)\n{'=' * 78}")
        print(f"{'dataset':16s}{'contrast':28s}{'n':>3s}{'pos':>4s}{'base':>8s}{'prop':>8s}"
              f"{'d pp':>7s}{'d %':>7s}{'95% CI':>18s}{'t':>7s}{'d_z':>6s}"
              f"{'p':>11s}{'p_perm':>9s}{'role':>12s}")
        for r in gains.itertuples():
            p = (f"{r.p_holm:.2e}*" if r.role == "primary"
                 and not np.isnan(r.p_holm) else f"{r.p_raw:.2e}")
            print(f"{DS_LABEL.get(r.dataset, r.dataset):16s}{r.contrast:28s}"
                  f"{r.n_seeds:3d}{r.n_pos:4d}{r.baseline:8.4f}{r.proposed:8.4f}"
                  f"{r.delta_pp:+7.2f}{r.delta_rel_pct:+7.1f}"
                  f"{f'[{r.ci_lo_pp:+.2f}, {r.ci_hi_pp:+.2f}]':>18s}"
                  f"{r.t:7.2f}{r.d_z:6.2f}"
                  f"{p:>11s}{r.p_perm_exact:9.5f}{r.role:>12s}")
        print("\n  * 는 Holm 보정 (데이터셋별 주 대비 2건, V5 사전 지정과 동일). "
              "진단 대비는 보정하지 않은 raw p 이다.")

    nc = spans[spans.n_not_converged > 0]
    if len(nc):
        print(f"\n  [주의] 미수렴 조건이 있는 모델: "
              f"{', '.join(nc.model + '(' + nc.n_not_converged.astype(str) + ')')}"
              f" - 해당 값은 하한으로 보고할 것")

    print(f"\n  [save] p1_table4_span.csv, p1_table4_cells.csv"
          + (", p1_summary_gain.csv" if len(gains) else ""))


if __name__ == "__main__":
    main()