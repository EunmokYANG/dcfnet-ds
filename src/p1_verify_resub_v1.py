"""재투고 원고(Access-2026-40899)의 주요 수치를 보관된 시드별 CSV 에서 다시 계산해 대조한다.

읽는 파일 : outputs/tables/seeds_{ds}_full_scaler*.csv           (m4, 10 시드)
            outputs/tables/metrics_baseline_{ds}_full_scaler*_s*.csv, seeds_baseline_*.csv
                                                                 (HGB, RF, logreg, 전체 학습 파티션)
대조 대상 : Table II (m4 평균, 표준편차), Table III (대응 차이, 95% CI, t, Holm p, 양수 시드 수),
            Table V (모델별 span 과 noise)
실행      : python -m src.p1_verify_resub_v1
결과      : 항목별 PASS / FAIL 과 요약. FAIL 이 하나라도 있으면 종료 코드 1.
"""

import sys

import numpy as np
from scipy import stats

from src.p1_table4_v1 import load_proposed, load_baselines, cells_and_spans
import pandas as pd

# ---- 원고에 인쇄된 값 (소수 자릿수 = 비교 자릿수) ------------------------------
T2 = {  # (dataset, scaler): (mean, sd)  Table II, m4, 10 seeds
    ("ciciot2023", "minmax"): (0.6484, 0.0045), ("ciciot2023", "signedlog"): (0.6680, 0.0037),
    ("ciciot2023", "signedlog_minmax"): (0.6715, 0.0059), ("ciciot2023", "none"): (0.4365, 0.0158),
    ("nfunsw", "minmax"): (0.3866, 0.0109), ("nfunsw", "signedlog"): (0.4403, 0.0086),
    ("nfunsw", "signedlog_minmax"): (0.4553, 0.0077), ("nfunsw", "none"): (0.1489, 0.0108),
}
T3 = {  # (dataset, a, b): dict(delta, lo, hi, t, p, npos)  Table III; p = Holm for primary pair
    ("ciciot2023", "signedlog", "minmax"): dict(delta=1.96, lo=1.67, hi=2.25, t=15.11, p=2.1e-7, npos=10),
    ("ciciot2023", "signedlog_minmax", "minmax"): dict(delta=2.31, lo="1.70", hi=2.92, p=1.3e-5, npos=10),
    ("ciciot2023", "signedlog_minmax", "signedlog"): dict(delta=0.35, p=0.076, npos=6),
    ("nfunsw", "signedlog", "minmax"): dict(delta=5.37, lo=4.62, hi=6.12, t=16.13, p=6.0e-8, npos=10),
    ("nfunsw", "signedlog_minmax", "minmax"): dict(delta=6.87, lo=6.07, hi=7.66, p=2.3e-8, npos=10),
    ("nfunsw", "signedlog_minmax", "signedlog"): dict(delta="1.50", p=2.3e-5, npos=10),
}
PRIMARY = [("signedlog", "minmax"), ("signedlog_minmax", "minmax")]   # Holm family within dataset
T5 = {  # (dataset, model): (span_pp, noise_pp)  Table V
    ("ciciot2023", "m4"): ("23.5", "0.70"), ("nfunsw", "m4"): ("30.6", 1.01),
    ("ciciot2023", "hgb"): (0.51, 0.13), ("nfunsw", "hgb"): ("0.10", 0.56),
    ("ciciot2023", "rf"): (0.48, 0.17), ("nfunsw", "rf"): ("0.50", 0.15),
    ("ciciot2023", "logreg"): (37.45, None), ("nfunsw", "logreg"): (21.55, None),
}

RES = []


def dec(x):
    s = x if isinstance(x, str) else f"{x}"
    return len(s.split(".")[1]) if "." in s and "e" not in s else 0


def chk(name, got, exp, rel=None):
    if exp is None:
        return
    k, shown = dec(exp), exp
    exp = float(exp)
    if rel is not None:
        ok = abs(got - exp) <= rel * abs(exp)
        g = f"{got:.2e}"
    else:
        ok = abs(got - exp) <= 1.01 * 10 ** (-k) / 2 + 1e-12
        g = f"{got:.{k}f}"
    RES.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} 계산 {g:>10s}   원고 {shown}")


def main():
    prop = load_proposed("m4")
    m4 = prop[prop.model == "m4"]

    print("\n[Table II] m4 평균 / 표준편차 (10 시드)")
    for (ds, sc), (mu, sd) in T2.items():
        v = m4[(m4.dataset == ds) & (m4.scaler == sc)]["score"]
        print(f"  ({ds}, {sc}) n = {len(v)}")
        chk(f"{ds} {sc} mean", v.mean(), mu)
        chk(f"{ds} {sc} sd", v.std(ddof=1), sd)

    print("\n[Table III] 시드 대응 차이 (pp), 95% CI, t, p, 양수 시드 수")
    for ds in ["ciciot2023", "nfunsw"]:
        d = m4[m4.dataset == ds]
        res = {}
        for (a, b) in [(k[1], k[2]) for k in T3 if k[0] == ds]:
            A = d[d.scaler == a].set_index("seed")["score"]
            B = d[d.scaler == b].set_index("seed")["score"]
            c = sorted(set(A.index) & set(B.index))
            diff = (A.loc[c] - B.loc[c]).to_numpy() * 100
            n = len(diff); m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            t, p = stats.ttest_rel(A.loc[c], B.loc[c])
            res[(a, b)] = dict(n=n, delta=m, lo=m - tc * se, hi=m + tc * se, t=t, p=p,
                               npos=int((diff > 0).sum()), dz=m / diff.std(ddof=1))
        ps = sorted(PRIMARY, key=lambda k: res[k]["p"])          # Holm within dataset
        run = 0.0
        for i, k in enumerate(ps):
            run = max(run, (len(ps) - i) * res[k]["p"]); res[k]["p_holm"] = min(run, 1.0)
        for (a, b), r in res.items():
            e = T3[(ds, a, b)]
            tag = f"{ds} {a} - {b} (n={r['n']}, d_z={r['dz']:.2f})"
            chk(tag + " delta", r["delta"], e.get("delta"))
            chk(tag + " CI lo", r["lo"], e.get("lo"))
            chk(tag + " CI hi", r["hi"], e.get("hi"))
            chk(tag + " t", r["t"], e.get("t"))
            chk(tag + (" p (Holm)" if (a, b) in PRIMARY else " p (raw)"),
                r.get("p_holm", r["p"]), e.get("p"), rel=0.06)
            chk(tag + " positive seeds", r["npos"], e.get("npos"))

    print("\n[Table V] 모델별 span 과 noise (pp)")
    raw = pd.concat([prop, load_baselines()], ignore_index=True)
    _, spans = cells_and_spans(raw)
    for (ds, mdl), (sp, nz) in T5.items():
        s = spans[(spans.dataset == ds) & (spans.model == mdl)]
        if s.empty:
            RES.append(False); print(f"  FAIL  {ds} {mdl}: 결과 파일 없음"); continue
        r = s.iloc[0]
        print(f"  ({ds}, {mdl}) seeds {r.n_seeds_min}-{r.n_seeds_max}, best {r.best_scaler}, worst {r.worst_scaler}")
        chk(f"{ds} {mdl} span", r.span_pp, sp)
        chk(f"{ds} {mdl} noise", r.noise_pp, nz)

    nf = RES.count(False)
    print(f"\n[요약] {len(RES)} 항목 중 PASS {len(RES) - nf}, FAIL {nf}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
