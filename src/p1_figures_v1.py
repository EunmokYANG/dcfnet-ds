"""논문 1(스케일러 재검토) Figure 1~3 생성.

입력
  data/processed/{dataset}_full_scaler{scaler}_meta.json   (rms_u, dyn_range)
  data/processed/{dataset}_oob_{scaler}_meta.json          (배포 안정성 진단)
  outputs/tables/sweep_scaler.csv                          (PR-AUC 평균/표준편차)

출력
  outputs/figures/paper1_scaler/p1_fig1_rms_curve.png
  outputs/figures/paper1_scaler/p1_fig2_prauc_bar.png
  outputs/figures/paper1_scaler/p1_fig3_cond_vs_perf.png
  outputs/figures/paper1_scaler/p1_fig4_resolution_vs_perf.png
  outputs/figures/paper1_scaler/p1_fig5_architecture_sensitivity.png
  outputs/figures/paper1_scaler/p1_fig6_degree_dependence.png
  outputs/tables/paper1_scaler/p1_table1.csv               (Table 1 원본값)
  outputs/tables/paper1_scaler/p1_table2.csv               (Table 2 원본값)
  outputs/tables/paper1_scaler/p1_table3.csv               (Table 3 진단 메커니즘)
  outputs/tables/paper1_scaler/p1_table4.csv               (Table 4 아키텍처 민감도)
  outputs/tables/paper1_scaler/p1_table5.csv               (Table 5 Welch 검정)

사용
  python -m src.p1_figures_v1
  python -m src.p1_figures_v1 --fig 1        # 특정 그림만
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

# ---------------------------------------------------------------- 출력 경로
# 입력(sweep_scaler.csv, metrics_baseline_*.csv)은 C.TABLE_DIR 에서 그대로 읽는다.
# 논문 산출물만 논문별 하위 폴더로 분리한다. 논문 2 는 outputs/paper2_degree/.
PAPER = "p1"
OUT_FIG = C.FIG_DIR / "paper1_scaler"
OUT_TAB = C.TABLE_DIR / "paper1_scaler"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 설정
SCALERS = ["minmax", "standard", "signedlog", "signedlog_minmax", "none"]
LABELS = {"minmax": "Min–Max", "standard": "Z-score",
          "signedlog": "Signed-log", "signedlog_minmax": "Signed-log + Min–Max",
          "none": "No scaling"}
SHORT = {"minmax": "Min–Max", "standard": "Z-score",
         "signedlog": "Signed-log", "signedlog_minmax": "SLog+MM",
         "none": "No scaling"}
COLORS = {"minmax": "#4C72B0", "standard": "#DD8452",
          "signedlog": "#55A868", "signedlog_minmax": "#8172B3",
          "none": "#C44E52"}
MARKERS = {"minmax": "o", "standard": "s", "signedlog": "^",
           "signedlog_minmax": "v", "none": "D"}
BASELINE_MODELS = ["rf", "hgb", "logreg"]
DATASETS = ["ciciot2023", "nfunsw"]
DS_TITLE = {"ciciot2023": "CICIoT2023", "nfunsw": "NF-UNSW-NB15-v3"}

DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- 로딩
def load_meta():
    """{(dataset, scaler): meta dict}. tag = oob_{scaler} 로 통일한다."""
    return load_meta_oob()


def load_sweep():
    p = C.TABLE_DIR / "sweep_scaler.csv"
    if not p.exists():
        raise FileNotFoundError(f"스윕 결과 없음: {p}")
    return pd.read_csv(p)


def prauc(df, ds, sc):
    r = df[(df.dataset == ds) & (df.value == sc)]
    if r.empty:
        raise ValueError(f"sweep_scaler.csv 에 {ds}/{sc} 행이 없습니다")
    r = r.iloc[0]
    return float(r["PR-AUC(macro)_mean"]), float(r["PR-AUC(macro)_std"])


# ---------------------------------------------------------------- Figure 1
def figure1(meta):
    """스케일러별 rms(x^k) 곡선. 원자료 통계이며 모델 활성값이 아니다.

    모델은 DegreeScale 로 매 차수를 정규화하므로 내부 u_k 는 이 값에
    도달하지 않는다. 이 곡선은 정규화가 없을 때의 다항식 항 크기다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ks = np.arange(1, 5)

    for ax, ds in zip(axes, DATASETS):
        for sc in SCALERS:
            rms = meta[(ds, sc)]["rms_u"]
            rms = [r if np.isfinite(r) else np.nan for r in rms]
            ax.plot(ks, rms, marker=MARKERS[sc], color=COLORS[sc],
                    label=LABELS[sc], linewidth=1.3, markersize=4.5)
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("Interaction order $k$")
        ax.set_title(DS_TITLE[ds])

    axes[0].set_ylabel(r"$\mathrm{rms}(x^{k})$")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)

    out = OUT_FIG / f"{PAPER}_fig1_rms_curve.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Figure 2
def figure2(df):
    """스케일러별 PR-AUC(macro) 막대. 2패널, 오차막대 = 3시드 표준편차."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    xs = np.arange(len(SCALERS))

    for ax, ds in zip(axes, DATASETS):
        means, stds = zip(*[prauc(df, ds, sc) for sc in SCALERS])
        ax.bar(xs, means, yerr=stds, capsize=3, width=0.62,
               color=[COLORS[s] for s in SCALERS],
               edgecolor="black", linewidth=0.6,
               error_kw={"linewidth": 0.8})
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + 0.012, f"{m:.4f}",
                    ha="center", fontsize=6.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[s] for s in SCALERS], rotation=18,
                           ha="right")
        ax.set_ylim(0, max(means) * 1.28)
        ax.set_title(DS_TITLE[ds])
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("PR-AUC (macro)")

    out = OUT_FIG / f"{PAPER}_fig2_prauc_bar.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Figure 3
def figure3(meta, df):
    """조건수 vs 성능 산점도. x = log10(u4/u1), 비단조 관계 시각화."""
    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    for ds, mk, fc in [("ciciot2023", "o", "full"),
                       ("nfunsw", "s", "open")]:
        xs, ys = [], []
        for sc in SCALERS:
            ratio = meta[(ds, sc)]["dynamic_range_u4_u1"]
            xs.append(np.log10(ratio))
            ys.append(prauc(df, ds, sc)[0])
        order = np.argsort(xs)                      # x 순으로 연결 (지그재그 방지)
        ax.plot(np.array(xs)[order], np.array(ys)[order],
                linestyle="--", color="gray", linewidth=0.7, zorder=1)
        for x, y, sc in zip(xs, ys, SCALERS):
            ax.scatter(x, y, marker=mk, s=52, zorder=3,
                       color=COLORS[sc] if fc == "full" else "white",
                       edgecolor=COLORS[sc], linewidth=1.4)
            off = (5, -11) if sc == "minmax" else (5, 5)
            ax.annotate(SHORT[sc], (x, y), textcoords="offset points",
                        xytext=off, fontsize=6.5, color=COLORS[sc])

    ax.set_xlabel(r"$\log_{10}\left(\mathrm{rms}(x^{4})/\mathrm{rms}(x)\right)$")
    ax.set_ylabel("PR-AUC (macro)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")

    h = [plt.Line2D([], [], marker="o", color="black", linestyle="",
                    markerfacecolor="black", markersize=6,
                    label=DS_TITLE["ciciot2023"]),
         plt.Line2D([], [], marker="s", color="black", linestyle="",
                    markerfacecolor="white", markersize=6,
                    label=DS_TITLE["nfunsw"])]
    ax.legend(handles=h, loc="lower left", frameon=False)

    out = OUT_FIG / f"{PAPER}_fig3_cond_vs_perf.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Table 1
def load_meta_oob():
    """배포 안정성 진단용 메타. tag = oob_{scaler}."""
    out = {}
    for ds in DATASETS:
        for sc in SCALERS:
            p = C.PROCESSED_DIR / f"{ds}_oob_{sc}_meta.json"
            if not p.exists():
                raise FileNotFoundError(f"메타 파일 없음: {p}")
            out[(ds, sc)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def table1():
    """Table 1 (배포 안정성) 원본값을 CSV 로 저장."""
    mo = load_meta_oob()
    rows = []
    for ds in DATASETS:
        for sc in SCALERS:
            m = mo[(ds, sc)]
            rows.append({
                "dataset": DS_TITLE[ds],
                "split": m["split"],
                "scaler": LABELS[sc],
                "absmax_train": m["absmax_train"],
                "absmax_test": m["absmax_test"],
                "exceed_rel": m["exceed_rel"],
                "exceed_rel_deg4": m["exceed_rel_deg4"],
                "worst_feature": m["exceed_worst_feature"],
                "n_oob_values": m["oob_value_count"],
                "n_oob_rows": m["oob_row_count"],
                "n_test_rows": m["test_shape"][0],
                "oob_row_pct": m["oob_row_pct"],
            })
    out = OUT_TAB / f"{PAPER}_table1.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Table 2
def table2(meta, df):
    """Table 2 원본값을 CSV 로 저장. 논문 표에 그대로 옮길 수 있다."""
    rows = []
    for ds in DATASETS:
        for sc in SCALERS:
            m = meta[(ds, sc)]
            mu, sd = prauc(df, ds, sc)
            rows.append({
                "dataset": DS_TITLE[ds],
                "scaler": LABELS[sc],
                "PR-AUC(macro)": f"{mu:.4f} ± {sd:.4f}",
                "rms_u1": m["rms_u"][0],
                "rms_u2": m["rms_u"][1],
                "rms_u3": m["rms_u"][2],
                "rms_u4": m["rms_u"][3],
                "u4/u1": m["dynamic_range_u4_u1"],
                "oob_value_pct": m["oob_value_pct"],
                "oob_row_pct": m["oob_row_pct"],
                "n_test_rows": m["test_shape"][0],
                "n_oob_rows": round(m["test_shape"][0]
                                    * m["oob_row_pct"] / 100),
            })
    out = OUT_TAB / f"{PAPER}_table2.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Table 3
def table3(df):
    """Table 3 (스케일러 진단 메커니즘) — 두 축과 성능을 한 표에."""
    mo = load_meta_oob()
    rows = []
    for ds in DATASETS:
        for sc in SCALERS:
            m = mo[(ds, sc)]
            mu, sd = prauc(df, ds, sc)
            rows.append({
                "dataset": DS_TITLE[ds],
                "scaler": LABELS[sc],
                "feat_rms_ratio": m["feat_rms_ratio"],
                "n_vanishing": m["n_vanishing_features"],
                "n_features": m["n_features"],
                "iqr_over_range_cont": m["iqr_over_range_median_continuous"],
                "n_continuous": m["n_continuous_features"],
                "dyn_range_k4_k1": m["dynamic_range_u4_u1"],
                "PR-AUC(macro)_mean": mu,
                "PR-AUC(macro)_std": sd,
            })
    out = OUT_TAB / f"{PAPER}_table3.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Figure 4
def figure4(df):
    """피처 내 해상도 vs 성능. 아핀 3종이 한 점에 겹치는 것이 요점."""
    mo = load_meta_oob()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    for ds, mk, fc in [("ciciot2023", "o", "full"),
                       ("nfunsw", "s", "open")]:
        for sc in SCALERS:
            x = mo[(ds, sc)]["iqr_over_range_median_continuous"]
            y = prauc(df, ds, sc)[0]
            ax.scatter(x, y, marker=mk, s=54, zorder=3,
                       color=COLORS[sc] if fc == "full" else "white",
                       edgecolor=COLORS[sc], linewidth=1.4)
        # 아핀 3종은 같은 x 에 겹치므로 세로 점선으로 묶어 표시
        xa = mo[(ds, "minmax")]["iqr_over_range_median_continuous"]
        ys = [prauc(df, ds, s)[0] for s in ("minmax", "standard", "none")]
        ax.plot([xa, xa], [min(ys), max(ys)], color="gray",
                linestyle=":", linewidth=1.0, zorder=1)
        ax.annotate("affine scalers\n(identical x)", (xa, max(ys)),
                    textcoords="offset points", xytext=(8, 2),
                    fontsize=6.5, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Within-feature resolution  (median IQR / range)")
    ax.set_ylabel("PR-AUC (macro)")

    h = [plt.Line2D([], [], marker="o", color="black", linestyle="",
                    markerfacecolor="black", markersize=6,
                    label=DS_TITLE["ciciot2023"]),
         plt.Line2D([], [], marker="s", color="black", linestyle="",
                    markerfacecolor="white", markersize=6,
                    label=DS_TITLE["nfunsw"])]
    h += [plt.Line2D([], [], marker="o", color=COLORS[s], linestyle="",
                     markersize=6, label=SHORT[s]) for s in SCALERS]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=6.5)

    out = OUT_FIG / f"{PAPER}_fig4_resolution_vs_perf.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Table 4
def table4(df):
    """Table 4 (아키텍처별 스케일러 민감도).

    트리 앙상블은 피처별 임계값 분할이라 단조 변환에 불변이다.
    따라서 스케일러를 바꿔도 성능이 변하지 않아야 하며, 이는 관측된
    m4 의 큰 격차가 모델 특이적 현상임을 확인하는 음성 대조군이 된다.
    """
    rows = []
    for ds in DATASETS:
        per_model = {m: {} for m in BASELINE_MODELS}
        for sc in SCALERS:
            p = C.TABLE_DIR / f"metrics_baseline_{ds}_oob_{sc}.csv"
            if not p.exists():
                raise FileNotFoundError(f"baseline 결과 없음: {p}")
            b = pd.read_csv(p)
            for m in BASELINE_MODELS:
                r = b[b.Variant == m]
                if len(r):
                    per_model[m][sc] = float(r["PR-AUC(macro)"].iloc[0])
        per_model["m4"] = {sc: prauc(df, ds, sc)[0] for sc in SCALERS}

        for m in BASELINE_MODELS + ["m4"]:
            v = per_model[m]
            vals = [v[sc] for sc in SCALERS if sc in v]
            row = {"dataset": DS_TITLE[ds], "model": m,
                   "spread_pp": round(100 * (max(vals) - min(vals)), 2),
                   "min": round(min(vals), 4), "max": round(max(vals), 4)}
            for sc in SCALERS:
                row[LABELS[sc]] = round(v.get(sc, float("nan")), 4)
            rows.append(row)

    out = OUT_TAB / f"{PAPER}_table4.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Figure 5
def figure5(df):
    """아키텍처별 스케일러 민감도. 트리는 평평하고 m4 만 크게 흔들린다."""
    order = ["rf", "hgb", "logreg", "m4"]
    nice = {"rf": "Random Forest", "hgb": "HistGradientBoosting",
            "logreg": "Logistic Regression", "m4": "Interaction model (K=4)"}
    style = {"rf": "-", "hgb": "-", "logreg": "--", "m4": "-"}
    col = {"rf": "#937860", "hgb": "#8C8C8C", "logreg": "#DA8BC3",
           "m4": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    xs = np.arange(len(SCALERS))
    for ax, ds in zip(axes, DATASETS):
        base = {}
        for sc in SCALERS:
            b = pd.read_csv(C.TABLE_DIR / f"metrics_baseline_{ds}_oob_{sc}.csv")
            for m in BASELINE_MODELS:
                r = b[b.Variant == m]
                if len(r):
                    base.setdefault(m, {})[sc] = float(r["PR-AUC(macro)"].iloc[0])
        base["m4"] = {sc: prauc(df, ds, sc)[0] for sc in SCALERS}

        for m in order:
            ys = [base[m][sc] for sc in SCALERS]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                    linestyle=style[m], color=col[m], label=nice[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[s] for s in SCALERS], rotation=20,
                           ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_title(DS_TITLE[ds])
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("PR-AUC (macro)")
    axes[1].legend(loc="center right", frameon=False, fontsize=6.5)

    out = OUT_FIG / f"{PAPER}_fig5_architecture_sensitivity.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")


# ---------------------------------------------------------------- Table 5
def table5(df, n_seeds=3):
    """Table 5 (스케일러 쌍별 Welch t-검정).

    sweep_scaler.csv 의 평균과 표준편차만으로 정확히 계산된다.
    Welch 검정은 평균, 표준편차, 표본수만 요구하므로 근사가 아니다.
    성능 순위상 인접한 쌍과, 표준 관행(Min-Max) 대비 조합을 검정한다.
    같은 데이터셋 안에서 Holm-Bonferroni 로 다중비교를 보정한다.
    """
    from scipy import stats

    pairs = [("signedlog_minmax", "signedlog"),
             ("signedlog", "minmax"),
             ("signedlog_minmax", "minmax"),
             ("minmax", "standard"),
             ("standard", "none")]

    rows = []
    for ds in DATASETS:
        raw = []
        for a, b in pairs:
            m1, s1 = prauc(df, ds, a)
            m2, s2 = prauc(df, ds, b)
            v1, v2 = s1 ** 2 / n_seeds, s2 ** 2 / n_seeds
            se = np.sqrt(v1 + v2)
            t = (m1 - m2) / se
            dof = (v1 + v2) ** 2 / (v1 ** 2 / (n_seeds - 1)
                                    + v2 ** 2 / (n_seeds - 1))
            p = 2 * stats.t.sf(abs(t), dof)
            # Hedges 보정 없이 Cohen's d (합동 표준편차)
            sp = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
            raw.append((a, b, m1 - m2, t, dof, p, (m1 - m2) / sp if sp else np.inf))

        # Holm-Bonferroni
        order = np.argsort([r[5] for r in raw])
        adj = [None] * len(raw)
        run = 0.0
        for rank, idx in enumerate(order):
            v = raw[idx][5] * (len(raw) - rank)
            run = max(run, min(v, 1.0))
            adj[idx] = run

        for (a, b, d, t, dof, p, cd), pa in zip(raw, adj):
            rows.append({
                "dataset": DS_TITLE[ds],
                "comparison": f"{LABELS[a]} vs {LABELS[b]}",
                "diff_pp": round(100 * d, 2),
                "t": round(t, 2),
                "df": round(dof, 1),
                "p_raw": round(p, 4),
                "p_holm": round(pa, 4),
                "cohens_d": round(cd, 2),
                "significant_0.05": pa < 0.05,
            })

    out = OUT_TAB / f"{PAPER}_table5.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  [save] {out.name}")

# ---------------------------------------------------------------- Figure 6
def figure6():
    """차수 상한 K 에 따른 signed-log 와 Min-Max 의 격차.

    "왜곡이 k 제곱으로 증폭된다"는 가설이 맞다면 K 가 커질수록 격차가
    벌어져야 한다. 실측은 그렇지 않으며, 그 결과를 그대로 보인다.
    """
    p = C.TABLE_DIR / "sweep_degree.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    d["scaler"] = d["tag"].str.replace(r"_degree\d+$", "", regex=True)
    d = d[d["scaler"].isin(["oob_minmax", "oob_signedlog"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ks = [1, 2, 3, 4]
    for ax, ds in zip(axes, DATASETS):
        sub = d[d.dataset == ds]
        for sc, lab, col in [("oob_minmax", "Min–Max", COLORS["minmax"]),
                             ("oob_signedlog", "Signed-log",
                              COLORS["signedlog"])]:
            g = sub[sub.scaler == sc].set_index("value").sort_index()
            m = [g.loc[k, "PR-AUC(macro)_mean"] for k in ks]
            e = [g.loc[k, "PR-AUC(macro)_std"] for k in ks]
            ax.errorbar(ks, m, yerr=e, marker="o", markersize=4.5,
                        capsize=3, linewidth=1.5, color=col, label=lab)
        for k in ks:
            a = sub[(sub.scaler == "oob_minmax") & (sub.value == k)]
            b = sub[(sub.scaler == "oob_signedlog") & (sub.value == k)]
            gap = (float(b["PR-AUC(macro)_mean"].iloc[0])
                   - float(a["PR-AUC(macro)_mean"].iloc[0])) * 100
            ax.annotate(f"{gap:+.1f}", (k, float(b["PR-AUC(macro)_mean"].iloc[0])),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="dimgray")
        ax.set_xticks(ks)
        ax.set_xlabel("Maximum interaction order $K$")
        ax.set_title(DS_TITLE[ds])
        ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.10 * (hi - lo))   # 격차 라벨 여백
    axes[0].set_ylabel("PR-AUC (macro)")
    axes[0].legend(loc="lower right", frameon=False)

    out = OUT_FIG / f"{PAPER}_fig6_degree_dependence.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  [save] {out.name}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, default=0,
                    help="1~6 지정 시 해당 그림만. 0 이면 전부")
    a = ap.parse_args()

    meta = load_meta()
    df = load_sweep()

    print("=" * 62)
    print("  논문 1 그림 생성")
    print("=" * 62)

    if a.fig in (0, 1):
        figure1(meta)
    if a.fig in (0, 2):
        figure2(df)
    if a.fig in (0, 3):
        figure3(meta, df)
    if a.fig in (0, 4):
        figure4(df)
    if a.fig in (0, 5):
        figure5(df)
    if a.fig in (0, 6):
        figure6()
    if a.fig == 0:
        table1()
        table2(meta, df)
        table3(df)
        table4(df)
        table5(df)

    print(f"  그림: {OUT_FIG}")
    print(f"  표  : {OUT_TAB}")


if __name__ == "__main__":
    main()


