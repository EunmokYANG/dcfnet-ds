"""논문 1 재투고(2차) 최종 그림 Fig. 1–7 을 인쇄 크기 기준으로 다시 그린다.

IEEE Access Access-2026-52475 재투고, 심사의견 R3-7 (그림 가독성·범례·축·약어) 대응.

규칙
  - 그림 크기 = 원고에 실리는 크기 (단단 3.4 in, 양단 7.0 in). 확대·축소 없이 인쇄된다.
  - 글꼴 8 pt 이상 (패널 안 주석 7 pt 이상), Times 계열 serif.
  - 색 = Okabe–Ito 색약 대응 팔레트, 스케일러마다 고유 마커 → 흑백 인쇄에서도 구분.
  - 오차막대 = 시드 간 표준편차 (정의는 캡션에 기재).

읽는 파일
  outputs/tables/p1_table4_cells.csv     Fig. 2, 5 (10시드 주조건, full-partition 대조군)
  outputs/tables/sweep_scaler.csv        Fig. 3, 4 (시드 1–3 평균·표준편차)
  outputs/tables/sweep_degree.csv        Fig. 6
  data/processed/{ds}_oob_{scaler}_meta.json  Fig. 3, 4, 7 (진단값)

산출물  outputs/figures/paper1_scaler/final/p1_final_fig{1..7}.(png|pdf)
실행    python -m src.p1_figures_final_v1
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src import config as C

OUT = C.FIG_DIR / "paper1_scaler" / "final"
W1, W2 = 3.4, 7.0                     # 단단·양단 폭 (inch)
DPI = 600

SC = ["minmax", "standard", "signedlog", "signedlog_minmax", "none"]
LAB = {"minmax": "Min–max", "standard": "Z-score", "signedlog": "Signed-log",
       "signedlog_minmax": "SLog+MM", "none": "No scaling"}
COL = {"minmax": "#0072B2", "standard": "#E69F00", "signedlog": "#009E73",
       "signedlog_minmax": "#CC79A7", "none": "#D55E00"}
MK = {"minmax": "o", "standard": "s", "signedlog": "^",
      "signedlog_minmax": "D", "none": "v"}
DS = [("ciciot2023", "CICIoT2023"), ("nfunsw", "NF-UNSW-NB15-v3")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.3, "lines.markersize": 5,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.45,
    "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": DPI, "pdf.fonttype": 42,
})


def save(fig, n):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"p1_final_fig{n}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"  [save] p1_final_fig{n}.png / .pdf")


def meta(ds, sc):
    p = C.PROCESSED_DIR / f"{ds}_oob_{sc}_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def cells():
    return pd.read_csv(C.TABLE_DIR / "p1_table4_cells.csv")


def cell(c, ds, model, sc):
    r = c[(c.dataset == ds) & (c.model == model) & (c.scaler == sc)].iloc[0]
    return float(r["mean"]), (float(r["sd"]) if int(r["n"]) > 1 else 0.0), int(r["n"])


def sweep(ds, sc):
    d = pd.read_csv(C.TABLE_DIR / "sweep_scaler.csv")
    r = d[(d.dataset == ds) & (d.value == sc)].iloc[0]
    return float(r["PR-AUC(macro)_mean"]), float(r["PR-AUC(macro)_std"])


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig = plt.figure(figsize=(W2, 2.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 36)
    cols = [  # (x0, width, title, lines)
        (0.5, 17.5, "1  Data", ["CICIoT2023", "1,176,851 flows", "43 features, 8 classes",
                                "stratified 70/30 split", "", "NF-UNSW-NB15-v3",
                                "2,242,931 flows", "47 features, 10 classes", "chronological 70/30 split"]),
        (20.5, 19.5, "2  Ten transforms", ["Affine: min–max, z-score,", "  RobustScaler, no scaling",
                                          "Clip + affine: winsorized", "  min–max",
                                          "Fixed nonlinear: signed-log,", "  asinh, SLog+MM",
                                          "Fitted nonlinear: quantile,", "  Yeo–Johnson",
                                          "fitted on training rows only"]),
        (42.5, 18.0, "3  Models", ["Interaction detector", "  orders K = 1–4", "  10 seeds (principal)",
                                   "", "Controls", "  logistic regression", "  random forest,",
                                   "  hist. gradient boosting"]),
        (63.0, 17.5, "4  Diagnostics", ["Between-feature balance", "  max/min feature RMS",
                                        "Within-feature resolution", "  median IQR/range",
                                        "Cross-degree range", "  rms(x⁴)/rms(x)",
                                        "", "affine maps cannot", "change resolution"]),
        (83.0, 16.5, "5  Evaluation", ["Macro PR-AUC,", "  macro F1, MCC", "Paired seed contrasts,", "  Holm-adjusted",
                                       "Checks: order K,", "  unseen vectors,", "  budget, encoding,",
                                       "  attribution"]),
    ]
    for k, (x0, w, title, lines) in enumerate(cols):
        ax.add_patch(FancyBboxPatch((x0, 1.0), w, 33.0, boxstyle="round,pad=0,rounding_size=1.2",
                                    linewidth=0.8, edgecolor="#333333", facecolor="#F4F4F4"))
        ax.add_patch(FancyBboxPatch((x0, 29.6), w, 4.4, boxstyle="round,pad=0,rounding_size=1.2",
                                    linewidth=0.0, facecolor="#0072B2"))
        ax.text(x0 + w / 2, 31.8, title, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        for j, t in enumerate(lines):
            bold = t in ("CICIoT2023", "NF-UNSW-NB15-v3", "Interaction detector", "Controls")
            ax.text(x0 + 1.0, 27.0 - j * 2.85, t, ha="left", va="center", fontsize=8,
                    fontweight="bold" if bold else "normal",
                    style="italic" if t in ("fitted on training rows only", "affine maps cannot",
                                            "change resolution") else "normal")
        if k < len(cols) - 1:
            nx = cols[k + 1][0]
            ax.annotate("", xy=(nx - 0.1, 17.5), xytext=(x0 + w + 0.1, 17.5),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333333",
                                        mutation_scale=9))
    save(fig, 1)


# ---------------------------------------------------------------- Fig. 2
def fig2():
    c = cells()
    fig, axs = plt.subplots(2, 1, figsize=(W1, 3.9))
    for ax, (ds, name), top in zip(axs, DS, (0.80, 0.56)):
        xs = np.arange(len(SC))
        for x, sc in zip(xs, SC):
            m, s, n = cell(c, ds, "m4", sc)
            ax.bar(x, m, yerr=s, width=0.64, color=COL[sc], edgecolor="black", linewidth=0.5,
                   capsize=2.5, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                   hatch="///" if sc == "standard" else None)
            ax.text(x, m + s + top * 0.015, f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs); ax.set_xticklabels([LAB[s] for s in SC])
        ax.set_ylim(0, top); ax.set_ylabel("Macro PR-AUC")
        ax.set_title(name, pad=3); ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=0.8)
    save(fig, 2)


# ---------------------------------------------------------------- Fig. 3 / 4 공통
def _scatter(xkey, xlabel, n, logx):
    fig, ax = plt.subplots(figsize=(W1, 3.2))
    for ds, fill in (("ciciot2023", True), ("nfunsw", False)):
        pts = []
        for sc in SC:
            m = meta(ds, sc)
            x = np.log10(m[xkey]) if xkey == "dynamic_range_u4_u1" else m[xkey]
            y, e = sweep(ds, sc)
            pts.append((x, y))
            ax.errorbar(x, y, yerr=e, fmt=MK[sc], ms=5.5, mew=1.1, color=COL[sc],
                        mfc=COL[sc] if fill else "white", mec=COL[sc],
                        elinewidth=0.8, capsize=2, zorder=3)
        if xkey == "dynamic_range_u4_u1":
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, ls="--", lw=0.7, color="gray", zorder=1)
        else:
            xa = meta(ds, "minmax")[xkey]
            ya = [sweep(ds, s)[0] for s in ("minmax", "standard", "none")]
            ax.plot([xa, xa], [min(ya), max(ya)], ls=":", lw=0.9, color="gray", zorder=1)
    if logx:
        ax.set_xscale("log")
    else:
        ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel(xlabel); ax.set_ylabel("Macro PR-AUC")
    hd = [Line2D([], [], marker="o", ls="", color="black", mfc="black", label="CICIoT2023"),
          Line2D([], [], marker="o", ls="", color="black", mfc="white", label="NF-UNSW-NB15-v3")]
    ax.legend(handles=hd, loc="upper right" if xkey == "dynamic_range_u4_u1" else "upper center",
              handletextpad=0.3, borderaxespad=0.4, fontsize=7.5)
    hs = [Line2D([], [], marker=MK[s], ls="", color=COL[s], label=LAB[s]) for s in SC]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               columnspacing=1.0, handletextpad=0.3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, n)


def fig3():
    _scatter("dynamic_range_u4_u1", r"Cross-degree dynamic range, $\log_{10}[\mathrm{rms}(x^4)/\mathrm{rms}(x)]$", 3, False)


def fig4():
    _scatter("iqr_over_range_median_continuous", "Within-feature resolution (median IQR/range)", 4, True)


# ---------------------------------------------------------------- Fig. 5
def fig5():
    c = cells()
    fam = [("m4", "Interaction model (K = 4)", "#D55E00", "-", "o"),
           ("logreg", "Logistic regression", "#CC79A7", "--", "s"),
           ("rf", "Random forest", "#0072B2", "-", "^"),
           ("hgb", "Hist. gradient boosting", "#009E73", "-", "D")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75), sharey=True)
    for ax, (ds, name) in zip(axs, DS):
        for key, lab, col, ls, mk in fam:
            mu, sd = zip(*[cell(c, ds, key, s)[:2] for s in SC])
            ax.errorbar(range(5), mu, yerr=sd, color=col, ls=ls, marker=mk, ms=4.5,
                        capsize=2, elinewidth=0.8, label=lab)
        ax.set_xticks(range(5)); ax.set_xticklabels([LAB[s] for s in SC])
        ax.set_ylim(0, 1.0); ax.set_title(name, pad=3); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Macro PR-AUC")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 5)


# ---------------------------------------------------------------- Fig. 6
def fig6():
    d = pd.read_csv(C.TABLE_DIR / "sweep_degree.csv")
    d["sc"] = d["tag"].str.replace(r"_degree\d+$", "", regex=True)
    ks = [1, 2, 3, 4]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    for ax, (ds, name) in zip(axs, DS):
        sub = d[d.dataset == ds]
        ser = {}
        for tag, sc in (("oob_minmax", "minmax"), ("oob_signedlog", "signedlog")):
            g = sub[sub.sc == tag].set_index("value").sort_index()
            m = [float(g.loc[k, "PR-AUC(macro)_mean"]) for k in ks]
            e = [float(g.loc[k, "PR-AUC(macro)_std"]) for k in ks]
            ser[sc] = m
            ax.errorbar(ks, m, yerr=e, color=COL[sc], marker=MK[sc], ms=5, capsize=2.5,
                        elinewidth=0.8, label=LAB[sc])
        for k, a, b in zip(ks, ser["minmax"], ser["signedlog"]):
            ax.annotate(f"+{(b - a) * 100:.1f}", (k, b), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color="#444444")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        ax.set_xticks(ks); ax.set_xlabel("Maximum interaction order K")
        ax.set_title(name, pad=3); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Macro PR-AUC")
    axs[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, 6)


# ---------------------------------------------------------------- Fig. 7
def fig7():
    ks = np.arange(1, 5)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75))
    for ax, (ds, name) in zip(axs, DS):
        for sc in SC:
            r = [v if v is not None and np.isfinite(v) else np.nan for v in meta(ds, sc)["rms_u"]]
            ax.plot(ks, r, color=COL[sc], marker=MK[sc], ms=4.5, label=LAB[sc],
                    ls="--" if sc == "signedlog_minmax" else "-")
        ax.set_yscale("log"); ax.set_xticks(ks)
        ax.set_xlabel("Interaction order k"); ax.set_title(name, pad=3)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel(r"$\mathrm{rms}(x^k)$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    save(fig, 7)


def main():
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f()
    print(f"  출력: {OUT}")


if __name__ == "__main__":
    main()
