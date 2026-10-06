"""재투고 원고 Fig. 2 와 Fig. 5 를 다시 그린다 (10시드 / full-partition 결과).

IEEE Access Access-2026-40899.

읽는 파일 : outputs/tables/p1_table4_cells.csv   (python -m src.p1_table4_v1 의 산출물)
산출물   : outputs/figures/paper1_scaler/p1_fig2_prauc_by_scaler_resub.(png|pdf)
           outputs/figures/paper1_scaler/p1_fig5_model_family_resub.(png|pdf)
실행     : python -m src.p1_figures_resub_v1
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C

ORDER = ["minmax", "standard", "signedlog", "signedlog_minmax", "none"]
LABEL = ["Min–Max", "Z-score", "Signed-log", "SLog+MM", "No scaling"]
DS = [("ciciot2023", "CICIoT2023", 0.85), ("nfunsw", "NF-UNSW-NB15-v3", 0.58)]
COL = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#c44e52"]
FAM = [("rf", "Random forest", dict(color="#8c6d4f", ls="-")),
       ("hgb", "HistGradientBoosting", dict(color="#7f7f7f", ls="-")),
       ("logreg", "Logistic regression", dict(color="#d084c4", ls="--")),
       ("m4", "Interaction model (K=4)", dict(color="#c44e52", ls="-"))]
OUT = C.FIG_DIR / "paper1_scaler"


def get(cells, ds, model):
    sub = cells[(cells.dataset == ds) & (cells.model == model)].set_index("scaler")
    mu = [float(sub.loc[s, "mean"]) for s in ORDER]
    sd = [float(sub.loc[s, "sd"]) if int(sub.loc[s, "n"]) > 1 else 0.0 for s in ORDER]
    return mu, sd


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    print(f"  [save] {OUT / stem}.png / .pdf")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
    cells = pd.read_csv(C.TABLE_DIR / "p1_table4_cells.csv")

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 7.16 / 2.4117))
    for ax, (ds, name, top) in zip(axs, DS):
        mu, sd = get(cells, ds, "m4")
        ax.bar(range(5), mu, yerr=sd, color=COL, edgecolor="black", linewidth=0.6,
               width=0.62, capsize=3, error_kw={"elinewidth": 0.8})
        for i, (v, e) in enumerate(zip(mu, sd)):
            ax.text(i, v + e + top * 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(5)); ax.set_xticklabels(LABEL, rotation=18, ha="right")
        ax.set_ylim(0, top); ax.set_title(name, fontsize=10)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5); ax.set_axisbelow(True)
    axs[0].set_ylabel("PR-AUC (macro)")
    fig.tight_layout()
    save(fig, "p1_fig2_prauc_by_scaler_resub")

    fig, axs = plt.subplots(1, 2, figsize=(7.16, 7.16 / 2.4043), sharey=True)
    for ax, (ds, name, _) in zip(axs, DS):
        for key, lab, sty in FAM:
            mu, sd = get(cells, ds, key)
            ax.errorbar(range(5), mu, yerr=sd, marker="o", ms=4, lw=1.5, capsize=2.5,
                        elinewidth=0.8, label=lab, **sty)
        ax.set_xticks(range(5)); ax.set_xticklabels(LABEL, rotation=18, ha="right")
        ax.set_ylim(0, 1.0); ax.set_title(name, fontsize=10)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5); ax.set_axisbelow(True)
    axs[0].set_ylabel("PR-AUC (macro)")
    h, l = axs[0].get_legend_handles_labels()
    axs[1].legend(h, l, loc="upper right", ncol=1, frameon=False, fontsize=7,
                  handlelength=2.2, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, "p1_fig5_model_family_resub")


if __name__ == "__main__":
    main()