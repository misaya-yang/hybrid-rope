"""
Figure 2 for NeurIPS paper: EVQ x YaRN orthogonal complementarity.

Data source:
  docs/exp/2026-03-03_passkey_mix_results.md

Only uses the paper-safe, internally consistent evidence block:
  - 10% passkey mix
  - fair scale=8 comparison
  - 3-seed means
  - 8K seed-wise retrieval dots

Usage:
  ~/miniconda3/bin/conda run -n aidemo python scripts/fig2_evq_yarn_orthogonality.py

Output:
  - paper_exports/fig2_evq_yarn_orthogonality.pdf/.png
  - docs/paperdraft/figs/fig2_evq_yarn_synergy.pdf/.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    }
)


C_GEO = "#2166ac"
C_EVG = "#d6604d"
C_GEO_LIGHT = "#92c5de"
C_EVG_LIGHT = "#f4a582"
C_GRID = "#d9d9d9"
C_TEXT = "#333333"


RETRIEVAL_LABELS = ["2K", "4K", "8K", "12K", "16K"]
RETRIEVAL_X = np.array([2, 4, 8, 12, 16], dtype=float)
RETRIEVAL = {
    "Geo": np.array([100, 59, 41, 57, 51], dtype=float),
    "Geo+YaRN": np.array([100, 100, 61, 59, 51], dtype=float),
    "EVQ": np.array([100, 69, 53, 63, 50], dtype=float),
    "EVQ+YaRN": np.array([100, 100, 100, 79, 68], dtype=float),
}

PPL_LABELS = ["2K", "8K", "12K", "16K"]
PPL_X = np.array([2, 8, 12, 16], dtype=float)
PPL = {
    "Geo": np.array([67.2, 161.9, 212.0, 253.2], dtype=float),
    "Geo+YaRN": np.array([68.1, 82.9, 118.9, 157.7], dtype=float),
    "EVQ": np.array([67.9, 150.3, 191.8, 229.5], dtype=float),
    "EVQ+YaRN": np.array([70.7, 70.9, 81.4, 107.5], dtype=float),
}

# 10% fair scale=8, seed-wise 8K retrieval.
SEEDWISE_8K = {
    "Geo": np.array([36, 46, 40], dtype=float),
    "Geo+YaRN": np.array([58, 62, 64], dtype=float),
    "EVQ": np.array([44, 60, 56], dtype=float),
    "EVQ+YaRN": np.array([100, 100, 100], dtype=float),
}


SERIES_STYLE = {
    "Geo": {"color": C_GEO, "marker": "o", "linestyle": "-", "label": "Geo"},
    "Geo+YaRN": {
        "color": C_GEO,
        "marker": "o",
        "linestyle": "--",
        "label": "Geo+YaRN",
    },
    "EVQ": {"color": C_EVG, "marker": "s", "linestyle": "-", "label": "EVQ"},
    "EVQ+YaRN": {
        "color": C_EVG,
        "marker": "s",
        "linestyle": "--",
        "label": "EVQ+YaRN",
    },
}


def setup_axis(ax):
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def add_gain_bracket(ax, x0, x1, y, text):
    ax.plot([x0, x0, x1, x1], [y - 2, y, y, y - 2], color=C_TEXT, lw=0.8)
    ax.text((x0 + x1) / 2, y + 1.5, text, ha="center", va="bottom", fontsize=6.5, color=C_TEXT)


def bar_style(name):
    is_evq = name.startswith("EVQ")
    return {
        "facecolor": C_EVG_LIGHT if is_evq else C_GEO_LIGHT,
        "edgecolor": C_EVG if is_evq else C_GEO,
        "hatch": "//" if "YaRN" in name else "",
    }


def draw_bar_panel(ax):
    categories = ["Geo", "Geo+YaRN", "EVQ", "EVQ+YaRN"]
    means = [RETRIEVAL[name][2] for name in categories]
    x = np.arange(len(categories))
    width = 0.64

    for idx, name in enumerate(categories):
        style = bar_style(name)
        ax.bar(
            x[idx],
            means[idx],
            width=width,
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.9,
            hatch=style["hatch"],
            zorder=2,
        )

        jitter = np.array([-0.08, 0.00, 0.08])
        ax.scatter(
            np.full(3, x[idx]) + jitter,
            SEEDWISE_8K[name],
            color=style["edgecolor"],
            s=14,
            zorder=3,
        )

        ax.text(
            x[idx],
            means[idx] + 3.0,
            f"{int(round(means[idx]))}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=style["edgecolor"],
            fontweight="bold",
        )

    add_gain_bracket(ax, x[1], x[3], 104, "+39pp at 8K")
    ax.annotate(
        "10%: 3/3 seeds = 100%",
        xy=(x[3], 100),
        xytext=(2.45, 88),
        fontsize=6.5,
        color=C_EVG,
        ha="left",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVG, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVG, linewidth=0.6),
    )

    ax.annotate(
        "training-time gain",
        xy=(x[0], 41),
        xytext=(0.55, 21),
        fontsize=5.8,
        color=C_EVG,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C_EVG, lw=0.7, mutation_scale=7),
    )
    ax.annotate(
        "inference-time gain",
        xy=(x[0], 41),
        xytext=(-0.35, 13),
        fontsize=5.8,
        color=C_GEO,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=0.7, mutation_scale=7),
    )

    ax.set_title("(a) 8K retrieval at fair scale=8", fontweight="bold", pad=6)
    ax.set_ylabel("Passkey retrieval (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Geo", "Geo\n+YaRN", "EVQ", "EVQ\n+YaRN"])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    setup_axis(ax)


def draw_retrieval_panel(ax):
    for name, values in RETRIEVAL.items():
        style = SERIES_STYLE[name]
        ax.plot(
            RETRIEVAL_X,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )

    ax.annotate(
        "EVQ+YaRN keeps\n4x retrieval perfect",
        xy=(8, 100),
        xytext=(9.6, 89),
        fontsize=6.2,
        color=C_EVG,
        ha="left",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVG, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVG, linewidth=0.6),
    )
    ax.annotate(
        "Geo+YaRN saturates\nnear 61%",
        xy=(8, 61),
        xytext=(10.1, 63),
        fontsize=6.0,
        color=C_GEO,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef5fb", edgecolor=C_GEO, linewidth=0.6),
    )

    ax.set_title("(b) Retrieval vs extrapolation length", fontweight="bold", pad=6)
    ax.set_xlabel("Evaluation length")
    ax.set_ylabel("Passkey retrieval (%)")
    ax.set_xticks(RETRIEVAL_X)
    ax.set_xticklabels(RETRIEVAL_LABELS)
    ax.set_ylim(35, 105)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    setup_axis(ax)
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="none", borderpad=0.3, handletextpad=0.3)


def draw_ppl_panel(ax):
    for name, values in PPL.items():
        style = SERIES_STYLE[name]
        ax.plot(
            PPL_X,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )

    ax.annotate(
        "82.9 \u2192 70.9 at 8K",
        xy=(8, 70.9),
        xytext=(10.0, 93),
        fontsize=6.0,
        color=C_EVG,
        ha="left",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVG, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVG, linewidth=0.6),
    )
    ax.annotate(
        "near-zero 4x PPL drift",
        xy=(8, 70.9),
        xytext=(6.5, 117),
        fontsize=6.0,
        color=C_EVG,
        ha="center",
    )

    ax.set_title("(c) PPL vs extrapolation length", fontweight="bold", pad=6)
    ax.set_xlabel("Evaluation length")
    ax.set_ylabel("Perplexity")
    ax.set_xticks(PPL_X)
    ax.set_xticklabels(PPL_LABELS)
    ax.set_ylim(55, 270)
    ax.set_yticks([60, 100, 140, 180, 220, 260])
    setup_axis(ax)


def main():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.95, 1.05, 1.05], "wspace": 0.38},
    )

    draw_bar_panel(axes[0])
    draw_retrieval_panel(axes[1])
    draw_ppl_panel(axes[2])

    legacy_dir = "paper_exports"
    paper_fig_dir = "docs/paperdraft/figs"
    os.makedirs(legacy_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)

    legacy_pdf = os.path.join(legacy_dir, "fig2_evq_yarn_orthogonality.pdf")
    legacy_png = os.path.join(legacy_dir, "fig2_evq_yarn_orthogonality.png")
    paper_pdf = os.path.join(paper_fig_dir, "fig2_evq_yarn_synergy.pdf")
    paper_png = os.path.join(paper_fig_dir, "fig2_evq_yarn_synergy.png")

    for path in [legacy_pdf, legacy_png, paper_pdf, paper_png]:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)

    print("Saved:")
    print(f"  {legacy_pdf}")
    print(f"  {legacy_png}")
    print(f"  {paper_pdf}")
    print(f"  {paper_png}")


if __name__ == "__main__":
    main()
