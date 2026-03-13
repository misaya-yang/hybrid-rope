"""
Figure 3 for NeurIPS paper: PE-dominant regime and scaling-law confirmation.

Panels:
  (a) 128-token extreme extrapolation against DAPE-style learnable PE baselines
  (b) Phase 11 raw PPL vs extrapolation ratio (3-seed mean ± std)
  (c) Phase 11 +YaRN PPL vs extrapolation ratio (3-seed mean ± std)

Usage:
  ~/miniconda3/bin/conda run -n aidemo python scripts/figures/fig3_pe_dominant_scaling.py

Output:
  - paper/figs/fig3_pe_dominant_scaling.pdf/.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    }
)


C_GEO = "#2166ac"
C_GEO_LIGHT = "#92c5de"
C_EVQ = "#d6604d"
C_EVQ_LIGHT = "#f4a582"
C_DAPE = "#4d4d4d"
C_LEARN = "#7b8da6"
C_GRID = "#d9d9d9"
C_TEXT = "#333333"

LENGTHS = np.array([256, 512, 1024, 2048, 4096, 8192], dtype=float)
RATIO_LABELS = ["1x", "2x", "4x", "8x", "16x", "32x"]


def setup_axis(ax):
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def load_json(path: str):
    return json.loads(Path(path).read_text())


def load_phase6_extreme_data():
    sweep = load_json("data/evq_128tok_results/results_phase6.json")
    ckpt = load_json("data/evq_128tok_results/results_checkpoint.json")

    fineweb_curve = sweep["experiments"]["6A_extended_sweep"]["complete_tau_curve"]["fineweb"]
    evq_tau5 = next(item for item in fineweb_curve if item["tau"] == 5.0)

    experiments = ckpt["experiments"]
    geo = experiments["125m_tau0.00_seed42"]
    learnable = experiments["125m_learnable_init1.00_seed42"]
    dape = experiments["125m_dape_lrmult100_seed42"]

    return {
        "Geo": {"ppl_128": geo["ppl"]["128"], "ppl_8192": geo["ppl"]["8192"], "color": C_GEO},
        "Learnable $\\tau$": {
            "ppl_128": learnable["ppl"]["128"],
            "ppl_8192": learnable["ppl"]["8192"],
            "color": C_LEARN,
        },
        "DAPE (32p)": {
            "ppl_128": dape["ppl"]["128"],
            "ppl_8192": dape["ppl"]["8192"],
            "color": C_DAPE,
        },
        "EVQ $\\tau$=5.0": {
            "ppl_128": evq_tau5["ppl_128"],
            "ppl_8192": evq_tau5["ppl_8192"],
            "color": C_EVQ,
        },
    }


def summarize_phase11(path: str, field: str):
    data = load_json(path)
    methods = {"geo": [], "evq2.0": [], "evq4.0": []}
    for rec in data.values():
        ppl = rec[field]
        methods[rec["method"]].append([ppl[str(int(length))] for length in LENGTHS])

    summary = {}
    for method, rows in methods.items():
        arr = np.asarray(rows, dtype=float)
        summary[method] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
    return summary


def draw_extreme_panel(ax):
    data = load_phase6_extreme_data()
    labels = list(data.keys())
    ppl_8k = [data[label]["ppl_8192"] for label in labels]
    colors = [data[label]["color"] for label in labels]
    x = np.arange(len(labels))

    ax.bar(x, ppl_8k, color=colors, edgecolor=colors, linewidth=0.8, alpha=0.88, zorder=2)

    for idx, value in enumerate(ppl_8k):
        ax.text(
            idx,
            value + 18,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=colors[idx],
            fontweight="bold",
        )

    geo_8k = data["Geo"]["ppl_8192"]
    evq_8k = data["EVQ $\\tau$=5.0"]["ppl_8192"]
    dape_8k = data["DAPE (32p)"]["ppl_8192"]

    ax.annotate(
        "EVQ beats DAPE\nby 27% at 64x",
        xy=(3, evq_8k),
        xytext=(2.1, 610),
        fontsize=6.2,
        color=C_EVQ,
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVQ, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVQ, linewidth=0.6),
    )
    ax.annotate(
        "Geo baseline",
        xy=(0, geo_8k),
        xytext=(0.25, 560),
        fontsize=5.8,
        color=C_GEO,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=0.7, mutation_scale=7),
    )

    ax.text(
        1.5,
        95,
        "125M FineWeb, 128$\\rightarrow$8192 (64x)\n15M tokens, direct DAPE-style regime",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#f7f7f7", edgecolor="none", alpha=0.95),
    )

    ax.set_title("(a) Extreme extrapolation beats learnable PE", fontweight="bold", pad=6)
    ax.set_ylabel("PPL at 8K")
    ax.set_xticks(x)
    ax.set_xticklabels(["Geo", "Learnable\n$\\tau$", "DAPE\n(32p)", "EVQ\n(0p)"])
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylim(0, 720)
    ax.set_yticks([0, 150, 300, 450, 600])
    setup_axis(ax)

    inset = inset_axes(ax, width="48%", height="38%", loc="upper left", borderpad=1.0)
    id_vals = [data[label]["ppl_128"] for label in labels]
    inset.bar(x, id_vals, color=colors, edgecolor=colors, linewidth=0.6, alpha=0.88)
    inset.set_ylim(178, 186)
    inset.set_xticks([])
    inset.set_yticks([180, 184])
    inset.tick_params(axis="y", labelsize=5.5)
    inset.set_title("ID PPL@128", fontsize=6.0, pad=2)
    for spine in ["top", "right"]:
        inset.spines[spine].set_visible(False)
    inset.grid(axis="y", color=C_GRID, linewidth=0.4, alpha=0.5)
    inset.text(
        1.5,
        184.9,
        "all within 1.6%",
        fontsize=5.3,
        color="#555555",
        ha="center",
        va="top",
    )


RAW_STYLE = {
    "geo": {"label": "Geo", "color": C_GEO, "marker": "o"},
    "evq2.0": {"label": "EVQ $\\tau$=2.0", "color": C_EVQ_LIGHT, "marker": "s"},
    "evq4.0": {"label": "EVQ $\\tau$=4.0", "color": C_EVQ, "marker": "^"},
}


YARN_STYLE = {
    "geo": {"label": "Geo+YaRN", "color": C_GEO, "marker": "o"},
    "evq2.0": {"label": "EVQ2+YaRN", "color": C_EVQ_LIGHT, "marker": "s"},
    "evq4.0": {"label": "EVQ4+YaRN", "color": C_EVQ, "marker": "^"},
}


def draw_curve_panel(ax, summary, style_map, title):
    x = np.arange(len(LENGTHS))
    for method in ["geo", "evq2.0", "evq4.0"]:
        mean = summary[method]["mean"]
        std = summary[method]["std"]
        style = style_map[method]
        ax.plot(
            x,
            mean,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.12, zorder=2)

    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlabel("Extrapolation ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS)
    setup_axis(ax)


def add_endpoint_label(ax, x, y, text, color, dy=0.0):
    ax.text(
        x,
        y + dy,
        text,
        fontsize=5.8,
        color=color,
        ha="left",
        va="center",
        fontweight="bold",
    )


def draw_raw_panel(ax):
    summary = summarize_phase11("results/core_text/phase11/results_phase11_raw.json", "ppl")
    draw_curve_panel(ax, summary, RAW_STYLE, "(b) Phase 11 raw scaling-law confirmation")
    ax.set_ylabel("Perplexity")
    ax.set_ylim(45, 285)
    ax.set_yticks([50, 100, 150, 200, 250])

    ax.annotate(
        "predicted $\\tau^*=4.0$\nobserved best raw curve",
        xy=(5, summary["evq4.0"]["mean"][5]),
        xytext=(3.2, 232),
        fontsize=6.2,
        color=C_EVQ,
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVQ, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVQ, linewidth=0.6),
    )
    ax.text(
        4.5,
        154,
        "-37.5%",
        fontsize=6.0,
        color=C_EVQ,
        ha="left",
        va="center",
        fontweight="bold",
    )
    add_endpoint_label(ax, 5.08, summary["geo"]["mean"][5], "Geo", C_GEO, dy=5)
    add_endpoint_label(ax, 5.08, summary["evq2.0"]["mean"][5], "EVQ2", C_EVQ_LIGHT, dy=-2)
    add_endpoint_label(ax, 5.08, summary["evq4.0"]["mean"][5], "EVQ4", C_EVQ, dy=-14)


def draw_yarn_panel(ax):
    summary = summarize_phase11("results/core_text/phase11/results_phase11_yarn.json", "yarn_auto")
    draw_curve_panel(ax, summary, YARN_STYLE, "(c) EVQ unlocks YaRN in PE-dominant regime")
    ax.set_ylabel("Perplexity")
    ax.set_ylim(45, 285)
    ax.set_yticks([50, 100, 150, 200, 250])

    geo_32 = summary["geo"]["mean"][5]
    evq4_32 = summary["evq4.0"]["mean"][5]
    ax.annotate(
        "99.6 vs 260.2 at 32x\nYaRN leverage: ~10x",
        xy=(5, evq4_32),
        xytext=(3.15, 230),
        fontsize=6.1,
        color=C_EVQ,
        ha="center",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_EVQ, lw=0.8, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3ef", edgecolor=C_EVQ, linewidth=0.6),
    )
    add_endpoint_label(ax, 5.08, summary["geo"]["mean"][5], "Geo+YaRN", C_GEO, dy=6)
    add_endpoint_label(ax, 5.08, summary["evq2.0"]["mean"][5], "EVQ2+YaRN", C_EVQ_LIGHT, dy=-1)
    add_endpoint_label(ax, 5.08, summary["evq4.0"]["mean"][5], "EVQ4+YaRN", C_EVQ, dy=-10)


def main():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.55, 2.65),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.02, 1.05, 1.05], "wspace": 0.34},
    )

    draw_extreme_panel(axes[0])
    draw_raw_panel(axes[1])
    draw_yarn_panel(axes[2])

    paper_fig_dir = "paper/figs"
    os.makedirs(paper_fig_dir, exist_ok=True)

    paper_pdf = os.path.join(paper_fig_dir, "fig3_pe_dominant_scaling.pdf")
    paper_png = os.path.join(paper_fig_dir, "fig3_pe_dominant_scaling.png")

    fig.savefig(paper_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paper_png, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {paper_pdf}")
    print(f"Saved {paper_png}")


if __name__ == "__main__":
    main()
