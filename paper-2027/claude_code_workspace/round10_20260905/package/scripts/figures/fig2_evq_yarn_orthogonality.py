"""
Figure 2 for the NeurIPS paper: primary EVQ x YaRN retrieval anchor.

Data source:
  data/curated/table2_evq_yarn_454m_passkey_10pct.json

Only uses the paper-safe 10% passkey-mix, matched YaRN scale=8 block:
  - 454M model
  - 3 seeds per configuration
  - PK@8K teacher-forced NLL-gap retrieval

Usage:
  python scripts/figures/fig2_evq_yarn_orthogonality.py

Output:
  - paper/figs/fig2_evq_yarn_synergy.pdf/.png
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


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }
)


C_GEO = "#2166ac"
C_EVQ = "#d6604d"
C_GEO_LIGHT = "#d9e7f5"
C_EVQ_LIGHT = "#f7ddd8"
C_GRID = "#d9d9d9"
C_TEXT = "#333333"

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "curated"
    / "table2_evq_yarn_454m_passkey_10pct.json"
)


def load_table2_summary() -> dict[str, dict[str, object]]:
    """Load reviewer-facing curated values used by Tables 2--3."""
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return payload["table2_full_sequence_summary"]


def load_seedwise_8k(summary: dict[str, dict[str, object]]) -> dict[str, np.ndarray]:
    seed_order = ("42", "123", "7")
    return {
        name: np.array(
            [summary[name]["pk_8k_seedwise"][seed] for seed in seed_order],
            dtype=float,
        )
        for name in ("Geo", "Geo+YaRN", "EVQ", "EVQ+YaRN")
    }


def bar_style(name: str) -> dict[str, object]:
    is_evq = name.startswith("EVQ")
    return {
        "facecolor": C_EVQ_LIGHT if is_evq else C_GEO_LIGHT,
        "edgecolor": C_EVQ if is_evq else C_GEO,
        "hatch": "//" if "YaRN" in name else "",
    }


def add_gain_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str) -> None:
    ax.plot([x0, x0, x1, x1], [y - 2, y, y, y - 2], color=C_TEXT, lw=0.8, clip_on=False)
    ax.text((x0 + x1) / 2, y + 1.2, text, ha="center", va="bottom", fontsize=7.3, color=C_TEXT)


def main() -> None:
    categories = ["Geo", "Geo+YaRN", "EVQ", "EVQ+YaRN"]
    summary = load_table2_summary()
    seedwise_8k = load_seedwise_8k(summary)
    means = np.array([seedwise_8k[name].mean() for name in categories])
    stds = np.array([seedwise_8k[name].std(ddof=1) for name in categories])
    rounded = np.array([summary[name]["pk"]["8k"] for name in categories], dtype=int)

    fig, ax = plt.subplots(figsize=(4.9, 2.45), constrained_layout=True)
    x = np.arange(len(categories), dtype=float)
    width = 0.62

    for idx, name in enumerate(categories):
        style = bar_style(name)
        ax.bar(
            x[idx],
            means[idx],
            width=width,
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=1.0,
            hatch=style["hatch"],
            zorder=2,
        )
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=stds[idx],
            fmt="none",
            ecolor=style["edgecolor"],
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        jitter = np.array([-0.09, 0.0, 0.09])
        ax.scatter(
            np.full(3, x[idx]) + jitter,
            seedwise_8k[name],
            color=style["edgecolor"],
            s=18,
            marker="s" if name.startswith("EVQ") else "o",
            zorder=5,
        )
        ax.text(
            x[idx],
            min(means[idx] + stds[idx] + 5.0, 107.0),
            f"{rounded[idx]}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=style["edgecolor"],
            fontweight="bold",
        )

    add_gain_bracket(ax, x[1], x[3], 109.0, "+39 pp at 8K")
    ax.annotate(
        "matched YaRN scale",
        xy=(x[1], means[1]),
        xytext=(0.55, 83),
        ha="center",
        va="center",
        fontsize=7.2,
        color=C_GEO,
        arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=0.8, mutation_scale=8),
    )
    ax.set_title("PK@8K, 454M passkey-mix, 3 seeds", fontweight="bold", pad=7)
    ax.set_ylabel("Teacher-forced passkey retrieval")
    ax.set_xticks(x)
    ax.set_xticklabels(["Geo", "Geo\n+YaRN", "EVQ", "EVQ\n+YaRN"])
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.5,
        -0.28,
        "Dots: individual seeds; whiskers: one standard deviation. PK is NLL-gap retrieval, not AR exact match.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.4,
        color="#666666",
    )

    paper_fig_dir = "paper/figs"
    os.makedirs(paper_fig_dir, exist_ok=True)
    paper_pdf = os.path.join(paper_fig_dir, "fig2_evq_yarn_synergy.pdf")
    paper_png = os.path.join(paper_fig_dir, "fig2_evq_yarn_synergy.png")

    for path in [paper_pdf, paper_png]:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)

    print("Saved:")
    print(f"  {paper_pdf}")
    print(f"  {paper_png}")


if __name__ == "__main__":
    main()
