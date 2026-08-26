#!/usr/bin/env python3
"""Render the 1.485B same-initialisation full-parameter crossover figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
LENGTHS = np.array([2, 4, 8, 16])
DELTA = np.array([0.0724, 0.0381, -0.0437, -0.1351])
CI_LOW = np.array([0.0664, 0.0332, -0.0494, -0.1420])
CI_HIGH = np.array([0.0779, 0.0428, -0.0380, -0.1281])
FAVOUR = np.array([3, 7, 122, 126])


def main() -> None:
    assert np.all(CI_LOW <= DELTA)
    assert np.all(DELTA <= CI_HIGH)
    assert np.all(CI_LOW[:2] > 0) and np.all(CI_HIGH[2:] < 0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax, axb) = plt.subplots(
        2,
        1,
        figsize=(6.9, 3.55),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1], "hspace": 0.10},
    )
    color = "#176B87"
    yerr = np.vstack((DELTA - CI_LOW, CI_HIGH - DELTA))
    ax.errorbar(
        LENGTHS,
        DELTA,
        yerr=yerr,
        color=color,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        elinewidth=1.2,
        capsize=3.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax.fill_between([7.3, 16.7], -0.155, 0, color="#DCEEF2", alpha=0.65, zorder=0)
    ax.text(15.7, -0.012, "EVQ-Cosh lower NLL", ha="right", va="top", color=color)
    ax.set_ylabel("EVQ-Cosh − Geo full NLL")
    ax.set_title("1.485B full-parameter crossover after 2.097B training tokens", fontweight="bold")
    ax.set_ylim(-0.155, 0.095)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    pct = 100 * FAVOUR / 128
    bars = axb.bar(LENGTHS, pct, width=[1.2, 1.5, 2.2, 3.0], color="#B9D8E1", edgecolor=color)
    for bar, count in zip(bars, FAVOUR):
        axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{count}/128", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axb.set_ylabel("documents\nfavouring EVQ")
    axb.set_ylim(0, 112)
    axb.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    axb.set_xticks(LENGTHS, ["2K", "4K", "8K", "16K"])
    axb.set_xlabel("evaluation length")
    axb.grid(axis="y", color="#E2E2E2", linewidth=0.5)
    axb.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.15)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_olmo_scale_crossover.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
