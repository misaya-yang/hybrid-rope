#!/usr/bin/env python3
"""Render the appendix fixed-support versus target-matched control figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
RATIOS = np.array([1, 2, 4, 8])
SEEDS = (42, 137, 256)
FIXED = np.array(
    [
        [0.033, -0.478, -0.205, -0.113],
        [0.029, -0.271, -0.191, -0.181],
        [0.017, -0.094, -0.132, -0.143],
    ]
)
TARGET = np.array(
    [
        [0.033, 0.061, 0.182, 0.279],
        [0.029, 0.053, 0.156, 0.399],
        [0.017, 0.067, 0.344, 0.701],
    ]
)


def main() -> None:
    assert np.all(FIXED[:, 1:] < 0)
    assert np.all(TARGET[:, 1:] > 0)
    assert np.allclose(FIXED[:, 0], TARGET[:, 0])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharex=True, sharey=True)
    panels = (
        (axes[0], FIXED, "(a) Pinned training support", "#176B87"),
        (axes[1], TARGET, "(b) Target-matched support", "#B14A3B"),
    )

    for ax, values, title, color in panels:
        for row, seed in zip(values, SEEDS):
            ax.plot(
                RATIOS,
                row,
                color=color,
                alpha=0.38,
                linewidth=1.1,
                marker="o",
                markersize=3.2,
                label=f"seed {seed}",
            )
        ax.plot(
            RATIOS,
            values.mean(axis=0),
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="mean",
            zorder=5,
        )
        ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
        ax.set_title(title, fontweight="bold", pad=7)
        ax.set_xticks(RATIOS, ["1×", "2×", "4×", "8×"])
        ax.set_xlabel("evaluation / training length")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Cosh − Geo tail NLL")
    axes[0].text(7.85, -0.055, "Cosh allocation favoured", ha="right", va="top", color="#176B87", fontsize=8)
    axes[1].text(7.85, 0.055, "Geometric control favoured", ha="right", va="bottom", color="#B14A3B", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2.0)

    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_exact_range_control.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
