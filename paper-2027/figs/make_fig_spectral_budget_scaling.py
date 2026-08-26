#!/usr/bin/env python3
"""Render the appendix spectral-budget scaling diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.full_rope_collision_audit import (  # noqa: E402
    evq_phi,
    geometric_phi,
    schedule_metrics,
)


OUT = Path(__file__).resolve().parent / "fig_spectral_budget_scaling.pdf"


def main() -> None:
    pairs = np.array([16, 32, 64])
    length = 4096
    base = 5e5
    tau = 2 * pairs / np.sqrt(length)
    geo_phi = [geometric_phi(int(k)) for k in pairs]
    slow_phi = [phi[np.power(base, -phi) * length <= 1] for phi in geo_phi]
    slow_pairs = np.array([len(phi) for phi in slow_phi])
    nominal = 2 * slow_pairs
    slow_rank = np.array([
        schedule_metrics(phi, length, base)["full_whitened_stable_rank"]
        for phi in slow_phi
    ])
    geo_rank = np.array([
        schedule_metrics(phi, length, base)["full_whitened_stable_rank"]
        for phi in geo_phi
    ])
    cosh_rank = np.array([
        schedule_metrics(evq_phi(int(k), float(t)), length, base)["full_whitened_stable_rank"]
        for k, t in zip(pairs, tau)
    ])
    assert np.array_equal(slow_pairs, np.array([6, 12, 24]))
    assert np.all(slow_rank < 2.001)
    assert np.all(cosh_rank >= geo_rank)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

    ax = axes[0]
    x = np.arange(len(pairs))
    width = 0.34
    bars_nominal = ax.bar(x - width / 2, nominal, width, color="#C8D9E8", edgecolor="#2166AC", label="nominal slow-band dimensions")
    bars_rank = ax.bar(x + width / 2, slow_rank, width, color="#D6604D", label="block-whitened effective rank")
    for bar, value in zip(bars_nominal, nominal):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, str(value), ha="center", fontsize=8)
    for bar, value in zip(bars_rank, slow_rank):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}", ha="center", fontsize=8, color="#9B2F23")
    ax.set_xticks(x, [f"K={value}" for value in pairs])
    ax.set_ylabel("dimensions")
    ax.set_title("(a) Slow-band collapse", fontweight="bold")
    ax.set_ylim(0, 54)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(pairs, geo_rank, color="#2166AC", marker="o", linewidth=2.1, label="Geo")
    ax.plot(pairs, cosh_rank, color="#D6604D", marker="s", linewidth=2.1, label="endpoint-anchored EVQ-Cosh")
    for xpos, value in zip(pairs, cosh_rank):
        ax.text(xpos, value + 0.65, f"{value:.2f}", ha="center", color="#9B2F23", fontsize=8)
    ax.set_xticks(pairs, [str(value) for value in pairs])
    ax.set_xlabel("rotary pairs K")
    ax.set_ylabel("full-table block-whitened r2")
    ax.set_title("(b) Reallocating the finite budget", fontweight="bold")
    ax.set_ylim(0, 20)
    ax.grid(color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.1)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
