#!/usr/bin/env python3
"""Render the two appendix range-composition panels as separate figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/curated/table2_evq_yarn_454m_passkey_10pct.json"

C_GEO = "#2166AC"
C_EVQ = "#D6604D"
C_GEO_LIGHT = "#D9E7F5"
C_EVQ_LIGHT = "#F7DDD8"


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    summary = payload["table2_full_sequence_summary"]
    categories = ("Geo", "Geo+YaRN", "EVQ", "EVQ+YaRN")
    seed_order = ("42", "123", "7")
    seedwise = {
        name: np.array([summary[name]["pk_8k_seedwise"][seed] for seed in seed_order], dtype=float)
        for name in categories
    }
    means = np.array([seedwise[name].mean() for name in categories])
    stds = np.array([seedwise[name].std(ddof=1) for name in categories])
    rounded = np.array([summary[name]["pk"]["8k"] for name in categories], dtype=int)
    assert tuple(rounded) == (41, 61, 53, 100)

    ratios = np.array([4, 8, 16, 32])
    ppl = {
        "Geo+YaRN-style": np.array([60.7, 97.5, 190.4, 260.2]),
        "EVQ-Cosh (τ=2)+YaRN-style": np.array([51.6, 66.7, 107.9, 141.7]),
        "EVQ-Cosh (τ=4)+YaRN-style": np.array([50.9, 60.9, 84.1, 99.6]),
    }
    assert np.all(ppl["EVQ-Cosh (τ=4)+YaRN-style"] < ppl["Geo+YaRN-style"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 2.65))
    x = np.arange(len(categories))
    for i, name in enumerate(categories):
        evq = name.startswith("EVQ")
        edge = C_EVQ if evq else C_GEO
        face = C_EVQ_LIGHT if evq else C_GEO_LIGHT
        ax.bar(i, means[i], width=0.62, facecolor=face, edgecolor=edge,
               linewidth=1.0, hatch="//" if "YaRN" in name else "", zorder=2)
        ax.errorbar(i, means[i], yerr=stds[i], fmt="none", ecolor=edge,
                    elinewidth=1.0, capsize=3, zorder=4)
        ax.scatter(i + np.array([-0.09, 0, 0.09]), seedwise[name], color=edge,
                   s=17, marker="s" if evq else "o", zorder=5)
        ax.text(i, min(means[i] + stds[i] + 4, 106), f"{rounded[i]}%",
                ha="center", color=edge, fontweight="bold")
    ax.set_title("454M: same range operator", fontweight="bold")
    ax.set_ylabel("PK@8K teacher-forced retrieval")
    ax.set_xticks(x, ["Geo", "Geo\n+YaRN", "EVQ", "EVQ\n+YaRN"])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_range_composition_454m.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.9, 7.4))
    styles = (
        ("Geo+YaRN-style", C_GEO, "o", "-"),
        ("EVQ-Cosh (τ=2)+YaRN-style", "#E89A88", "s", "-"),
        ("EVQ-Cosh (τ=4)+YaRN-style", C_EVQ, "s", "-"),
    )
    for name, color, marker, linestyle in styles:
        ax.plot(ratios, ppl[name], label=name, color=color, marker=marker,
                linestyle=linestyle, linewidth=2 if "τ=4" in name else 1.4,
                markersize=4.5)
    ax.set_title("125M: range-composition curve", fontweight="bold")
    ax.set_xlabel("extrapolation ratio")
    ax.set_ylabel("perplexity")
    ax.set_xticks(ratios, ["4×", "8×", "16×", "32×"])
    ax.grid(color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_range_composition_125m.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
