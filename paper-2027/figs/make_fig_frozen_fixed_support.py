"""Plot the mature-checkpoint fixed-support allocation intervention.

The two panels remain separate experimental units. Grey bars are standard
reference rows; the shaded lower block holds support, amplitude, checkpoint,
rows, decoder, precision, and hardware fixed while changing only interior z.

Source: data/curated/frozen_fixed_support_mature_20260823.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/curated/frozen_fixed_support_mature_20260823.json"
OUT = Path(__file__).with_name("fig_frozen_fixed_support.pdf")

INK = "#17212B"
MUTED = "#69737D"
GRID = "#E2E6E9"
REFERENCE = ["#C5CBD0", "#8D979F"]
PURE_Z = ["#6FA8CC", "#2A8C6A", "#D35F45"]
PURE_Z_BG = "#F0F5F8"


with SOURCE.open(encoding="utf-8") as handle:
    receipt = json.load(handle)


def values_for(key: str) -> np.ndarray:
    block = receipt[key]
    return 100.0 * np.array([
        block["reference_macro"]["native"],
        block["reference_macro"]["official_transformers_yarn_factor4"],
        block["macro"]["same_support_geometric"],
        block["macro"]["nearest_movement_profile_ramp"],
        block["macro"]["derived"],
    ])


olmo = values_for("olmo_16k_unseen9")
qwen = values_for("qwen_64k_core4")
olmo_ci = 100.0 * np.array(
    receipt["olmo_16k_unseen9"]["evaluation_row_intervals"]
    ["derived_minus_geometric"]
)
qwen_ci = 100.0 * np.array(
    receipt["qwen_64k_core4"]["evaluation_row_intervals"]
    ["derived_minus_geometric"]
)

assert receipt["changed"] == "interior normalized exponent allocation z"
assert np.allclose(olmo, [0.0, 7.94, 0.5555555556, 61.0370370370, 60.4722222222])
assert np.allclose(qwen, [54.50, 60.25, 57.75, 64.00, 66.50])
assert np.allclose(olmo_ci, [54.8796296296, 64.7962962963])
assert np.allclose(qwen_ci, [0.25, 17.50])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.65,
})

labels = [
    "Native",
    "Official YaRN",
    r"Uniform $z$",
    r"Coarse label-free $z$",
    r"Derived $z$",
]
y = np.arange(len(labels))
colors = REFERENCE + PURE_Z

fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.35), sharex=True, sharey=True)
panels = [
    (axes[0], olmo, "(a) OLMo 1.485B: unseen-nine at 16K", olmo_ci),
    (axes[1], qwen, "(b) Qwen 1.5B: core-four at 64K", qwen_ci),
]

for ax, values, title, interval in panels:
    ax.axhspan(1.55, 4.45, color=PURE_Z_BG, zorder=0)
    bars = ax.barh(y, values, height=0.58, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=2)
    ax.axhline(1.5, color=MUTED, lw=0.65, zorder=3)
    ax.text(69.5, 1.60, r"matched support; only $z$ changes",
            ha="right", va="top", color="#2F6DAA", fontsize=6.7,
            weight="bold")
    for bar, value in zip(bars, values):
        x = 1.1 if value == 0 else min(value + 1.0, 69.0)
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                va="center", ha="left" if value < 68 else "right",
                color=INK, fontsize=7.0, weight="bold")
    delta = values[-1] - values[2]
    ax.text(0.0, 1.01,
            rf"pure-$z$: {values[2]:.2f} $\rightarrow$ {values[-1]:.2f}"
            "\n" + rf"$\Delta={delta:+.2f}$ [{interval[0]:.2f}, {interval[1]:.2f}]",
            transform=ax.transAxes, ha="left", va="bottom", color="#B84E38",
            fontsize=7.0, weight="bold")
    ax.set_title(title, loc="left", weight="bold", pad=21)
    ax.set_xlim(0, 72)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("RULER macro (%)")
    ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()

fig.subplots_adjust(left=0.17, right=0.995, bottom=0.18, top=0.78, wspace=0.10)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
