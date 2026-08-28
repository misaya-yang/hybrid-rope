"""Plot the matched 8B adaptation remote-source-use intervention.

Source: data/curated/llama8b_causal_source_use_s42_20260714.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/curated/llama8b_causal_source_use_s42_20260714.json"
OUT = Path(__file__).with_name("fig_8b_causal_source_use.pdf")

INK = "#17212B"
GRID = "#E2E6E9"
NATIVE = "#98A2AA"
EVQ = "#D35F45"

with SOURCE.open(encoding="utf-8") as handle:
    receipt = json.load(handle)

remote = receipt["true_16k_remote_source_use"]
hit = 100.0 * np.array([
    remote["median_target_block_hit_at_16"]["native"],
    remote["median_target_block_hit_at_16"]["evq"],
])
deletion = np.array([
    remote["nll_change_after_all_head_gold_block_deletion"]["native"],
    remote["nll_change_after_all_head_gold_block_deletion"]["evq"],
])

assert receipt["paper_role"] == "mature_8b_probability_and_causal_remote_source_use"
assert remote["frozen_case_count"] == 10
assert remote["retrieval_head_count"] == 32
assert np.allclose(hit, [18.75, 64.06])
assert np.allclose(deletion, [-0.0095, 1.5055])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.65,
})

fig, axes = plt.subplots(1, 2, figsize=(7.25, 1.85), gridspec_kw={"wspace": 0.30})
x = np.arange(2)
labels = ["Native-LoRA", "EVQ-Cosh-LoRA"]
colors = [NATIVE, EVQ]

ax = axes[0]
bars = ax.bar(x, hit, width=0.52, color=colors, edgecolor="white", linewidth=0.5)
for bar, value in zip(bars, hit):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.2f}%",
            ha="center", va="bottom", color=INK, fontsize=8.0, weight="bold")
ax.set_ylim(0, 72)
ax.set_ylabel("median target-block hit@16 (%)")
ax.set_title("(a) The adapted model routes to the remote source", loc="left",
             weight="bold", pad=5)
ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)

ax = axes[1]
bars = ax.bar(x, deletion, width=0.52, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color=INK, lw=0.75)
for bar, value in zip(bars, deletion):
    offset = 0.07 if value >= 0 else -0.07
    ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.4f}",
            ha="center", va="bottom" if value >= 0 else "top", color=INK,
            fontsize=8.0, weight="bold")
ax.set_ylim(-0.20, 1.72)
ax.set_ylabel(r"$\Delta$ NLL after gold-block deletion")
ax.set_title("(b) Removing that source changes answer probability", loc="left",
             weight="bold", pad=5)
ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)

for ax in axes:
    ax.set_xticks(x, labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)

fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.82)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
