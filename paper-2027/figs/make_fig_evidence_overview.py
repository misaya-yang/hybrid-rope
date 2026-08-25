"""Result-first ICLR overview from frozen paper evidence owners.

Sources:
  - EXACT_RANGE_151M_3SEED_RESULT_20260820.json
  - table18_mla_3seed_aggregate.json
  - SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("fig_evidence_overview.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
INK = "#17212B"
MUTED = "#69737D"
GRID = "#E2E6E9"
SEED = "#A8B0B7"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.4,
    "axes.linewidth": 0.6,
})


fig, axes = plt.subplots(
    1, 3, figsize=(7.25, 2.18),
    gridspec_kw={"width_ratios": [1.05, 1.12, 1.0]},
)

# (a) Fixed-support allocation identification: training seed is the unit.
ratios = np.array([1, 2, 4, 8])
exact_seed = np.array([
    [0.032759063, -0.477501452, -0.204985037, -0.112843767],
    [0.029025115, -0.270500794, -0.190884590, -0.180826515],
    [0.016798005, -0.094181113, -0.132105887, -0.143473938],
])
exact_mean = exact_seed.mean(axis=0)
ax = axes[0]
ax.axhspan(-0.55, 0, color="#F1F5F8", zorder=0)
ax.axhline(0, color=INK, lw=0.7, zorder=1)
for values in exact_seed:
    ax.plot(ratios, values, color=SEED, marker="o", ms=2.6, lw=0.85, zorder=2)
ax.plot(ratios, exact_mean, color=BLUE, marker="o", ms=4.2, lw=1.7,
        label="mean (3 seeds)", zorder=3)
for x, value in zip(ratios[1:], exact_mean[1:]):
    ax.text(x, value - 0.045, "3/3", color=BLUE, fontsize=5.9,
            ha="center", va="top", weight="bold")
ax.text(1, 0.052, "+0.026", color=MUTED, fontsize=6.0, ha="center")
ax.set_xticks(ratios, [r"$1\times$", r"$2\times$", r"$4\times$", r"$8\times$"])
ax.set_ylim(-0.57, 0.10)
ax.set_yticks([-0.5, -0.25, 0])
ax.set_xlabel("relative to training length")
ax.set_ylabel(r"$\Delta$NLL (anchored EVQ-Cosh $-$ FMRoPE)")
ax.set_title("(a) Fixed support, interior allocation", loc="left", weight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=5.9, handlelength=1.5)
ax.grid(axis="y", color=GRID, lw=0.45)

# (b) Scarce-channel MLA: training seed is again the unit.
mla_geo = np.array([
    [34.509, 141.069],
    [36.374, 132.528],
    [35.451, 142.825],
])
mla_evq = np.array([
    [35.065, 93.686],
    [36.642, 92.742],
    [35.617, 100.336],
])
mla_relative = 100 * (mla_evq / mla_geo - 1)
mla_mean = mla_relative.mean(axis=0)
ax = axes[1]
ax.axhspan(-38, 0, color="#F1F5F8", zorder=0)
ax.axhline(0, color=INK, lw=0.7, zorder=1)
for values in mla_relative:
    ax.plot([0, 1], values, color=SEED, marker="o", ms=2.6, lw=0.85,
            zorder=2)
ax.plot([0, 1], mla_mean, color=BLUE, marker="s", ms=4.2, lw=1.7,
        label="mean (3 seeds)", zorder=3)
ax.text(0, mla_mean[0] + 2.0, f"{mla_mean[0]:+.1f}%", color=MUTED,
        fontsize=6.0, ha="center")
ax.text(1, mla_mean[1] - 2.4, f"{mla_mean[1]:.1f}%", color=BLUE,
        fontsize=6.2, ha="center", va="top", weight="bold")
ax.text(1, -22.5, "3/3", color=BLUE, fontsize=5.9,
        ha="center", weight="bold")
ax.set_xticks([0, 1], ["8K (train)", "16K (2×)"])
ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-38, 8)
ax.set_yticks([-30, -15, 0])
ax.set_xlabel("evaluation length")
ax.set_ylabel("PPL change, EVQ-Cosh vs Geo (%)")
ax.set_title("(b) 432M MLA, $K=16$", loc="left", weight="bold")
ax.legend(frameon=False, loc="lower left", fontsize=5.9, handlelength=1.5)
ax.grid(axis="y", color=GRID, lw=0.45)

# (c) Released-model fixed-support intervention.
frozen_scores = np.array([0.555556, 60.472222, 61.041667])
x = np.arange(len(frozen_scores))
ax = axes[2]
bars = ax.bar(
    x, frozen_scores, width=0.62,
    color=["#D8DDE1", "#DDEAF4", "#F6E2DC"],
    edgecolor=[MUTED, BLUE, ORANGE], lw=0.8,
)
for bar, value, color in zip(bars, frozen_scores, [MUTED, BLUE, ORANGE]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
            f"{value:.2f}", ha="center", va="bottom", fontsize=6.2,
            color=color, weight="bold")
ax.plot([0, 0, 1, 1], [62, 65, 65, 62], color=BLUE, lw=0.7)
ax.text(0.5, 66.5, r"$+59.92$ points  (95\% CI $[54.88,64.80]$)",
        ha="center", va="bottom", color=BLUE, fontsize=5.6, weight="bold")
ax.set_xticks(x, ["Geometric", "Derived", "Coarse\nramp"])
ax.set_ylim(0, 72)
ax.set_yticks([0, 30, 60])
ax.set_xlabel("interior allocation (same support)")
ax.set_ylabel("RULER macro (%)")
ax.set_title("(c) Frozen 1.485B OLMo, 16K", loc="left", weight="bold")
ax.grid(axis="y", color=GRID, lw=0.45)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=1.05)

assert np.allclose(exact_mean, [0.026194061, -0.280727786,
                                -0.175991838, -0.145714740], atol=5e-10)
assert np.all(exact_seed[:, 1:] < 0)
assert np.allclose(mla_mean, [0.93873862, -31.11944897], atol=1e-7)
assert np.all(mla_relative[:, 1] < 0)
assert np.isclose(frozen_scores[1] - frozen_scores[0], 59.916666, atol=1e-8)
assert np.isclose(frozen_scores[2] - frozen_scores[1], 0.569445, atol=1e-8)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
