"""Result-first ICLR overview from frozen paper evidence owners.

Sources:
  - EXACT_RANGE_151M_3SEED_RESULT_20260820.json
  - OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md
  - OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md
  - EVQ_8B_ADAPTATION_EVIDENCE_20260724.md
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
ax.set_ylabel(r"$\Delta$NLL (Cosh $-$ uniform)")
ax.set_title("(a) Fixed support, interior allocation", loc="left", weight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=5.9, handlelength=1.5)
ax.grid(axis="y", color=GRID, lw=0.45)

# (b) The largest from-initialization comparison: same scientific recipe.
lengths = np.array([2, 4, 8, 16])
geo_ppl = np.array([177.99, 161.19, 163.88, 182.73])
evq_ppl = np.array([191.36, 167.45, 156.87, 159.64])
ax = axes[1]
ax.axvspan(4.1, 16.8, color="#F7EFE9", zorder=0)
ax.axvline(4, color=MUTED, lw=0.7, ls="--")
ax.plot(lengths, geo_ppl, color=MUTED, marker="o", ms=3.6, lw=1.45,
        label="Geo")
ax.plot(lengths, evq_ppl, color=BLUE, marker="s", ms=3.6, lw=1.65,
        label="EVQ-Cosh")
ax.annotate("Geo 182.7", (16, geo_ppl[-1]), xytext=(-4, 5),
            textcoords="offset points", ha="right", color=MUTED, fontsize=6.0)
ax.annotate("EVQ 159.6", (16, evq_ppl[-1]), xytext=(-4, -10),
            textcoords="offset points", ha="right", color=BLUE, fontsize=6.0,
            weight="bold")
ax.text(10.2, 195.2, "beyond 4K", color=ORANGE, fontsize=6.0,
        ha="center", weight="bold")
ax.text(0.97, 0.04, "focused PPL scale", transform=ax.transAxes,
        color=MUTED, fontsize=5.6, ha="right")
ax.set_xticks(lengths, ["2K", "4K", "8K", "16K"])
ax.set_xlim(1.4, 16.6)
ax.set_ylim(148, 198)
ax.set_xlabel("evaluation length")
ax.set_ylabel("PPL")
ax.set_title("(b) 1.485B from initialization", loc="left", weight="bold")
ax.grid(axis="y", color=GRID, lw=0.45)

# (c) Real-document QA plus a separate mature-model causal-use check.
x = np.arange(3)
native_2wiki = np.array([22.0, 0.0, 0.0])
evq_2wiki = np.array([21.5, 17.5, 4.0])
width = 0.34
ax = axes[2]
bars_native = ax.bar(x - width / 2, native_2wiki, width, color="#D8DDE1",
                     edgecolor=MUTED, lw=0.6)
bars_evq = ax.bar(x + width / 2, evq_2wiki, width, color="#DDEAF4",
                  edgecolor=BLUE, lw=0.8)
for bars, values, color in (
    (bars_native, native_2wiki, MUTED),
    (bars_evq, evq_2wiki, BLUE),
):
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6,
                f"{value:g}", ha="center", va="bottom", fontsize=5.8,
                color=color, weight="bold" if color == BLUE else None)
ax.legend([bars_native, bars_evq], ["Native", "EVQ-Cosh"], frameon=False,
          ncol=2, loc="upper left", fontsize=5.3, handlelength=0.9,
          columnspacing=0.8, handletextpad=0.35, borderaxespad=0.15)
ax.text(0.98, 0.82, r"8B deletion: $\Delta$NLL $-0.01\;\to\;+1.51$",
        transform=ax.transAxes, ha="right", color=ORANGE, fontsize=5.4,
        weight="bold")
ax.set_xticks(x, [r"$1\times$", r"$2\times$", r"$4\times$"])
ax.set_ylim(0, 29)
ax.set_xlabel("relative to 4K physical cap")
ax.set_ylabel("2Wiki exact match (%)")
ax.set_title("(c) 1.485B real-document QA", loc="left", weight="bold")
ax.grid(axis="y", color=GRID, lw=0.45)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=1.05)

assert np.allclose(exact_mean, [0.026194061, -0.280727786,
                                -0.175991838, -0.145714740], atol=5e-10)
assert np.all(exact_seed[:, 1:] < 0)
assert np.isclose(100 * (24.068 / 108.958 - 1), -77.91, atol=0.02)
assert np.allclose(native_2wiki, [22.0, 0.0, 0.0])
assert np.allclose(evq_2wiki, [21.5, 17.5, 4.0])

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
