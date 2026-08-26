"""Build the paper's core figure from analytic schedules and the exact-range owner.

The figure visualizes one scientific chain rather than the experiment inventory:
fixed support admits different interior allocations; slow rotary pairs collapse
toward the same positional directions; the paired training intervention tests
whether changing that allocation changes learned behaviour.

Sources:
  - analytic EVQ-Cosh inverse CDF in the manuscript
  - EXACT_RANGE_151M_3SEED_RESULT_20260820.json
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.full_rope_collision_audit import gram_blocks, whitened_blocks


OUT = Path(__file__).with_name("fig_evidence_overview.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
INK = "#17212B"
MUTED = "#69737D"
SEED = "#A8B0B7"
GRID = "#E2E6E9"
PANEL = "#F3F5F7"


def evq_phi(indices: np.ndarray, pairs: int, tau: float) -> np.ndarray:
    """Midpoint inverse-CDF allocation, normalized to fixed endpoints."""
    u = (indices + 0.5) / pairs
    raw = 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
    return (raw - raw[0]) / (raw[-1] - raw[0])


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.2,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.4,
    "axes.linewidth": 0.65,
})


fig = plt.figure(figsize=(7.25, 2.72))
grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.08, 1.16], wspace=0.42)

# (a) A finite set of slots can be placed differently inside the same support.
ax = fig.add_subplot(grid[0, 0])
pairs = 32
k = np.arange(pairs)
geo = k / (pairs - 1)
cosh = evq_phi(k, pairs, tau=4.0)

ax.axvspan(0.72, 1.0, color="#F4ECE8", zorder=0)
ax.text(0.86, 0.08, "slow end", color=ORANGE, fontsize=6.2,
        ha="center", va="center")
ax.hlines([0.72, 0.32], 0, 1, color=INK, lw=0.55, zorder=1)

for index in [3, 7, 11, 15, 19, 23, 27]:
    ax.plot([geo[index], cosh[index]], [0.70, 0.34], color="#D5DADF",
            lw=0.55, zorder=1)

ax.scatter(geo, np.full(pairs, 0.72), s=10, color=BLUE, edgecolor="white",
           lw=0.25, zorder=3)
ax.scatter(cosh, np.full(pairs, 0.32), s=10, color=ORANGE,
           edgecolor="white", lw=0.25, zorder=3)
for y in [0.72, 0.32]:
    ax.scatter([0, 1], [y, y], s=38, facecolor="white", edgecolor=INK,
               lw=0.9, zorder=4)

support = FancyArrowPatch((0, 0.93), (1, 0.93), arrowstyle="<->",
                          mutation_scale=8, color=INK, lw=0.75)
ax.add_patch(support)
ax.text(0.5, 0.955, r"same sampled support $(a,R)$", ha="center",
        va="bottom", color=INK, fontsize=6.6)
ax.text(-0.045, 0.72, "Geo", color=BLUE, ha="right", va="center",
        fontsize=6.8, weight="bold")
ax.text(-0.045, 0.32, "EVQ-Cosh", color=ORANGE, ha="right", va="center",
        fontsize=6.8, weight="bold")
ax.text(0.5, -0.11, r"$x_k=-\log\omega_k=a+Rz_k$", transform=ax.transAxes,
        ha="center", color=INK, fontsize=7.5)
ax.text(0.0, -0.02, r"fast / high $\omega$", ha="left", va="top",
        color=MUTED, fontsize=6.2)
ax.text(1.0, -0.02, r"slow / low $\omega$", ha="right", va="top",
        color=MUTED, fontsize=6.2)
ax.set_xlim(-0.16, 1.03)
ax.set_ylim(-0.12, 1.04)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
ax.set_title("(a) Same support, different allocation", loc="left",
             weight="bold", pad=4)

# (b) The exact slow-block spectrum: 46 nominal dimensions, two effective ones.
ax = fig.add_subplot(grid[0, 1])
geometry_pairs = 64
length = 4096.0
base = 500_000.0
phi = np.arange(geometry_pairs) / geometry_pairs
omega = np.power(base, -phi)
slow = omega * length <= 1
white = whitened_blocks(gram_blocks(omega, length))
slow_white = white[np.ix_(slow, slow)]
slow_gram = slow_white.transpose(0, 2, 1, 3).reshape(2 * slow.sum(),
                                                            2 * slow.sum())
eigenvalues = np.linalg.eigvalsh(0.5 * (slow_gram + slow_gram.T))[::-1]
shares = 100.0 * np.clip(eigenvalues, 0.0, None) / eigenvalues.sum()
directions = np.arange(1, len(shares) + 1)
colors = [BLUE if index < 2 else "#D6DBDF" for index in range(len(shares))]
ax.bar(directions, shares, width=0.72, color=colors, edgecolor="white",
       linewidth=0.25, zorder=2)
ax.axhline(0, color=INK, lw=0.6)
ax.text(2.9, 47.8, ">99.99% of trace\nin two directions", color=BLUE,
        fontsize=6.2, weight="bold", va="top")
ax.text(0.96, 0.75, r"$r_2=2.00$", transform=ax.transAxes, ha="right",
        va="top", color=INK, fontsize=8.2, weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=PANEL,
                  edgecolor="none"))
ax.text(0.98, 0.62, r"23 slow pairs $=$ 46 nominal dimensions",
        transform=ax.transAxes, ha="right", va="top", color=MUTED,
        fontsize=5.9)
ax.set_xlim(0.2, len(shares) + 0.8)
ax.set_ylim(0, 54)
ax.set_xticks([1.5, 10, 20, 30, 40, 46],
              ["1--2", "10", "20", "30", "40", "46"])
ax.set_yticks([0, 25, 50])
ax.set_xlabel("whitened slow-block eigendirection")
ax.set_ylabel("share of trace (%)")
ax.grid(axis="y", color=GRID, lw=0.45, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(b) Slow-block effective spectrum", loc="left",
             weight="bold", pad=4)

# (c) Fixed-support allocation identification: training seed is the unit.
ratios = np.array([1, 2, 4, 8])
exact_seed = np.array([
    [0.032759063, -0.477501452, -0.204985037, -0.112843767],
    [0.029025115, -0.270500794, -0.190884590, -0.180826515],
    [0.016798005, -0.094181113, -0.132105887, -0.143473938],
])
exact_mean = exact_seed.mean(axis=0)
ax = fig.add_subplot(grid[0, 2])
ax.axhspan(-0.55, 0, color=PANEL, zorder=0)
ax.axhline(0, color=INK, lw=0.7, zorder=1)
for values in exact_seed:
    ax.plot(ratios, values, color=SEED, marker="o", ms=2.8, lw=0.85,
            zorder=2)
ax.plot(ratios, exact_mean, color=ORANGE, marker="o", ms=4.2, lw=1.7,
        zorder=3)
ax.text(1, 0.054, "+0.026", color=MUTED, fontsize=6.0, ha="center")
for x, value in zip(ratios[1:], exact_mean[1:]):
    ax.text(x, value - 0.045, "3/3", color=ORANGE, fontsize=5.9,
            ha="center", va="top", weight="bold")
ax.text(8, exact_mean[-1] + 0.035, "mean", ha="right", va="bottom",
        color=ORANGE, fontsize=5.9, weight="bold")
ax.set_xticks(ratios,
              [r"$1\times$", r"$2\times$", r"$4\times$", r"$8\times$"])
ax.set_ylim(-0.57, 0.10)
ax.set_yticks([-0.5, -0.25, 0])
ax.set_xlabel("relative to training length")
ax.set_ylabel(r"$\Delta$ tail NLL" + "\n" + r"(EVQ-Cosh $-$ FMRoPE)")
ax.set_title("(c) Fixed-support training effect", loc="left",
             weight="bold", pad=4)
ax.grid(axis="y", color=GRID, lw=0.45)
ax.spines[["top", "right"]].set_visible(False)

assert cosh[0] == 0.0 and cosh[-1] == 1.0
assert np.all(np.diff(cosh) > 0)
assert slow.sum() == 23 and len(shares) == 46
assert np.isclose(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2),
                  2.00013315870158, atol=1e-10)
assert shares[:2].sum() > 99.99
assert np.allclose(exact_mean, [0.026194061, -0.280727786,
                                -0.175991838, -0.145714740], atol=5e-10)
assert np.all(exact_seed[:, 1:] < 0)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
