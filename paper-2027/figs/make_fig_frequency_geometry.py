"""Detailed frequency allocation and full-subspace redundancy visualization."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.full_rope_collision_audit import gram_blocks, whitened_blocks


OUT = Path(__file__).with_name("fig_frequency_geometry.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
GREEN = "#2A8C6A"
INK = "#17212B"
REDUNDANCY_CMAP = LinearSegmentedColormap.from_list(
    "redundancy_blue",
    ["#FFFFFF", "#EDF5F9", "#D4E8F2", "#A8CDE1", "#5E96BD"],
)
REDUNDANCY_CMAP.set_bad("#FFFFFF")


def midpoint_cosh(pairs: int, tau: float) -> np.ndarray:
    u = (np.arange(pairs) + 0.5) / pairs
    return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau


assert np.count_nonzero(midpoint_cosh(64, 4.0) <= 0.5) == 55


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15),
                         gridspec_kw={"width_ratios": [1.10, 1.0]})

# (a) Exact-range identification grid versus the deployed grid.
pairs, tau = 32, 4.0
k = np.arange(pairs)
geo = k / pairs
deployed = midpoint_cosh(pairs, tau)
anchored = ((pairs - 1) / pairs) * (
    (deployed - deployed[0]) / (deployed[-1] - deployed[0])
)

ax = axes[0]
ax.plot(k, geo, color=BLUE, marker="o", ms=2.8, lw=1.3,
        label="FMRoPE")
ax.plot(k, anchored, color=GREEN, marker="o", ms=2.8, lw=1.5,
        label="anchored EVQ-Cosh")
ax.plot(k, deployed, color=ORANGE, marker="o", ms=2.5, lw=1.2, ls="--",
        label="deployed midpoint EVQ-Cosh")
for index in [0, pairs - 1]:
    ax.scatter(index, geo[index], s=34, facecolor="white", edgecolor=GREEN,
               lw=1.2, zorder=5)
ax.annotate("identical sampled endpoints", (pairs - 1, geo[-1]),
            xytext=(12, 0.70), color=GREEN, fontsize=8.4,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
ax.set_xlim(-0.8, pairs - 0.2)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("frequency-pair index")
ax.set_ylabel(r"$\phi_k=-\log(\omega_k)/\log b$")
ax.set_title("(a) Fixed endpoints, different allocation", weight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=8.4, handlelength=1.6)
ax.grid(color="#E4E8EB", lw=0.5)

# (b) Canonical full-subspace redundancy on the standard RoPE grid.
base, length, pairs = 500_000.0, 4096, 64
phi = np.arange(pairs) / pairs
omega = base ** (-phi)
white = whitened_blocks(gram_blocks(omega, length))
affinity = 0.5 * np.sum(white**2, axis=(2, 3))
np.fill_diagonal(affinity, np.nan)
slow = omega * length <= 1
slow_indices = np.flatnonzero(slow)
assert len(slow_indices) == 23 and slow_indices[0] == 41

ax = axes[1]
affinity_plot = affinity.copy()
affinity_plot[np.tril_indices(pairs)] = np.nan
image = ax.imshow(affinity_plot, origin="lower", cmap=REDUNDANCY_CMAP, vmin=0, vmax=1,
                  interpolation="nearest")
ax.add_patch(Rectangle((40.5, 40.5), 23, 23, fill=False, edgecolor=ORANGE,
                       lw=1.4))
ax.annotate("23 slow pairs\n46 dims, $r_2=2.00$", (51.5, 51.5),
            xytext=(8, 53), color=INK, fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
ax.set_xlabel("frequency-pair index")
ax.set_ylabel("frequency-pair index")
ax.set_title("(b) Phase-invariant redundancy", weight="bold")
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
colorbar.set_label(r"canonical redundancy $c_{ij}$", fontsize=8)
colorbar.ax.tick_params(labelsize=8.2)

fig.tight_layout(w_pad=1.2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
