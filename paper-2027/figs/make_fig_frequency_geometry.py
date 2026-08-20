"""Detailed frequency allocation and full-subspace redundancy visualization."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.full_rope_collision_audit import gram_blocks, whitened_blocks


OUT = Path(__file__).with_name("fig_frequency_geometry.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
GREEN = "#2A8C6A"
INK = "#17212B"


def midpoint_cosh(pairs: int, tau: float) -> np.ndarray:
    u = (np.arange(pairs) + 0.5) / pairs
    return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.5,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45),
                         gridspec_kw={"width_ratios": [1.18, 1.0]})

# (a) Exact-range identification grid versus the deployed grid.
pairs, tau = 32, 4.0
k = np.arange(pairs)
geo = k / pairs
deployed = midpoint_cosh(pairs, tau)
anchored = ((pairs - 1) / pairs) * (
    (deployed - deployed[0]) / (deployed[-1] - deployed[0])
)

ax = axes[0]
ax.plot(k, geo, color=BLUE, marker="o", ms=2.8, lw=1.3, label="standard Geo")
ax.plot(k, anchored, color=GREEN, marker="o", ms=2.8, lw=1.5,
        label="endpoint-normalised Cosh")
ax.plot(k, deployed, color=ORANGE, marker="o", ms=2.5, lw=1.2, ls="--",
        label="deployed midpoint EVQ-Cosh")
for index in [0, pairs - 1]:
    ax.scatter(index, geo[index], s=34, facecolor="white", edgecolor=GREEN,
               lw=1.2, zorder=5)
ax.annotate("identical sampled endpoints", (pairs - 1, geo[-1]),
            xytext=(17, 0.70), color=GREEN, fontsize=6.8,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
ax.set_xlim(-0.8, pairs - 0.2)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("frequency-pair index")
ax.set_ylabel(r"$\phi_k=-\log(\omega_k)/\log b$")
ax.set_title("(a) The allocation axis is not the range axis", weight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=6.7, handlelength=1.6)
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
image = ax.imshow(affinity, origin="lower", cmap="magma", vmin=0, vmax=1,
                  interpolation="nearest")
ax.add_patch(Rectangle((40.5, 40.5), 23, 23, fill=False, edgecolor="#39D0B1",
                       lw=1.4))
ax.annotate("23 slow pairs\nshare ~2 dimensions", (51.5, 51.5),
            xytext=(10, 53), color="white", fontsize=7.0, weight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
ax.set_xlabel("frequency-pair index")
ax.set_ylabel("frequency-pair index")
ax.set_title("(b) Phase-invariant subspace redundancy", weight="bold")
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
colorbar.set_label(r"canonical redundancy $c_{ij}$", fontsize=7)
colorbar.ax.tick_params(labelsize=6.5)

fig.tight_layout(w_pad=1.2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
