"""Build the ICLR overview from analytic schedules and verified results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


OUT = Path(__file__).with_name("fig_method_overview.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
GREEN = "#2A8C6A"
INK = "#17212B"
MUTED = "#69737D"


def evq_phi(indices: np.ndarray, pairs: int, tau: float) -> np.ndarray:
    u = (indices + 0.5) / pairs
    return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.2,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "figure.dpi": 220,
    "savefig.dpi": 300,
})

schedule_pairs, tau = 32, 4.0
k = np.arange(schedule_pairs)
uniform_z = k / (schedule_pairs - 1)
midpoint_q = evq_phi(k, schedule_pairs, tau)
cosh_z = (midpoint_q - midpoint_q[0]) / (midpoint_q[-1] - midpoint_q[0])
deployed_normalised = (
    (midpoint_q - midpoint_q[0]) / (midpoint_q[-1] - midpoint_q[0])
)
assert np.allclose(cosh_z, deployed_normalised, rtol=0.0, atol=1e-15)
assert cosh_z[0] == 0.0 and cosh_z[-1] == 1.0

fig = plt.figure(figsize=(7.25, 2.18))
grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.36, 1.15], wspace=0.38)

# (a) Support and allocation are separate coordinates.
ax = fig.add_subplot(grid[0, 0])
geo_support = 0.12 + 0.76 * uniform_z
changed_support = 0.03 + 0.94 * uniform_z
changed_allocation = 0.12 + 0.76 * cosh_z
ax.plot(k, geo_support, color=BLUE, lw=1.45, label=r"Geo: $(a,R,z_{\rm lin})$")
ax.plot(k, changed_support, color=MUTED, lw=1.15, ls="--",
        label=r"change $(a,R)$ only")
ax.plot(k, changed_allocation, color=GREEN, lw=1.65,
        label=r"change $z$ only")
ax.scatter([0, schedule_pairs - 1], geo_support[[0, -1]], s=25,
           facecolor="white", edgecolor=GREEN, lw=1.0, zorder=5)
ax.annotate("same support", xy=(schedule_pairs - 1, geo_support[-1]),
            xytext=(20.2, 0.68), color=GREEN, fontsize=6.4, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.65))
ax.text(0.5, -0.23, r"$x_k=-\log\omega_k=a+Rz_k$",
        transform=ax.transAxes, ha="center", color=INK, fontsize=7.4)
ax.set_xlim(-0.8, schedule_pairs - 0.2)
ax.set_ylim(0, 0.98)
ax.set_xlabel("frequency-pair index")
ax.set_ylabel(r"log-frequency $x_k$ (schematic)")
ax.grid(color="#E5E8EB", lw=0.45)
ax.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.6,
          borderpad=0.1, labelspacing=0.22)
ax.set_title("(a) Support vs. allocation", weight="bold", pad=5)

# (b) Anchored and deployed EVQ share exactly the same normalised shape.
ax = fig.add_subplot(grid[0, 1])
ax.plot(k, uniform_z, color=BLUE, lw=1.25, marker="o", ms=2.4,
        label="Geo allocation")
ax.plot(k, cosh_z, color=GREEN, lw=1.65, marker="o", ms=2.4,
        label="anchored Cosh")
ax.scatter(k[::3], deployed_normalised[::3], marker="x", s=18,
           color=ORANGE, lw=0.8, zorder=5, label="deployed, normalised")
ax.scatter([0, schedule_pairs - 1], cosh_z[[0, -1]], s=28,
           facecolor="white", edgecolor=GREEN, lw=1.1, zorder=5)
ax.annotate(r"same $z$ exactly", xy=(18, cosh_z[18]),
            xytext=(21.0, 0.30), color=ORANGE, fontsize=6.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
ax.set_xlim(-0.8, schedule_pairs - 0.2)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("frequency-pair index")
ax.set_ylabel(r"normalised allocation $z_k$")
ax.grid(color="#E5E8EB", lw=0.45)
ax.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.6,
          borderpad=0.1, labelspacing=0.25)
ax.set_title("(b) Same shape, controlled support", weight="bold", pad=5)

# (c) Frozen swaps expose table-by-weights co-adaptation.
ax = fig.add_subplot(grid[0, 2])
ppl = np.array([[7.14, 76.20], [23.05, 7.16]])
cmap = LinearSegmentedColormap.from_list(
    "ppl", ["#EAF4F1", "#F6E6D8", "#D35F45"]
)
ax.imshow(np.log10(ppl), cmap=cmap, vmin=np.log10(7.0), vmax=np.log10(80.0),
          aspect="equal")
for row in range(2):
    for col in range(2):
        ax.text(col, row, f"{ppl[row, col]:.2f}", ha="center", va="center",
                color=INK if ppl[row, col] < 40 else "white", fontsize=8.5,
                weight="bold")
for index in range(2):
    ax.add_patch(Rectangle((index - 0.48, index - 0.48), 0.96, 0.96,
                           fill=False, edgecolor=GREEN, lw=1.6))
ax.set_xticks([0, 1], ["Geo", "EVQ"])
ax.set_yticks([0, 1], ["Geo", "EVQ"])
ax.set_xlabel("runtime table")
ax.set_ylabel("trained weights")
ax.tick_params(length=0)
ax.text(0.5, -0.25, "self-consistent", transform=ax.transAxes, ha="center",
        color=GREEN, weight="bold")
ax.text(0.5, -0.38, "post-hoc swaps fail", transform=ax.transAxes, ha="center",
        color=ORANGE, weight="bold")
ax.set_title("(c) Co-adaptation (PPL)", weight="bold", pad=5)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
