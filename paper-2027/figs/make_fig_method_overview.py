"""Build the ICLR method overview from analytic schedules and verified headline results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("fig_method_overview.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
INK = "#17212B"
MUTED = "#69737D"


def evq_phi(k: np.ndarray, K: int, tau: float) -> np.ndarray:
    u = (k + 0.5) / K
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

base, K, tau = 500_000.0, 64, 2.0
L = 4096
k = np.arange(K)
phi_geo = k / (K - 1)
phi_evq = evq_phi(k, K, tau)
phi_slow = np.log(L) / np.log(base)
slow = base ** (-phi_geo) * L <= 1

assert int(np.sum(slow)) == 24
assert np.all(np.diff(phi_evq) > 0)

fig = plt.figure(figsize=(7.25, 2.12))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.35], wspace=0.36)

# (a) Low-frequency collapse in a standard finite table.
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(0, phi_slow, color="#EAF3FA", zorder=0)
ax.axvspan(phi_slow, 1, color="#FBE9E5", zorder=0)
ax.axvline(phi_slow, color="#A6ADB4", lw=0.7, ls="--")
ax.scatter(phi_geo[~slow], np.full(np.sum(~slow), 0.62), s=8.0,
           color=BLUE, edgecolor="white", lw=0.22, zorder=3)
ax.scatter(phi_geo[slow], np.full(np.sum(slow), 0.62), s=8.0,
           color=ORANGE, edgecolor="white", lw=0.22, zorder=3)
ax.text(phi_slow / 2, 0.93, "40 faster pairs", ha="center", va="top",
        fontsize=7, color=BLUE, weight="bold")
ax.text((phi_slow + 1) / 2, 0.93, "24 slow pairs", ha="center", va="top",
        fontsize=7, color=ORANGE, weight="bold")
ax.text((phi_slow + 1) / 2, 0.28,
        "48 nominal dimensions\n"
        r"$\longrightarrow\ r_2=2.00$",
        ha="center", va="center", fontsize=7.2, color=INK,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="#D8A79B", lw=0.7))
ax.text(0.5, -0.17, r"slow means $\omega L\leq1$; redundant, not unused",
        transform=ax.transAxes, ha="center", color=MUTED, style="italic")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1.03)
ax.set_yticks([])
ax.set_xticks([0, phi_slow, 1])
ax.set_xticklabels(["fast", r"$\omega L=1$", "slow"])
ax.tick_params(axis="x", length=2.5, pad=2)
ax.spines[["left", "right", "top"]].set_visible(False)
ax.set_title("(a) Static low-frequency collapse", weight="bold", pad=5)

# (b) Closed-form inverse-CDF construction.
ax = fig.add_subplot(gs[0, 1])
u = np.linspace(0, 1, 300)
phi = 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
ax.plot(u, u, color="#9AA1A8", lw=1.1, ls="--", label="geometric")
ax.plot(u, phi, color=ORANGE, lw=2.0, label="EVQ-Cosh")
ax.scatter((k + 0.5) / K, phi_evq, s=4, color=ORANGE, alpha=0.72)
ax.text(0.50, 0.16, "inverse-CDF\nquantiles", color=ORANGE,
        ha="center", va="center", fontsize=6.7)
ax.text(0.5, -0.30,
        r"$\phi_k=1-\tau^{-1}\operatorname{asinh}((1-u_k)\sinh\tau)$",
        transform=ax.transAxes, ha="center", color=INK, fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.23", facecolor="#F4F8FB", edgecolor="#B9D4E8", lw=0.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"quantile $u$")
ax.set_ylabel(r"log-frequency position $\phi$")
ax.legend(frameon=False, loc="upper left", handlelength=1.7, borderpad=0.1)
ax.grid(color="#E5E8EB", lw=0.5)
ax.set_title("(b) Closed form", weight="bold", pad=5)

# (c) The controls that connect geometry to trained behaviour.
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
box = dict(boxstyle="round,pad=0.42", facecolor="#F7F9FB", edgecolor="#C7D0D8", lw=0.75)
ax.text(0.5, 0.77,
        "FIX RANGE\n"
        "same endpoints, span, init, data, budget\n"
        "move only 30 interior frequencies\n"
        r"$\Delta$OOD NLL = $-0.478/-0.205/-0.113$",
        ha="center", va="center", color=INK, bbox=box, linespacing=1.25)
ax.annotate("", xy=(0.5, 0.49), xytext=(0.5, 0.57),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.35))
ax.text(0.5, 0.26,
        "CROSS WEIGHTS × RUNTIME TABLE\n"
        "self-consistent PPL: 7.14 / 7.16\n"
        "post-hoc swaps: 76.20 / 23.05",
        ha="center", va="center", color=INK, bbox=box, linespacing=1.28)
ax.text(0.5, 0.02, "allocation is real; trained use is table-conditioned",
        ha="center", va="bottom", color=ORANGE, weight="bold")
ax.set_title("(c) Identification and co-adaptation", weight="bold", pad=5)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
