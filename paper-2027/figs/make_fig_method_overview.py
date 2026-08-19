"""Build the ICML method overview from analytic schedules and verified headline results."""

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
L, M = 4096, 32768
k = np.arange(K)
phi_geo = k / K
phi_evq = evq_phi(k, K, tau)
phi_L = np.log(L / (2 * np.pi)) / np.log(base)
phi_M = np.log(M / (2 * np.pi)) / np.log(base)


def counts(phi: np.ndarray) -> tuple[int, int, int]:
    wavelength = 2 * np.pi * base**phi
    return (
        int(np.sum(wavelength <= L)),
        int(np.sum((wavelength > L) & (wavelength <= M))),
        int(np.sum(wavelength > M)),
    )


assert counts(phi_geo) == (32, 10, 22)
assert counts(phi_evq) == (43, 8, 13)
assert np.all(np.diff(phi_evq) > 0)

fig = plt.figure(figsize=(7.25, 2.12))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.35], wspace=0.36)

# (a) The actual finite-channel budget in a mature configuration.
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(0, phi_L, color="#EAF3FA", zorder=0)
ax.axvspan(phi_L, phi_M, color="#FFF2D9", zorder=0)
ax.axvspan(phi_M, 1, color="#F1EDF7", zorder=0)
ax.axvline(phi_L, color="#A6ADB4", lw=0.65, ls="--")
ax.axvline(phi_M, color="#A6ADB4", lw=0.65, ls="--")
ax.scatter(phi_geo, np.full(K, 0.67), s=7.5, color=BLUE, edgecolor="white", lw=0.22, zorder=3)
ax.scatter(phi_evq, np.full(K, 0.28), s=8.0, marker="s", color=ORANGE,
           edgecolor="white", lw=0.22, zorder=3)
ax.text(-0.025, 0.67, "Geo", ha="right", va="center", color=BLUE, weight="bold")
ax.text(-0.025, 0.28, "EVQ", ha="right", va="center", color=ORANGE, weight="bold")
centers = [phi_L / 2, (phi_L + phi_M) / 2, (phi_M + 1) / 2]
headers = ["completed cycle", "transition", "slow at 32K"]
for x, header in zip(centers, headers):
    ax.text(x, 0.98, header, ha="center", va="top", fontsize=6.3, color=INK)
for x, g, e in zip(centers, counts(phi_geo), counts(phi_evq)):
    ax.text(x, 0.04, f"{g} → {e}", ha="center", va="bottom", fontsize=7,
            color=INK, weight="bold")
ax.text(0.5, -0.17, "same 64 channels; different allocation",
        transform=ax.transAxes, ha="center", color=MUTED, style="italic")
ax.set_xlim(-0.08, 1.01)
ax.set_ylim(0, 1.03)
ax.set_yticks([])
ax.set_xticks([0, phi_L, phi_M, 1])
ax.set_xticklabels(["fast", r"$\lambda=L$", r"$\lambda=M$", "slow"])
ax.tick_params(axis="x", length=2.5, pad=2)
ax.spines[["left", "right", "top"]].set_visible(False)
ax.set_title("(a) A finite spectral budget", weight="bold", pad=5)

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

# (c) The two controls that isolate the scientific claim.
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
        "MATCH PHASE EXPOSURE AT 1.485B\n"
        "same Q/K adaptation and 4K/8K/16K phases\n"
        "8K RULER: 2.02% → 31.63%",
        ha="center", va="center", color=INK, bbox=box, linespacing=1.28)
ax.text(0.5, 0.02, "allocation remains binding under both controls",
        ha="center", va="bottom", color=ORANGE, weight="bold")
ax.set_title("(c) Two controls isolate allocation", weight="bold", pad=5)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
