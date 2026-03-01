#!/usr/bin/env python3
"""Generate EVQ warp curve figure for NeurIPS v5 paper — v2.

Two panels:
  (a) φ_k(τ) vs u_k  — warp curves with deviation from diagonal highlighted
  (b) Δφ_k = φ_k(τ) - u_k  — directly shows high-freq bias magnitude
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

N = 64
u = np.array([(k + 0.5) / N for k in range(N)])

def evq_warp(u_arr, tau):
    if abs(tau) < 1e-8:
        return u_arr.copy()
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))

taus = [0.0, 0.4, 0.8, 1.2, 2.0]
colors = ["#999999", "#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
linestyles = ["--", "-", "-", "-", "-"]
linewidths = [2.0, 1.5, 1.5, 1.5, 1.5]
alphas = [0.7, 1.0, 1.0, 1.0, 1.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5), gridspec_kw={"wspace": 0.38})

# --- Left panel: warp curves ---
ax1.plot([0, 1], [0, 1], color="#DDDDDD", ls=":", lw=1.0, zorder=0)
for tau, color, ls, lw, alpha in zip(taus, colors, linestyles, linewidths, alphas):
    phi = evq_warp(u, tau)
    label = r"$\tau\!=\!0$ (geometric)" if tau == 0 else r"$\tau\!=\!" + str(tau) + "$"
    ax1.plot(u, phi, color=color, ls=ls, lw=lw, alpha=alpha, label=label)

ax1.set_xlabel(r"Uniform quantile $u_k$")
ax1.set_ylabel(r"Normalised position $\phi_k(\tau)$")
ax1.set_title("(a) EVQ warp")
ax1.legend(loc="upper left", framealpha=0.92, edgecolor="none", handlelength=1.8)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")

# Annotate the high-freq bias direction
ax1.annotate("", xy=(0.15, 0.35), xytext=(0.15, 0.18),
             arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.2))
ax1.text(0.19, 0.26, r"$\tau\!\uparrow$: high-freq bias", fontsize=7,
         color="#E91E63", fontstyle="italic")

# --- Right panel: deviation Δφ = φ(τ) - u ---
ax2.axhline(0, color="#DDDDDD", ls=":", lw=1.0, zorder=0)
for tau, color, ls, lw, alpha in zip(taus, colors, linestyles, linewidths, alphas):
    if tau == 0:
        continue  # Δφ = 0 by definition
    phi = evq_warp(u, tau)
    delta = phi - u
    ax2.plot(u, delta, color=color, ls=ls, lw=lw, alpha=alpha, label=r"$\tau\!=\!" + str(tau) + "$")

# shade positive region
ax2.fill_between([0, 1], 0, 0.35, color="#E8F5E9", alpha=0.3, zorder=0)
ax2.text(0.02, 0.29, "high-freq\nenriched", fontsize=6.5, color="#388E3C", alpha=0.8)

ax2.set_xlabel(r"Uniform quantile $u_k$")
ax2.set_ylabel(r"Deviation $\Delta\phi_k = \phi_k(\tau) - u_k$")
ax2.set_title(r"(b) Redistribution magnitude")
ax2.legend(loc="upper right", framealpha=0.92, edgecolor="none", handlelength=1.8)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.02, 0.35)

for ax in (ax1, ax2):
    ax.grid(True, alpha=0.12, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.savefig("/sessions/vibrant-practical-hawking/evq_warp_curves.pdf")
plt.savefig("/sessions/vibrant-practical-hawking/evq_warp_curves.png")
print("Done: evq_warp_curves.pdf / .png")
