"""
Figure 1 for NeurIPS paper: EVQ-Cosh Frequency Allocation
3-panel: (a) Frequency allocation  (b) PPL Waterbed  (c) Retrieval Crossover

Usage: python scripts/fig1_neurips.py
Output:
  - paper_exports/fig1_neurips.pdf/.png
  - docs/paperdraft/figs/fig1_frequency_dynamics.pdf/.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# ── NeurIPS style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

# Colors
C_GEO = "#2166ac"       # blue for Geometric
C_HYB = "#d6604d"       # red for Hybrid/EVQ
C_COLLISION = "#e0e0e0"  # gray for collision zone
C_GEO_LIGHT = "#92c5de"
C_HYB_LIGHT = "#f4a582"

# ── Data ───────────────────────────────────────────────────────────

# Panel (a): Frequency allocation
d_head = 64
K = d_head // 2  # 32 channels
base = 500_000
tau = 1.5
L_train = 2048

u_k = np.linspace(0.5 / K, 1 - 0.5 / K, K)  # midpoint quantiles

# Geometric: uniform in [0, 1]
phi_geo = u_k.copy()

# EVQ-cosh
phi_evq = 1 - (1 / tau) * np.arcsinh((1 - u_k) * np.sinh(tau))

# Collision boundary: φ > c means λ > L_train
c_boundary = np.log(L_train) / np.log(base)  # ≈ 0.581

# Panel (b): PPL training dynamics (Phase9F, 750M, L=2048)
ckpt_tokens = np.array([250, 500, 750, 1000])  # millions
ckpt_pct = ["25%", "50%", "75%", "100%"]

geo_ppl_2k = np.array([35.195, 26.995, 23.523, 21.980])
geo_ppl_8k = np.array([97.774, 99.432, 108.804, 115.010])
hyb_ppl_2k = np.array([35.212, 26.780, 23.058, 21.648])
hyb_ppl_8k = np.array([104.051, 104.356, 115.214, 121.583])
hyb_tokens = np.array([250, 500, 750, 1000])

# Panel (c): Passkey L=8K retrieval (20-trial checkpoint eval, consistent)
geo_ret_8k = np.array([55, 70, 60, 60])
hyb_ret_8k = np.array([45, 65, 75, 80])

# ── Figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.0, 2.4),
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
)

# ═══════════════════════════════════════════════════════════════════
# Panel (a): Frequency Allocation Stem Plot
# ═══════════════════════════════════════════════════════════════════
ax = axes[0]

# Collision zone background
ax.axhspan(c_boundary, 1.05, color=C_COLLISION, alpha=0.45, zorder=0)
ax.text(K // 2, (c_boundary + 1.05) / 2, "collision zone\n($\\lambda_k > L_{\\mathrm{train}}$)",
        fontsize=6.5, color="#555555", ha="center", va="center",
        fontweight="bold", style="italic")
ax.axhline(c_boundary, color="#888888", linewidth=0.7, linestyle="--", zorder=1)

# Stem plots
offset = 0.25  # horizontal offset between geo and evq stems
for k_idx in range(K):
    # Geometric stems
    ax.plot([k_idx - offset, k_idx - offset], [0, phi_geo[k_idx]],
            color=C_GEO, linewidth=0.6, zorder=2)
    ax.plot(k_idx - offset, phi_geo[k_idx], "o", color=C_GEO,
            markersize=2.5, zorder=3)
    # EVQ stems
    ax.plot([k_idx + offset, k_idx + offset], [0, phi_evq[k_idx]],
            color=C_HYB, linewidth=0.6, zorder=2)
    ax.plot(k_idx + offset, phi_evq[k_idx], "s", color=C_HYB,
            markersize=2.0, zorder=3)

# Spacing annotations
# Low-freq end (high k): spacing expands
k_annot = 27
ax.annotate("",
            xy=(k_annot + offset, phi_evq[k_annot]),
            xytext=(k_annot + offset, phi_evq[k_annot - 1]),
            arrowprops=dict(arrowstyle="<->", color=C_HYB, lw=0.8))
ax.text(k_annot + offset + 1.5, (phi_evq[k_annot] + phi_evq[k_annot - 1]) / 2,
        "expand", fontsize=5.5, color=C_HYB, ha="left", va="center")

# High-freq end (low k): spacing compresses — use slightly higher k to avoid crowding
k_annot2 = 6
ax.annotate("",
            xy=(k_annot2 + offset, phi_evq[k_annot2]),
            xytext=(k_annot2 + offset, phi_evq[k_annot2 - 1]),
            arrowprops=dict(arrowstyle="<->", color=C_HYB, lw=0.8))
ax.text(k_annot2 + offset + 1.8, (phi_evq[k_annot2] + phi_evq[k_annot2 - 1]) / 2,
        "compress", fontsize=5.5, color=C_HYB, ha="left", va="center")

# Labels and formatting
ax.set_xlabel("Channel index $k$")
ax.set_ylabel("Log-frequency $\\varphi_k$")
ax.set_xlim(-1.5, K + 0.5)
ax.set_ylim(-0.02, 1.05)
ax.set_xticks([0, 8, 16, 24, 31])

# Legend
ax.plot([], [], "o", color=C_GEO, markersize=3, label="Geometric ($\\tau$=0)")
ax.plot([], [], "s", color=C_HYB, markersize=3, label="EVQ-Cosh ($\\tau$=1.5)")
ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
          borderpad=0.3, handletextpad=0.3)

ax.set_title("(a) Frequency allocation", fontweight="bold", pad=6)

# ═══════════════════════════════════════════════════════════════════
# Panel (b): PPL Waterbed Training Dynamics
# Narrative: Geo L=8K gets WORSE with training, Hybrid doesn't
# ═══════════════════════════════════════════════════════════════════
ax = axes[1]

# --- L=2K lines: thin, muted (context, not focus) ---
ax.plot(ckpt_tokens, geo_ppl_2k, "-o", color=C_GEO_LIGHT, markersize=2.5,
        label="Geo $L$=2K", zorder=2, linewidth=0.8, alpha=0.7)
ax.plot(hyb_tokens, hyb_ppl_2k, "-o", color=C_HYB_LIGHT, markersize=2.5,
        label="Hybrid $L$=2K", zorder=2, linewidth=0.8, alpha=0.7)

# --- L=8K lines: BOLD, the main story ---
ax.plot(ckpt_tokens, geo_ppl_8k, "-^", color=C_GEO, markersize=5,
        label="Geo $L$=8K", zorder=4, linewidth=2.2)
ax.plot(hyb_tokens, hyb_ppl_8k, "-s", color=C_HYB, markersize=5,
        label="Hybrid $L$=8K", zorder=4, linewidth=2.2)

# Shade the Geo regression zone (50% → 100%)
ax.fill_between(ckpt_tokens[1:], geo_ppl_8k[1:],
                [geo_ppl_8k[1]] * 3,  # flat baseline at 50% level
                color=C_GEO, alpha=0.08, zorder=1)

# Annotation: Geo regression after 50%
ax.annotate("Geo: long-context\nregression after 50%",
            xy=(875, 112), xytext=(550, 120),
            fontsize=6, color=C_GEO, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=1.0,
                            mutation_scale=8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f0fa",
                      edgecolor=C_GEO, alpha=0.95, linewidth=0.6))

# Annotation: Hybrid also rises (waterbed too, but retrieval unharmed)
ax.annotate("Hybrid: also rises\n(but retrieval $\\uparrow$!)",
            xy=(875, 118.4), xytext=(400, 75),
            fontsize=5.5, color=C_HYB, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=C_HYB, lw=0.8,
                            mutation_scale=7),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fce8e4",
                      edgecolor=C_HYB, alpha=0.95, linewidth=0.6))

# Small "+17.6%" label at Geo endpoint
ax.text(1010, 115.0, "+17.6%", fontsize=5.5, color=C_GEO,
        ha="left", va="center", fontweight="bold")
# Small "+16.8%" label at Hybrid endpoint
ax.text(1010, 121.6, "+16.8%", fontsize=5.5, color=C_HYB,
        ha="left", va="center", fontweight="bold")

# "both improve" label for L=2K
ax.text(800, 27, "$L$=2K: both$\\downarrow$", fontsize=5.5,
        color="#888888", ha="center", va="center", style="italic")

ax.set_xlabel("Training tokens (M)")
ax.set_ylabel("Perplexity (at $L$=8K)")
ax.set_xticks(ckpt_tokens)
ax.set_xticklabels(["250M", "500M", "750M", "1B"])
ax.set_ylim(15, 130)
ax.legend(loc="center left", framealpha=0.9, edgecolor="none",
          borderpad=0.3, handletextpad=0.3, fontsize=6,
          ncol=1)

ax.set_title("(b) PPL training dynamics", fontweight="bold", pad=6)

# ═══════════════════════════════════════════════════════════════════
# Panel (c): Passkey L=8K Retrieval Crossover
# ═══════════════════════════════════════════════════════════════════
ax = axes[2]

ax.plot(ckpt_tokens, geo_ret_8k, "-o", color=C_GEO, markersize=5,
        label="Geometric", zorder=3, linewidth=2.0)
ax.plot(hyb_tokens, hyb_ret_8k, "-s", color=C_HYB, markersize=5,
        label="Hybrid (EVQ-Cosh)", zorder=3, linewidth=2.0)

# Shade the gap at 100% to emphasize the +20pp
ax.fill_betweenx([60, 80], 980, 1020, color=C_HYB, alpha=0.10, zorder=1)
ax.annotate("+20pp",
            xy=(1000, 70), xytext=(1000, 70),
            fontsize=7, color="#333333", ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffcc",
                      edgecolor="#999999", alpha=0.9, linewidth=0.5))

# Geo regression annotation
ax.annotate("Geo: regresses",
            xy=(750, 60), xytext=(600, 47),
            fontsize=6, color=C_GEO, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_GEO, lw=0.8,
                            mutation_scale=7),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f0fa",
                      edgecolor=C_GEO, alpha=0.9, linewidth=0.5))

# Hybrid monotonic improvement annotation
ax.annotate("Hybrid:\nmonotonic $\\uparrow$",
            xy=(875, 77.5), xytext=(700, 87),
            fontsize=6, color=C_HYB, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_HYB, lw=0.8,
                            mutation_scale=7),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce8e4",
                      edgecolor=C_HYB, alpha=0.9, linewidth=0.5))

ax.set_xlabel("Training tokens (M)")
ax.set_ylabel("Passkey retrieval at $L$=8K (%)")
ax.set_xticks(ckpt_tokens)
ax.set_xticklabels(["250M", "500M", "750M", "1B"])
ax.set_ylim(35, 95)
ax.set_yticks([40, 50, 60, 70, 80, 90])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
ax.legend(loc="lower left", framealpha=0.9, edgecolor="none",
          borderpad=0.3, handletextpad=0.3)

ax.set_title("(c) Passkey retrieval ($L$=8K)", fontweight="bold", pad=6)

# ── Save ───────────────────────────────────────────────────────────
import os

legacy_dir = "paper_exports"
paper_fig_dir = "docs/paperdraft/figs"
os.makedirs(legacy_dir, exist_ok=True)
os.makedirs(paper_fig_dir, exist_ok=True)

legacy_pdf = os.path.join(legacy_dir, "fig1_neurips.pdf")
legacy_png = os.path.join(legacy_dir, "fig1_neurips.png")
paper_pdf = os.path.join(paper_fig_dir, "fig1_frequency_dynamics.pdf")
paper_png = os.path.join(paper_fig_dir, "fig1_frequency_dynamics.png")

for path in [legacy_pdf, legacy_png, paper_pdf, paper_png]:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)

print("Saved:")
print(f"  {legacy_pdf}")
print(f"  {legacy_png}")
print(f"  {paper_pdf}")
print(f"  {paper_png}")
print(f"\nFrequency data check:")
print(f"  Collision boundary c = {c_boundary:.3f}")
print(f"  Geo spacing (low-freq, k=28-29): {phi_geo[29]-phi_geo[28]:.4f}")
print(f"  EVQ spacing (low-freq, k=28-29): {phi_evq[29]-phi_evq[28]:.4f}")
print(f"  Ratio: {(phi_evq[29]-phi_evq[28])/(phi_geo[29]-phi_geo[28]):.2f}x")
print(f"  Geo spacing (high-freq, k=2-3):  {phi_geo[3]-phi_geo[2]:.4f}")
print(f"  EVQ spacing (high-freq, k=2-3):  {phi_evq[3]-phi_evq[2]:.4f}")
print(f"  Ratio: {(phi_evq[3]-phi_evq[2])/(phi_geo[3]-phi_geo[2]):.2f}x")
