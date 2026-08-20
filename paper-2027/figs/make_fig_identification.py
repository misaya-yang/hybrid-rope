"""Primary I figure: allocation identification under exact range control.

Data source (verbatim, no re-derivation):
  - 151.9M seed-42 raw-backed fixed-range arm:
      MATCHED_RANGE_COSH_500M_S42_20260724.md
  - 50.9M exact-range factorial, 12 configs x 3 seeds:
      M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical palette (dataviz validator: all checks PASS, light mode)
BLUE, RED, PURPLE = "#2166AC", "#D6604D", "#7F3B8F"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.32, 2.82),
                               gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 1.05})

# ---------------- Panel A: 151.9M exact-range control ----------------
lengths = ["256", "512", "1K", "2K"]
x = np.arange(len(lengths))
fixed_s42 = [+0.03276, -0.47750, -0.20499, -0.11284]

axA.axhline(0, color=INK, lw=0.7, zorder=1)
axA.axvspan(0.5, 3.5, color="#EDF3F8", zorder=0)
bars = axA.bar(x, fixed_s42, width=0.48,
               color=[MUTED, BLUE, BLUE, BLUE], edgecolor="white",
               linewidth=0.8, zorder=3)
for bar, v in zip(bars, fixed_s42):
    offset = 3 if v >= 0 else -3
    va = "bottom" if v >= 0 else "top"
    axA.annotate(f"{v:+.2f}", (bar.get_x() + bar.get_width() / 2, v),
                 textcoords="offset points", xytext=(0, offset), ha="center",
                 va=va, fontsize=6.3, color=INK)

axA.set_xticks(x); axA.set_xticklabels(lengths)
axA.set_xlabel("evaluation length", labelpad=1.5)
axA.set_ylabel(r"$\Delta$NLL (cosh $-$ uniform)", labelpad=2)
axA.annotate("lower = cosh better", xy=(0.985, 0.06), xycoords="axes fraction",
             fontsize=5.9, color=MUTED, ha="right", va="bottom")
axA.text(0, 0.27, "train window", color=MUTED, fontsize=6.0, ha="center")
axA.text(2, 0.33, "same support, OOD", color=BLUE, fontsize=6.2,
         ha="center", weight="bold")
axA.set_ylim(-0.62, 0.42)
axA.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
axA.set_title("(a)  151.9M: only the 30 interior frequencies differ",
              loc="left", pad=10)
axA.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

# ---------------- Panel B: 50.9M exact-range factorial ----------------
names = [r"cosh $0.75\times$", r"cosh $1.00\times$ (rule)", r"cosh $1.25\times$",
         "matched exponential"]
mean  = [-0.009115, -0.009879, -0.012100, -0.010619]
lo    = [-0.017780, -0.021040, -0.020828, -0.020808]
hi    = [-0.001006, +0.001540, -0.002945, -0.000696]
wins  = ["8/12", "7/12", "10/12", "9/12"]
cols  = [BLUE, PURPLE, BLUE, RED]

y = np.arange(len(names))[::-1]
axB.axvline(0, color=INK, lw=0.7, zorder=1)
for yi, m, l, h, c in zip(y, mean, lo, hi, cols):
    axB.plot([l, h], [yi, yi], color=c, lw=1.8, solid_capstyle="round", zorder=3)
    axB.scatter([m], [yi], s=22, color=c, edgecolor="white", linewidth=0.8, zorder=4)
for yi, w_ in zip(y, wins):
    axB.annotate(w_, (0.0044, yi), fontsize=6.2, color=MUTED, va="center", ha="left")

axB.set_yticks(y); axB.set_yticklabels(names)
axB.set_xlabel(r"$\Delta$ weighted OOD NLL vs. geometric" "\n"
               r"(mean, 95% configuration bootstrap)", labelpad=1.5)
axB.set_xlim(-0.0245, 0.0088)
axB.set_xticks([-0.02, -0.01, 0.0])
axB.set_title("(b)  50.9M factorial: 12 configs $\\times$ 3 seeds", loc="left", pad=12)
axB.grid(axis="x", color=GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.annotate("configs\nbeating Geo", (0.0044, y[0] + 0.78), fontsize=5.8,
             color=MUTED, va="center", ha="left", linespacing=1.1)
axB.set_ylim(y[-1] - 0.55, y[0] + 1.25)

fig.savefig("figs/fig_identification.pdf", bbox_inches="tight", pad_inches=0.02)
print("written")
