"""Primary I figure: allocation identification under exact range control.

Data source (verbatim, no re-derivation):
  - 151.9M seed-42 raw-backed arm (fixed + target-matched):
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
lengths = ["512", "1K", "2K"]
x = np.arange(len(lengths))
fixed_s42  = [-0.47750, -0.20499, -0.11284]       # seed 42, fixed matched range
target_s42 = [+0.06109, +0.18182, +0.27857]       # seed 42, target-matched range

w = 0.30
axA.axhline(0, color=INK, lw=0.7, zorder=1)
axA.bar(x - w/2, fixed_s42, width=w, color=BLUE, edgecolor="white", linewidth=0.8,
        label="Fixed matched range (seed 42)", zorder=3)
axA.bar(x + w/2, target_s42, width=w, color=RED, edgecolor="white", linewidth=0.8,
        label="Target-retargeted (seed 42)", zorder=3)
for xi, v in zip(x - w/2, fixed_s42):
    axA.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(-13, -3),
                 ha="right", fontsize=6.2, color=INK)
for xi, v in zip(x + w/2, target_s42):
    axA.annotate(f"+{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=6.2, color=INK)

axA.set_xticks(x); axA.set_xticklabels(lengths)
axA.set_xlabel("evaluation length", labelpad=1.5)
axA.set_ylabel(r"$\Delta$NLL (cosh $-$ uniform)", labelpad=2)
axA.annotate("lower = cosh better", xy=(0.985, 0.06), xycoords="axes fraction",
             fontsize=5.9, color=MUTED, ha="right", va="bottom")
axA.set_ylim(-0.62, 0.42)
axA.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
axA.set_title("(a)  151.9M: only the 30 interior frequencies differ",
              loc="left", pad=26)
axA.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

axA.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.02), frameon=False,
           ncol=1, handlelength=1.0, handletextpad=0.5, borderaxespad=0,
           labelspacing=0.22)

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
