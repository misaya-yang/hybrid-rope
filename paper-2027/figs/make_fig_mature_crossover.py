"""Visualize the verified mature-model effective-context crossovers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).with_name("fig_mature_crossover.pdf")
BLUE = "#2F6DAA"
ORANGE = "#D35F45"
INK = "#17212B"
MUTED = "#69737D"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.2,
    "axes.titlesize": 8.1,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.7,
    "ytick.labelsize": 6.7,
    "axes.linewidth": 0.6,
})


def decorate(ax, title, ylabel):
    ax.axvspan(1.5, 4.25, color="#F7EFE9", zorder=0)
    ax.axvline(1.5, color="#B9BFC5", lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color=INK, lw=0.65, zorder=1)
    ax.set_xlim(0.42, 4.18)
    ax.set_xticks([0.5, 1, 2, 4], [r"$0.5\times$", r"$1\times$",
                                      r"$2\times$", r"$4\times$"])
    ax.set_xlabel("relative to physical training cap")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E3E7EA", lw=0.5)
    ax.set_title(title, weight="bold", pad=4)
    ax.text(0.66, 0.92, "beyond cap", transform=ax.transAxes, color=ORANGE,
            fontsize=6.2, weight="bold")


fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.02))

# (a) Natural-text modeling under two distinct scale protocols.
x_olmo = np.array([0.5, 1.0, 2.0, 4.0])
olmo_rel_ppl = np.array([7.51, 3.88, -4.28, -12.64])
x_llama = np.array([1.0, 2.0, 4.0])
llama_rel_ppl = 100 * (np.array([10.068, 24.068, 127.911]) /
                       np.array([6.817, 108.958, 991.475]) - 1)
decorate(axes[0], "(a) Natural-text modeling", r"EVQ $-$ Native PPL (\%)")
axes[0].plot(x_olmo, olmo_rel_ppl, color=BLUE, marker="o", ms=3.8, lw=1.5,
             label="OLMo 1.485B · init→1K")
axes[0].plot(x_llama, llama_rel_ppl, color=ORANGE, marker="s", ms=3.8, lw=1.5,
             label="LLaMA 8B · LoRA 300")
axes[0].set_ylim(-98, 62)
axes[0].text(0.03, 0.05, "lower is better", transform=axes[0].transAxes,
             color=MUTED, fontsize=6.1)
axes[0].text(2.35, -7, "OLMo init→1K", color=BLUE, fontsize=5.9)
axes[0].text(2.25, -90, "LLaMA LoRA·300", color=ORANGE, fontsize=5.9)

# (b) Held-out multi-hop QA after matched Q/K-only adaptation.
native_2wiki = np.array([22.0, 0.0, 0.0])
evq_2wiki = np.array([21.5, 17.5, 4.0])
decorate(axes[1], "(b) OLMo 1.485B · 2Wiki", "exact match (%)")
axes[1].plot([1, 2, 4], native_2wiki, color=BLUE, marker="o", ms=4, lw=1.6,
             label="Native")
axes[1].plot([1, 2, 4], evq_2wiki, color=ORANGE, marker="s", ms=4, lw=1.6,
             label="EVQ-Cosh")
axes[1].set_ylim(-1, 25)
axes[1].annotate("17.5", (2, 17.5), xytext=(4, 5), textcoords="offset points",
                 color=ORANGE, fontsize=6.7, weight="bold")
axes[1].text(3.05, 4.6, "EVQ-Cosh", color=ORANGE, fontsize=5.9)
axes[1].text(3.25, 0.9, "Native", color=BLUE, fontsize=5.9)

# (c) Official RULER macro differences under separate task-adaptation runs.
olmo_ruler_delta = np.array([-29.75, 29.61, 4.65])
llama_ruler_delta = np.array([77.60 - 94.44, 14.03 - 0.295])
decorate(axes[2], "(c) Task-adapted RULER", r"EVQ $-$ Native (pp)")
axes[2].plot([1, 2, 4], olmo_ruler_delta, color=BLUE, marker="o", ms=4, lw=1.6,
             label="OLMo Q/K · 300")
axes[2].plot([1, 2], llama_ruler_delta, color=ORANGE, marker="s", ms=4, lw=1.6,
             label="LLaMA Q/K/V/O · 516")
axes[2].set_ylim(-38, 38)
axes[2].text(0.03, 0.05, "positive = EVQ better", transform=axes[2].transAxes,
             color=MUTED, fontsize=6.1)
axes[2].text(2.12, 30.5, "OLMo Q/K·300", color=BLUE, fontsize=5.9)
axes[2].text(2.12, 14.5, "LLaMA Q/K/V/O·516", color=ORANGE, fontsize=5.8)

fig.tight_layout(w_pad=1.0)

assert np.isclose(llama_rel_ppl[1], -77.91, atol=0.02)
assert np.allclose(evq_2wiki, [21.5, 17.5, 4.0])
assert np.allclose(olmo_ruler_delta, [-29.75, 29.61, 4.65])

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
