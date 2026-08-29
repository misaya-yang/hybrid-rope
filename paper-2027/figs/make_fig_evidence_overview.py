"""Build Figure 1 from owner-backed analytic and experimental quantities.

The panels expose one causal decomposition:
  (a) a finite table is specified by support and interior allocation, with one
      small representative slow-block spectrum illustrating finite-budget
      redundancy;
  (b) the fixed-support allocation effect and the target-aware support-
      retargeted reversal are shown side by side rather than as a method race;
  (c) frozen OLMo/Qwen controls show that changing only interior ``z`` can have
      a large zero-training behavioural effect.

Sources:
  - analytic EVQ-Cosh inverse CDF in the manuscript
  - FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md
  - EXACT_RANGE_151M_3SEED_RESULT_20260820.json
  - data/curated/frozen_fixed_support_mature_20260823.json
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.full_rope_collision_audit import gram_blocks, whitened_blocks


OUT = Path(__file__).with_name("fig_evidence_overview.pdf")

# Okabe--Ito palette plus neutral greys. Line styles and markers redundantly
# encode every comparison so the figure remains legible in grayscale.
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
INK = "#17212B"
MUTED = "#69737D"
SEED = "#A8B0B7"
REFERENCE = ["#B5BDC4", "#737E87"]
GRID = "#E2E6E9"
FIXED_BG = "#FBEEE8"
TARGET_BG = "#EAF4FA"
PURE_Z_BG = "#F1F6F8"


def evq_phi(indices: np.ndarray, pairs: int, tau: float) -> np.ndarray:
    """Midpoint inverse-CDF allocation, normalized to fixed endpoints."""
    u = (indices + 0.5) / pairs
    raw = 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
    return (raw - raw[0]) / (raw[-1] - raw[0])


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.1,
    "ytick.labelsize": 6.1,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.65,
})


# ---------------------------------------------------------------------------
# Owner-backed quantities and invariants.
# ---------------------------------------------------------------------------

# Panel (a): the finite-table coordinate and one representative geometry case.
pairs = 32
k = np.arange(pairs)
uniform_z = k / (pairs - 1)
cosh_z = evq_phi(k, pairs, tau=4.0)

geometry_pairs = 64
geometry_length = 4096.0
geometry_base = 500_000.0
phi = np.arange(geometry_pairs) / geometry_pairs
omega = np.power(geometry_base, -phi)
slow = omega * geometry_length <= 1
white = whitened_blocks(gram_blocks(omega, geometry_length))
slow_white = white[np.ix_(slow, slow)]
slow_gram = slow_white.transpose(0, 2, 1, 3).reshape(
    2 * slow.sum(), 2 * slow.sum()
)
eigenvalues = np.linalg.eigvalsh(0.5 * (slow_gram + slow_gram.T))[::-1]
trace_shares = 100.0 * np.clip(eigenvalues, 0.0, None) / eigenvalues.sum()
effective_rank = eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2)

# Panel (b): paired training-seed contrasts. Rows are seeds 42, 137, and 256;
# columns are 1x/2x/4x/8x the training length. Negative values favour the
# anchored allocation. Both conditions share the 1x column by construction.
ratios = np.array([1, 2, 4, 8])
fixed_seed = np.array([
    [0.032759063, -0.477501452, -0.204985037, -0.112843767],
    [0.029025115, -0.270500794, -0.190884590, -0.180826515],
    [0.016798005, -0.094181113, -0.132105887, -0.143473938],
])
target_seed = np.array([
    [0.032759063, 0.061090410, 0.181823365, 0.278573841],
    [0.029025115, 0.052570835, 0.155599415, 0.399244130],
    [0.016798005, 0.067289382, 0.344166435, 0.700963058],
])
fixed_mean = fixed_seed.mean(axis=0)
target_mean = target_seed.mean(axis=0)

# Panel (c): percentages from the frozen mature-checkpoint receipt. Reference
# rows have their own support/transport; the lower arrays share support,
# amplitude, checkpoint, rows, decoder, precision, and hardware.
pure_z_labels = [r"Uniform $z$", r"Coarse $z$", r"Derived $z$"]
olmo_reference = np.array([0.0, 7.94])
qwen_reference = np.array([54.50, 60.25])
olmo_pure_z = np.array([0.5555555556, 61.0370370370, 60.4722222222])
qwen_pure_z = np.array([57.75, 64.00, 66.50])
olmo_derived_minus_uniform_ci = np.array([54.8796296296, 64.7962962963])
qwen_derived_minus_uniform_ci = np.array([0.25, 17.50])

assert np.isclose(cosh_z[0], 0.0) and np.isclose(cosh_z[-1], 1.0)
assert np.all(np.diff(cosh_z) > 0)
assert slow.sum() == 23 and len(trace_shares) == 46
assert np.isclose(effective_rank, 2.00013315870158, atol=1e-10)
assert trace_shares[:2].sum() > 99.99
assert np.allclose(
    fixed_mean,
    [0.026194061, -0.280727786, -0.175991838, -0.145714740],
    atol=5e-10,
)
assert np.allclose(
    target_mean,
    [0.026194061, 0.060316876, 0.227196405, 0.459593676],
    atol=5e-10,
)
assert np.all(fixed_seed[:, 1:] < 0)
assert np.all(target_seed[:, 1:] > 0)
assert np.allclose(olmo_reference, [0.0, 7.94])
assert np.allclose(qwen_reference, [54.50, 60.25])
assert np.allclose(olmo_pure_z, [0.5555555556, 61.0370370370, 60.4722222222])
assert np.allclose(qwen_pure_z, [57.75, 64.00, 66.50])
assert np.allclose(olmo_derived_minus_uniform_ci, [54.8796296296, 64.7962962963])
assert np.allclose(qwen_derived_minus_uniform_ci, [0.25, 17.50])


fig = plt.figure(figsize=(7.25, 2.95))
outer = fig.add_gridspec(
    1, 3, width_ratios=[1.08, 1.35, 1.45], wspace=0.30
)


# ---------------------------------------------------------------------------
# (a) Support, allocation, and one representative finite-budget instance.
# ---------------------------------------------------------------------------
a_grid = outer[0, 0].subgridspec(2, 1, height_ratios=[1.34, 0.70], hspace=0.13)
ax_alloc = fig.add_subplot(a_grid[0, 0])

row_uniform = 0.65
row_cosh = 0.31
ax_alloc.hlines([row_uniform, row_cosh], 0, 1, color=INK, lw=0.55, zorder=1)
for index in [3, 7, 11, 15, 19, 23, 27]:
    ax_alloc.plot(
        [uniform_z[index], cosh_z[index]],
        [row_uniform - 0.015, row_cosh + 0.015],
        color="#D5DADF",
        lw=0.55,
        zorder=1,
    )
ax_alloc.scatter(
    uniform_z,
    np.full(pairs, row_uniform),
    s=9,
    marker="o",
    color=OI_BLUE,
    edgecolor="white",
    lw=0.25,
    zorder=3,
)
ax_alloc.scatter(
    cosh_z,
    np.full(pairs, row_cosh),
    s=10,
    marker="D",
    color=OI_VERMILLION,
    edgecolor="white",
    lw=0.25,
    zorder=3,
)
for y in [row_uniform, row_cosh]:
    ax_alloc.scatter(
        [0, 1], [y, y], s=31, facecolor="white", edgecolor=INK, lw=0.85, zorder=4
    )

support = FancyArrowPatch(
    (0, 0.89),
    (1, 0.89),
    arrowstyle="<->",
    mutation_scale=7,
    color=INK,
    lw=0.75,
)
ax_alloc.add_patch(support)
ax_alloc.text(
    0.5,
    0.92,
    r"sampled support $(a,R)$",
    ha="center",
    va="bottom",
    color=INK,
    fontsize=6.2,
)
ax_alloc.text(
    -0.035,
    row_uniform,
    r"uniform $z$",
    color=OI_BLUE,
    ha="right",
    va="center",
    fontsize=6.3,
    weight="bold",
)
ax_alloc.text(
    -0.035,
    row_cosh,
    r"Cosh $z$",
    color=OI_VERMILLION,
    ha="right",
    va="center",
    fontsize=6.3,
    weight="bold",
)
ax_alloc.text(
    0.5,
    0.055,
    r"$x_k=-\log\omega_k=a+Rz_k$",
    ha="center",
    va="center",
    color=INK,
    fontsize=7.0,
)
ax_alloc.text(
    0.0, 0.14, r"fast / high $\omega$", ha="left", va="center", color=MUTED, fontsize=6.0
)
ax_alloc.text(
    1.0, 0.14, r"slow / low $\omega$", ha="right", va="center", color=MUTED, fontsize=6.0
)
ax_alloc.set_xlim(-0.18, 1.03)
ax_alloc.set_ylim(0.0, 1.03)
ax_alloc.set_xticks([])
ax_alloc.set_yticks([])
ax_alloc.spines[["top", "right", "bottom", "left"]].set_visible(False)

ax_budget = fig.add_subplot(a_grid[1, 0])
directions = np.arange(1, len(trace_shares) + 1)
budget_colors = [OI_BLUE if index < 2 else "#D6DBDF" for index in range(len(trace_shares))]
ax_budget.bar(
    directions,
    trace_shares,
    width=0.78,
    color=budget_colors,
    edgecolor="white",
    linewidth=0.20,
    zorder=2,
)
ax_budget.text(
    0.98,
    0.91,
    r"one instance: $K=64, L=4096$",
    transform=ax_budget.transAxes,
    ha="right",
    va="top",
    color=MUTED,
    fontsize=6.0,
)
ax_budget.text(
    0.98,
    0.66,
    "23 slow pairs\n" + r"46 nominal $\rightarrow r_2=2.00$",
    transform=ax_budget.transAxes,
    ha="right",
    va="top",
    color=INK,
    fontsize=6.2,
    weight="bold",
)
ax_budget.set_xlim(0.2, len(trace_shares) + 0.8)
ax_budget.set_ylim(0, 56)
ax_budget.set_xticks([])
ax_budget.set_yticks([0, 50])
ax_budget.set_ylabel("trace (%)", labelpad=1.5)
ax_budget.grid(axis="y", color=GRID, lw=0.40, zorder=0)
ax_budget.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# (b) Fixed-support allocation effect and support-retargeted reversal.
# ---------------------------------------------------------------------------
b_grid = outer[0, 1].subgridspec(1, 2, wspace=0.10)
ax_fixed = fig.add_subplot(b_grid[0, 0])
ax_target = fig.add_subplot(b_grid[0, 1], sharey=ax_fixed)
x = np.arange(len(ratios))


def plot_seed_condition(
    ax: Axes,
    seed_values: np.ndarray,
    mean_values: np.ndarray,
    color: str,
    marker: str,
    linestyle: str,
    title: str,
    shade: str,
    shade_limits: tuple[float, float],
) -> None:
    ax.axhspan(shade_limits[0], shade_limits[1], color=shade, zorder=0)
    ax.axhline(0, color=INK, lw=0.70, zorder=1)
    for values in seed_values:
        ax.plot(
            x,
            values,
            color=SEED,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.55,
            ms=2.7,
            lw=0.75,
            linestyle=linestyle,
            zorder=2,
        )
    ax.plot(
        x,
        mean_values,
        color=color,
        marker=marker,
        ms=4.0,
        lw=1.65,
        linestyle=linestyle,
        zorder=3,
    )
    for xpos, value in zip(x, mean_values):
        offset = 0.045 if value >= 0 else -0.045
        ax.text(
            xpos,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            color=color,
            fontsize=6.0,
            weight="bold",
        )
    ax.set_title(title, loc="left", weight="bold", pad=3)
    ax.set_xticks(x, [rf"${ratio}\times$" for ratio in ratios])
    ax.set_xlim(-0.32, len(ratios) - 0.68)
    ax.set_ylim(-0.56, 0.78)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xlabel("relative length")
    ax.grid(axis="y", color=GRID, lw=0.42, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


plot_seed_condition(
    ax_fixed,
    fixed_seed,
    fixed_mean,
    OI_VERMILLION,
    "o",
    "-",
    "Support fixed",
    FIXED_BG,
    (-0.56, 0.0),
)
plot_seed_condition(
    ax_target,
    target_seed,
    target_mean,
    OI_BLUE,
    "s",
    "--",
    "Support retargeted",
    TARGET_BG,
    (0.0, 0.78),
)
ax_fixed.set_ylabel(
    r"paired $\Delta$ tail NLL" + "\n" + r"(anchored $z$ $-$ geometric $z$)"
)
ax_fixed.text(
    0.03,
    0.04,
    "all 3 OOD signs < 0",
    transform=ax_fixed.transAxes,
    ha="left",
    va="bottom",
    color=OI_VERMILLION,
    fontsize=6.0,
    weight="bold",
)
ax_target.text(
    0.97,
    0.96,
    "all 3 OOD signs > 0",
    transform=ax_target.transAxes,
    ha="right",
    va="top",
    color=OI_BLUE,
    fontsize=6.0,
    weight="bold",
)
ax_target.tick_params(axis="y", labelleft=False)


# ---------------------------------------------------------------------------
# (c) Frozen mature checkpoints: only interior z changes in the colored rows.
# ---------------------------------------------------------------------------
c_grid = outer[0, 2].subgridspec(1, 2, wspace=0.10)
ax_olmo = fig.add_subplot(c_grid[0, 0])
ax_qwen = fig.add_subplot(c_grid[0, 1], sharex=ax_olmo, sharey=ax_olmo)
pure_y = np.array([2.0, 1.0, 0.0])
pure_colors = [OI_SKY, OI_GREEN, OI_VERMILLION]
pure_markers = ["o", "s", "D"]


def plot_frozen_checkpoint(
    ax: Axes,
    values: np.ndarray,
    references: np.ndarray,
    contrast_ci: np.ndarray,
    title: str,
) -> None:
    ax.axhspan(-0.35, 2.35, color=PURE_Z_BG, zorder=0)
    for ypos, value, color, marker in zip(
        pure_y, values, pure_colors, pure_markers
    ):
        ax.hlines(ypos, 0.0, value, color=color, lw=2.0, zorder=2)
        ax.scatter(
            value,
            ypos,
            s=25,
            marker=marker,
            color=color,
            edgecolor="white",
            lw=0.45,
            zorder=3,
        )
        ax.text(
            min(value + 1.0, 70.5),
            ypos,
            f"{value:.2f}",
            ha="left" if value < 69.5 else "right",
            va="center",
            color=INK,
            fontsize=6.1,
            weight="bold",
        )
    ax.vlines(
        references[0], -0.30, 2.30, color=REFERENCE[0], lw=0.85, linestyle="--", zorder=1
    )
    ax.vlines(
        references[1], -0.30, 2.30, color=REFERENCE[1], lw=0.90, linestyle=":", zorder=1
    )
    ax.text(
        references[0], 2.42, "N", ha="center", va="bottom", color=REFERENCE[0], fontsize=6.0
    )
    ax.text(
        references[1], 2.42, "Y", ha="center", va="bottom", color=REFERENCE[1], fontsize=6.0
    )
    delta = values[-1] - values[0]
    ax.text(
        0.02,
        0.015,
        rf"$\Delta_{{D-U}}={delta:+.2f}$"
        + "\n"
        + rf"[{contrast_ci[0]:.2f}, {contrast_ci[1]:.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=OI_VERMILLION,
        fontsize=6.0,
        weight="bold",
    )
    ax.set_title(title, loc="left", weight="bold", pad=3)
    ax.set_xlim(-1.5, 72)
    ax.set_ylim(-0.58, 2.62)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("RULER macro (%)")
    ax.grid(axis="x", color=GRID, lw=0.42, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


plot_frozen_checkpoint(
    ax_olmo,
    olmo_pure_z,
    olmo_reference,
    olmo_derived_minus_uniform_ci,
    "OLMo: unseen-nine\n16K",
)
plot_frozen_checkpoint(
    ax_qwen,
    qwen_pure_z,
    qwen_reference,
    qwen_derived_minus_uniform_ci,
    "Qwen: core-four\n64K",
)
ax_olmo.set_yticks(pure_y, pure_z_labels)
ax_qwen.tick_params(axis="y", labelleft=False)
ax_qwen.text(
    0.98,
    0.02,
    "gray refs: N / Y",
    transform=ax_qwen.transAxes,
    ha="right",
    va="bottom",
    color=MUTED,
    fontsize=6.0,
)


fig.text(
    0.055,
    0.975,
    "(a) Finite table = support + allocation",
    ha="left",
    va="top",
    fontsize=7.5,
    weight="bold",
    color=INK,
)
fig.text(
    0.365,
    0.975,
    "(b) Causal support–allocation decomposition",
    ha="left",
    va="top",
    fontsize=7.5,
    weight="bold",
    color=INK,
)
fig.text(
    0.690,
    0.975,
    r"(c) Frozen pure-$z$ zero-training",
    ha="left",
    va="top",
    fontsize=7.5,
    weight="bold",
    color=INK,
)

fig.subplots_adjust(left=0.055, right=0.995, bottom=0.15, top=0.82)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.025)
print(f"wrote {OUT}")
