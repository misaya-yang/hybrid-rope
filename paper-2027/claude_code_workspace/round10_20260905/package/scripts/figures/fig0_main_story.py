"""
Graphical abstract / main story figure for the EVQ-Cosh paper.

Usage:
  python scripts/figures/fig0_main_story.py

Outputs:
  paper/figs/fig0_main_story.pdf
  paper/figs/fig0_main_story.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }
)


C_GEO = "#2166ac"
C_EVQ = "#d6604d"
C_NAVY = "#0f3c5c"
C_TEXT = "#222222"
C_MUTED = "#666666"
C_GRID = "#e5e5e5"
C_PANEL = "#f8fafb"
C_YARN = "#7a4fb3"


def evq_phi(u: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=C_TEXT,
    )


def add_flow_arrow(fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes) -> None:
    p0 = ax_from.get_position()
    p1 = ax_to.get_position()
    y = (p0.y0 + p0.y1) / 2
    arrow = FancyArrowPatch(
        (p0.x1 + 0.015, y),
        (p1.x0 - 0.055, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=C_NAVY,
        alpha=0.95,
    )
    fig.patches.append(arrow)


def draw_budget_panel(ax: plt.Axes) -> None:
    add_panel_label(ax, "(a)", "RoPE has a finite spectral budget")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box = FancyBboxPatch(
        (0.04, 0.05),
        0.92,
        0.86,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.6,
        edgecolor="#d6dde3",
        facecolor=C_PANEL,
    )
    ax.add_patch(box)

    ax.text(
        0.10,
        0.83,
        r"$K=d_{\mathrm{rot}}/2$ fixed channels",
        ha="left",
        va="center",
        fontsize=7.6,
        color=C_TEXT,
        fontweight="bold",
    )
    ax.text(
        0.10,
        0.75,
        "geometric table = inherited allocation",
        ha="left",
        va="center",
        fontsize=6.5,
        color=C_MUTED,
    )

    left, right = 0.12, 0.88
    y_geo = 0.55
    y_evq = 0.31
    ax.plot([left, right], [y_geo, y_geo], color="#b8c1ca", lw=0.8)
    ax.plot([left, right], [y_evq, y_evq], color="#b8c1ca", lw=0.8)

    collision = Rectangle(
        (0.67, y_geo - 0.055),
        0.22,
        0.11,
        facecolor="#eeeeee",
        edgecolor="none",
        alpha=0.95,
        zorder=0,
    )
    ax.add_patch(collision)
    ax.text(
        0.78,
        y_geo + 0.11,
        "long-range\ncollision zone",
        ha="center",
        va="center",
        fontsize=5.7,
        color=C_MUTED,
    )

    k = np.linspace(0, 1, 20)
    x_geo = left + (right - left) * k
    x_evq = left + (right - left) * evq_phi(k, tau=1.5)

    ax.scatter(x_geo, np.full_like(x_geo, y_geo), s=18, color=C_GEO, zorder=3)
    ax.scatter(x_evq, np.full_like(x_evq, y_evq), s=18, color=C_EVQ, marker="s", zorder=3)

    ax.text(0.12, y_geo + 0.075, "Geometric", ha="left", va="center", fontsize=6.6, color=C_GEO, fontweight="bold")
    ax.text(0.12, y_evq + 0.075, "EVQ-Cosh", ha="left", va="center", fontsize=6.6, color=C_EVQ, fontweight="bold")
    ax.text(0.50, 0.12, "same K, different allocation", ha="center", va="center", fontsize=6.3, color=C_TEXT)

    ax.annotate(
        "less low-frequency\ncrowding",
        xy=(x_evq[-4], y_evq),
        xytext=(0.64, 0.21),
        fontsize=5.8,
        color=C_EVQ,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=C_EVQ, lw=0.8, mutation_scale=7),
    )


def draw_allocation_panel(ax: plt.Axes) -> None:
    add_panel_label(ax, "(b)", "EVQ-Cosh is a closed-form allocation")

    tau = 1.5
    k = np.arange(32)
    u = (k + 0.5) / len(k)
    phi_g = u
    phi_e = evq_phi(u, tau)
    x = np.linspace(0.0, 1.0, 256)

    ax.plot(x, x, "--", color="#9a9a9a", lw=1.0, label="geometric")
    ax.plot(x, evq_phi(x, tau), color=C_EVQ, lw=2.0, label="EVQ-Cosh")
    ax.scatter(u, phi_g, s=9, color=C_GEO, alpha=0.9)
    ax.scatter(u, phi_e, s=11, color=C_EVQ, marker="s", alpha=0.9)

    ax.fill_between(x, 0.58, 1.0, color="#f0f0f0", zorder=-2)
    ax.text(0.13, 0.88, "collision-prone\nlow frequencies", fontsize=5.7, color=C_MUTED, ha="left", va="center")
    ax.annotate(
        "non-geometric basin",
        xy=(0.72, evq_phi(np.array([0.72]), tau)[0]),
        xytext=(0.43, 0.38),
        fontsize=6.1,
        color=C_EVQ,
        arrowprops=dict(arrowstyle="-|>", color=C_EVQ, lw=0.8, mutation_scale=7),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"quantile $u_k$")
    ax.set_ylabel(r"log-frequency position $\phi_k$")
    ax.grid(True, color=C_GRID, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none")

    formula = (
        r"$\rho_\tau(\phi)=\dfrac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$"
        "\n"
        r"$\tau^*=d_{\mathrm{eff}}/\sqrt{L}$; replace inverse frequencies only"
    )
    ax.text(
        0.50,
        -0.28,
        formula,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef7fc", edgecolor="#b8d4e8", linewidth=0.6),
    )


def metric_card(
    ax: plt.Axes,
    y: float,
    title: str,
    metric: str,
    baseline_label: str,
    baseline_value: float,
    evq_label: str,
    evq_value: float,
    scale_max: float,
    higher_is_better: bool,
    callout: str,
) -> None:
    x0, y0, w, h = 0.04, y, 0.92, 0.245
    box = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.6,
        edgecolor="#d6dde3",
        facecolor=C_PANEL,
    )
    ax.add_patch(box)

    ax.text(x0 + 0.03, y0 + h - 0.055, title, ha="left", va="center", fontsize=6.8, color=C_TEXT, fontweight="bold")
    ax.text(x0 + 0.03, y0 + h - 0.120, metric, ha="left", va="center", fontsize=5.9, color=C_MUTED)

    bar_x, bar_w = x0 + 0.68, 0.17
    b1_y = y0 + 0.085
    b2_y = y0 + 0.035
    b_h = 0.038
    ax.add_patch(Rectangle((bar_x, b1_y), bar_w, b_h, facecolor="#e7edf4", edgecolor="none"))
    ax.add_patch(Rectangle((bar_x, b2_y), bar_w, b_h, facecolor="#f5e8e5", edgecolor="none"))

    b1 = min(baseline_value / scale_max, 1.0) * bar_w
    b2 = min(evq_value / scale_max, 1.0) * bar_w
    ax.add_patch(Rectangle((bar_x, b1_y), b1, b_h, facecolor=C_GEO, edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((bar_x, b2_y), b2, b_h, facecolor=C_EVQ, edgecolor="none", alpha=0.95))

    suffix = "%" if higher_is_better else ""
    ax.text(bar_x - 0.015, b1_y + b_h / 2, baseline_label, ha="right", va="center", fontsize=5.5, color=C_GEO)
    ax.text(bar_x - 0.015, b2_y + b_h / 2, evq_label, ha="right", va="center", fontsize=5.5, color=C_EVQ)
    ax.text(bar_x + bar_w + 0.012, b1_y + b_h / 2, f"{baseline_value:g}{suffix}", ha="left", va="center", fontsize=5.7, color=C_GEO)
    ax.text(bar_x + bar_w + 0.012, b2_y + b_h / 2, f"{evq_value:g}{suffix}", ha="left", va="center", fontsize=5.7, color=C_EVQ, fontweight="bold")

    arrow = r"$\uparrow$" if higher_is_better else r"$\downarrow$"
    ax.text(
        x0 + 0.03,
        y0 + 0.045,
        f"{callout} {arrow}",
        ha="left",
        va="center",
        fontsize=6.1,
        color=C_EVQ,
        fontweight="bold",
    )


def draw_evidence_panel(ax: plt.Axes) -> None:
    add_panel_label(ax, "(c)", "Three mechanism stress tests close the loop")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    metric_card(
        ax,
        0.69,
        "Substrate + range complementarity",
        r"PK@8K, 454M passkey-mix",
        "Geo+YaRN",
        61,
        "EVQ+YaRN",
        100,
        100,
        True,
        "+39 pp retrieval",
    )
    metric_card(
        ax,
        0.385,
        "PE-dominant extrapolation",
        r"PPL@8K, 125M FineWeb-Edu",
        "Geo",
        513.7,
        "EVQ",
        333.7,
        540,
        False,
        "-35.0% PPL",
    )
    metric_card(
        ax,
        0.08,
        "Scarce-channel MLA",
        r"16 RoPE channels, 432M MLA",
        "Geo",
        138.8,
        "EVQ",
        95.6,
        150,
        False,
        "-31.1% PPL",
    )

    ax.text(
        0.50,
        0.005,
        "Evidence isolates allocation shape, not learned parameters or tuned range scale.",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=C_MUTED,
    )


def main() -> None:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo, "paper", "figs")
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 3.42), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.985,
        top=0.78,
        bottom=0.25,
        width_ratios=[1.05, 1.13, 1.32],
        wspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    draw_budget_panel(axes[0])
    draw_allocation_panel(axes[1])
    draw_evidence_panel(axes[2])

    fig.suptitle(
        "RoPE's frequency table is a scarce spectral budget; EVQ-Cosh reallocates it analytically",
        x=0.50,
        y=0.985,
        ha="center",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.50,
        0.925,
        "Same rotary operator; zero learned parameters; complementary to inference-time range scaling.",
        ha="center",
        va="top",
        fontsize=7.0,
        color=C_MUTED,
    )

    add_flow_arrow(fig, axes[0], axes[1])
    add_flow_arrow(fig, axes[1], axes[2])

    pdf_path = os.path.join(out_dir, "fig0_main_story.pdf")
    png_path = os.path.join(out_dir, "fig0_main_story.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
