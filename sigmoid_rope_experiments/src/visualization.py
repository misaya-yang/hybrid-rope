from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt


def set_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def save_fig_both(fig: plt.Figure, out_stem: str | Path) -> Tuple[Path, Path]:
    p = Path(out_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf = p.with_suffix(".pdf")
    png = p.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png)
    return pdf, png


def legend_outside(ax: plt.Axes, ncol: int = 1) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=ncol)


def pick_colors() -> Iterable[str]:
    return ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

