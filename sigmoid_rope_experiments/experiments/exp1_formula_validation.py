from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.grid_search import GridSearchConfig, run_grid_search
from src.visualization import pick_colors, save_fig_both, set_plot_style


def _plot_k_vs_L(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = list(pick_colors())
    for idx, d in enumerate(sorted(df["d"].unique())):
        sub = df[df["d"] == d].sort_values("L")
        c = colors[idx % len(colors)]
        ax.plot(sub["L"], sub["k_formula"], color=c, linestyle="-", label=f"d={d} formula")
        ax.scatter(sub["L"], sub["k_optimal"], color=c, marker="o", s=32, label=f"d={d} grid-opt")
    ax.set_xscale("log")
    ax.set_xlabel("Context Length L")
    ax.set_ylabel("k")
    ax.set_title("Formula-Predicted vs Grid-Optimal k")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_validation_k")
    plt.close(fig)


def _plot_x0_vs_L(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = list(pick_colors())
    for idx, d in enumerate(sorted(df["d"].unique())):
        sub = df[df["d"] == d].sort_values("L")
        c = colors[idx % len(colors)]
        ax.plot(sub["L"], sub["x0_formula"], color=c, linestyle="-", label=f"d={d} formula")
        ax.scatter(sub["L"], sub["x0_optimal"], color=c, marker="o", s=32, label=f"d={d} grid-opt")
    ax.set_xscale("log")
    ax.set_xlabel("Context Length L")
    ax.set_ylabel("x0")
    ax.set_title("Formula-Predicted vs Grid-Optimal x0")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_validation_x0")
    plt.close(fig)


def _plot_score_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharey=True)
    methods = [
        ("score_standard", "Standard", "#d62728"),
        ("score_formula", "Sigmoid (Formula)", "#1f77b4"),
        ("score_optimal", "Sigmoid (Grid-Optimal)", "#2ca02c"),
    ]
    for ax, d in zip(axes, sorted(df["d"].unique())):
        sub = df[df["d"] == d].sort_values("L")
        for col, name, color in methods:
            ax.plot(sub["L"], sub[col], marker="o", color=color, label=name)
        ax.set_xscale("log")
        ax.set_title(f"d={d}")
        ax.set_xlabel("L")
    axes[0].set_ylabel("Phase Collision Score (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_fig_both(fig, out_dir / "formula_vs_gridsearch_score")
    plt.close(fig)


def run(
    root_dir: Path,
    device: torch.device,
    search_mode: str = "auto",
) -> Dict:
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = GridSearchConfig(
        mode=search_mode,
        checkpoint_path=str(data_dir / "grid_search_checkpoint.json"),
        csv_path=str(data_dir / "grid_search_results.csv"),
    )
    df = run_grid_search(cfg, device=device)
    if df.empty:
        raise RuntimeError("Grid search produced no rows.")

    _plot_k_vs_L(df, result_dir)
    _plot_x0_vs_L(df, result_dir)
    _plot_score_comparison(df, result_dir)

    preview_cols = ["d", "L", "k_optimal", "x0_optimal", "k_formula", "x0_formula", "score_standard", "score_formula", "score_optimal"]
    print("\n[EXP1] Key results preview:")
    print(df[preview_cols].sort_values(["d", "L"]).head(18).to_string(index=False))

    return {
        "rows": int(len(df)),
        "csv": str(data_dir / "grid_search_results.csv"),
        "fig_k": str(result_dir / "formula_validation_k.pdf"),
        "fig_x0": str(result_dir / "formula_validation_x0.pdf"),
        "fig_score": str(result_dir / "formula_vs_gridsearch_score.pdf"),
    }


if __name__ == "__main__":
    run(root_dir=Path(__file__).resolve().parents[1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

