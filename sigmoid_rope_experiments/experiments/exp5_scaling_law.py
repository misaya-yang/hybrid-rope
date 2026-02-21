from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.visualization import save_fig_both, set_plot_style


def run(root_dir: Path, device: torch.device) -> Dict:
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "grid_search_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing grid search result: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("grid_search_results.csv is empty.")

    d_values = sorted(df["d"].unique().tolist())
    l_values = sorted(df["L"].unique().tolist())

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    # (a) k vs L (formula vs grid search)
    for idx, d in enumerate(d_values):
        sub = df[df["d"] == d].sort_values("L")
        c = colors[idx % len(colors)]
        ax_a.plot(sub["L"], sub["k_formula"], color=c, linestyle="-", label=f"d={d} formula")
        ax_a.plot(sub["L"], sub["k_optimal"], color=c, linestyle="--", marker="o", label=f"d={d} optimal")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("L")
    ax_a.set_ylabel("k")
    ax_a.set_title("(a) k vs L")
    ax_a.legend(ncol=2, fontsize=8, frameon=True)

    # (b) x0 vs d, one line per L (optimal x0)
    for idx, L in enumerate(l_values):
        sub = df[df["L"] == L].sort_values("d")
        c = plt.get_cmap("viridis")(idx / max(1, len(l_values) - 1))
        ax_b.plot(sub["d"], sub["x0_optimal"], color=c, marker="o", label=f"L={L}")
    ax_b.set_xlabel("d")
    ax_b.set_ylabel("x0 (optimal)")
    ax_b.set_title("(b) x0 vs d")
    ax_b.legend(ncol=2, fontsize=8, frameon=True)

    # (c) score vs L, average over d
    g = df.groupby("L", as_index=False).agg(
        score_standard=("score_standard", "mean"),
        score_formula=("score_formula", "mean"),
        score_optimal=("score_optimal", "mean"),
    )
    ax_c.plot(g["L"], g["score_standard"], color="#d62728", marker="o", label="Standard")
    ax_c.plot(g["L"], g["score_formula"], color="#1f77b4", marker="o", label="Sigmoid (Formula)")
    ax_c.plot(g["L"], g["score_optimal"], color="#2ca02c", marker="o", label="Sigmoid (Optimal)")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("L")
    ax_c.set_ylabel("Phase Collision Score")
    ax_c.set_title("(c) Score vs L (mean over d)")
    ax_c.legend(frameon=True)

    # (d) relative improvement
    improve_formula = (g["score_standard"] - g["score_formula"]) / g["score_standard"]
    improve_opt = (g["score_standard"] - g["score_optimal"]) / g["score_standard"]
    ax_d.plot(g["L"], improve_formula, color="#1f77b4", marker="o", label="Formula improvement")
    ax_d.plot(g["L"], improve_opt, color="#2ca02c", marker="o", label="Optimal improvement")
    ax_d.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("L")
    ax_d.set_ylabel(r"$(S_{std}-S_{sig})/S_{std}$")
    ax_d.set_title("(d) Relative Improvement")
    ax_d.legend(frameon=True)

    fig.tight_layout()
    save_fig_both(fig, result_dir / "scaling_law_summary")
    plt.close(fig)

    summary = g.copy()
    summary["improve_formula"] = improve_formula
    summary["improve_optimal"] = improve_opt
    summary.to_csv(data_dir / "scaling_law_summary.csv", index=False, encoding="utf-8")

    print("\n[EXP5] Scaling law summary:")
    print(summary.to_string(index=False))

    return {
        "csv": str(data_dir / "scaling_law_summary.csv"),
        "fig": str(result_dir / "scaling_law_summary.pdf"),
    }


if __name__ == "__main__":
    run(root_dir=Path(__file__).resolve().parents[1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

