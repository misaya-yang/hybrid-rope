from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert

from src.metrics import compute_phase_collision_curve
from src.rope import RoPEFrequencyAllocator
from src.visualization import save_fig_both, set_plot_style


def run(root_dir: Path, device: torch.device) -> Dict:
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    d = 128
    base = 10000.0
    l_values = [4096, 16384, 65536, 131072]
    allocator = RoPEFrequencyAllocator(d=d, base=base)
    std_freqs = allocator.standard()

    rows: List[Dict] = []
    curves = {}
    for L in l_values:
        sig_freqs, k_val, x0_val = allocator.sigmoid_analytical(L=L)
        dist_std, col_std = compute_phase_collision_curve(std_freqs, L=L, num_points=2000, device=device)
        dist_sig, col_sig = compute_phase_collision_curve(sig_freqs, L=L, num_points=2000, device=device)
        curves[L] = {
            "k": float(k_val),
            "x0": float(x0_val),
            "standard": (dist_std, col_std),
            "sigmoid": (dist_sig, col_sig),
        }
        for dd, cs, cg in zip(dist_std.tolist(), col_std.tolist(), col_sig.tolist()):
            rows.append({"d": d, "L": L, "distance": int(dd), "collision_standard": float(cs), "collision_sigmoid": float(cg)})

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "phase_collision_multi_L.csv", index=False, encoding="utf-8")

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for ax, L in zip(axes, l_values):
        dist_std, col_std = curves[L]["standard"]
        dist_sig, col_sig = curves[L]["sigmoid"]
        ax.plot(dist_std, col_std, color="#d62728", label="Standard")
        ax.plot(dist_sig, col_sig, color="#1f77b4", label="Sigmoid-Analytical")
        ax.set_xscale("log")
        ax.set_title(f"L = {L}")
        ax.text(
            0.04,
            0.95,
            f"k={curves[L]['k']:.4f}\nx0={curves[L]['x0']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.set_xlabel(r"$|m-n|$")
    axes[0].set_ylabel("Phase Collision")
    axes[2].set_ylabel("Phase Collision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig_both(fig, result_dir / "phase_collision_multi_L")
    plt.close(fig)

    # Envelope plot
    set_plot_style()
    fig2, ax2 = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(l_values):
        color = cmap(i / max(1, len(l_values) - 1))
        dist_std, col_std = curves[L]["standard"]
        dist_sig, col_sig = curves[L]["sigmoid"]
        env_std = np.abs(hilbert(col_std))
        env_sig = np.abs(hilbert(col_sig))
        x_std = dist_std / float(L)
        x_sig = dist_sig / float(L)
        ax2.plot(x_std, env_std, color=color, linestyle="--", alpha=0.85, label=f"Std L={L}")
        ax2.plot(x_sig, env_sig, color=color, linestyle="-", alpha=0.95, label=f"Sig L={L}")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Normalized Distance $|m-n|/L$")
    ax2.set_ylabel("Envelope Amplitude")
    ax2.set_title("Phase-Collision Envelope Across Context Lengths")
    ax2.legend(ncol=2, fontsize=8, frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, result_dir / "phase_collision_envelope")
    plt.close(fig2)

    print("\n[EXP2] Analytical parameters by L:")
    for L in l_values:
        print(f"  L={L:6d} | k={curves[L]['k']:.6f} | x0={curves[L]['x0']:.3f}")

    return {
        "csv": str(data_dir / "phase_collision_multi_L.csv"),
        "fig_multi": str(result_dir / "phase_collision_multi_L.pdf"),
        "fig_env": str(result_dir / "phase_collision_envelope.pdf"),
    }


if __name__ == "__main__":
    run(root_dir=Path(__file__).resolve().parents[1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

