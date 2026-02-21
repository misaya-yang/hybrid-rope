from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.rope import RoPEFrequencyAllocator
from src.visualization import save_fig_both, set_plot_style


def run(root_dir: Path, device: torch.device) -> Dict:
    # Note: this module name is kept for compatibility with requested folder structure.
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    d = 128
    L = 131072
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    std = allocator.standard().cpu().numpy()
    sig_t, k_val, x0_val = allocator.sigmoid_analytical(L=L)
    sig = sig_t.cpu().numpy()
    idx = np.arange(len(std))

    ratio_std = std[1:] / std[:-1]
    ratio_sig = sig[1:] / sig[:-1]

    df_main = pd.DataFrame(
        {
            "index": idx.astype(int),
            "theta_standard": std.astype(np.float64),
            "theta_sigmoid": sig.astype(np.float64),
            "log_theta_standard": np.log(std).astype(np.float64),
            "log_theta_sigmoid": np.log(sig).astype(np.float64),
        }
    )
    df_main.to_csv(data_dir / "frequency_distribution.csv", index=False, encoding="utf-8")

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    axes[0].plot(idx, std, color="#d62728", label="Standard")
    axes[0].plot(idx, sig, color="#1f77b4", label="Sigmoid")
    axes[0].fill_between(idx, std, sig, color="#999999", alpha=0.2)
    axes[0].set_xlabel("Dimension Pair Index i")
    axes[0].set_ylabel(r"$\theta_i$")
    axes[0].set_title("Frequency Distribution")
    axes[0].legend(frameon=True)

    axes[1].plot(idx, np.log(std), color="#d62728", label="Standard")
    axes[1].plot(idx, np.log(sig), color="#1f77b4", label="Sigmoid")
    axes[1].set_xlabel("Dimension Pair Index i")
    axes[1].set_ylabel(r"$\log(\theta_i)$")
    axes[1].set_title("Log-Frequency")

    idx_ratio = np.arange(len(ratio_std))
    axes[2].plot(idx_ratio, ratio_std, color="#d62728", label="Standard")
    axes[2].plot(idx_ratio, ratio_sig, color="#1f77b4", label="Sigmoid")
    axes[2].set_xlabel("i")
    axes[2].set_ylabel(r"$\theta_{i+1} / \theta_i$")
    axes[2].set_title("Adjacent Frequency Ratio")
    axes[2].text(
        0.03,
        0.96,
        f"d={d}, L={L}\nk={k_val:.4f}, x0={x0_val:.2f}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.tight_layout()
    save_fig_both(fig, result_dir / "frequency_distribution")
    plt.close(fig)

    l_values: List[int] = [4096, 8192, 16384, 32768, 65536, 131072]
    records = []
    set_plot_style()
    fig2, ax = plt.subplots(figsize=(8.2, 5.0))
    cmap = plt.get_cmap("viridis")
    ax.plot(idx, np.log(std), color="black", linestyle="--", linewidth=1.4, label="Standard")
    for i, lv in enumerate(l_values):
        sig_l, k_l, x0_l = allocator.sigmoid_analytical(L=lv)
        sig_np = sig_l.cpu().numpy()
        color = cmap(i / max(1, len(l_values) - 1))
        ax.plot(idx, np.log(sig_np), color=color, label=f"Sigmoid L={lv}")
        for j in range(len(idx)):
            records.append(
                {
                    "L": int(lv),
                    "index": int(j),
                    "theta_sigmoid": float(sig_np[j]),
                    "log_theta_sigmoid": float(np.log(sig_np[j])),
                    "k": float(k_l),
                    "x0": float(x0_l),
                }
            )
    ax.set_xlabel("Dimension Pair Index i")
    ax.set_ylabel(r"$\log(\theta_i)$")
    ax.set_title("Sigmoid Frequency Curves Across Context Lengths")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, result_dir / "frequency_distribution_multi_L")
    plt.close(fig2)

    df_multi = pd.DataFrame(records)
    df_multi.to_csv(data_dir / "frequency_distribution_multi_L.csv", index=False, encoding="utf-8")

    print("\n[EXP4] Frequency summary:")
    print(f"  L={L}, d={d}, k={k_val:.6f}, x0={x0_val:.3f}")
    print(f"  mean(theta_sigmoid/theta_standard)={np.mean(sig / std):.6f}")

    return {
        "csv_main": str(data_dir / "frequency_distribution.csv"),
        "csv_multi": str(data_dir / "frequency_distribution_multi_L.csv"),
        "fig_main": str(result_dir / "frequency_distribution.pdf"),
        "fig_multi": str(result_dir / "frequency_distribution_multi_L.pdf"),
    }


if __name__ == "__main__":
    run(root_dir=Path(__file__).resolve().parents[1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

