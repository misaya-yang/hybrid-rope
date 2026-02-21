from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.metrics import compute_attention_score_decay
from src.rope import RoPEFrequencyAllocator
from src.visualization import save_fig_both, set_plot_style


def run(root_dir: Path, device: torch.device) -> Dict:
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    d = 128
    L = 131072
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    std_freq = allocator.standard()
    sig_freq, k_val, x0_val = allocator.sigmoid_analytical(L=L)

    dist_std, score_std = compute_attention_score_decay(std_freq, L=L, d=d, max_distance=50000, device=device)
    dist_sig, score_sig = compute_attention_score_decay(sig_freq, L=L, d=d, max_distance=50000, device=device)

    df = pd.DataFrame(
        {
            "distance": dist_std.astype(int),
            "score_standard": score_std.astype(np.float64),
            "score_sigmoid": score_sig.astype(np.float64),
            "score_diff": (score_sig - score_std).astype(np.float64),
        }
    )
    df.to_csv(data_dir / "attention_decay.csv", index=False, encoding="utf-8")

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    short_mask = dist_std <= 500
    axes[0].plot(dist_std[short_mask], score_std[short_mask], color="#d62728", label="Standard")
    axes[0].plot(dist_sig[short_mask], score_sig[short_mask], color="#1f77b4", label="Sigmoid")
    axes[0].set_xlabel(r"$|m-n|$")
    axes[0].set_ylabel("Expected Attention Score")
    axes[0].set_title("Short-Range (0~500)")
    axes[0].legend(frameon=True)

    nonzero = dist_std > 0
    axes[1].plot(dist_std[nonzero], score_std[nonzero], color="#d62728", label="Standard")
    axes[1].plot(dist_sig[nonzero], score_sig[nonzero], color="#1f77b4", label="Sigmoid")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$|m-n|$ (log scale)")
    axes[1].set_title("Global View")
    axes[1].text(
        0.03,
        0.95,
        f"d={d}, L={L}\nk={k_val:.4f}, x0={x0_val:.2f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    fig.tight_layout()
    save_fig_both(fig, result_dir / "attention_decay")
    plt.close(fig)

    set_plot_style()
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6))
    diff = score_sig - score_std
    x = dist_std
    ax2.plot(x, diff, color="#333333", linewidth=1.2)
    ax2.fill_between(x, 0.0, diff, where=(diff >= 0), color="#1f77b4", alpha=0.35, label="Sigmoid better")
    ax2.fill_between(x, 0.0, diff, where=(diff < 0), color="#d62728", alpha=0.3, label="Standard better")
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel(r"$|m-n|$")
    ax2.set_ylabel("Sigmoid - Standard")
    ax2.set_title("Attention Decay Difference")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, result_dir / "attention_decay_difference")
    plt.close(fig2)

    print("\n[EXP3] Attention decay summary:")
    print(f"  mean(diff): {diff.mean():.6f}")
    print(f"  max(diff):  {diff.max():.6f}")
    print(f"  min(diff):  {diff.min():.6f}")

    return {
        "csv": str(data_dir / "attention_decay.csv"),
        "fig_decay": str(result_dir / "attention_decay.pdf"),
        "fig_diff": str(result_dir / "attention_decay_difference.pdf"),
    }


if __name__ == "__main__":
    run(root_dir=Path(__file__).resolve().parents[1], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

