#!/usr/bin/env python3
"""
Phase transition threshold in mixed priors (Section 4.1).

Standalone script:
1) Builds Standard Geometric and Anchored Hybrid RoPE frequencies.
2) Computes generalized phase collision scores under mixed priors:
   D_p = p * D_pow + (1 - p) * D_unif.
3) Sweeps p in [0, 1], finds critical threshold p* (intersection),
   and generates publication-ready plots.

Outputs:
  - phase_transition_theory.pdf
  - phase_transition_theory.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase transition threshold in mixed priors")
    ap.add_argument("--L", type=int, default=8192, help="Maximum distance (Delta in [1, L])")
    ap.add_argument("--d", type=int, default=128, help="Head dimension")
    ap.add_argument("--base", type=float, default=10000.0, help="RoPE base")
    ap.add_argument("--alpha", type=float, default=0.2, help="Hybrid mixing weight")
    ap.add_argument("--poly_power", type=float, default=2.0, help="Polynomial power for anchored log-space interpolation")
    ap.add_argument("--num_p", type=int, default=100, help="Number of p points in [0, 1]")
    ap.add_argument("--out_dir", type=str, default=".", help="Directory to save figures")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    return ap.parse_args()


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_geometric_freqs(d: int, base: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # theta_i = base^{-2i/d}, i = 0..d/2-1
    n = d // 2
    i = torch.arange(n, dtype=dtype, device=device)
    return base ** (-2.0 * i / d)


def build_anchored_polynomial_freqs(
    theta_geo: torch.Tensor,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Preserve min/max of geometric frequencies and re-allocate intermediate points
    using polynomial interpolation in log-space.
    """
    n = theta_geo.numel()
    t = torch.linspace(0.0, 1.0, n, dtype=theta_geo.dtype, device=theta_geo.device)
    t_poly = t**power

    log_max = torch.log(theta_geo[0])      # high frequency anchor
    log_min = torch.log(theta_geo[-1])     # low frequency anchor
    log_theta = log_max + t_poly * (log_min - log_max)
    return torch.exp(log_theta)


def compute_collision_curve(theta: torch.Tensor, delta: torch.Tensor, d: int) -> torch.Tensor:
    """
    E(Delta) = ((2/d) * sum_i cos(theta_i * Delta))^2
    Returns shape [L].
    """
    angles = theta[:, None] * delta[None, :]
    cos_sum = torch.cos(angles).sum(dim=0) * (2.0 / d)
    return cos_sum.square()


def find_intersection(p: np.ndarray, y_geo: np.ndarray, y_hybrid: np.ndarray) -> Tuple[float, float, bool]:
    """
    Return (p_star, C_star, exact_crossing_found).
    If no sign change exists, return nearest |diff| point.
    """
    diff = y_hybrid - y_geo
    sign = np.sign(diff)
    idx_change = np.where(sign[:-1] * sign[1:] < 0)[0]

    if idx_change.size > 0:
        i = int(idx_change[0])
        p0, p1 = p[i], p[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        # Linear interpolation where diff crosses 0.
        p_star = float(p0 - d0 * (p1 - p0) / (d1 - d0))
        c0, c1 = y_geo[i], y_geo[i + 1]
        c_star = float(c0 + (p_star - p0) * (c1 - c0) / (p1 - p0))
        return p_star, c_star, True

    # Fallback: nearest point (no strict crossing in sampled range).
    j = int(np.argmin(np.abs(diff)))
    return float(p[j]), float(y_geo[j]), False


def main() -> None:
    args = parse_args()
    device = get_device(force_cpu=args.cpu)
    dtype = torch.float64

    if args.d % 2 != 0:
        raise ValueError("--d must be even because d/2 frequency pairs are required.")
    if args.L <= 1:
        raise ValueError("--L must be > 1.")

    # Academic style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )

    L = int(args.L)
    d = int(args.d)
    base = float(args.base)
    alpha = float(args.alpha)
    poly_power = float(args.poly_power)

    # Delta: 1..L
    delta = torch.arange(1, L + 1, dtype=dtype, device=device)

    # Frequencies
    theta_geo = build_geometric_freqs(d=d, base=base, device=device, dtype=dtype)
    theta_poly = build_anchored_polynomial_freqs(theta_geo=theta_geo, power=poly_power)
    theta_hybrid = (1.0 - alpha) * theta_geo + alpha * theta_poly

    # Collision curves E(Delta)
    E_geo = compute_collision_curve(theta_geo, delta, d=d)
    E_hybrid = compute_collision_curve(theta_hybrid, delta, d=d)

    # Priors
    D_unif = torch.full((L,), 1.0 / L, dtype=dtype, device=device)
    D_pow = (1.0 / delta)
    D_pow = D_pow / D_pow.sum()

    # Sweep p
    p_values = torch.linspace(0.0, 1.0, int(args.num_p), dtype=dtype, device=device)
    D_mix = p_values[:, None] * D_pow[None, :] + (1.0 - p_values[:, None]) * D_unif[None, :]

    C_geo = (D_mix * E_geo[None, :]).sum(dim=1)
    C_hybrid = (D_mix * E_hybrid[None, :]).sum(dim=1)

    # Move to CPU for plotting
    p_np = p_values.detach().cpu().numpy()
    C_geo_np = C_geo.detach().cpu().numpy()
    C_hybrid_np = C_hybrid.detach().cpu().numpy()

    p_star, c_star, has_cross = find_intersection(p_np, C_geo_np, C_hybrid_np)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # Regime shading
    ax.axvspan(0.0, p_star, color="#4C72B0", alpha=0.08)
    ax.axvspan(p_star, 1.0, color="#DD8452", alpha=0.08)

    ax.plot(
        p_np,
        C_geo_np,
        color="#4C72B0",
        linestyle="--",
        linewidth=2.0,
        label="Standard Geometric (Flat Optimum)",
    )
    ax.plot(
        p_np,
        C_hybrid_np,
        color="#DD8452",
        linestyle="-",
        linewidth=2.0,
        label="Anchored Hybrid (Convex Optimum)",
    )

    # Critical point
    ax.scatter([p_star], [c_star], color="black", s=28, zorder=5)
    crit_txt = f"Critical Threshold $p^*$ = {p_star:.3f}"
    if not has_cross:
        crit_txt += " (nearest)"
    ax.annotate(
        crit_txt,
        xy=(p_star, c_star),
        xytext=(p_star + 0.04, c_star + 0.01 * (np.max(C_geo_np) - np.min(C_geo_np))),
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        fontsize=9,
    )

    y_top = max(float(np.max(C_geo_np)), float(np.max(C_hybrid_np)))
    y_bottom = min(float(np.min(C_geo_np)), float(np.min(C_hybrid_np)))
    y_range = max(1e-12, y_top - y_bottom)

    ax.text(0.02, y_top - 0.06 * y_range, "Uniform Regime / Standard Wins", color="#2E4A7D", fontsize=9)
    ax.text(min(0.62, p_star + 0.02), y_top - 0.06 * y_range, "Linguistic Regime / Hybrid Wins", color="#8A4F24", fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Mixing Proportion $p$ (Uniform $\rightarrow$ Power-law)")
    ax.set_ylabel(r"Generalized Phase Collision Score $\mathcal{C}$")
    ax.set_title("Phase Transition Threshold in Mixed Priors")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "phase_transition_theory.pdf"
    png_path = out_dir / "phase_transition_theory.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Device: {device}")
    print(f"L={L}, d={d}, base={base}, alpha={alpha}, poly_power={poly_power}")
    print(f"p*={p_star:.6f}, C*={c_star:.8f}, exact_crossing={has_cross}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

