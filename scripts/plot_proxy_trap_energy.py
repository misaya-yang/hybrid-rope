#!/usr/bin/env python3
"""
Proxy-trap visualization for Theorem 3.5.

Outputs:
1) 3D energy surface E_diag(phi, lambda) under bimodal prior
2) 2D heatmap of the same surface
3) Density comparison rho*(phi) ~ 1/E_diag(phi) for uniform / power-law / bimodal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = PROJECT_ROOT / "sigmoid_rope_experiments"
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from src.visualization import save_fig_both, set_plot_style  # noqa: E402


def compute_E_diag_discrete(
    omega: torch.Tensor,
    prior: torch.Tensor,
    L: int,
    device: torch.device,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """
    E_diag(phi) = 0.5 * [1 + sum_delta D(delta) * cos(2 * omega(phi) * delta)].
    """
    if prior.numel() != L:
        raise ValueError(f"prior size mismatch: {prior.numel()} vs L={L}")
    omega = omega.to(device=device, dtype=torch.float64)
    prior = prior.to(device=device, dtype=torch.float64)
    out = torch.zeros_like(omega)
    for start in range(1, L + 1, chunk_size):
        end = min(L + 1, start + chunk_size)
        delta = torch.arange(start, end, device=device, dtype=torch.float64)  # [chunk]
        w = prior[start - 1 : end - 1]  # [chunk]
        c = torch.cos(2.0 * omega[:, None] * delta[None, :])  # [P, chunk]
        out += (c * w[None, :]).sum(dim=1)
    return 0.5 * (1.0 + out)


def prior_uniform(L: int, device: torch.device) -> torch.Tensor:
    p = torch.ones(L, device=device, dtype=torch.float64)
    p = p / p.sum()
    return p


def prior_powerlaw(L: int, gamma: float, device: torch.device) -> torch.Tensor:
    d = torch.arange(1, L + 1, device=device, dtype=torch.float64)
    p = d.pow(-gamma)
    p = p / p.sum()
    return p


def rho_from_E(E: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rho = 1.0 / np.clip(E, eps, None)
    rho = rho / np.trapezoid(rho, dx=1.0 / max(len(rho) - 1, 1))
    return rho


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate proxy-trap energy visualizations.")
    ap.add_argument("--L", type=int, default=131072)
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--phi_points", type=int, default=256)
    ap.add_argument("--lambda_points", type=int, default=121)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta_short", type=float, default=8.0)
    ap.add_argument("--delta_long_ratio", type=float, default=0.9)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--chunk_size", type=int, default=8192)
    ap.add_argument("--out_dir", type=str, default="results/theory_2026-02-22")
    ap.add_argument("--data_dir", type=str, default="data/theory_2026-02-22")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[proxy-trap] device={device}")

    phi = np.linspace(0.0, 1.0, args.phi_points, dtype=np.float64)
    omega = torch.tensor(args.base ** (-phi), dtype=torch.float64, device=device)

    delta_s = float(args.delta_short)
    delta_l = float(max(2.0, args.delta_long_ratio * args.L))
    lambda_grid = np.linspace(0.0, 1.0, args.lambda_points, dtype=np.float64)

    # Closed-form for Dirac bimodal prior:
    # E = 0.5 * [1 + lambda * cos(2*w*ds) + (1-lambda) * cos(2*w*dl)]
    c_s = torch.cos(2.0 * omega * delta_s).detach().cpu().numpy()  # [P]
    c_l = torch.cos(2.0 * omega * delta_l).detach().cpu().numpy()  # [P]
    E_surface = 0.5 * (1.0 + lambda_grid[:, None] * c_s[None, :] + (1.0 - lambda_grid[:, None]) * c_l[None, :])

    # Compare against diffuse priors for rho*(phi) shape.
    prior_u = prior_uniform(args.L, device=device)
    prior_p = prior_powerlaw(args.L, gamma=args.gamma, device=device)
    E_u = compute_E_diag_discrete(omega=omega, prior=prior_u, L=args.L, device=device, chunk_size=args.chunk_size)
    E_p = compute_E_diag_discrete(omega=omega, prior=prior_p, L=args.L, device=device, chunk_size=args.chunk_size)
    E_b = torch.tensor(E_surface[args.lambda_points // 2], dtype=torch.float64)  # lambda=0.5 slice

    E_u_np = E_u.detach().cpu().numpy()
    E_p_np = E_p.detach().cpu().numpy()
    E_b_np = E_b.detach().cpu().numpy()

    rho_u = rho_from_E(E_u_np)
    rho_p = rho_from_E(E_p_np)
    rho_b = rho_from_E(E_b_np)

    mid = (phi >= 0.3) & (phi <= 0.7)
    metrics: Dict[str, float] = {
        "E_uniform_min": float(E_u_np.min()),
        "E_powerlaw_min": float(E_p_np.min()),
        "E_bimodal_lambda0.5_min": float(E_b_np.min()),
        "rho_uniform_mid_mean": float(rho_u[mid].mean()),
        "rho_powerlaw_mid_mean": float(rho_p[mid].mean()),
        "rho_bimodal_mid_mean": float(rho_b[mid].mean()),
        "rho_bimodal_mid_over_powerlaw": float(rho_b[mid].mean() / max(rho_p[mid].mean(), 1e-12)),
        "delta_short": delta_s,
        "delta_long": delta_l,
    }

    np.savez_compressed(
        data_dir / "proxy_trap_energy_data.npz",
        phi=phi,
        lambda_grid=lambda_grid,
        E_surface=E_surface,
        E_uniform=E_u_np,
        E_powerlaw=E_p_np,
        E_bimodal=E_b_np,
        rho_uniform=rho_u,
        rho_powerlaw=rho_p,
        rho_bimodal=rho_b,
    )
    (data_dir / "proxy_trap_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    set_plot_style()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Figure 1: 3D surface.
    fig1 = plt.figure(figsize=(9.0, 6.2))
    ax1 = fig1.add_subplot(111, projection="3d")
    pp, ll = np.meshgrid(phi, lambda_grid)
    surf = ax1.plot_surface(pp, ll, E_surface, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel(r"$\lambda$ (short-mode weight)")
    ax1.set_zlabel(r"$E_{\mathrm{diag}}(\phi)$")
    ax1.set_title("Proxy Trap Energy Surface (Bimodal Prior)")
    fig1.colorbar(surf, shrink=0.6, pad=0.1, label=r"$E_{\mathrm{diag}}$")
    save_fig_both(fig1, out_dir / "proxy_trap_energy_surface_3d")
    plt.close(fig1)

    # Figure 2: heatmap.
    fig2, ax2 = plt.subplots(figsize=(8.6, 5.4))
    im = ax2.imshow(
        E_surface,
        origin="lower",
        aspect="auto",
        extent=[phi.min(), phi.max(), lambda_grid.min(), lambda_grid.max()],
        cmap="viridis",
    )
    ax2.set_xlabel(r"$\phi$")
    ax2.set_ylabel(r"$\lambda$ (short-mode weight)")
    ax2.set_title(r"Heatmap of $E_{\mathrm{diag}}(\phi)$ under Bimodal Prior")
    fig2.colorbar(im, ax=ax2, label=r"$E_{\mathrm{diag}}$")
    save_fig_both(fig2, out_dir / "proxy_trap_energy_heatmap")
    plt.close(fig2)

    # Figure 3: rho*(phi) comparison.
    fig3, ax3 = plt.subplots(figsize=(8.8, 5.4))
    ax3.plot(phi, rho_u, color="#d62728", label="Uniform prior")
    ax3.plot(phi, rho_p, color="#1f77b4", label="Power-law prior")
    ax3.plot(phi, rho_b, color="#2ca02c", label="Bimodal proxy prior (lambda=0.5)")
    ax3.axvspan(0.3, 0.7, color="#bbbbbb", alpha=0.2, label="Mid band")
    ax3.set_xlabel(r"$\phi$")
    ax3.set_ylabel(r"Normalized $\rho^*(\phi) \propto 1 / E_{\mathrm{diag}}(\phi)$")
    ax3.set_title("Density Redistribution and Mid-Frequency Dilution")
    ax3.legend(loc="best")
    save_fig_both(fig3, out_dir / "proxy_trap_density_dilution")
    plt.close(fig3)

    print("[proxy-trap] done")
    print(f"  data: {data_dir / 'proxy_trap_energy_data.npz'}")
    print(f"  summary: {data_dir / 'proxy_trap_summary.json'}")
    print(f"  fig dir: {out_dir}")
    print(f"  key metric rho_mid_bimodal / rho_mid_powerlaw = {metrics['rho_bimodal_mid_over_powerlaw']:.4f}")


if __name__ == "__main__":
    main()
