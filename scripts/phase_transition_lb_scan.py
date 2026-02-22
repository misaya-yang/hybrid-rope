#!/usr/bin/env python3
"""
Phase-transition re-check under different L/b ratios.

Motivation:
- The theorem condition for flat-optimal geometric behavior under uniform prior
  is asymptotic (L >> b). This script explicitly scans multiple L/b regimes.

What it does:
1) Builds standard geometric frequencies and a standalone anchored-hybrid variant.
2) Computes generalized phase-collision scores C_geo(p), C_hybrid(p) efficiently
   with chunked GPU computation (works for very large L).
3) Sweeps p in [0,1] and scans ratios L/b in a list.
4) Saves:
   - JSON summary of intersections.
   - CSV with all curve points.
   - Two publication-ready plots:
       (a) Delta C = C_hybrid - C_geo vs p (with zero line)
       (b) Raw scores C_geo/C_hybrid in a 2x2 panel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="L/b scan for mixed-prior phase transition")
    ap.add_argument("--d", type=int, default=128, help="Head dimension")
    ap.add_argument("--base", type=float, default=10000.0, help="RoPE base b")
    ap.add_argument("--alpha", type=float, default=0.2, help="Hybrid mixing alpha")
    ap.add_argument("--poly_power", type=float, default=2.0, help="Polynomial power for anchored hybrid tail")
    ap.add_argument("--ratios", type=str, default="1.6,10,100,1000", help="Comma-separated L/b ratios")
    ap.add_argument("--num_p", type=int, default=100, help="Number of p points in [0,1]")
    ap.add_argument("--chunk_size", type=int, default=262144, help="Delta chunk size for chunked computation")
    ap.add_argument("--out_dir", type=str, default="results/theory_2026-02-22", help="Output directory")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    return ap.parse_args()


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_theta_geo(d: int, base: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    i = torch.arange(d // 2, device=device, dtype=dtype)
    return base ** (-2.0 * i / d)


def build_theta_hybrid(theta_geo: torch.Tensor, alpha: float, poly_power: float) -> torch.Tensor:
    n = theta_geo.numel()
    t = torch.linspace(0.0, 1.0, n, dtype=theta_geo.dtype, device=theta_geo.device)
    t_poly = t**poly_power
    log_hi = torch.log(theta_geo[0])
    log_lo = torch.log(theta_geo[-1])
    theta_poly = torch.exp(log_hi + t_poly * (log_lo - log_hi))
    return (1.0 - alpha) * theta_geo + alpha * theta_poly


@torch.no_grad()
def compute_unif_and_pow_scores(
    theta: torch.Tensor,
    d: int,
    L: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, float]:
    """
    Compute:
      C_unif = (1/L) * sum_{Delta=1..L} E(Delta)
      C_pow  = (1/H_L) * sum_{Delta=1..L} E(Delta)/Delta
    where:
      E(Delta) = ((2/d) * sum_i cos(theta_i * Delta))^2
      H_L = sum_{j=1..L} 1/j
    """
    sum_E = torch.zeros((), dtype=dtype, device=device)
    sum_E_over_delta = torch.zeros((), dtype=dtype, device=device)

    # Harmonic normalization for power-law prior.
    # Split to avoid huge temporary vectors on extremely large L.
    H_L = torch.zeros((), dtype=dtype, device=device)
    for start in range(1, L + 1, chunk_size):
        end = min(L, start + chunk_size - 1)
        delta = torch.arange(start, end + 1, dtype=dtype, device=device)
        H_L += (1.0 / delta).sum()

        angles = theta[:, None] * delta[None, :]
        e = ((2.0 / d) * torch.cos(angles).sum(dim=0)).square()
        sum_E += e.sum()
        sum_E_over_delta += (e / delta).sum()

    c_unif = (sum_E / float(L)).item()
    c_pow = (sum_E_over_delta / H_L).item()
    return float(c_unif), float(c_pow)


def find_intersection(p: np.ndarray, y_geo: np.ndarray, y_hyb: np.ndarray) -> Tuple[float, float, bool]:
    diff = y_hyb - y_geo
    idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
    if idx.size:
        i = int(idx[0])
        p0, p1 = p[i], p[i + 1]
        d0, d1 = diff[i], diff[i + 1]
        p_star = float(p0 - d0 * (p1 - p0) / (d1 - d0))
        c0, c1 = y_geo[i], y_geo[i + 1]
        c_star = float(c0 + (p_star - p0) * (c1 - c0) / (p1 - p0))
        return p_star, c_star, True
    j = int(np.argmin(np.abs(diff)))
    return float(p[j]), float(y_geo[j]), False


def set_style() -> None:
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


def main() -> None:
    args = parse_args()
    if args.d % 2 != 0:
        raise ValueError("--d must be even.")

    device = get_device(args.cpu)
    dtype = torch.float64
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = [float(x.strip()) for x in str(args.ratios).split(",") if x.strip()]
    ratios = sorted(ratios)
    p = np.linspace(0.0, 1.0, int(args.num_p))

    theta_geo = build_theta_geo(args.d, args.base, device=device, dtype=dtype)
    theta_hyb = build_theta_hybrid(theta_geo, alpha=float(args.alpha), poly_power=float(args.poly_power))

    summary: List[Dict[str, float]] = []
    rows: List[Dict[str, float]] = []
    curves: Dict[float, Dict[str, np.ndarray]] = {}

    print(f"Device={device}, d={args.d}, base={args.base}, alpha={args.alpha}, poly_power={args.poly_power}")
    print(f"Ratios L/b: {ratios}")

    for ratio in ratios:
        L = int(round(float(args.base) * ratio))
        L = max(L, 2)
        print(f"\n[scan] ratio={ratio:.4g}, L={L} ...")

        c_unif_geo, c_pow_geo = compute_unif_and_pow_scores(
            theta=theta_geo,
            d=int(args.d),
            L=L,
            chunk_size=int(args.chunk_size),
            device=device,
            dtype=dtype,
        )
        c_unif_hyb, c_pow_hyb = compute_unif_and_pow_scores(
            theta=theta_hyb,
            d=int(args.d),
            L=L,
            chunk_size=int(args.chunk_size),
            device=device,
            dtype=dtype,
        )

        c_geo = (1.0 - p) * c_unif_geo + p * c_pow_geo
        c_hyb = (1.0 - p) * c_unif_hyb + p * c_pow_hyb
        delta = c_hyb - c_geo

        p_star, c_star, has_cross = find_intersection(p, c_geo, c_hyb)
        print(
            f"  C_unif(geo/hyb)={c_unif_geo:.8f}/{c_unif_hyb:.8f}, "
            f"C_pow(geo/hyb)={c_pow_geo:.8f}/{c_pow_hyb:.8f}, "
            f"p*={p_star:.6f}, cross={has_cross}"
        )

        curves[ratio] = {"p": p.copy(), "geo": c_geo.copy(), "hyb": c_hyb.copy(), "delta": delta.copy()}
        summary.append(
            {
                "ratio_L_over_b": float(ratio),
                "L": int(L),
                "C_unif_geo": float(c_unif_geo),
                "C_unif_hybrid": float(c_unif_hyb),
                "C_pow_geo": float(c_pow_geo),
                "C_pow_hybrid": float(c_pow_hyb),
                "p_star": float(p_star),
                "c_star": float(c_star),
                "exact_crossing": bool(has_cross),
                "delta_at_p0": float(delta[0]),
                "delta_at_p1": float(delta[-1]),
            }
        )
        for i in range(len(p)):
            rows.append(
                {
                    "ratio_L_over_b": float(ratio),
                    "L": int(L),
                    "p": float(p[i]),
                    "C_geo": float(c_geo[i]),
                    "C_hybrid": float(c_hyb[i]),
                    "delta_C_hybrid_minus_geo": float(delta[i]),
                }
            )

    # Save raw tables
    csv_path = out_dir / "phase_transition_lb_scan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("ratio_L_over_b,L,p,C_geo,C_hybrid,delta_C_hybrid_minus_geo\n")
        for r in rows:
            f.write(
                f"{r['ratio_L_over_b']},{r['L']},{r['p']},{r['C_geo']},{r['C_hybrid']},{r['delta_C_hybrid_minus_geo']}\n"
            )

    json_path = out_dir / "phase_transition_lb_scan_summary.json"
    payload = {
        "device": str(device),
        "d": int(args.d),
        "base": float(args.base),
        "alpha": float(args.alpha),
        "poly_power": float(args.poly_power),
        "chunk_size": int(args.chunk_size),
        "ratios": ratios,
        "summary": summary,
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "delta_plot_pdf": str(out_dir / "phase_transition_lb_scan_delta.pdf"),
            "delta_plot_png": str(out_dir / "phase_transition_lb_scan_delta.png"),
            "scores_plot_pdf": str(out_dir / "phase_transition_lb_scan_scores.pdf"),
            "scores_plot_png": str(out_dir / "phase_transition_lb_scan_scores.png"),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    set_style()

    # Plot A: delta curves (most direct for threshold/crossover)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cmap = plt.get_cmap("viridis")
    for idx, ratio in enumerate(ratios):
        c = cmap(idx / max(1, len(ratios) - 1))
        ax.plot(curves[ratio]["p"], curves[ratio]["delta"], color=c, lw=2.0, label=f"L/b={ratio:g}")
    ax.axhline(0.0, color="black", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel(r"Mixing Proportion $p$ (Uniform $\rightarrow$ Power-law)")
    ax.set_ylabel(r"$\Delta \mathcal{C} = \mathcal{C}_{hybrid} - \mathcal{C}_{geo}$")
    ax.set_title(r"Phase Transition Scan Across $L/b$ Ratios")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_transition_lb_scan_delta.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "phase_transition_lb_scan_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot B: raw scores per ratio (2x2)
    n = len(ratios)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.8 * nrows), squeeze=False)
    for idx, ratio in enumerate(ratios):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.plot(curves[ratio]["p"], curves[ratio]["geo"], color="#4C72B0", ls="--", lw=2.0, label="Standard Geometric")
        ax.plot(curves[ratio]["p"], curves[ratio]["hyb"], color="#DD8452", ls="-", lw=2.0, label="Anchored Hybrid")
        meta = next(x for x in summary if float(x["ratio_L_over_b"]) == float(ratio))
        p_star = float(meta["p_star"])
        c_star = float(meta["c_star"])
        if bool(meta["exact_crossing"]):
            ax.scatter([p_star], [c_star], color="black", s=20, zorder=4)
            ax.annotate(
                f"$p^*={p_star:.3f}$",
                xy=(p_star, c_star),
                xytext=(min(0.75, p_star + 0.06), c_star),
                arrowprops=dict(arrowstyle="->", lw=1.0),
                fontsize=8,
            )
        ax.set_title(f"L/b={ratio:g} (L={int(meta['L'])})")
        ax.set_xlabel("p")
        ax.set_ylabel(r"$\mathcal{C}$")
        if idx == 0:
            ax.legend(frameon=True)

    # hide empty subplots
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "phase_transition_lb_scan_scores.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "phase_transition_lb_scan_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {out_dir / 'phase_transition_lb_scan_delta.pdf'}")
    print(f"Saved: {out_dir / 'phase_transition_lb_scan_scores.pdf'}")


if __name__ == "__main__":
    main()

