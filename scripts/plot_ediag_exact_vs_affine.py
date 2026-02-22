#!/usr/bin/env python3
"""
E_diag(phi) exact-vs-affine validation for Section 3/4 theory.

E_diag(phi) = 1/2 * [1 + (1/ln L) * integral_{1..L} cos(2*b^{-phi}*Delta)/Delta dDelta]

This script:
1) Computes E_diag exactly using scipy numerical quadrature.
2) Fits an affine approximation A + B*phi on mid-band phi in [0.1, 0.9].
3) Plots exact vs fitted affine for:
   - (b=1e4, L=16384)
   - (b=5e5, L=16384)
4) Saves figure + CSV + JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot E_diag exact vs affine approximation.")
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--num_phi", type=int, default=401)
    ap.add_argument("--phi_fit_lo", type=float, default=0.1)
    ap.add_argument("--phi_fit_hi", type=float, default=0.9)
    ap.add_argument("--out_dir", type=str, default="results/theory_2026-02-22")
    return ap.parse_args()


def e_diag_exact_numeric(phi: float, b: float, L: int) -> float:
    w = 2.0 * (b ** (-phi))

    def integrand(delta: float) -> float:
        return math.cos(w * delta) / delta

    val, _ = quad(integrand, 1.0, float(L), limit=300, epsabs=1e-10, epsrel=1e-10)
    return 0.5 * (1.0 + val / math.log(L))


def fit_affine(phi: np.ndarray, y: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    mask = (phi >= lo) & (phi <= hi)
    p = np.polyfit(phi[mask], y[mask], deg=1)
    b = float(p[0])
    a = float(p[1])
    return a, b


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return 1.0
    return 1.0 - ss_res / ss_tot


def run_case(
    phi: np.ndarray,
    b: float,
    L: int,
    fit_lo: float,
    fit_hi: float,
) -> Dict[str, object]:
    exact = np.array([e_diag_exact_numeric(float(x), b=b, L=L) for x in phi], dtype=np.float64)
    a_fit, b_fit = fit_affine(phi, exact, lo=fit_lo, hi=fit_hi)
    approx = a_fit + b_fit * phi

    mask = (phi >= fit_lo) & (phi <= fit_hi)
    r2_mid = r2_score(exact[mask], approx[mask])
    mae_mid = float(np.mean(np.abs(exact[mask] - approx[mask])))
    maxe_mid = float(np.max(np.abs(exact[mask] - approx[mask])))

    # Theory-guided slope under bulk approximation: B_th = ln(b)/(2 ln(L))
    b_theory = math.log(b) / (2.0 * math.log(L))
    gamma_e = 0.5772156649015329
    a_theory = 0.5 - (gamma_e + math.log(2.0)) / (2.0 * math.log(L))
    approx_theory = a_theory + b_theory * phi

    return {
        "b": float(b),
        "L": int(L),
        "phi": phi,
        "exact": exact,
        "approx_fit": approx,
        "approx_theory": approx_theory,
        "A_fit": float(a_fit),
        "B_fit": float(b_fit),
        "A_theory": float(a_theory),
        "B_theory": float(b_theory),
        "r2_mid": float(r2_mid),
        "mae_mid": float(mae_mid),
        "maxe_mid": float(maxe_mid),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    phi = np.linspace(0.0, 1.0, int(args.num_phi), dtype=np.float64)

    case_1 = run_case(phi, b=1.0e4, L=int(args.L), fit_lo=float(args.phi_fit_lo), fit_hi=float(args.phi_fit_hi))
    case_2 = run_case(phi, b=5.0e5, L=int(args.L), fit_lo=float(args.phi_fit_lo), fit_hi=float(args.phi_fit_hi))

    # Save CSV
    csv_path = out_dir / "ediag_exact_vs_affine.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("case,b,L,phi,exact,approx_fit,approx_theory\n")
        for name, c in [("b1e4", case_1), ("b5e5", case_2)]:
            for i in range(len(phi)):
                f.write(
                    f"{name},{c['b']},{c['L']},{c['phi'][i]},{c['exact'][i]},{c['approx_fit'][i]},{c['approx_theory'][i]}\n"
                )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    for ax, title, c in [
        (axes[0], r"$b=10^4,\ L=16384$", case_1),
        (axes[1], r"$b=5\times10^5,\ L=16384$", case_2),
    ]:
        ax.plot(c["phi"], c["exact"], color="#1f77b4", lw=2.0, label="Exact (numerical integral)")
        ax.plot(c["phi"], c["approx_fit"], color="#d62728", lw=1.8, ls="--", label="Affine fit $A+B\\phi$")
        ax.plot(c["phi"], c["approx_theory"], color="#2ca02c", lw=1.5, ls="-.", label="Theory affine")
        ax.axvspan(float(args.phi_fit_lo), float(args.phi_fit_hi), color="gray", alpha=0.08)
        ax.set_title(
            title + "\n"
            + rf"$B_{{fit}}={c['B_fit']:.4f},\ B_{{th}}={c['B_theory']:.4f},\ R^2_{{mid}}={c['r2_mid']:.4f}$"
        )
        ax.set_xlabel(r"$\phi$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$E_{\mathrm{diag}}(\phi)$")
    axes[1].legend(loc="best", frameon=True)
    fig.tight_layout()

    pdf_path = out_dir / "ediag_exact_vs_affine.pdf"
    png_path = out_dir / "ediag_exact_vs_affine.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "L": int(args.L),
        "d": int(args.d),
        "num_phi": int(args.num_phi),
        "fit_range": [float(args.phi_fit_lo), float(args.phi_fit_hi)],
        "case_b1e4": {
            "A_fit": case_1["A_fit"],
            "B_fit": case_1["B_fit"],
            "A_theory": case_1["A_theory"],
            "B_theory": case_1["B_theory"],
            "r2_mid": case_1["r2_mid"],
            "mae_mid": case_1["mae_mid"],
            "maxe_mid": case_1["maxe_mid"],
        },
        "case_b5e5": {
            "A_fit": case_2["A_fit"],
            "B_fit": case_2["B_fit"],
            "A_theory": case_2["A_theory"],
            "B_theory": case_2["B_theory"],
            "r2_mid": case_2["r2_mid"],
            "mae_mid": case_2["mae_mid"],
            "maxe_mid": case_2["maxe_mid"],
        },
        "artifacts": {
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
        },
    }
    summary_path = out_dir / "ediag_exact_vs_affine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {summary_path}")
    print("[b=1e4]  B_fit={:.6f}, B_theory={:.6f}, R2_mid={:.6f}".format(case_1["B_fit"], case_1["B_theory"], case_1["r2_mid"]))
    print("[b=5e5]  B_fit={:.6f}, B_theory={:.6f}, R2_mid={:.6f}".format(case_2["B_fit"], case_2["B_theory"], case_2["r2_mid"]))


if __name__ == "__main__":
    main()

