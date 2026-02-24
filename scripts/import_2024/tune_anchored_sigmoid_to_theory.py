#!/usr/bin/env python3
"""
Tune anchored-sigmoid schedule parameters against theory bands.

This script fits three schedule parameters:
  - anchor_factor
  - slope_raw (effective slope = slope_raw / head_dim)
  - center_ratio

Objective:
  - minimize RMSE to target rho* curve (mid-band target)
  - maximize in-band coverage between diagonal and full-functional theory curves
  - discourage non-monotonic increasing artifacts in rho(phi)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_dhat_exact(phi: float, base: float, seq_len: int) -> float:
    z = 2.0 * (base ** (-phi))
    _, ci_z = sici(z)
    _, ci_zl = sici(z * seq_len)
    return float((ci_zl - ci_z) / np.log(seq_len))


def compute_theory_curves(phi: np.ndarray, base: float, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    e_diag = np.array([0.5 * (1.0 + compute_dhat_exact(float(p), base, seq_len)) for p in phi], dtype=np.float64)
    e_diag = np.maximum(e_diag, 1e-8)

    rho_diag = 1.0 / e_diag
    rho_diag = rho_diag / trapz(rho_diag, phi)

    rho_cosh = np.cosh(1.0 - phi)
    rho_cosh = rho_cosh / trapz(rho_cosh, phi)
    return rho_diag, rho_cosh


def density_from_inv_freq(inv_freq: np.ndarray, phi_grid: np.ndarray) -> np.ndarray | None:
    omega = np.asarray(inv_freq, dtype=np.float64).reshape(-1)
    omega = omega[np.isfinite(omega) & (omega > 0)]
    if omega.size < 2:
        return None

    phi_raw = -np.log(omega)
    phi_sorted = np.sort(phi_raw)
    span = float(phi_sorted[-1] - phi_sorted[0])
    if span <= 1e-12:
        return np.ones_like(phi_grid)

    phi_norm = (phi_sorted - phi_sorted[0]) / span
    u = np.linspace(0.0, 1.0, len(phi_norm))
    dphi_du = np.gradient(phi_norm, u)
    rho_samples = 1.0 / np.clip(dphi_du, 1e-8, None)
    rho = np.interp(phi_grid, phi_norm, rho_samples, left=rho_samples[0], right=rho_samples[-1])
    rho = np.maximum(rho, 1e-8)
    rho = rho / trapz(rho, phi_grid)
    return rho


def build_anchored_sigmoid_inv_freq(
    head_dim: int,
    base: float,
    anchor_factor: float,
    slope_raw: float,
    center_ratio: float,
) -> np.ndarray:
    k = head_dim // 2
    idx = np.arange(k, dtype=np.float64)
    base_inv = 1.0 / (base ** (2.0 * idx / float(head_dim)))

    slope = float(slope_raw) / float(head_dim)
    center = float(center_ratio) * float(k)
    sig = 1.0 / (1.0 + np.exp(-slope * (idx - center)))
    scale_factor = 1.0 + (float(anchor_factor) - 1.0) * sig
    return base_inv / scale_factor


def load_real_inv_freq(json_path: Path, key: str) -> np.ndarray | None:
    if not json_path.exists():
        return None
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    vals = obj.get(key)
    if isinstance(vals, list) and vals:
        return np.asarray(vals, dtype=np.float64)
    return None


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def monotonic_increase_penalty(rho: np.ndarray) -> float:
    # Theory curves are decreasing in phi; penalize increasing segments.
    diff = np.diff(rho)
    return float(np.mean(np.clip(diff, 0.0, None)))


def evaluate_candidate(
    rho: np.ndarray,
    rho_diag: np.ndarray,
    rho_cosh: np.ndarray,
    band_tol: float,
    w_rmse: float,
    w_cov: float,
    w_mono: float,
) -> Dict[str, float]:
    low = np.minimum(rho_diag, rho_cosh)
    high = np.maximum(rho_diag, rho_cosh)
    target = 0.5 * (low + high)
    value_rmse = rmse(rho, target)
    inside = (rho >= low * (1.0 - band_tol)) & (rho <= high * (1.0 + band_tol))
    coverage = float(np.mean(inside))
    mono_pen = monotonic_increase_penalty(rho)

    loss = w_rmse * value_rmse + w_cov * (1.0 - coverage) + w_mono * mono_pen
    return {
        "loss": float(loss),
        "rmse_to_mid": value_rmse,
        "coverage": coverage,
        "mono_penalty": mono_pen,
    }


def run_grid_search(
    phi: np.ndarray,
    contexts: List[int],
    base: float,
    head_dim: int,
    band_tol: float,
    w_rmse: float,
    w_cov: float,
    w_mono: float,
    anchor_grid: np.ndarray,
    slope_grid: np.ndarray,
    center_grid: np.ndarray,
) -> Dict[str, float]:
    theory = {}
    for c in contexts:
        theory[c] = compute_theory_curves(phi=phi, base=base, seq_len=c)

    best = {
        "loss": float("inf"),
        "anchor_factor": None,
        "slope_raw": None,
        "center_ratio": None,
    }

    for anchor in anchor_grid:
        for slope_raw in slope_grid:
            for center_ratio in center_grid:
                inv = build_anchored_sigmoid_inv_freq(
                    head_dim=head_dim,
                    base=base,
                    anchor_factor=float(anchor),
                    slope_raw=float(slope_raw),
                    center_ratio=float(center_ratio),
                )
                rho = density_from_inv_freq(inv, phi)
                if rho is None:
                    continue

                losses = []
                for c in contexts:
                    rho_diag, rho_cosh = theory[c]
                    met = evaluate_candidate(
                        rho=rho,
                        rho_diag=rho_diag,
                        rho_cosh=rho_cosh,
                        band_tol=band_tol,
                        w_rmse=w_rmse,
                        w_cov=w_cov,
                        w_mono=w_mono,
                    )
                    losses.append(met["loss"])

                mean_loss = float(np.mean(losses))
                if mean_loss < best["loss"]:
                    best = {
                        "loss": mean_loss,
                        "anchor_factor": float(anchor),
                        "slope_raw": float(slope_raw),
                        "center_ratio": float(center_ratio),
                    }

    return best


def plot_context_overlay(
    out_path: Path,
    phi: np.ndarray,
    seq_len: int,
    rho_diag: np.ndarray,
    rho_cosh: np.ndarray,
    rho_real: np.ndarray | None,
    rho_tuned: np.ndarray,
    title_suffix: str,
) -> None:
    low = np.minimum(rho_diag, rho_cosh)
    high = np.maximum(rho_diag, rho_cosh)

    plt.figure(figsize=(8, 4.8))
    plt.fill_between(phi, low, high, color="#4c72b0", alpha=0.15, label="Theory band")
    plt.plot(phi, rho_diag, "--", color="#4c72b0", linewidth=1.5, label="rho_diag")
    plt.plot(phi, rho_cosh, "-", color="#1f4e79", linewidth=1.8, label="rho_cosh")

    if rho_real is not None:
        plt.plot(phi, rho_real, color="#d62728", linewidth=1.6, alpha=0.8, label="current anchored_sigmoid")
    plt.plot(phi, rho_tuned, color="#2ca02c", linewidth=2.0, label="tuned anchored_sigmoid")

    plt.xlim(0.0, 1.0)
    plt.xlabel("phi")
    plt.ylabel("rho(phi)")
    plt.title(f"Theory Alignment @ L={seq_len} {title_suffix}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune anchored-sigmoid schedule against theory bands.")
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--contexts", type=str, default="16384,32768")
    ap.add_argument("--phi_points", type=int, default=500)
    ap.add_argument("--band_tol", type=float, default=0.20)
    ap.add_argument("--w_rmse", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=0.35)
    ap.add_argument("--w_mono", type=float, default=0.50)
    ap.add_argument("--anchor_min", type=float, default=2.0)
    ap.add_argument("--anchor_max", type=float, default=30.0)
    ap.add_argument("--anchor_steps", type=int, default=29)
    ap.add_argument("--slope_min", type=float, default=4.0)
    ap.add_argument("--slope_max", type=float, default=40.0)
    ap.add_argument("--slope_steps", type=int, default=19)
    ap.add_argument("--center_min", type=float, default=0.30)
    ap.add_argument("--center_max", type=float, default=0.70)
    ap.add_argument("--center_steps", type=int, default=17)
    ap.add_argument(
        "--invfreq_json",
        type=str,
        default="scripts/import_2024/real_inv_freq_20260223.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/reviewer_2026-02-24/tuning",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    phi = np.linspace(0.01, 0.99, args.phi_points)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_grid = np.linspace(args.anchor_min, args.anchor_max, args.anchor_steps)
    slope_grid = np.linspace(args.slope_min, args.slope_max, args.slope_steps)
    center_grid = np.linspace(args.center_min, args.center_max, args.center_steps)

    print(f"[tune] contexts={contexts} phi_points={args.phi_points}")
    print(
        f"[tune] grid sizes: anchor={len(anchor_grid)} slope={len(slope_grid)} center={len(center_grid)} "
        f"total={len(anchor_grid)*len(slope_grid)*len(center_grid)}"
    )

    best = run_grid_search(
        phi=phi,
        contexts=contexts,
        base=args.base,
        head_dim=args.head_dim,
        band_tol=args.band_tol,
        w_rmse=args.w_rmse,
        w_cov=args.w_cov,
        w_mono=args.w_mono,
        anchor_grid=anchor_grid,
        slope_grid=slope_grid,
        center_grid=center_grid,
    )
    print(
        f"[tune] best: loss={best['loss']:.6f} anchor={best['anchor_factor']:.3f} "
        f"slope_raw={best['slope_raw']:.3f} center_ratio={best['center_ratio']:.3f}"
    )

    inv_real = load_real_inv_freq(Path(args.invfreq_json), "anchored_sigmoid")
    rho_real = density_from_inv_freq(inv_real, phi) if inv_real is not None else None

    inv_tuned = build_anchored_sigmoid_inv_freq(
        head_dim=args.head_dim,
        base=args.base,
        anchor_factor=best["anchor_factor"],
        slope_raw=best["slope_raw"],
        center_ratio=best["center_ratio"],
    )
    rho_tuned = density_from_inv_freq(inv_tuned, phi)
    if rho_tuned is None:
        raise RuntimeError("Failed to build rho for tuned parameters.")

    summary: Dict[str, object] = {
        "base": args.base,
        "head_dim": args.head_dim,
        "contexts": contexts,
        "objective": {
            "band_tol": args.band_tol,
            "w_rmse": args.w_rmse,
            "w_cov": args.w_cov,
            "w_mono": args.w_mono,
        },
        "best_params": best,
        "per_context": {},
    }

    for c in contexts:
        rho_diag, rho_cosh = compute_theory_curves(phi=phi, base=args.base, seq_len=c)
        met_tuned = evaluate_candidate(
            rho=rho_tuned,
            rho_diag=rho_diag,
            rho_cosh=rho_cosh,
            band_tol=args.band_tol,
            w_rmse=args.w_rmse,
            w_cov=args.w_cov,
            w_mono=args.w_mono,
        )
        met_real = None
        if rho_real is not None:
            met_real = evaluate_candidate(
                rho=rho_real,
                rho_diag=rho_diag,
                rho_cosh=rho_cosh,
                band_tol=args.band_tol,
                w_rmse=args.w_rmse,
                w_cov=args.w_cov,
                w_mono=args.w_mono,
            )

        summary["per_context"][str(c)] = {
            "tuned": met_tuned,
            "current_anchored_sigmoid": met_real,
            "improvement_loss_pct": (
                100.0 * (met_real["loss"] - met_tuned["loss"]) / max(met_real["loss"], 1e-12)
                if met_real is not None
                else None
            ),
            "improvement_coverage_abs": (
                met_tuned["coverage"] - met_real["coverage"] if met_real is not None else None
            ),
        }

        plot_context_overlay(
            out_path=out_dir / f"tuned_vs_theory_L{c}.png",
            phi=phi,
            seq_len=c,
            rho_diag=rho_diag,
            rho_cosh=rho_cosh,
            rho_real=rho_real,
            rho_tuned=rho_tuned,
            title_suffix="(current vs tuned)",
        )

    (out_dir / "tuned_params.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[tune] wrote: {(out_dir / 'tuned_params.json')}")
    for c in contexts:
        obj = summary["per_context"][str(c)]
        tuned = obj["tuned"]
        print(
            f"[tune] L={c}: tuned loss={tuned['loss']:.6f}, "
            f"coverage={tuned['coverage']:.4f}, rmse={tuned['rmse_to_mid']:.4f}"
        )
        if obj["current_anchored_sigmoid"] is not None:
            cur = obj["current_anchored_sigmoid"]
            print(
                f"        current loss={cur['loss']:.6f}, coverage={cur['coverage']:.4f}, "
                f"rmse={cur['rmse_to_mid']:.4f}, improvement_loss_pct={obj['improvement_loss_pct']:.2f}%"
            )


if __name__ == "__main__":
    main()
