#!/usr/bin/env python3
"""
Bridge empirical attention-distance priors and theory predictions.

Inputs:
- Two prior-fit artifacts from scripts/run_attn_hist.py (baseline vs anchored)
- Optional inv_freq tensors for observed density comparison

Outputs:
- prior_fit_comparison.json
- prior_fit_comparison.png
- prior_fit_comparison.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build baseline-vs-anchored empirical prior bridge.")
    ap.add_argument("--baseline_prior_json", type=str, required=True)
    ap.add_argument("--anchored_prior_json", type=str, required=True)
    ap.add_argument("--baseline_label", type=str, default="baseline")
    ap.add_argument("--anchored_label", type=str, default="anchored_tuned")
    ap.add_argument("--baseline_inv_freq", type=str, default="")
    ap.add_argument("--anchored_inv_freq", type=str, default="")
    ap.add_argument("--base", type=float, default=500000.0, help="RoPE base used to map phi->omega.")
    ap.add_argument("--seq_len", type=int, default=32768, help="Target context for diagonal prediction.")
    ap.add_argument("--phi_points", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--head_dim", type=int, default=128, help="Used for default geometric baseline if no inv_freq provided.")
    ap.add_argument("--hf_phi_threshold", type=float, default=0.70, help="High-frequency mass integration threshold.")
    ap.add_argument("--out_dir", type=str, default="artifacts/reviewer_2026-02-24/prior_bridge")
    return ap.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prior_fit(path: Path) -> Dict[str, object]:
    obj = load_json(path)
    overall = obj.get("overall", {}) if isinstance(obj.get("overall"), dict) else {}
    alpha = overall.get("alpha")
    ci_low = overall.get("alpha_ci_low")
    ci_high = overall.get("alpha_ci_high")
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"Missing numeric overall.alpha in {path}")

    hist = obj.get("overall_hist")
    hist_arr = None
    if isinstance(hist, list) and hist and all(isinstance(x, (int, float)) for x in hist):
        hist_arr = np.asarray(hist, dtype=np.float64)
        s = float(np.sum(hist_arr))
        if s > 0:
            hist_arr = hist_arr / s

    return {
        "alpha": float(alpha),
        "alpha_ci_low": float(ci_low) if isinstance(ci_low, (int, float)) else None,
        "alpha_ci_high": float(ci_high) if isinstance(ci_high, (int, float)) else None,
        "hist": hist_arr,
        "meta": obj.get("meta", {}),
        "path": str(path),
    }


def powerlaw_prior(alpha: float, seq_len: int) -> np.ndarray:
    delta = np.arange(1, seq_len + 1, dtype=np.float64)
    w = np.power(np.maximum(delta, 1.0), -float(alpha))
    z = float(np.sum(w))
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError(f"Invalid normalization for alpha={alpha}, seq_len={seq_len}")
    return w / z


def predict_density_from_alpha(
    alpha: float,
    base: float,
    seq_len: int,
    phi_points: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 1.0, int(phi_points), dtype=np.float64)
    omega = np.power(float(base), -phi, dtype=np.float64)
    prior = powerlaw_prior(alpha=alpha, seq_len=seq_len)

    e_diag = np.zeros_like(phi, dtype=np.float64)
    for start in range(1, seq_len + 1, int(chunk_size)):
        end = min(seq_len, start + int(chunk_size) - 1)
        delta = np.arange(start, end + 1, dtype=np.float64)
        cosv = np.cos(np.outer(omega, delta))
        w = prior[start - 1 : end]
        e_diag += np.sum((cosv * cosv) * w[None, :], axis=1)

    e_diag = np.maximum(e_diag, 1e-8)
    rho = 1.0 / e_diag
    rho = rho / max(trapz(rho, phi), 1e-12)
    return phi, rho, e_diag


def load_inv_freq(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"inv_freq path not found: {p}")

    if p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
            return np.asarray(obj, dtype=np.float64)
        if isinstance(obj, dict):
            for key in ("inv_freq", "custom_inv_freq", "data", "tensor"):
                vals = obj.get(key)
                if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                    return np.asarray(vals, dtype=np.float64)
        raise RuntimeError(f"Unsupported JSON inv_freq payload in {p}")

    if p.suffix == ".npy":
        arr = np.load(str(p))
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    if p.suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required to load .pt/.pth inv_freq files") from exc
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().float().numpy().reshape(-1)
        if isinstance(obj, dict):
            for key in ("inv_freq", "custom_inv_freq", "data", "tensor"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    return val.detach().cpu().float().numpy().reshape(-1)
        raise RuntimeError(f"Unsupported tensor payload in {p}")

    raise RuntimeError(f"Unsupported inv_freq suffix for {p}")


def geometric_inv_freq(head_dim: int, base: float) -> np.ndarray:
    k = int(head_dim) // 2
    idx = np.arange(k, dtype=np.float64)
    return np.power(float(base), -(2.0 * idx / float(head_dim)), dtype=np.float64)


def density_from_inv_freq(inv_freq: np.ndarray, phi_grid: np.ndarray) -> Optional[np.ndarray]:
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
    rho = rho / max(trapz(rho, phi_grid), 1e-12)
    return rho


def high_freq_mass(phi: np.ndarray, rho: np.ndarray, threshold: float) -> float:
    mask = phi >= float(threshold)
    if int(np.sum(mask)) <= 1:
        return 0.0
    return float(trapz(rho[mask], phi[mask]))


def fit_line_from_alpha(alpha: float, hist: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    d = np.arange(1, len(hist), dtype=np.float64)
    y = np.asarray(hist[1:], dtype=np.float64)
    mask = np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 16:
        return None

    x = d[mask]
    yv = y[mask]
    c = float(np.exp(np.mean(np.log(yv) + float(alpha) * np.log(x))))
    y_fit = c * np.power(x, -float(alpha), dtype=np.float64)
    return x, y_fit


def ci_or_alpha(item: Dict[str, object], key: str) -> float:
    val = item.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return float(item["alpha"])


def make_plot(
    out_png: Path,
    out_pdf: Path,
    baseline_label: str,
    anchored_label: str,
    baseline_fit: Dict[str, object],
    anchored_fit: Dict[str, object],
    phi: np.ndarray,
    rho_pred_baseline: np.ndarray,
    rho_pred_anchored: np.ndarray,
    rho_obs_baseline: Optional[np.ndarray],
    rho_obs_anchored: Optional[np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    ax = axes[0]
    h0 = baseline_fit.get("hist")
    h1 = anchored_fit.get("hist")
    if isinstance(h0, np.ndarray) and isinstance(h1, np.ndarray):
        d0 = np.arange(1, len(h0), dtype=np.float64)
        d1 = np.arange(1, len(h1), dtype=np.float64)
        m0 = np.isfinite(h0[1:]) & (h0[1:] > 0)
        m1 = np.isfinite(h1[1:]) & (h1[1:] > 0)
        ax.plot(d0[m0], h0[1:][m0], color="#1f77b4", linewidth=1.6, label=f"{baseline_label} D_hat")
        ax.plot(d1[m1], h1[1:][m1], color="#d62728", linewidth=1.6, label=f"{anchored_label} D_hat")

        fit0 = fit_line_from_alpha(float(baseline_fit["alpha"]), h0)
        fit1 = fit_line_from_alpha(float(anchored_fit["alpha"]), h1)
        if fit0 is not None:
            ax.plot(fit0[0], fit0[1], "--", color="#1f77b4", alpha=0.8, label=f"{baseline_label} fit")
        if fit1 is not None:
            ax.plot(fit1[0], fit1[1], "--", color="#d62728", alpha=0.8, label=f"{anchored_label} fit")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance Δ")
        ax.set_ylabel("D_hat(Δ)")
    else:
        ax.text(
            0.5,
            0.5,
            "Histogram missing.\nRe-run run_attn_hist.py with --save_hist.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Empirical attention prior")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.array([0, 1], dtype=np.float64)
    alphas = np.array([float(baseline_fit["alpha"]), float(anchored_fit["alpha"])], dtype=np.float64)
    lo = np.array(
        [
            ci_or_alpha(baseline_fit, "alpha_ci_low"),
            ci_or_alpha(anchored_fit, "alpha_ci_low"),
        ],
        dtype=np.float64,
    )
    hi = np.array(
        [
            ci_or_alpha(baseline_fit, "alpha_ci_high"),
            ci_or_alpha(anchored_fit, "alpha_ci_high"),
        ],
        dtype=np.float64,
    )
    yerr = np.vstack((alphas - lo, hi - alphas))
    ax.errorbar(xs, alphas, yerr=yerr, fmt="o", capsize=4, linewidth=1.4, color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels([baseline_label, anchored_label], rotation=10)
    ax.set_ylabel("alpha")
    ax.set_title("Power-law fit with CI")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.plot(phi, rho_pred_baseline, color="#1f77b4", linewidth=1.7, label=f"{baseline_label} pred")
    ax.plot(phi, rho_pred_anchored, color="#d62728", linewidth=1.7, label=f"{anchored_label} pred")
    if rho_obs_baseline is not None:
        ax.plot(phi, rho_obs_baseline, "--", color="#1f77b4", linewidth=1.5, alpha=0.9, label=f"{baseline_label} obs")
    if rho_obs_anchored is not None:
        ax.plot(phi, rho_obs_anchored, "--", color="#d62728", linewidth=1.5, alpha=0.9, label=f"{anchored_label} obs")
    ax.set_xlabel("phi")
    ax.set_ylabel("rho(phi)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Predicted vs observed density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Empirical prior bridge: baseline vs anchored", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_fit = load_prior_fit(Path(args.baseline_prior_json))
    anchored_fit = load_prior_fit(Path(args.anchored_prior_json))

    phi, rho_pred_baseline, e_base = predict_density_from_alpha(
        alpha=float(baseline_fit["alpha"]),
        base=float(args.base),
        seq_len=int(args.seq_len),
        phi_points=int(args.phi_points),
        chunk_size=int(args.chunk_size),
    )
    _, rho_pred_anchored, e_anch = predict_density_from_alpha(
        alpha=float(anchored_fit["alpha"]),
        base=float(args.base),
        seq_len=int(args.seq_len),
        phi_points=int(args.phi_points),
        chunk_size=int(args.chunk_size),
    )

    inv_base = load_inv_freq(args.baseline_inv_freq)
    if inv_base is None:
        inv_base = geometric_inv_freq(head_dim=int(args.head_dim), base=float(args.base))
    inv_anch = load_inv_freq(args.anchored_inv_freq)

    rho_obs_baseline = density_from_inv_freq(inv_base, phi)
    rho_obs_anchored = density_from_inv_freq(inv_anch, phi) if inv_anch is not None else None

    hf_thr = float(args.hf_phi_threshold)
    hf_pred_base = high_freq_mass(phi, rho_pred_baseline, threshold=hf_thr)
    hf_pred_anch = high_freq_mass(phi, rho_pred_anchored, threshold=hf_thr)
    hf_obs_base = high_freq_mass(phi, rho_obs_baseline, threshold=hf_thr) if rho_obs_baseline is not None else None
    hf_obs_anch = high_freq_mass(phi, rho_obs_anchored, threshold=hf_thr) if rho_obs_anchored is not None else None

    pred_delta = float(hf_pred_anch - hf_pred_base)
    obs_delta = float(hf_obs_anch - hf_obs_base) if (hf_obs_anch is not None and hf_obs_base is not None) else None
    direction_consistent = None
    if obs_delta is not None:
        direction_consistent = bool(pred_delta * obs_delta >= 0.0)

    out_png = out_dir / "prior_fit_comparison.png"
    out_pdf = out_dir / "prior_fit_comparison.pdf"
    make_plot(
        out_png=out_png,
        out_pdf=out_pdf,
        baseline_label=args.baseline_label,
        anchored_label=args.anchored_label,
        baseline_fit=baseline_fit,
        anchored_fit=anchored_fit,
        phi=phi,
        rho_pred_baseline=rho_pred_baseline,
        rho_pred_anchored=rho_pred_anchored,
        rho_obs_baseline=rho_obs_baseline,
        rho_obs_anchored=rho_obs_anchored,
    )

    summary = {
        "meta": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_label": args.baseline_label,
            "anchored_label": args.anchored_label,
            "base": float(args.base),
            "seq_len": int(args.seq_len),
            "phi_points": int(args.phi_points),
            "chunk_size": int(args.chunk_size),
            "hf_phi_threshold": hf_thr,
        },
        "inputs": {
            "baseline_prior_json": str(Path(args.baseline_prior_json).resolve()),
            "anchored_prior_json": str(Path(args.anchored_prior_json).resolve()),
            "baseline_inv_freq": str(Path(args.baseline_inv_freq).resolve()) if args.baseline_inv_freq else "",
            "anchored_inv_freq": str(Path(args.anchored_inv_freq).resolve()) if args.anchored_inv_freq else "",
        },
        "alpha_fit": {
            args.baseline_label: {
                "alpha": baseline_fit["alpha"],
                "alpha_ci_low": baseline_fit["alpha_ci_low"],
                "alpha_ci_high": baseline_fit["alpha_ci_high"],
            },
            args.anchored_label: {
                "alpha": anchored_fit["alpha"],
                "alpha_ci_low": anchored_fit["alpha_ci_low"],
                "alpha_ci_high": anchored_fit["alpha_ci_high"],
            },
            "delta_alpha": float(anchored_fit["alpha"]) - float(baseline_fit["alpha"]),
        },
        "high_freq_mass": {
            "pred_baseline": hf_pred_base,
            "pred_anchored": hf_pred_anch,
            "pred_delta_anchored_minus_baseline": pred_delta,
            "obs_baseline": hf_obs_base,
            "obs_anchored": hf_obs_anch,
            "obs_delta_anchored_minus_baseline": obs_delta,
            "direction_consistent": direction_consistent,
        },
        "diagnostics": {
            "e_diag_baseline_mean": float(np.mean(e_base)),
            "e_diag_anchored_mean": float(np.mean(e_anch)),
        },
        "artifacts": {
            "png": str(out_png.resolve()),
            "pdf": str(out_pdf.resolve()),
            "json": str((out_dir / "prior_fit_comparison.json").resolve()),
        },
    }

    out_json = out_dir / "prior_fit_comparison.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
