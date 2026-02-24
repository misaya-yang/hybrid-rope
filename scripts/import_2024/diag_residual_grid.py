#!/usr/bin/env python3
"""
Build diagonal-surrogate residual grids across (b, L, prior).

Residual definition:
    residual_fro = ||K - diag(diag(K))||_F / ||K||_F

where
    K(phi_i, phi_j) = sum_Δ D(Δ) cos(w_i Δ) cos(w_j Δ), w_i = b^{-phi_i}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for tok in parse_csv(text):
        vals.append(float(tok))
    return vals


def parse_int_grid(text: str) -> List[int]:
    vals = []
    for tok in parse_csv(text):
        vals.append(int(tok))
    return vals


def build_prior(L: int, prior_family: str, sigma_ratio: float) -> np.ndarray:
    delta = np.arange(1, L + 1, dtype=np.float64)
    prior = prior_family.strip().lower()
    if prior == "uniform":
        w = np.ones(L, dtype=np.float64)
    elif prior == "powerlaw":
        w = 1.0 / np.maximum(delta, 1.0)
    elif prior == "bimodal":
        sigma = max(8.0, float(L) * float(sigma_ratio))
        g1 = np.exp(-((delta - 1.0) ** 2) / (2.0 * sigma * sigma))
        g2 = np.exp(-((delta - float(L)) ** 2) / (2.0 * sigma * sigma))
        w = 0.5 * g1 + 0.5 * g2
    else:
        raise ValueError(f"Unsupported prior_family={prior_family}")

    z = float(np.sum(w))
    if z <= 0:
        raise RuntimeError(f"Invalid prior normalization for prior={prior_family}, L={L}")
    return w / z


def compute_kernel_chunked(
    b: float,
    L: int,
    prior: np.ndarray,
    phi_points: int,
    chunk_size: int,
) -> np.ndarray:
    phi = np.linspace(0.0, 1.0, int(phi_points), dtype=np.float64)
    omega = np.power(float(b), -phi, dtype=np.float64)
    m = int(phi_points)
    K = np.zeros((m, m), dtype=np.float64)

    for start in range(1, L + 1, int(chunk_size)):
        end = min(L, start + int(chunk_size) - 1)
        delta = np.arange(start, end + 1, dtype=np.float64)
        c = np.cos(np.outer(omega, delta))  # [M, chunk]
        w = prior[start - 1 : end]          # [chunk]
        K += (c * w[None, :]) @ c.T

    return K


def diag_residual_metrics(K: np.ndarray) -> Dict[str, float]:
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape={K.shape}")

    diag_only = np.diag(np.diag(K))
    num = float(np.linalg.norm(K - diag_only, ord="fro"))
    den = float(np.linalg.norm(K, ord="fro"))
    residual_fro = num / max(den, 1e-12)

    abs_total = float(np.sum(np.abs(K)))
    abs_diag = float(np.sum(np.abs(np.diag(K))))
    diag_dominance = abs_diag / max(abs_total, 1e-12)

    return {
        "residual_fro": float(residual_fro),
        "diag_dominance": float(diag_dominance),
    }


def plot_heatmaps(
    rows: List[Dict[str, float]],
    priors: List[str],
    b_grid: List[float],
    L_grid: List[int],
    value_key: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    ncol = len(priors)
    fig, axes = plt.subplots(1, ncol, figsize=(4.3 * ncol, 4.8), squeeze=False)

    for j, prior in enumerate(priors):
        ax = axes[0, j]
        mat = np.full((len(L_grid), len(b_grid)), np.nan, dtype=np.float64)
        for iL, L in enumerate(L_grid):
            for ib, b in enumerate(b_grid):
                hit = next(
                    (
                        r for r in rows
                        if r["prior_family"] == prior and int(r["L"]) == int(L) and float(r["b"]) == float(b)
                    ),
                    None,
                )
                if hit is not None:
                    mat[iL, ib] = float(hit[value_key])

        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(f"prior={prior}")
        ax.set_xticks(np.arange(len(b_grid)))
        ax.set_xticklabels([f"{b:g}" for b in b_grid], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(L_grid)))
        ax.set_yticklabels([str(L) for L in L_grid])
        ax.set_xlabel("b")
        if j == 0:
            ax.set_ylabel("L")

        for i in range(len(L_grid)):
            for k in range(len(b_grid)):
                v = mat[i, k]
                if np.isfinite(v):
                    ax.text(k, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_key)

    fig.suptitle("Diagonal residual grid", y=0.99)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute diagonal residual grid over (b, L, prior)")
    ap.add_argument("--b_grid", type=str, default="1e3,1e4,1e5,5e5,1e6")
    ap.add_argument("--L_grid", type=str, default="4096,8192,16384,32768,65536")
    ap.add_argument("--prior_family", type=str, default="uniform,powerlaw,bimodal")
    ap.add_argument("--phi_points", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--bimodal_sigma_ratio", type=float, default=0.02)
    ap.add_argument("--residual_threshold", type=float, default=0.15)
    ap.add_argument("--out_dir", type=str, default="artifacts/reviewer_2026-02-24/diag_residual_grid")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    b_grid = parse_float_grid(args.b_grid)
    L_grid = parse_int_grid(args.L_grid)
    priors = parse_csv(args.prior_family)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for prior in priors:
        for L in L_grid:
            prior_w = build_prior(L=L, prior_family=prior, sigma_ratio=args.bimodal_sigma_ratio)
            for b in b_grid:
                K = compute_kernel_chunked(
                    b=float(b),
                    L=int(L),
                    prior=prior_w,
                    phi_points=int(args.phi_points),
                    chunk_size=int(args.chunk_size),
                )
                met = diag_residual_metrics(K)
                rows.append(
                    {
                        "prior_family": prior,
                        "L": int(L),
                        "b": float(b),
                        "phi_points": int(args.phi_points),
                        "residual_fro": met["residual_fro"],
                        "diag_dominance": met["diag_dominance"],
                    }
                )
                print(
                    f"[grid] prior={prior:8s} L={L:6d} b={b:9g} "
                    f"residual_fro={met['residual_fro']:.4f} diag_dom={met['diag_dominance']:.4f}",
                    flush=True,
                )

    csv_path = out_dir / "diag_residual_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prior_family", "L", "b", "phi_points", "residual_fro", "diag_dominance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    recommended = []
    threshold = float(args.residual_threshold)
    for row in rows:
        if float(row["residual_fro"]) <= threshold:
            recommended.append({
                "prior_family": row["prior_family"],
                "L": int(row["L"]),
                "b": float(row["b"]),
                "residual_fro": float(row["residual_fro"]),
            })

    summary = {
        "meta": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "b_grid": b_grid,
            "L_grid": L_grid,
            "prior_family": priors,
            "phi_points": int(args.phi_points),
            "chunk_size": int(args.chunk_size),
            "residual_threshold": threshold,
        },
        "rows": rows,
        "recommended_region": recommended,
        "artifacts": {
            "csv": str(csv_path),
            "png": str(out_dir / "diag_residual_grid.png"),
            "pdf": str(out_dir / "diag_residual_grid.pdf"),
            "recommended_md": str(out_dir / "recommended_domain.md"),
        },
    }

    json_path = out_dir / "diag_residual_grid.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Diagonal Residual Recommended Domain",
        "",
        f"Residual threshold: `residual_fro <= {threshold:.3f}`",
        "",
        "| prior | L | b | residual_fro |",
        "|---|---:|---:|---:|",
    ]
    for x in recommended:
        md_lines.append(f"| {x['prior_family']} | {x['L']} | {x['b']:.0f} | {x['residual_fro']:.4f} |")
    if not recommended:
        md_lines.append("| (none) | - | - | - |")

    md_path = out_dir / "recommended_domain.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_heatmaps(
        rows=rows,
        priors=priors,
        b_grid=b_grid,
        L_grid=L_grid,
        value_key="residual_fro",
        out_png=out_dir / "diag_residual_grid.png",
        out_pdf=out_dir / "diag_residual_grid.pdf",
    )

    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")


if __name__ == "__main__":
    main()
