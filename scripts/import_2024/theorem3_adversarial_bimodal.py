#!/usr/bin/env python3
"""
Adversarial bimodal prior scan for Theorem 3 fragility.

Scans over:
- mode gap (distance between two modes)
- mass ratio (long-range mass / short-range mass)
- width ratio (mode sharpness)

Outputs fragility maps based on local TV sensitivity of rho*(phi).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_grid(text: str) -> List[float]:
    return [float(x) for x in parse_csv(text)]


def build_bimodal_prior(
    L: int,
    mode_gap_ratio: float,
    mass_ratio: float,
    width_ratio: float,
) -> np.ndarray:
    d = np.arange(1, L + 1, dtype=np.float64)

    mu1 = max(2.0, 0.08 * float(L))
    mu2 = min(float(L), max(mu1 + 4.0, float(mode_gap_ratio) * float(L)))
    sigma = max(4.0, float(width_ratio) * float(L))

    g1 = np.exp(-0.5 * ((d - mu1) / sigma) ** 2)
    g2 = np.exp(-0.5 * ((d - mu2) / sigma) ** 2)

    m2 = max(1e-8, float(mass_ratio))
    m1 = 1.0
    w = m1 * g1 + m2 * g2
    w = np.maximum(w, 1e-16)
    w /= float(np.sum(w))
    return w


def compute_kernel(
    b: float,
    L: int,
    prior: np.ndarray,
    phi_points: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 1.0, int(phi_points), dtype=np.float64)
    omega = np.power(float(b), -phi, dtype=np.float64)
    m = int(phi_points)
    K = np.zeros((m, m), dtype=np.float64)

    for start in range(1, L + 1, int(chunk_size)):
        end = min(L, start + int(chunk_size) - 1)
        delta = np.arange(start, end + 1, dtype=np.float64)
        c = np.cos(np.outer(omega, delta))
        w = prior[start - 1 : end]
        K += (c * w[None, :]) @ c.T

    return phi, K


def diag_density(K: np.ndarray) -> np.ndarray:
    d = np.diag(K).astype(np.float64)
    d = np.maximum(d, 1e-12)
    rho = 1.0 / d
    rho = np.maximum(rho, 1e-12)
    rho /= float(np.sum(rho))
    return rho


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(a - b)))


def local_fragility(
    rho_map: Dict[Tuple[int, int], np.ndarray],
    i_gap: int,
    i_mass: int,
    n_gap: int,
    n_mass: int,
) -> float:
    center = rho_map[(i_gap, i_mass)]
    tvs: List[float] = []
    neigh = [
        (i_gap - 1, i_mass),
        (i_gap + 1, i_mass),
        (i_gap, i_mass - 1),
        (i_gap, i_mass + 1),
    ]
    for ig, im in neigh:
        if 0 <= ig < n_gap and 0 <= im < n_mass and (ig, im) in rho_map:
            tvs.append(tv_distance(center, rho_map[(ig, im)]))
    if not tvs:
        return float("nan")
    return float(np.mean(tvs))


def plot_fragility_heatmaps(
    maps: Dict[float, np.ndarray],
    mode_gap_grid: List[float],
    mass_ratio_grid: List[float],
    out_png: Path,
    out_pdf: Path,
) -> None:
    widths = sorted(maps.keys())
    ncol = len(widths)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.8), squeeze=False)

    for j, wr in enumerate(widths):
        ax = axes[0, j]
        mat = maps[wr]
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(f"width_ratio={wr:.3f}")
        ax.set_xlabel("mode_gap_ratio")
        if j == 0:
            ax.set_ylabel("mass_ratio")
        ax.set_xticks(np.arange(len(mode_gap_grid)))
        ax.set_xticklabels([f"{x:.2f}" for x in mode_gap_grid], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(mass_ratio_grid)))
        ax.set_yticklabels([f"{x:.2f}" for x in mass_ratio_grid])

        for i in range(len(mass_ratio_grid)):
            for k in range(len(mode_gap_grid)):
                v = mat[i, k]
                if np.isfinite(v):
                    ax.text(k, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fragility_tv")

    fig.suptitle("Theorem-3 adversarial bimodal fragility map", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adversarial bimodal fragility scan for Theorem 3")
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--b", type=float, default=500000.0)
    ap.add_argument("--phi_points", type=int, default=96)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--mode_gap_grid", type=str, default="0.25,0.40,0.55,0.70,0.85")
    ap.add_argument("--mass_ratio_grid", type=str, default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--width_ratio_grid", type=str, default="0.01,0.02,0.04")
    ap.add_argument("--out_dir", type=str, default="artifacts/reviewer_2026-02-24/theorem3_fragility")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mode_gap_grid = parse_float_grid(args.mode_gap_grid)
    mass_ratio_grid = parse_float_grid(args.mass_ratio_grid)
    width_ratio_grid = parse_float_grid(args.width_ratio_grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    maps: Dict[float, np.ndarray] = {}

    for wr in width_ratio_grid:
        rho_map: Dict[Tuple[int, int], np.ndarray] = {}
        for ig, gap in enumerate(mode_gap_grid):
            for im, mr in enumerate(mass_ratio_grid):
                prior = build_bimodal_prior(
                    L=int(args.L),
                    mode_gap_ratio=float(gap),
                    mass_ratio=float(mr),
                    width_ratio=float(wr),
                )
                _, K = compute_kernel(
                    b=float(args.b),
                    L=int(args.L),
                    prior=prior,
                    phi_points=int(args.phi_points),
                    chunk_size=int(args.chunk_size),
                )
                rho_map[(ig, im)] = diag_density(K)

        frag = np.full((len(mass_ratio_grid), len(mode_gap_grid)), np.nan, dtype=np.float64)
        for ig, gap in enumerate(mode_gap_grid):
            for im, mr in enumerate(mass_ratio_grid):
                f = local_fragility(
                    rho_map=rho_map,
                    i_gap=ig,
                    i_mass=im,
                    n_gap=len(mode_gap_grid),
                    n_mass=len(mass_ratio_grid),
                )
                frag[im, ig] = f
                row = {
                    "width_ratio": float(wr),
                    "mode_gap_ratio": float(gap),
                    "mass_ratio": float(mr),
                    "fragility_tv": float(f),
                }
                rows.append(row)
                print(
                    f"[fragility] width={wr:.3f} gap={gap:.2f} mass={mr:.2f} tv={f:.4f}",
                    flush=True,
                )

        maps[float(wr)] = frag

    summary = {
        "meta": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "L": int(args.L),
            "b": float(args.b),
            "phi_points": int(args.phi_points),
            "chunk_size": int(args.chunk_size),
            "mode_gap_grid": mode_gap_grid,
            "mass_ratio_grid": mass_ratio_grid,
            "width_ratio_grid": width_ratio_grid,
        },
        "rows": rows,
        "artifacts": {
            "json": str(out_dir / "theorem3_fragility_map.json"),
            "png": str(out_dir / "theorem3_fragility_map.png"),
            "pdf": str(out_dir / "theorem3_fragility_map.pdf"),
        },
    }

    out_json = out_dir / "theorem3_fragility_map.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_fragility_heatmaps(
        maps=maps,
        mode_gap_grid=mode_gap_grid,
        mass_ratio_grid=mass_ratio_grid,
        out_png=out_dir / "theorem3_fragility_map.png",
        out_pdf=out_dir / "theorem3_fragility_map.pdf",
    )

    print(f"[ok] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
