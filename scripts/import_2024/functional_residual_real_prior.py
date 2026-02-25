#!/usr/bin/env python3
"""
Quantify diagonal approximation vs full-functional optimum under empirical priors.

For each (b, L, prior_real):
- diagonal residual: ||K - diag(diag(K))||_F / ||K||_F
- objective gap: J_full[rho_diag] - J_full[rho_full]
- TV distance: 0.5 * ||rho_diag - rho_full||_1

`rho_full` is optimized on simplex by mirror descent with entropy regularization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_grid(text: str) -> List[float]:
    return [float(x) for x in parse_csv(text)]


def parse_int_grid(text: str) -> List[int]:
    return [int(x) for x in parse_csv(text)]


def load_real_prior_hist(prior_json: Path) -> np.ndarray:
    obj = json.loads(prior_json.read_text(encoding="utf-8"))
    hist = obj.get("overall_hist")
    if not isinstance(hist, list) or len(hist) < 8:
        raise RuntimeError(f"Missing usable `overall_hist` in {prior_json}")
    arr = np.asarray(hist, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    s = float(np.sum(arr))
    if s <= 0:
        raise RuntimeError(f"Invalid histogram normalization in {prior_json}")
    arr = arr / s
    return arr


def prior_for_L(hist: np.ndarray, L: int) -> np.ndarray:
    # hist[d] corresponds to distance d, including d=0; we need d in [1..L]
    if len(hist) <= 2:
        raise RuntimeError("Histogram too short")

    src_x = np.arange(1, len(hist), dtype=np.float64)
    src_y = np.asarray(hist[1:], dtype=np.float64)
    src_y = np.maximum(src_y, 0.0)

    tgt_x = np.arange(1, L + 1, dtype=np.float64)
    if int(tgt_x[-1]) <= int(src_x[-1]):
        y = src_y[:L].copy()
    else:
        y = np.interp(tgt_x, src_x, src_y, left=src_y[0], right=src_y[-1])

    y = np.maximum(y, 1e-16)
    y /= float(np.sum(y))
    return y


def compute_kernel_chunked(
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


def objective_full(K: np.ndarray, rho: np.ndarray, entropy_reg: float) -> float:
    quad = float(rho @ (K @ rho))
    if entropy_reg <= 0:
        return quad
    ent = float(np.sum(rho * np.log(np.clip(rho, 1e-12, None))))
    return quad + float(entropy_reg) * ent


def optimize_full_density(
    K: np.ndarray,
    steps: int,
    lr: float,
    entropy_reg: float,
) -> np.ndarray:
    m = K.shape[0]
    rho = np.full(m, 1.0 / m, dtype=np.float64)

    for _ in range(int(steps)):
        grad = 2.0 * (K @ rho)
        if entropy_reg > 0:
            grad = grad + float(entropy_reg) * (1.0 + np.log(np.clip(rho, 1e-12, None)))

        grad = grad - float(np.mean(grad))
        rho = rho * np.exp(-float(lr) * grad)
        rho = np.maximum(rho, 1e-18)
        rho /= float(np.sum(rho))

    return rho


def diag_residual_fro(K: np.ndarray) -> float:
    diag_only = np.diag(np.diag(K))
    num = float(np.linalg.norm(K - diag_only, ord="fro"))
    den = float(np.linalg.norm(K, ord="fro"))
    return num / max(den, 1e-12)


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(a - b)))


def plot_heatmap(
    rows: List[Dict[str, float]],
    b_grid: List[float],
    L_grid: List[int],
    key: str,
    out_paths: List[Path],
    title: str,
) -> None:
    mat = np.full((len(L_grid), len(b_grid)), np.nan, dtype=np.float64)
    for iL, L in enumerate(L_grid):
        for ib, b in enumerate(b_grid):
            hit = next((r for r in rows if int(r["L"]) == int(L) and float(r["b"]) == float(b)), None)
            if hit is not None:
                mat[iL, ib] = float(hit[key])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("b")
    ax.set_ylabel("L")
    ax.set_xticks(np.arange(len(b_grid)))
    ax.set_xticklabels([f"{b:g}" for b in b_grid], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(L_grid)))
    ax.set_yticklabels([str(L) for L in L_grid])

    for i in range(len(L_grid)):
        for j in range(len(b_grid)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=key)
    fig.tight_layout()
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute full-functional residual gap under real prior")
    ap.add_argument("--prior_json", type=str, required=True, help="run_attn_hist prior fit json with overall_hist")
    ap.add_argument("--b_grid", type=str, default="1e3,1e4,1e5,5e5")
    ap.add_argument("--L_grid", type=str, default="4096,8192,16384,32768")
    ap.add_argument("--phi_points", type=int, default=96)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--mirror_steps", type=int, default=700)
    ap.add_argument("--mirror_lr", type=float, default=0.8)
    ap.add_argument("--entropy_reg", type=float, default=1e-4)
    ap.add_argument("--gap_threshold", type=float, default=0.002)
    ap.add_argument("--tv_threshold", type=float, default=0.12)
    ap.add_argument("--out_dir", type=str, default="artifacts/reviewer_2026-02-24/functional_residual_real_prior")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    b_grid = parse_float_grid(args.b_grid)
    L_grid = parse_int_grid(args.L_grid)

    prior_hist = load_real_prior_hist(Path(args.prior_json))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for L in L_grid:
        prior = prior_for_L(prior_hist, int(L))
        for b in b_grid:
            _, K = compute_kernel_chunked(
                b=float(b),
                L=int(L),
                prior=prior,
                phi_points=int(args.phi_points),
                chunk_size=int(args.chunk_size),
            )
            rho_d = diag_density(K)
            rho_f = optimize_full_density(
                K=K,
                steps=int(args.mirror_steps),
                lr=float(args.mirror_lr),
                entropy_reg=float(args.entropy_reg),
            )

            j_diag = objective_full(K, rho_d, entropy_reg=float(args.entropy_reg))
            j_full = objective_full(K, rho_f, entropy_reg=float(args.entropy_reg))
            row = {
                "b": float(b),
                "L": int(L),
                "diag_residual_fro": float(diag_residual_fro(K)),
                "objective_diag": float(j_diag),
                "objective_full": float(j_full),
                "objective_gap": float(j_diag - j_full),
                "tv_rho_diag_full": float(tv_distance(rho_d, rho_f)),
            }
            rows.append(row)
            print(
                f"[real-prior] L={L:6d} b={b:9g} residual={row['diag_residual_fro']:.4f} "
                f"gap={row['objective_gap']:.5f} tv={row['tv_rho_diag_full']:.4f}",
                flush=True,
            )

    csv_path = out_dir / "functional_residual_real_prior.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "b",
                "L",
                "diag_residual_fro",
                "objective_diag",
                "objective_full",
                "objective_gap",
                "tv_rho_diag_full",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    recommended = [
        r
        for r in rows
        if float(r["objective_gap"]) <= float(args.gap_threshold)
        and float(r["tv_rho_diag_full"]) <= float(args.tv_threshold)
    ]

    summary = {
        "meta": {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "prior_json": str(Path(args.prior_json).resolve()),
            "b_grid": b_grid,
            "L_grid": L_grid,
            "phi_points": int(args.phi_points),
            "chunk_size": int(args.chunk_size),
            "mirror_steps": int(args.mirror_steps),
            "mirror_lr": float(args.mirror_lr),
            "entropy_reg": float(args.entropy_reg),
            "gap_threshold": float(args.gap_threshold),
            "tv_threshold": float(args.tv_threshold),
        },
        "rows": rows,
        "recommended_region": recommended,
        "artifacts": {
            "csv": str(csv_path),
            "gap_png": str(out_dir / "functional_objective_gap.png"),
            "tv_png": str(out_dir / "functional_tv_gap.png"),
            "gap_pdf": str(out_dir / "functional_objective_gap.pdf"),
            "tv_pdf": str(out_dir / "functional_tv_gap.pdf"),
            "recommended_md": str(out_dir / "recommended_domain.md"),
        },
    }

    json_path = out_dir / "functional_residual_real_prior.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_heatmap(
        rows=rows,
        b_grid=b_grid,
        L_grid=L_grid,
        key="objective_gap",
        out_paths=[out_dir / "functional_objective_gap.png", out_dir / "functional_objective_gap.pdf"],
        title="J_full[rho_diag] - J_full[rho_full]",
    )
    plot_heatmap(
        rows=rows,
        b_grid=b_grid,
        L_grid=L_grid,
        key="tv_rho_diag_full",
        out_paths=[out_dir / "functional_tv_gap.png", out_dir / "functional_tv_gap.pdf"],
        title="TV(rho_diag, rho_full)",
    )

    md_lines = [
        "# Functional Residual Recommended Domain",
        "",
        f"Thresholds: `objective_gap <= {float(args.gap_threshold):.6f}` and `TV <= {float(args.tv_threshold):.6f}`",
        "",
        "| L | b | diag_residual_fro | objective_gap | tv_rho_diag_full |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in recommended:
        md_lines.append(
            f"| {int(r['L'])} | {float(r['b']):g} | {float(r['diag_residual_fro']):.4f} | {float(r['objective_gap']):.6f} | {float(r['tv_rho_diag_full']):.4f} |"
        )
    (out_dir / "recommended_domain.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
