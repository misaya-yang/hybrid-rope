#!/usr/bin/env python3
"""
Phase-transition scan for mixed priors:
    D_p(delta) = p * D_powerlaw(delta) + (1-p) * D_uniform(delta)

Goal:
    Quantify where different RoPE frequency shapes become optimal as p changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = PROJECT_ROOT / "sigmoid_rope_experiments"
if str(PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE_ROOT))

from src.rope import RoPEFrequencyAllocator  # noqa: E402
from src.visualization import save_fig_both, set_plot_style  # noqa: E402


def geometric_inv_freq(head_dim: int, base: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    k = head_dim // 2
    idx = torch.arange(k, dtype=dtype)
    return 1.0 / (float(base) ** (2.0 * idx / float(head_dim)))


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def get_custom_inv_freq(
    method: str,
    head_dim: int,
    base: float,
    max_seq_len: int,
    rigid_j0: int = 12,
) -> torch.Tensor:
    method = method.lower()
    base_inv = geometric_inv_freq(head_dim=head_dim, base=base, dtype=torch.float64)
    if method == "baseline":
        return base_inv

    scale = max(float(max_seq_len) / 8192.0, 1.0)
    if method == "pi":
        return base_inv / scale

    if method == "yarn":
        k = head_dim // 2
        idx = torch.arange(k, dtype=torch.float64)
        start = int(0.20 * k)
        end = int(0.90 * k)
        if end <= start:
            end = min(k - 1, start + 1)
        ramp = (idx - start) / float(max(1, end - start))
        ramp = torch.clamp(ramp, 0.0, 1.0)
        ramp = smoothstep(ramp)
        temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
        yarn_scale = (scale**ramp) * (temperature ** (0.5 * ramp))
        return base_inv / yarn_scale

    if method == "anchored_hybrid":
        k = head_dim // 2
        rigid_j0 = min(max(0, int(rigid_j0)), k)
        if scale <= 1.0:
            return base_inv

        tail_base = float(base) * (scale**2)
        tail_base = max(tail_base, float(base) * 4.0)
        tail_inv = geometric_inv_freq(head_dim=head_dim, base=tail_base, dtype=torch.float64)
        out = base_inv.clone()
        if rigid_j0 < k:
            t = torch.arange(k - rigid_j0, dtype=torch.float64)
            if t.numel() == 1:
                ramp = torch.ones_like(t)
            else:
                t = t / float(t.numel() - 1)
                ramp = 0.5 - 0.5 * torch.cos(math.pi * t)
            alpha = min(0.40, max(0.08, 0.16 * math.log2(scale)))
            blend = alpha * ramp
            out[rigid_j0:] = (1.0 - blend) * base_inv[rigid_j0:] + blend * tail_inv[rigid_j0:]
        out[:rigid_j0] = base_inv[:rigid_j0]
        return out

    raise ValueError(f"Unknown method: {method}")


def compute_s_delta(freqs: torch.Tensor, L: int, device: torch.device, chunk_size: int = 8192) -> np.ndarray:
    freqs_dev = freqs.to(device=device, dtype=torch.float64)
    vals: List[np.ndarray] = []
    for start in range(1, L + 1, chunk_size):
        end = min(L + 1, start + chunk_size)
        delta = torch.arange(start, end, device=device, dtype=torch.float64)
        s = torch.cos(delta[:, None] * freqs_dev[None, :]).mean(dim=1)
        vals.append(s.detach().cpu().numpy())
    return np.concatenate(vals, axis=0)


def prior_uniform(L: int) -> np.ndarray:
    p = np.ones(L, dtype=np.float64)
    p /= p.sum()
    return p


def prior_powerlaw(L: int, gamma: float) -> np.ndarray:
    d = np.arange(1, L + 1, dtype=np.float64)
    p = np.power(d, -gamma)
    p /= p.sum()
    return p


def find_crossing_x(x: np.ndarray, y: np.ndarray) -> List[float]:
    out: List[float] = []
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if y0 == 0.0:
            out.append(float(x[i]))
            continue
        if y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)
            out.append(float(x[i] + t * (x[i + 1] - x[i])))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate phase-transition scan data/figures.")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--base", type=float, default=10000.0)
    ap.add_argument("--L", type=int, default=131072)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--p_points", type=int, default=41, help="number of points in p in [0,1]")
    ap.add_argument("--rigid_j0", type=int, default=12)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--chunk_size", type=int, default=8192)
    ap.add_argument("--target_cross_p", type=float, default=0.5)
    ap.add_argument("--scan_k_min", type=float, default=8.0)
    ap.add_argument("--scan_k_max", type=float, default=30.0)
    ap.add_argument("--scan_k_points", type=int, default=19)
    ap.add_argument("--enable_sigmoid_k_scan", action="store_true")
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
    print(f"[phase-transition] device={device}")

    allocator = RoPEFrequencyAllocator(d=args.head_dim, base=args.base)
    method_freqs: Dict[str, torch.Tensor] = {
        "baseline": get_custom_inv_freq("baseline", args.head_dim, args.base, args.L, args.rigid_j0),
        "pi": get_custom_inv_freq("pi", args.head_dim, args.base, args.L, args.rigid_j0),
        "yarn": get_custom_inv_freq("yarn", args.head_dim, args.base, args.L, args.rigid_j0),
        "anchored_hybrid": get_custom_inv_freq("anchored_hybrid", args.head_dim, args.base, args.L, args.rigid_j0),
        # add one smooth convex family point for reference
        "sigmoid_formula": allocator.sigmoid(k=16.05 / args.head_dim, x0=0.47 * (args.head_dim // 2)),
    }

    print("[phase-transition] computing S(delta) for methods ...")
    s2: Dict[str, np.ndarray] = {}
    for name, freqs in method_freqs.items():
        s = compute_s_delta(freqs=freqs, L=args.L, device=device, chunk_size=args.chunk_size)
        s2[name] = np.square(s, dtype=np.float64)
        print(f"  - {name}: done")

    d_uni = prior_uniform(args.L)
    d_pow = prior_powerlaw(args.L, gamma=args.gamma)
    p_grid = np.linspace(0.0, 1.0, args.p_points, dtype=np.float64)

    rows: List[Dict[str, float | str]] = []
    score_map: Dict[str, List[float]] = {k: [] for k in method_freqs.keys()}
    for p in p_grid:
        d_mix = p * d_pow + (1.0 - p) * d_uni
        for name in method_freqs.keys():
            score = float(np.dot(d_mix, s2[name]))
            score_map[name].append(score)
            rows.append({"p": float(p), "method": name, "collision_score": score})

    df = pd.DataFrame(rows)
    csv_path = data_dir / "phase_transition_scan.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    scan_best = None
    if args.enable_sigmoid_k_scan:
        print("[phase-transition] scanning sigmoid k for target crossing ...")
        base_scores = np.asarray(score_map["baseline"], dtype=np.float64)
        k_values = np.linspace(args.scan_k_min / args.head_dim, args.scan_k_max / args.head_dim, args.scan_k_points)
        x0 = 0.47 * (args.head_dim // 2)
        for k_val in k_values:
            f_scan = allocator.sigmoid(k=float(k_val), x0=float(x0))
            s2_scan = np.square(compute_s_delta(freqs=f_scan, L=args.L, device=device, chunk_size=args.chunk_size), dtype=np.float64)
            score_scan = []
            for p in p_grid:
                d_mix = p * d_pow + (1.0 - p) * d_uni
                score_scan.append(float(np.dot(d_mix, s2_scan)))
            score_scan = np.asarray(score_scan, dtype=np.float64)
            diff = score_scan - base_scores
            cands = find_crossing_x(p_grid, diff)
            if not cands:
                continue
            c = cands[0]
            item = {
                "k": float(k_val),
                "x0": float(x0),
                "crossing": float(c),
                "distance_to_target": float(abs(c - args.target_cross_p)),
                "score_p0": float(score_scan[0]),
                "score_p1": float(score_scan[-1]),
            }
            if scan_best is None or item["distance_to_target"] < scan_best["distance_to_target"]:
                scan_best = item
                score_map["sigmoid_tuned"] = score_scan.tolist()

    diff_hybrid_std = np.asarray(score_map["anchored_hybrid"]) - np.asarray(score_map["baseline"])
    crossings = find_crossing_x(p_grid, diff_hybrid_std)
    diff_sigmoid_std = np.asarray(score_map["sigmoid_formula"]) - np.asarray(score_map["baseline"])
    crossings_sigmoid = find_crossing_x(p_grid, diff_sigmoid_std)
    crossings_tuned = []
    if "sigmoid_tuned" in score_map:
        diff_tuned = np.asarray(score_map["sigmoid_tuned"]) - np.asarray(score_map["baseline"])
        crossings_tuned = find_crossing_x(p_grid, diff_tuned)

    summary = {
        "head_dim": int(args.head_dim),
        "base": float(args.base),
        "L": int(args.L),
        "gamma": float(args.gamma),
        "p_grid_points": int(args.p_points),
        "methods": list(score_map.keys()),
        "hybrid_minus_baseline_crossings": crossings,
        "sigmoid_formula_minus_baseline_crossings": crossings_sigmoid,
        "sigmoid_tuned_minus_baseline_crossings": crossings_tuned,
        "sigmoid_scan_best": scan_best,
        "score_at_p0": {k: float(v[0]) for k, v in score_map.items()},
        "score_at_p1": {k: float(v[-1]) for k, v in score_map.items()},
    }
    json_path = data_dir / "phase_transition_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    set_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    color_map = {
        "baseline": "#d62728",
        "anchored_hybrid": "#1f77b4",
        "sigmoid_formula": "#2ca02c",
        "sigmoid_tuned": "#17becf",
        "pi": "#8c564b",
        "yarn": "#9467bd",
    }
    for name, vals in score_map.items():
        ax.plot(p_grid, vals, label=name, color=color_map.get(name, None))

    for x in crossings:
        ax.axvline(x, linestyle="--", color="#444444", alpha=0.6)
        ax.text(x, ax.get_ylim()[0], f"p*={x:.2f}", rotation=90, va="bottom", ha="right", fontsize=9)
    for x in crossings_tuned:
        ax.axvline(x, linestyle=":", color="#17becf", alpha=0.8)
        ax.text(x, ax.get_ylim()[1], f"tuned p*={x:.2f}", rotation=90, va="top", ha="left", fontsize=9, color="#17becf")

    ax.set_xlabel("Mixture ratio p (power-law weight)")
    ax.set_ylabel("Phase Collision Score (lower is better)")
    ax.set_title("Phase Transition Under Mixed Priors")
    ax.legend(loc="best")
    pdf_path, png_path = save_fig_both(fig, out_dir / "phase_transition_scan")
    plt.close(fig)

    print("[phase-transition] done")
    print(f"  csv: {csv_path}")
    print(f"  summary: {json_path}")
    print(f"  fig: {pdf_path}")
    print(f"  fig: {png_path}")
    if crossings:
        print(f"  crossings (anchored_hybrid vs baseline): {crossings}")
    else:
        print("  crossings (anchored_hybrid vs baseline): none")
    if crossings_sigmoid:
        print(f"  crossings (sigmoid_formula vs baseline): {crossings_sigmoid}")
    if crossings_tuned:
        print(f"  crossings (sigmoid_tuned vs baseline): {crossings_tuned}")
    if scan_best is not None:
        print(
            "  best sigmoid scan:",
            f"k={scan_best['k']:.6f}, x0={scan_best['x0']:.2f}, ",
            f"crossing={scan_best['crossing']:.4f}, |cross-target|={scan_best['distance_to_target']:.4f}",
        )


if __name__ == "__main__":
    main()
