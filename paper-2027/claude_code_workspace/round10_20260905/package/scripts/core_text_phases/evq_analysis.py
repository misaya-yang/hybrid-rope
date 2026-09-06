#!/usr/bin/env python3
"""EVQ τ-sweep analysis: publication-quality figures for NeurIPS v5.

Usage:
    python scripts/core_text_phases/evq_analysis.py --input results_final.json
    python scripts/core_text_phases/evq_analysis.py --input results_final.json --out_dir paper_draft/figs/

Generates:
  1. fig_ppl_vs_tau.pdf        — PPL vs τ curves (sub-plots per eval length)
  2. fig_waterbed.pdf          — Waterbed trade-off scatter
  3. fig_phase_collision.pdf   — Phase collision heatmap
  4. fig_freq_distribution.pdf — Frequency allocation curves per τ
  5. summary_table.csv         — Tabular summary for LaTeX
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── NeurIPS style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
})

# Colors
C_GEO = "#2166ac"       # blue for τ=0 (Geometric baseline)
C_EVQ = "#d6604d"       # red for EVQ
C_MID = "#4daf4a"       # green for intermediate τ
C_LIGHT_BG = "#f7f7f7"

# Color palette for τ sweep (8 values max)
TAU_CMAP = matplotlib.colormaps.get_cmap("coolwarm").resampled(10)

def tau_color(tau: float, tau_min: float = 0.0, tau_max: float = 2.0) -> Any:
    """Map τ value to color from blue (low) to red (high)."""
    if tau_max == tau_min:
        return TAU_CMAP(0.5)
    t = np.clip((tau - tau_min) / (tau_max - tau_min), 0, 1)
    return TAU_CMAP(t)


# ── EVQ-cosh formula (replicated from run_evq_sweep.py) ───────────

def evq_cosh_inv_freq(head_dim: int, tau: float, base: float = 500000.0) -> np.ndarray:
    """Compute EVQ-cosh inverse frequencies."""
    K = head_dim // 2
    idx = np.arange(K, dtype=np.float64)
    u = idx / max(K - 1, 1)  # 0..1

    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * sinh_tau)

    theta_max = 1.0 / base  # lowest frequency (highest period)
    theta_min = 1.0          # highest frequency (period=2π)
    inv_freq = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return inv_freq


# ── Data loading ──────────────────────────────────────────────────

def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def extract_tau_ppl_matrix(data: Dict) -> Tuple[List[float], List[int], np.ndarray, Dict]:
    """Extract a (tau × eval_length) PPL matrix, averaging over seeds.

    Returns: taus, eval_lengths, ppl_matrix[tau_idx, len_idx], raw_per_seed
    """
    experiments = data.get("experiments", {})

    # Gather unique τ and seeds
    tau_seed_map: Dict[float, Dict[int, Dict[str, float]]] = {}
    for run_id, exp in experiments.items():
        if "tau" not in exp or "ppl" not in exp:
            continue
        tau = exp["tau"]
        seed = exp.get("seed", 0)
        ppl = exp["ppl"]
        tau_seed_map.setdefault(tau, {})[seed] = ppl

    taus = sorted(tau_seed_map.keys())
    if not taus:
        return [], [], np.array([]), {}

    # Gather eval lengths from first available experiment
    all_lengths = set()
    for seed_map in tau_seed_map.values():
        for ppl_dict in seed_map.values():
            all_lengths.update(int(k) for k in ppl_dict.keys())
    eval_lengths = sorted(all_lengths)

    # Build matrix (mean over seeds)
    n_taus = len(taus)
    n_lens = len(eval_lengths)
    ppl_mean = np.full((n_taus, n_lens), np.nan)
    ppl_std = np.full((n_taus, n_lens), np.nan)

    raw_per_seed = {}
    for ti, tau in enumerate(taus):
        seed_map = tau_seed_map[tau]
        seeds = sorted(seed_map.keys())
        raw_per_seed[tau] = {}
        for li, L in enumerate(eval_lengths):
            vals = [seed_map[s].get(str(L), np.nan) for s in seeds]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                ppl_mean[ti, li] = np.mean(vals)
                ppl_std[ti, li] = np.std(vals) if len(vals) > 1 else 0.0
                raw_per_seed[tau][L] = vals

    return taus, eval_lengths, ppl_mean, raw_per_seed


# ── Figure 1: PPL vs τ ──────────────────────────────────────────

def plot_ppl_vs_tau(
    taus: List[float],
    eval_lengths: List[int],
    ppl_mean: np.ndarray,
    out_path: str,
    raw_per_seed: Optional[Dict] = None,
):
    """One sub-plot per eval length, x-axis = τ, y-axis = PPL."""
    n_panels = len(eval_lengths)
    if n_panels == 0:
        print("  [SKIP] ppl_vs_tau: no data")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels, 2.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for li, (L, ax) in enumerate(zip(eval_lengths, axes)):
        ppls = ppl_mean[:, li]
        valid = ~np.isnan(ppls)
        t_valid = np.array(taus)[valid]
        p_valid = ppls[valid]

        # Main line
        ax.plot(t_valid, p_valid, "o-", color=C_EVQ, markersize=4, linewidth=1.4, zorder=3)

        # Scatter per-seed if available
        if raw_per_seed:
            for tau in t_valid:
                seeds_vals = raw_per_seed.get(tau, {}).get(L, [])
                if len(seeds_vals) > 1:
                    ax.scatter(
                        [tau] * len(seeds_vals), seeds_vals,
                        s=10, color=C_EVQ, alpha=0.3, zorder=2
                    )

        # Highlight τ=0 (geometric baseline)
        if 0.0 in taus:
            idx0 = taus.index(0.0)
            if not np.isnan(ppl_mean[idx0, li]):
                ax.axhline(ppl_mean[idx0, li], color=C_GEO, ls="--", lw=0.8, alpha=0.6)
                ax.text(
                    max(taus) * 0.95, ppl_mean[idx0, li],
                    r"Geo ($\tau$=0)", fontsize=6, ha="right", va="bottom",
                    color=C_GEO, alpha=0.8,
                )

        ax.set_xlabel(r"$\tau$")
        ax.set_title(f"L = {L:,}" if L < 10000 else f"L = {L // 1024}K")
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.set_xlim(min(taus) - 0.05, max(taus) + 0.05)

    axes[0].set_ylabel("Perplexity")
    fig.suptitle(r"PPL vs $\tau$ at Different Context Lengths", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ── Figure 2: Waterbed trade-off ─────────────────────────────────

def plot_waterbed(data: Dict, out_path: str, taus: Optional[List[float]] = None):
    """x = long-context PPL change %, y = short-context PPL change %."""
    waterbed = data.get("waterbed", {})
    if not waterbed:
        print("  [SKIP] waterbed: no data")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    xs, ys, labels, colors = [], [], [], []
    for run_id, wb in sorted(waterbed.items()):
        # Extract short and long change
        short_keys = [k for k in wb if "short" in k and "change" in k]
        long_keys = [k for k in wb if "long" in k and "change" in k]
        if not short_keys or not long_keys:
            # Try per_length fallback
            per_l = wb.get("per_length", {})
            if not per_l:
                continue
            lengths = sorted(int(k) for k in per_l.keys())
            if len(lengths) < 2:
                continue
            short_change = per_l[str(lengths[0])]
            long_change = per_l[str(lengths[-1])]
        else:
            short_change = wb[short_keys[0]]
            long_change = wb[long_keys[0]]

        # Parse τ from run_id
        tau_val = 0.0
        if "_tau" in run_id:
            try:
                tau_val = float(run_id.split("_tau")[1].split("_")[0])
            except ValueError:
                pass

        xs.append(long_change)   # x: long-context change (negative = improved)
        ys.append(short_change)  # y: short-context change (positive = degraded)
        labels.append(f"τ={tau_val:.1f}")

        tau_range = max(taus) - min(taus) if taus else 2.0
        colors.append(tau_color(tau_val, min(taus) if taus else 0.0, max(taus) if taus else 2.0))

    if not xs:
        print("  [SKIP] waterbed: no valid data points")
        plt.close(fig)
        return

    xs, ys = np.array(xs), np.array(ys)

    # Waterbed quadrant shading
    ax.axhspan(0, max(max(ys) * 1.2, 5), xmin=0, xmax=0.5, color="#FFECEC", alpha=0.3, zorder=0)
    ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.5)

    # Scatter points
    ax.scatter(xs, ys, c=colors, s=40, edgecolors="white", linewidths=0.5, zorder=3)

    # Label each point
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(
            lab, (x, y), textcoords="offset points", xytext=(4, 4),
            fontsize=6, alpha=0.8,
        )

    # Quadrant annotations
    ax.text(
        0.05, 0.95, "Waterbed holds\n(trade-off)",
        transform=ax.transAxes, fontsize=6, va="top", ha="left",
        style="italic", color="#999",
    )

    ax.set_xlabel("Long-context PPL change (%)")
    ax.set_ylabel("Short-context PPL change (%)")
    ax.set_title("Waterbed Trade-off", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ── Figure 3: Phase collision heatmap ────────────────────────────

def plot_phase_collision(data: Dict, out_path: str):
    """Heatmap: x = τ, y = distance range, color = collision score."""
    experiments = data.get("experiments", {})

    # Collect collision data per τ (average over seeds)
    tau_collision: Dict[float, Dict[str, List[float]]] = {}
    for run_id, exp in experiments.items():
        if "tau" not in exp or "phase_collision" not in exp:
            continue
        tau = exp["tau"]
        pc = exp["phase_collision"]
        tau_collision.setdefault(tau, {})
        for k, v in pc.items():
            tau_collision[tau].setdefault(k, []).append(v)

    if not tau_collision:
        print("  [SKIP] phase_collision: no data")
        return

    taus = sorted(tau_collision.keys())
    # Use distance keys (d=X format)
    dist_keys = [k for k in sorted(tau_collision[taus[0]].keys()) if k.startswith("d=")]
    if not dist_keys:
        print("  [SKIP] phase_collision: no distance keys")
        return

    # Build matrix
    matrix = np.zeros((len(dist_keys), len(taus)))
    for ti, tau in enumerate(taus):
        for di, dk in enumerate(dist_keys):
            vals = tau_collision[tau].get(dk, [0.0])
            matrix[di, ti] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(3.5, 0.6 * len(taus) + 1.5), 3.5))

    # Custom diverging colormap: blue (low collision) to red (high)
    cmap = LinearSegmentedColormap.from_list("collision", ["#2166ac", "#f7f7f7", "#d6604d"])

    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap,
        vmin=-1.0, vmax=1.0,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:.1f}" if t == int(t) else f"{t:.2f}" for t in taus], fontsize=7)
    ax.set_yticks(range(len(dist_keys)))
    ax.set_yticklabels([dk.replace("d=", "Δ=") for dk in dist_keys], fontsize=7)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Token Distance")
    ax.set_title("Phase Collision: cos(θ·Δ)", fontsize=9)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax, label="avg cos(θ·Δ)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ── Figure 4: Frequency distribution curves ──────────────────────

def plot_freq_distribution(
    taus: List[float],
    head_dim: int,
    base: float,
    out_path: str,
):
    """Overlay inv_freq curves for different τ values."""
    if not taus:
        print("  [SKIP] freq_distribution: no taus")
        return

    K = head_dim // 2
    idx = np.arange(K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    tau_min, tau_max = min(taus), max(taus)

    for tau in taus:
        inv = evq_cosh_inv_freq(head_dim, tau, base)
        color = tau_color(tau, tau_min, tau_max)
        label = f"τ={tau:.1f}" if tau == int(tau) else f"τ={tau:.2f}"

        # Left: log-scale inv_freq
        ax1.semilogy(idx, inv, color=color, label=label, linewidth=1.2, alpha=0.85)

        # Right: normalized spacing ratio
        if len(inv) > 1:
            ratios = inv[:-1] / inv[1:]
            ax2.plot(idx[:-1], ratios, color=color, label=label, linewidth=1.0, alpha=0.85)

    # Geometric baseline reference
    geo_inv = evq_cosh_inv_freq(head_dim, 0.0, base)
    ax1.semilogy(idx, geo_inv, color=C_GEO, ls="--", lw=1.0, alpha=0.5, label="Geo (τ=0)")

    ax1.set_xlabel("Frequency Index k")
    ax1.set_ylabel(r"$\theta_k$ (log scale)")
    ax1.set_title("(a) Frequency Allocation", fontsize=9)
    ax1.legend(fontsize=5.5, ncol=2, loc="upper right", framealpha=0.7)
    ax1.grid(True, alpha=0.2, linewidth=0.4)

    geo_ratios = geo_inv[:-1] / geo_inv[1:]
    ax2.axhline(geo_ratios[0], color=C_GEO, ls="--", lw=0.8, alpha=0.5, label="Geo (constant)")
    ax2.set_xlabel("Frequency Index k")
    ax2.set_ylabel(r"$\theta_k / \theta_{k+1}$")
    ax2.set_title("(b) Adjacent Frequency Ratio", fontsize=9)
    ax2.legend(fontsize=5.5, ncol=2, loc="upper right", framealpha=0.7)
    ax2.grid(True, alpha=0.2, linewidth=0.4)

    fig.suptitle(
        f"EVQ-Cosh Frequency Distribution (d_head={head_dim}, base={base:.0f})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ── Summary table ────────────────────────────────────────────────

def generate_summary_table(
    taus: List[float],
    eval_lengths: List[int],
    ppl_mean: np.ndarray,
    data: Dict,
    out_path: str,
):
    """CSV summary table for easy LaTeX inclusion."""
    if not taus or not eval_lengths:
        print("  [SKIP] summary_table: no data")
        return

    lines = []
    # Header
    len_cols = [f"PPL@{L//1024}K" if L >= 1024 else f"PPL@{L}" for L in eval_lengths]
    header = "tau," + ",".join(len_cols) + ",waterbed_holds"
    lines.append(header)

    waterbed = data.get("waterbed", {})

    for ti, tau in enumerate(taus):
        ppls = [f"{ppl_mean[ti, li]:.2f}" if not np.isnan(ppl_mean[ti, li]) else "—"
                for li in range(len(eval_lengths))]

        # Check waterbed
        wb_key_candidates = [k for k in waterbed if f"_tau{tau:.2f}_" in k]
        if wb_key_candidates:
            wb = waterbed[wb_key_candidates[0]]
            holds = "Y" if wb.get("waterbed_holds") else "N"
        elif tau == 0.0:
            holds = "baseline"
        else:
            holds = "—"

        line = f"{tau:.2f}," + ",".join(ppls) + f",{holds}"
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out_path}")

    # Also print to console
    print("\n  Summary Table:")
    print(f"  {'τ':>6s}", end="")
    for L in eval_lengths:
        label = f"{L//1024}K" if L >= 1024 else str(L)
        print(f"  {label:>8s}", end="")
    print(f"  {'WB':>4s}")
    print("  " + "-" * (8 + 10 * len(eval_lengths) + 6))

    for ti, tau in enumerate(taus):
        print(f"  {tau:6.2f}", end="")
        for li in range(len(eval_lengths)):
            v = ppl_mean[ti, li]
            if np.isnan(v):
                print(f"  {'—':>8s}", end="")
            else:
                print(f"  {v:8.2f}", end="")
        # Waterbed
        wb_key_candidates = [k for k in waterbed if f"_tau{tau:.2f}_" in k]
        if wb_key_candidates:
            wb = waterbed[wb_key_candidates[0]]
            holds = "Y" if wb.get("waterbed_holds") else "N"
        elif tau == 0.0:
            holds = "—"
        else:
            holds = "—"
        print(f"  {holds:>4s}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EVQ τ-sweep analysis and figure generation")
    parser.add_argument("--input", required=True, help="Path to results_final.json")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument("--base", type=float, default=500000.0, help="RoPE base frequency")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVQ τ-Sweep Analysis")
    print(f"  Input: {input_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    data = load_results(str(input_path))

    # Extract τ-PPL matrix
    taus, eval_lengths, ppl_mean, raw_per_seed = extract_tau_ppl_matrix(data)
    print(f"  Found {len(taus)} τ values: {taus}")
    print(f"  Eval lengths: {eval_lengths}")
    n_experiments = len(data.get("experiments", {}))
    print(f"  Total experiments: {n_experiments}")

    if not taus:
        print("\n  [ERROR] No valid experiments found in input!")
        sys.exit(1)

    # Figure 1: PPL vs τ
    plot_ppl_vs_tau(
        taus, eval_lengths, ppl_mean,
        str(out_dir / "fig_ppl_vs_tau.pdf"),
        raw_per_seed=raw_per_seed,
    )

    # Figure 2: Waterbed trade-off
    plot_waterbed(data, str(out_dir / "fig_waterbed.pdf"), taus=taus)

    # Figure 3: Phase collision heatmap
    plot_phase_collision(data, str(out_dir / "fig_phase_collision.pdf"))

    # Figure 4: Frequency distribution curves
    plot_freq_distribution(
        taus, args.head_dim, args.base,
        str(out_dir / "fig_freq_distribution.pdf"),
    )

    # Summary table
    generate_summary_table(
        taus, eval_lengths, ppl_mean, data,
        str(out_dir / "summary_table.csv"),
    )

    print(f"\n{'='*60}")
    print(f"  Analysis complete! Figures saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
