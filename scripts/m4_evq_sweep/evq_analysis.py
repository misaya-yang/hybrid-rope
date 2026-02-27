#!/usr/bin/env python3
"""Analyse EVQ τ-sweep results and generate NeurIPS-quality figures.

Usage:
    python scripts/m4_evq_sweep/evq_analysis.py --input ~/evq_m4_sweep/results_final.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# NeurIPS figure style
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette for τ values
TAU_COLORS = {
    0.0: "#999999",
    0.2: "#2196F3",
    0.4: "#4CAF50",
    0.6: "#FF9800",
    0.8: "#E91E63",
    1.0: "#9C27B0",
    1.5: "#795548",
    2.0: "#607D8B",
}


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def get_tau_color(tau: float) -> str:
    return TAU_COLORS.get(tau, "#333333")


# ---------------------------------------------------------------------------
# Figure 1: PPL vs τ at each eval length
# ---------------------------------------------------------------------------

def plot_ppl_vs_tau(results: Dict, out_dir: Path) -> None:
    """4-panel: PPL at each eval length as function of τ."""
    exps = results["experiments"]
    # Collect (tau, length) → ppl
    tau_ppl: Dict[float, Dict[str, float]] = {}
    for rid, data in exps.items():
        tau = data["tau"]
        ppl = data.get("ppl", {})
        if ppl:
            tau_ppl[tau] = ppl

    if not tau_ppl:
        print("  [SKIP] No PPL data found")
        return

    taus = sorted(tau_ppl.keys())
    lengths = sorted(set(k for p in tau_ppl.values() for k in p.keys()), key=int)
    n_panels = len(lengths)

    fig, axes = plt.subplots(1, n_panels, figsize=(2.2 * n_panels, 2.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for i, L in enumerate(lengths):
        ax = axes[i]
        xs = [t for t in taus if L in tau_ppl[t]]
        ys = [tau_ppl[t][L] for t in xs]
        colors = [get_tau_color(t) for t in xs]

        ax.plot(xs, ys, "o-", color="#333333", lw=1.2, markersize=5, zorder=2)
        for x, y, c in zip(xs, ys, colors):
            ax.scatter([x], [y], color=c, s=40, zorder=3, edgecolors="white", linewidths=0.5)

        ax.set_xlabel(r"$\tau$")
        ax.set_title(f"L = {int(L):,}")
        if i == 0:
            ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.15, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(f"PPL vs EVQ parameter $\\tau$  ({results['metadata'].get('tier', '?')})",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    out = out_dir / "figure_ppl_vs_tau.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Waterbed trade-off scatter
# ---------------------------------------------------------------------------

def plot_waterbed(results: Dict, out_dir: Path) -> None:
    """Scatter: long-context improvement vs short-context degradation."""
    wb = results.get("waterbed", {})
    if not wb:
        print("  [SKIP] No waterbed data")
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for rid, data in sorted(wb.items()):
        # Find the short and long change keys
        short_val = None
        long_val = None
        for k, v in data.items():
            if "short" in k and "change" in k and isinstance(v, (int, float)):
                short_val = v
            if "long" in k and "change" in k and isinstance(v, (int, float)):
                long_val = v

        if short_val is None or long_val is None:
            continue

        # Extract tau from run_id
        tau = float(rid.split("tau")[1].split("_")[0])
        color = get_tau_color(tau)
        holds = data.get("waterbed_holds", False)
        marker = "o" if holds else "x"

        ax.scatter(-long_val, short_val, color=color, s=60, marker=marker,
                   edgecolors="white" if holds else color, linewidths=0.5, zorder=3)
        ax.annotate(f"τ={tau}", (-long_val, short_val), textcoords="offset points",
                    xytext=(5, 5), fontsize=6.5, color=color)

    # Waterbed zone
    ax.fill_between([0, 50], 0, 50, color="#FFEBEE", alpha=0.3, zorder=0)
    ax.text(0.95, 0.05, "Waterbed\nzone", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#C62828", alpha=0.6)

    ax.axhline(0, color="#ccc", ls=":", lw=0.8)
    ax.axvline(0, color="#ccc", ls=":", lw=0.8)
    ax.set_xlabel("Long-context PPL improvement (%)")
    ax.set_ylabel("Short-context PPL degradation (%)")
    ax.set_title("Waterbed trade-off verification")
    ax.grid(True, alpha=0.12, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = out_dir / "figure_waterbed.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Phase collision bar chart
# ---------------------------------------------------------------------------

def plot_phase_collision(results: Dict, out_dir: Path) -> None:
    """Bar chart: phase collision total score by τ."""
    exps = results["experiments"]
    taus = []
    totals = []
    shorts = []
    mids = []
    longs = []

    for rid, data in sorted(exps.items(), key=lambda x: x[1]["tau"]):
        pc = data.get("phase_collision", {})
        if "total" not in pc:
            continue
        if data["tau"] in [d["tau"] for d in [exps[r] for r in taus]]:
            continue  # dedup
        taus.append(rid)
        totals.append(pc["total"])
        shorts.append(pc.get("short_avg", 0))
        mids.append(pc.get("mid_avg", 0))
        longs.append(pc.get("long_avg", 0))

    if not taus:
        print("  [SKIP] No phase collision data")
        return

    tau_vals = [exps[r]["tau"] for r in taus]
    x = np.arange(len(taus))
    w = 0.22

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.bar(x - w, shorts, w, label="Short (≤100)", color="#90CAF9", edgecolor="white")
    ax.bar(x, mids, w, label="Mid (100–5K)", color="#FFE082", edgecolor="white")
    ax.bar(x + w, longs, w, label="Long (>5K)", color="#EF9A9A", edgecolor="white")
    ax.plot(x, totals, "ko-", lw=1.2, markersize=4, label="Weighted total", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in tau_vals], rotation=30)
    ax.set_ylabel("Phase collision score")
    ax.set_title("Phase collision vs EVQ parameter")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.12, lw=0.5, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = out_dir / "figure_phase_collision.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Frequency distribution overlay
# ---------------------------------------------------------------------------

def plot_freq_distribution(results: Dict, out_dir: Path) -> None:
    """Overlay inv_freq curves for different τ."""
    exps = results["experiments"]
    base = results["metadata"].get("base", 500000.0)

    fig, ax = plt.subplots(figsize=(4, 2.8))

    seen_taus = set()
    for rid, data in sorted(exps.items(), key=lambda x: x[1]["tau"]):
        tau = data["tau"]
        if tau in seen_taus:
            continue
        seen_taus.add(tau)

        # Reconstruct inv_freq from EVQ formula
        head_dim = 64  # standard
        K = head_dim // 2
        idx = np.arange(K)
        u = idx / K
        if abs(tau) < 1e-8:
            phi = u
        else:
            phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
        inv_freq = base ** (-phi)

        color = get_tau_color(tau)
        label = r"$\tau\!=\!0$ (geo)" if tau == 0 else r"$\tau\!=\!" + str(tau) + "$"
        ls = "--" if tau == 0 else "-"
        ax.semilogy(idx, inv_freq, color=color, ls=ls, lw=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Frequency index $k$")
    ax.set_ylabel(r"$\omega_k = b^{-\phi_k}$")
    ax.set_title(f"Frequency allocation (base={base:.0e})")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.12, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = out_dir / "figure_freq_distribution.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary table (CSV)
# ---------------------------------------------------------------------------

def write_summary_table(results: Dict, out_dir: Path) -> None:
    """CSV summary: τ, PPL@each_length, phase_collision, waterbed."""
    import csv

    exps = results["experiments"]
    wb = results.get("waterbed", {})
    rows = []

    for rid, data in sorted(exps.items(), key=lambda x: (x[1]["seed"], x[1]["tau"])):
        row = {
            "run_id": rid,
            "tau": data["tau"],
            "seed": data["seed"],
        }
        for L, ppl in sorted(data.get("ppl", {}).items(), key=lambda x: int(x[0])):
            row[f"PPL@{L}"] = ppl
        row["collision_total"] = data.get("phase_collision", {}).get("total", "")
        row["train_time_min"] = round(data.get("train_time_sec", 0) / 60, 1)

        wb_data = wb.get(rid, {})
        row["waterbed_holds"] = wb_data.get("waterbed_holds", "")

        rows.append(row)

    if not rows:
        print("  [SKIP] No data for summary table")
        return

    fieldnames = list(rows[0].keys())
    out = out_dir / "summary_table.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse EVQ sweep results")
    parser.add_argument("--input", required=True, help="Path to results_final.json")
    parser.add_argument("--out_dir", default="", help="Output directory (default: same as input)")
    args = parser.parse_args()

    results = load_results(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.input).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Analysing: {args.input}")
    print(f"  Output:    {out_dir}")
    print(f"  Runs:      {len(results.get('experiments', {}))}\n")

    plot_ppl_vs_tau(results, out_dir)
    plot_waterbed(results, out_dir)
    plot_phase_collision(results, out_dir)
    plot_freq_distribution(results, out_dir)
    write_summary_table(results, out_dir)

    print(f"\n  All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
