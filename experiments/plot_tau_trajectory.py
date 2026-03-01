#!/usr/bin/env python3
"""Plot τ trajectory and PPL comparison for learnable-τ EVQ experiments.

Figure A: τ vs training step (learnable runs overlaid, dashed = best fixed τ)
Figure B: PPL@16K vs τ (fixed sweep as curve, learnable as star markers)

Usage:
    python experiments/plot_tau_trajectory.py --work_dir ~/evq_125m_learnable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# NeurIPS style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results(work_dir: Path) -> dict:
    """Load results from checkpoint or final JSON."""
    for name in ["results_final.json", "results_checkpoint.json"]:
        p = work_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"No results JSON found in {work_dir}")


def load_tau_trajectories(work_dir: Path) -> Dict[str, List[dict]]:
    """Find all tau_trajectory.json files in run directories."""
    trajectories = {}
    for traj_path in sorted(work_dir.glob("*/tau_trajectory.json")):
        run_id = traj_path.parent.name
        with open(traj_path) as f:
            trajectories[run_id] = json.load(f)
    return trajectories


def get_fixed_tau_ppls(results: dict, eval_key: str = "16384") -> List[Tuple[float, float]]:
    """Extract (τ, PPL@eval_key) pairs for fixed-τ runs."""
    pairs = []
    for run_id, data in results.get("experiments", {}).items():
        if "learnable" in run_id or "PI" in run_id:
            continue
        tau = data.get("tau", None)
        ppl = data.get("ppl", {}).get(eval_key, None)
        if tau is not None and ppl is not None:
            pairs.append((tau, ppl))
    return sorted(pairs, key=lambda x: x[0])


def get_learnable_ppls(results: dict, eval_key: str = "16384") -> List[Tuple[str, float, float]]:
    """Extract (run_id, final_tau, PPL@eval_key) for learnable runs."""
    items = []
    for run_id, data in results.get("experiments", {}).items():
        if "learnable" not in run_id:
            continue
        tau = data.get("tau", None)
        ppl = data.get("ppl", {}).get(eval_key, None)
        if tau is not None and ppl is not None:
            items.append((run_id, tau, ppl))
    return items


def plot_tau_trajectory(work_dir: Path, trajectories: Dict[str, List[dict]],
                        best_fixed_tau: float | None = None) -> Path:
    """Figure A: τ vs training step."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(trajectories), 3)))
    for i, (run_id, traj) in enumerate(trajectories.items()):
        steps = [e["step"] for e in traj]
        taus = [e["tau"] for e in traj]
        # Extract init from run_id for label
        label = run_id.replace("125m_learnable_", "").replace("_seed42", "")
        ax.plot(steps, taus, color=colors[i], linewidth=1.5, label=label)

    if best_fixed_tau is not None:
        ax.axhline(best_fixed_tau, color="gray", linestyle="--", linewidth=1,
                    label=f"best fixed τ={best_fixed_tau:.2f}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("τ")
    ax.set_title("Learned τ trajectory during training")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out = work_dir / "fig_tau_trajectory.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_ppl_vs_tau(work_dir: Path, fixed_pairs: List[Tuple[float, float]],
                    learnable_items: List[Tuple[str, float, float]],
                    eval_key: str = "16384") -> Path:
    """Figure B: PPL@eval_key vs τ."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Fixed sweep as curve
    if fixed_pairs:
        taus_f, ppls_f = zip(*fixed_pairs)
        ax.plot(taus_f, ppls_f, "o-", color="steelblue", linewidth=1.5,
                markersize=5, label="Fixed τ (sweep)")

    # Learnable as stars
    for run_id, tau, ppl in learnable_items:
        label = run_id.replace("125m_learnable_", "").replace("_seed42", "")
        ax.plot(tau, ppl, "*", markersize=14, color="crimson", zorder=5,
                label=f"Learned: {label}")

    ax.set_xlabel("τ (final)")
    ax.set_ylabel(f"PPL @ L={eval_key}")
    ax.set_title(f"Perplexity vs τ (L={eval_key})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out = work_dir / f"fig_ppl_vs_tau_{eval_key}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot learnable-τ results")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--eval_key", type=str, default="16384",
                        help="Eval length key for PPL comparison (default: 16384)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    print(f"Loading results from {work_dir}")

    results = load_results(work_dir)
    trajectories = load_tau_trajectories(work_dir)
    fixed_pairs = get_fixed_tau_ppls(results, args.eval_key)
    learnable_items = get_learnable_ppls(results, args.eval_key)

    print(f"  Fixed runs: {len(fixed_pairs)}")
    print(f"  Learnable trajectories: {len(trajectories)}")
    print(f"  Learnable PPL entries: {len(learnable_items)}")

    # Find best fixed τ
    best_fixed_tau = None
    if fixed_pairs:
        best_fixed_tau = min(fixed_pairs, key=lambda x: x[1])[0]
        best_fixed_ppl = min(fixed_pairs, key=lambda x: x[1])[1]
        print(f"  Best fixed: τ={best_fixed_tau:.2f}, PPL={best_fixed_ppl:.2f}")

    # Plot
    if trajectories:
        plot_tau_trajectory(work_dir, trajectories, best_fixed_tau)
    else:
        print("  No tau trajectories found, skipping Figure A")

    if fixed_pairs or learnable_items:
        plot_ppl_vs_tau(work_dir, fixed_pairs, learnable_items, args.eval_key)
    else:
        print("  No PPL data found, skipping Figure B")

    # Print summary table
    print(f"\n  Summary (PPL@{args.eval_key}):")
    print(f"  {'Method':35s}  {'τ':>8s}  {'PPL':>8s}")
    print(f"  {'-'*55}")
    for tau, ppl in fixed_pairs:
        print(f"  {'Fixed τ=' + f'{tau:.2f}':35s}  {tau:8.4f}  {ppl:8.2f}")
    for run_id, tau, ppl in learnable_items:
        label = f"Learnable ({run_id.split('_seed')[0].split('learnable_')[1]})"
        print(f"  {label:35s}  {tau:8.4f}  {ppl:8.2f}")


if __name__ == "__main__":
    main()
