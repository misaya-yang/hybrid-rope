#!/usr/bin/env python3
"""Phase 16 analysis: r-sweep + τ-scaling results.

Reads results_final.json from Phase 16 work dirs and generates:
  1. r-sweep summary table (PPL + Passkey vs r, 2-seed mean ± std)
  2. r convexity check (is r=0 optimal or noise-equivalent?)
  3. τ* calibration table (predicted vs measured τ*)
  4. Paper-ready LaTeX tables

Usage:
    python analyze_phase16.py --base_dir /root/autodl-tmp
    python analyze_phase16.py --rsweep_dir /path/to/phase16_rsweep
    python analyze_phase16.py --tau_dirs /path/to/L512 /path/to/L1024
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    """Load results JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_ppl(experiments: dict, run_id: str) -> Dict[str, float]:
    """Extract PPL dict from experiment entry."""
    return experiments.get(run_id, {}).get("ppl", {})


def extract_passkey_rate(experiments: dict, run_id: str) -> Dict[str, float]:
    """Extract passkey retrieval rates by length from experiment entry."""
    pk = experiments.get(run_id, {}).get("passkey", {})
    if not pk or not isinstance(pk, dict):
        return {}
    rates = {}
    summary = pk.get("summary", {})
    for key, val in summary.items():
        if isinstance(val, dict) and "retrieval_rate" in val:
            # key format: "L=2048_d=0.50" etc — extract the length
            parts = key.split("_")
            for p in parts:
                if p.startswith("L="):
                    L = p[2:]
                    # Average across depths for this length
                    if L not in rates:
                        rates[L] = []
                    rates[L].append(val["retrieval_rate"])
    # Average depths
    return {L: np.mean(vs) for L, vs in rates.items()}


# ---------------------------------------------------------------------------
# Experiment A: r-sweep analysis
# ---------------------------------------------------------------------------

def analyze_rsweep(results: dict) -> None:
    """Analyze r-sweep results and print tables."""
    meta = results.get("metadata", {})
    experiments = results.get("experiments", {})
    r_values = meta.get("r_values", [])
    seeds = meta.get("seeds", [])
    fixed_tau = meta.get("fixed_tau", "?")
    tier = meta.get("tier", "?")

    if not r_values:
        print("  No r_values found in metadata, trying to infer...")
        r_values = sorted(set(
            int(k.split("_r")[1].split("_")[0])
            for k in experiments if "_r" in k
        ))
    if not seeds:
        seeds = sorted(set(
            int(k.split("seed")[1])
            for k in experiments if "seed" in k
        ))

    # Collect all eval lengths
    all_lengths = set()
    for exp in experiments.values():
        ppl = exp.get("ppl", {})
        all_lengths.update(int(k) for k in ppl.keys())
    eval_lengths = sorted(all_lengths)

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT A: r-SWEEP ANALYSIS")
    print(f"  tier={tier}  tau={fixed_tau}  seeds={seeds}  r_values={r_values}")
    print(f"{'='*80}")

    # --- PPL table ---
    print(f"\n  PPL (mean ± std across seeds)")
    header = f"  {'r':>4}"
    for L in eval_lengths:
        header += f"  {'PPL@'+str(L):>12}"
    print(header)
    print(f"  {'─'*4}" + f"  {'─'*12}" * len(eval_lengths))

    best_ppl = {str(L): (float("inf"), -1) for L in eval_lengths}

    for r in r_values:
        row = f"  {r:>4}"
        for L in eval_lengths:
            vals = []
            for seed in seeds:
                run_id = f"{tier}_r{r}_tau{fixed_tau:.2f}_seed{seed}"
                ppl = extract_ppl(experiments, run_id)
                v = ppl.get(str(L))
                if v is not None:
                    vals.append(v)
            if vals:
                mean = np.mean(vals)
                if len(vals) > 1:
                    std = np.std(vals, ddof=1)
                    row += f"  {mean:>6.1f}±{std:<4.1f}"
                else:
                    row += f"  {mean:>11.1f}"
                if mean < best_ppl[str(L)][0]:
                    best_ppl[str(L)] = (mean, r)
            else:
                row += f"  {'—':>12}"
        print(row)

    print(f"\n  Best r by eval length:")
    for L in eval_lengths:
        ppl_val, r_best = best_ppl[str(L)]
        print(f"    PPL@{L}: r={r_best} (PPL={ppl_val:.1f})")

    # --- Passkey table ---
    print(f"\n  Passkey Retrieval Rate (mean across seeds)")
    pk_lengths = set()
    for exp in experiments.values():
        pk = exp.get("passkey", {})
        if isinstance(pk, dict) and "summary" in pk:
            for key in pk["summary"]:
                for p in key.split("_"):
                    if p.startswith("L="):
                        pk_lengths.add(int(p[2:]))
    pk_lengths = sorted(pk_lengths)

    if pk_lengths:
        header = f"  {'r':>4}"
        for L in pk_lengths:
            header += f"  {'PK@'+str(L):>10}"
        print(header)
        print(f"  {'─'*4}" + f"  {'─'*10}" * len(pk_lengths))

        for r in r_values:
            row = f"  {r:>4}"
            for L in pk_lengths:
                vals = []
                for seed in seeds:
                    run_id = f"{tier}_r{r}_tau{fixed_tau:.2f}_seed{seed}"
                    rates = extract_passkey_rate(experiments, run_id)
                    v = rates.get(str(L))
                    if v is not None:
                        vals.append(v)
                if vals:
                    mean = np.mean(vals) * 100
                    row += f"  {mean:>9.0f}%"
                else:
                    row += f"  {'—':>10}"
            print(row)

    # --- Convexity check ---
    print(f"\n  Convexity check (is PPL monotonically increasing with r?)")
    for L in eval_lengths:
        means = []
        for r in r_values:
            vals = []
            for seed in seeds:
                run_id = f"{tier}_r{r}_tau{fixed_tau:.2f}_seed{seed}"
                ppl = extract_ppl(experiments, run_id)
                v = ppl.get(str(L))
                if v is not None:
                    vals.append(v)
            means.append(np.mean(vals) if vals else float("nan"))

        monotonic = all(
            means[i] <= means[i + 1] + 0.5  # allow 0.5 PPL noise
            for i in range(len(means) - 1)
            if not (math.isnan(means[i]) or math.isnan(means[i + 1]))
        )
        trend = "MONOTONIC ✓" if monotonic else "NON-MONOTONIC"
        print(f"    PPL@{L}: {trend}  "
              f"(r=0: {means[0]:.1f}, r={r_values[-1]}: {means[-1]:.1f}, "
              f"Δ={means[-1]-means[0]:+.1f})")


# ---------------------------------------------------------------------------
# Experiment B: τ-scaling analysis
# ---------------------------------------------------------------------------

def analyze_tau_scaling(results_by_L: Dict[int, dict]) -> None:
    """Analyze τ-scaling results across different L_train values."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT B: τ* SCALING LAW ANALYSIS")
    print(f"{'='*80}")

    HEAD_DIM = 64  # fixed for all tiers

    tau_star_table = []

    for L_train, results in sorted(results_by_L.items()):
        experiments = results.get("experiments", {})
        meta = results.get("metadata", {})
        taus = meta.get("taus", [])
        tier = meta.get("tier", "?")

        if not taus:
            taus = sorted(set(
                float(k.split("tau")[1].split("_")[0])
                for k in experiments if "tau" in k
            ))

        predicted_tau = HEAD_DIM / math.sqrt(L_train)

        print(f"\n  L_train={L_train}  tier={tier}  predicted τ*={predicted_tau:.2f}")

        # Collect PPL at various eval lengths for each τ
        all_lengths = set()
        for exp in experiments.values():
            all_lengths.update(int(k) for k in exp.get("ppl", {}).keys())
        eval_lengths = sorted(all_lengths)

        header = f"  {'τ':>6}  {'τ-τ*':>6}"
        for L in eval_lengths:
            ratio = L / L_train
            header += f"  {'PPL@'+str(L)+f'({ratio:.0f}x)':>16}"
        print(header)
        print(f"  {'─'*6}  {'─'*6}" + f"  {'─'*16}" * len(eval_lengths))

        # Find best τ at each extrapolation ratio
        best_by_length = {}
        tau_ppl_data = {}

        for tau in taus:
            run_id = f"{tier}_tau{tau:.2f}_seed42"
            ppl = extract_ppl(experiments, run_id)

            delta = tau - predicted_tau
            row = f"  {tau:>6.2f}  {delta:>+6.2f}"

            tau_ppl_data[tau] = {}
            for L in eval_lengths:
                v = ppl.get(str(L))
                if v is not None:
                    ratio = L / L_train
                    row += f"  {v:>16.1f}"
                    tau_ppl_data[tau][L] = v
                    # Track best at extrapolation ratios ≥ 2x
                    if ratio >= 2.0:
                        if L not in best_by_length or v < best_by_length[L][1]:
                            best_by_length[L] = (tau, v)
                else:
                    row += f"  {'—':>16}"
            print(row)

        # Determine empirical τ*: the τ that minimizes PPL at moderate extrapolation
        # Use the longest eval length with ≥4x ratio as the reference
        ref_lengths = [L for L in eval_lengths if L / L_train >= 4]
        if ref_lengths:
            ref_L = ref_lengths[-1]  # longest
            best_tau, best_val = best_by_length.get(ref_L, (None, None))
            if best_tau is not None:
                deviation = ((best_tau / predicted_tau) - 1) * 100
                tau_star_table.append({
                    "L_train": L_train,
                    "predicted": predicted_tau,
                    "measured": best_tau,
                    "deviation_pct": deviation,
                    "ref_L": ref_L,
                })
                print(f"\n  → Empirical τ* at {ref_L} ({ref_L/L_train:.0f}x): "
                      f"τ={best_tau:.2f}  (predicted: {predicted_tau:.2f}, "
                      f"deviation: {deviation:+.0f}%)")

    # --- Cross-L summary ---
    if tau_star_table:
        print(f"\n  {'─'*60}")
        print(f"  τ* SCALING LAW SUMMARY: τ*(L) = d_head/√L = 64/√L")
        print(f"  {'─'*60}")
        print(f"  {'L_train':>8}  {'Predicted':>10}  {'Measured':>10}  "
              f"{'Deviation':>10}  {'Ref Length':>10}")
        for entry in tau_star_table:
            print(f"  {entry['L_train']:>8}  {entry['predicted']:>10.2f}  "
                  f"{entry['measured']:>10.2f}  {entry['deviation_pct']:>+9.0f}%  "
                  f"{entry['ref_L']:>10}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_rsweep(results: dict) -> str:
    """Generate LaTeX table for r-sweep results."""
    meta = results.get("metadata", {})
    experiments = results.get("experiments", {})
    r_values = meta.get("r_values", [])
    seeds = meta.get("seeds", [])
    fixed_tau = meta.get("fixed_tau", "?")
    tier = meta.get("tier", "?")

    # Pick representative eval lengths for paper
    target_lengths = [2048, 4096, 8192]
    available = set()
    for exp in experiments.values():
        available.update(int(k) for k in exp.get("ppl", {}).keys())
    eval_lengths = [L for L in target_lengths if L in available]

    n_cols = 1 + len(eval_lengths)  # r + PPL columns
    col_spec = "r" + "c" * len(eval_lengths)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{r-sweep: PPL vs.\ number of geometric channels $r$.}",
        r"\label{tab:rsweep}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = "$r$"
    for L in eval_lengths:
        header += f" & PPL@{L//1000}K"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for r in r_values:
        row = f"{r}"
        for L in eval_lengths:
            vals = []
            for seed in seeds:
                run_id = f"{tier}_r{r}_tau{fixed_tau:.2f}_seed{seed}"
                ppl = extract_ppl(experiments, run_id)
                v = ppl.get(str(L))
                if v is not None:
                    vals.append(v)
            if vals:
                mean = np.mean(vals)
                if len(vals) > 1:
                    std = np.std(vals, ddof=1)
                    row += f" & ${mean:.1f} \\pm {std:.1f}$"
                else:
                    row += f" & {mean:.1f}"
            else:
                row += " & ---"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_tau_scaling(tau_star_table: list) -> str:
    """Generate LaTeX table for τ* scaling law."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation of $\tau^* = d_{\mathrm{head}} / \sqrt{L}$ "
        r"across training lengths.}",
        r"\label{tab:tau-scaling}",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"$L_{\mathrm{train}}$ & Predicted $\tau^*$ & Measured $\tau^*$ "
        r"& Deviation \\",
        r"\midrule",
    ]

    for entry in tau_star_table:
        lines.append(
            f"{entry['L_train']} & {entry['predicted']:.2f} & "
            f"{entry['measured']:.2f} & "
            f"{entry['deviation_pct']:+.0f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 16 analysis: r-sweep + τ-scaling"
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Base directory containing phase16_* subdirs"
    )
    parser.add_argument(
        "--rsweep_dir", type=str, default=None,
        help="Direct path to r-sweep results dir"
    )
    parser.add_argument(
        "--tau_dirs", type=str, nargs="*", default=None,
        help="Direct paths to τ-scaling results dirs"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also print LaTeX tables"
    )
    args = parser.parse_args()

    # Resolve directories
    rsweep_dir = None
    tau_dirs = {}

    if args.base_dir:
        base = Path(args.base_dir)
        rsweep_candidate = base / "phase16_rsweep"
        if rsweep_candidate.exists():
            rsweep_dir = rsweep_candidate
        for suffix, L in [("phase16_tau_L512", 512), ("phase16_tau_L1024", 1024)]:
            candidate = base / suffix
            if candidate.exists():
                tau_dirs[L] = candidate

    if args.rsweep_dir:
        rsweep_dir = Path(args.rsweep_dir)
    if args.tau_dirs:
        for d in args.tau_dirs:
            p = Path(d)
            # Infer L from dir name
            name = p.name.lower()
            for candidate_L in [256, 512, 1024, 2048, 4096]:
                if str(candidate_L) in name:
                    tau_dirs[candidate_L] = p
                    break

    # --- Experiment A analysis ---
    if rsweep_dir:
        results_path = rsweep_dir / "results_final.json"
        if not results_path.exists():
            results_path = rsweep_dir / "results_checkpoint.json"
        if results_path.exists():
            results = load_results(results_path)
            analyze_rsweep(results)
            if args.latex:
                print(f"\n  LaTeX (r-sweep):")
                print(generate_latex_rsweep(results))
        else:
            print(f"  WARNING: No results found in {rsweep_dir}")

    # --- Experiment B analysis ---
    if tau_dirs:
        results_by_L = {}
        for L, d in sorted(tau_dirs.items()):
            results_path = d / "results_final.json"
            if not results_path.exists():
                results_path = d / "results_checkpoint.json"
            if results_path.exists():
                results_by_L[L] = load_results(results_path)
            else:
                print(f"  WARNING: No results found in {d}")

        if results_by_L:
            analyze_tau_scaling(results_by_L)

    if not rsweep_dir and not tau_dirs:
        print("  ERROR: No data directories specified.")
        print("  Use --base_dir, --rsweep_dir, or --tau_dirs")
        sys.exit(1)


if __name__ == "__main__":
    main()
