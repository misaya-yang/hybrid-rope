#!/usr/bin/env python3
"""
Post-sweep analysis: extract basin curves, L-exponent, τ_floor phase transition
from the weekend 50M τ theory sweep.

Usage:
    python scripts/m4_max_36gb/analyze_weekend_sweep.py
    python scripts/m4_max_36gb/analyze_weekend_sweep.py --plot  # requires matplotlib

Outputs:
    results/weekend_sweep/analysis/basin_data.json   -- structured PPL/passkey table
    results/weekend_sweep/analysis/L_exponent.json   -- log-log fit result
    results/weekend_sweep/analysis/summary.md        -- paper-ready summary
    results/weekend_sweep/analysis/basin_curves.pdf  -- optional plot
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "results" / "weekend_sweep"
ANALYSIS_DIR = SWEEP_DIR / "analysis"
L_VALUES = [256, 512, 1024, 2048]
TAU_STAR = {256: 4.0, 512: 2.8284, 1024: 2.0, 2048: 1.4142}


def load_runs(L: int) -> list:
    """Load all completed runs for a given L."""
    ckpt = SWEEP_DIR / f"L{L}" / "results_checkpoint.json"
    if not ckpt.exists():
        return []
    with open(ckpt) as f:
        data = json.load(f)
    return list(data.get("experiments", {}).values())


def extract_basin(L: int, eval_len: int = None) -> dict:
    """For given L, extract (τ, PPL) pairs averaged across seeds.

    eval_len: evaluation context length (default: 2*L for 2x extrapolation)
    """
    if eval_len is None:
        eval_len = 2 * L
    runs = load_runs(L)
    if not runs:
        return {}

    by_tau = {}
    for r in runs:
        tau = r.get("tau", 0.0)
        ppl_dict = r.get("ppl", {})
        ppl = ppl_dict.get(str(eval_len))
        if ppl is None:
            continue
        by_tau.setdefault(tau, []).append(ppl)

    return {
        tau: {
            "mean": mean(ppls),
            "std": stdev(ppls) if len(ppls) > 1 else 0.0,
            "n": len(ppls),
            "r": tau / TAU_STAR[L],
        }
        for tau, ppls in by_tau.items()
    }


def fit_L_exponent(L_to_basin: dict) -> dict:
    """Fit τ_opt ∝ L^{-γ}.

    For each L, τ_opt = tau with minimum mean PPL at 2× extrapolation.
    Then log-log regress τ_opt vs L.
    """
    Ls, tau_opts = [], []
    for L, basin in L_to_basin.items():
        if not basin:
            continue
        # Exclude tau=0 from optimum search (geometric baseline, not EVQ)
        # Also exclude L values with only 1 tau value (e.g., partial L=2048)
        non_zero_taus = [t for t in basin if t > 0]
        if len(non_zero_taus) < 2:
            continue
        tau_opt = min(non_zero_taus, key=lambda t: basin[t]["mean"])
        Ls.append(L)
        tau_opts.append(tau_opt)

    if len(Ls) < 2:
        return {"n": len(Ls), "error": "need ≥2 L values with ≥2 τ values each"}

    # Linear regression on log(tau) vs log(L)
    logL = [math.log(l) for l in Ls]
    logT = [math.log(t) for t in tau_opts]
    n = len(logL)
    mean_x, mean_y = sum(logL) / n, sum(logT) / n
    num = sum((logL[i] - mean_x) * (logT[i] - mean_y) for i in range(n))
    den = sum((logL[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return {"error": "degenerate"}
    slope = num / den  # this is -γ
    intercept = mean_y - slope * mean_x
    # R²
    ss_res = sum((logT[i] - (slope * logL[i] + intercept)) ** 2 for i in range(n))
    ss_tot = sum((logT[i] - mean_y) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "n": n,
        "Ls": Ls,
        "tau_opts": tau_opts,
        "gamma": -slope,  # τ ∝ L^{-γ}
        "prefactor": math.exp(intercept),
        "r2": r2,
        "theoretical_gamma": 0.5,
        "gap_to_theory": abs(-slope - 0.5),
    }


def basin_width(basin: dict, pct_threshold: float = 0.01) -> dict:
    """Compute τ range where PPL is within pct_threshold of minimum."""
    if not basin:
        return {}
    taus = sorted(basin.keys())
    ppls = [basin[t]["mean"] for t in taus]
    ppl_min = min(ppls)
    threshold = ppl_min * (1 + pct_threshold)
    inside = [t for t, p in zip(taus, ppls) if p <= threshold]
    if not inside:
        return {"width": 0.0, "tau_opt": taus[ppls.index(ppl_min)]}
    return {
        "tau_opt": taus[ppls.index(ppl_min)],
        "ppl_min": ppl_min,
        "ppl_threshold": threshold,
        "pct_threshold": pct_threshold,
        "tau_low": min(inside),
        "tau_high": max(inside),
        "width": max(inside) - min(inside),
        "width_relative_to_opt": (max(inside) - min(inside)) / taus[ppls.index(ppl_min)] if taus[ppls.index(ppl_min)] > 0 else 0.0,
        "n_inside": len(inside),
    }


def tau_floor_check(L: int, K: int = 32) -> dict:
    """Check τ_floor Proposition 2 prediction at given L.

    Proposition 2: τ_floor = 4/√K · (1 + 1/(2K) + ...) ≈ 0.71 at K=32.
    For L where τ*(L) < τ_floor, EVQ at τ* is effectively geometric.
    """
    import numpy as np  # noqa

    tau_floor = 4.0 / math.sqrt(K) * (1 + 1.0 / (2 * K))
    basin = extract_basin(L, eval_len=2 * L)
    if not basin:
        return {}

    below_floor_taus = [t for t in basin if t < tau_floor and t > 0]
    tau0 = basin.get(0.0, {}).get("mean")

    floor_comparison = {}
    for t in below_floor_taus:
        if tau0 is not None:
            floor_comparison[t] = {
                "ppl": basin[t]["mean"],
                "ppl_vs_geo": basin[t]["mean"] - tau0,
                "ppl_vs_geo_pct": (basin[t]["mean"] - tau0) / tau0 * 100 if tau0 > 0 else 0.0,
            }

    return {
        "L": L,
        "K": K,
        "tau_floor_predicted": tau_floor,
        "below_floor_taus": below_floor_taus,
        "comparison_vs_geo": floor_comparison,
        "interpretation": "τ below floor should behave similarly to τ=0 (Geo); large |ppl_vs_geo| invalidates Proposition 2",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true")
    p.add_argument("--eval_len_ratio", type=int, default=2,
                   help="Eval at eval_len_ratio × L (default 2 = 2× extrapolation)")
    args = p.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Weekend sweep analysis  (eval: {args.eval_len_ratio}× L extrapolation)")
    print("=" * 70)

    # Load all L basins
    L_to_basin = {}
    for L in L_VALUES:
        basin = extract_basin(L, eval_len=args.eval_len_ratio * L)
        L_to_basin[L] = basin
        if basin:
            print(f"\n[L={L}] τ*(L)={TAU_STAR[L]:.3f}, {len(basin)} τ values, "
                  f"seeds/τ: {[basin[t]['n'] for t in sorted(basin)]}")
            for tau in sorted(basin):
                r = basin[tau]["r"]
                ppl = basin[tau]["mean"]
                std = basin[tau]["std"]
                print(f"  τ={tau:.4f} (r={r:.2f}):  PPL = {ppl:.2f} ± {std:.2f}")

    # Save basin data
    basin_out = {
        "metadata": {"eval_len_ratio": args.eval_len_ratio, "L_values": L_VALUES, "TAU_STAR": TAU_STAR},
        "basins": {str(L): basin for L, basin in L_to_basin.items()},
    }
    (ANALYSIS_DIR / "basin_data.json").write_text(json.dumps(basin_out, indent=2))

    # L-exponent fit
    print("\n" + "=" * 70)
    print("L-exponent analysis  (τ_opt vs L log-log regression)")
    print("=" * 70)
    fit = fit_L_exponent(L_to_basin)
    print(json.dumps(fit, indent=2, default=str))
    (ANALYSIS_DIR / "L_exponent.json").write_text(json.dumps(fit, indent=2, default=str))

    # Basin widths
    print("\n" + "=" * 70)
    print("Basin width  (τ range where PPL within 1% of minimum)")
    print("=" * 70)
    widths = {}
    for L, basin in L_to_basin.items():
        if basin:
            w = basin_width(basin, pct_threshold=0.01)
            widths[L] = w
            print(f"[L={L}] τ_opt={w.get('tau_opt'):.3f}  "
                  f"τ∈[{w.get('tau_low', '?'):.3f}, {w.get('tau_high', '?'):.3f}]  "
                  f"width={w.get('width_relative_to_opt', 0)*100:.1f}% of τ_opt")
    (ANALYSIS_DIR / "basin_widths.json").write_text(json.dumps({str(k): v for k, v in widths.items()}, indent=2, default=str))

    # τ_floor check
    print("\n" + "=" * 70)
    print("τ_floor Proposition 2 check  (below-floor τ vs geometric)")
    print("=" * 70)
    floor_results = {}
    for L in L_VALUES:
        fc = tau_floor_check(L)
        if fc:
            floor_results[L] = fc
            print(f"\n[L={L}] τ_floor predicted = {fc['tau_floor_predicted']:.3f}")
            for t, c in fc.get('comparison_vs_geo', {}).items():
                print(f"  τ={t:.3f} (below floor): PPL vs Geo diff = {c['ppl_vs_geo']:+.2f} ({c['ppl_vs_geo_pct']:+.1f}%)")
    (ANALYSIS_DIR / "tau_floor_check.json").write_text(json.dumps({str(k): v for k, v in floor_results.items()}, indent=2, default=str))

    # Paper-ready summary
    md = []
    md.append("# Weekend 50M τ Theory Sweep — Results Summary\n")
    md.append(f"Eval at {args.eval_len_ratio}× extrapolation length.\n")

    md.append("\n## L-exponent fit\n")
    md.append(f"- γ (empirical) = **{fit.get('gamma', 0):.3f}**")
    md.append(f"- γ (theory) = 0.500")
    md.append(f"- Gap to theory = **{fit.get('gap_to_theory', 0):.3f}**")
    md.append(f"- R² = {fit.get('r2', 0):.3f}")
    md.append(f"- τ_opt values by L: {dict(zip(fit.get('Ls', []), [round(t, 3) for t in fit.get('tau_opts', [])]))}\n")

    md.append("\n## Basin widths (<1% PPL threshold)\n")
    md.append("| L | τ_opt | τ range | relative width |")
    md.append("|---|---|---|---|")
    for L, w in widths.items():
        md.append(f"| {L} | {w.get('tau_opt', '?'):.3f} | "
                  f"[{w.get('tau_low', '?'):.3f}, {w.get('tau_high', '?'):.3f}] | "
                  f"{w.get('width_relative_to_opt', 0)*100:.1f}% |")

    md.append("\n## τ_floor Proposition 2 validation\n")
    for L, fc in floor_results.items():
        md.append(f"\n### L={L}, τ_floor = {fc['tau_floor_predicted']:.3f}")
        for t, c in fc.get('comparison_vs_geo', {}).items():
            md.append(f"- τ={t:.3f}: PPL vs Geo = {c['ppl_vs_geo']:+.2f} ({c['ppl_vs_geo_pct']:+.1f}%)")
    (ANALYSIS_DIR / "summary.md").write_text("\n".join(md))
    print(f"\n\n✓ Summary written to {ANALYSIS_DIR / 'summary.md'}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            # Basin curves
            for L, basin in L_to_basin.items():
                if not basin: continue
                rs = sorted([basin[t]["r"] for t in basin])
                taus_sorted = sorted(basin.keys())
                ppls = [basin[t]["mean"] for t in taus_sorted]
                stds = [basin[t]["std"] for t in taus_sorted]
                rs = [basin[t]["r"] for t in taus_sorted]
                axs[0].errorbar(rs, ppls, yerr=stds, label=f"L={L}", marker="o")
            axs[0].set_xlabel("τ / τ*(L)")
            axs[0].set_ylabel(f"PPL at {args.eval_len_ratio}× extrapolation")
            axs[0].set_title("Basin shape")
            axs[0].legend()
            axs[0].grid(alpha=0.3)

            # L-exponent
            if fit.get("Ls") and fit.get("tau_opts"):
                axs[1].scatter(fit["Ls"], fit["tau_opts"], s=80, label="empirical τ_opt")
                # theory line (γ=0.5)
                import numpy as np
                xs = np.array(sorted(fit["Ls"]))
                axs[1].plot(xs, xs.astype(float) ** (-0.5) * (fit["tau_opts"][0] * fit["Ls"][0] ** 0.5),
                            "--", alpha=0.5, label="theory γ=0.5")
                axs[1].plot(xs, xs.astype(float) ** (-fit["gamma"]) * fit["prefactor"],
                            ":", alpha=0.7, label=f"fit γ={fit['gamma']:.3f}")
                axs[1].set_xscale("log")
                axs[1].set_yscale("log")
                axs[1].set_xlabel("L")
                axs[1].set_ylabel("τ_opt")
                axs[1].set_title(f"L-exponent fit: γ={fit['gamma']:.3f} (theory 0.5)")
                axs[1].legend()
                axs[1].grid(alpha=0.3, which="both")

            plt.tight_layout()
            plt.savefig(ANALYSIS_DIR / "basin_curves.pdf")
            print(f"✓ Plot saved to {ANALYSIS_DIR / 'basin_curves.pdf'}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
