#!/usr/bin/env python3
"""
Test 3 修正版: 扫描距离先验 D(Δ) 的 power-law exponent α，找到使 R² 最大化的 α。

理论中 D(Δ) 是"距离先验"——距离 Δ 处 token-pair 的重要性权重。
正确的操作化不是 token co-occurrence，而是 attention 距离分布。

候选 D(Δ) 形式:
  1. power-law: D(Δ) ∝ Δ^{-α}，α ∈ [0, 2]
  2. exponential: D(Δ) ∝ exp(-Δ/λ)
  3. uniform: D(Δ) = 1/L
  4. 真实 attention: 从训练好的模型提取 (需要 GPU，本脚本不做)
  5. 真实 token co-occurrence: 已证明不是正确定义

NLP 中常见假设: D(Δ) ∝ 1/Δ (α=1)，因为近距离 context 更重要。

Usage:
    python test3_distance_prior_sweep.py              # 扫全部
    python test3_distance_prior_sweep.py --base 500000  # 指定 base
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# Core: Broadband R² computation for given D(Δ)
# ============================================================

def compute_broadband_r2_from_D(
    D: np.ndarray,
    base: int,
    n_grid: int = 64,
) -> dict:
    """
    Compute broadband R² for a single (D, base) config.
    D: shape (L,), normalized distance prior for Δ=1,...,L
    """
    L = len(D)
    phi = np.linspace(0, 1, n_grid)
    dphi = phi[1] - phi[0]
    omega = base ** (-phi)
    M = np.minimum(phi[:, None], phi[None, :])

    deltas = np.arange(1, L + 1, dtype=np.float64)

    # Kernel K_ij = Σ_Δ D(Δ) cos(ω_i Δ) cos(ω_j Δ)
    cos_table = np.cos(np.outer(omega, deltas))  # (n_grid, L)
    weighted = cos_table * D[np.newaxis, :]
    K = weighted @ cos_table.T  # (n_grid, n_grid)

    # Fit: K ≈ c₀ + α_fit·I/Δφ + β·M
    # Step 1: off-diagonal → β, c₀
    mask = ~np.eye(n_grid, dtype=bool)
    K_off = K[mask]
    M_off = M[mask]
    A_fit = np.column_stack([np.ones_like(K_off), M_off])
    coeffs, _, _, _ = np.linalg.lstsq(A_fit, K_off, rcond=None)
    c0, beta = coeffs

    # Step 2: diagonal → alpha_fit
    K_diag = np.diag(K)
    resid_diag = K_diag - c0 - beta * phi
    alpha_fit = resid_diag.mean() * dphi

    # Reconstruct (allow negative alpha for diagnostics)
    I_mat = np.eye(n_grid) * (alpha_fit / dphi)
    K_approx = c0 + beta * M + I_mat

    # R² full
    ss_res = np.sum((K - K_approx) ** 2)
    ss_tot = np.sum((K - K.mean()) ** 2)
    r2_full = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # R² mid-band (exclude edge 10%)
    lo, hi = int(n_grid * 0.1), int(n_grid * 0.9)
    K_mid = K[lo:hi, lo:hi]
    K_approx_mid = K_approx[lo:hi, lo:hi]
    ss_res_mid = np.sum((K_mid - K_approx_mid) ** 2)
    ss_tot_mid = np.sum((K_mid - K_mid.mean()) ** 2)
    r2_mid = 1 - ss_res_mid / ss_tot_mid if ss_tot_mid > 0 else 0

    # τ from fit
    if alpha_fit > 0 and beta > 0:
        tau_fit = math.sqrt(beta / alpha_fit)
    else:
        tau_fit = float("nan")

    return {
        "r2_full": round(r2_full, 6),
        "r2_mid": round(r2_mid, 6),
        "alpha_fit": float(alpha_fit),
        "beta": float(beta),
        "c0": float(c0),
        "tau_fit": round(tau_fit, 4) if not math.isnan(tau_fit) else None,
        "K_norm": float(np.linalg.norm(K, 'fro')),
        "residual_frac": round(math.sqrt(ss_res / ss_tot) if ss_tot > 0 else 1, 6),
    }


# ============================================================
# Distance prior generators
# ============================================================

def make_power_law(L: int, alpha: float) -> np.ndarray:
    """D(Δ) ∝ Δ^{-alpha}, normalized."""
    deltas = np.arange(1, L + 1, dtype=np.float64)
    D = deltas ** (-alpha)
    return D / D.sum()


def make_exponential(L: int, lam: float) -> np.ndarray:
    """D(Δ) ∝ exp(-Δ/λ), normalized."""
    deltas = np.arange(1, L + 1, dtype=np.float64)
    D = np.exp(-deltas / lam)
    return D / D.sum()


def make_uniform(L: int) -> np.ndarray:
    """D(Δ) = 1/L."""
    return np.ones(L, dtype=np.float64) / L


def make_log_uniform(L: int) -> np.ndarray:
    """D(Δ) ∝ 1/(Δ·ln(L)), i.e. uniform on log-scale."""
    deltas = np.arange(1, L + 1, dtype=np.float64)
    D = 1.0 / deltas
    return D / D.sum()


# ============================================================
# Main sweep
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test 3: Distance prior sweep for broadband R²")
    parser.add_argument("--base", type=int, nargs="+",
                        default=[10_000, 100_000, 500_000, 10_000_000])
    parser.add_argument("--L", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / "m4_max_36gb"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Part 1: Power-law exponent sweep
    # ============================================================
    alphas = np.arange(0.0, 2.05, 0.1)

    print("=" * 80)
    print("PART 1: Power-law sweep D(Δ) ∝ Δ^{-α}")
    print("=" * 80)

    all_results = {"power_law_sweep": {}, "special_priors": {}, "best_configs": []}

    for base in args.base:
        for L in args.L:
            key = f"base={base}_L={L}"
            print(f"\n--- {key} ---")
            print(f"  {'α':>6s}  {'R²_full':>10s}  {'R²_mid':>10s}  {'α_fit':>10s}  "
                  f"{'β':>10s}  {'τ_fit':>10s}")
            print(f"  " + "-" * 65)

            sweep = []
            best_r2_mid = -999
            best_alpha = None

            for alpha in alphas:
                D = make_power_law(L, alpha)
                r = compute_broadband_r2_from_D(D, base)
                r["alpha_prior"] = round(float(alpha), 2)
                sweep.append(r)

                marker = ""
                if r["r2_mid"] > best_r2_mid:
                    best_r2_mid = r["r2_mid"]
                    best_alpha = alpha
                if r["r2_mid"] > 0.99:
                    marker = " *** >0.99!"
                elif r["r2_mid"] > 0.95:
                    marker = " ** >0.95"

                tau_str = f"{r['tau_fit']:10.2f}" if r['tau_fit'] is not None else "       NaN"
                print(f"  {alpha:6.2f}  {r['r2_full']:10.6f}  {r['r2_mid']:10.6f}  "
                      f"{r['alpha_fit']:10.4e}  {r['beta']:10.4e}  {tau_str}{marker}")

            all_results["power_law_sweep"][key] = sweep
            print(f"\n  ★ Best α = {best_alpha:.2f}, R²_mid = {best_r2_mid:.6f}")

            all_results["best_configs"].append({
                "base": base, "L": L,
                "best_alpha": round(float(best_alpha), 2),
                "best_r2_mid": round(best_r2_mid, 6),
            })

    # ============================================================
    # Part 2: Fine-grained sweep around best α
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 2: Fine-grained sweep around best α (base=500K, L=2048)")
    print("=" * 80)

    # Find best α from Part 1 for base=500K, L=2048
    base_focus = 500_000
    L_focus = 2048
    key_focus = f"base={base_focus}_L={L_focus}"
    sweep_focus = all_results["power_law_sweep"].get(key_focus, [])
    if sweep_focus:
        best_coarse = max(sweep_focus, key=lambda x: x["r2_mid"])
        alpha_center = best_coarse["alpha_prior"]
    else:
        alpha_center = 1.0

    alphas_fine = np.arange(max(0, alpha_center - 0.5), alpha_center + 0.55, 0.05)
    print(f"\n  Sweeping α ∈ [{alphas_fine[0]:.2f}, {alphas_fine[-1]:.2f}] "
          f"around coarse best {alpha_center:.2f}")
    print(f"  {'α':>6s}  {'R²_full':>10s}  {'R²_mid':>10s}")
    print(f"  " + "-" * 30)

    fine_results = []
    for alpha in alphas_fine:
        D = make_power_law(L_focus, alpha)
        r = compute_broadband_r2_from_D(D, base_focus)
        r["alpha_prior"] = round(float(alpha), 3)
        fine_results.append(r)
        marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
        print(f"  {alpha:6.3f}  {r['r2_full']:10.6f}  {r['r2_mid']:10.6f}{marker}")

    best_fine = max(fine_results, key=lambda x: x["r2_mid"])
    print(f"\n  ★ Best fine α = {best_fine['alpha_prior']:.3f}, R²_mid = {best_fine['r2_mid']:.6f}")
    all_results["fine_sweep_500K_2048"] = fine_results

    # ============================================================
    # Part 3: Special priors comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 3: Special priors comparison (base=500K, L=2048)")
    print("=" * 80)

    special = {}
    for name, D in [
        ("uniform", make_uniform(L_focus)),
        ("1/Δ (log-uniform)", make_log_uniform(L_focus)),
        ("exp(-Δ/100)", make_exponential(L_focus, 100)),
        ("exp(-Δ/500)", make_exponential(L_focus, 500)),
        ("exp(-Δ/1000)", make_exponential(L_focus, 1000)),
        ("Δ^{-0.5}", make_power_law(L_focus, 0.5)),
        ("Δ^{-1.0}", make_power_law(L_focus, 1.0)),
        ("Δ^{-1.5}", make_power_law(L_focus, 1.5)),
        ("Δ^{-2.0}", make_power_law(L_focus, 2.0)),
    ]:
        r = compute_broadband_r2_from_D(D, base_focus)
        special[name] = r
        marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
        tau_str = f"{r['tau_fit']:.2f}" if r['tau_fit'] is not None else "NaN"
        print(f"  {name:25s}  R²_full={r['r2_full']:.4f}  R²_mid={r['r2_mid']:.4f}  "
              f"τ_fit={tau_str}{marker}")

    all_results["special_priors"]["base=500000_L=2048"] = special

    # Also test with real D(Δ) from our measurements
    print("\n  Real D(Δ) from datasets (if available):")
    for dname in ["fineweb_edu", "openwebtext", "wikitext", "c4", "tinystories"]:
        pt_path = results_dir / f"D_{dname}.pt"
        if pt_path.exists():
            import torch
            D_real = torch.load(pt_path, weights_only=True).numpy().astype(np.float64)
            D_trunc = D_real[:L_focus].copy()
            D_trunc /= D_trunc.sum()
            r = compute_broadband_r2_from_D(D_trunc, base_focus)
            special[f"real_{dname}"] = r
            print(f"  {dname:25s}  R²_full={r['r2_full']:.4f}  R²_mid={r['r2_mid']:.4f}")

    # ============================================================
    # Part 4: Cross-base analysis at best α
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 4: Cross-base R² at optimal α (from fine sweep)")
    print("=" * 80)

    best_alpha_val = best_fine["alpha_prior"]
    print(f"\n  Using α = {best_alpha_val:.3f}")
    print(f"  {'base':>10s}  {'L':>6s}  {'R²_full':>10s}  {'R²_mid':>10s}  {'τ_fit':>10s}")
    print(f"  " + "-" * 50)

    for base in args.base:
        for L in args.L:
            D = make_power_law(L, best_alpha_val)
            r = compute_broadband_r2_from_D(D, base)
            tau_str = f"{r['tau_fit']:10.2f}" if r['tau_fit'] is not None else "       NaN"
            marker = " ***" if r["r2_mid"] > 0.99 else (" **" if r["r2_mid"] > 0.95 else "")
            print(f"  {base:10d}  {L:6d}  {r['r2_full']:10.6f}  {r['r2_mid']:10.6f}  "
                  f"{tau_str}{marker}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  1. Best power-law exponent (base=500K, L=2048): α = {best_fine['alpha_prior']:.3f}")
    print(f"     R²_mid = {best_fine['r2_mid']:.6f}")
    if best_fine["r2_mid"] > 0.99:
        print(f"     ✓ R²_mid > 0.99 achieved! Paper claim verified.")
    elif best_fine["r2_mid"] > 0.95:
        print(f"     ~ R²_mid > 0.95 but < 0.99. Close but not exact match.")
    else:
        print(f"     ✗ R²_mid < 0.95. Broadband approximation quality depends on prior choice.")

    print(f"\n  2. Best configs across bases:")
    for bc in all_results["best_configs"]:
        print(f"     base={bc['base']:>10d}  L={bc['L']:>5d}  "
              f"best_α={bc['best_alpha']:.2f}  R²_mid={bc['best_r2_mid']:.6f}")

    # Save
    save_path = args.save or str(results_dir / "test3_prior_sweep_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {save_path}")


if __name__ == "__main__":
    main()
