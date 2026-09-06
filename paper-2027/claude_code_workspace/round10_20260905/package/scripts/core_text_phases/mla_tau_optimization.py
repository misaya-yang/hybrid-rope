#!/usr/bin/env python3
"""MLA τ* optimization: maximize min-gap of log half-periods in extrapolation range.

Professor's insight: instead of grid search on rugged τ landscape,
directly optimize for uniform frequency coverage in target extrapolation range.

For K=16 frequencies (MLA d_rope=32), base=500000:
  - Compute EVQ φ_k(τ) for each τ
  - Convert to log-periods: log_T_k = log(2π) + φ_k × log(base)
  - Define extrapolation range: [L_train, L_target] → [φ_min, φ_max]
  - Count frequencies in range, compute min gap
  - Find τ that maximizes min gap (= most uniform coverage)
"""
import numpy as np
import math
from scipy.optimize import minimize_scalar

K = 16  # d_rope/2
BASE = 500000.0
LOG_BASE = math.log(BASE)


def evq_phi(tau, K=16):
    """Compute φ_k(τ) for k=0..K-1."""
    idx = np.arange(K, dtype=np.float64)
    u = (idx + 0.5) / K
    if abs(tau) < 1e-8:
        return u
    sinh_tau = math.sinh(tau)
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * sinh_tau)


def log_periods(tau, K=16, base=500000.0):
    """Log of full period (2π/inv_freq) for each channel."""
    phi = evq_phi(tau, K)
    # period = 2π × base^φ → log_period = log(2π) + φ × log(base)
    return math.log(2 * math.pi) + phi * math.log(base)


def min_gap_in_range(tau, L_train, L_target, K=16, base=500000.0):
    """Compute min gap of log-periods for frequencies whose period falls in [L_train, L_target].
    Returns (min_gap, n_freqs_in_range, freqs_in_range_indices)."""
    log_T = log_periods(tau, K, base)
    log_L_min = math.log(L_train)
    log_L_max = math.log(L_target)

    # Find frequencies in range
    in_range = (log_T >= log_L_min) & (log_T <= log_L_max)
    indices = np.where(in_range)[0]

    if len(indices) < 2:
        return 0.0, len(indices), indices

    # Compute gaps between consecutive in-range frequencies
    log_T_in = log_T[indices]
    gaps = np.diff(log_T_in)
    return gaps.min(), len(indices), indices


def total_coverage_score(tau, L_train, L_target, K=16, base=500000.0):
    """Combined score: min_gap × n_freqs (we want both large gap AND many freqs)."""
    mg, n, _ = min_gap_in_range(tau, L_train, L_target, K, base)
    return mg * n  # reward both uniformity and count


def analyze_tau(tau, L_train=4096, L_target=16384):
    """Full analysis for a given tau."""
    phi = evq_phi(tau, K)
    log_T = log_periods(tau, K, BASE)
    periods = np.exp(log_T)

    log_L_min = math.log(L_train)
    log_L_max = math.log(L_target)

    in_range = (log_T >= log_L_min) & (log_T <= log_L_max)
    n_in = in_range.sum()

    # Also check broader range [L_train/2, L_target*2]
    in_broad = (log_T >= math.log(L_train/2)) & (log_T <= math.log(L_target*2))
    n_broad = in_broad.sum()

    # Min gap in extrapolation range
    mg, _, indices = min_gap_in_range(tau, L_train, L_target, K, BASE)

    # Min gap across ALL frequencies (full spectrum uniformity)
    all_gaps = np.diff(log_T)
    full_min_gap = all_gaps.min()
    full_max_gap = all_gaps.max()
    uniformity = full_min_gap / full_max_gap  # 1.0 = perfectly uniform

    return {
        'tau': tau,
        'n_in_extrap': n_in,
        'n_in_broad': n_broad,
        'min_gap_extrap': mg,
        'full_min_gap': full_min_gap,
        'full_max_gap': full_max_gap,
        'uniformity': uniformity,
        'indices_in_range': indices,
        'periods': periods,
    }


def main():
    L_train = 4096
    print("=" * 70)
    print(f"MLA τ* Optimization: K={K}, base={BASE}, L_train={L_train}")
    print("=" * 70)

    # Analyze experimental tau values
    print(f"\n--- Experimental results (50M MLA sweep) ---")
    print(f"{'tau':>5} | {'PPL@4K':>8} | {'PPL@8K':>8} | {'PPL@16K':>8} | {'#freq[4K,16K]':>13} | {'min_gap':>8} | {'uniformity':>10}")
    print("-" * 80)

    exp_results = {
        0.0: (156.3, 204.6, 348.0),
        1.8: (168.2, 177.6, 336.5),
        2.0: (178.4, 236.5, 373.5),
        2.2: (181.7, 162.4, 264.3),
        2.5: (163.1, 216.2, 346.8),
        3.0: (169.7, 171.2, 300.1),
    }

    for tau, (p4, p8, p16) in exp_results.items():
        for L_target in [16384]:
            a = analyze_tau(tau, L_train, L_target)
            print(f"{tau:5.1f} | {p4:8.1f} | {p8:8.1f} | {p16:8.1f} | {a['n_in_extrap']:13d} | {a['min_gap_extrap']:8.4f} | {a['uniformity']:10.4f}")

    # Fine-grained sweep to find optimal tau
    print(f"\n--- Fine τ sweep: min-gap in [L_train, 4×L_train] = [{L_train}, {L_train*4}] ---")
    print(f"{'tau':>5} | {'#freq':>5} | {'min_gap':>8} | {'score':>8} | {'uniformity':>10}")
    print("-" * 55)

    best_tau = 0.0
    best_score = 0.0

    for tau_100 in range(0, 500, 5):  # tau from 0.0 to 5.0 in steps of 0.05
        tau = tau_100 / 100.0
        L_target = L_train * 4  # 4× extrapolation
        mg, n, _ = min_gap_in_range(tau, L_train, L_target, K, BASE)
        a = analyze_tau(tau, L_train, L_target)
        score = total_coverage_score(tau, L_train, L_target, K, BASE)

        if score > best_score:
            best_score = score
            best_tau = tau

        if tau_100 % 20 == 0 or abs(tau - 2.8) < 0.06:
            print(f"{tau:5.2f} | {n:5d} | {mg:8.4f} | {score:8.4f} | {a['uniformity']:10.4f}")

    print(f"\n{'='*70}")
    print(f"OPTIMAL τ* = {best_tau:.2f} (score={best_score:.4f})")
    print(f"{'='*70}")

    # Detailed analysis of optimal and nearby taus
    print(f"\n--- Detailed analysis around optimal ---")
    for tau in [best_tau - 0.2, best_tau - 0.1, best_tau, best_tau + 0.1, best_tau + 0.2, 2.2, 2.8, 3.0]:
        tau = round(tau, 2)
        if tau < 0:
            continue
        a = analyze_tau(tau, L_train, L_train * 4)
        print(f"\nτ={tau:.2f}: {a['n_in_extrap']} freqs in [{L_train}, {L_train*4}], "
              f"min_gap={a['min_gap_extrap']:.4f}, uniformity={a['uniformity']:.4f}")
        print(f"  Periods: {', '.join(f'{p:.0f}' for p in a['periods'])}")
        print(f"  In-range indices: {a['indices_in_range']}")

    # Frequency distribution visualization
    print(f"\n--- Frequency periods for key τ values ---")
    for tau in [0.0, 1.414, 2.2, 2.8, 3.0]:
        phi = evq_phi(tau, K)
        periods = 2 * math.pi * BASE ** phi
        active_4k = sum(1 for p in periods if p < L_train)
        extrap_4x = sum(1 for p in periods if L_train <= p <= L_train * 4)
        print(f"\nτ={tau:.3f}: {active_4k} active(@4K) + {extrap_4x} extrap([4K,16K]) + {K-active_4k-extrap_4x} dead(>16K)")
        for k in range(K):
            marker = ""
            if periods[k] < L_train:
                marker = "ACTIVE"
            elif periods[k] <= L_train * 4:
                marker = "EXTRAP ←"
            else:
                marker = "dead"
            print(f"  k={k:2d}: period={periods[k]:12.0f}  φ={phi[k]:.4f}  {marker}")


if __name__ == "__main__":
    main()
