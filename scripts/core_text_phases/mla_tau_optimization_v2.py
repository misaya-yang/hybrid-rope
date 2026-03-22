#!/usr/bin/env python3
"""MLA τ* optimization v2: min-gap maximization + sliding window analysis.

Key insight from v1: with K=16, base=500000, only 1-2 frequencies ever fall
in the [4K, 16K] extrapolation range. The non-monotonicity is caused by
frequencies entering/leaving the window.

This version:
1. Finds τ that maximizes min-gap for various extrapolation ranges
2. Analyzes the "frequency crossing" events that cause non-monotonicity
3. Computes the theoretical optimal τ for MLA as a function of d_rope/d_total ratio
4. Professor's suggestion: optimize for uniform coverage in target range
"""
import numpy as np
import math

K = 16  # d_rope/2 for MLA
BASE = 500000.0
LOG_BASE = math.log(BASE)


def evq_phi(tau, K=16):
    idx = np.arange(K, dtype=np.float64)
    u = (idx + 0.5) / K
    if abs(tau) < 1e-8:
        return u
    sinh_tau = math.sinh(tau)
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * sinh_tau)


def periods(tau, K=16, base=500000.0):
    phi = evq_phi(tau, K)
    return 2 * math.pi * base ** phi


def count_in_range(tau, L_lo, L_hi, K=16, base=500000.0):
    T = periods(tau, K, base)
    return int(((T >= L_lo) & (T <= L_hi)).sum())


def min_gap_log_periods_in_range(tau, L_lo, L_hi, K=16, base=500000.0):
    T = periods(tau, K, base)
    mask = (T >= L_lo) & (T <= L_hi)
    log_T = np.log(T[mask])
    if len(log_T) < 2:
        return 0.0, len(log_T)
    gaps = np.diff(np.sort(log_T))
    return gaps.min(), len(log_T)


def main():
    L_train = 4096

    print("=" * 80)
    print("PART 1: Why non-monotonic — frequency crossing events")
    print("=" * 80)
    print(f"\nAs τ increases, each channel k's period changes.")
    print(f"When period crosses into [4096, 16384], extrapolation improves.")
    print()

    # For each channel k, find the τ where its period = L_train and = 4*L_train
    print(f"{'k':>3} | {'τ where period=4096':>20} | {'τ where period=16384':>20} | {'GEO period':>12}")
    print("-" * 70)

    for k in range(K):
        geo_period = 2 * math.pi * BASE ** ((k + 0.5) / K)

        # Binary search for τ where period_k = L_train
        tau_enter = None
        tau_exit = None

        for target_L in [L_train, L_train * 4]:
            # period_k = 2π × base^(φ_k(τ)) = target_L
            # φ_k(τ) = log(target_L / 2π) / log(base)
            target_phi = math.log(target_L / (2 * math.pi)) / LOG_BASE

            # Binary search for τ
            lo, hi = 0.01, 10.0
            found = False
            for _ in range(100):
                mid = (lo + hi) / 2
                phi_k = evq_phi(mid, K)[k]
                if phi_k < target_phi:
                    lo = mid
                else:
                    hi = mid
                if abs(phi_k - target_phi) < 1e-8:
                    found = True
                    break

            tau_val = (lo + hi) / 2 if found or abs(evq_phi((lo+hi)/2, K)[k] - target_phi) < 0.001 else None

            if target_L == L_train:
                tau_enter = tau_val
            else:
                tau_exit = tau_val

        enter_str = f"{tau_enter:.3f}" if tau_enter and 0 < tau_enter < 10 else "N/A"
        exit_str = f"{tau_exit:.3f}" if tau_exit and 0 < tau_exit < 10 else "N/A"
        print(f"{k:3d} | {enter_str:>20} | {exit_str:>20} | {geo_period:12.0f}")

    print()
    print("=" * 80)
    print("PART 2: Fine-grained τ sweep — count + min-gap in extrapolation ranges")
    print("=" * 80)

    for extrap_mult in [2, 4, 8]:
        L_target = L_train * extrap_mult
        print(f"\n--- Extrapolation range: [{L_train}, {L_target}] ({extrap_mult}×) ---")
        print(f"{'τ':>5} | {'#freq':>5} | {'min_gap':>8} | {'best_period_in_range':>20}")
        print("-" * 55)

        best_tau = 0.0
        best_mg = 0.0

        for tau_100 in range(0, 500, 1):  # 0.01 resolution
            tau = tau_100 / 100.0
            mg, n = min_gap_log_periods_in_range(tau, L_train, L_target, K, BASE)

            if mg > best_mg:
                best_mg = mg
                best_tau = tau

            # Print at interesting points
            if n >= 2 and mg > 0:
                T = periods(tau, K, BASE)
                in_range = T[(T >= L_train) & (T <= L_target)]
                in_str = ", ".join(f"{p:.0f}" for p in in_range)
                print(f"{tau:5.2f} | {n:5d} | {mg:8.4f} | {in_str:>20}")

        print(f">>> Best τ for {extrap_mult}× extrap: τ={best_tau:.2f}, min_gap={best_mg:.4f}")

    print()
    print("=" * 80)
    print("PART 3: Experimental validation — correlate theory with PPL")
    print("=" * 80)

    exp = {
        0.0: (156.3, 204.6, 348.0),
        1.8: (168.2, 177.6, 336.5),
        2.0: (178.4, 236.5, 373.5),
        2.2: (181.7, 162.4, 264.3),
        2.5: (163.1, 216.2, 346.8),
        3.0: (169.7, 171.2, 300.1),
    }

    print(f"\n{'τ':>5} | {'PPL@4K':>8} | {'PPL@8K':>8} | {'PPL@16K':>8} | {'#act':>4} | {'#ext2x':>6} | {'#ext4x':>6} | {'best_ext_period':>15}")
    print("-" * 90)

    for tau, (p4, p8, p16) in exp.items():
        T = periods(tau, K, BASE)
        n_active = int((T < L_train).sum())
        n_ext2x = count_in_range(tau, L_train, L_train * 2, K, BASE)
        n_ext4x = count_in_range(tau, L_train, L_train * 4, K, BASE)
        ext_periods = T[(T >= L_train) & (T <= L_train * 4)]
        ext_str = ", ".join(f"{p:.0f}" for p in ext_periods) if len(ext_periods) > 0 else "none"
        print(f"{tau:5.1f} | {p4:8.1f} | {p8:8.1f} | {p16:8.1f} | {n_active:4d} | {n_ext2x:6d} | {n_ext4x:6d} | {ext_str:>15}")

    print()
    print("=" * 80)
    print("PART 4: MLA correction factor — d_rope/d_total effect")
    print("=" * 80)
    print()
    print("In MHA: K = d_head/2 = 32, RoPE covers 100% of attention")
    print("In MLA: K = d_rope/2 = 16, RoPE covers d_rope/(d_rope+kv_lora_rank) of attention")
    print()

    for d_rope, kv_rank in [(32, 128), (32, 192), (32, 256), (64, 256)]:
        K_val = d_rope // 2
        rope_frac = d_rope / (d_rope + kv_rank)
        n_active_geo = 0
        n_ext_geo = 0
        T_geo = periods(0.0, K_val, BASE)
        n_active_geo = int((T_geo < L_train).sum())
        n_ext_geo = count_in_range(0.0, L_train, L_train * 4, K_val, BASE)

        # Find optimal tau for this K
        best_tau = 0.0
        best_score = 0.0
        for tau_100 in range(0, 500, 1):
            tau = tau_100 / 100.0
            mg, n = min_gap_log_periods_in_range(tau, L_train, L_train * 4, K_val, BASE)
            score = mg * n
            if score > best_score:
                best_score = score
                best_tau = tau

        print(f"d_rope={d_rope}, kv_rank={kv_rank}, K={K_val}, RoPE%={rope_frac:.1%}")
        print(f"  GEO: {n_active_geo} active, {n_ext_geo} extrap([4K,16K])")
        print(f"  Optimal τ (min-gap): {best_tau:.2f}")
        print()

    print("=" * 80)
    print("PART 5: Recommendation for τ=2.8~3.0 range (professor's suggestion)")
    print("=" * 80)
    print()

    for tau_10 in range(27, 32):
        tau = tau_10 / 10.0
        T = periods(tau, K, BASE)
        n_active = int((T < L_train).sum())
        ext_2x = T[(T >= L_train) & (T <= L_train * 2)]
        ext_4x = T[(T >= L_train) & (T <= L_train * 4)]
        print(f"τ={tau:.1f}: {n_active} active, {len(ext_2x)} ext@2×, {len(ext_4x)} ext@4×", end="")
        if len(ext_4x) > 0:
            print(f"  periods={[f'{p:.0f}' for p in ext_4x]}", end="")
        # 4K degradation estimate: more active freqs compressed = more redundancy
        compression = n_active / 8.0  # ratio vs GEO's 8 active
        print(f"  compression={compression:.2f}×")


if __name__ == "__main__":
    main()
