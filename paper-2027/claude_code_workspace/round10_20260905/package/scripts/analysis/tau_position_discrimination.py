#!/usr/bin/env python3
"""
Key insight from tau_scaling_analysis.py:
  - Broadband surrogate τ = √(β/α) gives wrong L-scaling (L^{-0.085} vs L^{-0.5})
  - BUT: at d_head = L, the surrogate is EXACT (ratio=0.975)
  - For d_head << L, discrete finite-K effects dominate

This script tries: compute the ACTUAL position discrimination quality for L positions
using K = d_head/2 RoPE channels, and optimize τ to maximize it.

Key quantity: for L positions, the RoPE inner product between positions m and n is:
  R(m,n) = (1/K) Σ_k cos(ω_k (m-n))

where ω_k = b^{-φ_k(τ)}.

Two positions are "confused" if R(m,n) ≈ R(m,m) = 1.

The position discrimination at distance Δ is:
  D(Δ) = 1 - R(Δ) = 1 - (1/K) Σ_k cos(ω_k Δ)

Objective: maximize the MINIMUM discrimination across all distances 1..L-1.
  max_τ min_{Δ=1..L-1} D(Δ, τ)

Or alternatively: minimize the collision probability:
  P_collision = Σ_{Δ=1}^{L-1} R(Δ)² = Σ_Δ [(1/K) Σ_k cos(ω_k Δ)]²
"""

import numpy as np


def evq_phi(u_arr, tau):
    if abs(tau) < 1e-6:
        return u_arr.copy()
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))


def rope_correlation(phi_arr, delta, b):
    """Compute R(Δ) = (1/K) Σ_k cos(ω_k Δ) for given allocation."""
    omega = b ** (-phi_arr)
    return np.mean(np.cos(omega * delta))


def min_discrimination(phi_arr, L, b):
    """Min discrimination: min_{Δ=1..L-1} [1 - R(Δ)]"""
    K = len(phi_arr)
    omega = b ** (-phi_arr)
    min_D = 2.0  # max possible is 2
    for delta in range(1, L):
        R = np.mean(np.cos(omega * delta))
        D = 1 - R
        if D < min_D:
            min_D = D
    return min_D


def collision_probability(phi_arr, L, b):
    """Total collision: Σ_{Δ=1}^{L-1} R(Δ)²"""
    K = len(phi_arr)
    omega = b ** (-phi_arr)
    total = 0.0
    for delta in range(1, L):
        R = np.mean(np.cos(omega * delta))
        total += R * R
    return total


def weighted_collision(phi_arr, L, b, power=0):
    """Weighted collision: Σ Δ^power × R(Δ)²"""
    omega = b ** (-phi_arr)
    total = 0.0
    for delta in range(1, L):
        R = np.mean(np.cos(omega * delta))
        weight = delta ** power
        total += weight * R * R
    return total


def long_range_collision(phi_arr, L, b, fraction=0.5):
    """Collision only at distances > fraction × L."""
    omega = b ** (-phi_arr)
    start = int(fraction * L)
    total = 0.0
    count = 0
    for delta in range(start, L):
        R = np.mean(np.cos(omega * delta))
        total += R * R
        count += 1
    return total / max(count, 1)


def optimize_tau_grid(d_head, L, b, obj_func, minimize=True, n_grid=200):
    """Grid search for optimal τ."""
    K = d_head // 2
    u = (2 * np.arange(K) + 1) / (2 * K)

    tau_max = max(d_head / np.sqrt(L) * 5, 15)
    taus = np.linspace(0.05, tau_max, n_grid)

    best_tau = 0
    best_val = 1e20 if minimize else -1e20

    for tau in taus:
        phi = evq_phi(u, tau)
        val = obj_func(phi, L, b)
        if minimize and val < best_val:
            best_val = val
            best_tau = tau
        elif not minimize and val > best_val:
            best_val = val
            best_tau = tau

    return best_tau, best_val


def main():
    print("="*80)
    print("POSITION DISCRIMINATION OPTIMIZATION")
    print("Finding: which ACTUAL position-based objective gives τ* = d_head/√L?")
    print("="*80)

    # Test configs: keep L small enough for direct computation
    configs = [
        (16, 32, 10_000),
        (16, 64, 10_000),
        (16, 128, 10_000),
        (16, 256, 10_000),
        (32, 32, 10_000),
        (32, 64, 10_000),
        (32, 128, 10_000),
        (32, 256, 10_000),
        (32, 512, 10_000),
        (64, 128, 10_000),
        (64, 256, 10_000),
        (64, 512, 10_000),
        (64, 1024, 10_000),
    ]

    objectives = {
        'MinDiscrim': (lambda phi, L, b: -min_discrimination(phi, L, b), True),  # maximize → minimize negative
        'TotalCollision': (lambda phi, L, b: collision_probability(phi, L, b), True),
        'WeightedColl_Δ¹': (lambda phi, L, b: weighted_collision(phi, L, b, power=1), True),
        'WeightedColl_Δ²': (lambda phi, L, b: weighted_collision(phi, L, b, power=2), True),
        'LongRange50%': (lambda phi, L, b: long_range_collision(phi, L, b, 0.5), True),
        'LongRange75%': (lambda phi, L, b: long_range_collision(phi, L, b, 0.75), True),
    }

    for obj_name, (obj_func, minimize) in objectives.items():
        print(f"\n{'─'*80}")
        print(f"OBJECTIVE: {obj_name}")
        print(f"{'─'*80}")
        print(f"  {'d_head':>6s} {'L':>6s} {'τ_opt':>8s} {'τ*=d/√L':>8s} {'ratio':>8s}")
        print(f"  {'-'*45}")

        for d_head, L, b in configs:
            tau_emp = d_head / np.sqrt(L)
            tau_opt, _ = optimize_tau_grid(d_head, L, b, obj_func, minimize)
            ratio = tau_opt / tau_emp if tau_emp > 0 else 0
            print(f"  {d_head:>6d} {L:>6d} {tau_opt:>8.3f} {tau_emp:>8.3f} {ratio:>8.3f}")

    # Detailed sweep for one case
    print("\n" + "="*80)
    print("DETAILED SWEEP: d_head=32, L=256, b=10000 (τ*=2.0)")
    print("="*80)
    d_head, L, b = 32, 256, 10_000
    K = d_head // 2
    u = (2 * np.arange(K) + 1) / (2 * K)
    tau_emp = d_head / np.sqrt(L)

    print(f"\n  {'τ':>6s} {'MinDisc':>10s} {'TotColl':>10s} {'WColl_Δ¹':>12s} {'WColl_Δ²':>12s} {'LR50':>10s}")
    print(f"  {'-'*65}")

    for tau in np.linspace(0.1, 8, 40):
        phi = evq_phi(u, tau)
        md = min_discrimination(phi, L, b)
        tc = collision_probability(phi, L, b)
        wc1 = weighted_collision(phi, L, b, 1)
        wc2 = weighted_collision(phi, L, b, 2)
        lr = long_range_collision(phi, L, b, 0.5)
        marker = "  ←τ*" if abs(tau - tau_emp) < 0.15 else ""
        print(f"  {tau:>6.2f} {md:>10.6f} {tc:>10.4f} {wc1:>12.2f} {wc2:>12.0f} {lr:>10.6f}{marker}")

    print(f"\n  τ* = d/√L = {tau_emp:.3f}")

    # Check b-dependence
    print("\n" + "="*80)
    print("BASE DEPENDENCE: d_head=32, L=128")
    print("="*80)
    d_head, L = 32, 128
    tau_emp = d_head / np.sqrt(L)
    print(f"  τ* = d/√L = {tau_emp:.3f}")
    print(f"\n  {'base':>10s} {'TotColl_τ_opt':>14s} {'WColl_Δ²_τ_opt':>15s}")
    print(f"  {'-'*45}")

    for b in [500, 1000, 5000, 10_000, 50_000, 500_000]:
        tau1, _ = optimize_tau_grid(d_head, L, b,
                                     lambda phi, L, b: collision_probability(phi, L, b), True)
        tau2, _ = optimize_tau_grid(d_head, L, b,
                                     lambda phi, L, b: weighted_collision(phi, L, b, power=2), True)
        print(f"  {b:>10d} {tau1:>14.3f} {tau2:>15.3f}")


if __name__ == "__main__":
    main()
