#!/usr/bin/env python3
"""
Direct numerical optimization of τ on the discrete K-channel problem.

Key insight from tau_scaling_analysis.py:
  - Broadband surrogate gives τ_surr = √(β/α) ~ √(d_head), scaling as L^{-0.085}
  - Empirical τ* = d_head/√L, scaling as L^{-0.5}
  - The surrogate MISSES the correct scaling

Strategy: Directly optimize τ over the exact kernel matrix for K channels,
using several candidate objectives. Find which objective yields τ* = d_head/√L.

Candidate objectives:
  1. Off-diagonal Frobenius norm: Σ_{i≠j} K_ij²  (minimize collisions)
  2. Log-determinant: log|K| (maximize information capacity)
  3. Condition number: κ(K) (minimize ill-conditioning)
  4. Minimum eigenvalue gap (maximize discrimination)
  5. Weighted off-diag: Σ_{i≠j} K_ij² × w(|φ_i-φ_j|)
"""

import numpy as np

def exact_kernel(phi1, phi2, L, b):
    """Exact collision kernel for uniform distance prior on [0,L]."""
    omega1 = b ** (-phi1)
    omega2 = b ** (-phi2)
    sum_omega = omega1 + omega2
    diff_omega = omega1 - omega2

    if abs(diff_omega) < 1e-15:
        term1 = L
    else:
        term1 = np.sin(diff_omega * L) / diff_omega

    term2 = np.sin(sum_omega * L) / sum_omega
    return (term1 + term2) / (2 * L)


def evq_phi(u_arr, tau):
    """EVQ-cosh frequency allocation."""
    if abs(tau) < 1e-6:
        return u_arr.copy()  # geometric limit
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))


def compute_kernel_matrix(phi_arr, L, b):
    """Compute exact kernel matrix for given channel allocations."""
    K_ch = len(phi_arr)
    K_mat = np.zeros((K_ch, K_ch))
    for i in range(K_ch):
        for j in range(K_ch):
            K_mat[i, j] = exact_kernel(phi_arr[i], phi_arr[j], L, b)
    return K_mat


def obj_offdiag_frob(K_mat):
    """Off-diagonal Frobenius norm squared."""
    K_ch = K_mat.shape[0]
    diag = np.diag(np.diag(K_mat))
    offdiag = K_mat - diag
    return np.sum(offdiag**2)


def obj_neg_logdet(K_mat):
    """Negative log-determinant (minimize = maximize determinant)."""
    try:
        sign, logdet = np.linalg.slogdet(K_mat)
        if sign > 0:
            return -logdet
        else:
            return 1e10  # invalid
    except:
        return 1e10


def obj_condition(K_mat):
    """Condition number."""
    try:
        eigvals = np.linalg.eigvalsh(K_mat)
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) == 0:
            return 1e10
        return eigvals[-1] / eigvals[0]
    except:
        return 1e10


def obj_neg_min_eigval(K_mat):
    """Negative minimum eigenvalue (minimize = maximize min eigval)."""
    try:
        eigvals = np.linalg.eigvalsh(K_mat)
        return -eigvals[0]
    except:
        return 1e10


def obj_offdiag_sum(K_mat):
    """Sum of absolute off-diagonal elements."""
    K_ch = K_mat.shape[0]
    total = 0.0
    for i in range(K_ch):
        for j in range(K_ch):
            if i != j:
                total += abs(K_mat[i, j])
    return total


def obj_trace_normalized_offdiag(K_mat):
    """Off-diagonal Frobenius / trace² — normalized collision metric."""
    tr = np.trace(K_mat)
    if tr < 1e-15:
        return 1e10
    K_ch = K_mat.shape[0]
    diag = np.diag(np.diag(K_mat))
    offdiag = K_mat - diag
    return np.sum(offdiag**2) / (tr**2)


def optimize_tau(d_head, L, b, obj_func, tau_range=None):
    """Grid search for optimal τ given an objective function."""
    K = d_head // 2
    u = (2 * np.arange(K) + 1) / (2 * K)  # midpoints

    if tau_range is None:
        tau_range = np.concatenate([
            np.linspace(0.01, 1, 40),
            np.linspace(1, 5, 40),
            np.linspace(5, 20, 40),
            np.linspace(20, 50, 20),
        ])

    best_tau = 0
    best_val = 1e20
    results = []

    for tau in tau_range:
        phi = evq_phi(u, tau)
        K_mat = compute_kernel_matrix(phi, L, b)
        val = obj_func(K_mat)
        results.append((tau, val))
        if val < best_val:
            best_val = val
            best_tau = tau

    return best_tau, best_val, results


def main():
    print("="*80)
    print("DIRECT OPTIMIZATION: Which objective gives τ* = d_head/√L?")
    print("="*80)

    objectives = {
        'OffDiag_Frob²': obj_offdiag_frob,
        '-log|K|': obj_neg_logdet,
        'Condition#': obj_condition,
        '-λ_min': obj_neg_min_eigval,
        'OffDiag_|sum|': obj_offdiag_sum,
        'NormCollision': obj_trace_normalized_offdiag,
    }

    configs = [
        (32, 128, 10_000, "d=32, L=128, b=10K"),
        (64, 128, 10_000, "d=64, L=128, b=10K"),
        (64, 256, 10_000, "d=64, L=256, b=10K"),
        (64, 512, 10_000, "d=64, L=512, b=10K"),
        (64, 1024, 10_000, "d=64, L=1024, b=10K"),
        (64, 2048, 10_000, "d=64, L=2048, b=10K"),
        (64, 4096, 10_000, "d=64, L=4096, b=10K"),
        (128, 4096, 10_000, "d=128, L=4096, b=10K"),
        (32, 32, 10_000, "d=32, L=32, b=10K"),
        (64, 256, 500_000, "d=64, L=256, b=500K"),
        (64, 2048, 500_000, "d=64, L=2048, b=500K"),
    ]

    for obj_name, obj_func in objectives.items():
        print(f"\n{'─'*80}")
        print(f"OBJECTIVE: {obj_name}")
        print(f"{'─'*80}")
        print(f"  {'Config':<30s} {'τ_opt':>8s} {'τ*=d/√L':>8s} {'ratio':>8s}")
        print(f"  {'-'*60}")

        for d_head, L, b, desc in configs:
            tau_emp = d_head / np.sqrt(L)

            # Use finer grid around expected optimal
            tau_range = np.concatenate([
                np.linspace(0.05, max(tau_emp * 3, 10), 100),
                np.linspace(max(tau_emp * 3, 10), 50, 50),
            ])

            try:
                tau_opt, _, _ = optimize_tau(d_head, L, b, obj_func, tau_range)
                ratio = tau_opt / tau_emp if tau_emp > 0 else 0
                print(f"  {desc:<30s} {tau_opt:>8.3f} {tau_emp:>8.3f} {ratio:>8.3f}")
            except Exception as e:
                print(f"  {desc:<30s} ERROR: {e}")

    # Detailed τ sweep for a specific case to see objective landscapes
    print("\n" + "="*80)
    print("DETAILED τ SWEEP: d_head=64, L=1024, b=10000")
    print("="*80)
    d_head, L, b = 64, 1024, 10_000
    K = d_head // 2
    u = (2 * np.arange(K) + 1) / (2 * K)
    tau_emp = d_head / np.sqrt(L)  # = 2.0

    taus = np.linspace(0.1, 8, 80)
    print(f"\n  {'τ':>6s}", end="")
    for name in objectives:
        print(f"  {name:>14s}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in objectives:
        print(f"  {'-'*14}", end="")
    print()

    for tau in taus[::5]:  # print every 5th for readability
        phi = evq_phi(u, tau)
        K_mat = compute_kernel_matrix(phi, L, b)
        print(f"  {tau:>6.2f}", end="")
        for name, func in objectives.items():
            val = func(K_mat)
            if abs(val) < 1e4:
                print(f"  {val:>14.4f}", end="")
            else:
                print(f"  {val:>14.2e}", end="")
        print()

    print(f"\n  τ* = d/√L = {tau_emp:.3f}")


if __name__ == "__main__":
    main()
