#!/usr/bin/env python3
"""
Verify GPT-5's softmax transport theory for τ* ∝ d_head/√L.

Core claim: The missing 1/L factor comes from post-softmax attention transport,
not from the pre-softmax collision kernel.

Proposed objective:
  F(τ) = (1/2M) ∫(ρ_τ - 1)² dφ  +  λ(M/L) ∫ q(L b^{-φ}) ρ_τ(φ) dφ

where:
  q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)²
  η(φ) = (1-φ)²/2 - 1/6
  ∫η²dφ = 1/45

Key prediction:
  benefit ~ (M/L)τ²,  cost ~ τ⁴/M
  → τ* ~ M/√L = d_head/(2√L)
"""
import numpy as np
import time


# ========== EVQ-cosh family ==========

def rho_tau(phi, tau):
    """Cosh density: ρ_τ(φ) = τ cosh(τ(1-φ)) / sinh(τ)"""
    if abs(tau) < 1e-10:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)


def evq_cosh_positions(tau, K):
    if abs(tau) < 1e-10:
        return np.linspace(0.5/K, 1 - 0.5/K, K)
    u = np.linspace(0.5/K, 1 - 0.5/K, K)
    phi = 1.0 - (1.0/tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
    return np.clip(phi, 1e-6, 1 - 1e-6)


# ========== GPT-5's q(x) function ==========

def q_func(x):
    """q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)²
    Transport energy of a single frequency channel.
    q(x) → x⁴/45 for x≪1 (dead channel)
    q(x) → 1/2  for x≫1 (alive channel)
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    small = np.abs(x) < 1e-6
    large = ~small

    # Small x: Taylor expansion q ≈ x⁴/45
    result[small] = x[small]**4 / 45

    # General case
    xl = x[large]
    sinx = np.sin(xl)
    result[large] = 0.5 + np.sin(2*xl)/(4*xl) - (sinx/xl)**2

    return result


# ========== Verify q(x) by direct numerical integration ==========

def q_func_numerical(x, n_pts=2000):
    """Verify: q(x) = ∫₀¹ (cos(xt) - sin(x)/x)² dt"""
    t = np.linspace(0, 1, n_pts)
    dt = t[1] - t[0]

    if abs(x) < 1e-10:
        return 0.0

    mean = np.sin(x) / x
    integrand = (np.cos(x * t) - mean)**2
    return np.sum(integrand) * dt


# ========== η(φ) and its properties ==========

def eta(phi):
    """η(φ) = (1-φ)²/2 - 1/6  (second-order expansion coefficient)"""
    return (1 - phi)**2 / 2 - 1.0/6


# ========== GPT-5's proposed objective ==========

def softmax_transport_objective(tau, M, L, b, lam, n_phi=500):
    """
    F(τ) = (1/2M) ∫(ρ_τ - 1)² dφ  +  λ(M/L) ∫ q(L b^{-φ}) ρ_τ(φ) dφ

    Note: second term has NEGATIVE sign in optimization (it's utility, we want to maximize it)
    So: F(τ) = stiffness - λ × utility
    """
    phi = np.linspace(0, 1, n_phi)
    dphi = phi[1] - phi[0]

    rho = rho_tau(phi, tau)

    # Term 1: stiffness (cost of deviating from geometric)
    stiffness = np.sum((rho - 1)**2) * dphi / (2 * M)

    # Term 2: softmax-weighted transport utility
    omega = b ** (-phi)
    x = omega * L  # = L b^{-φ}
    q_vals = q_func(x)
    utility = np.sum(q_vals * rho) * dphi * M / L

    # Minimize stiffness - λ × utility (we want high utility, low stiffness)
    return stiffness - lam * utility


def golden_min(f, a, b_bound, tol=1e-5, maxiter=100):
    gr = (np.sqrt(5) + 1) / 2
    c = b_bound - (b_bound - a) / gr
    d = a + (b_bound - a) / gr
    for _ in range(maxiter):
        if abs(b_bound - a) < tol: break
        if f(c) < f(d): b_bound = d
        else: a = c
        c = b_bound - (b_bound - a) / gr
        d = a + (b_bound - a) / gr
    return (a + b_bound) / 2


def main():
    t0 = time.time()
    b = 500_000

    print("=" * 80)
    print("VERIFICATION OF GPT-5's SOFTMAX TRANSPORT THEORY")
    print("=" * 80)

    # ===== Step 0: Verify q(x) =====
    print("\n[0] Verify q(x) closed-form vs numerical integration")
    for x in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0]:
        qc = q_func(x)
        qn = q_func_numerical(x)
        print(f"  x={x:>6.2f}: q_closed={qc:.6f}, q_numerical={qn:.6f}, diff={abs(qc-qn):.2e}")

    # ===== Step 1: Verify η(φ) expansion =====
    print("\n[1] Verify ρ_τ ≈ 1 + τ²η(φ) for small τ")
    phi_test = np.linspace(0, 1, 100)
    for tau_test in [0.1, 0.5, 1.0]:
        rho_exact = rho_tau(phi_test, tau_test)
        rho_approx = 1 + tau_test**2 * eta(phi_test)
        max_err = np.max(np.abs(rho_exact - rho_approx))
        print(f"  τ={tau_test}: max|ρ_exact - (1+τ²η)| = {max_err:.6f}")

    # Verify ∫η²dφ = 1/45
    int_eta2 = np.sum(eta(phi_test)**2) * (phi_test[1]-phi_test[0])
    print(f"\n  ∫η²dφ = {int_eta2:.6f} (theory: {1/45:.6f})")

    # ===== Step 2: Verify stiffness ~ τ⁴/(90M) =====
    print("\n[2] Verify stiffness scaling")
    M = 32
    for tau_test in [0.1, 0.3, 0.5, 0.8, 1.0]:
        phi = np.linspace(0, 1, 1000)
        dphi = phi[1] - phi[0]
        rho = rho_tau(phi, tau_test)
        stiff_exact = np.sum((rho - 1)**2) * dphi / (2*M)
        stiff_theory = tau_test**4 / (90*M)
        print(f"  τ={tau_test}: stiff_exact={stiff_exact:.6e}, τ⁴/(90M)={stiff_theory:.6e}, "
              f"ratio={stiff_exact/stiff_theory:.4f}")

    # ===== Step 3: Check Q₁(L) =====
    print("\n[3] Q₁(L) = ∫ η(φ) q(L b^{-φ}) dφ  (should be Θ(1))")
    phi = np.linspace(0, 1, 2000)
    dphi = phi[1] - phi[0]
    eta_vals = eta(phi)
    for L in [128, 256, 512, 1024, 2048, 4096, 8192]:
        x = L * b ** (-phi)
        q_vals = q_func(x)
        Q1 = np.sum(eta_vals * q_vals) * dphi
        print(f"  L={L:>5d}: Q₁ = {Q1:.6f}")

    # ===== Step 4: THE KEY TEST — optimize the proposed objective =====
    print("\n" + "=" * 80)
    print("[4] KEY TEST: Optimize GPT-5's softmax transport objective")
    print("    F(τ) = stiffness(τ) - λ × utility(τ)")
    print("=" * 80)

    d_head = 64
    M = d_head // 2  # = 32
    Ls = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Try multiple λ values
    for lam in [0.001, 0.01, 0.1, 0.5, 1.0]:
        print(f"\n  λ = {lam}")
        print(f"  {'L':>6s} {'τ_opt':>10s} {'τ*=d/√L':>10s} {'ratio':>8s}")
        print("  " + "-" * 40)

        taus = []
        for L in Ls:
            obj = lambda tau, _L=L: softmax_transport_objective(tau, M, _L, b, lam)
            tau_opt = golden_min(obj, 0.01, 15.0, tol=1e-5)
            tau_emp = d_head / np.sqrt(L)
            taus.append(tau_opt)
            print(f"  {L:>6d} {tau_opt:>10.4f} {tau_emp:>10.4f} {tau_opt/tau_emp:>8.3f}")

        logL = np.log(Ls)
        log_tau = np.log(np.maximum(taus, 0.001))
        slope = np.polyfit(logL, log_tau, 1)[0]
        print(f"  → L-exponent: {slope:.4f}  (target: -0.500)")

    # ===== Step 5: Find the λ that best matches empirical τ* =====
    print("\n" + "=" * 80)
    print("[5] Find optimal λ that matches τ* = d_head/√L")
    print("=" * 80)

    best_gap = 999
    best_lam = 0

    for lam_exp in np.linspace(-4, 2, 100):
        lam = 10**lam_exp
        taus = []
        for L in Ls:
            obj = lambda tau, _L=L, _lam=lam: softmax_transport_objective(tau, M, _L, b, _lam)
            tau_opt = golden_min(obj, 0.01, 15.0, tol=1e-4)
            taus.append(tau_opt)

        logL = np.log(Ls)
        slope = np.polyfit(logL, np.log(np.maximum(taus, 0.001)), 1)[0]
        gap = abs(slope + 0.5)

        if gap < best_gap:
            best_gap = gap
            best_lam = lam
            best_slope = slope
            best_taus = taus.copy()

    print(f"  Best λ = {best_lam:.6f}")
    print(f"  Best slope = {best_slope:.4f} (target: -0.500, gap: {best_gap:.4f})")
    print(f"\n  {'L':>6s} {'τ_opt':>10s} {'τ*=d/√L':>10s} {'ratio':>8s}")
    print("  " + "-" * 40)
    for i, L in enumerate(Ls):
        tau_emp = d_head / np.sqrt(L)
        print(f"  {L:>6d} {best_taus[i]:>10.4f} {tau_emp:>10.4f} {best_taus[i]/tau_emp:>8.3f}")

    # ===== Step 6: Check d_head scaling =====
    print("\n" + "=" * 80)
    print("[6] d_head SCALING (L=2048)")
    print("=" * 80)

    L_test = 2048
    lam = best_lam
    d_heads = [32, 64, 128, 256]
    d_taus = []

    print(f"  Using λ = {lam:.6f}")
    print(f"\n  {'d_head':>8s} {'M':>4s} {'τ_opt':>10s} {'d/√L':>10s} {'ratio':>8s}")
    print("  " + "-" * 45)

    for d in d_heads:
        M_d = d // 2
        obj = lambda tau, _M=M_d: softmax_transport_objective(tau, _M, L_test, b, lam)
        t = golden_min(obj, 0.01, 15.0, tol=1e-5)
        te = d / np.sqrt(L_test)
        d_taus.append(t)
        print(f"  {d:>8d} {M_d:>4d} {t:>10.4f} {te:>10.4f} {t/te:>8.3f}")

    d_slope = np.polyfit(np.log(d_heads), np.log(np.maximum(d_taus, 0.001)), 1)[0]
    print(f"\n  τ ~ d_head^{d_slope:.3f} (expected: 1.0)")

    # ===== Step 7: Analytical prediction vs numerical =====
    print("\n" + "=" * 80)
    print("[7] ANALYTICAL PREDICTION: τ* = √(45 λ Q₁ M²/L)")
    print("=" * 80)

    phi = np.linspace(0, 1, 2000)
    dphi = phi[1] - phi[0]
    eta_vals = eta(phi)

    print(f"\n  {'L':>6s} {'Q₁':>10s} {'τ_analytic':>12s} {'τ_numerical':>12s} {'τ*=d/√L':>10s}")
    print("  " + "-" * 55)

    for i, L in enumerate(Ls):
        x = L * b ** (-phi)
        q_vals = q_func(x)
        Q1 = np.sum(eta_vals * q_vals) * dphi

        tau_analytic = np.sqrt(45 * best_lam * Q1 * M**2 / L) if Q1 > 0 else 0
        tau_num = best_taus[i]
        tau_emp = d_head / np.sqrt(L)

        print(f"  {L:>6d} {Q1:>10.6f} {tau_analytic:>12.4f} {tau_num:>12.4f} {tau_emp:>10.4f}")

    # ===== VERDICT =====
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if best_gap < 0.05:
        print(f"  ★★★ GPT-5 THEORY CONFIRMED: L-exponent = {best_slope:.4f} (gap {best_gap:.4f})")
        print("  The softmax transport objective recovers L^{-0.5}!")
    elif best_gap < 0.15:
        print(f"  ★★ PARTIALLY CONFIRMED: L-exponent = {best_slope:.4f} (gap {best_gap:.4f})")
        print("  Significantly better than kernel-only L^{-0.17}")
    else:
        print(f"  NOT CONFIRMED: L-exponent = {best_slope:.4f} (gap {best_gap:.4f})")

    print(f"\n  Comparison:")
    print(f"    kernel-only (uniform surrogate)     : L^{{-0.085}}")
    print(f"    kernel-only (self-consistent)        : L^{{-0.172}}")
    print(f"    GPT-5 softmax transport              : L^{{{best_slope:.4f}}}")
    print(f"    empirical                            : L^{{-0.500}}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
