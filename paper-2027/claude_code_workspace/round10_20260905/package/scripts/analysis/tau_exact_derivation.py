#!/usr/bin/env python3
"""
RIGOROUS DERIVATION: τ* from softmax transport, no truncation.

=== Theory chain ===

GIVEN (exact):
  ρ_τ(φ) = τ cosh(τ(1-φ)) / sinh(τ)               ... EVQ-cosh family
  q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)²         ... single-channel transport energy
  J₀ = (1/L)Π₀                                      ... softmax Jacobian at uniform baseline

OBJECTIVE (exact):
  F(τ) = C(τ)/M  -  λ(M/L) U(τ,L)

where:
  C(τ) = (1/2)∫₀¹ (ρ_τ - 1)² dφ                    ... stiffness
  U(τ,L) = ∫₀¹ q(Lb^{-φ}) ρ_τ(φ) dφ                ... transport utility

STEP 1: Prove C(τ) = τ²/(4sinh²τ) + τcothτ/4 - 1/2  (exact closed form)
STEP 2: Compute U(τ,L) and U'(τ,L) numerically (no closed form)
STEP 3: Solve C'(τ*)/M = λ(M/L)U'(τ*,L) exactly (implicit equation)
STEP 4: Derive corrected τ* formula with next-order terms
STEP 5: Validate against numerical optimization
"""
import numpy as np
import time

b_rope = 500_000

# ================================================================
# EXACT FUNCTIONS
# ================================================================

def rho_tau(phi, tau):
    if abs(tau) < 1e-10:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)

def rho_tau_deriv(phi, tau, dtau=1e-7):
    """∂ρ_τ/∂τ by finite difference."""
    return (rho_tau(phi, tau+dtau) - rho_tau(phi, tau-dtau)) / (2*dtau)

def q_func(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-6
    large = ~small
    result[small] = x[small]**4 / 45
    xl = x[large]
    result[large] = 0.5 + np.sin(2*xl)/(4*xl) - (np.sin(xl)/xl)**2
    return result


# ================================================================
# STEP 1: C(τ) exact closed form
# ================================================================

def C_exact(tau):
    """C(τ) = (1/2)∫₀¹(ρ_τ - 1)²dφ  — exact closed form.

    Derivation:
      ∫₀¹ ρ_τ² dφ = (τ²/sinh²τ) × (1/τ) ∫₀^τ cosh²(x) dx
                   = (τ/sinh²τ) × [τ/2 + sinh(2τ)/4]
                   = τ²/(2sinh²τ) + τ·2sinhτcoshτ/(4sinh²τ)
                   = τ²/(2sinh²τ) + τcothτ/2

      ∫₀¹ ρ_τ dφ = 1  (normalized)

      C = (1/2)(∫ρ² - 2∫ρ + 1) = (1/2)(∫ρ² - 1)
        = τ²/(4sinh²τ) + τcothτ/4 - 1/2
    """
    if abs(tau) < 1e-10:
        return tau**4 / 90  # leading order
    s = np.sinh(tau)
    return tau**2 / (4*s**2) + tau/(4*np.tanh(tau)) - 0.5

def C_numerical(tau, n=100000):
    """High-resolution numerical verification."""
    phi = np.linspace(0, 1, n)
    dphi = 1.0 / (n - 1)
    rho = rho_tau(phi, tau)
    return 0.5 * np.trapz((rho - 1)**2, phi)

def C_deriv_exact(tau):
    """C'(τ) — exact closed form.

    C(τ) = τ²/(4sinh²τ) + τcothτ/4 - 1/2

    d/dτ[τ²/(4sinh²τ)] = [2τsinh²τ - τ²·2sinhτcoshτ] / (4sinh⁴τ)
                        = τ/(2sinh²τ) - τ²coshτ/(2sinh³τ)
                        = τ/(2sinh²τ)(1 - τcothτ)

    d/dτ[τcothτ/4] = (1/4)[cothτ + τ(-1/sinh²τ)]
                    = (1/4)[cothτ - τ/sinh²τ]

    C'(τ) = τ/(2sinh²τ) - τ²cothτ/(2sinh²τ) + cothτ/4 - τ/(4sinh²τ)
           = τ/(4sinh²τ) - τ²cothτ/(2sinh²τ) + cothτ/4
           = τ/(4sinh²τ)(1 - 2τcothτ) + cothτ/4
    """
    if abs(tau) < 1e-10:
        return tau**3 * 4 / 90  # = 2τ³/45
    s = np.sinh(tau)
    c = np.cosh(tau)
    coth = c / s
    return tau/(4*s**2)*(1 - 2*tau*coth) + coth/4

def C_deriv_numerical(tau, eps=1e-6):
    return (C_exact(tau+eps) - C_exact(tau-eps)) / (2*eps)


# ================================================================
# STEP 2: U(τ,L) and U'(τ,L) — numerical
# ================================================================

def U_func(tau, L, n=5000):
    """U(τ,L) = ∫₀¹ q(Lb^{-φ}) ρ_τ(φ) dφ"""
    phi = np.linspace(0, 1, n)
    x = L * b_rope**(-phi)
    qv = q_func(x)
    rho = rho_tau(phi, tau)
    return np.trapz(qv * rho, phi)

def U_deriv(tau, L, n=5000, dtau=1e-6):
    """∂U/∂τ by central difference."""
    return (U_func(tau+dtau, L, n) - U_func(tau-dtau, L, n)) / (2*dtau)


# ================================================================
# STEP 3: Exact balance equation
# C'(τ*)/M = λ(M/L) U'(τ*,L)
# → C'(τ*) = λ M²/L × U'(τ*,L)
# ================================================================

def balance_residual(tau, M, L, lam):
    """C'(τ) - λ(M²/L)U'(τ,L).  Zero at τ*."""
    return C_deriv_exact(tau) - lam * M**2 / L * U_deriv(tau, L)

def find_tau_star(M, L, lam, tol=1e-6):
    """Find τ* by bisection on the balance equation."""
    # At small τ: C' ~ 2τ³/45 (growing), λM²U'/L ~ 2λM²Q₁τ/L (growing slower)
    # At large τ: C' → 1/4, U' → 0. So C' > λM²U'/L eventually.
    # At τ=0: both are 0, but C'' > 0 and U'' might also > 0
    # We want F'(τ) = C'(τ)/M - λ(M/L)U'(τ,L) = 0

    # Actually F(τ) = C(τ)/M - λ(M/L)U(τ,L)
    # F'(τ) = C'(τ)/M - λ(M/L)U'(τ,L)

    # We need the MINIMUM, so F'(τ*) = 0, F''(τ*) > 0
    # C'(0) = 0, U'(0) = 0. For τ small:
    #   C'(τ) ≈ 2τ³/45, U'(τ) ≈ 2Q₁τ
    #   F'(τ) ≈ 2τ³/(45M) - 2λMQ₁τ/L = 2τ[τ²/(45M) - λMQ₁/L]
    #   F'(τ) = 0 → τ = 0 or τ² = 45λMQ₁M/L = 45λQ₁M²/L

    # So τ* is where C'(τ)/M = λ(M/L)U'(τ,L)
    # i.e., C'(τ) = λM²U'(τ,L)/L

    a, b_hi = 0.001, 20.0
    # Find sign change: g(τ) = C'(τ)×L/(λM²) - U'(τ,L)
    def g(tau):
        return C_deriv_exact(tau) * L / (lam * M**2) - U_deriv(tau, L)

    # g(small τ) ≈ (2τ³/45)×L/(λM²) - 2Q₁τ
    # At very small τ: g < 0 (U' dominates since τ³ << τ)
    # At large τ: g > 0 (C' → const, U' → 0)

    # Bisection
    for _ in range(200):
        mid = (a + b_hi) / 2
        if g(mid) < 0:
            a = mid
        else:
            b_hi = mid
        if b_hi - a < tol:
            break
    return (a + b_hi) / 2


# ================================================================
# STEP 4: Full objective optimization (golden section, for comparison)
# ================================================================

def full_objective(tau, M, L, lam):
    return C_exact(tau) / M - lam * M / L * U_func(tau, L)

def golden_min(f, a, b, tol=1e-5, maxiter=100):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in range(maxiter):
        if abs(b - a) < tol: break
        if f(c) < f(d): b = d
        else: a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (a + b) / 2


def main():
    t0 = time.time()
    d_head = 64
    M = d_head // 2

    print("=" * 80)
    print("EXACT τ* DERIVATION — NO TRUNCATION")
    print("=" * 80)

    # ===== STEP 1: Verify C(τ) closed form at high precision =====
    print("\n[STEP 1] C(τ) exact closed form verification (n=100000)")
    print(f"  {'τ':>6s} {'C_closed':>14s} {'C_numerical':>14s} {'rel_err':>12s}")
    all_ok = True
    for tau in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        cc = C_exact(tau)
        cn = C_numerical(tau, n=100000)
        re = abs(cc - cn) / max(abs(cc), 1e-30)
        ok = re < 1e-4
        if not ok: all_ok = False
        print(f"  {tau:>6.2f} {cc:>14.8f} {cn:>14.8f} {re:>12.2e} {'✓' if ok else '✗'}")
    print(f"  C(τ) closed form: {'CONFIRMED ✓' if all_ok else 'CHECK NEEDED'}")

    # Verify C'(τ) closed form
    print(f"\n  C'(τ) verification:")
    for tau in [0.5, 1.0, 2.0, 5.0]:
        ca = C_deriv_exact(tau)
        cn = C_deriv_numerical(tau)
        print(f"    C'({tau}) = {ca:.8f}  (numerical: {cn:.8f}, diff: {abs(ca-cn):.2e})")

    # ===== STEP 2-3: Solve exact balance for each L =====
    print("\n" + "=" * 80)
    print("[STEP 2-3] Solve EXACT balance equation: C'(τ*) = λ(M²/L)U'(τ*,L)")
    print("=" * 80)

    Ls = [128, 256, 512, 1024, 2048, 4096, 8192]

    # First, find λ that makes τ*(L=2048) ≈ d_head/√2048 ≈ 1.414
    target_tau = d_head / np.sqrt(2048)
    print(f"\n  Calibrating λ so that τ*(L=2048) = {target_tau:.4f}")

    # At τ* = 1.414: C'(1.414) = λ(M²/2048)U'(1.414, 2048)
    # λ = C'(1.414) × 2048 / (M² × U'(1.414, 2048))
    cd = C_deriv_exact(target_tau)
    ud = U_deriv(target_tau, 2048)
    lam_cal = cd * 2048 / (M**2 * ud)
    print(f"  C'({target_tau:.3f}) = {cd:.8f}")
    print(f"  U'({target_tau:.3f}, 2048) = {ud:.8f}")
    print(f"  → λ = {lam_cal:.6f}")

    # Now solve exact balance for all L
    print(f"\n  Exact balance solutions (λ = {lam_cal:.6f}):")
    print(f"  {'L':>6s} {'τ_exact':>10s} {'τ_golden':>10s} {'d/√L':>10s} {'τ_ex/emp':>10s} {'match':>6s}")
    print("  " + "-" * 58)

    tau_exact = []
    tau_golden = []
    for L in Ls:
        te_bisect = find_tau_star(M, L, lam_cal)
        te_golden = golden_min(lambda tau, _L=L: full_objective(tau, M, _L, lam_cal), 0.01, 15.0)
        te_emp = d_head / np.sqrt(L)
        tau_exact.append(te_bisect)
        tau_golden.append(te_golden)
        match = abs(te_bisect - te_golden) < 0.01
        print(f"  {L:>6d} {te_bisect:>10.4f} {te_golden:>10.4f} {te_emp:>10.4f} "
              f"{te_bisect/te_emp:>10.4f} {'✓' if match else '✗':>6s}")

    # Fit power law
    logL = np.log(Ls)
    slope_exact = np.polyfit(logL, np.log(tau_exact), 1)[0]
    slope_golden = np.polyfit(logL, np.log(tau_golden), 1)[0]
    slope_emp = -0.5

    print(f"\n  L-exponent (exact balance) : {slope_exact:.4f}")
    print(f"  L-exponent (golden min)    : {slope_golden:.4f}")
    print(f"  L-exponent (empirical)     : {slope_emp:.4f}")

    # ===== STEP 4: Local exponent at each L =====
    print("\n" + "=" * 80)
    print("[STEP 4] LOCAL EXPONENT: d ln τ* / d ln L at each point")
    print("=" * 80)

    print(f"\n  {'L':>6s} {'τ*':>8s} {'local_exp':>12s}")
    print("  " + "-" * 30)
    for i in range(1, len(Ls)-1):
        loc_exp = (np.log(tau_exact[i+1]) - np.log(tau_exact[i-1])) / \
                  (np.log(Ls[i+1]) - np.log(Ls[i-1]))
        print(f"  {Ls[i]:>6d} {tau_exact[i]:>8.4f} {loc_exp:>12.4f}")

    # ===== STEP 5: Corrected formula =====
    print("\n" + "=" * 80)
    print("[STEP 5] CORRECTED τ* FORMULA")
    print("=" * 80)

    # The exact balance is: C'(τ) = λM²U'(τ,L)/L
    # Leading order: τ₀² = 45λQ₁M²/L  (from τ⁴ vs τ² balance)
    # We need a correction: τ* = τ₀ × f(τ₀)
    # where f captures the deviation from the small-τ regime.

    # Strategy: fit τ*/τ₀ as a function of τ₀ (or equivalently of d/√L)
    # to find the correction function.

    phi_grid = np.linspace(0, 1, 5000)
    dphi = phi_grid[1] - phi_grid[0]

    print("\n  Correction factor τ_exact / τ_leading:")
    print(f"  {'L':>6s} {'τ₀':>10s} {'τ_exact':>10s} {'ratio':>10s} {'d/√L':>10s}")
    print("  " + "-" * 48)

    ratios = []
    tau0s = []
    for i, L in enumerate(Ls):
        # Q₁(L)
        eta_v = (1-phi_grid)**2/2 - 1.0/6
        x = L * b_rope**(-phi_grid)
        Q1 = np.trapz(eta_v * q_func(x), phi_grid)

        tau0 = np.sqrt(45 * lam_cal * Q1 * M**2 / L)
        ratio = tau_exact[i] / tau0
        tau0s.append(tau0)
        ratios.append(ratio)
        te = d_head / np.sqrt(L)
        print(f"  {L:>6d} {tau0:>10.4f} {tau_exact[i]:>10.4f} {ratio:>10.4f} {te:>10.4f}")

    # Fit correction: τ* ≈ τ₀ / (1 + c₁τ₀² + c₂τ₀⁴)
    # Since ratio < 1 at large τ₀ and → 1 at small τ₀
    # Try: ratio ≈ 1 - a×τ₀²
    tau0_arr = np.array(tau0s)
    ratio_arr = np.array(ratios)

    # Linear fit: (1-ratio) vs τ₀²
    delta = 1 - ratio_arr
    tau0_sq = tau0_arr**2
    # fit delta = c₁ τ₀² + c₂ τ₀⁴
    A = np.column_stack([tau0_sq, tau0_sq**2])
    coeffs = np.linalg.lstsq(A, delta, rcond=None)[0]
    c1, c2 = coeffs

    print(f"\n  Correction fit: τ*/τ₀ ≈ 1 - {c1:.4f}τ₀² - {c2:.6f}τ₀⁴")

    # Verify correction
    print(f"\n  Verification of corrected formula:")
    print(f"  {'L':>6s} {'τ_exact':>10s} {'τ_corrected':>12s} {'error%':>10s}")
    print("  " + "-" * 45)
    for i, L in enumerate(Ls):
        tau_c = tau0s[i] * (1 - c1*tau0s[i]**2 - c2*tau0s[i]**4)
        err_pct = (tau_c - tau_exact[i]) / tau_exact[i] * 100
        print(f"  {L:>6d} {tau_exact[i]:>10.4f} {tau_c:>12.4f} {err_pct:>+10.2f}%")

    # ===== STEP 6: Final practical formula =====
    print("\n" + "=" * 80)
    print("[STEP 6] PRACTICAL FORMULA")
    print("=" * 80)

    # From the exact analysis, let's also try: τ* = d_head/√L × g(d_head/√L)
    # where g(s) is a correction that depends on s = d/√L itself

    print("\n  Direct fit: τ* = A × (d/√L)^α")
    log_te = np.log(np.array([d_head/np.sqrt(L) for L in Ls]))
    log_tex = np.log(tau_exact)
    pfit = np.polyfit(log_te, log_tex, 1)
    A_fit = np.exp(pfit[1])
    alpha_fit = pfit[0]
    print(f"  τ* = {A_fit:.4f} × (d/√L)^{alpha_fit:.4f}")

    # Check quality
    print(f"\n  {'L':>6s} {'τ_exact':>10s} {'τ_fit':>10s} {'error%':>10s}")
    print("  " + "-" * 35)
    max_err = 0
    for i, L in enumerate(Ls):
        te = d_head / np.sqrt(L)
        tau_f = A_fit * te**alpha_fit
        err = (tau_f - tau_exact[i]) / tau_exact[i] * 100
        max_err = max(max_err, abs(err))
        print(f"  {L:>6d} {tau_exact[i]:>10.4f} {tau_f:>10.4f} {err:>+10.2f}%")
    print(f"  Max error: {max_err:.2f}%")

    # Also try: τ* = (d/√L) × [1 - β(d/√L)²]
    # which is a Padé-like correction
    print("\n  Padé correction: τ* = (d/√L) × [1 - β(d/√L)²]")
    te_arr = np.array([d_head/np.sqrt(L) for L in Ls])
    beta_arr = (1 - np.array(tau_exact)/te_arr) / te_arr**2
    beta_mean = np.mean(beta_arr[2:])  # exclude small-L points where τ is large
    print(f"  β values: {[f'{x:.4f}' for x in beta_arr]}")
    print(f"  β (mean, L≥512): {beta_mean:.4f}")

    # ===== STEP 7: d_head scaling check =====
    print("\n" + "=" * 80)
    print("[STEP 7] d_head SCALING CHECK")
    print("=" * 80)

    L_test = 2048
    d_heads = [32, 64, 128, 256]
    d_taus = []

    for d in d_heads:
        Md = d // 2
        # Recalibrate λ: at each d_head, λ should be the same physical quantity
        # λ captures σ²_pos / (2T²κ), which shouldn't depend on d or M
        # So we use the SAME λ
        t = find_tau_star(Md, L_test, lam_cal)
        te = d / np.sqrt(L_test)
        d_taus.append(t)
        print(f"  d={d:>3d}: τ_exact={t:.4f}, d/√L={te:.4f}, ratio={t/te:.4f}")

    d_slope = np.polyfit(np.log(d_heads), np.log(d_taus), 1)[0]
    print(f"\n  τ ~ d_head^{d_slope:.4f} (expected: 1.0)")

    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
  EXACT RESULTS (from full balance equation, no truncation):

  1. Objective:
     F(τ) = C(τ)/M - λ(M/L)U(τ,L)
     C(τ) = τ²/(4sinh²τ) + τcothτ/4 - 1/2    [exact, verified]
     U(τ,L) = ∫₀¹ q(Lb^{{-φ}}) ρ_τ(φ) dφ      [numerical]

  2. Balance equation:
     C'(τ*) = λ(M²/L) U'(τ*,L)               [implicit, solved by bisection]

  3. Calibration:
     λ = {lam_cal:.6f}  (from matching τ*(2048) = d_head/√2048)

  4. Global L-exponent: {slope_exact:.4f}  (empirical: -0.500)

  5. Practical formula:
     τ* = {A_fit:.4f} × (d_head/√L)^{{{alpha_fit:.4f}}}
     or equivalently:
     τ* ≈ (d_head/√L) × [1 - {beta_mean:.4f}×(d_head/√L)²]

  6. d_head exponent: {d_slope:.4f}  (expected: 1.0)
""")

    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
