#!/usr/bin/env python3
"""
REFINED τ* FORMULA — explore multiple correction approaches.

From the exact balance equation, we know:
  - Leading order: τ₀ = √(45λQ₁) × M/√L  (gives L^{-0.5})
  - Full solution: L^{-0.626} (steeper due to higher-order C(τ) terms)

Goal: find the cleanest practical formula that captures the exact solution.
"""
import numpy as np
import time

b_rope = 500_000

def rho_tau(phi, tau):
    if abs(tau) < 1e-10:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)

def q_func(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-6
    large = ~small
    result[small] = x[small]**4 / 45
    xl = x[large]
    result[large] = 0.5 + np.sin(2*xl)/(4*xl) - (np.sin(xl)/xl)**2
    return result

def C_exact(tau):
    if abs(tau) < 1e-10:
        return tau**4 / 90
    s = np.sinh(tau)
    return tau**2 / (4*s**2) + tau/(4*np.tanh(tau)) - 0.5

def C_deriv_exact(tau):
    if abs(tau) < 1e-10:
        return tau**3 * 4 / 90
    s = np.sinh(tau)
    c = np.cosh(tau)
    coth = c / s
    return tau/(4*s**2)*(1 - 2*tau*coth) + coth/4

def U_func(tau, L, n=5000):
    phi = np.linspace(0, 1, n)
    x = L * b_rope**(-phi)
    qv = q_func(x)
    rho = rho_tau(phi, tau)
    return np.trapz(qv * rho, phi)

def U_deriv(tau, L, n=5000, dtau=1e-6):
    return (U_func(tau+dtau, L, n) - U_func(tau-dtau, L, n)) / (2*dtau)

def find_tau_star(M, L, lam, tol=1e-6):
    def g(tau):
        return C_deriv_exact(tau) * L / (lam * M**2) - U_deriv(tau, L)
    a, b_hi = 0.001, 20.0
    for _ in range(200):
        mid = (a + b_hi) / 2
        if g(mid) < 0:
            a = mid
        else:
            b_hi = mid
        if b_hi - a < tol:
            break
    return (a + b_hi) / 2


def main():
    t0 = time.time()
    d_head = 64
    M = d_head // 2
    Ls = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Calibrate λ
    target_tau = d_head / np.sqrt(2048)
    cd = C_deriv_exact(target_tau)
    ud = U_deriv(target_tau, 2048)
    lam = cd * 2048 / (M**2 * ud)

    # Get exact solutions
    tau_exact = []
    for L in Ls:
        tau_exact.append(find_tau_star(M, L, lam))

    tau_exact = np.array(tau_exact)
    te_arr = np.array([d_head / np.sqrt(L) for L in Ls])

    print("=" * 80)
    print("REFINED τ* FORMULA SEARCH")
    print("=" * 80)

    # ===== APPROACH 1: Power law τ* = A × s^α where s = d/√L =====
    print("\n[A1] Power law: τ* = A × (d/√L)^α")
    pfit = np.polyfit(np.log(te_arr), np.log(tau_exact), 1)
    A1, alpha1 = np.exp(pfit[1]), pfit[0]
    tau_a1 = A1 * te_arr**alpha1
    err_a1 = np.abs(tau_a1 - tau_exact) / tau_exact * 100
    print(f"  A={A1:.4f}, α={alpha1:.4f}")
    print(f"  Max error: {np.max(err_a1):.2f}%, RMSE: {np.sqrt(np.mean((tau_a1/tau_exact - 1)**2))*100:.2f}%")

    # ===== APPROACH 2: Reciprocal correction τ* = s / (1 + a·s^2) =====
    print("\n[A2] Rational correction: τ* = s / (1 + a·s²)")
    # τ*/s = 1/(1+a·s²), so a = (s/τ* - 1) / s²
    a_vals = (te_arr / tau_exact - 1) / te_arr**2
    print(f"  a values: {[f'{x:.4f}' for x in a_vals]}")
    # This form doesn't work well because ratio > 1 at small L

    # ===== APPROACH 3: τ* = s × (1 + a·s²) (upward correction for large s) =====
    print("\n[A3] Polynomial correction: τ* = s × (1 + a·s² + b·s⁴)")
    ratio = tau_exact / te_arr
    # fit: ratio = 1 + a·s² + b·s⁴
    S2 = te_arr**2
    S4 = te_arr**4
    A_mat = np.column_stack([S2, S4])
    ab_fit = np.linalg.lstsq(A_mat, ratio - 1, rcond=None)[0]
    a3, b3 = ab_fit
    tau_a3 = te_arr * (1 + a3*S2 + b3*S4)
    err_a3 = np.abs(tau_a3 - tau_exact) / tau_exact * 100
    print(f"  τ* = s × (1 + {a3:.5f}·s² + {b3:.7f}·s⁴)")
    print(f"  Max error: {np.max(err_a3):.2f}%, RMSE: {np.sqrt(np.mean((tau_a3/tau_exact - 1)**2))*100:.2f}%")
    for i, L in enumerate(Ls):
        print(f"    L={L:>5d}: τ*={tau_exact[i]:.4f}, fit={tau_a3[i]:.4f}, err={err_a3[i]:.2f}%")

    # ===== APPROACH 4: [1,1] Padé: τ* = s(1 + a·s) / (1 + c·s) =====
    print("\n[A4] [1,1] Padé: τ* = s(1 + a·s) / (1 + c·s)")
    # ratio = (1 + a·s)/(1 + c·s)
    # ratio + ratio·c·s = 1 + a·s
    # a·s - ratio·c·s = ratio - 1
    # s(a - ratio·c) = ratio - 1
    A_mat4 = np.column_stack([te_arr, -ratio * te_arr])
    ac_fit = np.linalg.lstsq(A_mat4, ratio - 1, rcond=None)[0]
    a4, c4 = ac_fit
    tau_a4 = te_arr * (1 + a4*te_arr) / (1 + c4*te_arr)
    err_a4 = np.abs(tau_a4 - tau_exact) / tau_exact * 100
    print(f"  a={a4:.5f}, c={c4:.5f}")
    print(f"  Max error: {np.max(err_a4):.2f}%, RMSE: {np.sqrt(np.mean((tau_a4/tau_exact - 1)**2))*100:.2f}%")
    for i, L in enumerate(Ls):
        print(f"    L={L:>5d}: τ*={tau_exact[i]:.4f}, fit={tau_a4[i]:.4f}, err={err_a4[i]:.2f}%")

    # ===== APPROACH 5: [2,0] Padé with s^(1+ε) tweak =====
    # Actually let's try the self-consistent approach: what if we treat τ itself as a correction parameter?
    # τ_next = √(45λQ₁(τ)M²/L × correction(τ))
    # where correction captures the ratio C'(τ)/(2τ³/45) and U'(τ)/(2Q₁τ)

    print("\n[A5] Self-consistent correction analysis")
    print("  Decompose: C'(τ)/[2τ³/45] and U'(τ,L)/[2Q₁(L)τ]")
    phi_grid = np.linspace(0, 1, 5000)
    eta_v = (1 - phi_grid)**2/2 - 1.0/6

    for i, L in enumerate(Ls):
        tau = tau_exact[i]
        x = L * b_rope**(-phi_grid)
        Q1 = np.trapz(eta_v * q_func(x), phi_grid)

        c_ratio = C_deriv_exact(tau) / (2*tau**3/45)  # should → 1 as τ→0
        u_ratio = U_deriv(tau, L) / (2*Q1*tau) if Q1 > 0 else 0  # should → 1 as τ→0
        print(f"    L={L:>5d}, τ*={tau:.4f}: C'_ratio={c_ratio:.4f}, U'_ratio={u_ratio:.4f}, "
              f"combined={c_ratio/u_ratio:.4f}")

    # ===== APPROACH 6: Define σ = d_head²/L (natural variable) =====
    print("\n[A6] Natural variable σ = d²/L")
    sigma = np.array([d_head**2 / L for L in Ls])
    # τ* vs σ
    pfit6 = np.polyfit(np.log(sigma), np.log(tau_exact), 1)
    print(f"  τ* ∝ σ^{pfit6[0]:.4f}  (leading order: σ^0.5, i.e., d/√L)")
    # So exponent in σ is 0.5 × α where α is the exponent in d/√L

    # ===== APPROACH 7: Two-regime formula =====
    print("\n[A7] Two-regime formula")
    print("  Small s (s < 1): τ* ≈ s  (leading order)")
    print("  Large s (s > 2): check behavior")

    # For the paper, the cleanest formula might be the implicit one.
    # Let's define the "effective exponent" function.
    print("\n[A8] Effective exponent function γ(s) where τ* = s^γ(s)")
    for i, L in enumerate(Ls):
        s = te_arr[i]
        if abs(np.log(s)) > 1e-6:
            gamma = np.log(tau_exact[i]) / np.log(s)
        else:
            gamma = 1.0
        print(f"    L={L:>5d}, s={s:.3f}: γ={gamma:.4f}")

    # ===== APPROACH 9: Closed-form via C'/C ratio =====
    print("\n" + "=" * 80)
    print("[A9] PHYSICAL FORMULA: using C and U structure")
    print("=" * 80)

    # The balance equation is:
    #   C'(τ) = λ M² U'(τ,L) / L
    # For small τ: C'(τ) ≈ 2τ³/45, U'(τ) ≈ 2Q₁τ
    # → τ₀² = 45λQ₁M²/L

    # For finite τ, define:
    #   c(τ) = C'(τ)/(2τ³/45)   (stiffness correction, → 1 as τ→0)
    #   u(τ) = U'(τ,L)/(2Q₁τ)   (utility correction, → 1 as τ→0)
    # Then balance becomes:
    #   c(τ) × (2τ³/45) = λM²/L × u(τ) × 2Q₁τ
    #   c(τ)/u(τ) × τ² = 45λQ₁M²/L = τ₀²
    #   τ² = τ₀² × u(τ)/c(τ)

    # So: τ* = τ₀ × √(u(τ*)/c(τ*))

    # Can we express c(τ) and u(τ) analytically?
    # c(τ) = C'(τ) × 45/(2τ³)
    # C'(τ) = τ/(4sinh²τ)(1 - 2τcothτ) + cothτ/4
    # c(τ) = 45/(2τ³) × [τ/(4sinh²τ)(1 - 2τcothτ) + cothτ/4]

    print("\n  c(τ) = C'(τ) × 45/(2τ³)  [stiffness correction factor]:")
    for tau in [0.01, 0.1, 0.5, 1.0, 1.414, 2.0, 3.0, 5.0, 8.0]:
        c_val = C_deriv_exact(tau) * 45 / (2 * tau**3)
        print(f"    c({tau:.3f}) = {c_val:.6f}")

    # c(τ) can be fit with a simple function
    tau_test = np.linspace(0.01, 10, 1000)
    c_vals = np.array([C_deriv_exact(t) * 45 / (2 * t**3) for t in tau_test])

    # Try: c(τ) ≈ 1/(1 + τ²/a)  or  c(τ) ≈ sech(τ/a)
    # At τ=0: c=1. At large τ: C'→1/4, so c→45/(8τ³)→0. So c decays.

    # Let's try c(τ) ≈ (3τ/sinh(τ))^(2/3) or similar hyperbolic form
    # Actually let's just numerically check what c and u do

    print("\n  u(τ,L) for L=2048:")
    for tau in [0.01, 0.1, 0.5, 1.0, 1.414, 2.0, 3.0, 5.0]:
        x = 2048 * b_rope**(-phi_grid)
        Q1 = np.trapz(eta_v * q_func(x), phi_grid)
        ud = U_deriv(tau, 2048)
        u_val = ud / (2 * Q1 * tau) if tau > 1e-6 else 1.0
        print(f"    u({tau:.3f}, 2048) = {u_val:.6f}")

    # ===== FINAL: Best practical formula =====
    print("\n" + "=" * 80)
    print("BEST PRACTICAL FORMULAS")
    print("=" * 80)

    # 1. Implicit (exact): solve C'(τ) = λM²U'(τ,L)/L
    # 2. Power law (max 3% error): τ* ≈ 0.94 × (d/√L)^1.25
    # 3. Polynomial correction (better):

    # Let's try: τ* = s × exp(a·s²) where s = d/√L
    print("\n  [F1] Exponential correction: τ* = s × exp(a·s²)")
    log_ratio = np.log(tau_exact / te_arr)
    a_exp = np.linalg.lstsq(S2.reshape(-1,1), log_ratio, rcond=None)[0][0]
    tau_f1 = te_arr * np.exp(a_exp * S2)
    err_f1 = np.abs(tau_f1 - tau_exact) / tau_exact * 100
    print(f"  a = {a_exp:.5f}")
    print(f"  Max error: {np.max(err_f1):.2f}%, RMSE: {np.sqrt(np.mean((tau_f1/tau_exact - 1)**2))*100:.2f}%")
    for i, L in enumerate(Ls):
        print(f"    L={L:>5d}: τ*={tau_exact[i]:.4f}, fit={tau_f1[i]:.4f}, err={err_f1[i]:.2f}%")

    # Try with two params: τ* = s × exp(a·s + b·s²)
    print("\n  [F2] τ* = s × exp(a·s + b·s²)")
    A_mat_f2 = np.column_stack([te_arr, S2])
    ab_f2 = np.linalg.lstsq(A_mat_f2, log_ratio, rcond=None)[0]
    a_f2, b_f2 = ab_f2
    tau_f2 = te_arr * np.exp(a_f2*te_arr + b_f2*S2)
    err_f2 = np.abs(tau_f2 - tau_exact) / tau_exact * 100
    print(f"  a = {a_f2:.5f}, b = {b_f2:.6f}")
    print(f"  Max error: {np.max(err_f2):.2f}%, RMSE: {np.sqrt(np.mean((tau_f2/tau_exact - 1)**2))*100:.2f}%")
    for i, L in enumerate(Ls):
        print(f"    L={L:>5d}: τ*={tau_exact[i]:.4f}, fit={tau_f2[i]:.4f}, err={err_f2[i]:.2f}%")

    # ===== MULTI d_head validation =====
    print("\n" + "=" * 80)
    print("MULTI d_head VALIDATION of best formula")
    print("=" * 80)

    for d in [32, 64, 128]:
        Md = d // 2
        print(f"\n  d_head = {d}:")
        taus_d = []
        for L in Ls:
            t = find_tau_star(Md, L, lam)
            s = d / np.sqrt(L)
            t_pred = s * (1 + a3*s**2 + b3*s**4)  # A3 formula
            t_pw = A1 * s**alpha1  # A1 power law
            taus_d.append(t)
            print(f"    L={L:>5d}: exact={t:.4f}, A3={t_pred:.4f} ({(t_pred/t-1)*100:+.1f}%), "
                  f"PL={t_pw:.4f} ({(t_pw/t-1)*100:+.1f}%)")

        slope_d = np.polyfit(np.log(Ls), np.log(taus_d), 1)[0]
        print(f"    L-exponent: {slope_d:.4f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
