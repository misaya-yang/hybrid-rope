#!/usr/bin/env python3
"""
V2: Deep analysis of the coefficient problem.

The L-scaling is PERFECT (L^{-0.497}), but |τ_opt| ≈ 0.014 × |τ*|.
Two questions:
  1. Is there a natural value of λ from the attention mechanism?
  2. Can we match both scaling AND absolute value?

Key insight from GPT-5's derivation:
  τ² ~ 45λ Q₁ M²/L

For τ* = d_head/√L = 2M/√L, we need:
  (2M/√L)² = 45λ Q₁ M²/L
  4M²/L = 45λ Q₁ M²/L
  4 = 45λ Q₁
  λ = 4/(45 × Q₁) ≈ 4/(45 × 0.031) ≈ 2.87

Let's verify this.
"""
import numpy as np
import time


def rho_tau(phi, tau):
    if abs(tau) < 1e-10:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)

def evq_cosh_positions(tau, K):
    if abs(tau) < 1e-10:
        return np.linspace(0.5/K, 1 - 0.5/K, K)
    u = np.linspace(0.5/K, 1 - 0.5/K, K)
    phi = 1.0 - (1.0/tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
    return np.clip(phi, 1e-6, 1 - 1e-6)

def q_func(x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    small = np.abs(x) < 1e-6
    large = ~small
    result[small] = x[small]**4 / 45
    xl = x[large]
    sinx = np.sin(xl)
    result[large] = 0.5 + np.sin(2*xl)/(4*xl) - (sinx/xl)**2
    return result

def eta(phi):
    return (1 - phi)**2 / 2 - 1.0/6

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

def softmax_obj(tau, M, L, b, lam, n_phi=500):
    phi = np.linspace(0, 1, n_phi)
    dphi = phi[1] - phi[0]
    rho = rho_tau(phi, tau)
    stiffness = np.sum((rho - 1)**2) * dphi / (2 * M)
    x = L * b ** (-phi)
    q_vals = q_func(x)
    utility = np.sum(q_vals * rho) * dphi * M / L
    return stiffness - lam * utility


def main():
    t0 = time.time()
    b = 500_000
    d_head = 64
    M = d_head // 2
    Ls = [128, 256, 512, 1024, 2048, 4096, 8192]

    # ===== Q₁ values =====
    phi = np.linspace(0, 1, 2000)
    dphi = phi[1] - phi[0]
    eta_vals = eta(phi)

    Q1_vals = {}
    for L in Ls:
        x = L * b ** (-phi)
        Q1_vals[L] = np.sum(eta_vals * q_func(x)) * dphi

    Q1_mean = np.mean(list(Q1_vals.values()))

    print("=" * 80)
    print("COEFFICIENT ANALYSIS")
    print("=" * 80)

    # For τ* = 2M/√L, need: (2M)² = 45λQ₁M², so λ = 4/(45Q₁)
    lam_natural = 4.0 / (45 * Q1_mean)
    print(f"\n  Q₁ mean = {Q1_mean:.6f}")
    print(f"  λ needed for τ* = d_head/√L: λ = 4/(45×Q₁) = {lam_natural:.4f}")
    print(f"  (Note: d_head/√L = 2M/√L, so τ² = 4M²/L)")

    # ===== Test with natural λ =====
    print(f"\n  Testing with λ = {lam_natural:.4f}:")
    print(f"  {'L':>6s} {'τ_opt':>10s} {'τ*=d/√L':>10s} {'ratio':>8s}")
    print("  " + "-" * 40)

    taus = []
    for L in Ls:
        obj = lambda tau, _L=L: softmax_obj(tau, M, _L, b, lam_natural)
        tau_opt = golden_min(obj, 0.01, 15.0, tol=1e-5)
        tau_emp = d_head / np.sqrt(L)
        taus.append(tau_opt)
        print(f"  {L:>6d} {tau_opt:>10.4f} {tau_emp:>10.4f} {tau_opt/tau_emp:>8.3f}")

    slope = np.polyfit(np.log(Ls), np.log(np.maximum(taus, 0.001)), 1)[0]
    print(f"  → L-exponent: {slope:.4f}")

    # ===== The stiffness term might need rescaling =====
    # GPT-5 wrote (1/2M)∫(ρ-1)²dφ but maybe the natural scale is different
    # Let's try: cost = (1/2M²) ∫(ρ-1)² (or other normalizations)
    print("\n" + "=" * 80)
    print("STIFFNESS NORMALIZATION VARIANTS")
    print("=" * 80)

    def obj_variant(tau, M, L, b, lam, stiff_norm, n_phi=500):
        phi = np.linspace(0, 1, n_phi)
        dphi = phi[1] - phi[0]
        rho = rho_tau(phi, tau)
        stiffness = np.sum((rho - 1)**2) * dphi / stiff_norm
        x = L * b ** (-phi)
        q_vals = q_func(x)
        utility = np.sum(q_vals * rho) * dphi * M / L
        return stiffness - lam * utility

    # The analytical solution gives:
    # τ² = (45 × λ × Q₁ × stiff_norm × M) / (L × 2)
    # (from d/dτ[τ⁴/(90×stiff_norm) - λ(M/L)Q₁τ²] = 0)
    # → 4τ³/(90×stiff_norm) = 2λ(M/L)Q₁τ
    # → τ² = 45λQ₁×stiff_norm×M/(L)

    # For τ = 2M/√L: 4M²/L = 45λQ₁×stiff_norm×M/L
    # → stiff_norm = 4M/(45λQ₁)

    variants = {
        '1/(2M)': 2*M,           # original GPT-5
        '1/(2M²)': 2*M**2,      # per-channel-squared
        '1/2': 2,                # no M normalization
        '1/(2K²)': 2*(M)**2,    # same as M²
    }

    for name, stiff_norm in variants.items():
        # For this stiff_norm with λ=1, what τ do we get?
        taus_v = []
        for L in Ls:
            obj = lambda tau, _L=L, _sn=stiff_norm: obj_variant(tau, M, _L, b, 1.0, _sn)
            tau_opt = golden_min(obj, 0.01, 15.0, tol=1e-5)
            taus_v.append(tau_opt)

        slope_v = np.polyfit(np.log(Ls), np.log(np.maximum(taus_v, 0.001)), 1)[0]
        t2048 = taus_v[Ls.index(2048)]
        target = d_head / np.sqrt(2048)
        print(f"\n  stiff_norm={name}: slope=L^{slope_v:.4f}, τ@2048={t2048:.4f} (target {target:.4f}), ratio={t2048/target:.3f}")

    # ===== THE KEY: what if stiffness ∝ 1/(2M²) and λ from attention scaling? =====
    print("\n" + "=" * 80)
    print("MATCH BOTH SCALING AND ABSOLUTE VALUE")
    print("=" * 80)

    # From τ² = 45λQ₁ × stiff_norm × M/L, want τ = 2M/√L:
    # 4M²/L = 45λQ₁ × stiff_norm × M/L
    # stiff_norm × λ = 4M / (45Q₁) ≈ 4×32/(45×0.031) ≈ 91.8

    target_product = 4 * M / (45 * Q1_mean)
    print(f"  Need: stiff_norm × λ = {target_product:.2f}")

    # If stiff_norm = 2M² = 2048, λ = target_product/2048 ≈ 0.045
    # If stiff_norm = 2M = 64, λ = target_product/64 ≈ 1.43
    # If stiff_norm = 2, λ = target_product/2 ≈ 45.9

    configs = [
        ('1/(2M²), λ=free', 2*M**2, target_product/(2*M**2)),
        ('1/(2M), λ=free',  2*M,    target_product/(2*M)),
        ('1/2, λ=free',     2,      target_product/2),
    ]

    for name, sn, lam_v in configs:
        taus_c = []
        for L in Ls:
            obj = lambda tau, _L=L, _sn=sn, _lam=lam_v: obj_variant(tau, M, _L, b, _lam, _sn)
            tau_opt = golden_min(obj, 0.01, 15.0, tol=1e-5)
            taus_c.append(tau_opt)

        slope_c = np.polyfit(np.log(Ls), np.log(np.maximum(taus_c, 0.001)), 1)[0]
        t2048 = taus_c[Ls.index(2048)]
        target = d_head / np.sqrt(2048)

        print(f"\n  {name}, λ={lam_v:.4f}:")
        print(f"    slope=L^{slope_c:.4f}, τ@2048={t2048:.4f} (target {target:.4f}), ratio={t2048/target:.3f}")

        # Full table
        if abs(t2048/target - 1) < 0.5:  # close enough to show details
            print(f"    {'L':>6s} {'τ_opt':>10s} {'τ*':>10s} {'ratio':>8s}")
            for i, L in enumerate(Ls):
                te = d_head / np.sqrt(L)
                print(f"    {L:>6d} {taus_c[i]:>10.4f} {te:>10.4f} {taus_c[i]/te:>8.3f}")

    # ===== Check d_head scaling with best config =====
    print("\n" + "=" * 80)
    print("d_head SCALING CHECK (best config)")
    print("=" * 80)

    sn_best = 2*M  # 1/(2M)
    lam_best = target_product / sn_best
    L_test = 2048

    d_heads = [32, 64, 128, 256]
    d_taus = []
    print(f"  Config: stiff_norm=2M, λ={lam_best:.4f}")
    print(f"  {'d':>6s} {'M':>4s} {'τ_opt':>10s} {'d/√L':>10s} {'ratio':>8s}")
    print("  " + "-" * 42)

    for d in d_heads:
        Md = d // 2
        sn_d = 2 * Md  # stiff_norm scales with M for this variant
        lam_d = 4 * Md / (45 * Q1_vals.get(L_test, Q1_mean) * sn_d)  # recompute for consistency

        obj = lambda tau, _Md=Md, _sn=sn_d, _lam=lam_d: obj_variant(tau, _Md, L_test, b, _lam, _sn)
        t = golden_min(obj, 0.01, 15.0, tol=1e-5)
        te = d / np.sqrt(L_test)
        d_taus.append(t)
        print(f"  {d:>6d} {Md:>4d} {t:>10.4f} {te:>10.4f} {t/te:>8.3f}")

    if len(d_taus) >= 2:
        d_slope = np.polyfit(np.log(d_heads), np.log(np.maximum(d_taus, 0.001)), 1)[0]
        print(f"\n  τ ~ d_head^{d_slope:.3f} (expected: 1.0)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
