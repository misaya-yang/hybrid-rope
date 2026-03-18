#!/usr/bin/env python3
"""
Attempt to derive τ* = d_head/√L from the ODE coefficients.

The ODE is: ρ''(φ) - τ² ρ(φ) = γ b^{-2φ}, where τ = √(β/α).

We compute α and β from the exact kernel numerically, then check
if τ = √(β/α) matches d_head/√L.
"""

import numpy as np

def exact_kernel(phi1, phi2, L, b):
    """Compute the exact collision kernel for uniform distance prior on [0,L]."""
    omega1 = b ** (-phi1)
    omega2 = b ** (-phi2)

    # K = (1/L) ∫₀ᴸ cos(ω₁Δ)cos(ω₂Δ) dΔ
    # = (1/2L) [sin((ω₁-ω₂)L)/(ω₁-ω₂) + sin((ω₁+ω₂)L)/(ω₁+ω₂)]
    sum_omega = omega1 + omega2
    diff_omega = omega1 - omega2

    if abs(diff_omega) < 1e-15:
        term1 = L  # lim sin(xL)/x as x→0 = L
    else:
        term1 = np.sin(diff_omega * L) / diff_omega

    term2 = np.sin(sum_omega * L) / sum_omega

    return (term1 + term2) / (2 * L)


def compute_kernel_matrix(K_channels, L, b):
    """Compute the K×K exact kernel matrix for uniformly spaced channels."""
    phi = np.linspace(0, 1, K_channels, endpoint=False) + 0.5/K_channels
    K_mat = np.zeros((K_channels, K_channels))
    for i in range(K_channels):
        for j in range(K_channels):
            K_mat[i, j] = exact_kernel(phi[i], phi[j], L, b)
    return K_mat, phi


def fit_broadband_surrogate(K_mat, phi):
    """
    Fit K_app(φ_i, φ_j) = α δ_{ij}/Δφ + β min(φ_i, φ_j) to the exact kernel.
    For the discrete case: K_app_ij = α/Δφ × I(i=j) + β × min(φ_i, φ_j)
    """
    K_ch = len(phi)
    dphi = 1.0 / K_ch  # uniform spacing

    # Separate diagonal and off-diagonal
    diag = np.diag(K_mat)
    alpha_est = np.mean(diag) * dphi  # α ≈ mean(K_ii) × Δφ

    # For off-diagonal: K_ij ≈ β × min(φ_i, φ_j)
    offdiag_vals = []
    min_vals = []
    for i in range(K_ch):
        for j in range(K_ch):
            if i != j:
                offdiag_vals.append(K_mat[i, j])
                min_vals.append(min(phi[i], phi[j]))

    offdiag_vals = np.array(offdiag_vals)
    min_vals = np.array(min_vals)

    # Least squares fit: β = (offdiag · min) / (min · min)
    beta_est = np.dot(offdiag_vals, min_vals) / np.dot(min_vals, min_vals)

    return alpha_est, beta_est


def main():
    print("="*70)
    print("NUMERICAL τ = √(β/α) vs EMPIRICAL τ* = d_head/√L")
    print("="*70)

    configs = [
        # (d_head, L, b, description)
        (64, 128, 500_000, "d=64, L=128, b=500K"),
        (64, 256, 500_000, "d=64, L=256, b=500K"),
        (64, 512, 500_000, "d=64, L=512, b=500K"),
        (64, 1024, 500_000, "d=64, L=1024, b=500K"),
        (64, 2048, 500_000, "d=64, L=2048, b=500K"),
        (64, 4096, 500_000, "d=64, L=4096, b=500K"),
        (64, 2048, 10_000, "d=64, L=2048, b=10K"),
        (64, 2048, 100_000, "d=64, L=2048, b=100K"),
        (128, 4096, 10_000, "d=128, L=4096, b=10K (Llama-2)"),
        (32, 32, 10_000, "d=32, L=32, b=10K (video DiT)"),
    ]

    print(f"\n{'Config':<35s} {'α':>8s} {'β':>10s} {'τ_theory':>10s} {'τ_empiric':>10s} {'ratio':>8s}")
    print("-" * 85)

    for d_head, L, b, desc in configs:
        K = d_head // 2
        K_mat, phi = compute_kernel_matrix(K, L, b)
        alpha, beta = fit_broadband_surrogate(K_mat, phi)

        tau_theory = np.sqrt(abs(beta / alpha)) if alpha > 0 else 0
        tau_empirical = d_head / np.sqrt(L)
        ratio = tau_theory / tau_empirical if tau_empirical > 0 else 0

        print(f"{desc:<35s} {alpha:>8.4f} {beta:>10.6f} {tau_theory:>10.3f} {tau_empirical:>10.3f} {ratio:>8.3f}")

    # Additional analysis: how do α and β scale with L?
    print("\n" + "="*70)
    print("SCALING ANALYSIS: α and β vs L (d_head=64, b=500K)")
    print("="*70)
    print(f"\n{'L':>6s} {'α':>10s} {'β':>12s} {'β/α':>10s} {'√(β/α)':>10s} {'d/√L':>10s}")
    print("-" * 65)

    d_head = 64
    b = 500_000
    K = d_head // 2
    alphas, betas, Ls = [], [], []

    for L in [128, 256, 512, 1024, 2048, 4096]:
        K_mat, phi = compute_kernel_matrix(K, L, b)
        alpha, beta = fit_broadband_surrogate(K_mat, phi)
        tau_th = np.sqrt(abs(beta/alpha))
        tau_emp = d_head / np.sqrt(L)
        print(f"{L:>6d} {alpha:>10.4f} {beta:>12.6f} {abs(beta/alpha):>10.4f} {tau_th:>10.3f} {tau_emp:>10.3f}")
        alphas.append(alpha)
        betas.append(beta)
        Ls.append(L)

    alphas = np.array(alphas)
    betas = np.array(betas)
    Ls = np.array(Ls)

    # Fit power laws: α ~ L^a, β ~ L^b
    from numpy.polynomial import polynomial as P
    log_L = np.log(Ls)
    log_alpha = np.log(np.abs(alphas))
    log_beta = np.log(np.abs(betas))

    # Linear regression in log-log space
    a_slope = np.polyfit(log_L, log_alpha, 1)[0]
    b_slope = np.polyfit(log_L, log_beta, 1)[0]

    print(f"\nPower law fits:")
    print(f"  α ~ L^{a_slope:.3f}")
    print(f"  β ~ L^{b_slope:.3f}")
    print(f"  → τ = √(β/α) ~ L^{(b_slope - a_slope)/2:.3f}")
    print(f"  Expected for τ=d/√L: L^{-0.5}")

    # Scaling with d_head
    print("\n" + "="*70)
    print("SCALING ANALYSIS: α and β vs d_head (L=2048, b=500K)")
    print("="*70)
    print(f"\n{'d_head':>6s} {'K':>4s} {'α':>10s} {'β':>12s} {'√(β/α)':>10s} {'d/√L':>10s}")
    print("-" * 55)

    L = 2048
    b = 500_000
    for d_head in [32, 64, 128, 256]:
        K = d_head // 2
        K_mat, phi = compute_kernel_matrix(K, L, b)
        alpha, beta = fit_broadband_surrogate(K_mat, phi)
        tau_th = np.sqrt(abs(beta/alpha))
        tau_emp = d_head / np.sqrt(L)
        print(f"{d_head:>6d} {K:>4d} {alpha:>10.4f} {beta:>12.6f} {tau_th:>10.3f} {tau_emp:>10.3f}")


if __name__ == "__main__":
    main()
