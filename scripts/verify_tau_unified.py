"""
Verification script for the unified τ formula:
    τ*(d_head, L) = max(d_head / √L, τ_bal)
where τ_bal = 1.4266890671... is the exact self-balance point of the EVQ-cosh density,
defined by sinh(τ)/τ - 1 = 1 - tanh(τ)/τ.

Tests:
  1. τ_bal is closed-form (Newton-Raphson convergence)
  2. Unified formula against 15 experimental anchors (MHA, MLA, Phase16 sweep)
  3. Structural coincidence: L_crit = (d/τ_bal)² ≈ d²/2.035

Run: python verify_tau_unified.py (numpy only, < 1 second)
"""
import numpy as np


def compute_tau_bal(tol=1e-12, max_iter=100):
    """Exact self-balance point: sinh(τ)/τ + tanh(τ)/τ = 2."""
    def F(t):
        return np.sinh(t) / t + np.tanh(t) / t - 2

    def Fp(t):
        return (np.cosh(t) / t - np.sinh(t) / t**2) + (
            1 / (np.cosh(t) ** 2 * t) - np.tanh(t) / t**2
        )

    t = 1.4
    for _ in range(max_iter):
        f = F(t)
        if abs(f) < tol:
            break
        t = t - f / Fp(t)
    return t


def tau_unified(d_head, L, tau_bal):
    """Unified formula: τ* = max(d_head/√L, τ_bal)."""
    return max(d_head / np.sqrt(L), tau_bal)


def main():
    tau_bal = compute_tau_bal()
    print(f"τ_bal = {tau_bal:.10f}")
    print(f"√2    = {np.sqrt(2):.10f}")
    print(f"Δ = {tau_bal - np.sqrt(2):.4f} (τ_bal is close to but not equal to √2)")
    print()

    # Anchors: (name, d_head, L_train, τ_observed)
    anchors = [
        ("MHA 50M L=2048",       64, 2048, 1.50),
        ("MHA 125M L=2048",      64, 2048, 1.50),
        ("Phase11 454M L=256",   64, 256,  4.00),
        ("Phase15 750M L=4096",  64, 4096, 1.50),
        ("Phase17 454M L=512",   64, 512,  2.80),
        ("P16 d32 L=256",        32, 256,  2.00),
        ("P16 d32 L=512",        32, 512,  1.77),
        ("P16 d32 L=1024",       32, 1024, 1.25),
        ("P16 d64 L=256",        64, 256,  4.00),
        ("P16 d64 L=512",        64, 512,  4.24),
        ("P16 d64 L=1024",       64, 1024, 2.50),
        ("P16 d128 L=256",      128, 256, 10.00),
        ("P16 d128 L=512",      128, 512,  5.66),
        ("P16 d128 L=1024",     128, 1024, 5.00),
        ("MLA (d_head=128) L=8192", 128, 8192, 1.414),
    ]

    print(f"{'Anchor':<28} {'d':>4} {'L':>6} {'d/√L':>7} {'τ*':>7} {'obs':>7} {'err%':>6}")
    print("-" * 80)
    errs = []
    for name, d, L, obs in anchors:
        r = d / np.sqrt(L)
        pred = tau_unified(d, L, tau_bal)
        err = abs(pred - obs) / obs * 100
        errs.append(err)
        print(f"{name:<28} {d:>4} {L:>6} {r:>7.3f} {pred:>7.3f} {obs:>7.3f} {err:>6.1f}")

    errs = np.array(errs)
    print("-" * 80)
    print(
        f"mean err = {errs.mean():.1f}%, max = {errs.max():.1f}%, "
        f"≤15% rate = {sum(errs < 15)}/{len(errs)}"
    )
    print()

    # Compare against pure d/√L
    errs_bare = np.array([
        abs(d / np.sqrt(L) - obs) / obs * 100 for _, d, L, obs in anchors
    ])
    print(f"Comparison: pure d/√L has mean err = {errs_bare.mean():.1f}%, "
          f"≤15% = {sum(errs_bare < 15)}/{len(anchors)}")
    print()

    # Crossover structural check
    print("Structural crossover L_crit = (d/τ_bal)² ≈ d² / 2.035:")
    for d in [32, 64, 96, 128, 192, 256]:
        L_crit = (d / tau_bal) ** 2
        print(f"  d={d:>3}: L_crit = {L_crit:7.0f}  (= d² / {d**2 / L_crit:.3f})")

    verdict = "PASS" if errs.mean() < 15 else "FAIL"
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
