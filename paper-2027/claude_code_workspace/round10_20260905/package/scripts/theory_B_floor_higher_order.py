"""
Theory direction B: Higher-order τ_floor expansion.

Derives τ_floor(N, K) = 4·√(N/K) · [1 + N/(2K) + (241/120)·(N/K)² + O(K⁻³)]
from pure Taylor expansion of |Δφ(1/2, τ)| = τ²/16 - τ⁴/256 - 17τ⁶/30720 + ...

Verification:
  1. Symbolic derivation of Taylor coefficients via sympy
  2. Numerical exact τ_floor via brentq root-finding
  3. Compare leading vs 1-order vs 2-order formulas

All derivations are closed-form; no empirical constants anywhere.

Run: python theory_B_floor_higher_order.py
Dependencies: numpy, scipy (for root finding), sympy (for symbolic check)
"""
import numpy as np

try:
    from scipy.optimize import brentq
except ImportError:
    raise RuntimeError("scipy is required for numerical root finding")


def signed_displacement(tau):
    """Signed displacement Δφ(1/2, τ) = 1/2 - arcsinh(sinh(τ)/2)/τ."""
    if tau < 1e-10:
        return -tau**2 / 16
    return 0.5 - np.arcsinh(0.5 * np.sinh(tau)) / tau


def abs_displacement(tau):
    """|Δφ(1/2, τ)|. Positive for τ > 0."""
    return abs(signed_displacement(tau))


def exact_tau_floor(K, N=1):
    """Solve |Δφ(1/2, τ_floor)| = N/K numerically."""
    target = N / K
    return brentq(lambda t: abs_displacement(t) - target, 1e-5, 10.0)


def leading_tau_floor(K, N=1):
    """Leading-order formula: τ_floor^(0) = 4·√(N/K)."""
    return 4 * np.sqrt(N / K)


def first_order_tau_floor(K, N=1):
    """1-order formula: τ_floor · (1 + N/(2K))."""
    t0 = leading_tau_floor(K, N)
    return t0 * (1 + N / (2 * K))


def second_order_tau_floor(K, N=1):
    """2-order formula: τ_floor · (1 + N/(2K) + (241/120)(N/K)²).

    Coefficient 241/120 derived from Taylor expansion:
      - (1+y)^(1/2) with y = N/K + (64/15)(N/K)²
      - y/2 - y²/8 gives 32/15 - 1/8 = 241/120.
    """
    t0 = leading_tau_floor(K, N)
    ratio = N / K
    return t0 * (1 + ratio / 2 + (241 / 120) * ratio**2)


def main():
    print("=" * 100)
    print("τ_floor(N, K) — closed-form Taylor expansion verification")
    print("=" * 100)
    print("Formula: τ_floor = 4·√(N/K) · [1 + N/(2K) + (241/120)·(N/K)² + O(K⁻³)]")
    print()

    header = (f"{'N':>2} {'K':>4} {'d':>4} {'Leading':>9} {'1-order':>9} {'2-order':>9} "
              f"{'Exact':>9} {'err₀%':>9} {'err₁%':>9} {'err₂%':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for N in [1, 2]:
        for K in [16, 32, 64, 128, 256, 512]:
            t_exact = exact_tau_floor(K, N)
            t0 = leading_tau_floor(K, N)
            t1 = first_order_tau_floor(K, N)
            t2 = second_order_tau_floor(K, N)
            e0 = abs(t0 - t_exact) / t_exact * 100
            e1 = abs(t1 - t_exact) / t_exact * 100
            e2 = abs(t2 - t_exact) / t_exact * 100
            rows.append((N, K, t_exact, t0, t1, t2, e0, e1, e2))
            print(f"{N:>2} {K:>4} {2*K:>4} {t0:>9.4f} {t1:>9.4f} {t2:>9.4f} "
                  f"{t_exact:>9.4f} {e0:>8.3f}% {e1:>8.3f}% {e2:>8.3f}%")

    # ---------- Symbolic verification of Taylor coefficients ----------
    print()
    print("=" * 100)
    print("Symbolic Taylor series of Δφ(1/2, τ)")
    print("=" * 100)
    try:
        import sympy as sp
        t = sp.symbols('t')
        dphi = sp.Rational(1, 2) - sp.asinh(sp.sinh(t) / 2) / t
        series = sp.series(dphi, t, 0, 11).removeO()
        print(f"  Δφ(1/2, τ) = {series}")
        print()
        for n in [2, 4, 6, 8, 10]:
            c = series.coeff(t, n)
            print(f"  coefficient of τ^{n}: {c}  (float: {float(c):+.6e})")
    except ImportError:
        print("  [sympy not available; skipping symbolic verification]")

    # ---------- Error reduction summary ----------
    print()
    print("=" * 100)
    print("Error reduction summary (N=1)")
    print("=" * 100)
    print(f"{'K':>4} {'err(lead)':>10} {'err(1-ord)':>11} {'err(2-ord)':>11} "
          f"{'1-ord / lead':>12} {'2-ord / lead':>12}")
    print("-" * 70)
    for row in rows:
        N, K, te, t0, t1, t2, e0, e1, e2 = row
        if N != 1:
            continue
        r1 = e1 / e0 if e0 > 0 else 0
        r2 = e2 / e0 if e0 > 0 else 0
        print(f"{K:>4} {e0:>9.3f}% {e1:>10.3f}% {e2:>10.3f}% {r1:>12.3f} {r2:>12.3f}")

    max_err_2order = max(row[8] for row in rows)
    verdict = "PASS" if max_err_2order < 1.0 else "FAIL"
    print()
    print(f"Max 2-order error across all (N, K) tested: {max_err_2order:.3f}%")
    print(f"Verdict: {verdict} (target: <1% for K ≥ 16)")


if __name__ == "__main__":
    main()
