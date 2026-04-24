"""Numerical verification of c_coll = 1.171 via the Q_1(L,b) integral.

Paper reference: Appendix sec:lambda-curvature. Under the lambda=1 unit
convention, the operating-point prefactor predicted by the softmax-transport
balance equation is c_pred = sqrt(45 * Q_1(L, b)). This script compares
c_pred against the observed c_coll (from tables/table_lambda_cv.tex); the
ratio, if near-constant, is the square root of the surrogate-vs-exact
curvature multiplier lambda_infty.

Local runtime target: < 5s on M4 Max.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def q_scalar(x: float) -> float:
    """Softmax-transport kernel integrand; q(0)=0 handled via guard."""
    if abs(x) < 1e-8:
        return 0.0
    return 0.5 + np.sin(2.0 * x) / (4.0 * x) - (np.sin(x) / x) ** 2


def a_scalar(phi: float) -> float:
    return (1.0 - phi) ** 2 / 2.0 - 1.0 / 6.0


def Q1(L: float, b: float) -> float:
    """Q_1(L, b) = integral_0^1 a(phi) * q(L * b^(-phi)) d phi."""
    def integrand(phi: float) -> float:
        x = L * b ** (-phi)
        return a_scalar(phi) * q_scalar(x)
    result, _ = quad(integrand, 0.0, 1.0, epsrel=1e-10, limit=200)
    return result


def main() -> None:
    b = 500_000.0
    configs = [
        # (d_head, L, tau*_coll from table_lambda_cv.tex)
        (32, 256, 2.340),
        (32, 512, 1.660),
        (32, 1024, 1.170),
        (64, 256, 4.680),
        (64, 512, 3.310),
        (64, 1024, 2.340),
        (128, 256, 9.370),
        (128, 512, 6.630),
        (128, 1024, 4.680),
    ]

    print(f"b = {b:g}")
    print(f"{'d_head':>7} {'L':>5} {'Q_1':>12} {'c_pred':>10} {'c_obs':>10} {'ratio':>10}")
    print("-" * 60)

    ratios: list[float] = []
    for d_head, L, tau_coll in configs:
        q1_val = Q1(L, b)
        c_pred = float(np.sqrt(45.0 * q1_val))
        c_obs = tau_coll / (d_head / np.sqrt(L))
        ratio = c_obs / c_pred
        ratios.append(ratio)
        print(f"{d_head:>7} {L:>5} {q1_val:>12.6f} {c_pred:>10.4f} {c_obs:>10.4f} {ratio:>10.4f}")

    arr = np.array(ratios)
    mean_r = float(arr.mean())
    std_r = float(arr.std(ddof=1))
    cv_pct = 100.0 * std_r / mean_r
    lam_inf = mean_r ** 2

    print("-" * 60)
    print(
        f"Mean c_obs / c_pred = {mean_r:.4f} +/- {std_r:.4f} (CV {cv_pct:.2f}%). "
        f"If near-constant: lambda_infty = ({mean_r:.4f})^2 = {lam_inf:.4f}"
    )


if __name__ == "__main__":
    main()
