"""Auditor 4: independent recompute of Q_1(L,b) and c_pred = sqrt(45 Q_1) for
the 8 (L,b) pairs disclosed in PAPER_HANDOVER_2026-04-27.md.

Q_1(L,b) = integral_0^1 a(phi) * q(L * b^{-phi}) d phi
a(phi)   = (1-phi)^2 / 2 - 1/6
q(x)     = 1/2 + sin(2x)/(4x) - (sin(x)/x)^2
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def q_scalar(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    return 0.5 + np.sin(2.0 * x) / (4.0 * x) - (np.sin(x) / x) ** 2


def a_scalar(phi: float) -> float:
    return (1.0 - phi) ** 2 / 2.0 - 1.0 / 6.0


def Q1(L: float, b: float) -> float:
    def integrand(phi: float) -> float:
        x = L * b ** (-phi)
        return a_scalar(phi) * q_scalar(x)
    val, _ = quad(integrand, 0.0, 1.0, epsrel=1e-10, limit=400)
    return val


def main() -> None:
    pairs = [
        (128, 10_000),
        (1024, 10_000),
        (4096, 10_000),
        (8192, 10_000),
        (128, 500_000),
        (2048, 500_000),
        (4096, 500_000),
        (8192, 500_000),
    ]
    # Handover doc disclosed values (Q_1, c_pred, deviation pct from c=1)
    handover = {
        (128, 10_000): (0.0317, 1.194, -16.3),
        (1024, 10_000): (0.0241, 1.042, -4.0),
        (4096, 10_000): (0.0141, 0.797, +25.4),
        (8192, 10_000): (0.0083, 0.611, +63.6),
        (128, 500_000): (0.0301, 1.164, -14.1),
        (2048, 500_000): (0.0305, 1.172, -14.7),
        (4096, 500_000): (0.0288, 1.138, -12.1),
        (8192, 500_000): (0.0265, 1.091, -8.4),
    }

    print(
        f"{'L':>6} {'b':>8} {'Q_1':>10} {'c_pred':>9} {'dev%':>8} | "
        f"{'Q_1_doc':>9} {'c_doc':>8} {'dev_doc%':>9} | {'verdict':>10}"
    )
    print("-" * 95)

    for L, b in pairs:
        q1 = Q1(L, b)
        c = np.sqrt(45.0 * q1)
        dev = 100.0 * (1.0 / c - 1.0)  # deployed c=1 vs predicted c
        q1_d, c_d, dev_d = handover[(L, b)]
        # tolerance: relative diff in c_pred < 1%
        err_q1 = abs(q1 - q1_d) / max(q1_d, 1e-12)
        err_c = abs(c - c_d) / max(c_d, 1e-12)
        verdict = "OK" if err_c < 0.01 else "DRIFT"
        print(
            f"{L:>6d} {b:>8d} {q1:>10.6f} {c:>9.4f} {dev:>8.2f} | "
            f"{q1_d:>9.4f} {c_d:>8.4f} {dev_d:>9.2f} | {verdict:>10}"
        )


if __name__ == "__main__":
    main()
