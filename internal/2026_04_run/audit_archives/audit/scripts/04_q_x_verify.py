"""Auditor 4: independent verification of q(x) = Var_{t~U[0,1]}[cos(xt)].

Compares numerical integration against the closed form
    q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)^2.

Reports max abs error across x in {0.1, 1, 5, 25, 100, 1000, 10000}.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def q_closed_form(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    return 0.5 + np.sin(2.0 * x) / (4.0 * x) - (np.sin(x) / x) ** 2


def q_numerical(x: float) -> float:
    """Var[cos(x t)] = E[cos^2(xt)] - (E[cos(xt)])^2, t~U[0,1]."""
    # E[cos(xt)] = (1/x) sin(x) when x != 0, else 1
    if abs(x) < 1e-12:
        return 0.0
    # Use quad with high tolerance.  Heavy oscillation for large x is handled
    # by chopping the integration domain into k*(pi/x) chunks.
    period = np.pi / abs(x) if abs(x) > 0 else 1.0
    n_chunks = max(1, int(np.ceil(1.0 / period)))
    n_chunks = min(n_chunks, 5000)  # cap for very large x

    def f_cos2(t: float) -> float:
        return np.cos(x * t) ** 2

    def f_cos(t: float) -> float:
        return np.cos(x * t)

    if n_chunks == 1:
        e_cos2, _ = quad(f_cos2, 0.0, 1.0, epsrel=1e-12, limit=400)
        e_cos, _ = quad(f_cos, 0.0, 1.0, epsrel=1e-12, limit=400)
    else:
        edges = np.linspace(0.0, 1.0, n_chunks + 1)
        e_cos2 = 0.0
        e_cos = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            v2, _ = quad(f_cos2, a, b, epsrel=1e-12, limit=400)
            v1, _ = quad(f_cos, a, b, epsrel=1e-12, limit=400)
            e_cos2 += v2
            e_cos += v1

    return e_cos2 - e_cos ** 2


def main() -> None:
    xs = [0.1, 1.0, 5.0, 25.0, 100.0, 1000.0, 10000.0]
    print(f"{'x':>10} {'q_closed':>14} {'q_numerical':>16} {'abs_err':>14} {'rel_err':>12}")
    print("-" * 70)
    max_abs = 0.0
    max_rel = 0.0
    for x in xs:
        cf = q_closed_form(x)
        nu = q_numerical(x)
        ae = abs(cf - nu)
        re = ae / max(abs(cf), 1e-12)
        if ae > max_abs:
            max_abs = ae
        if re > max_rel:
            max_rel = re
        print(f"{x:>10.4g} {cf:>14.10f} {nu:>16.10f} {ae:>14.4e} {re:>12.4e}")
    print("-" * 70)
    print(f"max abs error = {max_abs:.4e}")
    print(f"max rel error = {max_rel:.4e}")


if __name__ == "__main__":
    main()
