#!/usr/bin/env python3
"""CPU-only verification of the exact finite-grid TailSpline construction.

The checks establish the stated algebraic and discrete-optimization identities.
They do not load a checkpoint and are not evidence of benchmark advantage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .tables import analytic_exponents, atomic_json


FORMAT = "TAILSPLINE_CPU_VERIFICATION_V1"


def tailspline_increments(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    q = np.arange(1, n + 1, dtype=np.float64)
    values = 3.0 * (n + q) * (n - q + 1.0)
    return values / (n * (n + 1.0) * (2.0 * n + 1.0))


def one_sided_roughness(increments: np.ndarray) -> float:
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("increments must be one finite nonempty vector")
    return float(np.square(np.diff(values)).sum() + values[-1] ** 2)


def equality_constrained_minimizer(n: int) -> tuple[np.ndarray, float]:
    """Solve min ||D epsilon||^2 with sum(epsilon)=1 independently."""
    if n <= 0:
        raise ValueError("n must be positive")
    # D.T@D for rows (epsilon[q+1]-epsilon[q]) and the terminal epsilon[n].
    # Construct it directly to avoid relying on a platform BLAS for this tiny
    # exact tridiagonal system.
    hessian = np.diag(np.r_[1.0, np.full(n - 1, 2.0)])
    if n > 1:
        hessian += np.diag(np.full(n - 1, -1.0), 1)
        hessian += np.diag(np.full(n - 1, -1.0), -1)
    kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
    kkt[:n, :n] = hessian
    kkt[:n, n] = 1.0
    kkt[n, :n] = 1.0
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[n] = 1.0
    solution = np.linalg.solve(kkt, rhs)[:n]
    return solution, float(np.linalg.eigvalsh(hessian).min())


def verify_width(n: int) -> dict:
    pairs, low, high = n + 5, 2, n + 2
    tailspline = analytic_exponents(
        "tailspline", pairs, low=low, high=high,
    )
    increments = np.diff(tailspline[low : high + 1])
    closed_increments = tailspline_increments(n)
    numerical, minimum_eigenvalue = equality_constrained_minimizer(n)
    bm = analytic_exponents("bm", pairs, low=low, high=high)
    front = analytic_exponents(
        "mrpro_frontloaded", pairs, low=low, high=high,
    )
    mix075 = analytic_exponents("mix075", pairs, low=low, high=high)
    finite_weight = 3.0 * n / (2.0 * (2.0 * n + 1.0))
    exact_mix = (1.0 - finite_weight) * bm + finite_weight * front
    maximum_exponent_difference = float(np.max(np.abs(tailspline - mix075)))
    frequency_differences = {
        f"S{scale}": float(
            np.max(np.abs(np.power(scale, -(tailspline - mix075)) - 1.0))
        )
        for scale in (4, 8)
    }
    objective = one_sided_roughness(increments)
    expected_objective = 6.0 / (n * (n + 1.0) * (2.0 * n + 1.0))
    checks = {
        "closed_increment_identity": bool(
            np.allclose(increments, closed_increments, atol=2e-16, rtol=2e-15)
        ),
        "numerical_kkt_identity": bool(
            np.allclose(increments, numerical, atol=2e-15, rtol=0.0)
        ),
        "exact_bm_front_mixture_identity": bool(
            np.allclose(tailspline, exact_mix, atol=2e-16, rtol=2e-15)
        ),
        "unit_increment_mass": abs(float(increments.sum()) - 1.0) < 3e-16,
        "positive_strictly_decreasing_increments": bool(
            np.all(increments > 0.0) and np.all(np.diff(increments) < 0.0)
        ),
        "native_prefix_and_full_tail": bool(
            np.all(tailspline[: low + 1] == 0.0)
            and np.all(tailspline[high:] == 1.0)
        ),
        "strict_convexity": minimum_eigenvalue > 0.0,
        "minimum_objective_identity": math.isclose(
            objective, expected_objective, rel_tol=2e-15, abs_tol=2e-18,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"TailSpline n={n} checks failed: {failed}")
    return {
        "n": n,
        "checks": checks,
        "finite_grid_front_weight": finite_weight,
        "minimum_hessian_eigenvalue": minimum_eigenvalue,
        "objective": objective,
        "objective_closed_form": expected_objective,
        "maximum_exponent_difference_from_mix075": maximum_exponent_difference,
        "maximum_relative_frequency_difference_from_mix075": frequency_differences,
    }


def run_verification() -> dict:
    widths = [verify_width(n) for n in (17, 18)]
    return {
        "status": FORMAT,
        "widths": widths,
        "scope": "CPU algebra and finite-grid convex optimization only",
        "checkpoint_loaded": False,
        "real_model_evaluations": 0,
        "benchmark_advantage_demonstrated": False,
        "claim_boundary": (
            "The exact table uniquely minimizes its declared one-sided log-gap "
            "roughness objective; task superiority remains a GPU benchmark question."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_verification()
    atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "widths": [record["n"] for record in result["widths"]],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
