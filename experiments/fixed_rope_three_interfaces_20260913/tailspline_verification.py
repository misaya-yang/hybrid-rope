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


FORMAT = "TAILSPLINE_CPU_VERIFICATION_V2"


def tailspline_increments(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    q = np.arange(1, n + 1, dtype=np.float64)
    values = 3.0 * (n + q) * (n - q + 1.0)
    return values / (n * (n + 1.0) * (2.0 * n + 1.0))


def boundary_roughness(
    increments: np.ndarray, *, entry_penalty: float, tail_penalty: float = 1.0,
) -> float:
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("increments must be one finite nonempty vector")
    if (
        not math.isfinite(entry_penalty)
        or not math.isfinite(tail_penalty)
        or entry_penalty < 0.0
        or tail_penalty < 0.0
        or entry_penalty + tail_penalty <= 0.0
    ):
        raise ValueError("boundary penalties must be finite, nonnegative, and nonzero")
    return float(
        entry_penalty * values[0] ** 2
        + np.square(np.diff(values)).sum()
        + tail_penalty * values[-1] ** 2
    )


def one_sided_roughness(increments: np.ndarray) -> float:
    return boundary_roughness(increments, entry_penalty=0.0)


def symmetric_roughness(increments: np.ndarray) -> float:
    return boundary_roughness(increments, entry_penalty=1.0)


def equality_constrained_minimizer(
    n: int, *, entry_penalty: float = 0.0, tail_penalty: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Solve min ||D epsilon||^2 with sum(epsilon)=1 independently."""
    if n <= 0:
        raise ValueError("n must be positive")
    if (
        not math.isfinite(entry_penalty)
        or not math.isfinite(tail_penalty)
        or entry_penalty < 0.0
        or tail_penalty < 0.0
        or entry_penalty + tail_penalty <= 0.0
    ):
        raise ValueError("boundary penalties must be finite, nonnegative, and nonzero")
    # D.T@D for adjacent differences plus the two declared boundary penalties.
    # Construct it directly to keep this tiny exact tridiagonal system explicit.
    hessian = np.zeros((n, n), dtype=np.float64)
    if n > 1:
        difference = np.eye(n - 1, n, k=1) - np.eye(n - 1, n)
        hessian += difference.T @ difference
    hessian[0, 0] += entry_penalty
    hessian[-1, -1] += tail_penalty
    kkt = np.zeros((n + 1, n + 1), dtype=np.float64)
    kkt[:n, :n] = hessian
    kkt[:n, n] = 1.0
    kkt[n, :n] = 1.0
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[n] = 1.0
    solution = np.linalg.solve(kkt, rhs)[:n]
    return solution, float(np.linalg.eigvalsh(hessian).min())


def boundary_family_closed_form(n: int, entry_penalty: float) -> np.ndarray:
    """Solve the boundary-weight continuum without creating GPU candidates."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not math.isfinite(entry_penalty) or entry_penalty < 0.0:
        raise ValueError("entry_penalty must be finite and nonnegative")
    q = np.arange(1, n + 1, dtype=np.float64)
    coefficient = (
        1.0 + entry_penalty * n * (n + 2.0)
    ) / (2.0 * (1.0 + entry_penalty * n))
    offset = (n + 1.0) ** 2 / 2.0 - coefficient * (n + 1.0)
    unnormalized = -q * q / 2.0 + coefficient * q + offset
    return unnormalized / unnormalized.sum()


def bm_increments(n: int) -> np.ndarray:
    q = np.arange(1, n + 1, dtype=np.float64)
    return 6.0 * q * (n - q + 1.0) / (n * (n + 1.0) * (n + 2.0))


def profile_geometry(increments: np.ndarray) -> dict:
    values = np.asarray(increments, dtype=np.float64)
    locations = np.arange(1, len(values) + 1, dtype=np.float64)
    return {
        "entry_jump": float(values[0]),
        "tail_jump": float(values[-1]),
        "increment_centroid": float(np.sum(locations * values) / values.sum()),
        "one_sided_roughness": one_sided_roughness(values),
        "symmetric_roughness": symmetric_roughness(values),
    }


def verify_width(n: int) -> dict:
    pairs, low, high = n + 5, 2, n + 2
    tailspline = analytic_exponents(
        "tailspline", pairs, low=low, high=high,
    )
    increments = np.diff(tailspline[low : high + 1])
    closed_increments = tailspline_increments(n)
    numerical, minimum_eigenvalue = equality_constrained_minimizer(n)
    symmetric_numerical, symmetric_minimum_eigenvalue = equality_constrained_minimizer(
        n, entry_penalty=1.0,
    )
    audit_penalty = math.sqrt(2.0)
    audit_numerical, _ = equality_constrained_minimizer(
        n, entry_penalty=audit_penalty,
    )
    one_sided_closed = boundary_family_closed_form(n, 0.0)
    symmetric_closed = boundary_family_closed_form(n, 1.0)
    audit_closed = boundary_family_closed_form(n, audit_penalty)
    bm = analytic_exponents("bm", pairs, low=low, high=high)
    front = analytic_exponents(
        "mrpro_frontloaded", pairs, low=low, high=high,
    )
    mix075 = analytic_exponents("mix075", pairs, low=low, high=high)
    finite_weight = 3.0 * n / (2.0 * (2.0 * n + 1.0))
    exact_mix = (1.0 - finite_weight) * bm + finite_weight * front
    bm_epsilon = np.diff(bm[low : high + 1])
    mrpro = analytic_exponents("mrpro", pairs, low=low, high=high)
    mrpro_epsilon = np.diff(mrpro[low : high + 1])
    maximum_exponent_difference = float(np.max(np.abs(tailspline - mix075)))
    frequency_differences = {
        f"S{scale}": float(
            np.max(np.abs(np.power(scale, -(tailspline - mix075)) - 1.0))
        )
        for scale in (4, 8)
    }
    objective = one_sided_roughness(increments)
    expected_objective = 6.0 / (n * (n + 1.0) * (2.0 * n + 1.0))
    symmetric_objective = symmetric_roughness(bm_epsilon)
    expected_symmetric_objective = 12.0 / (n * (n + 1.0) * (n + 2.0))
    checks = {
        "closed_increment_identity": bool(
            np.allclose(increments, closed_increments, atol=2e-16, rtol=2e-15)
        ),
        "numerical_kkt_identity": bool(
            np.allclose(increments, numerical, atol=2e-15, rtol=0.0)
        ),
        "one_sided_closed_family_endpoint_identity": bool(
            np.allclose(increments, one_sided_closed, atol=2e-15, rtol=0.0)
        ),
        "symmetric_numerical_kkt_identity": bool(
            np.allclose(bm_epsilon, symmetric_numerical, atol=2e-15, rtol=0.0)
        ),
        "symmetric_closed_family_endpoint_identity": bool(
            np.allclose(bm_epsilon, symmetric_closed, atol=2e-15, rtol=0.0)
        ),
        "interior_boundary_family_kkt_identity": bool(
            np.allclose(audit_numerical, audit_closed, atol=2e-15, rtol=0.0)
        ),
        "symmetric_bm_increment_identity": bool(
            np.allclose(bm_epsilon, bm_increments(n), atol=2e-16, rtol=2e-15)
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
        "symmetric_strict_convexity": symmetric_minimum_eigenvalue > 0.0,
        "minimum_objective_identity": math.isclose(
            objective, expected_objective, rel_tol=2e-15, abs_tol=2e-18,
        ),
        "symmetric_minimum_objective_identity": math.isclose(
            symmetric_objective, expected_symmetric_objective,
            rel_tol=2e-15, abs_tol=2e-18,
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
        "symmetric_minimum_hessian_eigenvalue": symmetric_minimum_eigenvalue,
        "objective": objective,
        "objective_closed_form": expected_objective,
        "symmetric_objective": symmetric_objective,
        "symmetric_objective_closed_form": expected_symmetric_objective,
        "transport_geometry": {
            "tailspline": profile_geometry(increments),
            "bm": profile_geometry(bm_epsilon),
            "mrpro": profile_geometry(mrpro_epsilon),
        },
        "maximum_exponent_difference_from_bm": float(
            np.max(np.abs(tailspline - bm))
        ),
        "maximum_relative_frequency_difference_from_bm": {
            f"S{scale}": float(
                np.max(np.abs(np.power(scale, -(tailspline - bm)) - 1.0))
            )
            for scale in (4, 8)
        },
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
        "boundary_condition_audit": {
            "one_sided_entry_penalty": 0.0,
            "one_sided_tail_penalty": 1.0,
            "one_sided_unique_minimizer": "tailspline",
            "symmetric_entry_penalty": 1.0,
            "symmetric_tail_penalty": 1.0,
            "symmetric_unique_minimizer": "bm",
            "nonnegative_entry_penalty_has_unique_closed_form_continuum": True,
            "intermediate_penalties_are_gpu_candidates": False,
            "cpu_selects_boundary_condition": False,
        },
        "claim_boundary": (
            "TailSpline and BM uniquely minimize different declared boundary-roughness "
            "objectives. CPU algebra does not select the boundary condition or prove "
            "task superiority; that remains a real-model benchmark question."
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
