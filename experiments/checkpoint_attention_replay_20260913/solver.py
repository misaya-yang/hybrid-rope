"""Boundary-capable local quadratic solver for transition increments."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def project_simplex(values: Iterable[float], total: float) -> np.ndarray:
    """Euclidean projection onto ``x >= 0, sum(x) = total`` with exact zeros."""
    vector = np.asarray(list(values), dtype=np.float64)
    if (vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all()
            or not math.isfinite(total) or total < 0):
        raise ValueError("simplex projection inputs are invalid")
    if total == 0:
        return np.zeros_like(vector)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - total
    support = np.flatnonzero(ordered - cumulative / np.arange(1, len(vector) + 1) > 0)
    if not len(support):
        raise AssertionError("simplex projection has no active coordinate")
    index = int(support[-1])
    threshold = cumulative[index] / (index + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected *= total / projected.sum()
    return projected


def _kkt_residual(solution: np.ndarray, gradient: np.ndarray, zero_tolerance: float) -> float:
    active = solution > zero_tolerance
    if not active.any():
        return float(np.max(np.abs(gradient)))
    level = float(gradient[active].mean())
    active_error = float(np.max(np.abs(gradient[active] - level)))
    inactive_error = (
        float(np.max(np.maximum(level - gradient[~active], 0.0)))
        if (~active).any() else 0.0
    )
    return max(active_error, inactive_error)


def solve_increment_qp(
    initial_increments: Iterable[float],
    gradient_at_initial: Iterable[float],
    hessian: np.ndarray,
    *,
    tail_depth: float | None = None,
    max_iterations: int = 100_000,
    tolerance: float = 1e-11,
    zero_tolerance: float = 1e-10,
) -> dict:
    """Solve a PSD local QP on the closed simplex by projected gradient.

    The supplied model is in step coordinates,
    ``g.T @ (epsilon-epsilon0) + .5*(epsilon-epsilon0).T Q (...)``.
    Returning absolute increments avoids the common coordinate ambiguity.
    """
    initial = np.asarray(list(initial_increments), dtype=np.float64)
    gradient = np.asarray(list(gradient_at_initial), dtype=np.float64)
    matrix = np.asarray(hessian, dtype=np.float64)
    depth = float(initial.sum() if tail_depth is None else tail_depth)
    if (initial.ndim != 1 or not len(initial) or gradient.shape != initial.shape
            or matrix.shape != (len(initial), len(initial))):
        raise ValueError("QP arrays have inconsistent shapes")
    if (not np.isfinite(initial).all() or not np.isfinite(gradient).all()
            or not np.isfinite(matrix).all() or np.any(initial < 0)
            or not 0 <= depth <= 1 or max_iterations < 1
            or tolerance <= 0 or zero_tolerance < 0):
        raise ValueError("QP inputs are invalid")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -1e-10:
        raise ValueError("local QP Hessian must be positive semidefinite")
    current = project_simplex(initial, depth)
    linear = gradient - matrix @ initial
    lipschitz = float(max(eigenvalues[-1], 0.0))
    if lipschitz <= 1e-15:
        minimum = float(linear.min())
        indices = np.flatnonzero(np.isclose(linear, minimum, rtol=0, atol=tolerance))
        current = np.zeros_like(initial)
        current[indices] = depth / len(indices) if len(indices) else 0.0
        iterations = 1
    else:
        for iterations in range(1, max_iterations + 1):
            local_gradient = matrix @ current + linear
            proposed = project_simplex(current - local_gradient / lipschitz, depth)
            if np.linalg.norm(proposed - current, ord=np.inf) <= tolerance:
                current = proposed
                break
            current = proposed
    step = current - initial
    final_gradient = gradient + matrix @ step
    objective = float(gradient @ step + 0.5 * step @ matrix @ step)
    residual = _kkt_residual(current, final_gradient, zero_tolerance)
    return {
        "status": "INCREMENT_QP_COMPLETE" if residual <= max(1e-7, 10 * tolerance)
                  else "INCREMENT_QP_NUMERICAL_UNRESOLVED",
        "initial_increments": initial.tolist(),
        "increments": current.tolist(),
        "step": step.tolist(),
        "exponents": np.concatenate(([0.0], np.cumsum(current))).tolist(),
        "tail_depth": depth,
        "active_increment_indices": np.flatnonzero(current > zero_tolerance).tolist(),
        "zero_increment_indices": np.flatnonzero(current <= zero_tolerance).tolist(),
        "objective_change": objective,
        "kkt_residual": residual,
        "iterations": iterations,
        "zero_tolerance": zero_tolerance,
        "coordinate_contract": "step QP around initial_increments; output is absolute epsilon",
    }
