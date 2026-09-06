"""Deterministic same-support RoPE controls for mature-checkpoint audits.

These tables are CPU constructions.  They never inspect task labels or model
outputs.  All controls preserve the Native fast endpoint, move the Native slow
endpoint by the same factor, and use the same number of rotary pairs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scripts.analysis.export_uniqueness_budgeted_tables import (
    causal_distance_measure,
    conditional_pair_uniqueness,
    float32_sha256,
)


def _source(native: np.ndarray) -> np.ndarray:
    values = np.asarray(native, dtype=np.float64).reshape(-1)
    if (
        values.size < 2
        or not np.isfinite(values).all()
        or not (values > 0.0).all()
        or not np.all(values[:-1] > values[1:])
    ):
        raise ValueError("Native inverse frequencies must be finite, positive, and decreasing")
    return values


def _same_support_table(
    native: np.ndarray,
    movement: np.ndarray,
    *,
    factor: float,
) -> np.ndarray:
    source = _source(native)
    move = np.asarray(movement, dtype=np.float64).reshape(-1)
    if move.shape != source.shape or not np.isfinite(move).all():
        raise ValueError("movement must be finite and match the Native table")
    if not np.all((0.0 <= move) & (move <= 1.0)):
        raise ValueError("movement must lie in [0, 1]")
    if not math.isfinite(float(factor)) or float(factor) <= 1.0:
        raise ValueError("factor must be finite and greater than one")
    table = np.ascontiguousarray(
        source * (1.0 - move) + (source / float(factor)) * move,
        dtype="<f4",
    )
    native_f32 = np.ascontiguousarray(source, dtype="<f4")
    table[0] = native_f32[0]
    table[-1] = np.float32(native_f32[-1] / float(factor))
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError("same-support table is not strictly decreasing")
    return table


def phase_resolved_uniqueness_control(
    native: np.ndarray,
    *,
    native_context_length: int,
    factor: float = 4.0,
    exponent: float = 2.0,
    maximum_distance_stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the uniqueness table at a model-relative distance resolution.

    The original OLMo construction used 2048 points for a 4096-token Native
    window, i.e. a maximum distance stride of two.  Keeping that resolution
    model-relative avoids the stride-16 aliasing produced by a fixed 2048-point
    cap on a 32768-token Native window.
    """

    source = _source(native)
    length = int(native_context_length)
    stride = int(maximum_distance_stride)
    if length <= 0 or stride <= 0:
        raise ValueError("Native length and maximum distance stride must be positive")
    support_points = int(math.ceil(length / float(stride)))
    support, weight = causal_distance_measure(
        length=length,
        max_points=support_points,
    )
    uniqueness = conditional_pair_uniqueness(source, support, weight)
    span = float(uniqueness.max()) - float(uniqueness.min())
    normalized = (
        np.zeros_like(uniqueness)
        if span <= 0.0
        else (uniqueness - float(uniqueness.min())) / span
    )
    movement = (1.0 - normalized) ** float(exponent)
    table = _same_support_table(source, movement, factor=float(factor))
    receipt = {
        "construction": "phase_resolved_conditional_uniqueness",
        "factor": float(factor),
        "exponent": float(exponent),
        "native_context_length": length,
        "support_points": int(support.size),
        "maximum_distance_stride": stride,
        "fast_endpoint_exact_native": bool(table[0] == np.float32(source[0])),
        "slow_endpoint_exact_native_div_factor": bool(
            table[-1] == np.float32(np.float32(source[-1]) / float(factor))
        ),
        "order_crossing_indices": np.flatnonzero(table[:-1] <= table[1:]).tolist(),
        "active_sha256_float32": float32_sha256(table),
    }
    return table, movement, receipt


def geometric_same_support_control(
    native: np.ndarray,
    *,
    factor: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the log-linear table with the same two frequency endpoints."""

    source = _source(native)
    log_fast = math.log(float(source[0]))
    log_slow = math.log(float(source[-1]) / float(factor))
    table = np.ascontiguousarray(
        np.exp(np.linspace(log_fast, log_slow, source.size, dtype=np.float64)),
        dtype="<f4",
    )
    native_f32 = np.ascontiguousarray(source, dtype="<f4")
    table[0] = native_f32[0]
    table[-1] = np.float32(native_f32[-1] / float(factor))
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError("geometric same-support table is not strictly decreasing")
    return table, {
        "construction": "geometric_log_linear_same_support",
        "factor": float(factor),
        "pairs": int(source.size),
        "fast_endpoint_exact_native": True,
        "slow_endpoint_exact_native_div_factor": True,
        "active_sha256_float32": float32_sha256(table),
    }


def nearest_linear_ramp_control(
    native: np.ndarray,
    target_movement: np.ndarray,
    *,
    factor: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a movement profile onto the discrete YaRN linear-ramp family."""

    source = _source(native)
    target = np.asarray(target_movement, dtype=np.float64).reshape(-1)
    if target.shape != source.shape:
        raise ValueError("target movement must match the Native table")
    indices = np.arange(source.size, dtype=np.float64)
    best: tuple[float, int, int, np.ndarray] | None = None
    for lower in range(source.size - 1):
        for upper in range(lower + 1, source.size):
            ramp = np.clip((indices - float(lower)) / float(upper - lower), 0.0, 1.0)
            error = float(np.mean((target - ramp) ** 2))
            candidate = (error, lower, upper, ramp)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    error, lower, upper, ramp = best
    table = _same_support_table(source, ramp, factor=float(factor))
    return table, {
        "construction": "nearest_discrete_yarn_linear_ramp",
        "selection_uses_task_labels": False,
        "factor": float(factor),
        "lower_pair": int(lower),
        "upper_pair": int(upper),
        "movement_mse_to_uniqueness": error,
        "movement_max_abs_to_uniqueness": float(np.max(np.abs(target - ramp))),
        "fast_endpoint_exact_native": True,
        "slow_endpoint_exact_native_div_factor": True,
        "active_sha256_float32": float32_sha256(table),
    }


def build_same_support_controls(
    native: np.ndarray,
    *,
    native_context_length: int,
    factor: float = 4.0,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    uniqueness, movement, uniqueness_receipt = phase_resolved_uniqueness_control(
        native,
        native_context_length=int(native_context_length),
        factor=float(factor),
    )
    geometric, geometric_receipt = geometric_same_support_control(
        native,
        factor=float(factor),
    )
    ramp, ramp_receipt = nearest_linear_ramp_control(
        native,
        movement,
        factor=float(factor),
    )
    return {
        "converged_budgeted_s4": (uniqueness, uniqueness_receipt),
        "same_support_geometric_s4": (geometric, geometric_receipt),
        "nearest_yarn_ramp_s4": (ramp, ramp_receipt),
    }
