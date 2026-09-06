#!/usr/bin/env python3
"""Audit point sampling versus finite-cell projection of a frozen G(x) law."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_X_HIGH = 0.7382780681078285
DEFAULT_X_LOW = 0.366403835112904


@dataclass(frozen=True)
class ModelGrid:
    name: str
    pairs: int
    rope_theta: float
    native_length: int


MODELS = (
    ModelGrid("olmo_k64", 64, 500_000.0, 4_096),
    ModelGrid("qwen_k64", 64, 1_000_000.0, 32_768),
    ModelGrid("qwen_k32", 32, 1_000_000.0, 32_768),
)


def clipped_affine(x: np.ndarray, *, x_high: float, x_low: float) -> np.ndarray:
    if not x_high > x_low:
        raise ValueError("x_high must exceed x_low")
    return np.clip((x_high - x) / (x_high - x_low), 0.0, 1.0)


def clipped_affine_antiderivative(
    x: np.ndarray,
    *,
    x_high: float,
    x_low: float,
) -> np.ndarray:
    """Continuous antiderivative whose derivative is clipped_affine."""

    width = x_high - x_low
    value = np.empty_like(x, dtype=np.float64)
    slow = x <= x_low
    fast = x >= x_high
    middle = ~(slow | fast)
    value[slow] = x[slow]
    delta = x[middle] - x_low
    value[middle] = x_low + delta - delta**2 / (2.0 * width)
    value[fast] = x_low + width / 2.0
    return value


def cell_average(
    centers: np.ndarray,
    *,
    spacing: float,
    x_high: float,
    x_low: float,
) -> np.ndarray:
    """Project G onto equal-width midpoint/Voronoi cells of the log grid."""

    lower = centers - spacing / 2.0
    upper = centers + spacing / 2.0
    integral = clipped_affine_antiderivative(
        upper,
        x_high=x_high,
        x_low=x_low,
    ) - clipped_affine_antiderivative(
        lower,
        x_high=x_high,
        x_low=x_low,
    )
    return np.clip(integral / spacing, 0.0, 1.0)


def audit_grid(
    grid: ModelGrid,
    *,
    x_high: float,
    x_low: float,
) -> dict:
    pairs = int(grid.pairs)
    spacing = math.log(float(grid.rope_theta)) / pairs
    c_orth = 1.0 / (1.0 - float(grid.rope_theta) ** (-1.0 / pairs))
    x_zero = math.log(float(grid.native_length) / (2.0 * math.pi * c_orth))
    slots = np.arange(pairs, dtype=np.float64)
    centers = x_zero - spacing * slots
    point = clipped_affine(centers, x_high=x_high, x_low=x_low)
    cell = cell_average(
        centers,
        spacing=spacing,
        x_high=x_high,
        x_low=x_low,
    )
    difference = cell - point
    tolerance = 1e-10
    soft_point = np.flatnonzero(
        (point > tolerance) & (point < 1.0 - tolerance)
    )
    soft_cell = np.flatnonzero(
        (cell > tolerance) & (cell < 1.0 - tolerance)
    )
    changed = np.flatnonzero(np.abs(difference) > tolerance)
    fast_shoulder = np.flatnonzero(
        (point <= tolerance) & (cell > tolerance)
    )
    slow_shoulder = np.flatnonzero(
        (point >= 1.0 - tolerance) & (cell < 1.0 - tolerance)
    )

    def boundary(value: float) -> dict:
        continuous_slot = (x_zero - value) / spacing
        return {
            "x": value,
            "continuous_slot": continuous_slot,
            "bracketing_slots": [math.floor(continuous_slot), math.ceil(continuous_slot)],
        }

    return {
        "grid": asdict(grid),
        "delta_x": spacing,
        "transition_width": x_high - x_low,
        "eta": (x_high - x_low) / spacing,
        "x_zero": x_zero,
        "x_high_location": boundary(x_high),
        "x_low_location": boundary(x_low),
        "point_soft_slots": soft_point.tolist(),
        "cell_soft_slots": soft_cell.tolist(),
        "changed_slots": changed.tolist(),
        "fast_side_shoulders": fast_shoulder.tolist(),
        "slow_side_shoulders": slow_shoulder.tolist(),
        "point_movement": point.tolist(),
        "cell_movement": cell.tolist(),
        "cell_minus_point": difference.tolist(),
        "max_abs_cell_minus_point": float(np.max(np.abs(difference))),
        "mean_abs_cell_minus_point": float(np.mean(np.abs(difference))),
        "rms_cell_minus_point": float(np.sqrt(np.mean(difference**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-high", type=float, default=DEFAULT_X_HIGH)
    parser.add_argument("--x-low", type=float, default=DEFAULT_X_LOW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.x_high > args.x_low:
        raise ValueError("x-high must exceed x-low")

    named = {
        grid.name: audit_grid(
            grid,
            x_high=float(args.x_high),
            x_low=float(args.x_low),
        )
        for grid in MODELS
    }
    sweep = {
        str(pairs): audit_grid(
            ModelGrid(f"qwen_k{pairs}", pairs, 1_000_000.0, 32_768),
            x_high=float(args.x_high),
            x_low=float(args.x_low),
        )
        for pairs in (16, 32, 64, 128)
    }
    receipt = {
        "status": "FINITE_K_COUPLING_GEOMETRY_AUDIT_COMPLETE",
        "benchmark_scores_used": False,
        "law": {
            "family": "clipped_affine",
            "x_high": float(args.x_high),
            "x_low": float(args.x_low),
            "transition_width": float(args.x_high - args.x_low),
        },
        "cell_definition": "equal-width log-frequency Voronoi cell centered at each native slot, extended by half a lattice step at both endpoints",
        "named_models": named,
        "qwen_geometry_k_sweep": sweep,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
