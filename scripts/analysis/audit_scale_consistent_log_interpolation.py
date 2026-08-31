#!/usr/bin/env python3
"""Audit arithmetic versus scale-consistent log-frequency interpolation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.export_uniqueness_budgeted_tables import (
    causal_distance_measure,
    conditional_pair_uniqueness,
    float32_sha256,
    native_endpoint_inv_freq,
)


FACTORS = (2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def frozen_movement(native: np.ndarray) -> np.ndarray:
    support, weight = causal_distance_measure()
    uniqueness = conditional_pair_uniqueness(native, support, weight)
    span = float(uniqueness.max()) - float(uniqueness.min())
    normalized = (uniqueness - float(uniqueness.min())) / span
    return np.ascontiguousarray((1.0 - normalized) ** 2.0, dtype=np.float64)


def summarize(table: np.ndarray, native: np.ndarray, movement: np.ndarray, factor: int) -> dict[str, object]:
    displacement = -np.log(table / native)
    normalized = displacement / math.log(float(factor))
    table32 = np.ascontiguousarray(table, dtype="<f4")
    return {
        "sha256_float32": float32_sha256(table32),
        "crossing_indices": np.flatnonzero(table32[:-1] <= table32[1:]).tolist(),
        "fast_endpoint_ratio": float(table32[0] / np.float32(native[0])),
        "slow_endpoint_ratio": float(table32[-1] / np.float32(native[-1])),
        "maximum_normalized_displacement_error_from_m": float(
            np.max(np.abs(normalized - movement))
        ),
        "displacement": displacement.tolist(),
        "normalized_displacement": normalized.tolist(),
        "frequency_ratio": (table / native).tolist(),
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    native = native_endpoint_inv_freq().astype(np.float64)
    movement = frozen_movement(native)
    receipt: dict[str, object] = {
        "status": "SCALE_CONSISTENT_LOG_INTERPOLATION_AUDIT_COMPLETE",
        "factors": list(FACTORS),
        "movement_exponent": 2.0,
        "movement": movement.tolist(),
        "methods": {},
    }

    colors = {2: "#2563eb", 4: "#d97706", 8: "#dc2626"}
    figure, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)
    for factor in FACTORS:
        arithmetic = native * ((1.0 - movement) + movement / float(factor))
        logarithmic = native * np.power(float(factor), -movement)
        factor_rows: dict[str, object] = {}
        for name, table, style in (
            ("arithmetic", arithmetic, "--"),
            ("logarithmic", logarithmic, "-"),
        ):
            table32 = np.ascontiguousarray(table, dtype="<f4")
            if not np.all(table32[:-1] > table32[1:]):
                raise RuntimeError(f"{name} s{factor} is not strictly decreasing")
            np.save(output / f"{name}_s{factor}.npy", table32, allow_pickle=False)
            row = summarize(table, native, movement, factor)
            factor_rows[name] = row
            label = f"{name} s={factor}"
            axes[0].plot(row["displacement"], style, color=colors[factor], label=label)
            axes[1].plot(row["normalized_displacement"], style, color=colors[factor], label=label)
            axes[2].plot(row["frequency_ratio"], style, color=colors[factor], label=label)
        receipt["methods"][str(factor)] = factor_rows

    axes[0].set_title("Log-frequency displacement")
    axes[0].set_ylabel(r"$x_{long}-x_{native}$")
    axes[1].set_title("Normalized displacement")
    axes[1].plot(movement, ":", color="black", linewidth=2, label="frozen movement m")
    axes[1].set_ylabel(r"$\Delta x / \log s$")
    axes[2].set_title("Frequency ratio")
    axes[2].set_ylabel(r"$\omega_{long}/\omega_{native}$")
    axes[2].set_xlabel("rotary frequency-pair index k")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.savefig(output / "scale_interpolation_audit.png", dpi=180)
    plt.close(figure)
    (output / "audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
