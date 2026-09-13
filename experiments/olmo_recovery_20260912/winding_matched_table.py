#!/usr/bin/env python3
"""Build the author's closed-form Winding-Matched fixed RoPE table."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def circular_distance(angle: np.ndarray) -> np.ndarray:
    return np.abs((angle + math.pi) % (2.0 * math.pi) - math.pi)


def winding_matched(native_values, *, native_length: int, scale: float) -> dict:
    native = np.asarray(native_values, dtype=np.float64)
    if (
        native.ndim != 1
        or native.size == 0
        or not np.isfinite(native).all()
        or np.any(native <= 0.0)
        or np.any(native[:-1] <= native[1:])
        or native_length < 1
        or not math.isfinite(scale)
        or scale <= 1.0
    ):
        raise ValueError("invalid Native table, length, or scale")
    turns = native_length * native / (2.0 * math.pi)
    winding = np.floor((scale - 1.0) * turns).astype(np.int64)
    horizon = scale * native_length
    values64 = native / scale + (2.0 * math.pi * winding) / horizon
    if np.any(values64 < native / scale - 1e-15) or np.any(values64 > native + 1e-15):
        raise AssertionError("closed-form frequency left the non-acceleration interval")
    if np.any(values64[:-1] <= values64[1:]):
        raise AssertionError("closed-form frequency order failed")
    # Increasing any winding by one must leave the non-accelerating set.
    next_values = native / scale + (2.0 * math.pi * (winding + 1)) / horizon
    if np.any(next_values <= native):
        raise AssertionError("selected winding was not maximal")
    residual64 = circular_distance(horizon * values64 - native_length * native)
    values32 = values64.astype(np.float32)
    native32 = native.astype(np.float32)
    phase_extended32 = np.float32(horizon) * values32
    phase_native32 = np.float32(native_length) * native32
    residual32 = circular_distance(
        phase_extended32.astype(np.float64) - phase_native32.astype(np.float64)
    )
    exponent = -np.log(values64 / native) / math.log(scale)
    exponent_decreases = np.flatnonzero(np.diff(exponent) < -1e-10)
    return {
        "values_float32": values32.tolist(),
        "gain": 1.0 + 0.1 * math.log(scale),
        "construction": {
            "method": "winding_matched_rope_v1",
            "origin": "proposed by the project author on 2026-09-13",
            "native_length": native_length,
            "scale": scale,
            "horizon": horizon,
            "winding_numbers": winding.tolist(),
            "native_turns_float64": turns.tolist(),
            "exponents_float64": exponent.tolist(),
            "exponent_monotonicity_violations": exponent_decreases.tolist(),
            "max_endpoint_circular_residual_float64": float(residual64.max()),
            "max_endpoint_circular_residual_simulated_float32": float(residual32.max()),
            "min_adjacent_frequency_gap_float64": float(np.min(values64[:-1] - values64[1:])),
            "same_table_all_layers_and_lengths": True,
            "model_weight_updates": 0,
            "claim_boundary": "exact-arithmetic endpoint operator property; task utility requires generation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-length", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoConfig
    from .recovery_v2_runtime import table_for_config

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    native = table_for_config(config, "Native")["values_float32"]
    table = winding_matched(native, native_length=args.native_length, scale=args.scale)
    payload = {
        "status": "FROZEN",
        "label": args.label,
        "table": table,
        "scope": "author-proposed zero-training fixed table; no task claim before GPU generation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("output already exists")
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "label": args.label,
        "max_residual_float64": table["construction"]["max_endpoint_circular_residual_float64"],
        "max_residual_float32": table["construction"]["max_endpoint_circular_residual_simulated_float32"],
        "m_monotonicity_violations": len(table["construction"]["exponent_monotonicity_violations"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
