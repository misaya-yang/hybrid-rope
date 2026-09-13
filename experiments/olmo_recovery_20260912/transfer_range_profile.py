#!/usr/bin/env python3
"""Transfer a normalized exponent allocation to another RoPE model/horizon."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def transfer_profile(
    solver_result: dict,
    native_values: list[float] | np.ndarray,
    *,
    target_scale: float,
    gain_policy: str = "target_yarn",
    target_low: int | None = None,
    target_high: int | None = None,
    tail_depth: float = 1.0,
) -> dict:
    allocation = solver_result.get("allocation", solver_result.get("table", {}).get("construction", {}))
    exponents = np.asarray(allocation.get("exponents"), dtype=np.float64)
    native = np.asarray(native_values, dtype=np.float64)
    source_scale = float(allocation.get("scale"))
    source_gain = float(allocation.get("gain", solver_result.get("table", {}).get("gain")))
    if (
        exponents.shape != native.shape
        or not np.isfinite(exponents).all()
        or np.any(np.diff(exponents) < -1e-10)
        or exponents.min() < -1e-10
        or exponents.max() > 1.0 + 1e-10
        or not np.isfinite(native).all()
        or np.any(native <= 0.0)
        or np.any(native[:-1] <= native[1:])
        or not math.isfinite(target_scale)
        or target_scale <= 1.0
        or not math.isfinite(source_scale)
        or source_scale <= 1.0
        or not math.isfinite(source_gain)
        or source_gain <= 0.0
    ):
        raise ValueError("invalid source allocation, native table, or target scale")
    if (target_low is None) != (target_high is None):
        raise ValueError("target-low and target-high must be supplied together")
    if not math.isfinite(tail_depth) or not 0.0 < tail_depth <= 1.0:
        raise ValueError("tail_depth must be finite and in (0, 1]")
    if target_low is not None:
        source_low = int(allocation.get("low"))
        source_high = int(allocation.get("high"))
        if not (0 <= source_low < source_high < len(exponents)):
            raise ValueError("invalid source transition band")
        if not (0 <= target_low < target_high < len(exponents)):
            raise ValueError("invalid target transition band")
        source_x = np.linspace(0.0, 1.0, source_high - source_low + 1)
        target_x = np.linspace(0.0, 1.0, target_high - target_low + 1)
        remapped = np.empty_like(exponents)
        remapped[: target_low + 1] = 0.0
        remapped[target_low : target_high + 1] = np.interp(
            target_x, source_x, exponents[source_low : source_high + 1]
        )
        remapped[target_high:] = 1.0
        exponents = remapped
        band_policy = "normalized_transition_band_remap"
    else:
        source_low = int(allocation.get("low", -1))
        source_high = int(allocation.get("high", -1))
        band_policy = "literal_slot_transfer"
    full_depth_exponents = exponents.copy()
    exponents = full_depth_exponents * tail_depth
    target_yarn_gain = 1.0 + 0.1 * math.log(target_scale)
    if gain_policy == "target_yarn":
        gain = target_yarn_gain
    elif gain_policy == "relative_solver":
        gain = target_yarn_gain * source_gain / (1.0 + 0.1 * math.log(source_scale))
    else:
        raise ValueError(f"unsupported gain policy: {gain_policy}")
    values = (native * np.power(target_scale, -exponents)).astype(np.float32)
    if np.any(values[:-1] <= values[1:]):
        raise ValueError("transferred table is not strictly decreasing")
    return {
        "values_float32": values.tolist(),
        "gain": gain,
        "construction": {
            "method": "normalized_model_conditioned_profile_transfer_v1",
            "source_scale": source_scale,
            "target_scale": target_scale,
            "exponents": exponents.tolist(),
            "source_low": source_low,
            "source_high": source_high,
            "target_low": target_low,
            "target_high": target_high,
            "band_policy": band_policy,
            "full_depth_source": band_policy,
            "full_depth_exponents": full_depth_exponents.tolist(),
            "full_depth_exponent_sum": float(full_depth_exponents.sum()),
            "full_depth_exponent_max": float(full_depth_exponents.max()),
            "tail_depth": float(tail_depth),
            "final_exponent_sum": float(exponents.sum()),
            "final_exponent_max": float(exponents.max()),
            "gain_policy": gain_policy,
            "same_table_all_layers_and_lengths": True,
            "model_weight_updates": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-profile", choices=("solver_result", "C42V24"), default="solver_result")
    parser.add_argument("--solver-result", type=Path)
    parser.add_argument("--target-scale", type=float, required=True)
    parser.add_argument("--gain-policy", choices=("target_yarn", "relative_solver"), default="target_yarn")
    parser.add_argument("--target-low", type=int)
    parser.add_argument("--target-high", type=int)
    parser.add_argument("--tail-depth", type=float, default=1.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoConfig
    from .recovery_v2_runtime import table_for_config

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    native = table_for_config(config, "Native")["values_float32"]
    if args.source_profile == "solver_result":
        if args.solver_result is None:
            raise ValueError("solver_result source requires --solver-result")
        result = json.loads(args.solver_result.read_text())
        source_identity = str(args.solver_result)
    else:
        from experiments.rope_fast_5090_20260912.e3_tables import tables
        record = tables()["C42V24"]
        construction = record["construction"]
        exponents = construction["cumulative_exponents_float64"]
        first_positive = next(index for index, value in enumerate(exponents) if value > 0.0)
        first_full = next(index for index, value in enumerate(exponents) if value >= 1.0)
        result = {"allocation": {
            "exponents": exponents,
            "scale": float(construction["scale"]),
            "gain": float(record["gain"]),
            "low": first_positive - 1,
            "high": first_full,
        }}
        source_identity = "C42V24 exact polynomial reconstruction"
    table = transfer_profile(
        result,
        native,
        target_scale=args.target_scale,
        gain_policy=args.gain_policy,
        target_low=args.target_low,
        target_high=args.target_high,
        tail_depth=args.tail_depth,
    )
    payload = {
        "status": "FROZEN",
        "label": args.label,
        "source_profile": args.source_profile,
        "source_identity": source_identity,
        "table": table,
        "scope": "cross-model zero-training hypothesis; no transfer claim before target-model generation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("output already exists")
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "label": args.label, "gain": table["gain"]}, sort_keys=True))


if __name__ == "__main__":
    main()
