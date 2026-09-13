#!/usr/bin/env python3
"""Audit a frozen exponent allocation transported between deployment scales."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .tables import atomic_json, find_table, read_json, validate_table


def audit_scale_transfer(source: dict, target: dict) -> dict:
    if source.get("model_geometry") != target.get("model_geometry"):
        raise ValueError("scale-transfer receipts belong to different model geometries")
    source_scale = float(source.get("scale", float("nan")))
    target_scale = float(target.get("scale", float("nan")))
    if not 1.0 < source_scale < target_scale or not all(map(math.isfinite, (source_scale, target_scale))):
        raise ValueError("scale transfer requires 1 < source scale < target scale")
    pairs = int(source["model_geometry"]["pairs"])
    source_values, source_gain = validate_table(find_table(source), pairs=pairs)
    target_values, target_gain = validate_table(find_table(target), pairs=pairs)
    source_m = np.asarray(source.get("exponents"), dtype=np.float64)
    target_m = np.asarray(target.get("exponents"), dtype=np.float64)
    if source_m.shape != (pairs,) or target_m.shape != (pairs,):
        raise ValueError("scale-transfer receipts must expose complete exponent vectors")
    exponent_difference = target_m - source_m
    ratio_scale = target_scale / source_scale
    expected_frequency_ratio = np.power(ratio_scale, -source_m)
    observed_frequency_ratio = target_values.astype(np.float64) / source_values.astype(np.float64)
    endpoint_phase_ratio = ratio_scale * expected_frequency_ratio
    relative_error = np.abs(observed_frequency_ratio / expected_frequency_ratio - 1.0)
    same_allocation = bool(np.max(np.abs(exponent_difference)) <= 2e-6)
    return {
        "status": "FIXED_EXPONENT_SCALE_TRANSFER_AUDIT_V1",
        "source_candidate_id": source.get("candidate_id"),
        "target_candidate_id": target.get("candidate_id"),
        "source_scale": source_scale,
        "target_scale": target_scale,
        "scale_ratio": ratio_scale,
        "same_model_geometry": True,
        "same_exponent_allocation": same_allocation,
        "max_abs_exponent_difference": float(np.max(np.abs(exponent_difference))),
        "frequency_identity": "nu_target/nu_source=(S_target/S_source)^(-m)",
        "max_relative_frequency_identity_error": float(relative_error.max()),
        "source_gain": source_gain,
        "target_gain": target_gain,
        "gain_changed": bool(source_gain != target_gain),
        "common_absolute_distance_phase_ratio": observed_frequency_ratio.tolist(),
        "corresponding_endpoint_phase_ratio": endpoint_phase_ratio.tolist(),
        "endpoint_phase_ratio_range": [float(endpoint_phase_ratio.min()), float(endpoint_phase_ratio.max())],
        "interpretation": (
            "same m transported to a new deployment scale; common-length frequency effect and "
            "corresponding-endpoint text-length effect are distinct"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit_scale_transfer(read_json(args.source), read_json(args.target))
    if not result["same_exponent_allocation"]:
        raise ValueError("target table is not a frozen exponent-allocation transfer")
    atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "max_abs_exponent_difference": result["max_abs_exponent_difference"],
        "max_relative_frequency_identity_error": result["max_relative_frequency_identity_error"],
        "endpoint_phase_ratio_range": result["endpoint_phase_ratio_range"],
    }, indent=2))


if __name__ == "__main__":
    main()
