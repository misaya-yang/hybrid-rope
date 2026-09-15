"""Reflect a frozen candidate around Native to match per-slot phase dose.

This is an attribution control, not a new candidate or an optimization of task
quality. Invalid positive/order geometry is rejected, never clipped to fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reflection(native_values, candidate_values, *, native_length: int) -> dict:
    native = np.asarray(native_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if native.ndim != 1 or native.size < 3 or candidate.shape != native.shape or native_length < 2:
        raise ValueError("invalid table dimensions or native length")
    for table in (native, candidate):
        if not np.isfinite(table).all() or not np.all(table > 0) or not np.all(np.diff(table) < 0):
            raise ValueError("tables must be positive finite and strictly decreasing")
    if not np.array_equal(native[[0, -1]], candidate[[0, -1]]):
        raise ValueError("candidate must preserve Native endpoints")
    reflected64 = 2 * native.astype(np.float64) - candidate.astype(np.float64)
    reflected = reflected64.astype(np.float32)
    if not np.all(reflected > 0) or not np.all(np.diff(reflected) < 0):
        raise ValueError("phase reflection is not a valid ordered table; no clipping is allowed")
    plus = reflected.astype(np.float64) - native.astype(np.float64)
    minus = candidate.astype(np.float64) - native.astype(np.float64)
    absolute_residual = np.abs(plus + minus)
    # Rounding can affect absolute phase equality, so verify a scale-aware ULP bound.
    round_bound = np.abs(np.spacing(reflected)) / 2
    if not np.all(absolute_residual <= round_bound + np.finfo(float).eps):
        raise RuntimeError("FP32 reflection error exceeds rounding bound")
    return {"status": "CPU_PREPARED_PHASE_REFLECTION_CONTROL_V1",
            "role": "attribution_control", "values_float32": reflected.tolist(), "gain": 1.0,
            "construction": {"rule": "omega_reflection = 2*omega_native - omega_candidate",
                             "native_length": native_length, "endpoints_fixed": True,
                             "free_strength_parameters": 0, "model_outputs_used": False},
            "audit": {"maximum_phase_magnitude_mismatch_fp32": float((native_length - 1) * absolute_residual.max()),
                      "maximum_candidate_phase_displacement": float((native_length - 1) * np.abs(minus).max()),
                      "endpoints_bit_exact": bool(np.array_equal(reflected[[0, -1]], native[[0, -1]])),
                      "positive_ordered": True,
                      "minimum_log_gap_ratio": float(np.min(np.diff(-np.log(reflected.astype(float))) / np.diff(-np.log(native.astype(float)))))},
            "scope": "Same per-slot absolute phase displacement at every distance before FP32 rounding; signed response differs. Does not isolate NCP reference-risk optimality."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-table", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--native-length", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    def values(path):
        payload = json.loads(path.read_text())
        return payload.get("table", payload)["values_float32"]
    result = reflection(values(args.native_table), values(args.candidate_table), native_length=args.native_length)
    if args.out.exists() and json.loads(args.out.read_text()) != result:
        raise ValueError("refusing to overwrite another frozen control")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["audit"]))


if __name__ == "__main__":
    main()
