#!/usr/bin/env python3
"""Build the frozen OLMo Native Contrastive Proximal (NCP) table.

NCP uses only the runtime Native RoPE table and its public Native length.  It
does not load model weights, activations, gradients, calibration data, or task
outputs.  The first experiment is deliberately frozen to the OLMo 4K/64-pair
geometry audited for this candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import METHOD_ID


TABLE_STATUS = "FROZEN_NATIVE_CONTRASTIVE_PROXIMAL_TABLE_V1"
AUDIT_STATUS = "NATIVE_CONTRASTIVE_PROXIMAL_CPU_AUDIT_V1"
FOURIER_GRID_POINTS = 1024
FOURIER_MODES = 32
PROXIMAL_CURVATURE = 9.0 / 4.0
MAX_LOG_SHIFT = 2.0 / 9.0
MIN_LOG_GAP_FRACTION = 0.5
EXPECTED_OLMO_TABLE_SHA256 = "54b9dd1f73aafc69f7bb5ed1b7b49d49128002371cb378d03ca1fd1d108e0cb7"


def tensor_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_native(values: Any) -> np.ndarray:
    native = np.asarray(values, dtype=np.float32)
    if (
        native.ndim != 1
        or native.size < 4
        or not np.isfinite(native).all()
        or np.any(native <= 0.0)
        or np.any(native[:-1] <= native[1:])
    ):
        raise ValueError("Native inverse frequencies must be positive and strictly decreasing")
    return np.ascontiguousarray(native)


def reference_fourier(
    *, grid_points: int = FOURIER_GRID_POINTS, modes: int = FOURIER_MODES,
) -> tuple[float, np.ndarray]:
    """Return the Fourier series of the binary reference retrieval risk."""
    if grid_points < 4 or grid_points % 2 or not 0 < modes < grid_points // 2:
        raise ValueError("invalid Fourier discretization")
    angles = 2.0 * math.pi * np.arange(grid_points, dtype=np.float64) / grid_points
    cosine = np.cos(angles)
    phase_risk = np.logaddexp(
        0.0,
        cosine[None, :] - cosine[:, None],
    ).mean(axis=1)
    spectrum = np.fft.rfft(phase_risk).real / grid_points
    return float(spectrum[0]), np.ascontiguousarray(2.0 * spectrum[1 : modes + 1])


def _kernel(values: np.ndarray) -> np.ndarray:
    return np.sinc(values / (2.0 * math.pi)) ** 2


def reference_risk(phase: Any, *, a0: float, coefficients: np.ndarray) -> np.ndarray:
    phase_array = np.asarray(phase, dtype=np.float64)
    modes = np.arange(1, coefficients.size + 1, dtype=np.float64)
    values = phase_array[..., None] * modes
    return a0 + np.sum(coefficients * _kernel(values), axis=-1)


def reference_log_gradient(
    phase: Any, *, coefficients: np.ndarray,
) -> np.ndarray:
    """Return d R(phi) / d log(phi) for the Fourier risk."""
    phase_array = np.asarray(phase, dtype=np.float64)
    modes = np.arange(1, coefficients.size + 1, dtype=np.float64)
    values = phase_array[..., None] * modes
    kernel = _kernel(values)
    terms = 2.0 * (np.sinc(values / math.pi) - kernel)
    small = np.abs(values) < 1e-3
    series = -(values**2) / 6.0 + (values**4) / 90.0 - (values**6) / 3360.0
    terms = np.where(small, series, terms)
    return np.sum(coefficients * terms, axis=-1)


def _independent_shift(
    phase: float, *, coefficients: np.ndarray, tolerance: float = 1e-14,
) -> float:
    """Solve 9u/4 = q(phi exp(-u)) on the frozen feasible interval."""
    initial_gradient = float(reference_log_gradient(phase, coefficients=coefficients))
    if initial_gradient <= 1e-13:
        return 0.0

    def derivative(value: float) -> float:
        shifted = phase * math.exp(-value)
        return PROXIMAL_CURVATURE * value - float(
            reference_log_gradient(shifted, coefficients=coefficients)
        )

    low, high = 0.0, MAX_LOG_SHIFT
    if derivative(high) < 0.0:
        raise RuntimeError("NCP optimum exceeds the derived maximum log shift")
    for _ in range(96):
        midpoint = 0.5 * (low + high)
        if derivative(midpoint) > 0.0:
            high = midpoint
        else:
            low = midpoint
        if high - low <= tolerance:
            break
    return 0.5 * (low + high)


def build_ncp_arrays(native_values: Any, *, native_length: int) -> dict[str, Any]:
    """Return the exact FP32 NCP table and its construction audit values."""
    native = _validate_native(native_values)
    if native_length < 2:
        raise ValueError("native_length must be at least two")
    distance = native_length - 1
    phase = distance * native.astype(np.float64)
    a0, coefficients = reference_fourier()
    shifts = np.asarray([
        _independent_shift(float(value), coefficients=coefficients) for value in phase
    ])
    shifts[0] = 0.0
    shifts[-1] = 0.0

    candidate = (native.astype(np.float64) * np.exp(-shifts)).astype(np.float32)
    candidate[0] = native[0]
    candidate[-1] = native[-1]
    _validate_native(candidate)

    native_gaps = np.diff(-np.log(native.astype(np.float64)))
    candidate_gaps = np.diff(-np.log(candidate.astype(np.float64)))
    gap_ratios = candidate_gaps / native_gaps
    if float(gap_ratios.min()) < MIN_LOG_GAP_FRACTION - 1e-7:
        raise RuntimeError(
            "the coupled NCP gap constraint activates; this frozen OLMo experiment "
            "accepts only the audited independent solution"
        )

    native_risk = reference_risk(phase, a0=a0, coefficients=coefficients)
    candidate_phase = phase * np.exp(-shifts)
    candidate_risk = reference_risk(candidate_phase, a0=a0, coefficients=coefficients)
    active = shifts > 0.0
    residuals = (
        PROXIMAL_CURVATURE * shifts[active]
        - reference_log_gradient(candidate_phase[active], coefficients=coefficients)
    )
    proximal_native = float(np.mean(native_risk))
    proximal_candidate = float(
        np.mean(candidate_risk + 0.5 * PROXIMAL_CURVATURE * shifts**2)
    )
    checks = {
        "native_positive_strictly_decreasing": True,
        "candidate_positive_strictly_decreasing": bool(
            np.all(candidate > 0.0) and np.all(candidate[:-1] > candidate[1:])
        ),
        "endpoints_bit_exact": bool(
            candidate[0] == native[0] and candidate[-1] == native[-1]
        ),
        "one_sided_slowdown": bool(np.all(candidate <= native)),
        "maximum_log_shift_within_bound": bool(float(shifts.max()) <= MAX_LOG_SHIFT),
        "minimum_log_gap_preserved": bool(float(gap_ratios.min()) >= MIN_LOG_GAP_FRACTION),
        "stationary_equations_solved": bool(
            not residuals.size or float(np.max(np.abs(residuals))) < 1e-11
        ),
        "reference_risk_decreased": bool(float(np.mean(candidate_risk)) < proximal_native),
        "proximal_objective_decreased": bool(proximal_candidate < proximal_native),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"NCP construction failed CPU checks: {failed}")

    return {
        "native": native,
        "candidate": candidate,
        "phase": phase,
        "candidate_phase": candidate_phase,
        "log_shifts": shifts,
        "changed_indices": np.flatnonzero(candidate != native),
        "distance": distance,
        "a0": a0,
        "fourier_coefficients": coefficients,
        "reference_risk_native": native_risk,
        "reference_risk_candidate": candidate_risk,
        "proximal_objective_native": proximal_native,
        "proximal_objective_candidate": proximal_candidate,
        "minimum_log_gap_ratio": float(gap_ratios.min()),
        "maximum_stationarity_residual": float(
            np.max(np.abs(residuals)) if residuals.size else 0.0
        ),
        "checks": checks,
    }


def build_audit(
    arrays: dict[str, Any], *, model_id: str, config_sha256: str,
) -> dict[str, Any]:
    native = arrays["native"]
    candidate = arrays["candidate"]
    shifts = arrays["log_shifts"]
    distance = arrays["distance"]
    table_hash = tensor_sha256(candidate)
    return {
        "status": AUDIT_STATUS,
        "method_id": METHOD_ID,
        "model_id": model_id,
        "model_config_sha256": config_sha256,
        "public_inputs_only": True,
        "model_execution": False,
        "native_length": int(distance + 1),
        "pair_count": int(native.size),
        "fourier_grid_points": FOURIER_GRID_POINTS,
        "fourier_modes": FOURIER_MODES,
        "proximal_curvature": PROXIMAL_CURVATURE,
        "maximum_log_shift_bound": MAX_LOG_SHIFT,
        "minimum_log_gap_fraction": MIN_LOG_GAP_FRACTION,
        "changed_pair_indices_zero_based": arrays["changed_indices"].astype(int).tolist(),
        "changed_pair_count": int(arrays["changed_indices"].size),
        "maximum_relative_slowdown": float(np.max(1.0 - candidate / native)),
        "peak_shift_pair_zero_based": int(np.argmax(shifts)),
        "maximum_native_window_phase_change": float(
            np.max(distance * np.abs(candidate.astype(np.float64) - native.astype(np.float64)))
        ),
        "minimum_log_gap_ratio": arrays["minimum_log_gap_ratio"],
        "reference_risk_native_mean": float(np.mean(arrays["reference_risk_native"])),
        "reference_risk_candidate_mean": float(np.mean(arrays["reference_risk_candidate"])),
        "proximal_objective_native_mean": arrays["proximal_objective_native"],
        "proximal_objective_candidate_mean": arrays["proximal_objective_candidate"],
        "maximum_stationarity_residual": arrays["maximum_stationarity_residual"],
        "native_table_sha256_float32": tensor_sha256(native),
        "candidate_table_sha256_float32": table_hash,
        "expected_olmo_table_sha256_float32": EXPECTED_OLMO_TABLE_SHA256,
        "checks": {
            **arrays["checks"],
            "frozen_olmo_table_identity": table_hash == EXPECTED_OLMO_TABLE_SHA256,
        },
        "claim_boundary": (
            "CPU construction audit only. It verifies the frozen public-parameter NCP "
            "table and its reference-risk objective; it does not establish checkpoint "
            "or task improvement."
        ),
    }


def table_receipt(
    arrays: dict[str, Any], *, model_id: str, config_sha256: str,
) -> dict[str, Any]:
    values = arrays["candidate"]
    native = arrays["native"]
    return {
        "status": TABLE_STATUS,
        "candidate_id": f"{model_id}_native_contrastive_proximal_v1",
        "model_id": model_id,
        "role": "candidate",
        "values_float32": values.astype(np.float32).tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(values),
        "native_table_sha256_float32": tensor_sha256(native),
        "construction": {
            "method_id": METHOD_ID,
            "native_length": int(arrays["distance"] + 1),
            "changed_variable": "interior_native_rope_frequency_allocation",
            "fastest_and_slowest_frequency_fixed": True,
            "frequency_slot_assignment_fixed": True,
            "same_table_all_layers_and_inputs": True,
            "runtime_table_switching": False,
            "model_weights_or_outputs_used": False,
            "gain_rule": "Native gain 1",
            "proximal_curvature": PROXIMAL_CURVATURE,
            "maximum_log_shift": MAX_LOG_SHIFT,
            "minimum_log_gap_fraction": MIN_LOG_GAP_FRACTION,
            "fourier_grid_points": FOURIER_GRID_POINTS,
            "fourier_modes": FOURIER_MODES,
            "model_config_sha256": config_sha256,
        },
    }


def _write_frozen_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing frozen artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    from experiments.fixed_rope_three_interfaces_20260913.tables import (
        model_geometry,
        runtime_native_inv_freq,
    )

    geometry = model_geometry(config)
    expected_geometry = {"native_length": 4096, "pairs": 64, "base": 500000.0}
    if any(geometry[key] != value for key, value in expected_geometry.items()):
        raise ValueError(f"the frozen first NCP experiment requires {expected_geometry}")
    native = runtime_native_inv_freq(geometry)
    arrays = build_ncp_arrays(native, native_length=int(geometry["native_length"]))
    config_hash = file_sha256(args.config)
    audit = build_audit(arrays, model_id=args.model_id, config_sha256=config_hash)
    if not all(audit["checks"].values()):
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"frozen OLMo NCP audit failed: {failed}")
    _write_frozen_json(
        args.out / "ncp.json",
        table_receipt(arrays, model_id=args.model_id, config_sha256=config_hash),
    )
    _write_frozen_json(args.out / "cpu_audit.json", audit)
    print(json.dumps({
        "status": AUDIT_STATUS,
        "changed_pairs": audit["changed_pair_count"],
        "table_sha256_float32": audit["candidate_table_sha256_float32"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

