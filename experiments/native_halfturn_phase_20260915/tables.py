#!/usr/bin/env python3
"""Build and audit a fixed-support Native half-turn phase intervention.

The construction uses only the runtime Native RoPE table and its public Native
length.  It does not load model weights, activations, gradients, calibration
data, or task outputs.  ``contract`` moves eligible frequencies downward and
``reverse`` applies the same additive phase displacement upward.
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


TABLE_STATUS = "FROZEN_NATIVE_HALFTURN_TABLE_V1"
AUDIT_STATUS = "NATIVE_HALFTURN_CPU_AUDIT_V1"
DEFAULT_ETA = 0.25
DEFAULT_PHASE_CAP = math.pi


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


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def build_halfturn_arrays(
    native_values: Any,
    *,
    native_length: int,
    eta: float = DEFAULT_ETA,
    phase_cap: float = DEFAULT_PHASE_CAP,
) -> dict[str, Any]:
    """Return Native, phase-contracting and equal-dose reverse FP32 tables."""
    native = _validate_native(native_values)
    if native_length < 2:
        raise ValueError("native_length must be at least two")
    if not math.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    if not math.isfinite(phase_cap) or phase_cap <= 0.0:
        raise ValueError("phase_cap must be finite and positive")

    distance = native_length - 1
    phase = native.astype(np.float64) * distance
    lower = float(phase[-1])
    upper = min(float(phase[0]), float(phase_cap))
    active = (phase > lower) & (phase < upper) if lower < upper else np.zeros_like(phase, dtype=bool)
    displacement = np.zeros_like(phase)
    displacement[active] = eta * np.minimum(
        phase[active] - lower,
        upper - phase[active],
    )

    contract64 = (phase - displacement) / distance
    reverse64 = (phase + displacement) / distance
    contract = contract64.astype(np.float32)
    reverse = reverse64.astype(np.float32)
    for values in (contract, reverse):
        values[0] = native[0]
        values[-1] = native[-1]
        _validate_native(values)

    active_indices = np.flatnonzero(active)
    phase_delta_contract = (contract.astype(np.float64) - native.astype(np.float64)) * distance
    phase_delta_reverse = (reverse.astype(np.float64) - native.astype(np.float64)) * distance
    exact_bound = eta * max(upper - lower, 0.0) / 2.0
    fp32_dose_error = float(
        np.max(np.abs(np.abs(phase_delta_contract) - np.abs(phase_delta_reverse)))
    )

    checks = {
        "native_positive_strictly_decreasing": True,
        "eta_in_open_unit_interval": True,
        "endpoints_bit_exact_contract": bool(
            contract[0] == native[0] and contract[-1] == native[-1]
        ),
        "endpoints_bit_exact_reverse": bool(
            reverse[0] == native[0] and reverse[-1] == native[-1]
        ),
        "contract_positive_strictly_decreasing": bool(
            np.all(contract > 0.0) and np.all(contract[:-1] > contract[1:])
        ),
        "reverse_positive_strictly_decreasing": bool(
            np.all(reverse > 0.0) and np.all(reverse[:-1] > reverse[1:])
        ),
        "outside_band_bit_exact_contract": bool(np.array_equal(contract[~active], native[~active])),
        "outside_band_bit_exact_reverse": bool(np.array_equal(reverse[~active], native[~active])),
        "active_contract_direction": bool(not active.any() or np.all(contract[active] < native[active])),
        "active_reverse_direction": bool(not active.any() or np.all(reverse[active] > native[active])),
        "active_native_phase_below_cap": bool(not active.any() or np.all(phase[active] < upper)),
        "continuous_maps_strictly_increasing": bool(0.0 < 1.0 - eta and 1.0 + eta > 0.0),
        "precast_additive_phase_dose_exact": bool(
            np.array_equal(np.abs(contract64 - native.astype(np.float64)),
                           np.abs(reverse64 - native.astype(np.float64)))
        ),
        "maximum_phase_displacement_within_bound": bool(
            float(displacement.max(initial=0.0)) <= exact_bound + 1e-15
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"half-turn construction failed CPU checks: {failed}")

    log_fast = -math.log(float(native[0]))
    log_slow = -math.log(float(native[-1]))
    log_span = log_slow - log_fast
    normalized_z = {
        name: ((-np.log(values.astype(np.float64)) - log_fast) / log_span)
        for name, values in (("native", native), ("contract", contract), ("reverse", reverse))
    }
    checks["normalized_z_endpoints_fixed"] = all(
        abs(float(values[0])) <= 1e-12 and abs(float(values[-1]) - 1.0) <= 1e-7
        for values in normalized_z.values()
    )
    checks["normalized_z_strictly_increasing"] = all(
        bool(np.all(np.diff(values) > 0.0)) for values in normalized_z.values()
    )
    if not checks["normalized_z_endpoints_fixed"] or not checks["normalized_z_strictly_increasing"]:
        raise RuntimeError("half-turn construction violates normalized-z support")

    return {
        "native": native,
        "contract": contract,
        "reverse": reverse,
        "phase": phase,
        "displacement_precast": displacement,
        "phase_delta_contract_float32": phase_delta_contract,
        "phase_delta_reverse_float32": phase_delta_reverse,
        "normalized_z": normalized_z,
        "active_indices": active_indices,
        "lower_phase": lower,
        "upper_phase": upper,
        "distance": distance,
        "exact_phase_bound": exact_bound,
        "fp32_phase_dose_max_abs_error": fp32_dose_error,
        "checks": checks,
    }


def _table_receipt(
    *,
    values: np.ndarray,
    native: np.ndarray,
    model_id: str,
    arm: str,
    role: str,
    audit: dict[str, Any],
    eta: float,
    phase_cap: float,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "status": TABLE_STATUS,
        "candidate_id": f"{model_id}_native_halfturn_{arm}",
        "model_id": model_id,
        "role": role,
        "values_float32": values.astype(np.float32).tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(values),
        "native_table_sha256_float32": tensor_sha256(native),
        "construction": {
            "method_id": METHOD_ID,
            "arm": arm,
            "native_length": int(audit["distance"] + 1),
            "maximum_relative_distance": int(audit["distance"]),
            "eta": float(eta),
            "phase_cap": float(phase_cap),
            "lower_endpoint_phase": float(audit["lower_phase"]),
            "effective_upper_phase": float(audit["upper_phase"]),
            "active_pair_indices_zero_based": audit["active_indices"].astype(int).tolist(),
            "changed_variable": "interior_native_rope_frequency_allocation",
            "fastest_and_slowest_frequency_fixed": True,
            "frequency_slot_assignment_fixed": True,
            "model_weights_or_outputs_used": False,
            "gain_rule": "Native gain 1",
            "model_config_sha256": config_sha256,
        },
    }


def compare_v1(
    payload: dict[str, Any],
    arrays: dict[str, Any],
) -> dict[str, Any]:
    table = payload.get("table", payload)
    v1 = _validate_native(table.get("values_float32"))
    native = arrays["native"]
    if v1.shape != native.shape or float(table.get("gain", float("nan"))) != 1.0:
        raise ValueError("V1 comparison table has incompatible geometry or gain")
    expected_native_hash = table.get("native_table_sha256_float32")
    if expected_native_hash and expected_native_hash != tensor_sha256(native):
        raise ValueError("V1 comparison belongs to another Native table")
    distance = arrays["distance"]
    delta_v1 = (v1.astype(np.float64) - native.astype(np.float64)) * distance
    delta_contract = arrays["phase_delta_contract_float32"]
    delta_reverse = arrays["phase_delta_reverse_float32"]
    return {
        "identity": {
            "candidate_id": table.get("candidate_id"),
            "table_sha256_float32": tensor_sha256(v1),
        },
        "changed_pairs": int(np.count_nonzero(v1 != native)),
        "maximum_absolute_phase_change": float(np.max(np.abs(delta_v1))),
        "l2_phase_change": float(np.linalg.norm(delta_v1)),
        "signed_phase_change_sum": float(delta_v1.sum()),
        "cosine_with_contract_phase_direction": _cosine(delta_v1, delta_contract),
        "cosine_with_reverse_phase_direction": _cosine(delta_v1, delta_reverse),
        "contract_to_v1_l2_phase_ratio": float(
            np.linalg.norm(delta_contract) / np.linalg.norm(delta_v1)
        ) if np.linalg.norm(delta_v1) else None,
        "interpretation": (
            "Descriptive construction comparison only; V1 is checkpoint-calibrated and "
            "is not an equal-dose control for the half-turn intervention."
        ),
    }


def _rotation_distance(delta_frequency: np.ndarray, distance: int) -> np.ndarray:
    return 2.0 * np.abs(np.sin(0.5 * distance * delta_frequency.astype(np.float64)))


def build_audit(
    arrays: dict[str, Any],
    *,
    model_id: str,
    eta: float,
    phase_cap: float,
    config_sha256: str,
    v1_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    native = arrays["native"]
    contract = arrays["contract"]
    reverse = arrays["reverse"]
    distance = arrays["distance"]
    probe_distances = sorted({0, 1, 16, 64, 256, 512, 1024, distance})
    norm_errors = {}
    for current in probe_distances:
        left = _rotation_distance(contract - native, current)
        right = _rotation_distance(reverse - native, current)
        norm_errors[str(current)] = float(np.max(np.abs(left - right)))
    active = arrays["active_indices"]
    maximum_displacement = float(arrays["displacement_precast"].max(initial=0.0))
    result = {
        "status": AUDIT_STATUS,
        "method_id": METHOD_ID,
        "model_id": model_id,
        "model_config_sha256": config_sha256,
        "public_inputs_only": True,
        "model_execution": False,
        "eta": float(eta),
        "phase_cap": float(phase_cap),
        "native_length": int(distance + 1),
        "pair_count": int(native.size),
        "active_pair_indices_zero_based": active.astype(int).tolist(),
        "active_pair_count": int(active.size),
        "lower_endpoint_phase": float(arrays["lower_phase"]),
        "effective_upper_phase": float(arrays["upper_phase"]),
        "maximum_phase_displacement_precast": maximum_displacement,
        "maximum_phase_displacement_bound": float(arrays["exact_phase_bound"]),
        "phase_displacement_at_distance_256_bound": float(maximum_displacement * 256 / distance),
        "maximum_relative_frequency_change_contract": float(
            np.max((native - contract) / native)
        ),
        "maximum_relative_frequency_change_reverse": float(
            np.max((reverse - native) / native)
        ),
        "float32_equal_phase_dose_max_abs_error": arrays["fp32_phase_dose_max_abs_error"],
        "rotation_operator_distance_equality_errors": norm_errors,
        "checks": arrays["checks"],
        "table_sha256_float32": {
            "native": tensor_sha256(native),
            "contract": tensor_sha256(contract),
            "reverse": tensor_sha256(reverse),
        },
        "claim_boundary": (
            "CPU construction audit only. It proves endpoint, ordering, direction and "
            "equal additive-phase-dose properties; it does not predict checkpoint or task gains."
        ),
    }
    if v1_payload is not None:
        result["historical_v1_comparison"] = compare_v1(v1_payload, arrays)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--phase-cap", type=float, default=DEFAULT_PHASE_CAP)
    parser.add_argument("--v1-table", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    from experiments.fixed_rope_three_interfaces_20260913.tables import (
        model_geometry,
        runtime_native_inv_freq,
    )

    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    arrays = build_halfturn_arrays(
        native,
        native_length=int(geometry["native_length"]),
        eta=args.eta,
        phase_cap=args.phase_cap,
    )
    config_hash = file_sha256(args.config)
    v1_payload = json.loads(args.v1_table.read_text()) if args.v1_table else None
    audit = build_audit(
        arrays,
        model_id=args.model_id,
        eta=args.eta,
        phase_cap=args.phase_cap,
        config_sha256=config_hash,
        v1_payload=v1_payload,
    )
    roles = {
        "native": "native",
        "contract": "candidate",
        "reverse": "control",
    }
    for arm, role in roles.items():
        receipt = _table_receipt(
            values=arrays[arm],
            native=arrays["native"],
            model_id=args.model_id,
            arm=arm,
            role=role,
            audit=arrays,
            eta=args.eta,
            phase_cap=args.phase_cap,
            config_sha256=config_hash,
        )
        _write_json(args.out / f"{arm}.json", receipt)
    _write_json(args.out / "cpu_audit.json", audit)
    print(json.dumps({
        "status": AUDIT_STATUS,
        "active_pairs": audit["active_pair_count"],
        "tables": audit["table_sha256_float32"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

