#!/usr/bin/env python3
"""General public-input NCP preparation, including active gap constraints.

The input native table is authoritative. No model weights, activations, task
outputs, torch, or CUDA are used. Config-derived tables are only CPU design
geometries until checked against actual runtime native tables. Native length
must always be supplied explicitly; max_position_embeddings is not a claim
about the checkpoint's training window.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, brentq, minimize, nnls

from experiments.native_contrastive_proximal_20260915.tables import (
    MAX_LOG_SHIFT,
    MIN_LOG_GAP_FRACTION,
    PROXIMAL_CURVATURE,
    reference_fourier,
    reference_log_gradient,
    reference_risk,
    tensor_sha256,
)


def _native_table(values: Any) -> np.ndarray:
    native = np.asarray(values, dtype=np.float32)
    if (
        native.ndim != 1 or native.size < 2 or not np.isfinite(native).all()
        or np.any(native <= 0) or np.any(native[:-1] <= native[1:])
    ):
        raise ValueError("native table must contain at least two positive, decreasing FP32 values")
    return native


def _length(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError("native_length must explicitly specify an integer of at least two")
    return int(value)


def config_native(config: dict[str, Any], *, native_length: int) -> dict[str, Any]:
    """Read a narrowly supported vanilla full-RoPE configuration on the CPU.

    This adapter accepts only llama, olmo2 and qwen2 default-RoPE schemas. A
    scaled, partial, unknown, or nested/mixed configuration requires an explicit
    actual native table instead. The output uses FP64 exponentiation followed
    by FP32 casting and is NOT asserted to match a framework's FP32 pow path.
    """
    length = _length(native_length)
    if config.get("model_type") not in {"llama", "olmo2", "qwen2"}:
        raise ValueError("unknown model schema: provide the explicit runtime native table")
    if config.get("rope_type", "default") != "default":
        raise ValueError("scaled RoPE: provide the explicit runtime native table")
    for key in ("rope_scaling", "rope_parameters"):
        parameters = config.get(key)
        if parameters is not None:
            if not isinstance(parameters, dict) or set(parameters) - {"rope_type", "type", "rope_theta"}:
                raise ValueError("scaled or ambiguous RoPE: provide the explicit runtime native table")
            if any(parameters.get(name, "default") != "default" for name in ("rope_type", "type")):
                raise ValueError("scaled RoPE: provide the explicit runtime native table")
    if config.get("partial_rotary_factor", 1.0) != 1.0 or config.get("rotary_pct", 1.0) != 1.0:
        raise ValueError("partial RoPE: provide the explicit runtime native table")
    hidden, heads = config.get("hidden_size"), config.get("num_attention_heads")
    explicit_dim = config.get("head_dim")
    if explicit_dim is None:
        if (
            isinstance(hidden, bool) or isinstance(heads, bool)
            or not isinstance(hidden, int) or not isinstance(heads, int)
            or hidden <= 0 or heads <= 0 or hidden % heads
        ):
            raise ValueError("ambiguous head geometry: provide the explicit runtime native table")
        dimension = hidden // heads
    else:
        dimension = explicit_dim
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4 or dimension % 2:
        raise ValueError("rotary head dimension must be a positive even integer of at least four")
    if config.get("rotary_dim", dimension) != dimension:
        raise ValueError("partial RoPE: provide the explicit runtime native table")
    bases = [config.get("rope_theta")]
    bases += [config[key].get("rope_theta") for key in ("rope_parameters", "rope_scaling") if isinstance(config.get(key), dict)]
    bases = [float(base) for base in bases if base is not None]
    if not bases or any(not math.isfinite(base) or base <= 1 for base in bases) or len(set(bases)) != 1:
        raise ValueError("explicit unambiguous rope_theta is required")
    native = _native_table(np.exp(-np.arange(dimension // 2, dtype=float) * math.log(bases[0]) / (dimension // 2)))
    return {
        "native_values": native.tolist(),
        "native_length": length,
        "input_label": "config_derived_fp32_not_runtime_verified",
        "model_type": config["model_type"],
        "native_length_source": "explicit_argument_not_inferred_from_config",
        "config_max_position_embeddings_informational_only": config.get("max_position_embeddings"),
        "pair_count": native.size,
        "gain": 1.0,
    }


def build_public_ncp(
    native_values: Any, *, native_length: int, model_id: str,
    input_label: str = "explicit_runtime_native_table",
) -> dict[str, Any]:
    """Solve the same fixed NCP objective for arbitrary input table dimensions.

    Independent scalar roots are used when all gap constraints are satisfied.
    Otherwise the full convex problem is solved with fixed endpoints and its
    KKT conditions checked. This never rewrites the frozen OLMo experiment.
    Unsafe FP32 quantization (including collapsed adjacent frequencies) is
    rejected; small measured active-boundary rounding errors are reported.
    """
    native = _native_table(native_values)
    length = _length(native_length)
    phase = native.astype(float) * (length - 1)
    if not np.isfinite(phase).all():
        raise ValueError("native_length and native table produce nonfinite phases")
    gaps = np.diff(-np.log(native.astype(float)))
    size = native.size - 2
    a0, coefficients = reference_fourier()
    native_risk = reference_risk(phase, a0=a0, coefficients=coefficients)

    def q(values: Any) -> np.ndarray:
        return reference_log_gradient(values, coefficients=coefficients)

    def full(u: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], u, [0.0]))

    def gradient(u: np.ndarray) -> np.ndarray:
        return PROXIMAL_CURVATURE * u - q(phase[1:-1] * np.exp(-u))

    def objective(u: np.ndarray) -> float:
        risk = reference_risk(phase[1:-1] * np.exp(-u), a0=a0, coefficients=coefficients)
        return float(np.sum(risk - native_risk[1:-1] + 0.5 * PROXIMAL_CURVATURE * u**2))

    shifts = np.zeros(size)
    for index, current_phase in enumerate(phase[1:-1]):
        if float(q(current_phase)) <= 0:
            continue

        def derivative(u: float) -> float:
            return PROXIMAL_CURVATURE * u - float(q(current_phase * math.exp(-u)))

        if derivative(MAX_LOG_SHIFT) < -1e-12:
            raise RuntimeError("reference derivative violates the derived NCP displacement bound")
        shifts[index] = brentq(derivative, 0, MAX_LOG_SHIFT, xtol=1e-14, rtol=2e-15)

    allowance = (1 - MIN_LOG_GAP_FRACTION) * gaps
    independent_slack = allowance + np.diff(full(shifts))
    coupled = bool(np.min(independent_slack) < -1e-12)
    embedding = np.zeros((native.size, size))
    embedding[1:-1] = np.eye(size)
    difference = np.diff(embedding, axis=0)
    if coupled:
        result = minimize(
            objective, np.zeros(size), jac=gradient, method="SLSQP",
            bounds=[(0, MAX_LOG_SHIFT)] * size,
            constraints=[LinearConstraint(difference, -allowance, np.inf)],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"coupled NCP solve failed: {result.message}")
        shifts = result.x
    gap_slack = allowance + np.diff(full(shifts))
    slack = np.concatenate((shifts, MAX_LOG_SHIFT - shifts, gap_slack))
    if not np.isfinite(slack).all() or np.min(slack) < -1e-9:
        raise RuntimeError("NCP solution violates the feasible set")

    stationarity = gradient(shifts)
    complementarity = 0.0
    if size and not coupled:
        # Bound multipliers are known directly for independent scalar optima.
        # Avoid a redundant dense NNLS solve, especially for larger tables.
        stationarity[(shifts == 0) & (stationarity >= 0)] = 0
        stationarity[(shifts == MAX_LOG_SHIFT) & (stationarity <= 0)] = 0
    elif size:
        normals = np.concatenate((-np.eye(size), np.eye(size), -difference), axis=0)
        active = slack < 2e-8
        if np.any(active) and np.max(np.abs(stationarity)) > 1e-12:
            multipliers, _ = nnls(normals[active].T, -stationarity, maxiter=10000)
            stationarity = stationarity + np.einsum("ji,j->i", normals[active], multipliers)
            complementarity = float(np.max(np.abs(multipliers * slack[active])))
    residual = float(np.max(np.abs(stationarity), initial=0))
    if residual > 2e-7 or complementarity > 2e-8:
        raise RuntimeError(f"NCP solution failed KKT optimality verification: {residual}, {complementarity}")

    full_shifts = full(shifts)
    candidate = (native.astype(float) * np.exp(-full_shifts)).astype(np.float32)
    candidate[[0, -1]] = native[[0, -1]]
    _native_table(candidate)
    actual_shifts = np.log(native.astype(float) / candidate.astype(float))
    actual_gap_slack = allowance + np.diff(actual_shifts)
    if (
        np.any(candidate > native)
        or np.min(actual_gap_slack) < -2e-7
        or np.max(actual_shifts) > MAX_LOG_SHIFT + 2e-7
    ):
        raise RuntimeError("NCP FP32 quantization violates the constraint rounding tolerance")
    candidate_risk = reference_risk(phase * np.exp(-full_shifts), a0=a0, coefficients=coefficients)
    actual_risk = reference_risk(candidate.astype(float) * (length - 1), a0=a0, coefficients=coefficients)
    inequality_margin = float(np.mean(native_risk - candidate_risk - 0.5 * PROXIMAL_CURVATURE * full_shifts**2))
    if inequality_margin < -2e-13:
        raise RuntimeError("NCP does not satisfy its reference-risk descent inequality")
    audit = {
        "model_execution": False,
        "native_length": length,
        "native_length_source": "explicit_argument",
        "pair_count": native.size,
        "input_label": input_label,
        "coupled_gap_solver_used": coupled,
        "independent_minimum_gap_slack": float(np.min(independent_slack)),
        "minimum_gap_slack_precast": float(np.min(gap_slack)),
        "minimum_gap_slack_float32": float(np.min(actual_gap_slack)),
        "minimum_gap_ratio_float32": float(np.min((gaps + np.diff(actual_shifts)) / gaps)),
        "float32_absolute_constraint_tolerance": 2e-7,
        "maximum_kkt_stationarity_residual": residual,
        "maximum_kkt_complementarity_residual": complementarity,
        "reference_risk_native_mean": float(np.mean(native_risk)),
        "reference_risk_candidate_precast_mean": float(np.mean(candidate_risk)),
        "reference_risk_candidate_float32_mean": float(np.mean(actual_risk)),
        "reference_descent_inequality_margin_precast": inequality_margin,
        "changed_pair_count": int(np.count_nonzero(native != candidate)),
        "maximum_log_shift_precast": float(np.max(full_shifts)),
        "maximum_log_shift_float32": float(np.max(actual_shifts)),
        "endpoints_bit_exact": bool(np.array_equal(native[[0, -1]], candidate[[0, -1]])),
        "positive_strictly_decreasing_float32": True,
        "one_sided_slowdown_float32": True,
        "claim_boundary": (
            "Public-input CPU construction and binary reference-risk checks only. "
            "No model or task improvement is established. Config-derived geometries "
            "are not asserted to equal runtime FP32 native tables."
        ),
    }
    return {
        "status": "PUBLIC_NATIVE_NCP_GENERAL_CPU_V1",
        "candidate_id": f"{model_id}_native_ncp_public_v1",
        "model_id": model_id,
        "role": "candidate",
        "values_float32": candidate.tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(candidate),
        "native_table_sha256_float32": tensor_sha256(native),
        "construction": {
            "method_id": "native_contrastive_proximal_fixed_public_v1",
            "native_length": length,
            "proximal_curvature": PROXIMAL_CURVATURE,
            "maximum_log_shift_bound": MAX_LOG_SHIFT,
            "minimum_log_gap_fraction": MIN_LOG_GAP_FRACTION,
            "log_shifts_precast": full_shifts.tolist(),
            "model_weights_or_outputs_used": False,
            "gain_rule": "Native gain 1",
            "fastest_and_slowest_frequency_fixed": True,
            "frequency_slot_assignment_fixed": True,
        },
        "cpu_audit": audit,
    }


def build_matrix(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct once per declared geometry; this is not a parameter sweep."""
    ids = [case["model_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("matrix case model_id values must be unique")
    tables = [build_public_ncp(
        case["native_values"], native_length=case["native_length"],
        model_id=case["model_id"], input_label=case.get("input_label", "explicit_native_table"),
    ) for case in cases]
    return {
        "status": "PUBLIC_NATIVE_NCP_GEOMETRY_MATRIX_CPU_V1",
        "model_execution": False,
        "parameters_selected_using_model_or_task_outputs": False,
        "rows": [{"model_id": table["model_id"], "table_sha256_float32": table["table_sha256_float32"], **table["cpu_audit"]} for table in tables],
        "tables": tables,
        "claim_boundary": "Geometry construction coverage, not cross-model performance evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of explicit native_values/native_length/model_id cases")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matrix = build_matrix(json.loads(args.cases.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "geometry_matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    for index, table in enumerate(matrix["tables"]):
        (args.out / f"case_{index:02d}.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": matrix["status"], "cases": len(matrix["rows"]), "model_execution": False}))


if __name__ == "__main__":
    main()
