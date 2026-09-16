#!/usr/bin/env python3
"""CPU oracle and a smooth, equal-log-dose control for the frozen NCP candidate.

This module never imports torch, opens a checkpoint, or executes a model. It
reuses the frozen NCP Fourier risk; the independent oracle integrates the
original binary retrieval loss and its derivatives directly. The new control
is an attribution experiment, not a proposed stronger enhancement method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import LinearConstraint, minimize, nnls
from scipy.special import expit

from experiments.native_contrastive_proximal_20260915.tables import (
    MAX_LOG_SHIFT,
    MIN_LOG_GAP_FRACTION,
    reference_fourier,
    reference_log_gradient,
    reference_risk,
    tensor_sha256,
)


def direct_reference(phase: float) -> np.ndarray:
    """Return risk, log-gradient and log-curvature without a Fourier expansion.

    The periodic distractor-angle integral uses 256 points. Adaptive distance
    integration differentiates the original softplus loss, independently of
    both the Fourier kernels and the integration-by-parts bound derivation.
    """
    if not math.isfinite(phase) or phase < 0:
        raise ValueError("phase must be finite and nonnegative")
    distractors = np.cos(2 * np.pi * np.arange(256) / 256)

    def integrand(t: float) -> np.ndarray:
        theta = phase * t
        margin = distractors - math.cos(theta)
        probabilities = expit(margin)
        first = math.sin(theta) * probabilities.mean()
        second = (
            math.cos(theta) * probabilities.mean()
            + math.sin(theta) ** 2 * np.mean(probabilities * (1 - probabilities))
        )
        return 2 * (1 - t) * np.asarray([
            np.logaddexp(0, margin).mean(),
            theta * first,
            theta * first + theta**2 * second,
        ])

    value, error = quad_vec(integrand, 0, 1, epsabs=2e-11, epsrel=2e-11)
    if error > 1e-9:
        raise RuntimeError(f"direct reference integral did not converge: {error}")
    return value


def reference_log_curvature(phase: Any, coefficients: np.ndarray) -> np.ndarray:
    """Second log derivative of the frozen Fourier reference risk."""
    values = np.asarray(phase, dtype=np.float64)[..., None] * np.arange(
        1, len(coefficients) + 1,
    )
    terms = (
        2 * np.cos(values) - 6 * np.sinc(values / np.pi)
        + 4 * np.sinc(values / (2 * np.pi)) ** 2
    )
    small = np.abs(values) < 1e-3
    series = -values**2 / 3 + 2 * values**4 / 45 - values**6 / 560
    return np.sum(coefficients * np.where(small, series, terms), axis=-1)


def audit_reference() -> dict[str, Any]:
    """Check the fixed public risk implementation against direct integration."""
    phases = np.asarray([0.0, 1e-4, 0.1, 1.0, math.pi, 4.7, 10.0, 30.0])
    a0, coefficients = reference_fourier()
    direct = np.asarray([direct_reference(float(value)) for value in phases])
    fourier = np.column_stack([
        reference_risk(phases, a0=a0, coefficients=coefficients),
        reference_log_gradient(phases, coefficients=coefficients),
        reference_log_curvature(phases, coefficients),
    ])
    errors = np.max(np.abs(direct - fourier), axis=0)
    if np.max(errors) > 1e-9:
        raise RuntimeError(f"Fourier reference disagrees with the direct oracle: {errors}")
    return {
        "status": "NCP_INDEPENDENT_CPU_ORACLE_V1",
        "model_execution": False,
        "probe_phases": phases.tolist(),
        "direct_risk_log_gradient_log_curvature": direct.tolist(),
        "maximum_absolute_error": dict(zip(
            ("risk", "log_gradient", "log_curvature"), errors.tolist(),
        )),
        "claim_boundary": (
            "Numerical checks of a binary reference retrieval model; these are "
            "not Transformer NLL or task results and do not prove global bounds."
        ),
    }


def _table(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if (
        array.ndim != 1 or array.size < 4 or not np.isfinite(array).all()
        or np.any(array <= 0) or np.any(array[:-1] <= array[1:])
    ):
        raise ValueError("frequencies must be positive and strictly decreasing")
    return array


def build_dose_control(
    native_values: Any, source_values: Any, *, native_length: int,
    model_id: str = "unspecified",
) -> dict[str, Any]:
    """Match source sum(log(native/source)) with the smoothest feasible shape.

    For native log gaps g, minimize sum((u[k+1]-u[k])**2 / g[k]).
    This is Dirichlet energy on the native log-frequency coordinate, with no
    adjustable shape coefficient. Endpoints are zero, 0 <= u <= 2/9, and at
    least half of every native log gap is retained. The fixed total log dose
    is taken from the actual source FP32 table, not from its pre-cast optimizer.

    The strictly convex quadratic has one solution. When its equality-only
    solution is feasible no optimizer is needed. Otherwise a constrained
    solve is accepted only with feasible constraints and a small KKT residual.
    """
    native, source = _table(native_values), _table(source_values)
    if not isinstance(native_length, (int, np.integer)) or native_length < 2:
        raise ValueError("native_length must be an integer of at least two")
    if native.shape != source.shape or not np.array_equal(native[[0, -1]], source[[0, -1]]):
        raise ValueError("source must share native shape and bit-exact endpoints")
    source_u = np.log(native.astype(float) / source.astype(float))
    if np.any(source_u < 0) or np.max(source_u) > MAX_LOG_SHIFT + 2e-7:
        raise ValueError("source must obey the NCP slowdown bounds")
    gaps = np.diff(-np.log(native.astype(float)))
    if np.any(gaps + np.diff(source_u) < MIN_LOG_GAP_FRACTION * gaps - 2e-7):
        raise ValueError("source violates the half-native-gap constraint")
    budget = float(source_u.sum())
    size = native.size - 2
    embedding = np.zeros((native.size, size))
    embedding[1:-1] = np.eye(size)
    difference = np.diff(embedding, axis=0)
    gap_allowance = (1 - MIN_LOG_GAP_FRACTION) * gaps
    hessian = np.diag(2 / gaps[:-1] + 2 / gaps[1:])
    hessian += np.diag(-2 / gaps[1:-1], k=1) + np.diag(-2 / gaps[1:-1], k=-1)

    # Summing the gap inequalities back from the fixed slow endpoint gives
    # this componentwise upper envelope. It is itself feasible, so its sum is
    # the exact maximum dose; no separate LP or multi-threaded solver is needed.
    envelope = np.minimum(MAX_LOG_SHIFT, np.cumsum(gap_allowance[::-1])[::-1][1:])
    maximum_dose = float(envelope.sum())
    if budget > maximum_dose + 1e-10:
        raise ValueError("source log-dose exceeds the control feasible capacity")

    def differences(u: np.ndarray) -> np.ndarray:
        return np.diff(np.concatenate(([0.0], u, [0.0])))

    def gradient_at(u: np.ndarray) -> np.ndarray:
        slopes = differences(u) / gaps
        return 2 * (slopes[:-1] - slopes[1:])

    if budget == 0:
        shifts = np.zeros(size)
    else:
        direction = np.linalg.solve(hessian, np.ones(size))
        shifts = budget * direction / direction.sum()
    solver = "equality_closed_form"
    if (
        np.min(differences(shifts) + gap_allowance) < -1e-12
        or np.max(shifts) > MAX_LOG_SHIFT
    ):
        solver = "constrained_quadratic_slsqp"
        start = envelope * (budget / maximum_dose)
        result = minimize(
            lambda u: float(np.sum(differences(u)**2 / gaps)), start,
            jac=gradient_at, method="SLSQP",
            bounds=[(0, MAX_LOG_SHIFT)] * size,
            constraints=[
                LinearConstraint(np.ones((1, size)), budget, budget),
                LinearConstraint(difference, -gap_allowance, np.inf),
            ],
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"control quadratic solve failed: {result.message}")
        shifts = result.x

    slack = np.concatenate((shifts, MAX_LOG_SHIFT - shifts, differences(shifts) + gap_allowance))
    if np.min(slack) < -1e-9 or abs(shifts.sum() - budget) > 1e-9:
        raise RuntimeError("control quadratic solution violates its constraints")
    normals = np.concatenate((-np.eye(size), np.eye(size), -difference), axis=0)
    active_normals = normals[slack < 1e-8].T
    # Equality multiplier is free; remove it by centering the stationarity
    # equation before fitting nonnegative active-inequality multipliers.
    gradient = gradient_at(shifts)
    centered_gradient = gradient - gradient.mean()
    centered_normals = active_normals - active_normals.mean(axis=0)
    if centered_normals.shape[1] and np.max(np.abs(centered_gradient)) > 1e-12:
        multipliers, _ = nnls(centered_normals, -centered_gradient, maxiter=10000)
        stationarity = centered_gradient + np.einsum("ij,j->i", centered_normals, multipliers)
    else:
        stationarity = centered_gradient
    kkt_residual = float(np.max(np.abs(stationarity)))
    if kkt_residual > 2e-6:
        raise RuntimeError(f"control failed convex-optimality KKT check: {kkt_residual}")

    full_u = np.concatenate(([0.0], shifts, [0.0]))
    candidate = (native.astype(float) * np.exp(-full_u)).astype(np.float32)
    candidate[[0, -1]] = native[[0, -1]]
    _table(candidate)
    actual_u = np.log(native.astype(float) / candidate.astype(float))
    actual_gap_slack = gaps + np.diff(actual_u) - MIN_LOG_GAP_FRACTION * gaps
    # FP32 conversion can cross an active mathematical gap by a rounding ULP.
    # Keep both the exact pre-cast certificate and the measured installed error.
    if np.min(actual_gap_slack) < -2e-7 or np.max(actual_u) > MAX_LOG_SHIFT + 2e-7:
        raise RuntimeError("FP32 control exceeds the documented rounding tolerance")
    dose_error = float(actual_u.sum() - budget)
    if abs(dose_error) > native.size * 1.3e-7:
        raise RuntimeError("FP32 control log-dose mismatch exceeds rounding tolerance")
    distance = native_length - 1
    return {
        "status": "FROZEN_NATIVE_NCP_DOSE_CONTROL_V1",
        "candidate_id": f"{model_id}_ncp_dose_control",
        "model_id": model_id,
        "role": "control",
        "values_float32": candidate.tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(candidate),
        "native_table_sha256_float32": tensor_sha256(native),
        "construction": {
            "method_id": "native_ncp_equal_log_dose_minimum_dirichlet_v1",
            "native_length": native_length,
            "source_table_sha256_float32": tensor_sha256(source),
            "source_log_dose": budget,
            "control_log_dose_precast": float(full_u.sum()),
            "control_log_dose_float32": float(actual_u.sum()),
            "float32_log_dose_error": dose_error,
            "maximum_feasible_log_dose": maximum_dose,
            "maximum_log_shift_bound": MAX_LOG_SHIFT,
            "minimum_log_gap_fraction": MIN_LOG_GAP_FRACTION,
            "minimum_log_gap_slack_precast": float(np.min(differences(shifts) + gap_allowance)),
            "minimum_log_gap_slack_float32": float(np.min(actual_gap_slack)),
            "float32_absolute_constraint_tolerance": 2e-7,
            "maximum_source_phase_change": float(np.max(distance * np.abs(native.astype(float) - source))),
            "maximum_control_phase_change": float(np.max(distance * np.abs(native.astype(float) - candidate))),
            "objective": "sum((u[k+1]-u[k])**2/native_log_gap[k])",
            "solver": solver,
            "maximum_kkt_stationarity_residual": kkt_residual,
            "model_weights_or_outputs_used": False,
            "changed_variable": "interior_native_rope_frequency_allocation",
            "fastest_and_slowest_frequency_fixed": True,
            "frequency_slot_assignment_fixed": True,
            "gain_rule": "Native gain 1",
        },
        "claim_boundary": (
            "Attribution control matching total native-relative log-frequency slowdown, "
            "within reported FP32 rounding error. It does not match phase displacement, "
            "per-channel compatibility, or task performance. CPU preparation only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-table", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--native-length", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def read(path: Path) -> np.ndarray:
        payload = json.loads(path.read_text())
        table = payload.get("table", payload)
        if float(table.get("gain", 1.0)) != 1.0:
            raise ValueError(f"table must use gain=1: {path}")
        return _table(table["values_float32"])

    audit = audit_reference()
    control = build_dose_control(
        read(args.native_table), read(args.source_table),
        native_length=args.native_length, model_id=args.model_id,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in (("ncp_cpu_oracle.json", audit), ("ncp_dose_control.json", control)):
        (args.out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": audit["status"], "model_execution": False,
                      "control_sha256": control["table_sha256_float32"]}, sort_keys=True))


if __name__ == "__main__":
    main()
