"""Construct the three endpoint-matched candidate frequency tables."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .protocol import (
    CANDIDATES,
    EVQ_TAU,
    MORPH_GRID,
    PHASE_BINS,
    PHASE_LAMBDA,
    ROPE_BASE,
    ROTARY_PAIRS,
    ContractError,
    protocol_manifest,
    sha256_file,
    validate_table,
)


def array_sha256(values: Sequence[float], *, dtype: str = "<f4") -> str:
    array = np.asarray(values, dtype=dtype).copy(order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(grid)))


def _strict_quantiles(grid: np.ndarray, density: np.ndarray, count: int) -> np.ndarray:
    area = _trapz(density, grid)
    if not math.isfinite(area) or area <= 0.0:
        raise ContractError("candidate density has no finite positive integral")
    rho = density / area
    cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (rho[:-1] + rho[1:]) * np.diff(grid)))
    )
    cdf /= cdf[-1]
    probabilities = np.linspace(0.0, 1.0, count, dtype=np.float64)
    phi = np.interp(probabilities, cdf, grid)
    phi[0], phi[-1] = 0.0, 1.0
    if not np.all(np.diff(phi) > 0.0):
        raise ContractError("inverse-CDF candidate is not strictly increasing")
    return phi


def phase_chord_phi(collection: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(collection, allow_pickle=False) as payload:
        inv_freq = np.asarray(payload["inv_freq"], dtype=np.float64)
        mass = np.asarray(payload["mass"], dtype=np.float64).sum(axis=(0, 1))
        metadata = json.loads(str(payload["metadata"].item()))
    validate_table(inv_freq, label="R0 Native inv_freq")
    if mass.ndim != 1 or mass.size < 2 or not np.isfinite(mass).all():
        raise ContractError("R0 distance mass is malformed")
    mass = mass.copy()
    mass[0] = 0.0
    if mass.sum() <= 0.0:
        raise ContractError("R0 has no non-self attention mass")
    distance_probability = mass / mass.sum()
    grid = np.linspace(0.0, 1.0, PHASE_BINS, dtype=np.float64)
    log_span = float(np.log(inv_freq[0] / inv_freq[-1]))
    probe_frequency = inv_freq[0] * np.exp(-log_span * grid)
    distance = np.arange(mass.size, dtype=np.float64)
    kernel = 1.0 - np.cos(distance[:, None] * probe_frequency[None, :])
    demand = np.sum(distance_probability[:, None] * kernel, axis=0)
    demand = np.maximum(demand, np.finfo(np.float64).tiny)
    demand /= _trapz(demand, grid)
    mixed = (1.0 - PHASE_LAMBDA) * demand + PHASE_LAMBDA
    density = np.cbrt(mixed)
    return _strict_quantiles(grid, density, ROTARY_PAIRS), {
        "kernel": "1-cos(omega(phi)*Delta)",
        "self_distance_excluded": True,
        "distance_mass": "all R0 layers and heads",
        "density": "((1-lambda)*normalised_phase_demand + lambda)^(1/3)",
        "bins": PHASE_BINS,
        "lambda": PHASE_LAMBDA,
        "source_metadata": {
            key: metadata.get(key)
            for key in (
                "backend",
                "base",
                "head_dim",
                "heads",
                "layers",
                "length",
                "model_revision",
                "tokens_sha256",
                "training_or_parameter_updates",
            )
        },
    }


def anchored_evq_phi(count: int = ROTARY_PAIRS, tau: float = EVQ_TAU) -> np.ndarray:
    probability = np.linspace(0.0, 1.0, count, dtype=np.float64)
    if abs(float(tau)) < 1e-12:
        return probability
    phi = 1.0 - np.arcsinh((1.0 - probability) * math.sinh(float(tau))) / float(tau)
    phi[0], phi[-1] = 0.0, 1.0
    if not np.all(np.diff(phi) > 0.0):
        raise ContractError("anchored EVQ-Cosh phi is not strictly increasing")
    return phi


def exponential_phi(count: int, parameter: float) -> np.ndarray:
    uniform = np.linspace(0.0, 1.0, count, dtype=np.float64)
    value = float(parameter)
    if abs(value) < 1e-10:
        return uniform
    phi = np.expm1(value * uniform) / math.expm1(value)
    phi[0], phi[-1] = 0.0, 1.0
    if not np.all(np.diff(phi) > 0.0):
        raise ContractError("exponential control phi is not strictly increasing")
    return phi


def matched_exponential_control(
    phase_phi: np.ndarray,
    native: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    uniform = np.linspace(0.0, 1.0, phase_phi.size, dtype=np.float64)
    native_table = None if native is None else np.asarray(native, dtype=np.float64)

    def displacement(phi: np.ndarray) -> np.ndarray:
        if native_table is None:
            return phi - uniform
        table = inv_freq_from_phi(phi, native_table)
        return np.log(table) - np.log(native_table)

    phase_displacement = displacement(phase_phi)
    target_rms = float(np.sqrt(np.mean(np.square(phase_displacement))))
    if target_rms == 0.0:
        return uniform, 0.0
    phase_mean = float(np.mean(phase_displacement))
    if native_table is None:
        sign = 1.0 if phase_mean >= 0.0 else -1.0
    else:
        # The exponential parameter and log-frequency displacement have the
        # same mean sign because phi-to-log-frequency reverses direction.
        sign = -1.0 if phase_mean >= 0.0 else 1.0

    def rms(magnitude: float) -> float:
        phi = exponential_phi(phase_phi.size, sign * magnitude)
        return float(np.sqrt(np.mean(np.square(displacement(phi)))))

    lower, upper = 0.0, 1.0
    while rms(upper) < target_rms and upper < 128.0:
        upper *= 2.0
    if rms(upper) < target_rms:
        raise ContractError("phase displacement exceeds exponential control family")
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if rms(middle) < target_rms:
            lower = middle
        else:
            upper = middle
    parameter = sign * 0.5 * (lower + upper)
    control = exponential_phi(phase_phi.size, parameter)
    control_displacement = displacement(control)
    observed = float(np.sqrt(np.mean(np.square(control_displacement))))
    if not math.isclose(observed, target_rms, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("control RMS displacement matching failed")
    if phase_mean != 0.0 and np.mean(control_displacement) * phase_mean >= 0.0:
        raise ContractError("control must bend opposite to the phase candidate")
    return control, parameter


def inv_freq_from_phi(phi: np.ndarray, native: np.ndarray) -> np.ndarray:
    log_high = math.log(float(native[0]))
    log_low = math.log(float(native[-1]))
    inv_freq = np.exp(log_high + phi * (log_low - log_high))
    inv_freq[0], inv_freq[-1] = native[0], native[-1]
    validate_table(inv_freq, label="candidate inv_freq")
    return inv_freq


def log_morph(native: Sequence[float], target: Sequence[float], t: float) -> np.ndarray:
    native_table = np.asarray(validate_table(native, label="native"), dtype=np.float64)
    target_table = np.asarray(validate_table(target, label="target"), dtype=np.float64)
    amount = float(t)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ContractError("morph amount t must lie in [0,1]")
    if amount == 0.0:
        return native_table.copy()
    if amount == 1.0:
        return target_table.copy()
    table = np.exp((1.0 - amount) * np.log(native_table) + amount * np.log(target_table))
    table[0], table[-1] = native_table[0], native_table[-1]
    validate_table(table, label=f"morph t={amount:g}")
    return table


def _table_receipt(name: str, phi: np.ndarray, inv_freq: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    log_displacement = np.log(inv_freq) - np.log(native)
    return {
        "name": name,
        "phi": phi.tolist(),
        "inv_freq": inv_freq.tolist(),
        "inv_freq_float32_sha256": array_sha256(inv_freq),
        "endpoint_identity": {
            "fast": bool(inv_freq[0] == native[0]),
            "slow": bool(inv_freq[-1] == native[-1]),
        },
        "log_displacement": {
            "rms": float(np.sqrt(np.mean(np.square(log_displacement)))),
            "mean": float(np.mean(log_displacement)),
            "l1": float(np.sum(np.abs(log_displacement))),
            "max_abs": float(np.max(np.abs(log_displacement))),
        },
        "minimum_adjacent_log_gap": float(np.min(-np.diff(np.log(inv_freq)))),
    }


def build_target_manifest(collection: Path) -> dict[str, Any]:
    collection = collection.resolve()
    with np.load(collection, allow_pickle=False) as payload:
        native = np.asarray(payload["inv_freq"], dtype=np.float64)
    validate_table(native, label="R0 Native inv_freq")
    phase_phi, phase_metadata = phase_chord_phi(collection)
    control_phi, control_parameter = matched_exponential_control(phase_phi, native)
    evq_phi = anchored_evq_phi()
    phis = {
        CANDIDATES[0]: phase_phi,
        CANDIDATES[1]: control_phi,
        CANDIDATES[2]: evq_phi,
    }
    tables = {
        name: inv_freq_from_phi(phi, native) for name, phi in phis.items()
    }
    candidates = {
        name: _table_receipt(name, phis[name], tables[name], native)
        for name in CANDIDATES
    }
    candidates[CANDIDATES[0]]["construction"] = phase_metadata
    candidates[CANDIDATES[1]]["construction"] = {
        "family": "endpoint-inclusive exponential coordinate warp",
        "parameter": control_parameter,
        "matching": "RMS log-frequency displacement from Native",
        "direction": "opposite mean displacement to phase-chord candidate",
        "attention_profile_shape_used": False,
        "phase_candidate_scalar_rms_used": True,
    }
    candidates[CANDIDATES[2]]["construction"] = {
        "family": "endpoint-anchored EVQ-Cosh",
        "tau": EVQ_TAU,
        "tau_rule": "head_dim/sqrt(native_context)",
    }
    phase_rms = candidates[CANDIDATES[0]]["log_displacement"]["rms"]
    control_rms = candidates[CANDIDATES[1]]["log_displacement"]["rms"]
    if not math.isclose(float(phase_rms), float(control_rms), rel_tol=0.0, abs_tol=1e-11):
        raise ContractError("phase/control log-displacement RMS mismatch")
    morph_tables: dict[str, list[dict[str, Any]]] = {}
    for name in CANDIDATES:
        rows = []
        for t in MORPH_GRID:
            table = log_morph(native, tables[name], t)
            rows.append(
                {
                    "t": t,
                    "inv_freq_float32_sha256": array_sha256(table),
                    "minimum_adjacent_log_gap": float(
                        np.min(-np.diff(np.log(table)))
                    ),
                }
            )
        morph_tables[name] = rows
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "FINITE_TARGETS_FROZEN",
        "protocol": protocol_manifest(),
        "source": {
            "r0_collection": str(collection),
            "r0_collection_sha256": sha256_file(collection),
        },
        "native": _table_receipt(
            "Native", np.linspace(0.0, 1.0, ROTARY_PAIRS), native, native
        ),
        "candidates": candidates,
        "morph_tables": morph_tables,
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest
