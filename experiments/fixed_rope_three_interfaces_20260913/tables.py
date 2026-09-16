#!/usr/bin/env python3
"""Freeze exact static RoPE tables and matched three-interface controls.

This module never fits a table to task scores.  It either wraps an existing
saved table, derives a named analytic baseline, or applies one declared depth
to a saved normalized exponent profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import TABLE_FORMAT


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def model_geometry(config: dict) -> dict:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if hidden % heads:
        raise ValueError("hidden size must be divisible by attention heads")
    parameters = config.get("rope_parameters") or {}
    attention_head_dim = int(config.get("head_dim") or hidden // heads)
    partial = float(config.get("partial_rotary_factor") or parameters.get("partial_rotary_factor") or 1.0)
    head_dim = int(attention_head_dim * partial)
    if not math.isfinite(partial) or not 0 < partial <= 1 or head_dim != attention_head_dim * partial:
        raise ValueError("partial RoPE dimension must be a positive integer")
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    base = float(config.get("rope_theta") or parameters.get("rope_theta") or 0.0)
    native_length = int(config.get("max_position_embeddings", 0))
    if not math.isfinite(base) or base <= 1 or native_length <= 0:
        raise ValueError("model config lacks valid RoPE base/native length")
    return {
        "model_type": str(config.get("model_type", "unknown")),
        "attention_head_dim": attention_head_dim,
        "partial_rotary_factor": partial,
        "head_dim": head_dim,
        "pairs": head_dim // 2,
        "base": base,
        "native_length": native_length,
    }


def native_inv_freq(geometry: dict) -> np.ndarray:
    pairs = int(geometry["pairs"])
    return np.power(float(geometry["base"]), -np.arange(pairs, dtype=np.float64) / pairs)


def runtime_native_inv_freq(geometry: dict) -> np.ndarray:
    """Use the same Torch-FP32 construction installed by the project runtime."""
    from scripts.experiments.cross_audit.tables import native_table

    return native_table(int(geometry["head_dim"]), float(geometry["base"])).astype(np.float32)


def default_band(geometry: dict) -> tuple[int, int]:
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    turns = native * float(geometry["native_length"]) / (2.0 * math.pi)
    fast = np.flatnonzero(turns > 32.0)
    slow = np.flatnonzero(turns < 1.0)
    if not len(fast) or not len(slow) or int(slow[0]) <= int(fast[-1]):
        raise ValueError("native geometry lacks a 32-turn/1-turn transition")
    return int(fast[-1]), int(slow[0])


def analytic_exponents(
    method: str, pairs: int, *, low: int, high: int, depth: float = 1.0,
) -> np.ndarray:
    if not 0 <= low < high < pairs:
        raise ValueError("invalid transition band")
    if not math.isfinite(depth) or not 0 < depth <= 1:
        raise ValueError("depth must be in (0,1]")
    n = high - low
    q = np.clip(np.arange(pairs, dtype=np.float64) - low, 0, n)
    if method == "mrpro":
        profile = q * (q + 1.0) / (n * (n + 1.0))
    elif method == "mrpro_frontloaded":
        profile = 1.0 - (n - q) * (n - q + 1.0) / (n * (n + 1.0))
    elif method == "bm":
        profile = q * (q + 1.0) * (3.0 * n + 2.0 - 2.0 * q)
        profile /= n * (n + 1.0) * (n + 2.0)
    elif method == "uni":
        profile = q / n
    elif method == "mix075":
        front = q * (2.0 * n + 1.0 - q) / (n * (n + 1.0))
        bm = q * (q + 1.0) * (3.0 * n + 2.0 - 2.0 * q)
        bm /= n * (n + 1.0) * (n + 2.0)
        profile = 0.25 * bm + 0.75 * front
    elif method == "tailspline":
        if depth != 1.0:
            raise ValueError("TailSpline is parameter-free and requires depth=1")
        # Exact finite-grid one-sided minimum-bending solution.  This is not
        # the asymptotic 0.25*BM + 0.75*front approximation: its finite-grid
        # front weight is 3n/[2(2n+1)].
        profile = q * (3.0 * n * n + 3.0 * n + 1.0 - q * q)
        profile /= n * (n + 1.0) * (2.0 * n + 1.0)
    elif method == "tailspline_dose_control":
        uniform = q / n
        pro = q * (q + 1.0) / (n * (n + 1.0))
        front = 2.0 * uniform - pro
        weight = 3.0 * n / (2.0 * (2.0 * n + 1.0))
        profile = (1.0 - weight) * uniform + weight * front
    else:
        raise ValueError(f"unsupported analytic exponent method: {method}")
    return depth * profile


def find_table(value: dict) -> dict:
    current: Any = value
    for _ in range(4):
        if isinstance(current, dict) and "values_float32" in current and "gain" in current:
            return current
        if not isinstance(current, dict):
            break
        for key in ("table", "installed_table", "static_table"):
            if isinstance(current.get(key), dict):
                current = current[key]
                break
        else:
            break
    raise ValueError("JSON does not contain a static table with values_float32 and gain")


def validate_table(table: dict, *, pairs: int) -> tuple[np.ndarray, float]:
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    gain = float(table.get("gain", float("nan")))
    if (
        values.shape != (pairs,)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or np.any(values[:-1] <= values[1:])
        or not math.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("invalid static RoPE table")
    return values, gain


def exponents_from_table(
    values: np.ndarray, native: np.ndarray, scale: float, *, require_monotone: bool = True,
) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 1:
        raise ValueError("scale must exceed one")
    exponents = -np.log(values.astype(np.float64) / native) / math.log(scale)
    if (
        not np.isfinite(exponents).all()
        or exponents.min() < -2e-6
        or exponents.max() > 1.0 + 2e-6
        or (require_monotone and np.any(np.diff(exponents) < -2e-6))
    ):
        raise ValueError("table is outside the monotone Native-relative exponent family")
    exponents = np.clip(exponents, 0.0, 1.0)
    exponents[np.abs(exponents) <= 2e-6] = 0.0
    exponents[np.abs(1.0 - exponents) <= 2e-6] = 1.0
    return exponents


def profile_metadata(exponents: np.ndarray, *, tolerance: float = 2e-6) -> dict:
    increments = np.diff(exponents)
    support = np.flatnonzero(increments > tolerance)
    negative = np.flatnonzero(increments < -tolerance)
    if len(support):
        band = [int(support[0]), int(support[-1] + 1)]
        weights = increments[support]
        centroid = float(np.sum((support + 1) * weights) / np.sum(weights))
    else:
        band = None
        centroid = None
    return {
        "exponents": exponents.tolist(),
        "increments": increments.tolist(),
        "increment_support": support.astype(int).tolist(),
        "negative_increment_indices": negative.astype(int).tolist(),
        "monotone_exponents": not len(negative),
        "band_envelope": band,
        "depth": float(exponents[-1]),
        "sum_m": float(exponents.sum()),
        "increment_centroid": centroid,
        "zero_tolerance": tolerance,
    }


def make_receipt(
    *, candidate_id: str, model_id: str, role: str, scale: float,
    geometry: dict, values: np.ndarray, gain: float, construction: dict,
    source: str, parent_candidate_id: str | None = None,
    changed_variables: list[str] | None = None,
    allow_nonmonotone_exponents: bool = False,
) -> dict:
    if role not in {"candidate", "baseline", "control", "native"}:
        raise ValueError("unknown table role")
    validate_table({"values_float32": values, "gain": gain}, pairs=int(geometry["pairs"]))
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    if role == "native":
        if not np.array_equal(values.astype(np.float32), runtime_native_inv_freq(geometry)) or float(gain) != 1.0:
            raise ValueError("native receipt must contain the exact FP32 Native table at gain 1")
        exponents = np.zeros_like(native)
    else:
        exponents = exponents_from_table(
            values, native, scale, require_monotone=not allow_nonmonotone_exponents,
        )
    metadata = profile_metadata(exponents)
    reconstructed = (native * np.power(float(scale), -exponents)).astype(np.float32)
    residual_slots = np.flatnonzero(reconstructed != values.astype(np.float32))
    exponent_reference = runtime_native_inv_freq(geometry)
    return {
        "status": TABLE_FORMAT,
        "candidate_id": candidate_id,
        "model_id": model_id,
        "role": role,
        "scale": float(scale),
        "model_geometry": geometry,
        "table_sha256_float32": tensor_sha256(values),
        "gain": float(gain),
        "exponent_reference": "project runtime Torch-FP32 Native",
        "exponent_reference_sha256_float32": tensor_sha256(exponent_reference),
        "exponent_reconstruction": {
            "exact": not len(residual_slots),
            "residual_slots": residual_slots.astype(int).tolist(),
            "note": "deployment always uses exact values_float32; clipped exponents are analysis coordinates",
        },
        **metadata,
        "parent_candidate_id": parent_candidate_id,
        "changed_variables": list(changed_variables or []),
        "source": source,
        "table": {
            "values_float32": values.astype(np.float32).tolist(),
            "gain": float(gain),
            "construction": {
                **construction,
                **metadata,
                "same_table_all_layers_and_lengths": True,
                "runtime_table_switching": False,
                "model_weight_updates": 0,
                "nonmonotone_exponents_allowed": bool(allow_nonmonotone_exponents),
            },
        },
    }


def build_analytic(
    config: dict, *, method: str, scale: float, low: int | None,
    high: int | None, depth: float, gain: float | None,
) -> tuple[np.ndarray, float, dict]:
    geometry = model_geometry(config)
    native = native_inv_freq(geometry)
    runtime_native = runtime_native_inv_freq(geometry)
    if method == "native":
        return runtime_native, 1.0 if gain is None else gain, {
            "method": "identity", "precision_identity": "project runtime Torch-FP32 Native",
        }
    if method == "yarn":
        if low is not None or high is not None or depth != 1.0:
            raise ValueError("official static YaRN does not accept band/depth overrides")
        from scripts.experiments.cross_audit.tables import transform

        values, official_gain, construction = transform(
            runtime_native, dim=int(geometry["head_dim"]),
            base=float(geometry["base"]), reference_length=int(geometry["native_length"]),
            scale=float(scale), method="yarn",
        )
        return values, float(official_gain if gain is None else gain), {
            **construction,
            "identity": "official static YaRN frequency map on a frozen checkpoint; no YaRN SFT",
        }
    if method in {"tailspline", "tailspline_dose_control"}:
        if not math.isfinite(scale) or scale <= 1.0:
            raise ValueError("TailSpline deployment scale must exceed one")
        if depth != 1.0:
            raise ValueError("TailSpline and its dose control require depth=1")
        if (low is None) != (high is None):
            raise ValueError("low and high must be supplied together")
        canonical_low, canonical_high = default_band(geometry)
        if low is not None and (int(low), int(high)) != (canonical_low, canonical_high):
            raise ValueError("TailSpline requires the canonical MrRoPE 32/1-turn boundaries")
        canonical_gain = 1.0 + 0.1 * math.log(float(scale))
        if gain is not None and not math.isclose(
            float(gain), canonical_gain, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("TailSpline requires the shared YaRN/MrRoPE gain")
        n = canonical_high - canonical_low
        exponents = analytic_exponents(
            method, int(geometry["pairs"]), low=canonical_low,
            high=canonical_high, depth=1.0,
        )
        values = (
            runtime_native.astype(np.float64)
            * np.power(float(scale), -exponents)
        ).astype(np.float32)
        if method == "tailspline_dose_control":
            weight = 3.0 * n / (2.0 * (2.0 * n + 1.0))
            return values, canonical_gain, {
                "method": "tailspline_same_log_displacement_control",
                "low": canonical_low,
                "high": canonical_high,
                "transition_gaps": n,
                "tail_depth": 1.0,
                "formula": "C=(1-w)*U+w*Front; U_q=q/n; Front=2U-MrPro; w=3n/[2(2n+1)]",
                "finite_grid_front_weight": weight,
                "matched_candidate": "tailspline",
                "matched_quantity": "sum of Native-relative exponents and therefore total log-frequency displacement",
                "boundary_rule": "canonical MrRoPE native-grid 32-turn/1-turn boundaries",
                "gain_rule": "shared YaRN/MrRoPE 1+0.1*ln(scale) cos/sin amplitude",
                "identity": "parameter-free same-dose shape control C for exact finite-grid TailSpline",
                "fitted_coefficients": 0,
            }
        return values, canonical_gain, {
            "method": "tailspline_exact_finite_grid",
            "low": canonical_low,
            "high": canonical_high,
            "transition_gaps": n,
            "tail_depth": 1.0,
            "formula": "m_q=q*(3*n^2+3*n+1-q^2)/(n*(n+1)*(2*n+1))",
            "increment_formula": "epsilon_q=3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1))",
            "finite_grid_front_weight": 3.0 * n / (2.0 * (2.0 * n + 1.0)),
            "boundary_rule": "canonical MrRoPE native-grid 32-turn/1-turn boundaries",
            "gain_rule": "shared YaRN/MrRoPE 1+0.1*ln(scale) cos/sin amplitude",
            "identity": "parameter-free exact finite-grid TailSpline Native-relative static table",
            "fitted_coefficients": 0,
        }
    if (low is None) != (high is None):
        raise ValueError("low and high must be supplied together")
    if low is None and method in {"mrpro", "uni"}:
        from scripts.experiments.cross_audit.tables import transform

        exact_method = "mrpro" if method == "mrpro" else "mruni"
        values, official_gain, construction = transform(
            runtime_native, dim=int(geometry["head_dim"]),
            base=float(geometry["base"]), reference_length=int(geometry["native_length"]),
            scale=float(scale), method=exact_method,
        )
        return values, float(official_gain if gain is None else gain), {
            **construction,
            "identity": "canonical analytic Native-relative static table",
        }
    if low is None and method == "bm":
        import torch
        from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq

        values_tensor, official_gain, construction = boundary_matched_inv_freq(
            torch.from_numpy(runtime_native.copy()), base=float(geometry["base"]),
            reference_length=int(geometry["native_length"]), scale=float(scale),
        )
        values = values_tensor.cpu().float().numpy()
        precision_identity = "canonical FP32-native boundary-matched construction"
        if geometry["model_type"] == "llama":
            exponents = np.asarray(construction["exponents"], dtype=np.float64)
            values = (native * np.power(float(scale), -exponents)).astype(np.float32)
            precision_identity = "exact frozen Llama BM float64-native construction"
        return values, float(official_gain if gain is None else gain), {
            **construction,
            "identity": "canonical boundary-matched Native-relative static table",
            "precision_identity": precision_identity,
        }
    actual_low, actual_high = default_band(geometry) if low is None else (int(low), int(high))
    exponents = analytic_exponents(
        method, int(geometry["pairs"]), low=actual_low, high=actual_high, depth=depth,
    )
    values = (runtime_native.astype(np.float64) * np.power(float(scale), -exponents)).astype(np.float32)
    actual_gain = 1.0 + 0.1 * math.log(scale) if gain is None else float(gain)
    return values, actual_gain, {
        "method": method,
        "low": actual_low,
        "high": actual_high,
        "tail_depth": float(depth),
        "identity": "analytic Native-relative static table",
    }


def build_depth_control(
    config: dict, parent: dict, *, scale: float, depth: float, gain: float | None,
) -> tuple[np.ndarray, float, dict]:
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    if (
        parent.get("status") != TABLE_FORMAT
        or parent.get("model_geometry") != geometry
        or float(parent.get("scale", float("nan"))) != float(scale)
    ):
        raise ValueError("parent receipt has another format, geometry, or deployment scale")
    parent_table = find_table(parent)
    values, parent_gain = validate_table(parent_table, pairs=int(geometry["pairs"]))
    if tensor_sha256(values) != parent.get("table_sha256_float32"):
        raise ValueError("parent table differs from its receipt hash")
    if float(parent_gain) != float(parent.get("gain", float("nan"))):
        raise ValueError("parent table and receipt gain differ")
    full = exponents_from_table(values, native, scale)
    if not math.isfinite(depth) or not 0 < depth <= 1:
        raise ValueError("depth must be in (0,1]")
    exponents = full * depth
    installed = (
        values.copy() if depth == 1.0
        else (native * np.power(float(scale), -exponents)).astype(np.float32)
    )
    return installed, float(parent_gain if gain is None else gain), {
        "method": "normalized_profile_depth",
        "tail_depth": float(depth),
        "parent_table_sha256_float32": tensor_sha256(values),
        "full_depth_exponents": full.tolist(),
    }


def build_tail_denominator_control(
    config: dict, parent: dict, *, scale: float, slow_start: int,
    denominator_fraction: float,
) -> tuple[np.ndarray, float, dict]:
    """Keep all faster slots exact and change only an ultra-slow plateau.

    ``denominator_fraction=0.9`` changes the selected tail from ``/S`` to
    ``/(0.9*S)``.  A nonmonotone exponent step is legal only when the installed
    frequencies remain strictly descending; the receipt records that fact.
    """
    geometry = model_geometry(config)
    if (
        parent.get("status") != TABLE_FORMAT
        or parent.get("model_geometry") != geometry
        or float(parent.get("scale", float("nan"))) != float(scale)
    ):
        raise ValueError("parent receipt has another format, geometry, or scale")
    if not 0 < slow_start < int(geometry["pairs"]):
        raise ValueError("slow_start is outside the rotary table")
    if not math.isfinite(denominator_fraction) or not 0 < denominator_fraction <= 1:
        raise ValueError("denominator_fraction must be in (0,1]")
    table = find_table(parent)
    parent_values, gain = validate_table(table, pairs=int(geometry["pairs"]))
    values = parent_values.copy()
    runtime_native = runtime_native_inv_freq(geometry).astype(np.float64)
    values[slow_start:] = (
        runtime_native[slow_start:] / (float(scale) * denominator_fraction)
    ).astype(np.float32)
    validate_table({"values_float32": values, "gain": gain}, pairs=int(geometry["pairs"]))
    return values, gain, {
        "method": "tail_only_denominator_control",
        "slow_start": int(slow_start),
        "denominator": float(scale) * denominator_fraction,
        "denominator_fraction_of_S": float(denominator_fraction),
        "unchanged_prefix_sha256_float32": tensor_sha256(parent_values[:slow_start]),
        "parent_table_sha256_float32": tensor_sha256(parent_values),
        "identity": "k < slow_start exact parent; k >= slow_start runtime Native divided by fraction*S",
    }


def build_tail_cap_control(
    config: dict, parent: dict, *, scale: float, denominator_fraction: float,
) -> tuple[np.ndarray, float, dict]:
    """Replace the deepest part of a monotone profile by a softer plateau.

    For ``denominator_fraction=0.9`` at ``S=4``, the exponent cap is
    ``log_4(3.6)``.  Every faster slot below that cap remains bit-exact, while
    the end of the transition and the slow plateau are flattened smoothly.
    """
    geometry = model_geometry(config)
    if (
        parent.get("status") != TABLE_FORMAT
        or parent.get("model_geometry") != geometry
        or float(parent.get("scale", float("nan"))) != float(scale)
    ):
        raise ValueError("parent receipt has another format, geometry, or scale")
    if not math.isfinite(denominator_fraction) or not 0 < denominator_fraction <= 1:
        raise ValueError("denominator_fraction must be in (0,1]")
    table = find_table(parent)
    parent_values, gain = validate_table(table, pairs=int(geometry["pairs"]))
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    parent_exponents = exponents_from_table(parent_values, native, scale)
    denominator = float(scale) * denominator_fraction
    exponent_cap = math.log(denominator) / math.log(float(scale))
    exponents = np.minimum(parent_exponents, exponent_cap)
    changed = np.flatnonzero(exponents != parent_exponents)
    if not len(changed):
        raise ValueError("tail cap does not change the parent table")
    values = (native * np.power(float(scale), -exponents)).astype(np.float32)
    validate_table({"values_float32": values, "gain": gain}, pairs=int(geometry["pairs"]))
    return values, gain, {
        "method": "monotone_tail_exponent_cap",
        "first_changed_slot": int(changed[0]),
        "denominator": denominator,
        "denominator_fraction_of_S": float(denominator_fraction),
        "exponent_cap": exponent_cap,
        "unchanged_prefix_sha256_float32": tensor_sha256(parent_values[: int(changed[0])]),
        "parent_table_sha256_float32": tensor_sha256(parent_values),
        "identity": "parent exponent profile capped at log_S(fraction*S)",
    }


def build_exponent_mix_control(
    config: dict, left: dict, right: dict, *, scale: float, right_weight: float,
) -> tuple[np.ndarray, float, dict]:
    """Interpolate two frozen Native-relative exponent allocations."""
    geometry = model_geometry(config)
    if not math.isfinite(right_weight) or not 0.0 <= right_weight <= 1.0:
        raise ValueError("right_weight must be in [0,1]")
    for parent in (left, right):
        if (
            parent.get("status") != TABLE_FORMAT
            or parent.get("model_geometry") != geometry
            or float(parent.get("scale", float("nan"))) != float(scale)
        ):
            raise ValueError("mix parent has another format, geometry, or scale")
    left_values, left_gain = validate_table(find_table(left), pairs=int(geometry["pairs"]))
    right_values, right_gain = validate_table(find_table(right), pairs=int(geometry["pairs"]))
    if left_gain != right_gain:
        raise ValueError("exponent mix requires matched gains")
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    left_exponents = exponents_from_table(left_values, native, scale)
    right_exponents = exponents_from_table(right_values, native, scale)
    exponents = (1.0 - right_weight) * left_exponents + right_weight * right_exponents
    values = (native * np.power(float(scale), -exponents)).astype(np.float32)
    validate_table({"values_float32": values, "gain": left_gain}, pairs=int(geometry["pairs"]))
    return values, left_gain, {
        "method": "frozen_exponent_allocation_mix",
        "left_candidate_id": left["candidate_id"],
        "right_candidate_id": right["candidate_id"],
        "right_weight": float(right_weight),
        "left_table_sha256_float32": tensor_sha256(left_values),
        "right_table_sha256_float32": tensor_sha256(right_values),
    }


def build_gain_control(
    config: dict, parent: dict, *, scale: float, gain: float,
) -> tuple[np.ndarray, float, dict]:
    """Keep every deployed frequency bit-exact and change only RoPE gain."""
    geometry = model_geometry(config)
    if (
        parent.get("status") != TABLE_FORMAT
        or parent.get("model_geometry") != geometry
        or float(parent.get("scale", float("nan"))) != float(scale)
    ):
        raise ValueError("gain parent has another format, geometry, or scale")
    values, parent_gain = validate_table(find_table(parent), pairs=int(geometry["pairs"]))
    if not math.isfinite(gain) or gain <= 0.0 or float(gain) == float(parent_gain):
        raise ValueError("gain control requires a different finite positive gain")
    return values.copy(), float(gain), {
        "method": "gain_only_control",
        "parent_table_sha256_float32": tensor_sha256(values),
        "parent_gain": float(parent_gain),
        "new_gain": float(gain),
        "frequency_values_bit_exact": True,
    }


def normalized_log_band_coordinates(
    native_values: np.ndarray, *, low: int, high: int,
) -> tuple[np.ndarray, float]:
    """Return the clamped, actual-table log coordinate of one fixed band."""
    native = np.asarray(native_values, dtype=np.float64)
    if (
        native.ndim != 1
        or not 0 <= low < high < len(native)
        or not np.isfinite(native).all()
        or np.any(native <= 0.0)
        or np.any(native[:-1] <= native[1:])
    ):
        raise ValueError("invalid Native table or fixed transport band")
    span = float(math.log(native[low] / native[high]))
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("fixed transport band has no positive log-frequency span")
    coordinates = np.log(native[low] / native) / span
    return np.clip(coordinates, 0.0, 1.0), span


def fixed_u_alpha(*, log_band_span: float, scale_from: float, scale_to: float) -> float:
    """Unique coefficient preserving normalized within-band log frequency."""
    if (
        not math.isfinite(log_band_span)
        or log_band_span <= 0.0
        or not math.isfinite(scale_from)
        or not math.isfinite(scale_to)
        or not 1.0 < scale_from < scale_to
    ):
        raise ValueError("fixed-u transport requires A>0 and 1<S_from<S_to")
    left = math.log(scale_from)
    right = math.log(scale_to)
    alpha = left * (log_band_span + right) / (right * (log_band_span + left))
    if not 0.0 < alpha < 1.0:
        raise AssertionError("fixed-u interpolation coefficient left (0,1)")
    return float(alpha)


def build_scale_transport_control(
    config: dict,
    parent: dict,
    *,
    scale_from: float,
    scale_to: float,
    low: int,
    high: int,
    mode: str,
    gain: float | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Move one saved parent table to a larger scale at fixed ``m`` or fixed ``u``.

    Both arms preserve the parent's band, Native prefix, and fully scaled
    suffix.  By default they also preserve its gain; an explicit common target
    gain can be frozen for a matched final-method comparison.
    """
    geometry = model_geometry(config)
    if (
        parent.get("status") != TABLE_FORMAT
        or parent.get("model_geometry") != geometry
        or float(parent.get("scale", float("nan"))) != float(scale_from)
    ):
        raise ValueError("transport parent has another format, geometry, or source scale")
    parent_values, parent_gain = validate_table(
        find_table(parent), pairs=int(geometry["pairs"]),
    )
    if tensor_sha256(parent_values) != parent.get("table_sha256_float32"):
        raise ValueError("transport parent table differs from its receipt hash")
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    parent_exponents = exponents_from_table(parent_values, native, scale_from)
    coordinates, span = normalized_log_band_coordinates(native, low=low, high=high)
    tolerance = 2e-6
    if (
        np.max(np.abs(parent_exponents[: low + 1])) > tolerance
        or np.max(np.abs(parent_exponents[high:] - 1.0)) > tolerance
    ):
        raise ValueError("transport parent does not have the declared Native prefix and full tail")
    if mode == "fixed_m":
        target_exponents = parent_exponents.copy()
        alpha = 1.0
    elif mode == "fixed_u":
        alpha = fixed_u_alpha(
            log_band_span=span, scale_from=scale_from, scale_to=scale_to,
        )
        target_exponents = alpha * parent_exponents + (1.0 - alpha) * coordinates
    else:
        raise ValueError("transport mode must be fixed_m or fixed_u")
    if (
        np.any(np.diff(target_exponents) < -2e-10)
        or target_exponents.min() < -2e-10
        or target_exponents.max() > 1.0 + 2e-10
    ):
        raise AssertionError("scale transport left the monotone exponent family")
    target_exponents = np.clip(target_exponents, 0.0, 1.0)
    target_values = (
        native * np.power(float(scale_to), -target_exponents)
    ).astype(np.float32)
    target_gain = float(parent_gain if gain is None else gain)
    validate_table(
        {"values_float32": target_values, "gain": target_gain},
        pairs=int(geometry["pairs"]),
    )
    parent_u = (
        span * coordinates + math.log(scale_from) * parent_exponents
    ) / (span + math.log(scale_from))
    target_u = (
        span * coordinates + math.log(scale_to) * target_exponents
    ) / (span + math.log(scale_to))
    return target_values, target_gain, {
        "method": f"scale_transport_{mode}",
        "scale_from": float(scale_from),
        "scale_to": float(scale_to),
        "low": int(low),
        "high": int(high),
        "actual_log_band_span": span,
        "alpha": float(alpha),
        "parent_table_sha256_float32": tensor_sha256(parent_values),
        "parent_exponents": parent_exponents.tolist(),
        "target_exponents": target_exponents.tolist(),
        "actual_native_log_coordinates": coordinates.tolist(),
        "max_normalized_u_residual": (
            float(np.max(np.abs(target_u - parent_u))) if mode == "fixed_u" else None
        ),
        "parent_gain": float(parent_gain),
        "target_gain": target_gain,
        "same_gain_as_parent": target_gain == float(parent_gain),
        "same_native_prefix_and_full_target_tail": True,
        "identity": (
            "unique fixed-band normalized log-frequency coordinate transport"
            if mode == "fixed_u"
            else "same parent Native-relative exponents at the target scale"
        ),
    }


def build_bm_skew_control(
    config: dict, *, scale: float, low: int, high: int, skew: float, gain: float,
) -> tuple[np.ndarray, float, dict]:
    """Skew BM internally while preserving endpoints and symmetric-grid dose."""
    geometry = model_geometry(config)
    if not 0 <= low < high < int(geometry["pairs"]):
        raise ValueError("invalid transition band")
    if not math.isfinite(skew):
        raise ValueError("skew must be finite")
    n = high - low
    q = np.clip(np.arange(int(geometry["pairs"]), dtype=np.float64) - low, 0, n)
    t = q / n
    base = analytic_exponents("bm", int(geometry["pairs"]), low=low, high=high)
    perturbation = t * t * (1.0 - t) * (1.0 - t) * (1.0 - 2.0 * t)
    exponents = base + skew * perturbation
    if exponents.min() < -2e-8 or exponents.max() > 1.0 + 2e-8 or np.any(np.diff(exponents) < -2e-8):
        raise ValueError("skew makes the exponent allocation infeasible")
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    values = (native * np.power(float(scale), -exponents)).astype(np.float32)
    validate_table({"values_float32": values, "gain": gain}, pairs=int(geometry["pairs"]))
    return values, float(gain), {
        "method": "boundary_matched_antisymmetric_skew",
        "low": int(low), "high": int(high), "skew": float(skew),
        "perturbation": "t^2(1-t)^2(1-2t)",
        "sum_m": float(exponents.sum()),
        "identity": "BM plus an endpoint-and-slope-preserving antisymmetric perturbation",
    }


def build_exact_c42(
    config: dict, *, scale: float, increment_swaps: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Construct exact C42, optionally permuting increments at fixed mass/moment.

    Swap indices are one-based positions in the 18-dimensional active
    increment vector.  The non-trivial ``(3,15),(1,9)`` permutation preserves
    both its multiset and its exact first moment, unlike a Pro-to-reverse
    comparison whose centroid necessarily changes.
    """
    from fractions import Fraction
    from experiments.rope_fast_5090_20260912.e3_tables import exact_increments

    geometry = model_geometry(config)
    if (
        int(geometry["pairs"]) != 64
        or float(geometry["base"]) != 500_000.0
        or float(scale) != 4.0
    ):
        raise ValueError("exact C42 requires 64 RoPE pairs, base 500000, and S=4")
    increments = exact_increments("C42")
    original = list(increments)
    swaps = list(increment_swaps or [])
    touched: set[int] = set()
    for left, right in swaps:
        if not 1 <= left <= len(increments) or not 1 <= right <= len(increments) or left == right:
            raise ValueError("C42 increment swap indices must be distinct and in [1,18]")
        if left in touched or right in touched:
            raise ValueError("C42 increment swaps must be disjoint")
        touched.update((left, right))
        increments[left - 1], increments[right - 1] = increments[right - 1], increments[left - 1]
    original_mass = sum(original, Fraction(0))
    changed_mass = sum(increments, Fraction(0))
    original_moment = sum(Fraction(index) * value for index, value in enumerate(original, 1))
    changed_moment = sum(Fraction(index) * value for index, value in enumerate(increments, 1))
    if sorted(increments) != sorted(original) or changed_mass != original_mass or changed_moment != original_moment:
        raise ValueError("requested C42 permutation does not preserve multiset, mass, and first moment")
    exponents_fraction = [Fraction(0) for _ in range(64)]
    cumulative = Fraction(0)
    for slot, increment in enumerate(increments, start=15):
        cumulative += increment
        exponents_fraction[slot] = cumulative
    for slot in range(33, 64):
        exponents_fraction[slot] = Fraction(1)
    if sum(exponents_fraction, Fraction(0)) != Fraction(42):
        raise AssertionError("C42 fixed-total invariant changed")
    exponents = np.asarray([float(value) for value in exponents_fraction], dtype=np.float64)
    values = (native_inv_freq(geometry) * np.power(float(scale), -exponents)).astype(np.float32)
    gain = 1.0 + 0.1 * math.log(float(scale))
    return values, gain, {
        "method": "exact_C42_increment_order_permutation" if swaps else "exact_C42",
        "transition_slots": [15, 32],
        "increment_swaps_one_based": [[int(left), int(right)] for left, right in swaps],
        "increment_mass_exact": str(changed_mass),
        "increment_centroid_r_exact": str(changed_moment / changed_mass),
        "sum_m_exact": "42",
        "multiset_preserved_exact": True,
        "mass_preserved_exact": True,
        "first_moment_preserved_exact": True,
        "increments_rational": [f"{value.numerator}/{value.denominator}" for value in increments],
        "identity": "exact C42 polynomial increments; optional order-only permutation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analytic = subparsers.add_parser("analytic")
    analytic.add_argument("--config", type=Path, required=True)
    analytic.add_argument(
        "--method",
        choices=(
            "native", "yarn", "mrpro", "mrpro_frontloaded", "bm", "uni",
            "mix075", "tailspline", "tailspline_dose_control", "c42",
        ),
        required=True,
    )
    analytic.add_argument("--scale", type=float, required=True)
    analytic.add_argument("--low", type=int)
    analytic.add_argument("--high", type=int)
    analytic.add_argument("--depth", type=float, default=1.0)
    analytic.add_argument("--gain", type=float)

    depth = subparsers.add_parser("depth")
    depth.add_argument("--config", type=Path, required=True)
    depth.add_argument("--parent", type=Path, required=True)
    depth.add_argument("--scale", type=float, required=True)
    depth.add_argument("--depth", type=float, required=True)
    depth.add_argument("--gain", type=float)

    tail = subparsers.add_parser("tail")
    tail.add_argument("--config", type=Path, required=True)
    tail.add_argument("--parent", type=Path, required=True)
    tail.add_argument("--scale", type=float, required=True)
    tail.add_argument("--slow-start", type=int, required=True)
    tail.add_argument("--denominator-fraction", type=float, required=True)

    tail_cap = subparsers.add_parser("tail-cap")
    tail_cap.add_argument("--config", type=Path, required=True)
    tail_cap.add_argument("--parent", type=Path, required=True)
    tail_cap.add_argument("--scale", type=float, required=True)
    tail_cap.add_argument("--denominator-fraction", type=float, required=True)

    mix = subparsers.add_parser("mix")
    mix.add_argument("--config", type=Path, required=True)
    mix.add_argument("--left", type=Path, required=True)
    mix.add_argument("--right", type=Path, required=True)
    mix.add_argument("--scale", type=float, required=True)
    mix.add_argument("--right-weight", type=float, required=True)

    gain_control = subparsers.add_parser("gain")
    gain_control.add_argument("--config", type=Path, required=True)
    gain_control.add_argument("--parent", type=Path, required=True)
    gain_control.add_argument("--scale", type=float, required=True)
    gain_control.add_argument("--gain", type=float, required=True)

    scale_transport = subparsers.add_parser("scale-transport")
    scale_transport.add_argument("--config", type=Path, required=True)
    scale_transport.add_argument("--parent", type=Path, required=True)
    scale_transport.add_argument("--scale-from", type=float, required=True)
    scale_transport.add_argument("--scale-to", type=float, required=True)
    scale_transport.add_argument("--low", type=int, required=True)
    scale_transport.add_argument("--high", type=int, required=True)
    scale_transport.add_argument("--mode", choices=("fixed_m", "fixed_u"), required=True)
    scale_transport.add_argument("--gain", type=float)

    bm_skew = subparsers.add_parser("bm-skew")
    bm_skew.add_argument("--config", type=Path, required=True)
    bm_skew.add_argument("--scale", type=float, required=True)
    bm_skew.add_argument("--low", type=int, required=True)
    bm_skew.add_argument("--high", type=int, required=True)
    bm_skew.add_argument("--skew", type=float, required=True)
    bm_skew.add_argument("--gain", type=float, required=True)

    c42_order = subparsers.add_parser("c42-order")
    c42_order.add_argument("--config", type=Path, required=True)
    c42_order.add_argument("--scale", type=float, required=True)
    c42_order.add_argument(
        "--swap", action="append", default=[], metavar="LEFT:RIGHT",
        help="one-based disjoint positions in the 18-dimensional C42 increment vector",
    )

    wrap = subparsers.add_parser("wrap")
    wrap.add_argument("--config", type=Path, required=True)
    wrap.add_argument("--source-table", type=Path, required=True)
    wrap.add_argument("--scale", type=float, required=True)

    for subparser in (
        analytic, depth, tail, tail_cap, mix, gain_control, scale_transport,
        bm_skew, c42_order, wrap,
    ):
        subparser.add_argument("--candidate-id", required=True)
        subparser.add_argument("--model-id", required=True)
        subparser.add_argument("--role", choices=("candidate", "baseline", "control", "native"), required=True)
        subparser.add_argument("--parent-candidate-id")
        subparser.add_argument("--changed-variable", action="append", default=[])
        subparser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = read_json(args.config)
    geometry = model_geometry(config)
    if args.command == "analytic":
        if args.method == "c42":
            if args.low is not None or args.high is not None or args.depth != 1.0 or args.gain is not None:
                raise ValueError("exact C42 does not accept band, depth, or gain overrides")
            values, gain, construction = build_exact_c42(config, scale=args.scale)
        else:
            values, gain, construction = build_analytic(
                config, method=args.method, scale=args.scale, low=args.low, high=args.high,
                depth=args.depth, gain=args.gain,
            )
        source = f"analytic:{args.method}"
    elif args.command == "depth":
        parent = read_json(args.parent)
        values, gain, construction = build_depth_control(
            config, parent, scale=args.scale, depth=args.depth, gain=args.gain,
        )
        source = str(args.parent)
    elif args.command == "tail":
        parent = read_json(args.parent)
        values, gain, construction = build_tail_denominator_control(
            config, parent, scale=args.scale, slow_start=args.slow_start,
            denominator_fraction=args.denominator_fraction,
        )
        source = str(args.parent)
    elif args.command == "tail-cap":
        parent = read_json(args.parent)
        values, gain, construction = build_tail_cap_control(
            config, parent, scale=args.scale,
            denominator_fraction=args.denominator_fraction,
        )
        source = str(args.parent)
    elif args.command == "mix":
        left = read_json(args.left)
        right = read_json(args.right)
        values, gain, construction = build_exponent_mix_control(
            config, left, right, scale=args.scale, right_weight=args.right_weight,
        )
        source = f"mix:{args.left}:{args.right}"
    elif args.command == "gain":
        parent = read_json(args.parent)
        values, gain, construction = build_gain_control(
            config, parent, scale=args.scale, gain=args.gain,
        )
        source = str(args.parent)
    elif args.command == "scale-transport":
        parent = read_json(args.parent)
        values, gain, construction = build_scale_transport_control(
            config, parent, scale_from=args.scale_from, scale_to=args.scale_to,
            low=args.low, high=args.high, mode=args.mode, gain=args.gain,
        )
        source = str(args.parent)
    elif args.command == "bm-skew":
        values, gain, construction = build_bm_skew_control(
            config, scale=args.scale, low=args.low, high=args.high,
            skew=args.skew, gain=args.gain,
        )
        source = "analytic:boundary_matched_antisymmetric_skew"
    elif args.command == "c42-order":
        swaps = []
        for value in args.swap:
            try:
                left, right = (int(part) for part in value.split(":"))
            except (TypeError, ValueError) as error:
                raise ValueError("--swap must be LEFT:RIGHT") from error
            swaps.append((left, right))
        if not swaps:
            raise ValueError("c42-order requires at least one --swap")
        values, gain, construction = build_exact_c42(
            config, scale=args.scale, increment_swaps=swaps,
        )
        source = "analytic:exact_C42_order_permutation"
    else:
        source_payload = read_json(args.source_table)
        source_table = find_table(source_payload)
        values, gain = validate_table(source_table, pairs=int(geometry["pairs"]))
        construction = {
            **source_table.get("construction", {}),
            "method": "wrapped_exact_saved_table",
            "wrapped_source_status": source_payload.get("status"),
            "wrapped_source_label": source_payload.get("label"),
        }
        source = str(args.source_table)
    receipt = make_receipt(
        candidate_id=args.candidate_id, model_id=args.model_id, role=args.role,
        scale=(args.scale_to if args.command == "scale-transport" else args.scale),
        geometry=geometry, values=values, gain=gain,
        construction=construction, source=source,
        parent_candidate_id=args.parent_candidate_id,
        changed_variables=args.changed_variable,
        allow_nonmonotone_exponents=args.command == "tail",
    )
    atomic_json(args.out, receipt)
    print(json.dumps({
        "status": receipt["status"], "candidate_id": receipt["candidate_id"],
        "table_sha256_float32": receipt["table_sha256_float32"], "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
