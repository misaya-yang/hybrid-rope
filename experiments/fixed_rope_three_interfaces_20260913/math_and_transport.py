#!/usr/bin/env python3
"""Verify the 2026-09-14 identities and build one fixed-u transport capsule.

This utility is CPU-only.  A capsule is a construction receipt, not evidence
that a checkpoint or task improved.  Use ``tables.py wrap`` or the preparation
pipeline to turn it into the existing resident runner's frozen-table format.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .tables import fixed_u_alpha, normalized_log_band_coordinates, tensor_sha256


FORMAT = "FIXED_U_TRANSPORT_CAPSULE_V1"
VERIFY_FORMAT = "ROPE_TODAY_MATH_VERIFICATION_V1"


def analytic_profiles(pairs: int, *, low: int, high: int) -> dict[str, np.ndarray]:
    if not 0 <= low < high < pairs:
        raise ValueError("invalid analytic band")
    n = high - low
    q = np.clip(np.arange(pairs, dtype=np.float64) - low, 0, n)
    pro = q * (q + 1.0) / (n * (n + 1.0))
    front = q * (2.0 * n + 1.0 - q) / (n * (n + 1.0))
    bm = q * (q + 1.0) * (3.0 * n + 2.0 - 2.0 * q)
    bm /= n * (n + 1.0) * (n + 2.0)
    uniform = q / n
    mix075 = 0.25 * bm + 0.75 * front
    return {"mrpro": pro, "front": front, "bm": bm, "uniform": uniform, "mix075": mix075}


def exponents_from_actual_table(
    native_inv_freq: np.ndarray, parent_inv_freq: np.ndarray, scale: float,
) -> np.ndarray:
    native = np.asarray(native_inv_freq, dtype=np.float64)
    parent = np.asarray(parent_inv_freq, dtype=np.float64)
    if (
        native.ndim != 1
        or parent.shape != native.shape
        or len(native) < 2
        or not np.isfinite(native).all()
        or not np.isfinite(parent).all()
        or np.any(native <= 0.0)
        or np.any(parent <= 0.0)
        or np.any(native[:-1] <= native[1:])
        or np.any(parent[:-1] <= parent[1:])
        or not math.isfinite(scale)
        or scale <= 1.0
    ):
        raise ValueError("invalid Native/parent table or source scale")
    values = -np.log(parent / native) / math.log(scale)
    if values.min() < -2e-6 or values.max() > 1.0 + 2e-6 or np.any(np.diff(values) < -2e-6):
        raise ValueError("parent is outside the monotone Native-relative exponent family")
    values = np.clip(values, 0.0, 1.0)
    values[np.abs(values) <= 2e-6] = 0.0
    values[np.abs(1.0 - values) <= 2e-6] = 1.0
    return values


def normalized_u(
    native_inv_freq: np.ndarray,
    exponents: np.ndarray,
    *,
    scale: float,
    low: int,
    high: int,
) -> np.ndarray:
    coordinates, span = normalized_log_band_coordinates(native_inv_freq, low=low, high=high)
    return (span * coordinates + math.log(scale) * exponents) / (span + math.log(scale))


def transport_fixed_u(
    native_inv_freq: np.ndarray,
    parent_inv_freq: np.ndarray,
    *,
    scale_from: float,
    scale_to: float,
    low: int,
    high: int,
) -> dict:
    native = np.asarray(native_inv_freq, dtype=np.float64)
    parent = np.asarray(parent_inv_freq, dtype=np.float64)
    parent_m = exponents_from_actual_table(native, parent, scale_from)
    coordinates, span = normalized_log_band_coordinates(native, low=low, high=high)
    if (
        np.max(np.abs(parent_m[: low + 1])) > 2e-6
        or np.max(np.abs(parent_m[high:] - 1.0)) > 2e-6
    ):
        raise ValueError("declared band does not match the parent's exact prefix/tail")
    alpha = fixed_u_alpha(
        log_band_span=span, scale_from=scale_from, scale_to=scale_to,
    )
    target_m = alpha * parent_m + (1.0 - alpha) * coordinates
    target_m = np.clip(target_m, 0.0, 1.0)
    if np.any(np.diff(target_m) < -2e-10):
        raise AssertionError("fixed-u transport is not monotone")
    fixed_m = (native * np.power(scale_to, -parent_m)).astype(np.float32)
    fixed_u = (native * np.power(scale_to, -target_m)).astype(np.float32)
    parent_u = normalized_u(native, parent_m, scale=scale_from, low=low, high=high)
    target_u = normalized_u(native, target_m, scale=scale_to, low=low, high=high)
    return {
        "alpha": float(alpha),
        "actual_log_band_span": float(span),
        "coordinates": coordinates,
        "parent_exponents": parent_m,
        "target_exponents": target_m,
        "fixed_m_inv_freq": fixed_m,
        "fixed_u_inv_freq": fixed_u,
        "max_normalized_u_residual": float(np.max(np.abs(parent_u - target_u))),
    }


def build_capsule(payload: dict) -> tuple[dict, dict[str, np.ndarray]]:
    required = (
        "native_inv_freq", "parent_inv_freq", "native_length", "scale_from",
        "scale_to", "low", "high", "gain",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"transport capsule lacks {missing}")
    native = np.asarray(payload["native_inv_freq"], dtype=np.float32)
    parent = np.asarray(payload["parent_inv_freq"], dtype=np.float32)
    result = transport_fixed_u(
        native, parent, scale_from=float(payload["scale_from"]),
        scale_to=float(payload["scale_to"]), low=int(payload["low"]),
        high=int(payload["high"]),
    )
    gain = float(payload["gain"])
    native_length = int(payload["native_length"])
    if not math.isfinite(gain) or gain <= 0.0 or native_length <= 0:
        raise ValueError("capsule gain/native length is invalid")
    arrays = {
        "native_inv_freq": native,
        "parent_inv_freq": parent,
        "fixed_m_inv_freq": result["fixed_m_inv_freq"],
        "fixed_u_inv_freq": result["fixed_u_inv_freq"],
    }
    receipt = {
        "status": FORMAT,
        "model_id": str(payload.get("model_id", "unspecified")),
        "native_length": native_length,
        "scale_from": float(payload["scale_from"]),
        "scale_to": float(payload["scale_to"]),
        "low": int(payload["low"]),
        "high": int(payload["high"]),
        "gain": gain,
        "alpha": result["alpha"],
        "actual_log_band_span": result["actual_log_band_span"],
        "max_normalized_u_residual": result["max_normalized_u_residual"],
        "array_sha256_float32": {name: tensor_sha256(value) for name, value in arrays.items()},
        "parent_exponents": result["parent_exponents"].tolist(),
        "target_exponents": result["target_exponents"].tolist(),
        "actual_native_log_coordinates": result["coordinates"].tolist(),
        "table": {
            "values_float32": arrays["fixed_u_inv_freq"].tolist(),
            "gain": gain,
            "construction": {
                "method": "fixed_band_normalized_log_frequency_coordinate_transport",
                "formula": "m1=alpha*m0+(1-alpha)*t_actual",
                "alpha": result["alpha"],
                "same_gain_as_parent": True,
                "same_native_prefix_and_full_target_tail": True,
                "model_weight_updates": 0,
            },
        },
        "scope": "CPU construction receipt; no checkpoint or task result",
    }
    return receipt, arrays


def _check(checks: list[dict], name: str, condition: bool, detail: object = None) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append({"name": name, "passed": True, "detail": detail})


def run_verification() -> dict:
    checks: list[dict] = []
    p = analytic_profiles(64, low=16, high=34)
    internal = slice(17, 34)
    profile_names = ("mrpro", "front", "bm", "mix075")
    _check(checks, "analytic_profile_endpoints", all(
        p[name][16] == 0.0 and p[name][34] == 1.0 for name in profile_names
    ))
    _check(checks, "analytic_profiles_monotone", all(
        np.all(np.diff(p[name]) >= 0.0) for name in profile_names
    ))
    _check(checks, "front_gt_bm", bool(np.all(p["front"][internal] > p["bm"][internal])))
    _check(checks, "bm_gt_mrpro", bool(np.all(p["bm"][internal] > p["mrpro"][internal])))
    _check(checks, "mix_gt_bm", bool(np.all(p["mix075"][internal] > p["bm"][internal])))
    _check(checks, "mix_gt_uniform", bool(np.all(p["mix075"][internal] > p["uniform"][internal])))
    n = 18
    q = np.arange(n + 1, dtype=np.float64)
    _check(checks, "front_minus_bm_identity", bool(np.allclose(
        (p["front"] - p["bm"])[16:35], 2*q*(n-q)*(n-q+1)/(n*(n+1)*(n+2)), atol=2e-15,
    )))
    _check(checks, "bm_minus_pro_identity", bool(np.allclose(
        (p["bm"] - p["mrpro"])[16:35], 2*q*(q+1)*(n-q)/(n*(n+1)*(n+2)), atol=2e-15,
    )))
    _check(checks, "mix_minus_uniform_identity", bool(np.allclose(
        (p["mix075"] - p["uniform"])[16:35], q*(n-q)*(n+q+3)/(2*n*(n+1)*(n+2)), atol=2e-15,
    )))
    for name in ("mrpro", "front", "bm", "mix075"):
        _check(checks, f"{name}_increment_mass", abs(float(np.diff(p[name]).sum()) - 1.0) < 1e-14)

    exponent = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    moved = exponent.copy()
    moved[1:3] -= 0.07
    native = np.geomspace(1.0, 0.01, len(exponent))
    before = native * np.power(4.0, -exponent)
    after = native * np.power(4.0, -moved)
    _check(checks, "gap_move_prefix_unchanged", after[0] == before[0])
    _check(checks, "gap_move_middle_exponent", bool(np.allclose(moved[1:3] - exponent[1:3], -0.07)))
    _check(checks, "gap_move_frequency_ratio", bool(np.allclose(after[1:3] / before[1:3], 4.0 ** 0.07)))
    _check(checks, "gap_move_suffix_unchanged", bool(np.array_equal(after[3:], before[3:])))

    llama_native = np.power(500_000.0, -np.arange(64, dtype=np.float64) / 64.0)
    llama_parent = llama_native * np.power(4.0, -p["mix075"])
    moved_u = transport_fixed_u(
        llama_native, llama_parent, scale_from=4.0, scale_to=8.0, low=16, high=34,
    )
    _check(checks, "llama_alpha", abs(moved_u["alpha"] - 0.757685348286866) < 2e-15, moved_u["alpha"])
    _check(checks, "alpha_in_unit_interval", 0.0 < moved_u["alpha"] < 1.0)
    _check(checks, "fixed_u_endpoints", moved_u["target_exponents"][16] == 0.0 and moved_u["target_exponents"][34] == 1.0)
    _check(checks, "fixed_u_monotone", bool(np.all(np.diff(moved_u["target_exponents"]) >= 0.0)))
    _check(checks, "fixed_u_coordinate_preserved", moved_u["max_normalized_u_residual"] < 3e-16)
    _check(checks, "fixed_u_faster_than_fixed_m", bool(np.all(
        moved_u["fixed_u_inv_freq"][internal] > moved_u["fixed_m_inv_freq"][internal]
    )))
    _check(checks, "llama_midpoint_exponent", abs(moved_u["target_exponents"][25] - 0.634588845) < 6e-10)
    ratio = float(moved_u["fixed_u_inv_freq"][25] / moved_u["fixed_m_inv_freq"][25])
    _check(checks, "llama_midpoint_frequency_ratio", abs(ratio - 1.093632637) < 2e-7, ratio)

    olmo_profile = analytic_profiles(64, low=14, high=31)["mix075"]
    olmo_native = np.power(500_000.0, -np.arange(64, dtype=np.float64) / 64.0)
    olmo_parent = olmo_native * np.power(4.0, -olmo_profile)
    olmo = transport_fixed_u(
        olmo_native, olmo_parent, scale_from=4.0, scale_to=8.0, low=14, high=31,
    )
    _check(checks, "olmo_alpha", abs(olmo["alpha"] - 0.7615159085) < 8e-11, olmo["alpha"])
    direct16 = transport_fixed_u(
        llama_native, llama_parent, scale_from=4.0, scale_to=16.0, low=16, high=34,
    )
    step_parent = moved_u["fixed_u_inv_freq"].astype(np.float64)
    step16 = transport_fixed_u(
        llama_native, step_parent, scale_from=8.0, scale_to=16.0, low=16, high=34,
    )
    _check(checks, "fixed_u_composition", bool(np.allclose(
        direct16["target_exponents"], step16["target_exponents"], atol=3e-8, rtol=0.0,
    )))
    minimum_denominator = 4.0 ** (1.0 - 1.0 / 57.0)
    _check(checks, "tail_minimum_denominator", abs(minimum_denominator - 3.9038896701) < 5e-11)
    _check(checks, "tail_3p9_outside_monotone_contract", math.log(3.9, 4.0) < 1.0 - 1.0 / 57.0)
    y00, y01, y10, y11 = 0.2, 0.3, 0.5, 0.9
    d_f = 0.5 * ((y10 - y00) + (y11 - y01))
    d_g = 0.5 * ((y01 - y00) + (y11 - y10))
    _check(checks, "gain_factorial_symmetric_sum", abs(d_f + d_g - (y11 - y00)) < 1e-15)
    weights = np.array([1/3, 1/2, 1/6])
    _check(checks, "olmo_stratified_auc_weights", abs(float(weights.sum()) - 1.0) < 1e-15)
    if len(checks) != 31:
        raise AssertionError(f"verification suite changed size: {len(checks)}")
    return {
        "status": VERIFY_FORMAT,
        "checks": checks,
        "passed": len(checks),
        "scope": "CPU algebra and synthetic identities only; no checkpoint evidence",
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--capsule", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    if args.verify:
        result = run_verification()
        atomic_json(args.out / "verification.json", result)
        print(json.dumps({"status": result["status"], "passed": result["passed"]}, sort_keys=True))
        return
    payload = json.loads(args.capsule.read_text())
    if not isinstance(payload, dict):
        raise ValueError("capsule input must be a JSON object")
    receipt, arrays = build_capsule(payload)
    for name, value in arrays.items():
        np.save(args.out / f"{name}.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)
    atomic_json(args.out / "table_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"], "alpha": receipt["alpha"],
        "fixed_u_sha256": receipt["array_sha256_float32"]["fixed_u_inv_freq"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
