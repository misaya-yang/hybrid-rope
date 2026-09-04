#!/usr/bin/env python3
"""Export a frozen log-p2 factor bracket from the exact Native/s4 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def factor_slug(factor: float) -> str:
    return f"{factor:.8g}".replace(".", "p")


def derive_movement(
    native: np.ndarray, reference: np.ndarray, reference_factor: float
) -> np.ndarray:
    if native.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError("Native and reference tables must be float32")
    if native.ndim != 1 or native.shape != reference.shape or native.size == 0:
        raise ValueError("Native/reference table shape mismatch")
    if not np.all(native[:-1] > native[1:]) or not np.all(reference[:-1] > reference[1:]):
        raise ValueError("Native/reference tables must be strictly decreasing")
    if not np.all(native > 0.0) or not np.all(reference > 0.0):
        raise ValueError("Native/reference tables must be positive")
    movement = -np.log(reference.astype(np.float64) / native) / math.log(reference_factor)
    if np.min(movement) < -1e-6 or np.max(movement) > 1.0 + 1e-6:
        raise ValueError("reference movement leaves [0,1]")
    if abs(float(movement[0])) > 1e-6 or abs(float(movement[-1]) - 1.0) > 1e-5:
        raise ValueError("reference endpoints do not encode m_0=0,m_last=1")
    return movement


def realize(native: np.ndarray, movement: np.ndarray, factor: float) -> np.ndarray:
    table = np.ascontiguousarray(
        native.astype(np.float64) * np.power(float(factor), -movement), dtype="<f4"
    )
    table[0] = np.float32(native[0])
    table[-1] = np.float32(np.float32(native[-1]) / float(factor))
    if not np.all(table[:-1] > table[1:]):
        raise ValueError(f"factor {factor:g} introduces a frequency crossing")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-table", type=Path, required=True)
    parser.add_argument("--reference-log-s4", type=Path, required=True)
    parser.add_argument("--reference-factor", type=float, default=4.0)
    parser.add_argument(
        "--factors", type=float, nargs="+", default=(4.0, 5.0, 6.0, 7.0, 8.0)
    )
    parser.add_argument(
        "--gain-coefficients", type=float, nargs="+", default=(0.074,)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    factors = tuple(float(value) for value in args.factors)
    gain_coefficients = tuple(float(value) for value in args.gain_coefficients)
    if (
        not factors
        or len(set(factors)) != len(factors)
        or any(not math.isfinite(value) or value <= 1.0 for value in factors)
        or not math.isfinite(args.reference_factor)
        or args.reference_factor <= 1.0
        or not gain_coefficients
        or len(set(gain_coefficients)) != len(gain_coefficients)
        or any(not math.isfinite(value) or value < 0.0 for value in gain_coefficients)
    ):
        raise ValueError("factors and gain coefficient are invalid")

    native_path = args.native_table.expanduser().resolve()
    reference_path = args.reference_log_s4.expanduser().resolve()
    native = np.load(native_path, allow_pickle=False)
    reference = np.load(reference_path, allow_pickle=False)
    movement = derive_movement(native, reference, float(args.reference_factor))
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    tables: dict[str, dict[str, object]] = {}
    arms: dict[str, dict[str, object]] = {}
    for factor in sorted(factors):
        table = (
            np.ascontiguousarray(reference, dtype="<f4")
            if factor == float(args.reference_factor)
            else realize(native, movement, factor)
        )
        name = f"legacy_u_p2_log_s{factor_slug(factor)}"
        path = output / f"{name}.npy"
        np.save(path, table, allow_pickle=False)
        tables[name] = {
            "factor": factor,
            "path": path.name,
            "float32_sha256": tensor_sha256(table),
            "file_sha256": file_sha256(path),
            "table_support": "native_div_factor",
            "crossing_indices": np.flatnonzero(table[:-1] <= table[1:]).tolist(),
        }
        for coefficient in sorted(gain_coefficients):
            arm_name = f"{name}_c{factor_slug(coefficient)}"
            arm_entry = {
                "table": name,
                "factor": factor,
                "gain_coefficient": coefficient,
                "attention_scaling": 1.0 + coefficient * math.log(factor),
            }
            arms[arm_name] = arm_entry
            alt_slug = f"{coefficient:.2f}".replace(".", "p")
            if alt_slug != factor_slug(coefficient):
                alt_arm_name = f"{name}_c{alt_slug}"
                if alt_arm_name not in arms:
                    arms[alt_arm_name] = arm_entry

    manifest = {
        "status": "LOG_P2_FACTOR_FRONTIER_FROZEN_V1",
        "question": "largest one-table factor inside the registered Native-damage budget",
        "law": "omega_prime_k = omega_k * factor ** (-m_k)",
        "selection_data": "1x PG-19 plus five held-out generation tasks only",
        "long_endpoints_locked_until_selection": [2, 4, 8],
        "reference_factor": float(args.reference_factor),
        "gain_coefficients": list(gain_coefficients),
        "native": {
            "path": str(native_path),
            "float32_sha256": tensor_sha256(native),
            "file_sha256": file_sha256(native_path),
        },
        "reference_log_s4": {
            "path": str(reference_path),
            "float32_sha256": tensor_sha256(reference),
            "file_sha256": file_sha256(reference_path),
        },
        "movement": movement.tolist(),
        "tables": tables,
        "arms": arms,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "tables": len(tables)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
