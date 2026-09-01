#!/usr/bin/env python3
"""Export zero-refit frozen coupling tables for an arbitrary full-RoPE grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_X_HIGH = 0.7382780681078285
DEFAULT_X_LOW = 0.366403835112904
SOURCE_PAIRS = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clipped_affine(x: np.ndarray, *, x_high: float, x_low: float) -> np.ndarray:
    if not x_high > x_low:
        raise ValueError("x_high must exceed x_low")
    return np.clip((x_high - x) / (x_high - x_low), 0.0, 1.0)


def grid_coordinate(
    *,
    pairs: int,
    rope_theta: float,
    native_length: int,
    c_orth_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if pairs <= 0 or rope_theta <= 1.0 or native_length <= 0:
        raise ValueError("invalid RoPE grid")
    slots = np.arange(pairs, dtype=np.float64)
    omega = np.power(float(rope_theta), -slots / float(pairs))
    c_orth = 1.0 / (1.0 - float(rope_theta) ** (-1.0 / float(pairs)))
    used_c_orth = c_orth if c_orth_override is None else float(c_orth_override)
    if not math.isfinite(used_c_orth) or used_c_orth <= 0.0:
        raise ValueError("invalid c_orth")
    x = np.log(float(native_length) * omega / (2.0 * math.pi * used_c_orth))
    return omega, x, c_orth


def config_identity(config_path: Path, native_length: int | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("checkpoint head_dim must be positive and even")
    partial = float(config.get("partial_rotary_factor", 1.0))
    if partial != 1.0:
        raise ValueError("only full-head RoPE checkpoints are admitted")
    rope_scaling = config.get("rope_scaling")
    if rope_scaling not in (None, {}):
        raise ValueError("checkpoint already has RoPE scaling")
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_theta = dict(config.get("rope_parameters") or {}).get("rope_theta")
    if rope_theta is None:
        raise ValueError("checkpoint rope_theta is unavailable")
    maximum = int(config["max_position_embeddings"])
    realized_native = maximum if native_length is None else int(native_length)
    if realized_native != maximum:
        raise ValueError("native_length must equal checkpoint max_position_embeddings")
    return {
        "model_type": str(config["model_type"]),
        "head_dim": head_dim,
        "pairs": head_dim // 2,
        "rope_theta": float(rope_theta),
        "native_length": realized_native,
        "partial_rotary_factor": partial,
        "rope_scaling": rope_scaling,
        "config_sha256": sha256_file(config_path),
    }


def normalized_index_movement(
    target_pairs: int,
    *,
    rope_theta: float,
    native_length: int,
    x_high: float,
    x_low: float,
) -> np.ndarray:
    _, source_x, _ = grid_coordinate(
        pairs=SOURCE_PAIRS,
        rope_theta=rope_theta,
        native_length=native_length,
    )
    source_m = clipped_affine(source_x, x_high=x_high, x_low=x_low)
    source_z = np.linspace(0.0, 1.0, SOURCE_PAIRS, dtype=np.float64)
    target_z = np.linspace(0.0, 1.0, target_pairs, dtype=np.float64)
    return np.interp(target_z, source_z, source_m)


def build_tables(
    identity: dict[str, Any],
    *,
    scale: float,
    x_high: float,
    x_low: float,
    include_wrong_c_orth: bool,
    native_override: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    if not math.isfinite(scale) or scale <= 1.0:
        raise ValueError("scale must exceed one")
    pairs = int(identity["pairs"])
    omega, x, c_orth = grid_coordinate(
        pairs=pairs,
        rope_theta=float(identity["rope_theta"]),
        native_length=int(identity["native_length"]),
    )
    if native_override is not None:
        omega = np.asarray(native_override, dtype=np.float64)
        if (
            omega.shape != (pairs,)
            or not np.isfinite(omega).all()
            or not (omega > 0.0).all()
            or not np.all(omega[:-1] > omega[1:])
        ):
            raise ValueError("runtime Native inverse-frequency tensor is invalid")
        x = np.log(
            float(identity["native_length"]) * omega / (2.0 * math.pi * c_orth)
        )
    movement = {
        "dimensionless_x": clipped_affine(x, x_high=x_high, x_low=x_low),
        "normalized_raw_index": normalized_index_movement(
            pairs,
            rope_theta=float(identity["rope_theta"]),
            native_length=int(identity["native_length"]),
            x_high=x_high,
            x_low=x_low,
        ),
    }
    c_orth_values = {"target": c_orth}
    if include_wrong_c_orth:
        _, _, source_c_orth = grid_coordinate(
            pairs=SOURCE_PAIRS,
            rope_theta=float(identity["rope_theta"]),
            native_length=int(identity["native_length"]),
        )
        wrong_x = np.log(
            float(identity["native_length"])
            * omega
            / (2.0 * math.pi * source_c_orth)
        )
        movement["wrong_source_c_orth"] = clipped_affine(
            wrong_x,
            x_high=x_high,
            x_low=x_low,
        )
        c_orth_values["wrong_source"] = source_c_orth
    tables = {
        name: np.ascontiguousarray(omega * np.power(scale, -values), dtype="<f4")
        for name, values in movement.items()
    }
    return tables, movement, c_orth_values


def export(args: argparse.Namespace) -> dict[str, Any]:
    identity = config_identity(args.config, args.native_length)
    formula_native, _, _ = grid_coordinate(
        pairs=int(identity["pairs"]),
        rope_theta=float(identity["rope_theta"]),
        native_length=int(identity["native_length"]),
    )
    native = formula_native
    if args.native_inv is not None:
        probe = np.load(args.native_inv, allow_pickle=False)
        if probe.dtype != np.dtype("float32"):
            raise ValueError("runtime Native tensor must be float32")
        native = np.asarray(probe, dtype=np.float64)
        identity["native_tensor_file_sha256"] = sha256_file(args.native_inv)
        identity["native_tensor_source"] = "runtime rotary initializer"
    else:
        identity["native_tensor_source"] = "config formula"
    identity["native_sha256_float32"] = sha256_bytes(
        np.ascontiguousarray(native, dtype="<f4").tobytes(order="C")
    )
    identity["formula_native_max_abs_difference"] = float(
        np.max(np.abs(native - formula_native))
    )
    tables, movements, c_orth = build_tables(
        identity,
        scale=float(args.scale),
        x_high=float(args.x_high),
        x_low=float(args.x_low),
        include_wrong_c_orth=bool(args.include_wrong_c_orth),
        native_override=native,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "status": "FROZEN_COUPLING_TRANSPORT_EXPORTED",
        "benchmark_scores_used": False,
        "checkpoint": identity,
        "law": {
            "family": "clipped_affine",
            "x_high": float(args.x_high),
            "x_low": float(args.x_low),
            "scale": float(args.scale),
            "formula": "omega_prime_i = omega_i * scale ** (-movement_i)",
        },
        "source_grid_for_normalized_index": {
            "pairs": SOURCE_PAIRS,
            "rope_theta": float(identity["rope_theta"]),
            "native_length": int(identity["native_length"]),
            "definition": "counterfactual K64 point samples at the target checkpoint base/native length, transported by i/(K-1)",
        },
        "c_orth": c_orth,
        "gain": {
            "coefficient": float(args.gain_coefficient),
            "attention_scaling": 1.0
            + float(args.gain_coefficient) * math.log(float(args.scale)),
        },
        "tables": {},
    }
    for name, table in tables.items():
        if not np.isfinite(table).all() or not (table > 0.0).all():
            raise RuntimeError(f"{name} table is non-finite or non-positive")
        crossings = np.flatnonzero(table[:-1] <= table[1:]).tolist()
        if crossings:
            raise RuntimeError(f"{name} table has frequency crossings")
        path = args.output / f"{name}_s{float(args.scale):g}.npy"
        np.save(path, table, allow_pickle=False)
        receipt["tables"][name] = {
            "path": path.name,
            "tensor_sha256": sha256_bytes(table.tobytes(order="C")),
            "file_sha256": sha256_file(path),
            "movement": movements[name].tolist(),
            "nonbinary_slots": np.flatnonzero(
                (movements[name] > 1e-10) & (movements[name] < 1.0 - 1e-10)
            ).tolist(),
            "crossing_indices": crossings,
        }
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-length", type=int)
    parser.add_argument("--native-inv", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--x-high", type=float, default=DEFAULT_X_HIGH)
    parser.add_argument("--x-low", type=float, default=DEFAULT_X_LOW)
    parser.add_argument("--gain-coefficient", type=float, default=0.074)
    parser.add_argument("--include-wrong-c-orth", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
