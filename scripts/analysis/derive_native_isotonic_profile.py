#!/usr/bin/env python3
"""Derive the pinned Iso(1-u) RoPE profile from Native geometry only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import mpmath as mp
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(payload.tobytes()).hexdigest()


def native_from_config(config_path: Path) -> np.ndarray:
    config = json.loads(config_path.read_text())
    head_dim = int(config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"])
    rope_theta = float(config.get("rope_theta", 10_000.0))
    try:
        import torch
    except ImportError as error:  # pragma: no cover - work-machine path
        raise RuntimeError("torch is required to reproduce the checkpoint float32 table") from error
    indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    base = torch.tensor(rope_theta, dtype=torch.float32)
    native = 1.0 / torch.pow(base, indices / float(head_dim))
    return native.cpu().numpy().astype("<f4", copy=False)


def triangular_fourier(alpha: mp.mpf, length: int) -> mp.mpc:
    if alpha == 0:
        return mp.mpc(1)
    exponent = mp.j * alpha
    step = mp.expm1(exponent)
    numerator = mp.expm1((length + 1) * exponent) - (length + 1) * step
    denominator = (length * (length + 1) / 2) * step ** 2
    return numerator / denominator


def phase_gram(native: Iterable[float], length: int, dps: int) -> mp.matrix:
    mp.mp.dps = int(dps)
    omega = [mp.mpf(float(value)) for value in native]
    pairs = len(omega)
    gram = mp.matrix(2 * pairs, 2 * pairs)
    cache: dict[tuple[int, int, int], mp.mpc] = {}

    def transform(i: int, j: int, sign: int) -> mp.mpc:
        key = (i, j, sign)
        if key not in cache:
            cache[key] = triangular_fourier(omega[i] + sign * omega[j], length)
        return cache[key]

    for i in range(pairs):
        for j in range(i, pairs):
            plus = transform(i, j, 1)
            minus = transform(i, j, -1)
            cos_cos = (mp.re(minus) + mp.re(plus)) / 2
            sin_sin = (mp.re(minus) - mp.re(plus)) / 2
            cos_sin = (mp.im(plus) - mp.im(minus)) / 2
            sin_cos = (mp.im(plus) + mp.im(minus)) / 2
            block = ((cos_cos, cos_sin), (sin_cos, sin_sin))
            for row in range(2):
                for col in range(2):
                    gram[2 * i + row, 2 * j + col] = block[row][col]
                    gram[2 * j + col, 2 * i + row] = block[row][col]
    return gram


def conditional_uniqueness(native: np.ndarray, length: int, dps: int) -> np.ndarray:
    gram = phase_gram(native, length, dps)
    inverse = gram ** -1
    values = []
    # The exact Gram is positive definite; this tolerance only absorbs the
    # residual of the point-MP inverse before authoritative float64 rounding.
    tolerance = mp.mpf("1e-25")
    for pair in range(len(native)):
        start = 2 * pair
        inverse_block = mp.matrix(
            [[inverse[start + i, start + j] for j in range(2)] for i in range(2)]
        )
        residual = inverse_block ** -1
        marginal_trace = gram[start, start] + gram[start + 1, start + 1]
        value = mp.re((residual[0, 0] + residual[1, 1]) / marginal_trace)
        if value < -tolerance or value > 1 + tolerance:
            raise RuntimeError(f"conditional uniqueness outside [0,1] at pair {pair}: {value}")
        values.append(float(min(mp.mpf(1), max(mp.mpf(0), value))))
    return np.asarray(values, dtype=np.float64)


def isotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    blocks: list[list[float | int]] = []
    for index, value in enumerate(source):
        blocks.append([index, index, float(value), 1.0])
        while len(blocks) >= 2 and float(blocks[-2][2]) > float(blocks[-1][2]):
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left[3]) + float(right[3])
            mean = (float(left[2]) * float(left[3]) + float(right[2]) * float(right[3])) / weight
            blocks.append([int(left[0]), int(right[1]), mean, weight])
    result = np.empty_like(source)
    for first, last, value, _weight in blocks:
        result[int(first) : int(last) + 1] = float(value)
    return result


def pinned_profile(uniqueness: np.ndarray) -> np.ndarray:
    if uniqueness.ndim != 1 or uniqueness.size < 2:
        raise ValueError("at least two rotary pairs are required")
    target = np.clip(1.0 - uniqueness, 0.0, 1.0)
    movement = np.empty_like(target)
    movement[0] = 0.0
    movement[-1] = 1.0
    if target.size > 2:
        movement[1:-1] = isotonic_nondecreasing(target[1:-1])
    if np.any(np.diff(movement) < -1e-14):
        raise RuntimeError("pinned movement is not nondecreasing")
    return movement


def self_test() -> None:
    mp.mp.dps = 60
    length = 9
    alpha = mp.mpf("0.137")
    direct = sum((length - delta) * mp.exp(mp.j * alpha * delta) for delta in range(length))
    direct /= length * (length + 1) / 2
    assert abs(triangular_fourier(alpha, length) - direct) < mp.mpf("1e-50")
    observed = isotonic_nondecreasing(np.asarray([0.0, 0.8, 0.4, 1.0]))
    assert np.allclose(observed, [0.0, 0.6, 0.6, 1.0])
    pinned = pinned_profile(np.asarray([0.9, 0.8, 0.3, 0.1]))
    assert np.allclose(pinned, [0.0, 0.2, 0.7, 1.0])


def parse_dps_grid(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if len(values) < 2 or values != sorted(set(values)) or values[0] < 40:
        raise ValueError("--dps-grid requires at least two increasing unique values >= 40")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-length", type=int, required=True)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--dps-grid", default="360,440,520")
    parser.add_argument("--convergence-tol", type=float, default=1e-9)
    parser.add_argument("--expected-native-sha256")
    parser.add_argument("--native-ulp-shift", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--legacy-log-table", type=Path)
    parser.add_argument("--expected-legacy-table-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.native_length <= 0 or not math.isfinite(args.factor) or args.factor <= 1:
        raise ValueError("native length must be positive and factor must exceed one")
    if args.output.exists():
        raise FileExistsError(args.output)

    native = native_from_config(args.config.resolve())
    source_native_hash = float32_sha256(native)
    if args.expected_native_sha256 and source_native_hash != args.expected_native_sha256:
        raise RuntimeError(f"Native tensor hash drift: {source_native_hash}")
    if args.native_ulp_shift:
        target = np.float32(np.inf if args.native_ulp_shift > 0 else -np.inf)
        native = np.nextafter(native, target, dtype=np.float32)
    native_hash = float32_sha256(native)

    dps_grid = parse_dps_grid(args.dps_grid)
    profiles = []
    uniqueness_values = []
    for dps in dps_grid:
        uniqueness = conditional_uniqueness(native, args.native_length, dps)
        uniqueness_values.append(uniqueness)
        profile = pinned_profile(uniqueness)
        profiles.append(profile)
        print(json.dumps({
            "status": "PRECISION_POINT_COMPLETE",
            "dps": dps,
            "uniqueness_min": float(uniqueness.min()),
            "uniqueness_max": float(uniqueness.max()),
            "movement_sha256_float64": hashlib.sha256(
                np.ascontiguousarray(profile, dtype="<f8").tobytes()
            ).hexdigest(),
        }, sort_keys=True), flush=True)
    u_delta = float(np.max(np.abs(uniqueness_values[-1] - uniqueness_values[-2])))
    m_delta = float(np.max(np.abs(profiles[-1] - profiles[-2])))
    u_delta_index = int(np.argmax(np.abs(uniqueness_values[-1] - uniqueness_values[-2])))
    m_delta_index = int(np.argmax(np.abs(profiles[-1] - profiles[-2])))
    if max(u_delta, m_delta) > args.convergence_tol:
        raise RuntimeError(
            "high-precision profile did not converge: "
            f"u={u_delta:.3g}@{u_delta_index}, m={m_delta:.3g}@{m_delta_index}"
        )

    movement = profiles[-1]
    table = np.ascontiguousarray(
        native.astype(np.float64) * np.power(float(args.factor), -movement),
        dtype="<f4",
    )
    if not np.isfinite(table).all() or not np.all(table[:-1] > table[1:]):
        raise RuntimeError("derived table is not finite, positive, and strictly decreasing")

    args.output.mkdir(parents=True)
    table_path = args.output / "exact_u_iso_log_s4.npy"
    native_path = args.output / "native_inv_freq.npy"
    movement_path = args.output / "movement.npy"
    uniqueness_path = args.output / "conditional_uniqueness.npy"
    np.save(native_path, native, allow_pickle=False)
    np.save(table_path, table, allow_pickle=False)
    np.save(movement_path, movement.astype("<f8"), allow_pickle=False)
    np.save(uniqueness_path, uniqueness_values[-1].astype("<f8"), allow_pickle=False)
    factorial: dict[str, dict[str, object]] = {}

    def save_factorial(
        name: str,
        candidate_movement: np.ndarray,
        candidate_table: np.ndarray | None = None,
    ) -> None:
        candidate = (
            np.ascontiguousarray(candidate_table, dtype="<f4")
            if candidate_table is not None
            else np.ascontiguousarray(
                native.astype(np.float64) * np.power(float(args.factor), -candidate_movement),
                dtype="<f4",
            )
        )
        if not np.isfinite(candidate).all() or not np.all(candidate[:-1] > candidate[1:]):
            raise RuntimeError(f"{name} is not finite, positive, and strictly decreasing")
        path = args.output / f"{name}.npy"
        np.save(path, candidate, allow_pickle=False)
        factorial[name] = {
            "movement": candidate_movement.tolist(),
            "table_sha256_float32": float32_sha256(candidate),
            "table_file_sha256": sha256_file(path),
        }

    save_factorial("exact_u_iso_log_s4", movement)
    exact_p2 = pinned_profile(1.0 - np.square(1.0 - uniqueness_values[-1]))
    save_factorial("exact_u_p2_log_s4", exact_p2)
    if args.legacy_log_table is not None:
        legacy_path = args.legacy_log_table.resolve()
        legacy = np.load(legacy_path, allow_pickle=False).astype("<f4", copy=False)
        legacy_hash = float32_sha256(legacy)
        if args.expected_legacy_table_sha256 and legacy_hash != args.expected_legacy_table_sha256:
            raise RuntimeError(f"legacy table hash drift: {legacy_hash}")
        if legacy.shape != native.shape:
            raise RuntimeError("legacy table shape drift")
        legacy_p2 = -np.log(legacy.astype(np.float64) / native.astype(np.float64)) / math.log(args.factor)
        legacy_p2[0], legacy_p2[-1] = 0.0, 1.0
        legacy_iso = pinned_profile(1.0 - np.sqrt(np.clip(legacy_p2, 0.0, 1.0)))
        save_factorial("legacy_u_p2_log_s4", legacy_p2, legacy)
        save_factorial("legacy_u_iso_log_s4", legacy_iso)
    receipt = {
        "status": "NATIVE_ISOTONIC_PROFILE_DERIVED",
        "construction": "full-lag high-precision conditional uniqueness; pinned Iso(1-u); log-frequency",
        "config_sha256": sha256_file(args.config.resolve()),
        "native_length": int(args.native_length),
        "factor": float(args.factor),
        "dps_grid": dps_grid,
        "convergence_tolerance": float(args.convergence_tol),
        "max_consecutive_uniqueness_delta": u_delta,
        "max_consecutive_movement_delta": m_delta,
        "max_consecutive_uniqueness_delta_index": u_delta_index,
        "max_consecutive_movement_delta_index": m_delta_index,
        "native_sha256_float32": native_hash,
        "native_file_sha256": sha256_file(native_path),
        "source_native_sha256_float32": source_native_hash,
        "native_ulp_shift": int(args.native_ulp_shift),
        "table_sha256_float32": float32_sha256(table),
        "table_file_sha256": sha256_file(table_path),
        "movement_file_sha256": sha256_file(movement_path),
        "uniqueness_file_sha256": sha256_file(uniqueness_path),
        "movement": movement.tolist(),
        "conditional_uniqueness": uniqueness_values[-1].tolist(),
        "endpoint_pins": [0.0, 1.0],
        "order_crossings": np.flatnonzero(table[:-1] <= table[1:]).tolist(),
        "factorial": factorial,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
