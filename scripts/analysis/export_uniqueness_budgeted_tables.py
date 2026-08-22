#!/usr/bin/env python3
"""Rebuild the frozen OLMo-2 uniqueness-budgeted RoPE tables on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


EXPECTED_DEFAULT_HASHES = {
    2.0: "f94a34381cfb3d05621db41bd3778779b16812246c392bd645013a1a06f80814",
    4.0: "a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3",
}


def float32_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def native_endpoint_inv_freq(
    *,
    head_dim: int = 128,
    rope_base: float = 500_000.0,
) -> np.ndarray:
    indices = torch.arange(0, int(head_dim), 2, dtype=torch.float32)
    base = torch.tensor(float(rope_base), dtype=torch.float32)
    values = 1.0 / torch.pow(base, indices / float(head_dim))
    return values.numpy().astype(np.float64)


def causal_distance_measure(
    *,
    length: int = 4096,
    max_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.arange(int(length), dtype=np.float64)
    weight = float(length) - support
    if support.size > int(max_points):
        stride = int(math.ceil(support.size / float(max_points)))
        reduced_support: list[int] = []
        reduced_weight: list[float] = []
        for lower in range(0, support.size, stride):
            upper = min(lower + stride, support.size)
            block = weight[lower:upper]
            total = float(block.sum())
            centroid = float((support[lower:upper] * block).sum() / total)
            reduced_support.append(
                int(np.clip(np.round(centroid), support[lower], support[upper - 1]))
            )
            reduced_weight.append(total)
        support = np.asarray(reduced_support, dtype=np.float64)
        weight = np.asarray(reduced_weight, dtype=np.float64)
    return support, weight / float(weight.sum())


def conditional_pair_uniqueness(
    omega: np.ndarray,
    support: np.ndarray,
    weight: np.ndarray,
    *,
    ridge: float = 1e-10,
) -> np.ndarray:
    theta = np.outer(
        np.asarray(support, dtype=np.float64),
        np.asarray(omega, dtype=np.float64),
    )
    basis = np.empty((theta.shape[0], 2 * theta.shape[1]), dtype=np.float64)
    basis[:, 0::2] = np.cos(theta)
    basis[:, 1::2] = np.sin(theta)
    design = basis * np.sqrt(np.asarray(weight, dtype=np.float64))[:, None]
    dimensions = np.arange(design.shape[1])
    values = np.empty(theta.shape[1], dtype=np.float64)
    for pair in range(theta.shape[1]):
        columns = np.arange(2 * pair, 2 * pair + 2)
        target = design[:, columns]
        others = design[:, np.delete(dimensions, columns)]
        coefficient, *_ = np.linalg.lstsq(others, target, rcond=float(ridge))
        with np.errstate(all="ignore"):
            residual = target - others @ coefficient
        conditional = float((residual ** 2).sum()) / 2.0
        marginal = float((target ** 2).sum()) / 2.0
        values[pair] = 0.0 if marginal <= 0.0 else conditional / marginal
    return np.clip(values, 0.0, 1.0)


def uniqueness_budgeted_table(
    native: np.ndarray,
    uniqueness: np.ndarray,
    *,
    factor: float,
    exponent: float = 2.0,
) -> np.ndarray:
    if not math.isfinite(float(factor)) or float(factor) <= 1.0:
        raise ValueError("factor must be finite and greater than one")
    if not math.isfinite(float(exponent)) or float(exponent) <= 0.0:
        raise ValueError("exponent must be finite and positive")
    source = np.asarray(native, dtype=np.float64).reshape(-1)
    values = np.asarray(uniqueness, dtype=np.float64).reshape(-1)
    if values.shape != source.shape:
        raise ValueError("uniqueness must match the Native table")
    span = float(values.max()) - float(values.min())
    normalized = (
        np.zeros_like(values)
        if span <= 0.0
        else (values - float(values.min())) / span
    )
    movement = (1.0 - normalized) ** float(exponent)
    target = source * (1.0 - movement) + (source / float(factor)) * movement
    result = np.ascontiguousarray(target, dtype="<f4")
    if (
        not np.isfinite(result).all()
        or not (result > 0.0).all()
        or not np.all(result[:-1] > result[1:])
    ):
        raise RuntimeError("derived table is not finite, positive, and decreasing")
    return result


def build_default_tables() -> dict[float, np.ndarray]:
    native = native_endpoint_inv_freq()
    support, weight = causal_distance_measure()
    uniqueness = conditional_pair_uniqueness(native, support, weight)
    return {
        factor: uniqueness_budgeted_table(
            native,
            uniqueness,
            factor=factor,
            exponent=2.0,
        )
        for factor in sorted(EXPECTED_DEFAULT_HASHES)
    }


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = {}
    for factor, table in build_default_tables().items():
        digest = float32_sha256(table)
        if digest != EXPECTED_DEFAULT_HASHES[factor]:
            raise RuntimeError(
                f"frozen s={factor:g} table hash drift: {digest}"
            )
        path = output / f"budgeted_s{factor:g}_p2.npy"
        _atomic_npy(path, table)
        records[str(int(factor))] = {
            "factor": factor,
            "path": path.name,
            "sha256_float32": digest,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "status": "UNIQUENESS_BUDGETED_TABLES_EXPORTED",
        "native_length": 4096,
        "support_points": 2048,
        "distance_measure": "causal pair count",
        "exponent": 2.0,
        "tables": records,
    }
    receipt_path = output / "receipt.json"
    with tempfile.NamedTemporaryFile(
        dir=output,
        prefix="receipt.json.",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
