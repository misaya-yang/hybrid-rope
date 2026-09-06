#!/usr/bin/env python3
"""Build a target-independent protected-band anchored EVQ-Cosh table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_geo_inv_freq,
)


STATUS = "ZERO_PARAMETER_NATIVE_PROTECTED_BAND_COSH_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f4").tobytes()
    ).hexdigest()


def cosh_warp(value: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 - np.arcsinh((1.0 - value) * math.sinh(tau)) / tau


def build_table(
    *,
    tau: float,
    native_length: int,
    minimum_wavelength_ratio: float,
    maximum_wavelength_ratio: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    if native_length <= 0:
        raise ValueError("native_length must be positive")
    if not 0.0 < minimum_wavelength_ratio < maximum_wavelength_ratio:
        raise ValueError("protected wavelength ratios are invalid")
    native = endpoint_geo_inv_freq().detach().cpu().to(torch.float32).numpy()
    wavelengths = 2.0 * math.pi / native.astype(np.float64)
    protected = np.flatnonzero(
        (wavelengths >= minimum_wavelength_ratio * native_length)
        & (wavelengths <= maximum_wavelength_ratio * native_length)
    ).astype(int)
    if protected.size == 0:
        raise RuntimeError("analytic protected band selected no rotary pairs")
    anchors = np.unique(np.concatenate(([0], protected, [native.size - 1]))).astype(int)
    native_z = np.linspace(0.0, 1.0, native.size, dtype=np.float64)
    active_z = native_z.copy()
    for left, right in zip(anchors[:-1], anchors[1:]):
        count = int(right - left - 1)
        if count <= 0:
            continue
        local_u = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
        active_z[left + 1 : right] = (
            native_z[left]
            + (native_z[right] - native_z[left]) * cosh_warp(local_u, tau)
        )
    native_x = -np.log(native.astype(np.float64))
    active_x = float(native_x[0]) + (float(native_x[-1]) - float(native_x[0])) * active_z
    table = np.exp(-active_x).astype("<f4")
    table[anchors] = native[anchors]
    if (
        table.shape != (64,)
        or not np.isfinite(table).all()
        or not np.all(table[:-1] > table[1:])
        or not np.array_equal(table[anchors], native[anchors])
    ):
        raise RuntimeError("protected-band table identity failed")
    receipt = {
        "status": STATUS,
        "construction": "exact Native O(L_native) wavelength band; local anchored EVQ-Cosh allocation in each complementary gap",
        "tau": float(tau),
        "native_length": int(native_length),
        "protected_wavelength_ratio": [
            float(minimum_wavelength_ratio),
            float(maximum_wavelength_ratio),
        ],
        "protected_pair_indices": protected.tolist(),
        "exact_native_anchor_indices": anchors.tolist(),
        "learned_parameters": 0,
        "optimizer_steps": 0,
        "uses_L_target": False,
        "uses_ood_labels": False,
        "uses_attention_prior": False,
        "uses_collision_objective": False,
        "attention_scaling": 1.0,
        "native_sha256_float32": float32_sha256(native),
        "active_sha256_float32": float32_sha256(table),
        "fixed_native_sampled_support": True,
        "candidate_differs_from_native": bool(not np.array_equal(table, native)),
        "maximum_normalized_coordinate_shift": float(
            np.max(np.abs(active_z - native_z))
        ),
        "minimum_adjacent_log_gap": float(
            np.min(np.diff(-np.log(table.astype(np.float64))))
        ),
    }
    return table, receipt


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--minimum-wavelength-ratio", type=float, default=1.0)
    parser.add_argument("--maximum-wavelength-ratio", type=float, default=4.0)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    table_path = args.output_table.expanduser().resolve()
    receipt_path = args.output_receipt.expanduser().resolve()
    if table_path.exists() or receipt_path.exists():
        raise FileExistsError("protected-band outputs must use new paths")
    table, receipt = build_table(
        tau=float(args.tau),
        native_length=int(args.native_length),
        minimum_wavelength_ratio=float(args.minimum_wavelength_ratio),
        maximum_wavelength_ratio=float(args.maximum_wavelength_ratio),
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=table_path.parent,
        prefix=table_path.name + ".",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, table, allow_pickle=False)
        temporary.replace(table_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt.update({
        "table_file_sha256": sha256_file(table_path),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
    })
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
