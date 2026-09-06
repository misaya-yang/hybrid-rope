#!/usr/bin/env python3
"""Build one target-independent, Native-support anchored EVQ-Cosh table."""

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
from scripts.lib.rope.schedules import evq_cosh_phi


STATUS = "ZERO_PARAMETER_NATIVE_SUPPORT_ANCHORED_EVQ_COSH_V1"


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


def build_table(tau: float) -> tuple[np.ndarray, dict[str, object]]:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be finite and positive")
    native = endpoint_geo_inv_freq().detach().cpu().to(torch.float32).numpy()
    phi = evq_cosh_phi(
        native.size,
        tau=float(tau),
        midpoint=False,
        dtype=torch.float64,
    ).numpy()
    z = (phi - float(phi[0])) / (float(phi[-1]) - float(phi[0]))
    native_x = -np.log(native.astype(np.float64))
    active_x = float(native_x[0]) + (float(native_x[-1]) - float(native_x[0])) * z
    table = np.exp(-active_x).astype("<f4")
    table[0] = np.float32(native[0])
    table[-1] = np.float32(native[-1])
    if (
        table.shape != (64,)
        or not np.isfinite(table).all()
        or not np.all(table[:-1] > table[1:])
        or not np.array_equal(table[[0, -1]], native[[0, -1]])
    ):
        raise RuntimeError("anchored EVQ-Cosh table identity failed")
    receipt = {
        "status": STATUS,
        "construction": "endpoint-grid EVQ-Cosh coordinates normalized to exact Native sampled support",
        "tau": float(tau),
        "pair_count": int(table.size),
        "model_profile_inputs": ["Native inverse-frequency tensor"],
        "uses_L_target": False,
        "uses_ood_labels": False,
        "uses_attention_prior": False,
        "uses_collision_objective": False,
        "learned_parameters": 0,
        "optimizer_steps": 0,
        "attention_scaling": 1.0,
        "native_sha256_float32": float32_sha256(native),
        "active_sha256_float32": float32_sha256(table),
        "fixed_native_sampled_support": True,
        "candidate_differs_from_native": bool(not np.array_equal(table, native)),
        "maximum_normalized_coordinate_shift": float(
            np.max(np.abs(z - np.linspace(0.0, 1.0, table.size)))
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
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    table_path = args.output_table.expanduser().resolve()
    receipt_path = args.output_receipt.expanduser().resolve()
    if table_path.exists() or receipt_path.exists():
        raise FileExistsError("zero-parameter table outputs must use new paths")
    table, receipt = build_table(float(args.tau))
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
