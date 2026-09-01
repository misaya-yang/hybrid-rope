#!/usr/bin/env python3
"""Build the three frozen tables for the dilation-distribution fork."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_float32(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


def maxent_dilation(q: np.ndarray, factor: float, lambda_: float) -> np.ndarray:
    if abs(lambda_) < 1e-10:
        return np.power(factor, q)
    return np.power(1.0 + q * (factor**lambda_ - 1.0), 1.0 / lambda_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--frequency-reference-table", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=float, default=500_000.0)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_901)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output-dir must be new")
    if args.head_dim < 8 or args.head_dim % 2 or args.factor <= 1.0:
        raise ValueError("head-dim/factor identity is invalid")

    count = args.head_dim // 2
    native = np.power(
        float(args.base),
        -np.arange(count, dtype=np.float64) / count,
    )
    reference = np.load(args.reference_table, allow_pickle=False).astype(np.float64)
    if reference.shape != native.shape or not np.all(reference > 0.0):
        raise RuntimeError("reference table identity drift")
    reference_r = native / reference

    permutation = np.arange(count)
    rng = np.random.default_rng(args.seed)
    permutation[1:-1] = rng.permutation(permutation[1:-1])
    permuted_r = reference_r[permutation]
    permuted_r[0], permuted_r[-1] = 1.0, float(args.factor)

    q = np.linspace(0.0, 1.0, count, dtype=np.float64)
    dilation = {
        "a1_permuted": permuted_r,
        "a2_haar": maxent_dilation(q, float(args.factor), 0.0),
        "a3_maxent": maxent_dilation(q, float(args.factor), float(args.lambda_)),
    }
    args.output_dir.mkdir(parents=True)
    reports = {}
    for name, r in dilation.items():
        table = np.ascontiguousarray(native / r, dtype="<f4")
        np.save(args.output_dir / f"{name}.npy", table, allow_pickle=False)
        reports[name] = {
            "table_sha256_float32": sha256_float32(table),
            "minimum_dilation": float(r.min()),
            "maximum_dilation": float(r.max()),
            "frequency_endpoints": [float(table[0]), float(table[-1])],
            "order_crossing_indices": np.flatnonzero(table[:-1] <= table[1:]).tolist(),
        }

    if args.frequency_reference_table is not None:
        frequency_reference = np.load(
            args.frequency_reference_table,
            allow_pickle=False,
        ).astype(np.float32)
        if frequency_reference.shape != native.shape:
            raise RuntimeError("frequency reference table identity drift")
        frequency_permuted = frequency_reference.copy()
        frequency_permuted[1:-1] = frequency_reference[permutation[1:-1]]
        np.save(
            args.output_dir / "a4_frequency_permuted.npy",
            frequency_permuted,
            allow_pickle=False,
        )
        if not np.array_equal(
            np.sort(frequency_permuted[1:-1]),
            np.sort(frequency_reference[1:-1]),
        ):
            raise RuntimeError("A4 changed the interior frequency multiset")
        reports["a4_frequency_permuted"] = {
            "table_sha256_float32": sha256_float32(frequency_permuted),
            "reference_table_sha256_float32": sha256_float32(frequency_reference),
            "frequency_endpoints": [
                float(frequency_permuted[0]),
                float(frequency_permuted[-1]),
            ],
            "order_crossing_indices": np.flatnonzero(
                frequency_permuted[:-1] <= frequency_permuted[1:]
            ).tolist(),
            "interior_frequency_multiset_preserved_exactly": True,
        }

    if not np.allclose(
        np.sort(permuted_r[1:-1]),
        np.sort(reference_r[1:-1]),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("A1 changed the interior dilation multiset")
    if any(
        not math.isclose(float(r[0]), 1.0) or not math.isclose(float(r[-1]), args.factor)
        for r in dilation.values()
    ):
        raise RuntimeError("dilation endpoints drift")
    if any(
        reports[name]["order_crossing_indices"]
        for name in ("a2_haar", "a3_maxent")
    ):
        raise RuntimeError("ordered allocation produced frequency crossings")

    receipt = {
        "status": "DILATION_ALLOCATION_FORK_READY_V1",
        "question": "distribution-only allocation or slot-dilation coupling",
        "reference_table": str(args.reference_table.resolve()),
        "reference_table_sha256_float32": sha256_float32(reference),
        "factor": float(args.factor),
        "lambda_selected_from_1x_only": float(args.lambda_),
        "quantiles": "q_i=i/(K-1); exact dilation endpoints 1 and s",
        "a1_permutation_seed": int(args.seed),
        "a1_permutation": permutation.tolist(),
        "tables": reports,
    }
    if args.frequency_reference_table is not None:
        receipt["a4_preregistration"] = {
            "H1": "slot/readout coupling: same physical frequencies in different slots collapse",
            "H2": "unordered physical-spectrum geometry: same physical frequencies remain near parity",
            "reference_1x_pg19_nll": 3.104233813285828,
            "reference_16k_core4": 0.3375,
            "support_H2": "abs(delta_1x_nll)<=0.01 and abs(delta_16k_core4)<=0.05",
            "support_H1": "delta_1x_nll>=0.10 or candidate_16k_core4<=0.10",
            "otherwise": "unresolved; do not add permutations or tasks",
            "run_order": "1x PG19 first; open 16K core4 only when 1x is near parity",
        }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
