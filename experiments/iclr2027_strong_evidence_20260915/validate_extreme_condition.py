#!/usr/bin/env python3
"""Validate one long-context condition before any GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913 import tables


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--rows-per-task", type=int, default=5)
    parser.add_argument("--ppl-documents", type=int, default=5)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    geometry = tables.model_geometry(config)
    native = int(geometry["native_length"])
    if args.target != native * args.scale:
        raise ValueError(f"target/native mismatch: {args.target}/{native} != S={args.scale}")

    assets = json.loads((args.root / "assets/manifest.json").read_text())
    panel_record = assets.get("panels", {}).get(str(args.target), {})
    panel_path = args.root / "assets" / str(panel_record.get("inputs", ""))
    panel_manifest_path = args.root / "assets" / str(panel_record.get("manifest", ""))
    panel_manifest = json.loads(panel_manifest_path.read_text())
    expected_tasks = [
        "niah_single_1", "niah_single_2", "niah_single_3",
        "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multivalue", "niah_multiquery",
    ]
    expected_rows = len(expected_tasks) * args.rows_per_task
    if (
        assets.get("status") != "COMPLETE"
        or assets.get("model_id") != args.model_id
        or float(assets.get("scale", float("nan"))) != args.scale
        or assets.get("lengths") != [args.target]
        or assets.get("rows_per_task") != args.rows_per_task
        or assets.get("tasks") != expected_tasks
        or panel_record.get("rows") != expected_rows
        or panel_manifest.get("rows") != expected_rows
        or panel_manifest.get("tasks") != expected_tasks
        or panel_manifest.get("length_cap") != args.target
        or sha256(panel_path) != panel_record.get("inputs_sha256")
        or sha256(panel_manifest_path) != panel_record.get("manifest_sha256")
    ):
        raise ValueError("clean generation assets do not match the condition")

    ppl = json.loads((args.root / "ppl/manifest.json").read_text())
    lm_path = args.root / "ppl/lm.npy"
    array = np.load(lm_path, mmap_mode="r", allow_pickle=False)
    if (
        ppl.get("status") != "COMPLETE"
        or ppl.get("model_id") != args.model_id
        or ppl.get("lengths") != [args.target]
        or ppl.get("documents") != args.ppl_documents
        or list(array.shape) != [args.ppl_documents, args.target + 1]
        or str(array.dtype) != "int64"
        or sha256(lm_path) != ppl.get("lm_array_sha256")
    ):
        raise ValueError("PPL assets do not match the condition")

    table_hashes = {}
    expected_gain = 1.0 + 0.1 * math.log(args.scale)
    expected_band = list(tables.default_band(geometry))
    for arm, method, role in (
        ("tailspline", "tailspline", "candidate"),
        ("mrpro", "mrpro", "baseline"),
    ):
        path = args.root / f"tables/{arm}.json"
        receipt = json.loads(path.read_text())
        actual, gain = tables.validate_table(
            tables.find_table(receipt), pairs=int(geometry["pairs"]),
        )
        expected, expected_method_gain, _ = tables.build_analytic(
            config, method=method, scale=float(args.scale), low=None, high=None,
            depth=1.0, gain=None,
        )
        if (
            receipt.get("status") != TABLE_FORMAT
            or receipt.get("candidate_id") != f"extreme_{args.model_id}_s{args.scale}_{arm}"
            or receipt.get("model_id") != args.model_id
            or receipt.get("role") != role
            or float(receipt.get("scale", float("nan"))) != args.scale
            or receipt.get("band_envelope") != expected_band
            or receipt.get("changed_variables") != ["internal_frequency_allocation"]
            or not np.array_equal(actual, expected)
            or gain != expected_method_gain
            or gain != expected_gain
            or receipt.get("table_sha256_float32") != tables.tensor_sha256(actual)
        ):
            raise ValueError(f"{arm} table is not the exact S={args.scale} construction")
        table_hashes[arm] = receipt["table_sha256_float32"]

    ready = {
        "status": "EXTREME_CONDITION_READY_V1",
        "model_id": args.model_id,
        "model_artifact_name": args.model.name,
        "native_length": native,
        "target_length": args.target,
        "scale": args.scale,
        "identity_check": f"{args.target}/{native}={args.scale}",
        "band": expected_band,
        "gain": expected_gain,
        "ruler_niah8_asset_rows": expected_rows,
        "gpu_task_subset": {
            "tasks": expected_tasks,
            "rows_per_task": args.rows_per_task,
            "rows_per_arm": 8 * args.rows_per_task,
        },
        "ppl_documents_per_arm": args.ppl_documents,
        "panel_sha256": sha256(panel_path),
        "lm_array_sha256": sha256(lm_path),
        "table_sha256_float32": table_hashes,
        "gpu_execution": False,
    }
    output = args.root / "ready.json"
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(ready, sort_keys=True))


if __name__ == "__main__":
    main()
