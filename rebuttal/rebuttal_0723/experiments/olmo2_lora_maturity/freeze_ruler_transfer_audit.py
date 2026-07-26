#!/usr/bin/env python3
"""Freeze the matched 4K held-out RULER transfer audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .prepare_data import atomic_json, sha256_file


DATA_RELATIVE = Path("data/ruler_olmo2_transfer_v3_4k_s20260727")
ARM_RUNS = {
    "untouched_native": "instruct_untouched_native_ruler_transfer_4k_v1",
    "native_lora_final": (
        "instruct_native_lora_ruler_transfer_4k_s20260725_v1"
    ),
    "evq_unadapted": "instruct_evq_unadapted_ruler_transfer_4k_v1",
    "evq_stage_a": (
        "instruct_evq_stage_a_ruler_transfer_4k_s20260725_v1"
    ),
    "evq_lora_final": (
        "instruct_evq_ruler_transfer_4k_s20260725_v1"
    ),
}
CODE_RELATIVES = (
    Path(
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "prepare_instruct_ruler_transfer.py"
    ),
    Path(
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "evaluate_instruct_ruler_transfer.py"
    ),
    Path(
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "freeze_ruler_transfer_audit.py"
    ),
)
TASKS = ("niah_multikey_3", "vt")
ARMS = tuple(ARM_RUNS)
EXPECTED_RESULT_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_COMPLETE"
EXPECTED_DATA_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_PREPARED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _row_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    mapped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["local_index"]))
        if key in mapped:
            raise RuntimeError(f"duplicate row {key}")
        mapped[key] = row
    return mapped


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data_root = base / DATA_RELATIVE
    data_manifest_path = data_root / "manifest.json"
    data_manifest = _read_json(data_manifest_path)
    if data_manifest.get("status") != EXPECTED_DATA_STATUS:
        raise RuntimeError("transfer data status drift")
    if tuple(data_manifest["tasks"]) != TASKS:
        raise RuntimeError("transfer task drift")
    if data_manifest["lengths"] != [4_096]:
        raise RuntimeError("transfer length drift")
    if int(data_manifest["samples_per_cell"]) != 20:
        raise RuntimeError("transfer sample-count drift")
    data_manifest_sha = sha256_file(data_manifest_path)

    run_receipts: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    canonical_row_hashes: dict[str, list[str]] = {}
    for arm, run_name in ARM_RUNS.items():
        run = base / "runs" / run_name
        results_path = run / "results.json"
        examples_path = run / "examples.jsonl"
        results = _read_json(results_path)
        rows = _read_jsonl(examples_path)
        if results.get("status") != EXPECTED_RESULT_STATUS:
            raise RuntimeError(f"{arm} result status drift")
        if results["data"]["manifest_sha256"] != data_manifest_sha:
            raise RuntimeError(f"{arm} data manifest drift")
        protocol = results["protocol"]
        if (
            tuple(protocol["tasks"]) != TASKS
            or protocol["lengths"] != [4_096]
            or int(protocol["limit_per_cell"]) != 20
        ):
            raise RuntimeError(f"{arm} protocol drift")
        if len(rows) != 40:
            raise RuntimeError(f"{arm} example-count drift")
        mapped = _row_map(rows)
        hashes = {
            task: [
                str(mapped[(task, index)]["row_sha256"])
                for index in range(20)
            ]
            for task in TASKS
        }
        canonical_row_hashes[arm] = [
            value for task in TASKS for value in hashes[task]
        ]
        run_receipts[arm] = {
            "run": str(run),
            "results_sha256": sha256_file(results_path),
            "examples_sha256": sha256_file(examples_path),
            "frequency": results["frequency"]["active_frequency"],
            "frequency_sha256_float32": results["frequency"][
                "active_sha256_float32"
            ],
            "adapter_sha256": (
                results["adapter"]["sha256"]
                if results["adapter"] is not None
                else None
            ),
            "cells": results["results"]["cells"],
            "row_sha256": hashes,
        }
        raw_rows[arm] = rows

    reference_hashes = canonical_row_hashes["untouched_native"]
    for arm in ARMS[1:]:
        if canonical_row_hashes[arm] != reference_hashes:
            raise RuntimeError(f"{arm} evaluation-row order drift")

    paired: dict[str, dict[str, Any]] = {}
    maps = {arm: _row_map(raw_rows[arm]) for arm in ARMS}
    for task in TASKS:
        task_pairs: dict[str, Any] = {}
        for arm in ARMS[1:]:
            base_scores = [
                float(
                    maps["untouched_native"][(task, index)][
                        "official_string_match_all"
                    ]
                )
                for index in range(20)
            ]
            arm_scores = [
                float(
                    maps[arm][(task, index)][
                        "official_string_match_all"
                    ]
                )
                for index in range(20)
            ]
            task_pairs[f"untouched_native_vs_{arm}"] = {
                "mean_delta_arm_minus_untouched": (
                    sum(arm_scores) - sum(base_scores)
                )
                / 20,
                "untouched_higher_rows": sum(
                    base_score > arm_score
                    for base_score, arm_score in zip(
                        base_scores, arm_scores, strict=True
                    )
                ),
                "arm_higher_rows": sum(
                    arm_score > base_score
                    for base_score, arm_score in zip(
                        base_scores, arm_scores, strict=True
                    )
                ),
                "tied_rows": sum(
                    arm_score == base_score
                    for base_score, arm_score in zip(
                        base_scores, arm_scores, strict=True
                    )
                ),
            }
        paired[task] = task_pairs

    summary = {
        "status": "OLMO2_4K_RULER_TRANSFER_AUDIT_FROZEN",
        "concerns": ["R27bE.2", "R27bE.5", "AC.2", "AC.4"],
        "question": (
            "Does the 4K-only EVQ conversion recipe retain or transfer "
            "capability to RULER tasks not used for adaptation?"
        ),
        "metric_boundary": (
            "Official autoregressive RULER string_match_all at 4K. "
            "niah_multikey_3 changes key/value type and distractor structure; "
            "variable tracking is a different task. This is a 20-row screen, "
            "not a complete RULER score."
        ),
        "data": {
            "manifest_sha256": data_manifest_sha,
            "seed": int(data_manifest["seed"]),
            "tasks": list(TASKS),
            "length": 4_096,
            "rows_per_task": 20,
            "ruler_commit": data_manifest["ruler_commit"],
        },
        "arms": run_receipts,
        "paired_against_untouched_native": paired,
        "registered_gate": {
            "rule": (
                "Do not evaluate a task at 8K when the EVQ final adapter "
                "scores below 0.50 at 4K."
            ),
            "evq_final_scores": {
                task: float(
                    run_receipts["evq_lora_final"]["cells"][task]["4096"][
                        "official_string_match_all"
                    ]
                )
                for task in TASKS
            },
            "decision": "STOP_8K_FOR_BOTH_TASKS",
        },
        "interpretation": [
            (
                "The untouched Native checkpoint retains nonzero 4K "
                "competence: 0.55 on UUID distractor retrieval and 0.25 "
                "mean reference recall on variable tracking."
            ),
            (
                "Native-LoRA declines to 0.40 and 0.15, showing a smaller "
                "generic curriculum narrowing effect."
            ),
            (
                "EVQ injection, EVQ Stage A, and final EVQ-LoRA each score "
                "0.00 on both tasks. Stage A's strong natural-text NLL does "
                "not restore these capabilities; Stage B recovers only the "
                "trained numeric single-needle task."
            ),
            (
                "Therefore the frozen 69/100 and 67/100 8K result remains "
                "valid single-task conversion evidence but is not broad "
                "downstream transfer or no-forgetting evidence."
            ),
        ],
        "parent_freezes": {
            "matched_native_evq_inventory_sha256": (
                "717b894e6924043165b1b06f292f2f0c8e30532b942e4c0fd93cf9c16544ef2d"
            ),
            "matched_8k100_inventory_sha256": (
                "5fd00a1ab7f35cbd279d6fc0c9c2702586e783905c2e873b101132b12b33769b"
            ),
            "evq_replication_inventory_sha256": (
                "b2a997ce74ce9ec929f13182d72a98cf9204535a2d101a1b14c7ea78f901ac7e"
            ),
        },
    }

    output.mkdir(parents=True)
    _copy(
        data_manifest_path,
        output / "artifacts" / DATA_RELATIVE / "manifest.json",
    )
    for task in TASKS:
        relative = Path(
            data_manifest["cells"][task]["4096"]["relative_path"]
        )
        _copy(
            data_root / relative,
            output / "artifacts" / DATA_RELATIVE / relative,
        )
    for arm, run_name in ARM_RUNS.items():
        for name in ("results.json", "examples.jsonl"):
            _copy(
                base / "runs" / run_name / name,
                output / "artifacts" / "runs" / run_name / name,
            )
    for relative in CODE_RELATIVES:
        _copy(base / "code" / relative, output / "code" / relative)
    atomic_json(output / "transfer_summary.json", summary)

    files = []
    for path in sorted(
        candidate
        for candidate in output.rglob("*")
        if candidate.is_file()
    ):
        files.append(
            {
                "relative_path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory = {
        "status": "OLMO2_4K_RULER_TRANSFER_FILE_INVENTORY",
        "base": str(base),
        "files": files,
    }
    atomic_json(output / "inventory.json", inventory)
    receipt = {
        "status": "OLMO2_4K_RULER_TRANSFER_EVIDENCE_FROZEN",
        "output": str(output),
        "file_count": len(files),
        "inventory_sha256": sha256_file(output / "inventory.json"),
    }
    atomic_json(output / "FREEZE_RECEIPT.json", receipt)
    for path in output.rglob("*"):
        path.chmod(path.stat().st_mode & ~0o222)
    output.chmod(output.stat().st_mode & ~0o222)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
