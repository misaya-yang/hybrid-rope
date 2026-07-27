#!/usr/bin/env python3
"""Create the no-GPU READY receipt for post-exact 4K retention."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .evaluate_instruct_ruler_transfer import (
    RETENTION_READY_STATUS,
    _validate_data,
    bound_code_sha256 as evaluator_bound_code_sha256,
)
from .gate_olmo2_4k_retention import (
    IMMEDIATE_PARENT_SHA256,
    MODEL_SHA256,
    PRE_QUERY_GAP_PARENT_SHA256,
    TASKS,
)
from .prepare_data import atomic_json, sha256_file
from .train_4k_stage_a import ready_checkpoint_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--exact-gate", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--immediate-parent-adapter", type=Path, required=True)
    parser.add_argument("--pre-parent-adapter", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--immediate-parent-output", type=Path, required=True)
    parser.add_argument("--pre-parent-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def adapter_entry(path: Path, expected_sha256: str) -> dict[str, Any]:
    entry = file_entry(path)
    if entry["sha256"] != expected_sha256:
        raise RuntimeError("retention adapter SHA drift")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise RuntimeError("retention adapter payload drift")
    entry["metadata"] = payload["metadata"]
    entry["state_tensors"] = len(payload["state"])
    return entry


def main() -> None:
    args = parse_args()
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "checkpoint",
            "ready_receipt",
            "data_root",
            "training_result",
            "exact_gate",
            "candidate_adapter",
            "immediate_parent_adapter",
            "pre_parent_adapter",
            "candidate_output",
            "immediate_parent_output",
            "pre_parent_output",
            "gate_output",
            "receipt_output",
        )
    }
    for name in (
        "candidate_output",
        "immediate_parent_output",
        "pre_parent_output",
        "gate_output",
        "receipt_output",
    ):
        if paths[name].exists():
            raise FileExistsError(paths[name])

    checkpoint_digest = ready_checkpoint_digest(
        paths["checkpoint"],
        paths["ready_receipt"],
    )
    if checkpoint_digest != MODEL_SHA256:
        raise RuntimeError("retention checkpoint is not OLMo-2 1.485B")
    training = json.loads(
        paths["training_result"].read_text(encoding="utf-8")
    )
    exact_gate = json.loads(
        paths["exact_gate"].read_text(encoding="utf-8")
    )
    if (
        training.get("status")
        != "OLMO2_4K_QUERY_GAP_EOS_REPAIR_COMPLETE_V1"
        or exact_gate.get("status") != "PASS"
        or exact_gate.get("gate")
        != "OLMO2_MINIMAL_FULL_STRING_EXACT_EOS_V1"
        or exact_gate.get("expanded_evaluation_authorized") is not True
    ):
        raise RuntimeError(
            "broad 4K retention is forbidden before minimal exact passes"
        )
    candidate_sha = str(training.get("adapter_sha256"))
    if training.get("parent_adapter_sha256") != IMMEDIATE_PARENT_SHA256:
        raise RuntimeError("retention candidate lineage drift")
    adapters = {
        "candidate": adapter_entry(
            paths["candidate_adapter"], candidate_sha
        ),
        "immediate_parent": adapter_entry(
            paths["immediate_parent_adapter"],
            IMMEDIATE_PARENT_SHA256,
        ),
        "pre_query_gap_parent": adapter_entry(
            paths["pre_parent_adapter"],
            PRE_QUERY_GAP_PARENT_SHA256,
        ),
    }
    if (
        adapters["candidate"]["metadata"].get("parent_adapter_sha256")
        != IMMEDIATE_PARENT_SHA256
        or adapters["immediate_parent"]["metadata"].get(
            "parent_adapter_sha256"
        )
        != PRE_QUERY_GAP_PARENT_SHA256
    ):
        raise RuntimeError("retention adapter metadata lineage drift")

    data_receipt, selected_rows = _validate_data(
        root=paths["data_root"],
        checkpoint=paths["checkpoint"],
        requested_tasks=TASKS,
        requested_lengths=(4_096,),
        limit_per_cell=20,
    )
    if len(selected_rows) != len(TASKS) * 20:
        raise RuntimeError("retention 13-task row-count drift")
    module_root = Path(__file__).resolve().parent
    evaluator = module_root / "evaluate_instruct_ruler_transfer.py"
    gate = module_root / "gate_olmo2_4k_retention.py"
    outputs = {
        "candidate": paths["candidate_output"],
        "immediate_parent": paths["immediate_parent_output"],
        "pre_query_gap_parent": paths["pre_parent_output"],
    }
    adapter_paths = {
        "candidate": paths["candidate_adapter"],
        "immediate_parent": paths["immediate_parent_adapter"],
        "pre_query_gap_parent": paths["pre_parent_adapter"],
    }

    def evaluation_argv(role: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            (
                "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
                "evaluate_instruct_ruler_transfer"
            ),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--ready-receipt",
            str(paths["ready_receipt"]),
            "--experiment-ready-receipt",
            str(paths["receipt_output"]),
            "--experiment-role",
            role,
            "--data-root",
            str(paths["data_root"]),
            "--output",
            str(outputs[role]),
            "--frequency",
            "evq",
            "--adapter",
            str(adapter_paths[role]),
            "--rank",
            "64",
            "--alpha",
            "128",
            "--tasks",
            *TASKS,
            "--lengths",
            "4096",
            "--limit-per-cell",
            "20",
        ]

    receipt = {
        "status": RETENTION_READY_STATUS,
        "objective": (
            "After exact+EOS admission, test whether the repair adds no "
            "catastrophic 4K forgetting relative to a0ccd and whether the "
            "full query-gap pipeline retains 4K capability relative to 95ce."
        ),
        "scope_boundary": (
            "This is a fixed 13-task 4K RULER and matched natural-NLL gate; "
            "it is not global retention relative to untouched OLMo-2."
        ),
        "inputs": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "composite_sha256": checkpoint_digest,
                "ready_receipt_sha256": sha256_file(
                    paths["ready_receipt"]
                ),
            },
            "data": {
                "path": str(paths["data_root"]),
                "manifest_sha256": data_receipt["manifest_sha256"],
                "tokenizer_sha256": data_receipt["tokenizer_sha256"],
                "ruler_commit": data_receipt["ruler_commit"],
                "cells": {
                    task: data_receipt["cells"][task]["4096"]
                    for task in TASKS
                },
            },
            "training_result": file_entry(paths["training_result"]),
            "exact_gate": file_entry(paths["exact_gate"]),
        },
        "registered_adapters": adapters,
        "registered_outputs": {
            role: str(path) for role, path in outputs.items()
        },
        "evaluator": file_entry(evaluator),
        "evaluator_bound_code_sha256": evaluator_bound_code_sha256(),
        "gate": file_entry(gate),
        "registered_working_directory": str(
            Path(__file__).resolve().parents[4]
        ),
        "required_execution_order": [
            "candidate",
            "immediate_parent",
            "pre_query_gap_parent",
            "retention_gate",
        ],
        "registered_commands": {
            role: evaluation_argv(role) for role in outputs
        },
        "protocol": {
            "tasks": list(TASKS),
            "lengths": [4_096],
            "rows_per_task": 20,
            "arms": list(outputs),
            "greedy": True,
            "primary_metric": "official_task_score",
            "row_pairing_required": True,
            "task_delta_floor": -0.10,
            "family_delta_floor": -0.05,
            "global_delta_floor": -0.05,
            "maximum_task_lost_mass": 2.0,
            "maximum_excess_loss_rows": 2,
            "minimum_retained_parent_mass": 0.80,
            "natural_nll_delta_ceiling": 0.10,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "authorization": (
            "READY is offline evidence only and does not authorize GPU use."
        ),
    }
    receipt["registered_commands"]["retention_gate"] = [
        sys.executable,
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "gate_olmo2_4k_retention"
        ),
        "--training-result",
        str(paths["training_result"]),
        "--exact-gate",
        str(paths["exact_gate"]),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--candidate-result",
        str(paths["candidate_output"] / "results.json"),
        "--candidate-examples",
        str(paths["candidate_output"] / "examples.jsonl"),
        "--immediate-parent-result",
        str(paths["immediate_parent_output"] / "results.json"),
        "--immediate-parent-examples",
        str(paths["immediate_parent_output"] / "examples.jsonl"),
        "--pre-parent-result",
        str(paths["pre_parent_output"] / "results.json"),
        "--pre-parent-examples",
        str(paths["pre_parent_output"] / "examples.jsonl"),
        "--output",
        str(paths["gate_output"]),
    ]
    atomic_json(paths["receipt_output"], receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(paths["receipt_output"]),
                "receipt_sha256": sha256_file(paths["receipt_output"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
