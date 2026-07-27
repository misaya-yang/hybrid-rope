#!/usr/bin/env python3
"""Create a no-GPU READY receipt for direct-parent 4K retention."""

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
from .gate_olmo2_4k_direct_retention import GATE_NAME
from .gate_olmo2_4k_retention import (
    EXACT_GATE_NAME,
    MODEL_SHA256,
    TASKS,
    TRAINING_STATUS,
)
from .prepare_data import atomic_json, sha256_file
from .preflight_olmo2_4k_retention import adapter_entry, file_entry
from .train_4k_stage_a import ready_checkpoint_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--exact-gate", type=Path, required=True)
    parser.add_argument(
        "--frequency",
        choices=("native", "evq"),
        required=True,
    )
    parser.add_argument(
        "--allow-exact-stop-diagnostic",
        action="store_true",
        help=(
            "Register retention as a user-authorized diagnostic after an "
            "exact-capability STOP; this does not promote capability."
        ),
    )
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        "checkpoint",
        "ready_receipt",
        "data_root",
        "training_result",
        "exact_gate",
        "candidate_adapter",
        "parent_adapter",
        "candidate_output",
        "parent_output",
        "gate_output",
        "receipt_output",
    )
    paths = {name: getattr(args, name).resolve() for name in names}
    for name in (
        "candidate_output",
        "parent_output",
        "gate_output",
        "receipt_output",
    ):
        if paths[name].exists():
            raise FileExistsError(paths[name])

    checkpoint_sha256 = ready_checkpoint_digest(
        paths["checkpoint"], paths["ready_receipt"]
    )
    if checkpoint_sha256 != MODEL_SHA256:
        raise RuntimeError("direct retention checkpoint identity drift")
    training = json.loads(
        paths["training_result"].read_text(encoding="utf-8")
    )
    exact_gate = json.loads(
        paths["exact_gate"].read_text(encoding="utf-8")
    )
    exact_pass = (
        exact_gate.get("status") == "PASS"
        and exact_gate.get("expanded_evaluation_authorized") is True
    )
    exact_stop_diagnostic = (
        bool(args.allow_exact_stop_diagnostic)
        and exact_gate.get("status") == "STOP"
        and exact_gate.get("expanded_evaluation_authorized") is False
    )
    if (
        training.get("status") != TRAINING_STATUS
        or exact_gate.get("gate") != EXACT_GATE_NAME
        or not (exact_pass or exact_stop_diagnostic)
    ):
        raise RuntimeError("direct retention is forbidden before exact PASS")

    candidate_sha256 = str(training.get("adapter_sha256"))
    parent_sha256 = str(training.get("parent_adapter_sha256"))
    adapters = {
        "candidate": adapter_entry(
            paths["candidate_adapter"], candidate_sha256
        ),
        "immediate_parent": adapter_entry(
            paths["parent_adapter"], parent_sha256
        ),
    }
    if (
        adapters["candidate"]["metadata"].get("parent_adapter_sha256")
        != parent_sha256
    ):
        raise RuntimeError("direct retention candidate lineage drift")

    data_receipt, selected_rows = _validate_data(
        root=paths["data_root"],
        checkpoint=paths["checkpoint"],
        requested_tasks=TASKS,
        requested_lengths=(4_096,),
        limit_per_cell=20,
    )
    if len(selected_rows) != len(TASKS) * 20:
        raise RuntimeError("direct retention 13-task row-count drift")

    module_root = Path(__file__).resolve().parent
    evaluator = module_root / "evaluate_instruct_ruler_transfer.py"
    gate = module_root / "gate_olmo2_4k_direct_retention.py"
    outputs = {
        "candidate": paths["candidate_output"],
        "immediate_parent": paths["parent_output"],
    }
    adapter_paths = {
        "candidate": paths["candidate_adapter"],
        "immediate_parent": paths["parent_adapter"],
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
            str(args.frequency),
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

    receipt: dict[str, Any] = {
        "status": RETENTION_READY_STATUS,
        "objective": (
            "Test whether the 32-step realized-gap answer-plus-EOS "
            "continuation preserves the fixed 13-task 4K matrix relative "
            "to its direct 300-step EVQ parent."
        ),
        "scope_boundary": (
            "Fixed 13-task 4K RULER and matched natural NLL only; not "
            "global retention relative to untouched OLMo-2."
        ),
        "inputs": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "composite_sha256": checkpoint_sha256,
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
            "retention_gate",
        ],
        "registered_commands": {
            role: evaluation_argv(role) for role in outputs
        },
        "protocol": {
            "frequency": str(args.frequency),
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
        "post_stop_retention_diagnostic_authorized": (
            exact_stop_diagnostic
        ),
    }
    receipt["registered_commands"]["retention_gate"] = [
        sys.executable,
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "gate_olmo2_4k_direct_retention"
        ),
        "--training-result",
        str(paths["training_result"]),
        "--exact-gate",
        str(paths["exact_gate"]),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--expected-frequency",
        str(args.frequency),
        "--candidate-result",
        str(paths["candidate_output"] / "results.json"),
        "--candidate-examples",
        str(paths["candidate_output"] / "examples.jsonl"),
        "--parent-result",
        str(paths["parent_output"] / "results.json"),
        "--parent-examples",
        str(paths["parent_output"] / "examples.jsonl"),
        "--output",
        str(paths["gate_output"]),
    ]
    receipt["gate_name"] = GATE_NAME
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
