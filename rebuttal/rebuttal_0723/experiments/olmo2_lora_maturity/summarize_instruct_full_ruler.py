#!/usr/bin/env python3
"""Aggregate complete per-task OLMo-2 RULER result directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .evaluate_instruct_ruler_transfer import RESULT_STATUS
from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_transfer import TASK_CONFIGS


SUMMARY_STATUS = "OLMO2_INSTRUCT_FULL_RULER_SUMMARY_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Arm name and root containing completed results.json files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_arm(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"invalid --arm value: {value!r}")
    return name, Path(raw_path).resolve()


def load_arm(name: str, root: Path) -> dict[str, Any]:
    direct = root / "results.json"
    paths = [direct] if direct.is_file() else sorted(
        root.glob("*/results.json")
    )
    if not paths:
        raise RuntimeError(f"{name}: no completed result files under {root}")
    expected_tasks = set(TASK_CONFIGS)
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    raw_files: list[dict[str, Any]] = []
    checkpoint_sha256 = None
    adapter_sha256 = None
    frequency_receipts: dict[str, dict[str, Any]] = {}

    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != RESULT_STATUS:
            raise RuntimeError(f"{name}: incomplete result {path}")
        observed_checkpoint = str(receipt["checkpoint_sha256"])
        if checkpoint_sha256 is None:
            checkpoint_sha256 = observed_checkpoint
        elif checkpoint_sha256 != observed_checkpoint:
            raise RuntimeError(f"{name}: checkpoint hash drift")
        adapter = receipt.get("adapter")
        observed_adapter = (
            None if adapter is None else str(adapter["sha256"])
        )
        if adapter_sha256 is None:
            adapter_sha256 = observed_adapter
        elif adapter_sha256 != observed_adapter:
            raise RuntimeError(f"{name}: adapter hash drift")
        frequency = receipt["frequency"]
        frequency_receipts[
            str(frequency["active_sha256_float32"])
        ] = frequency
        for task, task_cells in receipt["results"]["cells"].items():
            for raw_length, cell in task_cells.items():
                key = (str(task), int(raw_length))
                if key in cells:
                    raise RuntimeError(f"{name}: duplicate cell {key}")
                if int(cell["examples"]) != 20:
                    raise RuntimeError(
                        f"{name}: {key} has {cell['examples']} examples"
                    )
                cells[key] = {
                    "examples": int(cell["examples"]),
                    "official_metric": str(cell["official_metric"]),
                    "official_task_score": float(
                        cell["official_task_score"]
                    ),
                    "reference_recall": float(cell["reference_recall"]),
                    "all_references_found": float(
                        cell["all_references_found"]
                    ),
                }
        raw_files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    tasks = sorted({task for task, _ in cells})
    lengths = sorted({length for _, length in cells})
    if set(tasks) != expected_tasks:
        raise RuntimeError(
            f"{name}: task coverage drift: {tasks} != "
            f"{sorted(expected_tasks)}"
        )
    expected_cells = {
        (task, length) for task in tasks for length in lengths
    }
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        extra = sorted(set(cells) - expected_cells)
        raise RuntimeError(
            f"{name}: non-Cartesian coverage missing={missing} extra={extra}"
        )
    if len(frequency_receipts) != 1:
        raise RuntimeError(f"{name}: frequency receipt drift")

    by_length: dict[str, Any] = {}
    for length in lengths:
        selected = [
            cell
            for (task, cell_length), cell in cells.items()
            if cell_length == length
        ]
        score = sum(
            float(cell["official_task_score"]) for cell in selected
        ) / len(selected)
        recall = sum(
            float(cell["reference_recall"]) for cell in selected
        ) / len(selected)
        if not math.isfinite(score) or not math.isfinite(recall):
            raise RuntimeError(f"{name}: non-finite L{length} aggregate")
        by_length[str(length)] = {
            "tasks": len(selected),
            "examples": sum(int(cell["examples"]) for cell in selected),
            "macro_official_task_score": score,
            "macro_reference_recall": recall,
        }

    ordered_cells = {
        task: {
            str(length): cells[(task, length)] for length in lengths
        }
        for task in tasks
    }
    overall = sum(
        float(cell["official_task_score"]) for cell in cells.values()
    ) / len(cells)
    return {
        "root": str(root),
        "checkpoint_sha256": checkpoint_sha256,
        "adapter_sha256": adapter_sha256,
        "frequency": next(iter(frequency_receipts.values())),
        "tasks": tasks,
        "lengths": lengths,
        "cells": ordered_cells,
        "by_length": by_length,
        "overall_macro_official_task_score": overall,
        "raw_results": raw_files,
    }


def main() -> None:
    args = parse_args()
    parsed = [parse_arm(value) for value in args.arm]
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm name")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = {
        "status": SUMMARY_STATUS,
        "metric_boundary": (
            "Official autoregressive RULER task-specific scoring, 20 fixed "
            "examples per task and length. This summary aggregates immutable "
            "per-task result receipts and does not rescore predictions."
        ),
        "arms": {
            name: load_arm(name, root) for name, root in parsed
        },
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
