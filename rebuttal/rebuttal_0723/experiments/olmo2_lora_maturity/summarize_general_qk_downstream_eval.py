#!/usr/bin/env python3
"""Curate the completed general-QK QA/RULER matrix into one metrics owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .evaluate_2wiki_phase_adaptation import (
    RESULT_STATUS as TWO_WIKI_STATUS,
)
from .evaluate_instruct_ruler_transfer import (
    RESULT_STATUS as RULER_STATUS,
)
from .preflight_general_qk_downstream_eval import STATUS as READY_STATUS


STATUS = "OLMO2_GENERAL_QK_DOWNSTREAM_MATRIX_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def artifact_hashes(root: Path) -> dict[str, str]:
    names = ("run_manifest.json", "examples.jsonl", "results.json")
    return {
        name: sha256_file(root / name)
        for name in names
        if (root / name).is_file()
    }


def summarize_ruler(result: dict[str, Any]) -> dict[str, Any]:
    by_length: dict[str, list[float]] = {}
    for task_cells in result["results"]["cells"].values():
        for length, cell in task_cells.items():
            by_length.setdefault(str(length), []).append(
                float(cell["official_task_score"])
            )
    return {
        "examples": int(result["results"]["examples"]),
        "macro_official_task_score": float(
            result["results"]["macro_official_task_score"]
        ),
        "length_macro_official_task_score": {
            length: sum(values) / len(values)
            for length, values in sorted(by_length.items())
        },
        "tasks_per_length": {
            length: len(values)
            for length, values in sorted(by_length.items())
        },
    }


def summarize_2wiki(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "examples": int(result["results"]["examples"]),
        "macro_token_f1": float(result["results"]["macro_token_f1"]),
        "macro_normalized_exact": float(
            result["results"]["macro_normalized_exact"]
        ),
        "cells": {
            str(length): {
                "mean_token_f1": float(cell["mean_token_f1"]),
                "normalized_exact": float(
                    cell["normalized_exact"]
                ),
                "terminal_eos": float(cell["terminal_eos"]),
                "examples": int(cell["examples"]),
                "mean_input_tokens": float(cell["mean_input_tokens"]),
            }
            for length, cell in sorted(result["results"]["cells"].items())
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    ready_path = args.ready_receipt.resolve()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != READY_STATUS:
        raise RuntimeError("general-QK downstream READY status drift")
    output_root = args.output_root.resolve()
    if output_root != Path(ready["output_root"]).resolve():
        raise RuntimeError("general-QK downstream output-root drift")

    rows: list[dict[str, Any]] = []
    for job in ready["jobs"]:
        root = output_root / str(job["name"])
        result_path = root / "results.json"
        if not result_path.is_file():
            raise RuntimeError(f"incomplete downstream job: {job['name']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_status = (
            RULER_STATUS
            if job["benchmark"] == "ruler"
            else TWO_WIKI_STATUS
        )
        if (
            result.get("status") != expected_status
            or result.get("frequency", {}).get("active_frequency")
            is None
        ):
            raise RuntimeError(
                f"downstream result identity drift: {job['name']}"
            )
        expected_script = ready["bound_code_sha256"][
            (
                "ruler_evaluator"
                if job["benchmark"] == "ruler"
                else "two_wiki_evaluator"
            )
        ]
        if result.get("script_sha256") != expected_script:
            raise RuntimeError(
                f"downstream evaluator SHA drift: {job['name']}"
            )
        metrics = (
            summarize_ruler(result)
            if job["benchmark"] == "ruler"
            else summarize_2wiki(result)
        )
        adapter_receipt = result.get("adapter")
        rows.append(
            {
                "job": job,
                "active_frequency": result["frequency"][
                    "active_frequency"
                ],
                "active_frequency_sha256_float32": result["frequency"][
                    "active_sha256_float32"
                ],
                "adapter_sha256": (
                    adapter_receipt.get("sha256")
                    if isinstance(adapter_receipt, dict)
                    else None
                ),
                "metrics": metrics,
                "artifacts": artifact_hashes(root),
            }
        )

    summary = {
        "status": STATUS,
        "boundary": (
            "Single-seed post-submission evidence. Training uses generic "
            "LongAlign/Tulu rows at physical length <=4K. RULER and the "
            "deterministically length-filled 2Wiki protocol are capability "
            "endpoints; YaRN factors are target-length settings, not a "
            "fully optimized hyperparameter sweep."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ready_receipt_sha256": sha256_file(ready_path),
        "matched_training_gate": ready["matched_training_gate"],
        "jobs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
