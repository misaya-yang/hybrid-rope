#!/usr/bin/env python3
"""Create the no-GPU READY receipt for Native rotary-band diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .diagnose_4k_native_band_importance import (
    deterministic_calibration_rows,
    deterministic_query_positions,
)
from .native_protected_evq import (
    CALIBRATION_ROWS,
    DIAGNOSTIC_PREPARED_STATUS,
    QUERY_POSITIONS,
    SEQUENCE_LENGTH,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import load_fixed_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=CALIBRATION_ROWS)
    parser.add_argument(
        "--query-position-count", type=int, default=QUERY_POSITIONS
    )
    parser.add_argument("--pair-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_804)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=4_000_000_000
    )
    return parser.parse_args()


def file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("diagnostic preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("diagnostic preflight unexpectedly sees CUDA")
    output = args.diagnostic_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if output.exists() or receipt_output.exists():
        raise FileExistsError(output if output.exists() else receipt_output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient diagnostic output storage")
    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, checkpoint_ready
    )
    training_view = args.training_view.resolve()
    view = load_fixed_view(training_view)
    selected_rows = deterministic_calibration_rows(
        split=view.split,
        lengths=view.lengths,
        rows=int(args.rows),
        seed=int(args.seed),
    )
    query_positions = deterministic_query_positions(
        count=int(args.query_position_count)
    )
    files = {
        name: file_receipt(training_view / name)
        for name in (
            "manifest.json",
            "input_ids.npy",
            "assistant_mask.npy",
            "lengths.npy",
            "split.npy",
        )
    }
    for name, entry in files.items():
        expected = view.manifest["files"].get(name)
        if expected is not None and (
            int(expected["bytes"]) != entry["bytes"]
            or str(expected["sha256"]) != entry["sha256"]
        ):
            raise RuntimeError(f"training-view manifest drift for {name}")
    here = Path(__file__).resolve()
    diagnostic = here.with_name("diagnose_4k_native_band_importance.py")
    method = here.with_name("native_protected_evq.py")
    receipt = {
        "status": DIAGNOSTIC_PREPARED_STATUS,
        "scientific_question": (
            "Is Native 4K attention importance sufficiently concentrated "
            "and split-stable to justify protecting a small rotary subset?"
        ),
        "protocol": {
            "rows": int(args.rows),
            "row_indices": [int(value) for value in selected_rows],
            "query_position_count": int(args.query_position_count),
            "query_positions": [
                int(value) for value in query_positions
            ],
            "pair_chunk_size": int(args.pair_chunk_size),
            "physical_sequence_length": SEQUENCE_LENGTH,
            "seed": int(args.seed),
            "training_or_optimizer_steps": 0,
        },
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "composite_sha256": checkpoint_digest,
                "ready_receipt": file_receipt(checkpoint_ready),
            },
            "training_view": {
                "path": str(training_view),
                "files": files,
            },
        },
        "source": {
            "preflight": file_receipt(here),
            "diagnostic": file_receipt(diagnostic),
            "method": file_receipt(method),
        },
        "diagnostic_output": str(output),
        "storage": {
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "stop_condition": (
            "A failed concentration/stability gate ends this method before "
            "training; thresholds are not revised after observation."
        ),
    }
    atomic_json(receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": DIAGNOSTIC_PREPARED_STATUS,
                "receipt": str(receipt_output),
                "receipt_sha256": sha256_file(receipt_output),
                "diagnostic_output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
