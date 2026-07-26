#!/usr/bin/env python3
"""Create a no-GPU READY receipt for one progressive-morph arm."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .train_4k_progressive_morph import (
    LENGTH,
    READY_STATUS,
    protocol_from_args,
)
from .train_4k_stage_a import composite_checkpoint_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--longalign-view", type=Path, required=True)
    parser.add_argument("--tulu-view", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--retention-weight", type=float, default=0.0)
    parser.add_argument("--retention-positions", type=int, default=8)
    parser.add_argument("--retention-temperature", type=float, default=1.0)
    parser.add_argument(
        "--longalign-objective",
        choices=("full", "assistant"),
        default="full",
    )
    parser.add_argument(
        "--morph-schedule",
        choices=("linear", "immediate"),
        default="linear",
    )
    parser.add_argument(
        "--family-pattern",
        choices=("balanced", "longalign_3_to_1"),
        default="balanced",
    )
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_view(
    path: Path,
    *,
    require_all_train_rows_4k: bool,
    tulu: bool,
) -> dict[str, Any]:
    required = (
        "manifest.json",
        "input_ids.npy",
        "assistant_mask.npy",
        "lengths.npy",
        "split.npy",
    )
    files = {}
    for name in required:
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        files[name] = file_entry(candidate)
    manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    input_ids = np.load(
        path / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    mask = np.load(
        path / "assistant_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lengths = np.load(
        path / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(
        path / "split.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        input_ids.shape != mask.shape
        or input_ids.shape[1] != LENGTH
        or lengths.shape != (input_ids.shape[0],)
        or split.shape != (input_ids.shape[0],)
    ):
        raise RuntimeError(f"fixed-view shape drift: {path}")
    train_rows = np.flatnonzero(split == 0)
    if len(train_rows) == 0:
        raise RuntimeError(f"view has no training rows: {path}")
    if np.any(lengths <= 1) or np.any(lengths > LENGTH):
        raise RuntimeError(f"view length escaped 4K: {path}")
    if (
        require_all_train_rows_4k
        and np.any(lengths[train_rows] != LENGTH)
    ):
        raise RuntimeError("LongAlign training rows are not all 4K")
    if tulu:
        source = manifest.get("source", {})
        if (
            source.get("id")
            != "allenai/tulu-3-sft-olmo-2-mixture-0225"
            or source.get("revision")
            != "d91a0785ade02942520280fb484866fce41e448f"
        ):
            raise RuntimeError("Tulu source/revision drift")
        rows_path = path / "rows.jsonl"
        if not rows_path.is_file():
            raise FileNotFoundError(rows_path)
        files["rows.jsonl"] = file_entry(rows_path)
        rows = [
            json.loads(line)
            for line in rows_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        if len(rows) != len(input_ids):
            raise RuntimeError("Tulu row metadata count drift")
        markers = ("ruler", "niah", "needle-in-a-haystack")
        for index in train_rows:
            source_id = str(rows[int(index)]["source_id"]).lower()
            if any(marker in source_id for marker in markers):
                raise RuntimeError(
                    "explicit RULER/NIAH source in Tulu view"
                )
    return {
        "path": str(path),
        "manifest_sha256": files["manifest.json"]["sha256"],
        "files": files,
        "shape": [int(value) for value in input_ids.shape],
        "training_rows": int(len(train_rows)),
        "minimum_train_length": int(lengths[train_rows].min()),
        "maximum_train_length": int(lengths[train_rows].max()),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    longalign = args.longalign_view.resolve()
    tulu = args.tulu_view.resolve()
    background = args.background_dir.resolve()
    trainer = args.trainer.resolve()
    run_output = args.run_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if (
        run_output.exists()
        or run_output.with_name(run_output.name + ".incomplete").exists()
    ):
        raise FileExistsError(run_output)
    if receipt_output.exists():
        raise FileExistsError(receipt_output)
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    if float(args.retention_weight) < 0.0:
        raise ValueError("retention weight must be nonnegative")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    head_dim = int(
        config.get(
            "head_dim",
            int(config.get("hidden_size", 0))
            // max(int(config.get("num_attention_heads", 1)), 1),
        )
    )
    if (
        config.get("model_type") != "olmo2"
        or int(config.get("max_position_embeddings", -1)) != LENGTH
        or head_dim != 128
    ):
        raise RuntimeError("OLMo-2 checkpoint contract drift")
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    background_manifest = background / "manifest.json"
    if not background_manifest.is_file():
        raise FileNotFoundError(background_manifest)

    protocol = protocol_from_args(
        SimpleNamespace(
            steps=args.steps,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            rank=args.rank,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            compile_mode=args.compile_mode,
            retention_weight=args.retention_weight,
            retention_positions=args.retention_positions,
            retention_temperature=args.retention_temperature,
            longalign_objective=args.longalign_objective,
            morph_schedule=args.morph_schedule,
            family_pattern=args.family_pattern,
            natural_eval_rows=args.natural_eval_rows,
            seed=args.seed,
        )
    )
    receipt = {
        "status": READY_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "existing_evidence": (
            "Abrupt EVQ plus LongAlign recovers NLL but loses broad "
            "RULER capability; task-matched binding does not establish "
            "unseen-task transfer."
        ),
        "smallest_missing_evidence": (
            "Whether gradual Native-to-EVQ adaptation on non-RULER "
            "LongAlign/Tulu data retains mature-model capability."
        ),
        "smallest_executable_plan": (
            "One 4K-only progressive-morph QKVO LoRA arm, followed by "
            "a frozen multi-task RULER screen and full evaluation only "
            "for a passing candidate."
        ),
        "stop_condition": (
            "Stop the arm after the registered step budget; do not add "
            "task-matched binding data if the multi-task screen fails."
        ),
        "protocol": protocol,
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "composite_sha256": composite_checkpoint_sha256(
                    checkpoint
                ),
                "model_safetensors": file_entry(weights),
                "config": file_entry(checkpoint / "config.json"),
            },
            "longalign_view": validate_view(
                longalign,
                require_all_train_rows_4k=True,
                tulu=False,
            ),
            "tulu_view": validate_view(
                tulu,
                require_all_train_rows_4k=False,
                tulu=True,
            ),
            "background": {
                "path": str(background),
                "manifest_sha256": sha256_file(
                    background_manifest
                ),
            },
        },
        "trainer": {
            "path": str(trainer),
            **file_entry(trainer),
        },
        "output": str(run_output),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "liger_kernel": importlib.metadata.version(
                "liger-kernel"
            ),
        },
        "evidence_boundary": (
            "No custom binding/RULER training rows; upstream datasets "
            "are not claimed semantically free of every retrieval-like "
            "example."
        ),
    }
    atomic_json(receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "receipt": str(receipt_output),
                "receipt_sha256": sha256_file(receipt_output),
                "protocol": protocol,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
