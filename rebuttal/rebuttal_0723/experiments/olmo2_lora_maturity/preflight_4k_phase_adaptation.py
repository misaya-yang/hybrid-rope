#!/usr/bin/env python3
"""Freeze one no-GPU READY receipt for physical-4K phase adaptation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .phase_adaptation import LENGTH, READY_STATUS, PhaseAdaptationView
from .train_4k_phase_adaptation import (
    bound_code_sha256,
    registered_protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--natural-view", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def manifest_files(root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    files: dict[str, dict[str, Any]] = {
        "manifest.json": {
            "bytes": (root / "manifest.json").stat().st_size,
            "sha256": sha256_file(root / "manifest.json"),
        }
    }
    for name in manifest.get("files", {}):
        path = root / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    parent = args.parent_adapter.resolve()
    training_view = PhaseAdaptationView(args.training_view.resolve())
    natural_view = args.natural_view.resolve()
    ready = args.ready_receipt.resolve()
    output = args.output.resolve()
    if ready.exists():
        raise FileExistsError(ready)
    if output.exists() or output.with_name(
        output.name + ".incomplete"
    ).exists():
        raise FileExistsError(output)
    if (
        int(args.micro_batch_size) != 4
        or int(args.gradient_accumulation_steps) != 2
        or int(args.steps) <= 0
        or training_view.manifest["tokenizer_sha256"]
        != sha256_file(checkpoint / "tokenizer.json")
    ):
        raise RuntimeError("phase-adaptation preflight contract drift")
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, checkpoint_ready
    )
    payload = torch.load(
        parent, map_location="cpu", weights_only=False
    )
    metadata = dict(payload["metadata"])
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if metadata.get(name) != expected:
            raise RuntimeError(
                f"parent adapter drift for {name}: "
                f"{metadata.get(name)!r} != {expected!r}"
            )
    natural_files = manifest_files(natural_view)
    free = shutil.disk_usage(output.parent).free
    if free < 2_000_000_000:
        raise RuntimeError("less than 2GB free for phase adaptation")

    command = [
        sys.executable,
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments."
            "olmo2_lora_maturity.train_4k_phase_adaptation"
        ),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-ready-receipt",
        str(checkpoint_ready),
        "--parent-adapter",
        str(parent),
        "--training-view",
        str(training_view.root),
        "--natural-view",
        str(natural_view),
        "--ready-receipt",
        str(ready),
        "--output",
        str(output),
        "--frequency",
        str(args.frequency),
        "--steps",
        str(args.steps),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-steps",
        str(args.warmup_steps),
        "--compile-mode",
        str(args.compile_mode),
        "--validation-rows",
        str(args.validation_rows),
        "--seed",
        str(args.seed),
    ]
    receipt = {
        "status": READY_STATUS,
        "authorization_boundary": (
            "READY validates inputs and protocol; user authorization is "
            "still required before GPU execution."
        ),
        "output": str(output),
        "protocol": registered_protocol(args),
        "bound_code_sha256": bound_code_sha256(),
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "composite_sha256": checkpoint_digest,
                "ready_receipt": {
                    "path": str(checkpoint_ready),
                    "sha256": sha256_file(checkpoint_ready),
                },
            },
            "parent_adapter": {
                "path": str(parent),
                "bytes": parent.stat().st_size,
                "sha256": sha256_file(parent),
                "metadata": metadata,
            },
            "training_view": {
                "path": str(training_view.root),
                "manifest_sha256": sha256_file(
                    training_view.root / "manifest.json"
                ),
                "status": training_view.manifest["status"],
                "training_rows": len(training_view.training_rows),
                "validation_rows": len(training_view.validation_rows),
            },
            "natural_view": {
                "path": str(natural_view),
                "files": natural_files,
            },
        },
        "free_space_bytes": free,
        "entry_command": command,
    }
    ready.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ready, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
