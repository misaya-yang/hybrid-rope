#!/usr/bin/env python3
"""Create a no-GPU READY receipt for the OLMo-2 1B Instruct EVQ run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .prepare_data import atomic_json, sha256_file, verify_collection
from .prepare_probe_background import verify_background


EXPECTED_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
EXPECTED_WEIGHT_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
EXPECTED_WEIGHT_BYTES = 2_969_854_224
EXPECTED_TOKENIZER_SHA256 = (
    "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
)
CODE_FILES = (
    "rebuttal/rebuttal_0723/experiments/small_model_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/contract.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/train.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_data.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/prepare_probe_background.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_screen.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_4k_stage_a.py",
    "scripts/lib/rope/schedules.py",
)


def verify_checkpoint(checkpoint: Path) -> dict[str, Any]:
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected_config = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4_096,
        "rope_theta": 500_000,
        "vocab_size": 100_352,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"instruct checkpoint config drift for {key}: "
                f"{config.get(key)!r}"
            )
    revision = (
        checkpoint / "revision.txt"
    ).read_text(encoding="utf-8").strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError("instruct checkpoint revision drift")
    weight = checkpoint / "model.safetensors"
    if weight.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError("instruct checkpoint byte-size drift")
    weight_sha = sha256_file(weight)
    if weight_sha != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("instruct checkpoint checksum drift")
    tokenizer_sha = sha256_file(checkpoint / "tokenizer.json")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("instruct tokenizer checksum drift")
    return {
        "status": "verified",
        "checkpoint_path": str(checkpoint.resolve()),
        "model_id": "allenai/OLMo-2-0425-1B-Instruct",
        "revision": revision,
        "config": expected_config,
        "files": {
            "model.safetensors": {
                "bytes": int(weight.stat().st_size),
                "sha256": weight_sha,
                "mtime_ns": int(weight.stat().st_mtime_ns),
            }
        },
        "composite_sha256": weight_sha,
        "tokenizer_sha256": tokenizer_sha,
    }


def verify_code(code_root: Path) -> dict[str, Any]:
    files = {}
    for relative in CODE_FILES:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    return {"status": "verified", "files": files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=8_000_000_000
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must hide CUDA")

    import torch
    import transformers

    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees CUDA")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient output storage")
    receipt = {
        "format_version": 1,
        "status": "OLMO2_INSTRUCT_4K_CONVERSION_READY",
        "scientific_contract": {
            "base_checkpoint": "OLMo-2-0425-1B-Instruct",
            "frequency_intervention": "endpoint EVQ-Cosh tau=2",
            "maximum_optimizer_sequence_length": 4_096,
            "stage_a": "4K full-token LongAlign next-token CE",
            "evaluation_only_lengths": [4_096, 8_192, 16_384],
            "forbidden": [
                "position interpolation",
                "virtual position IDs beyond 4095",
                "attention-temperature or operator changes",
                "teacher or hidden-state distillation",
            ],
        },
        "checkpoint": verify_checkpoint(args.checkpoint.resolve()),
        "training_data": verify_collection(
            args.prepared_data.resolve() / "collection_manifest.json"
        ),
        "background": verify_background(args.background_dir.resolve()),
        "code": verify_code(args.code_root.resolve()),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "storage": {
            "output_root": str(output_root),
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
    }
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint": receipt["checkpoint"]["composite_sha256"],
                "free_bytes": free_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
