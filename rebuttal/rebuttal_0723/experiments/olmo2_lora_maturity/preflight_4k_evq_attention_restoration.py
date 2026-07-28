#!/usr/bin/env python3
"""Create a no-GPU preparation receipt for EVQ attention restoration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
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

from .evq_attention_restoration import (
    LINEARARD_COMMIT,
    LINEARARD_KERNEL_SHA256,
    METHOD_ID,
    PREPARED_STATUS,
    SEQUENCE_LENGTH,
)
from .train_4k_evq_attention_restoration import protocol_from_args
from .train_4k_stage_a import composite_checkpoint_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--linearard-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--gpu-ready-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1024.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.9)
    parser.add_argument("--maximum-gradient-norm", type=float, default=5.0)
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument(
        "--self-relation-weight", type=float, default=0.25
    )
    parser.add_argument("--seed", type=int, default=20_260_803)
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def validate_checkpoint(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head_dim = int(config["hidden_size"]) // int(
        config["num_attention_heads"]
    )
    expected = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": SEQUENCE_LENGTH,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"checkpoint config drift for {key}: "
                f"{config.get(key)!r} != {value!r}"
            )
    if head_dim != 128:
        raise RuntimeError("OLMo-2 attention head dimension drift")
    weight_files = sorted(path.glob("model-*.safetensors"))
    if not weight_files:
        weight_files = [path / "model.safetensors"]
    if any(not candidate.is_file() for candidate in weight_files):
        raise FileNotFoundError("checkpoint safetensor weights are incomplete")
    return {
        "path": str(path.resolve()),
        "config": file_entry(config_path),
        "weight_files": [
            {
                "path": candidate.name,
                "bytes": int(candidate.stat().st_size),
                "mtime_ns": int(candidate.stat().st_mtime_ns),
            }
            for candidate in weight_files
        ],
        "composite_sha256": composite_checkpoint_sha256(path),
        "architecture": {
            **expected,
            "head_dim": head_dim,
        },
    }


def validate_training_view(path: Path) -> dict[str, Any]:
    required = (
        "manifest.json",
        "input_ids.npy",
        "lengths.npy",
        "split.npy",
    )
    files = {
        name: file_entry(path / name)
        for name in required
        if (path / name).is_file()
    }
    if set(files) != set(required):
        missing = sorted(set(required) - set(files))
        raise FileNotFoundError(f"training-view files missing: {missing}")
    input_ids = np.load(
        path / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    lengths = np.load(
        path / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(path / "split.npy", mmap_mode="r", allow_pickle=False)
    if (
        input_ids.ndim != 2
        or input_ids.shape[1] != SEQUENCE_LENGTH
        or lengths.shape != (input_ids.shape[0],)
        or split.shape != (input_ids.shape[0],)
    ):
        raise RuntimeError("training-view shape drift")
    training_rows = np.flatnonzero(split == 0)
    if len(training_rows) == 0:
        raise RuntimeError("training view contains no training rows")
    if np.any(lengths[training_rows] != SEQUENCE_LENGTH):
        raise RuntimeError("restoration rows are not all exactly 4K")
    return {
        "path": str(path.resolve()),
        "files": files,
        "shape": [int(value) for value in input_ids.shape],
        "dtype": str(input_ids.dtype),
        "training_rows": int(len(training_rows)),
        "minimum_training_length": int(lengths[training_rows].min()),
        "maximum_training_length": int(lengths[training_rows].max()),
    }


def validate_linearard(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINEARARD_COMMIT:
        raise RuntimeError(
            f"LinearARD commit drift: {commit} != {LINEARARD_COMMIT}"
        )
    kernel_root = path / "kernels" / "attention_KL"
    files = {}
    for name, expected_sha256 in LINEARARD_KERNEL_SHA256.items():
        candidate = kernel_root / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        entry = file_entry(candidate)
        if entry["sha256"] != expected_sha256:
            raise RuntimeError(f"LinearARD {name} hash drift")
        files[name] = entry
    license_path = path / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(license_path)
    license_text = license_path.read_text(
        encoding="utf-8", errors="replace"
    )
    if "MIT License" not in license_text:
        raise RuntimeError("LinearARD license identity drift")
    return {
        "path": str(path.resolve()),
        "git_commit": commit,
        "kernel_files": files,
        "license": file_entry(license_path),
        "upstream_runtime": {
            "torch": "2.9.0+cu128",
            "transformers": "4.45.2",
            "triton": "3.5.0",
        },
    }


def command(
    *,
    mode: str,
    args: argparse.Namespace,
    receipt: Path,
    output: Path,
) -> list[str]:
    return [
        "/root/miniconda3/bin/python",
        str(args.trainer.resolve()),
        "--mode",
        mode,
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--training-view",
        str(args.training_view.resolve()),
        "--linearard-root",
        str(args.linearard_root.resolve()),
        "--receipt",
        str(receipt.resolve()),
        "--output",
        str(output.resolve()),
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
        "--minimum-lr-ratio",
        str(args.minimum_lr_ratio),
        "--maximum-gradient-norm",
        str(args.maximum_gradient_norm),
        "--attention-weight",
        str(args.attention_weight),
        "--context-weight",
        str(args.context_weight),
        "--self-relation-weight",
        str(args.self_relation_weight),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    for name in (
        "checkpoint",
        "training_view",
        "linearard_root",
        "trainer",
        "gpu_ready_output",
        "run_output",
        "receipt_output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.receipt_output.exists():
        raise FileExistsError(args.receipt_output)
    if args.gpu_ready_output.exists():
        raise FileExistsError(args.gpu_ready_output)
    if args.run_output.exists() or args.run_output.with_name(
        args.run_output.name + ".incomplete"
    ).exists():
        raise FileExistsError(args.run_output)
    if not args.trainer.is_file():
        raise FileNotFoundError(args.trainer)
    if int(args.steps) != 144:
        raise ValueError("registered first run requires exactly 144 steps")
    if int(args.micro_batch_size) != 1:
        raise ValueError("registered first run requires micro batch 1")
    if int(args.gradient_accumulation_steps) != 4:
        raise ValueError("registered first run requires accumulation 4")
    if int(args.rank) != 512 or float(args.alpha) != 1024.0:
        raise ValueError("registered first run requires rank 512 / alpha 1024")
    protocol = protocol_from_args(SimpleNamespace(**vars(args)))
    helper = args.trainer.with_name("evq_attention_restoration.py")
    preflight = Path(__file__).resolve()
    if not helper.is_file():
        raise FileNotFoundError(helper)
    checkpoint = validate_checkpoint(args.checkpoint)
    training_view = validate_training_view(args.training_view)
    linearard = validate_linearard(args.linearard_root)
    receipt = {
        "status": PREPARED_STATUS,
        "method_id": METHOD_ID,
        "classification": "DESIGN_ONLY_RUNTIME_UNVERIFIED",
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "scientific_question": (
            "Can a Native teacher restore tested 4K attention function "
            "after a mature OLMo-2 student is fixed to the complete EVQ "
            "frequency grid, before any long-task continuation?"
        ),
        "difference_from_failed_arms": {
            "failed_a1": (
                "linear Native-to-EVQ frequency morph plus token CE"
            ),
            "failed_a2": (
                "the same morph plus sparse output-logit KL"
            ),
            "this_candidate": (
                "fixed full EVQ from step 0; dense post-RoPE QK "
                "attention KL and A@V context matching; no CE or morph"
            ),
        },
        "protocol": protocol,
        "inputs": {
            "checkpoint": checkpoint,
            "training_view": training_view,
            "linearard_root": linearard,
        },
        "source": {
            "trainer": file_entry(args.trainer),
            "helper": file_entry(helper),
            "preflight": file_entry(preflight),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "triton": importlib.metadata.version("triton"),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "gpu_smoke_required": {
            "reason": (
                "Upstream kernels target torch 2.9/Triton 3.5; this "
                "server has torch 2.8/Triton 3.4 and no active GPU."
            ),
            "checks": [
                "BF16 D128 QK forward-loss parity",
                "QK backward-gradient parity",
                "alias-gradient parity for Q/Q, K/K, V/V calls",
                "full OLMo teacher+student memory fit",
                "finite composite loss and nonzero QKV LoRA gradients",
                "exact Native teacher and full-EVQ student frequency hashes",
            ],
            "command": command(
                mode="gpu-smoke",
                args=args,
                receipt=args.receipt_output,
                output=args.gpu_ready_output,
            ),
        },
        "future_train_command": command(
            mode="train",
            args=args,
            receipt=args.gpu_ready_output,
            output=args.run_output,
        ),
        "stop_conditions": [
            "GPU kernel parity exceeds registered tolerances",
            "full-model probe does not fit or produces non-finite values",
            "any trainable tensor escapes Q/K/V LoRA A/B",
            "Native teacher or full-EVQ student frequency identity drifts",
            "4K restoration screen misses any registered capability gate",
        ],
        "promotion_boundary": (
            "Training completion is not capability evidence. The adapter "
            "must pass held-out 4K 2Wiki, RULER13, natural NLL and an "
            "independent retention slice before any 8K/16K evaluation."
        ),
    }
    atomic_json(args.receipt_output, receipt)


if __name__ == "__main__":
    main()
