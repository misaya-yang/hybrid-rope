#!/usr/bin/env python3
"""Create the no-GPU READY receipt for the physical-8K Llama-3-8B experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from .common import (
    READY_STATUS,
    TRAIN_LENGTH,
    VIEW_STATUS,
    atomic_json,
    canonical_json_sha256,
    sha256_file,
)


CRITICAL_CODE = (
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/__init__.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/common.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/prepare_ruler_sources.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/prepare_training_view.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/train.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/evaluate.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/preflight.py",
    "rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/run.sh",
    "experiments/lora_evq_v2/train_evq_lora.py",
    "experiments/lora_evq_v2/legacy_lora_protocol.py",
    "scripts/lib/rope/schedules.py",
)
REQUIRED_PACKAGES = {
    "torch": "2.8.0+cu128",
    "transformers": "4.57.6",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "liger-kernel": "0.7.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--evq-parent", type=Path, required=True)
    parser.add_argument("--native-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def receipt(path: Path) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_model(
    checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise RuntimeError("model manifest format drift")
    files = {}
    for record in manifest["files"]:
        path = checkpoint / record["name"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"model file drift: {path}")
        files[record["name"]] = {
            "sha256": record["sha256"],
            "size_bytes": int(record["size_bytes"]),
        }
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected = {
        "model_type": "llama",
        "hidden_size": 4_096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": TRAIN_LENGTH,
        "rope_theta": 500_000.0,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError(
                f"model geometry drift for {name}: "
                f"{config.get(name)!r} != {value!r}"
            )
    return {
        "path": str(checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "geometry": expected,
    }


def verify_view(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != VIEW_STATUS:
        raise RuntimeError("training-view status drift")
    protocol = manifest["protocol"]
    if (
        int(protocol["physical_sequence_length"]) != TRAIN_LENGTH
        or int(protocol["train_rows"]) != 1_376
        or int(protocol["validation_rows"]) != 52
        or int(protocol["registered_global_batch_size"]) != 8
        or int(protocol["epochs"]) != 3
    ):
        raise RuntimeError("training-view protocol drift")
    files = {}
    for name, record in manifest["storage"]["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"training-view file drift: {path}")
        files[name] = dict(record)
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "protocol": protocol,
    }


def verify_parent(root: Path, method: str) -> dict[str, Any]:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "experiment_meta.json",
        "custom_inv_freq.pt",
    )
    files = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = receipt(path)
    metadata = json.loads(
        (root / "experiment_meta.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (root / "adapter_config.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("status") != "complete"
        or int(metadata.get("global_step", -1)) != 300
        or metadata.get("rope_method") != method
        or int(metadata.get("max_seq_len", -1)) != TRAIN_LENGTH
    ):
        raise RuntimeError(f"{method} parent identity drift")
    if (
        int(config.get("r", -1)) != 64
        or int(config.get("lora_alpha", -1)) != 128
        or set(config.get("target_modules", []))
        != {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise RuntimeError(f"{method} parent LoRA drift")
    return {
        "path": str(root),
        "files": files,
        "parent_global_step": 300,
        "parent_seed": int(metadata["protocol"]["seed"]),
        "physical_training_length": int(metadata["max_seq_len"]),
    }


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    checkpoint = args.checkpoint.resolve()
    model_manifest = args.model_manifest.resolve()
    training_view = args.training_view.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    packages = {
        name: importlib.metadata.version(name)
        for name in REQUIRED_PACKAGES
    }
    for name, expected in REQUIRED_PACKAGES.items():
        if packages[name] != expected:
            raise RuntimeError(
                f"runtime package drift for {name}: "
                f"{packages[name]} != {expected}"
            )
    code = {}
    for relative in CRITICAL_CODE:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code[relative] = receipt(path)

    model = verify_model(checkpoint, model_manifest)
    view = verify_view(training_view)
    arms = {
        "evq_cosh": {
            "parent_adapter": verify_parent(
                args.evq_parent.resolve(),
                "evq_cosh",
            ),
        },
        "native_geo": {
            "parent_adapter": verify_parent(
                args.native_parent.resolve(),
                "native_geo",
            ),
        },
    }
    protocol = {
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "physical_training_length": TRAIN_LENGTH,
        "virtual_position_ids": False,
        "epochs": 3,
        "global_batch_size": 8,
        "peak_learning_rate": 2e-5,
        "warmup_ratio": 0.05,
        "optimizer": "fused_adamw",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "scheduler": "cosine_to_0.1x_peak",
        "precision": "bfloat16",
        "attention": "PyTorch Flash-only SDPA with GQA",
        "gradient_checkpointing": False,
        "loss": "Liger fused linear cross entropy",
        "compile_mode": "max-autotune-no-cudagraphs",
        "primary_arm": "evq_cosh",
        "native_gate": (
            "run only after useful EVQ 8K or directional 16K screen"
        ),
    }
    stat = os.statvfs(training_view)
    result = {
        "status": READY_STATUS,
        "no_gpu_preflight": {
            "cuda_available": bool(torch.cuda.is_available()),
            "python": platform.python_version(),
            "packages": packages,
            "torch_cuda_build": torch.version.cuda,
            "compiled_architectures": (
                torch.cuda.get_arch_list()
                if hasattr(torch.cuda, "get_arch_list")
                else []
            ),
            "free_bytes_on_data_disk": int(
                stat.f_bavail * stat.f_frsize
            ),
        },
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "inputs": {
            "checkpoint": model,
            "training_view": view,
        },
        "arms": arms,
        "code": {
            "root": str(code_root),
            "files": code,
        },
        "launch_policy": {
            "probe": (
                "probe micro-batch 2/global 8 first; fall back to "
                "micro-batch 1/global 8 on OOM"
            ),
            "selection_metric": (
                "sustained physical input tokens/s with finite loss"
            ),
            "stop": (
                "stop after EVQ n=20 screen if no material 8K capability "
                "and no directional 16K transfer"
            ),
        },
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
