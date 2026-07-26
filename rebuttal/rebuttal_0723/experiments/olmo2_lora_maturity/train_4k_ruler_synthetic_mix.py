#!/usr/bin/env python3
"""Continue one EVQ adapter on a fixed 4K RULER-task/replay mixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
    save_adapter,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .prepare_4k_ruler_synthetic_mix import LENGTH, STATUS
from .train_4k_tulu_recovery import (
    evaluate_assistant_nll,
    train_one_or_more_epochs,
)
from .train_screen import apply_frequency, load_fixed_view


RESULT_STATUS = "OLMO2_4K_RULER_SYNTHETIC_MIX_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument("--seed", type=int, default=20_420_726)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    parent_adapter = args.parent_adapter.resolve()
    training_view = args.training_view.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")

    view = load_fixed_view(training_view)
    if view.manifest.get("status") != STATUS:
        raise RuntimeError("synthetic-mix data status drift")
    if tuple(view.input_ids.shape)[1] != LENGTH:
        raise RuntimeError("synthetic mix must be exact 4K storage")
    validation_rows = np.flatnonzero(view.split == 1)
    if len(validation_rows) != 20:
        raise RuntimeError("expected 20 held-out synthetic validation rows")

    incomplete.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    model = load_model(checkpoint)
    frequency = apply_frequency(model, "evq")
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("synthetic-mix training does not admit a readout")
    parent_metadata = load_adapter(parent_adapter, model, None)
    expected_parent = {
        "frequency": "evq",
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if parent_metadata.get(name) != expected:
            raise RuntimeError(
                f"parent adapter metadata drift for {name}: "
                f"{parent_metadata.get(name)!r} != {expected!r}"
            )
    model.to("cuda")

    validation_before = evaluate_assistant_nll(
        model=model,
        view=view,
        rows=validation_rows,
        batch_size=int(args.micro_batch_size),
    )
    training = train_one_or_more_epochs(
        model=model,
        view=view,
        epochs=int(args.epochs),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        seed=int(args.seed),
        compile_mode=args.compile_mode,
        log_path=incomplete / "train_log.jsonl",
    )
    training["objective"] = (
        "answer_only_synthetic_vt_cwe_fwe_qa_with_niah_and_natural_replay"
    )
    validation_after = evaluate_assistant_nll(
        model=model,
        view=view,
        rows=validation_rows,
        batch_size=int(args.micro_batch_size),
    )
    metadata = {
        "base_checkpoint_sha256": parent_metadata[
            "base_checkpoint_sha256"
        ],
        "frequency": "evq",
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "adaptation_description": (
            f"qkvo_r{int(args.rank)}_alpha{float(args.alpha):g}"
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
        "stage": "ruler_synthetic_mix_continuation",
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "training_view_sha256": sha256_file(
            training_view / "manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha = save_adapter(
        incomplete / "adapter.pt", model, None, metadata
    )
    result = {
        "status": RESULT_STATUS,
        "metric_boundary": view.manifest["metric_boundary"],
        "checkpoint": str(checkpoint),
        "parent_adapter": {
            "path": str(parent_adapter),
            "sha256": sha256_file(parent_adapter),
            "metadata": parent_metadata,
        },
        "training_view": {
            "path": str(training_view),
            "manifest_sha256": sha256_file(
                training_view / "manifest.json"
            ),
            "manifest": view.manifest,
        },
        "adapter_sha256": adapter_sha,
        "adapter_metadata": metadata,
        "frequency": frequency,
        "runtime": {
            **runtime,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "compile_cache": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
            "allocator": allocator,
        },
        "training": training,
        "synthetic_validation_before": validation_before,
        "synthetic_validation_after": validation_after,
    }
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
