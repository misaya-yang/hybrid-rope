#!/usr/bin/env python3
"""Create the no-GPU receipt for released-Native chord training."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .far_only_evq_residual import METHOD_ID, PREPARED_STATUS
from .continuous_8k_adaptation import Continuous8KView
from .phase_adaptation import PhaseAdaptationView
from .train_4k_far_only_evq_residual import (
    TRAINING_VIEW_MANIFEST_SHA256,
    bound_code_sha256,
    load_training_view,
    registered_protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest


TRAINER_MODULE = (
    "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
    "train_4k_far_only_evq_residual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--training-mode",
        choices=("phase_gap_4k", "continuous_8k"),
        default="phase_gap_4k",
    )
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--prepared-output", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--projection-rank", type=int, default=64)
    parser.add_argument("--residual-pairs", type=int, default=8)
    parser.add_argument("--initial-logit-gain", type=float, default=0.1)
    parser.add_argument("--content-value-dim", type=int, default=0)
    parser.add_argument("--content-projection-rank", type=int, default=64)
    parser.add_argument("--initial-content-gain", type=float, default=0.1)
    parser.add_argument("--first-token-loss-weight", type=float)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_821)
    return parser.parse_args()


def _trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        training_view=args.training_view,
        training_mode=str(args.training_mode),
        steps=int(args.steps),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        projection_rank=int(args.projection_rank),
        residual_pairs=int(args.residual_pairs),
        initial_logit_gain=float(args.initial_logit_gain),
        content_value_dim=int(args.content_value_dim),
        content_projection_rank=int(args.content_projection_rank),
        initial_content_gain=float(args.initial_content_gain),
        first_token_loss_weight=args.first_token_loss_weight,
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        compile_mode=str(args.compile_mode),
        validation_rows=int(args.validation_rows),
        seed=int(args.seed),
    )


def _input_receipt(
    *,
    args: argparse.Namespace,
    checkpoint_sha256: str,
    view: PhaseAdaptationView | Continuous8KView,
) -> dict[str, Any]:
    return {
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "composite_sha256": checkpoint_sha256,
            "ready_receipt_sha256": sha256_file(
                args.checkpoint_ready_receipt.resolve()
            ),
        },
        "training_view": {
            "path": str(view.root),
            "manifest_sha256": sha256_file(view.root / "manifest.json"),
            "status": str(view.manifest["status"]),
            "shape": [
                int(value) for value in view.manifest["shape"]
            ],
            "training_rows": int(len(view.training_rows)),
            "validation_rows": int(len(view.validation_rows)),
        },
    }


def command(
    *,
    mode: str,
    args: argparse.Namespace,
) -> list[str]:
    output = (
        args.smoke_output if mode == "smoke" else args.run_output
    )
    values = [
        sys.executable,
        "-m",
        TRAINER_MODULE,
        "--mode",
        mode,
        "--training-mode",
        str(args.training_mode),
        "--authorize",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--checkpoint-ready-receipt",
        str(args.checkpoint_ready_receipt.resolve()),
        "--training-view",
        str(args.training_view.resolve()),
        "--prepared-receipt",
        str(args.prepared_output.resolve()),
        "--output",
        str(output.resolve()),
        "--steps",
        str(args.steps),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--projection-rank",
        str(args.projection_rank),
        "--residual-pairs",
        str(args.residual_pairs),
        "--initial-logit-gain",
        str(args.initial_logit_gain),
        "--content-value-dim",
        str(args.content_value_dim),
        "--content-projection-rank",
        str(args.content_projection_rank),
        "--initial-content-gain",
        str(args.initial_content_gain),
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
    if args.first_token_loss_weight is not None:
        values.extend(
            [
                "--first-token-loss-weight",
                str(args.first_token_loss_weight),
            ]
        )
    if mode == "train":
        values.extend(
            [
                "--gpu-ready-receipt",
                str(
                    (
                        args.smoke_output.resolve()
                        / "gpu_ready.json"
                    )
                ),
            ]
        )
    return values


def main() -> None:
    args = parse_args()
    if int(args.steps) != 300:
        raise ValueError("registered first run requires exactly 300 steps")
    expected_shape = (
        (2, 4)
        if str(args.training_mode) == "continuous_8k"
        else (4, 2)
    )
    if (
        int(args.micro_batch_size),
        int(args.gradient_accumulation_steps),
    ) != expected_shape:
        raise ValueError(
            "registered run requires micro/accum "
            f"{expected_shape[0]}/{expected_shape[1]}"
        )
    if (
        int(args.projection_rank) != 64
        or int(args.residual_pairs) != 8
    ):
        raise ValueError("registered first run requires rank 64 / 8 pairs")
    if int(args.content_value_dim) not in {0, 32}:
        raise ValueError("registered content value dim must be zero or 32")
    if int(args.content_projection_rank) != 64:
        raise ValueError("registered content projection rank must be 64")
    if (
        args.first_token_loss_weight is not None
        and float(args.first_token_loss_weight) != 0.5
    ):
        raise ValueError("registered weighted run requires first-token weight 0.5")
    for path in (
        args.prepared_output,
        args.smoke_output,
        args.run_output,
    ):
        resolved = path.resolve()
        if resolved.exists() or resolved.with_name(
            resolved.name + ".incomplete"
        ).exists():
            raise FileExistsError(resolved)
    checkpoint_sha256 = ready_checkpoint_digest(
        args.checkpoint.resolve(),
        args.checkpoint_ready_receipt.resolve(),
    )
    trainer_args = _trainer_args(args)
    view = load_training_view(trainer_args)
    if (
        str(args.training_mode) == "phase_gap_4k"
        and
        TRAINING_VIEW_MANIFEST_SHA256 is not None
        and sha256_file(view.root / "manifest.json")
        != TRAINING_VIEW_MANIFEST_SHA256
    ):
        raise RuntimeError("registered RULER training-view manifest drift")
    dependencies = {}
    for module_name, distribution_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("liger_kernel", "liger-kernel"),
    ):
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(
                f"required runtime module is missing: {module_name}"
            )
        dependencies[distribution_name] = importlib.metadata.version(
            distribution_name
        )
    protocol = registered_protocol(trainer_args)
    receipt = {
        "status": PREPARED_STATUS,
        "classification": "NO_GPU_PREPARATION_NOT_EXPERIMENT_RESULT",
        "method_id": METHOD_ID,
        "protocol": protocol,
        "bound_code_sha256": bound_code_sha256(),
        "inputs": _input_receipt(
            args=args,
            checkpoint_sha256=checkpoint_sha256,
            view=view,
        ),
        "runtime_dependencies": dependencies,
        "outputs": {
            "gpu_ready": str(
                args.smoke_output.resolve() / "gpu_ready.json"
            ),
            "smoke_directory": str(args.smoke_output.resolve()),
            "run_directory": str(args.run_output.resolve()),
        },
        "commands": {
            "required_environment": {
                "OLMO_FAR_PASS_CHORD_GPU_AUTHORIZED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_CACHE_DIR": "<persistent-data-disk-path>",
            },
            "smoke": command(mode="smoke", args=args),
            "train_after_smoke_passes": command(mode="train", args=args),
        },
        "stop_conditions": [
            "short Native route is not bitwise deterministic",
            "Flash-only BF16 augmented D160 attention is ineligible",
            "active loss or gradients are non-finite or zero",
            "augmented prefill/decode cache width differs",
            "full training protocol or input hash differs from this receipt",
        ],
    }
    atomic_json(args.prepared_output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
