#!/usr/bin/env python3
"""Create the no-GPU preparation receipt for far-only EVQ residual training."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .far_only_evq_residual import METHOD_ID, PREPARED_STATUS
from .phase_adaptation import PhaseAdaptationView
from .train_4k_far_only_evq_residual import (
    bound_code_sha256,
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
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--prepared-output", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    parser.add_argument("--projection-rank", type=int, default=64)
    parser.add_argument("--residual-head-dim", type=int, default=128)
    parser.add_argument("--initial-logit-gain", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
    parser.add_argument("--validation-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_804)
    return parser.parse_args()


def _trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        steps=int(args.steps),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        projection_rank=int(args.projection_rank),
        residual_head_dim=int(args.residual_head_dim),
        initial_logit_gain=float(args.initial_logit_gain),
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
    view: PhaseAdaptationView,
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
        "python",
        "-m",
        TRAINER_MODULE,
        "--mode",
        mode,
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
        "--residual-head-dim",
        str(args.residual_head_dim),
        "--initial-logit-gain",
        str(args.initial_logit_gain),
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
    if int(args.steps) != 100:
        raise ValueError("registered first run requires exactly 100 steps")
    if (
        int(args.micro_batch_size) != 1
        or int(args.gradient_accumulation_steps) != 4
    ):
        raise ValueError("registered first run requires micro 1 / accum 4")
    if (
        int(args.projection_rank) != 64
        or int(args.residual_head_dim) != 128
    ):
        raise ValueError("registered first run requires rank 64 / D128")
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
    view = PhaseAdaptationView(args.training_view.resolve())
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
    protocol = registered_protocol(_trainer_args(args))
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
            "smoke": command(mode="smoke", args=args),
            "train_after_smoke_passes": command(mode="train", args=args),
        },
        "stop_conditions": [
            "short Native route is not bitwise deterministic",
            "Flash-only BF16 augmented D256 attention is ineligible",
            "active loss or gradients are non-finite or zero",
            "augmented prefill/decode cache width differs",
            "full training protocol or input hash differs from this receipt",
        ],
    }
    atomic_json(args.prepared_output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
