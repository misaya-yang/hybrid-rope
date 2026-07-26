#!/usr/bin/env python3
"""Recover instruction following with one frozen-view 4K Tulu-only pass."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    save_adapter,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    seed_everything,
    sha256_file,
)

from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


LENGTH = 4_096
READY_STATUS = "OLMO2_4K_TULU_RECOVERY_READY"
RESULT_STATUS = "OLMO2_4K_TULU_RECOVERY_COMPLETE"


def validate_ready(
    *,
    receipt_path: Path,
    checkpoint: Path,
    parent_adapter: Path,
    tulu_view: Path,
    background: Path,
    evaluator: Path,
    runtime_protocol: dict[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("Tulu-recovery READY status drift")
    inputs = receipt["inputs"]
    expected_paths = {
        "checkpoint": checkpoint,
        "parent_adapter": parent_adapter,
        "tulu_view": tulu_view,
        "background": background,
    }
    for name, path in expected_paths.items():
        if Path(inputs[name]["path"]).resolve() != path.resolve():
            raise RuntimeError(f"READY receipt {name} path drift")
    for relative, expected in inputs["checkpoint"]["files"].items():
        path = checkpoint / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"checkpoint file drift: {path}")
    if (
        parent_adapter.stat().st_size
        != int(inputs["parent_adapter"]["bytes"])
        or sha256_file(parent_adapter)
        != inputs["parent_adapter"]["sha256"]
    ):
        raise RuntimeError("parent adapter drift")
    for relative, expected in inputs["tulu_view"]["files"].items():
        path = tulu_view / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"Tulu view file drift: {path}")
    for relative, expected in inputs["background"]["files"].items():
        path = background / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"background file drift: {path}")
    trainer = Path(__file__).resolve()
    if (
        Path(receipt["code"]["trainer"]["path"]).resolve() != trainer
        or receipt["code"]["trainer"]["sha256"]
        != sha256_file(trainer)
    ):
        raise RuntimeError("trainer changed after READY receipt")
    if (
        Path(receipt["code"]["evaluator"]["path"]).resolve()
        != evaluator.resolve()
        or receipt["code"]["evaluator"]["sha256"]
        != sha256_file(evaluator)
    ):
        raise RuntimeError("evaluator changed after READY receipt")
    code_root = Path(receipt["code"]["root"]).resolve()
    for relative, expected in receipt["code"][
        "critical_files"
    ].items():
        path = code_root / relative
        if (
            path.stat().st_size != int(expected["bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"critical code drift: {path}")
    frozen_protocol = receipt["protocol"]
    for name, actual in runtime_protocol.items():
        if frozen_protocol.get(name) != actual:
            raise RuntimeError(
                f"runtime protocol drift for {name}: "
                f"{actual!r} != {frozen_protocol.get(name)!r}"
            )
    current_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "liger_kernel": importlib.metadata.version("liger-kernel"),
    }
    for name, actual in current_environment.items():
        if receipt["environment"].get(name) != actual:
            raise RuntimeError(
                f"runtime environment drift for {name}: "
                f"{actual!r} != "
                f"{receipt['environment'].get(name)!r}"
            )
    return receipt


@torch.no_grad()
def evaluate_assistant_nll(
    *,
    model: Any,
    view: Any,
    rows: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    if len(rows) == 0:
        raise RuntimeError("Tulu validation split is empty")
    model.eval()
    loss_module = fused_loss_module()
    total_nll = 0.0
    total_tokens = 0
    backbone = TrainingBackbone(model.model)
    for offset in range(0, len(rows), int(batch_size)):
        indices = rows[offset : offset + int(batch_size)]
        context, labels, supervised = natural_batch(
            view=view,
            indices=indices,
            objective="assistant",
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(context)
            value = loss_module(
                model.lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )
            value = value.loss if hasattr(value, "loss") else value
        if not torch.isfinite(value):
            raise RuntimeError("non-finite Tulu validation NLL")
        total_nll += float(value) * int(supervised)
        total_tokens += int(supervised)
        del context, labels, hidden, value
    mean_nll = total_nll / total_tokens
    return {
        "rows": int(len(rows)),
        "supervised_tokens": int(total_tokens),
        "mean_nll": float(mean_nll),
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
    }


def train_one_or_more_epochs(
    *,
    model: Any,
    view: Any,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
    compile_mode: str,
    log_path: Path,
) -> dict[str, Any]:
    training_rows = torch.from_numpy(view.training_rows.copy())
    global_batch = int(
        micro_batch_size * gradient_accumulation_steps
    )
    if len(training_rows) % global_batch != 0:
        raise RuntimeError(
            "clean one-pass protocol requires an exactly divisible split"
        )
    steps_per_epoch = len(training_rows) // global_batch
    total_steps = int(epochs) * steps_per_epoch
    warmup_steps = max(
        1, int(round(float(warmup_ratio) * total_steps))
    )
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=compile_mode,
    )
    loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 61_003)
    permutations = [
        training_rows[torch.randperm(len(training_rows), generator=generator)]
        for _ in range(int(epochs))
    ]
    ordered_rows = torch.cat(permutations)
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    supervised_tokens = 0
    recent: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, total_steps + 1):
        lr = cosine_lr(
            step,
            total_steps,
            warmup_steps,
            float(learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_rows = ordered_rows[
            (step - 1) * global_batch : step * global_batch
        ]
        micro_batches = []
        step_supervised = 0
        for micro in range(int(gradient_accumulation_steps)):
            indices = batch_rows[
                micro * int(micro_batch_size)
                : (micro + 1) * int(micro_batch_size)
            ].numpy()
            context, labels, supervised = natural_batch(
                view=view,
                indices=indices,
                objective="assistant",
            )
            micro_batches.append((context, labels, int(supervised)))
            step_supervised += int(supervised)
        if step_supervised <= 0:
            raise RuntimeError(
                f"Tulu recovery step {step} has no supervised tokens"
            )
        weighted_step_nll = 0.0
        for context, labels, supervised in micro_batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(context)
                value = loss_module(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                value = value.loss if hasattr(value, "loss") else value
                weight = float(supervised) / float(step_supervised)
                loss = value * weight
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite Tulu recovery loss at step {step}"
                )
            loss.backward()
            weighted_step_nll += float(value.detach()) * weight
            processed_tokens += int(context.numel())
            del context, labels, hidden, value, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        supervised_tokens += step_supervised
        recent.append(float(weighted_step_nll))

        if step == 1 or step % 25 == 0 or step == total_steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": 1 + (step - 1) // steps_per_epoch,
                    "loss": recent[-1],
                    "mean_loss_last_25": float(
                        np.mean(recent[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "processed_input_tokens": processed_tokens,
                    "supervised_tokens": supervised_tokens,
                    "elapsed_seconds": now - started,
                    "interval_tokens_per_second": (
                        (processed_tokens - last_log_tokens)
                        / max(now - last_log_time, 1e-9)
                    ),
                    "peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
            )
            last_log_time = now
            last_log_tokens = processed_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "objective": "official_tulu3_assistant_content_and_eos_only",
        "epochs": int(epochs),
        "steps": int(total_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "training_rows": int(len(training_rows)),
        "processed_input_tokens": int(processed_tokens),
        "supervised_tokens": int(supervised_tokens),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": int(global_batch),
        "learning_rate": float(learning_rate),
        "warmup_steps": int(warmup_steps),
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": float(elapsed),
        "tokens_per_second": processed_tokens / elapsed,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "loss_backend": "liger_fused_linear_cross_entropy",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--tulu-view", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.epochs) != 1:
        raise ValueError("the frozen clean-arm protocol is exactly one epoch")
    if not 0.0 < float(args.warmup_ratio) < 1.0:
        raise ValueError("warmup ratio must be in (0, 1)")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    incomplete_output = output.with_name(output.name + ".incomplete")
    if incomplete_output.exists():
        raise FileExistsError(incomplete_output)
    checkpoint = args.checkpoint.resolve()
    parent_adapter = args.parent_adapter.resolve()
    tulu_path = args.tulu_view.resolve()
    background = args.background_dir.resolve()
    evaluator = args.evaluator.resolve()
    ready_path = args.ready_receipt.resolve()
    runtime_protocol = {
        "epochs": int(args.epochs),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "compile_mode": args.compile_mode,
        "natural_eval_rows": int(args.natural_eval_rows),
        "seed": int(args.seed),
    }
    ready = validate_ready(
        receipt_path=ready_path,
        checkpoint=checkpoint,
        parent_adapter=parent_adapter,
        tulu_view=tulu_path,
        background=background,
        evaluator=evaluator,
        runtime_protocol=runtime_protocol,
    )
    view = load_fixed_view(tulu_path)
    if tuple(view.input_ids.shape)[1] != LENGTH:
        raise RuntimeError("Tulu recovery requires exact 4K rows")
    validation_rows = np.flatnonzero(view.split == 1)

    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if not cache:
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")
    incomplete_output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    gpu_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if not any(
        marker in gpu_name
        for marker in ready["protocol"]["allowed_gpu_name_substrings"]
    ):
        raise RuntimeError(f"unexpected GPU for clean arm: {gpu_name}")
    if capability != ready["protocol"][
        "required_compute_capability"
    ]:
        raise RuntimeError(
            f"compute-capability drift: {capability}"
        )
    model = load_model(checkpoint)
    frequency = apply_frequency(model, "evq")
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("Tulu recovery does not admit a readout")
    parent_metadata = load_adapter(parent_adapter, model, None)
    if (
        parent_metadata.get("frequency") != "evq"
        or parent_metadata.get("frequency_sha256_float32")
        != frequency["active_sha256_float32"]
        or int(parent_metadata.get("training_sequence_length", -1))
        != LENGTH
    ):
        raise RuntimeError("parent is not the frozen 4K EVQ Stage-A adapter")
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
        log_path=incomplete_output / "train_log.jsonl",
    )
    metadata = {
        "base_checkpoint_sha256": ready["inputs"]["checkpoint"][
            "composite_sha256"
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
        "stage": "tulu_recovery",
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "training_view_sha256": sha256_file(
            tulu_path / "manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha = save_adapter(
        incomplete_output / "adapter.pt", model, None, metadata
    )
    training_receipt = {
        "status": "OLMO2_4K_TULU_RECOVERY_TRAINING_COMPLETE",
        "metric_boundary": (
            "One deterministic pass over the frozen 3,968-row official-"
            "source Tulu view; no explicit custom binding or RULER "
            "benchmark rows were added."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ready_receipt_sha256": sha256_file(ready_path),
        "checkpoint": str(checkpoint),
        "parent_adapter": str(parent_adapter),
        "adapter_sha256": adapter_sha,
        "runtime": {
            **runtime,
            "gpu_name": gpu_name,
            "compute_capability": capability,
            "compile_cache": cache,
            "allocator": allocator,
        },
        "frequency": frequency,
        "training": training,
        "tulu_validation_before": validation_before,
        "protocol": {
            "maximum_training_length": LENGTH,
            "training_data": (
                "allenai/tulu-3-sft-olmo-2-mixture-0225"
            ),
            "training_objective": (
                "assistant_content_and_eos_only"
            ),
            "explicit_custom_binding_or_ruler_training_rows": 0,
            "frozen_view_passes": int(args.epochs),
            "seed": int(args.seed),
        },
    }
    atomic_json(
        incomplete_output / "training_complete.json",
        training_receipt,
    )
    incomplete_output.replace(output)

    validation_after = evaluate_assistant_nll(
        model=model,
        view=view,
        rows=validation_rows,
        batch_size=int(args.micro_batch_size),
    )
    natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=background,
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    result = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Strictly <=4K, one deterministic pass over a frozen 3,968-"
            "row official-source Tulu assistant-only view after LongAlign-"
            "only Stage A; no explicit custom binding or RULER benchmark "
            "rows were added"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ready_receipt_sha256": sha256_file(ready_path),
        "checkpoint": str(checkpoint),
        "parent_adapter": str(parent_adapter),
        "adapter_sha256": adapter_sha,
        "runtime": runtime,
        "frequency": frequency,
        "training": training,
        "tulu_validation_before": validation_before,
        "tulu_validation_after": validation_after,
        "natural_nll": natural_nll,
        "protocol": {
            "maximum_training_length": LENGTH,
            "training_data": (
                "allenai/tulu-3-sft-olmo-2-mixture-0225"
            ),
            "training_objective": (
                "assistant_content_and_eos_only"
            ),
            "explicit_custom_binding_or_ruler_training_rows": 0,
            "frozen_view_passes": int(args.epochs),
            "seed": int(args.seed),
        },
    }
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
