#!/usr/bin/env python3
"""Train or evaluate the 4K-only dense-label OLMo-2 EVQ-LoRA stage."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    save_adapter,
    trainable_named_parameters,
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


def composite_checkpoint_sha256(checkpoint: Path) -> str:
    shards = sorted(checkpoint.glob("model-*.safetensors"))
    if not shards:
        single_file = checkpoint / "model.safetensors"
        if not single_file.is_file():
            raise RuntimeError(
                f"no safetensor weights under {checkpoint}"
            )
        shards = [single_file]
    return ":".join(sha256_file(path) for path in shards)


def ready_checkpoint_digest(checkpoint: Path, receipt_path: Path) -> str:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") not in {
        "OLMO2_STEP30_4K_CONVERSION_READY",
        "OLMO2_INSTRUCT_4K_CONVERSION_READY",
        "OLMO2_INSTRUCT_PRO6000_RULER_MATRIX_READY",
    }:
        raise RuntimeError("4K conversion READY receipt is not valid")
    entry = receipt["checkpoint"]
    if Path(entry["checkpoint_path"]).resolve() != checkpoint.resolve():
        raise RuntimeError("READY receipt names a different checkpoint")
    for name, expected in entry["files"].items():
        path = checkpoint / name
        stat = path.stat()
        if (
            stat.st_size != int(expected["bytes"])
            or stat.st_mtime_ns != int(expected["mtime_ns"])
        ):
            raise RuntimeError(
                f"checkpoint changed after READY receipt: {path}"
            )
    return str(entry["composite_sha256"])


def train_dense_4k(
    *,
    model: Any,
    view_path: Path,
    target_supervised_tokens: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    seed: int,
    compile_mode: str,
    log_path: Path,
) -> dict[str, Any]:
    view = load_fixed_view(view_path)
    if tuple(view.input_ids.shape)[1] != 4_096:
        raise RuntimeError("Stage A requires the frozen 4K view")
    if np.any(view.lengths[view.training_rows] != 4_096):
        raise RuntimeError("Stage A training rows must be exactly 4K")

    named_parameters = trainable_named_parameters(model, None)
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError("Stage A has no trainable LoRA parameters")
    global_batch_size = int(
        micro_batch_size * gradient_accumulation_steps
    )
    labels_per_step = global_batch_size * (4_096 - 1)
    total_steps = int(
        math.ceil(int(target_supervised_tokens) / labels_per_step)
    )
    warmup_steps = max(
        1, int(round(float(warmup_ratio) * total_steps))
    )

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
    row_pool = torch.from_numpy(view.training_rows.copy())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 40_001)

    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_input_tokens = 0
    supervised_tokens = 0
    recent_losses: list[float] = []
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
        raw_losses: list[float] = []
        step_supervised = 0
        for _ in range(int(gradient_accumulation_steps)):
            selection = torch.randint(
                len(row_pool),
                (int(micro_batch_size),),
                generator=generator,
            )
            indices = row_pool[selection].numpy()
            context, labels, local_supervised = natural_batch(
                view=view,
                indices=indices,
                objective="full",
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(context)
                raw_loss = loss_module(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                if hasattr(raw_loss, "loss"):
                    raw_loss = raw_loss.loss
                loss = raw_loss / float(
                    gradient_accumulation_steps
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite Stage-A loss at step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            step_supervised += int(local_supervised)
            processed_input_tokens += int(context.numel())
            del context, labels, hidden, raw_loss, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        supervised_tokens += step_supervised
        mean_loss = float(np.mean(raw_losses))
        recent_losses.append(mean_loss)

        if step == 1 or step % 25 == 0 or step == total_steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            interval_tokens = (
                processed_input_tokens - last_log_tokens
            )
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "total_steps": total_steps,
                    "loss": mean_loss,
                    "mean_loss_last_25": float(
                        np.mean(recent_losses[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "supervised_tokens": supervised_tokens,
                    "processed_input_tokens": processed_input_tokens,
                    "elapsed_seconds": now - started,
                    "interval_tokens_per_second": (
                        interval_tokens
                        / max(now - last_log_time, 1e-9)
                    ),
                    "peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                },
            )
            last_log_time = now
            last_log_tokens = processed_input_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "objective": "full_token_next_token_ce",
        "training_sequence_length": 4_096,
        "steps": total_steps,
        "target_supervised_tokens": int(target_supervised_tokens),
        "actual_supervised_tokens": supervised_tokens,
        "processed_input_tokens": processed_input_tokens,
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": global_batch_size,
        "learning_rate": float(learning_rate),
        "warmup_steps": warmup_steps,
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_input_tokens / elapsed,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "loss_backend": "liger_fused_linear_cross_entropy",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("base", "train"), required=True
    )
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument(
        "--target-supervised-tokens", type=int, default=20_000_000
    )
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--natural-tail-tokens", type=int, default=1_024)
    parser.add_argument("--skip-natural-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.mode == "base" and int(args.target_supervised_tokens) != 0:
        raise ValueError("base mode requires target-supervised-tokens 0")
    if not 0.0 < float(args.warmup_ratio) < 1.0:
        raise ValueError("warmup ratio must be in (0, 1)")
    output.mkdir(parents=True)

    seed_everything(int(args.seed))
    runtime = configure_cuda()
    checkpoint = args.checkpoint.resolve()
    prepared = args.prepared_data.resolve()
    background = args.background_dir.resolve()
    ready_receipt = args.ready_receipt.resolve()
    view_path = prepared / "longalign_paired_L4096"

    model = load_model(checkpoint)
    frequency = apply_frequency(model, args.frequency)
    adaptation = "none"
    adaptation_description = "none"
    if args.mode == "train":
        readout = install_adaptation(
            model,
            "qkvo_answer",
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        if readout is not None:
            raise RuntimeError("Stage A must not add a readout")
        adaptation = "qkvo_answer"
        adaptation_description = (
            f"qkvo_r{int(args.rank)}_alpha{float(args.alpha):g}"
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.to("cuda")

    training = None
    if args.mode == "train":
        training = train_dense_4k(
            model=model,
            view_path=view_path,
            target_supervised_tokens=int(
                args.target_supervised_tokens
            ),
            micro_batch_size=int(args.micro_batch_size),
            gradient_accumulation_steps=int(
                args.gradient_accumulation_steps
            ),
            learning_rate=float(args.learning_rate),
            warmup_ratio=float(args.warmup_ratio),
            seed=int(args.seed),
            compile_mode=args.compile_mode,
            log_path=output / "train_log.jsonl",
        )

    natural_nll = (
        {}
        if args.skip_natural_eval
        else evaluate_natural_nll(
            model=model,
            background_dir=background,
            lengths=(4_096, 8_192, 16_384),
            rows=int(args.natural_eval_rows),
            tail_tokens=int(args.natural_tail_tokens),
        )
    )
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": adaptation,
        "adaptation_description": adaptation_description,
        "rank": int(args.rank) if args.mode == "train" else None,
        "alpha": float(args.alpha) if args.mode == "train" else None,
        "training_sequence_length": (
            4_096 if args.mode == "train" else None
        ),
        "training_view_sha256": sha256_file(
            view_path / "manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha256 = (
        save_adapter(
            output / "adapter.pt",
            model,
            None,
            adapter_metadata,
        )
        if args.mode == "train"
        else None
    )
    receipt = {
        "status": "OLMO2_4K_STAGE_A_COMPLETE",
        "metric_boundary": (
            "4K-only full-token adaptation with 4K/8K/16K natural-text "
            "NLL evaluation; no downstream capability claim"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "mode": args.mode,
        "frequency": frequency,
        "adaptation": adaptation,
        "adapter_sha256": adapter_sha256,
        "runtime": runtime,
        "training": training,
        "natural_nll": natural_nll,
        "protocol": {
            "maximum_training_length": (
                4_096 if args.mode == "train" else None
            ),
            "target_supervised_tokens": int(
                args.target_supervised_tokens
            ),
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "rank": int(args.rank) if args.mode == "train" else None,
            "alpha": (
                float(args.alpha) if args.mode == "train" else None
            ),
            "learning_rate": float(args.learning_rate),
            "warmup_ratio": float(args.warmup_ratio),
            "compile_mode": args.compile_mode,
            "training_view": str(view_path),
            "natural_eval_lengths": [4_096, 8_192, 16_384],
            "natural_evaluation_skipped": bool(
                args.skip_natural_eval
            ),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "training": training,
                "natural_nll": natural_nll,
                "adapter_sha256": adapter_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
