#!/usr/bin/env python3
"""Adapt Native OLMo-2 to full EVQ with a gradual 4K frequency morph."""

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
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    logits_from_hidden,
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

from .train_4k_stage_a import composite_checkpoint_sha256
from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


LENGTH = 4_096
READY_STATUS = "OLMO2_4K_PROGRESSIVE_MORPH_READY"
RESULT_STATUS = "OLMO2_4K_PROGRESSIVE_MORPH_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--longalign-view", type=Path, required=True)
    parser.add_argument("--tulu-view", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_500)
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
    parser.add_argument(
        "--retention-weight",
        type=float,
        default=0.0,
        help="Sparse Native-teacher KL weight; zero disables the teacher.",
    )
    parser.add_argument("--retention-positions", type=int, default=8)
    parser.add_argument("--retention-temperature", type=float, default=1.0)
    parser.add_argument(
        "--longalign-objective",
        choices=("full", "assistant"),
        default="full",
        help="Supervision applied to LongAlign rows.",
    )
    parser.add_argument(
        "--morph-schedule",
        choices=("linear", "immediate"),
        default="linear",
        help="Frequency schedule from Native to the full EVQ endpoint.",
    )
    parser.add_argument(
        "--family-pattern",
        choices=("balanced", "longalign_3_to_1"),
        default="balanced",
        help="Optimizer-step mixture of LongAlign-like and Tulu rows.",
    )
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_729)
    return parser.parse_args()


def protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "steps": int(args.steps),
        "maximum_training_length": LENGTH,
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "compile_mode": str(args.compile_mode),
        "retention_weight": float(args.retention_weight),
        "retention_positions": int(args.retention_positions),
        "retention_temperature": float(
            args.retention_temperature
        ),
        "longalign_objective": str(args.longalign_objective),
        "family_pattern": str(args.family_pattern),
        "family_cycle": (
            [
                f"longalign_{args.longalign_objective}",
                "tulu_assistant",
                f"longalign_{args.longalign_objective}",
                (
                    "tulu_native_kl"
                    if float(args.retention_weight) > 0.0
                    else "tulu_assistant"
                ),
            ]
            if str(args.family_pattern) == "balanced"
            else [
                f"longalign_{args.longalign_objective}",
                f"longalign_{args.longalign_objective}",
                f"longalign_{args.longalign_objective}",
                (
                    "tulu_native_kl"
                    if float(args.retention_weight) > 0.0
                    else "tulu_assistant"
                ),
            ]
        ),
        "morph": f"{args.morph_schedule}_native_to_full_evq",
        "natural_eval_rows": int(args.natural_eval_rows),
        "seed": int(args.seed),
        "explicit_custom_binding_or_ruler_training_rows": 0,
    }


def validate_ready(
    *,
    receipt_path: Path,
    checkpoint: Path,
    longalign_view: Path,
    tulu_view: Path,
    background: Path,
    output: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("progressive-morph READY status drift")
    if receipt.get("protocol") != protocol:
        raise RuntimeError("progressive-morph protocol drift")
    expected_paths = {
        "checkpoint": checkpoint,
        "longalign_view": longalign_view,
        "tulu_view": tulu_view,
        "background": background,
    }
    for name, path in expected_paths.items():
        entry = receipt["inputs"][name]
        if Path(entry["path"]).resolve() != path.resolve():
            raise RuntimeError(f"READY {name} path drift")
    if Path(receipt["output"]).resolve() != output.resolve():
        raise RuntimeError("READY output path drift")
    if (
        receipt["inputs"]["checkpoint"]["composite_sha256"]
        != composite_checkpoint_sha256(checkpoint)
    ):
        raise RuntimeError("checkpoint hash drift")
    for name, path in (
        ("longalign_view", longalign_view),
        ("tulu_view", tulu_view),
        ("background", background),
    ):
        manifest = path / "manifest.json"
        if (
            sha256_file(manifest)
            != receipt["inputs"][name]["manifest_sha256"]
        ):
            raise RuntimeError(f"{name} manifest hash drift")
    trainer = Path(__file__).resolve()
    if (
        Path(receipt["trainer"]["path"]).resolve() != trainer
        or receipt["trainer"]["sha256"] != sha256_file(trainer)
    ):
        raise RuntimeError("trainer changed after READY")
    return receipt


def select_rows(
    *,
    pool: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> np.ndarray:
    selection = torch.randint(
        len(pool), (int(count),), generator=generator
    )
    return pool[selection].numpy()


def retention_position_tensor(
    *,
    lengths: np.ndarray,
    positions_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    if int(positions_per_row) <= 0:
        raise ValueError("retention positions must be positive")
    fractions = torch.arange(
        1,
        int(positions_per_row) + 1,
        device=device,
        dtype=torch.float32,
    ) / float(int(positions_per_row) + 1)
    last_context_index = torch.from_numpy(
        np.maximum(lengths.astype(np.int64) - 2, 0)
    ).to(device=device, dtype=torch.float32)
    positions = torch.floor(
        last_context_index[:, None] * fractions[None, :]
    ).to(dtype=torch.long)
    return positions.clamp_(0, LENGTH - 2)


@torch.no_grad()
def native_teacher_probabilities(
    *,
    teacher: Any,
    context: torch.Tensor,
    positions: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = teacher.model(
            input_ids=context,
            use_cache=False,
            return_dict=False,
        )[0]
        row_ids = torch.arange(
            hidden.shape[0], device=hidden.device
        )[:, None]
        selected = hidden[row_ids, positions]
        logits = logits_from_hidden(
            teacher, selected, None
        ).float()
    probabilities = torch.softmax(
        logits / float(temperature), dim=-1
    )
    del hidden, selected, logits
    return probabilities


def set_morph_frequency(
    *,
    model: Any,
    native: torch.Tensor,
    evq: torch.Tensor,
    progress: float,
) -> None:
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError("morph progress escaped [0, 1]")
    realized = torch.lerp(native, evq, float(progress))
    rotary = model.model.rotary_emb
    rotary.inv_freq.copy_(
        realized.to(
            device=rotary.inv_freq.device,
            dtype=rotary.inv_freq.dtype,
        )
    )
    rotary.original_inv_freq = rotary.inv_freq


def train(
    *,
    model: Any,
    teacher: Any | None,
    longalign: Any,
    tulu: Any,
    native_frequency: torch.Tensor,
    evq_frequency: torch.Tensor,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    compile_mode: str,
    retention_weight: float,
    retention_positions: int,
    retention_temperature: float,
    longalign_objective: str,
    morph_schedule: str,
    family_pattern: str,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if float(retention_weight) > 0.0 and teacher is None:
        raise RuntimeError("retention weight requires a Native teacher")
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("progressive morph has no trainable parameters")
    warmup_steps = max(
        1, int(round(float(warmup_ratio) * int(steps)))
    )
    longalign_pool = torch.from_numpy(
        longalign.training_rows.copy()
    )
    tulu_pool = torch.from_numpy(tulu.training_rows.copy())
    if np.any(longalign.lengths[longalign.training_rows] != LENGTH):
        raise RuntimeError("LongAlign morph rows must all be exactly 4K")

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
    generator.manual_seed(int(seed) + 73_001)
    if longalign_objective not in {"full", "assistant"}:
        raise ValueError("invalid LongAlign objective")
    if morph_schedule not in {"linear", "immediate"}:
        raise ValueError("invalid morph schedule")
    if family_pattern == "balanced":
        family_cycle = (
            f"longalign_{longalign_objective}",
            "tulu_assistant",
            f"longalign_{longalign_objective}",
            (
                "tulu_native_kl"
                if float(retention_weight) > 0.0
                else "tulu_assistant"
            ),
        )
    elif family_pattern == "longalign_3_to_1":
        family_cycle = (
            f"longalign_{longalign_objective}",
            f"longalign_{longalign_objective}",
            f"longalign_{longalign_objective}",
            (
                "tulu_native_kl"
                if float(retention_weight) > 0.0
                else "tulu_assistant"
            ),
        )
    else:
        raise ValueError("invalid family pattern")

    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    supervised_tokens = 0
    retention_vectors = 0
    recent_losses: list[float] = []
    recent_ce: list[float] = []
    recent_kl: list[float] = []
    frequency_checkpoints: dict[str, dict[str, Any]] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(steps) + 1):
        progress = (
            float(step) / float(steps)
            if morph_schedule == "linear"
            else 1.0
        )
        set_morph_frequency(
            model=model,
            native=native_frequency,
            evq=evq_frequency,
            progress=progress,
        )
        if step in {1, max(1, steps // 2), steps}:
            frequency_checkpoints[str(step)] = {
                "progress": progress,
                "sha256_float32": tensor_sha256(
                    model.model.rotary_emb.inv_freq
                ),
            }
        lr = cosine_lr(
            step, int(steps), warmup_steps, float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        family = family_cycle[(step - 1) % len(family_cycle)]
        raw_step_losses: list[float] = []
        raw_step_ce: list[float] = []
        raw_step_kl: list[float] = []
        step_supervised = 0

        for _ in range(int(gradient_accumulation_steps)):
            is_longalign = family.startswith("longalign_")
            view = longalign if is_longalign else tulu
            objective = (
                longalign_objective
                if is_longalign
                else "assistant"
            )
            pool = (
                longalign_pool
                if is_longalign
                else tulu_pool
            )
            indices = select_rows(
                pool=pool,
                count=int(micro_batch_size),
                generator=generator,
            )
            context, labels, local_supervised = natural_batch(
                view=view,
                indices=indices,
                objective=objective,
            )
            teacher_probabilities = None
            positions = None
            if family == "tulu_native_kl":
                positions = retention_position_tensor(
                    lengths=np.asarray(
                        view.lengths[indices], dtype=np.int64
                    ),
                    positions_per_row=int(retention_positions),
                    device=context.device,
                )
                teacher_probabilities = native_teacher_probabilities(
                    teacher=teacher,
                    context=context,
                    positions=positions,
                    temperature=float(retention_temperature),
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(context)
                raw_ce = loss_module(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                raw_ce = (
                    raw_ce.loss
                    if hasattr(raw_ce, "loss")
                    else raw_ce
                )
            raw_kl = torch.zeros(
                (), device=context.device, dtype=torch.float32
            )
            if teacher_probabilities is not None:
                row_ids = torch.arange(
                    hidden.shape[0], device=hidden.device
                )[:, None]
                selected = hidden[row_ids, positions]
                student_logits = logits_from_hidden(
                    model, selected, None
                ).float()
                student_log_probabilities = torch.log_softmax(
                    student_logits / float(retention_temperature),
                    dim=-1,
                )
                raw_kl = F.kl_div(
                    student_log_probabilities,
                    teacher_probabilities,
                    reduction="batchmean",
                ) * float(retention_temperature) ** 2
                retention_vectors += int(positions.numel())
                del selected, student_logits
                del student_log_probabilities

            raw_loss = raw_ce + float(retention_weight) * raw_kl
            loss = raw_loss / float(gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite progressive-morph loss at step {step}"
                )
            loss.backward()
            raw_step_losses.append(float(raw_loss.detach()))
            raw_step_ce.append(float(raw_ce.detach()))
            raw_step_kl.append(float(raw_kl.detach()))
            step_supervised += int(local_supervised)
            processed_tokens += int(context.numel())
            del context, labels, hidden, raw_ce, raw_kl, raw_loss, loss
            del teacher_probabilities, positions

        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        supervised_tokens += step_supervised
        mean_loss = float(np.mean(raw_step_losses))
        mean_ce = float(np.mean(raw_step_ce))
        mean_kl = float(np.mean(raw_step_kl))
        recent_losses.append(mean_loss)
        recent_ce.append(mean_ce)
        recent_kl.append(mean_kl)

        if step == 1 or step % 25 == 0 or step == int(steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "total_steps": int(steps),
                    "morph_progress": progress,
                    "family": family,
                    "loss": mean_loss,
                    "ce_loss": mean_ce,
                    "retention_kl": mean_kl,
                    "mean_loss_last_25": float(
                        np.mean(recent_losses[-25:])
                    ),
                    "mean_ce_last_25": float(
                        np.mean(recent_ce[-25:])
                    ),
                    "mean_kl_last_25": float(
                        np.mean(recent_kl[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "processed_input_tokens": processed_tokens,
                    "supervised_tokens": supervised_tokens,
                    "retention_vectors": retention_vectors,
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

    set_morph_frequency(
        model=model,
        native=native_frequency,
        evq=evq_frequency,
        progress=1.0,
    )
    final_sha = tensor_sha256(model.model.rotary_emb.inv_freq)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "objective": (
            f"{morph_schedule}_frequency_morph_with_natural_ce"
        ),
        "steps": int(steps),
        "morph": f"{morph_schedule}_native_to_full_evq",
        "family_cycle": list(family_cycle),
        "processed_input_tokens": int(processed_tokens),
        "supervised_tokens": int(supervised_tokens),
        "retention_vectors": int(retention_vectors),
        "retention_weight": float(retention_weight),
        "retention_positions": int(retention_positions),
        "retention_temperature": float(retention_temperature),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": int(
            micro_batch_size * gradient_accumulation_steps
        ),
        "learning_rate": float(learning_rate),
        "warmup_steps": int(warmup_steps),
        "compile_mode": str(compile_mode),
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": float(elapsed),
        "tokens_per_second": processed_tokens / elapsed,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "frequency_checkpoints": frequency_checkpoints,
        "final_frequency_sha256_float32": final_sha,
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "loss_backend": "liger_fused_linear_cross_entropy",
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.warmup_ratio) < 1.0:
        raise ValueError("warmup ratio must be in (0, 1)")
    if float(args.retention_weight) < 0.0:
        raise ValueError("retention weight must be nonnegative")
    if float(args.retention_temperature) <= 0.0:
        raise ValueError("retention temperature must be positive")
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    longalign_path = args.longalign_view.resolve()
    tulu_path = args.tulu_view.resolve()
    background = args.background_dir.resolve()
    ready_path = args.ready_receipt.resolve()
    protocol = protocol_from_args(args)
    ready = validate_ready(
        receipt_path=ready_path,
        checkpoint=checkpoint,
        longalign_view=longalign_path,
        tulu_view=tulu_path,
        background=background,
        output=output,
        protocol=protocol,
    )

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
    incomplete.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()

    longalign = load_fixed_view(longalign_path)
    tulu = load_fixed_view(tulu_path)
    if (
        tuple(longalign.input_ids.shape)[1] != LENGTH
        or tuple(tulu.input_ids.shape)[1] != LENGTH
    ):
        raise RuntimeError("progressive morph requires fixed 4K views")

    model = load_model(checkpoint)
    native_frequency = (
        model.model.rotary_emb.inv_freq.detach().clone().float()
    )
    frequency = apply_frequency(model, "evq")
    evq_frequency = (
        model.model.rotary_emb.inv_freq.detach().clone().float()
    )
    set_morph_frequency(
        model=model,
        native=native_frequency,
        evq=evq_frequency,
        progress=0.0,
    )
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("progressive morph does not admit a readout")
    model.to("cuda")

    teacher = None
    if float(args.retention_weight) > 0.0:
        teacher = load_model(checkpoint)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        teacher.to("cuda")

    training = train(
        model=model,
        teacher=teacher,
        longalign=longalign,
        tulu=tulu,
        native_frequency=native_frequency,
        evq_frequency=evq_frequency,
        steps=int(args.steps),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        compile_mode=str(args.compile_mode),
        retention_weight=float(args.retention_weight),
        retention_positions=int(args.retention_positions),
            retention_temperature=float(
                args.retention_temperature
            ),
            longalign_objective=str(args.longalign_objective),
            morph_schedule=str(args.morph_schedule),
            family_pattern=str(args.family_pattern),
            seed=int(args.seed),
        log_path=incomplete / "train_log.jsonl",
    )
    if (
        training["final_frequency_sha256_float32"]
        != frequency["evq_sha256_float32"]
    ):
        raise RuntimeError("final trained frequency is not full EVQ")
    del teacher
    torch.cuda.empty_cache()

    metadata = {
        "base_checkpoint_sha256": ready["inputs"]["checkpoint"][
            "composite_sha256"
        ],
        "frequency": "evq",
        "frequency_sha256_float32": frequency[
            "evq_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "adaptation_description": (
            f"qkvo_r{int(args.rank)}_alpha{float(args.alpha):g}"
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
        "stage": f"{args.morph_schedule}_native_to_evq_morph",
        "training_view_sha256": sha256_file(
            longalign_path / "manifest.json"
        ),
        "tulu_view_sha256": sha256_file(
            tulu_path / "manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha = save_adapter(
        incomplete / "adapter.pt", model, None, metadata
    )
    training_complete = {
        "status": "OLMO2_4K_PROGRESSIVE_MORPH_TRAINING_COMPLETE",
        "metric_boundary": (
            "4K-only gradual Native-to-full-EVQ adaptation using "
            "LongAlign/Tulu; no custom binding or RULER rows."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ready_receipt_sha256": sha256_file(ready_path),
        "checkpoint": str(checkpoint),
        "adapter_sha256": adapter_sha,
        "runtime": {
            **runtime,
            "compile_cache": cache,
            "allocator": allocator,
        },
        "frequency": frequency,
        "protocol": protocol,
        "training": training,
    }
    atomic_json(
        incomplete / "training_complete.json",
        training_complete,
    )
    incomplete.replace(output)

    natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=background,
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    result = {
        **training_complete,
        "status": RESULT_STATUS,
        "natural_nll": natural_nll,
    }
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
