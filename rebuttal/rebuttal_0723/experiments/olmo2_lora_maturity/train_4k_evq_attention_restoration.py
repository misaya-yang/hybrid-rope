#!/usr/bin/env python3
"""Restore 4K attention function under a fixed full-EVQ frequency grid."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    load_model,
    save_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .evq_attention_restoration import (
    FINAL_LAYER_INDEX,
    METHOD_ID,
    PREPARED_STATUS,
    READY_STATUS,
    RESULT_STATUS,
    SEQUENCE_LENGTH,
    RelationCapture,
    configure_relation_capture_attention,
    dense_relation_forward_kl,
    import_linearard_kernel,
    install_qkv_lora,
    relation_loss,
)
from .train_4k_stage_a import composite_checkpoint_sha256
from .train_screen import apply_frequency, load_fixed_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("gpu-smoke", "train"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--linearard-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "teacher_frequency": "native_endpoint_rope",
        "student_frequency": "fixed_full_evq_endpoint_from_step_0",
        "frequency_morph": False,
        "training_length": SEQUENCE_LENGTH,
        "position_ids": "contiguous_0_to_4095",
        "steps": int(args.steps),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "student_tokens": int(
            args.steps
            * args.micro_batch_size
            * args.gradient_accumulation_steps
            * SEQUENCE_LENGTH
        ),
        "teacher_tokens": int(
            args.steps
            * args.micro_batch_size
            * args.gradient_accumulation_steps
            * SEQUENCE_LENGTH
        ),
        "lora_targets": ["q_proj", "k_proj", "v_proj"],
        "lora_layers": "all_16",
        "lora_rank": int(args.rank),
        "lora_alpha": float(args.alpha),
        "lora_dropout": 0.0,
        "distilled_layer": FINAL_LAYER_INDEX,
        "objective": {
            "post_rope_causal_qk_forward_kl": float(
                args.attention_weight
            ),
            "post_attention_context_normalized_mse": float(
                args.context_weight
            ),
            "qq_kk_vv_relation_mean": float(
                args.self_relation_weight
            ),
            "language_model_ce": 0.0,
            "output_logit_kl": 0.0,
        },
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "minimum_lr_ratio": float(args.minimum_lr_ratio),
        "maximum_gradient_norm": float(args.maximum_gradient_norm),
        "compile": False,
        "gradient_checkpointing": False,
        "seed": int(args.seed),
    }


def learning_rate(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum_ratio: float,
) -> float:
    if step <= warmup_steps:
        return peak * float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return peak * (
        float(minimum_ratio) + (1.0 - float(minimum_ratio)) * cosine
    )


def validate_receipt(
    args: argparse.Namespace,
    *,
    expected_status: str,
) -> dict[str, Any]:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != expected_status:
        raise RuntimeError(
            f"expected receipt status {expected_status!r}, "
            f"got {receipt.get('status')!r}"
        )
    protocol = protocol_from_args(args)
    if receipt.get("protocol") != protocol:
        raise RuntimeError("receipt protocol drift")
    expected_paths = {
        "checkpoint": args.checkpoint.resolve(),
        "training_view": args.training_view.resolve(),
        "linearard_root": args.linearard_root.resolve(),
    }
    for name, path in expected_paths.items():
        if Path(receipt["inputs"][name]["path"]).resolve() != path:
            raise RuntimeError(f"receipt {name} path drift")
    if (
        receipt["inputs"]["checkpoint"]["composite_sha256"]
        != composite_checkpoint_sha256(args.checkpoint)
    ):
        raise RuntimeError("checkpoint hash drift")
    for name in ("manifest.json", "input_ids.npy", "lengths.npy", "split.npy"):
        path = args.training_view / name
        if (
            sha256_file(path)
            != receipt["inputs"]["training_view"]["files"][name]["sha256"]
        ):
            raise RuntimeError(f"training-view {name} hash drift")
    source_entries = receipt["source"]
    trainer = Path(__file__).resolve()
    helper = Path(
        __file__
    ).with_name("evq_attention_restoration.py").resolve()
    if (
        source_entries["trainer"]["sha256"] != sha256_file(trainer)
        or source_entries["helper"]["sha256"] != sha256_file(helper)
    ):
        raise RuntimeError("training source changed after receipt")
    return receipt


def deterministic_training_rows(
    *,
    view: Any,
    examples: int,
    seed: int,
) -> np.ndarray:
    rows = np.asarray(view.training_rows, dtype=np.int64)
    if np.any(np.asarray(view.lengths[rows]) != SEQUENCE_LENGTH):
        raise RuntimeError("restoration training requires full 4K rows")
    generator = np.random.default_rng(int(seed))
    ordered: list[np.ndarray] = []
    remaining = int(examples)
    while remaining > 0:
        permutation = generator.permutation(rows)
        take = min(remaining, len(permutation))
        ordered.append(permutation[:take])
        remaining -= take
    return np.concatenate(ordered)


def model_pair(
    *,
    checkpoint: Path,
    rank: int,
    alpha: float,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], int]:
    teacher = load_model(checkpoint)
    student = load_model(checkpoint)
    teacher_frequency = apply_frequency(teacher, "native")
    student_frequency = apply_frequency(student, "evq")
    if (
        teacher_frequency["active_sha256_float32"]
        == student_frequency["active_sha256_float32"]
    ):
        raise RuntimeError("Native teacher and EVQ student frequencies match")
    trainable = install_qkv_lora(
        student, rank=int(rank), alpha=float(alpha)
    )
    trainable_names = [
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable_names) != 96:
        raise RuntimeError(
            f"expected 96 QKV LoRA tensors, got {len(trainable_names)}"
        )
    if any(
        not (
            any(
                marker in name
                for marker in (".q_proj.", ".k_proj.", ".v_proj.")
            )
            and name.endswith((".a", ".b"))
        )
        for name in trainable_names
    ):
        raise RuntimeError("trainable scope escaped QKV LoRA A/B tensors")
    teacher_probe = teacher.model.layers[0].self_attn.q_proj.weight
    student_probe = student.model.layers[0].self_attn.q_proj.base.weight
    if teacher_probe.data_ptr() == student_probe.data_ptr():
        raise RuntimeError("teacher and student unexpectedly share storage")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    student.train()
    teacher.config.use_cache = False
    student.config.use_cache = False
    if float(getattr(teacher.config, "attention_dropout", 0.0)) != 0.0:
        raise RuntimeError("teacher attention dropout is nonzero")
    if float(getattr(student.config, "attention_dropout", 0.0)) != 0.0:
        raise RuntimeError("student attention dropout is nonzero")
    configure_relation_capture_attention(teacher)
    configure_relation_capture_attention(student)
    teacher.to(device)
    student.to(device)
    return (
        teacher,
        student,
        teacher_frequency,
        student_frequency,
        trainable,
    )


def one_loss(
    *,
    teacher: Any,
    student: Any,
    input_ids: torch.Tensor,
    capture: RelationCapture,
    kernel: Any,
    protocol: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    capture.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
            relation_capture=capture,
            relation_mode="teacher",
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
            relation_capture=capture,
            relation_mode="student",
        )
        teacher_capture, student_capture = capture.require_pair()
        expected = (
            int(input_ids.shape[0]),
            16,
            SEQUENCE_LENGTH,
            128,
        )
        for name, value in (
            ("teacher_query", teacher_capture.query),
            ("teacher_key", teacher_capture.key),
            ("teacher_value", teacher_capture.value),
            ("student_query", student_capture.query),
            ("student_key", student_capture.key),
            ("student_value", student_capture.value),
        ):
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"{name} shape drift: {tuple(value.shape)} != {expected}"
                )
        total, components = relation_loss(
            teacher=teacher_capture,
            student=student_capture,
            kernel=kernel,
            attention_weight=protocol["objective"][
                "post_rope_causal_qk_forward_kl"
            ],
            context_weight=protocol["objective"][
                "post_attention_context_normalized_mse"
            ],
            self_relation_weight=protocol["objective"][
                "qq_kk_vv_relation_mean"
            ],
        )
    return total, components


def kernel_parity(kernel: Any, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(91)
    shape = (1, 16, 64, 128)
    student_q = torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    student_k = torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher_q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    teacher_k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(shape[-1])
    custom = kernel(
        student_q,
        student_k,
        teacher_q,
        teacher_k,
        attn_mask=None,
        causal=True,
        sm_scale_s=scale,
        sm_scale_t=scale,
    )
    custom.backward()
    custom_q_grad = student_q.grad.detach().float().clone()
    custom_k_grad = student_k.grad.detach().float().clone()

    reference_q = student_q.detach().float().requires_grad_(True)
    reference_k = student_k.detach().float().requires_grad_(True)
    reference = dense_relation_forward_kl(
        reference_q,
        reference_k,
        teacher_q.float(),
        teacher_k.float(),
        student_scale=scale,
        teacher_scale=scale,
    )
    reference.backward()
    loss_error = abs(float(custom.detach().float() - reference.detach()))
    q_grad_error = float(
        (custom_q_grad - reference_q.grad).abs().max()
    )
    k_grad_error = float(
        (custom_k_grad - reference_k.grad).abs().max()
    )
    if (
        loss_error > 0.05
        or q_grad_error > 0.02
        or k_grad_error > 0.02
    ):
        raise RuntimeError(
            "LinearARD kernel parity failed: "
            f"loss={loss_error}, q_grad={q_grad_error}, "
            f"k_grad={k_grad_error}"
        )
    alias_student = torch.randn(
        shape,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    alias_teacher = torch.randn(
        shape, device=device, dtype=torch.bfloat16
    )
    alias_custom = kernel(
        alias_student,
        alias_student,
        alias_teacher,
        alias_teacher,
        attn_mask=None,
        causal=True,
        sm_scale_s=scale,
        sm_scale_t=scale,
    )
    alias_custom.backward()
    alias_custom_gradient = alias_student.grad.detach().float().clone()
    alias_reference_input = (
        alias_student.detach().float().requires_grad_(True)
    )
    alias_reference = dense_relation_forward_kl(
        alias_reference_input,
        alias_reference_input,
        alias_teacher.float(),
        alias_teacher.float(),
        student_scale=scale,
        teacher_scale=scale,
    )
    alias_reference.backward()
    alias_loss_error = abs(
        float(alias_custom.detach().float() - alias_reference.detach())
    )
    alias_gradient_error = float(
        (
            alias_custom_gradient - alias_reference_input.grad
        ).abs().max()
    )
    if alias_loss_error > 0.05 or alias_gradient_error > 0.02:
        raise RuntimeError(
            "LinearARD alias-gradient parity failed: "
            f"loss={alias_loss_error}, grad={alias_gradient_error}"
        )
    return {
        "shape": list(shape),
        "dtype": "bfloat16",
        "loss_absolute_error": loss_error,
        "q_gradient_max_absolute_error": q_grad_error,
        "k_gradient_max_absolute_error": k_grad_error,
        "alias_loss_absolute_error": alias_loss_error,
        "alias_gradient_max_absolute_error": alias_gradient_error,
        "thresholds": {
            "loss_absolute_error": 0.05,
            "gradient_max_absolute_error": 0.02,
        },
    }


def gpu_smoke(args: argparse.Namespace) -> None:
    prepared = validate_receipt(args, expected_status=PREPARED_STATUS)
    if args.output.exists():
        raise FileExistsError(args.output)
    environment = configure_cuda()
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    kernel = import_linearard_kernel(args.linearard_root)
    parity = kernel_parity(kernel, device)
    view = load_fixed_view(args.training_view)
    rows = deterministic_training_rows(
        view=view,
        examples=int(args.micro_batch_size),
        seed=int(args.seed),
    )
    input_ids = torch.from_numpy(
        np.asarray(view.input_ids[rows], dtype=np.int64)
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    (
        teacher,
        student,
        teacher_frequency,
        student_frequency,
        trainable,
    ) = model_pair(
        checkpoint=args.checkpoint,
        rank=int(args.rank),
        alpha=float(args.alpha),
        device=device,
    )
    capture = RelationCapture()
    started = time.perf_counter()
    total, components = one_loss(
        teacher=teacher,
        student=student,
        input_ids=input_ids,
        capture=capture,
        kernel=kernel,
        protocol=protocol_from_args(args),
    )
    total.backward()
    torch.cuda.synchronize()
    finite_gradients = True
    gradient_rows = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        gradient_rows.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": (
                    gradient is not None
                    and bool(torch.isfinite(gradient).all())
                ),
                "nonzero": (
                    gradient is not None
                    and bool(torch.count_nonzero(gradient))
                ),
            }
        )
    finite_gradients = all(
        row["present"] and row["finite"] for row in gradient_rows
    )
    b_gradients = [
        row for row in gradient_rows if row["name"].endswith(".b")
    ]
    if len(gradient_rows) != 96 or len(b_gradients) != 48:
        raise RuntimeError("GPU smoke trainable-gradient scope drift")
    nonzero_b_gradients = sum(row["nonzero"] for row in b_gradients)
    if not torch.isfinite(total) or not finite_gradients:
        raise RuntimeError("full-model GPU smoke produced non-finite values")
    if nonzero_b_gradients != 48:
        raise RuntimeError(
            "full-model GPU smoke did not reach every QKV LoRA B tensor"
        )
    ready = {
        **prepared,
        "status": READY_STATUS,
        "prepared_receipt": {
            "path": str(args.receipt.resolve()),
            "sha256": sha256_file(args.receipt),
        },
        "gpu_smoke": {
            "environment": environment,
            "kernel_parity": parity,
            "full_model": {
                "batch_shape": list(input_ids.shape),
                "loss": float(total.detach()),
                "components": {
                    name: float(value.detach())
                    for name, value in components.items()
                },
                "finite_gradients": finite_gradients,
                "trainable_gradient_tensors": len(gradient_rows),
                "nonzero_lora_b_gradient_tensors": nonzero_b_gradients,
                "trainable_parameters": trainable,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            },
            "teacher_frequency": teacher_frequency,
            "student_frequency": student_frequency,
        },
    }
    atomic_json(args.output, ready)


def train(args: argparse.Namespace) -> None:
    ready = validate_receipt(args, expected_status=READY_STATUS)
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    incomplete.mkdir(parents=True)
    environment = configure_cuda()
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    protocol = protocol_from_args(args)
    kernel = import_linearard_kernel(args.linearard_root)
    view = load_fixed_view(args.training_view)
    total_examples = int(
        args.steps
        * args.micro_batch_size
        * args.gradient_accumulation_steps
    )
    rows = deterministic_training_rows(
        view=view,
        examples=total_examples,
        seed=int(args.seed),
    )
    (
        teacher,
        student,
        teacher_frequency,
        student_frequency,
        trainable,
    ) = model_pair(
        checkpoint=args.checkpoint,
        rank=int(args.rank),
        alpha=float(args.alpha),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in student.parameters()
            if parameter.requires_grad
        ],
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    capture = RelationCapture()
    log_path = incomplete / "train_log.jsonl"
    row_cursor = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(args.steps) + 1):
        step_started = time.perf_counter()
        lr = learning_rate(
            step=step,
            total_steps=int(args.steps),
            warmup_steps=int(args.warmup_steps),
            peak=float(args.learning_rate),
            minimum_ratio=float(args.minimum_lr_ratio),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        component_totals: dict[str, float] = {}
        total_value = 0.0
        for _ in range(int(args.gradient_accumulation_steps)):
            batch_rows = rows[
                row_cursor : row_cursor + int(args.micro_batch_size)
            ]
            row_cursor += int(args.micro_batch_size)
            input_ids = torch.from_numpy(
                np.asarray(view.input_ids[batch_rows], dtype=np.int64)
            ).to(device)
            total, components = one_loss(
                teacher=teacher,
                student=student,
                input_ids=input_ids,
                capture=capture,
                kernel=kernel,
                protocol=protocol,
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite restoration loss")
            (
                total / float(args.gradient_accumulation_steps)
            ).backward()
            total_value += float(total.detach())
            for name, value in components.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0)
                    + float(value.detach())
                )
            del input_ids, total, components
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in student.parameters()
                if parameter.requires_grad
            ],
            max_norm=float(args.maximum_gradient_norm),
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        record = {
            "step": step,
            "learning_rate": lr,
            "loss": total_value
            / float(args.gradient_accumulation_steps),
            "components": {
                name: value
                / float(args.gradient_accumulation_steps)
                for name, value in component_totals.items()
            },
            "gradient_norm": float(gradient_norm),
            "step_seconds": time.perf_counter() - step_started,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        }
        append_jsonl(log_path, record)
    active_student_frequency = (
        student.model.rotary_emb.inv_freq.detach().cpu().float()
    )
    if (
        tensor_sha256(active_student_frequency)
        != student_frequency["active_sha256_float32"]
    ):
        raise RuntimeError("student frequency changed during restoration")
    metadata = {
        "status": RESULT_STATUS,
        "protocol": protocol,
        "base_checkpoint_sha256": composite_checkpoint_sha256(
            args.checkpoint
        ),
        "frequency": "evq",
        "frequency_sha256_float32": student_frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkv_attention_restoration",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": SEQUENCE_LENGTH,
        "teacher_frequency": teacher_frequency,
        "student_frequency": student_frequency,
        "checkpoint_composite_sha256": composite_checkpoint_sha256(
            args.checkpoint
        ),
        "ready_receipt_sha256": sha256_file(args.receipt),
        "training_row_indices_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(rows).tobytes()
        ).hexdigest(),
    }
    adapter_sha256 = save_adapter(
        incomplete / "adapter.pt",
        student,
        None,
        metadata,
    )
    results = {
        **metadata,
        "environment": environment,
        "trainable_parameters": trainable,
        "adapter_sha256": adapter_sha256,
        "steps_completed": int(args.steps),
        "student_tokens": protocol["student_tokens"],
        "teacher_tokens": protocol["teacher_tokens"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "evaluation_status": "NOT_RUN",
        "claim_status": "TRAINING_ONLY_NOT_CAPABILITY_EVIDENCE",
        "ready_receipt": ready.get("gpu_smoke", {}),
    }
    atomic_json(incomplete / "results.json", results)
    incomplete.replace(output)


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.training_view = args.training_view.resolve()
    args.linearard_root = args.linearard_root.resolve()
    args.receipt = args.receipt.resolve()
    args.output = args.output.resolve()
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    if int(args.micro_batch_size) != 1:
        raise ValueError("registered first run requires micro batch 1")
    if int(args.gradient_accumulation_steps) != 4:
        raise ValueError("registered first run requires accumulation 4")
    if int(args.rank) != 512 or float(args.alpha) != 1024.0:
        raise ValueError("registered first run requires rank 512 / alpha 1024")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if args.mode == "gpu-smoke":
        gpu_smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
