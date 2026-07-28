#!/usr/bin/env python3
"""Restore Native 4K attention under a diagnostic-selected hybrid EVQ grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
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
    configure_relation_capture_attention,
    import_linearard_kernel,
)
from .native_protected_evq import (
    ADAPTATION_NAME,
    DIAGNOSTIC_STATUS,
    FREQUENCY_NAME,
    METHOD_ID,
    PREPARED_STATUS,
    READY_STATUS,
    RESULT_STATUS,
    SEQUENCE_LENGTH,
    MultiLayerRelationCapture,
    all_layer_attention_restoration_loss,
    apply_native_protected_evq,
    install_masked_qk_lora,
    native_protected_evq_inv_freq,
    qk_unprotected_output_mask,
)
from .train_4k_evq_attention_restoration import (
    deterministic_training_rows,
    kernel_parity,
    learning_rate,
)
from .train_4k_stage_a import composite_checkpoint_sha256
from .train_screen import apply_frequency, load_fixed_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("gpu-smoke", "train"), required=True
    )
    parser.add_argument(
        "--arm",
        choices=("protected", "full-evq-control"),
        default="protected",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=20_260_805)
    return parser.parse_args()


def load_selection(path: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if (
        receipt.get("status") != DIAGNOSTIC_STATUS
        or receipt.get("passed") is not True
        or selection.get("passed") is not True
    ):
        raise RuntimeError(
            "protected-EVQ training requires a passed diagnostic receipt"
        )
    protected = tuple(
        int(value) for value in selection["protected_pair_indices"]
    )
    if (
        list(protected) != sorted(set(protected))
        or len(protected) > int(
            selection["thresholds"]["maximum_protected_pairs"]
        )
    ):
        raise RuntimeError("diagnostic protected-pair set drift")
    return receipt, protected


def protocol_from_args(
    args: argparse.Namespace,
    *,
    selection_receipt: Path,
    protected_pairs: tuple[int, ...],
) -> dict[str, Any]:
    selected_protected = tuple(protected_pairs)
    applied_protected = (
        selected_protected
        if str(args.arm) == "protected"
        else ()
    )
    hybrid = native_protected_evq_inv_freq(applied_protected)
    mask_config = type(
        "Config",
        (),
        {
            "hidden_size": 2_048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
        },
    )()
    output_mask = qk_unprotected_output_mask(
        mask_config, applied_protected
    )
    steps = int(args.steps)
    micro = int(args.micro_batch_size)
    accumulation = int(args.gradient_accumulation_steps)
    return {
        "method_id": METHOD_ID,
        "arm": str(args.arm),
        "frequency": FREQUENCY_NAME,
        "frequency_sha256_float32": tensor_sha256(hybrid),
        "native_frequency_sha256_float32": tensor_sha256(
            endpoint_geo_inv_freq()
        ),
        "evq_frequency_sha256_float32": tensor_sha256(
            endpoint_evq_inv_freq()
        ),
        "diagnostic_selected_pair_indices": list(selected_protected),
        "protected_native_pair_indices": list(applied_protected),
        "unprotected_evq_pair_indices": [
            index
            for index in range(64)
            if index not in set(applied_protected)
        ],
        "qk_output_mask_sha256": tensor_sha256(output_mask),
        "selection_receipt_sha256": sha256_file(selection_receipt),
        "teacher": "same checkpoint with exact Native RoPE",
        "student": (
            "same frozen checkpoint with protected Native pairs and exact "
            "EVQ frequencies elsewhere"
        ),
        "adaptation": ADAPTATION_NAME,
        "lora_targets": ["q_proj", "k_proj"],
        "lora_layers": "all_16",
        "lora_direct_update_coordinates": "unprotected_pairs_only",
        "lora_rank": int(args.rank),
        "lora_alpha": float(args.alpha),
        "lora_dropout": 0.0,
        "distilled_layers": list(range(16)),
        "objective": {
            "all_layer_post_rope_causal_qk_forward_kl": float(
                args.attention_weight
            ),
            "all_layer_post_attention_context_normalized_mse": float(
                args.context_weight
            ),
            "language_model_ce": 0.0,
            "output_logit_kl": 0.0,
            "task_supervision": 0.0,
        },
        "training_length": SEQUENCE_LENGTH,
        "position_ids": "contiguous_0_to_4095",
        "steps": steps,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": micro * accumulation,
        "student_tokens": (
            steps * micro * accumulation * SEQUENCE_LENGTH
        ),
        "teacher_tokens": (
            steps * micro * accumulation * SEQUENCE_LENGTH
        ),
        "optimizer": "fused_AdamW",
        "weight_decay": 0.0,
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "minimum_lr_ratio": float(args.minimum_lr_ratio),
        "maximum_gradient_norm": float(args.maximum_gradient_norm),
        "compile": False,
        "gradient_checkpointing": False,
        "seed": int(args.seed),
    }


def validate_bound_receipt(
    args: argparse.Namespace,
    *,
    expected_status: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _, protected = load_selection(args.selection_receipt.resolve())
    applied_protected = (
        protected if str(args.arm) == "protected" else ()
    )
    protocol = protocol_from_args(
        args,
        selection_receipt=args.selection_receipt.resolve(),
        protected_pairs=protected,
    )
    if (
        receipt.get("status") != expected_status
        or receipt.get("protocol") != protocol
    ):
        raise RuntimeError("Native-protected EVQ receipt protocol drift")
    expected_paths = {
        "checkpoint": args.checkpoint.resolve(),
        "training_view": args.training_view.resolve(),
        "selection_receipt": args.selection_receipt.resolve(),
        "linearard_root": args.linearard_root.resolve(),
    }
    for name, expected in expected_paths.items():
        observed = Path(receipt["inputs"][name]["path"]).resolve()
        if observed != expected:
            raise RuntimeError(f"receipt input path drift for {name}")
    if (
        receipt["inputs"]["checkpoint"]["composite_sha256"]
        != composite_checkpoint_sha256(args.checkpoint.resolve())
        or receipt["inputs"]["selection_receipt"]["sha256"]
        != sha256_file(args.selection_receipt.resolve())
    ):
        raise RuntimeError("checkpoint or selection hash drift")
    for name, entry in receipt["inputs"]["training_view"]["files"].items():
        path = args.training_view.resolve() / name
        if (
            int(path.stat().st_size) != int(entry["bytes"])
            or sha256_file(path) != str(entry["sha256"])
        ):
            raise RuntimeError(f"training-view drift for {name}")
    source = receipt["source"]
    here = Path(__file__).resolve()
    expected_source = {
        "trainer": here,
        "method": here.with_name("native_protected_evq.py"),
        "relation_loss": here.with_name("evq_attention_restoration.py"),
    }
    for name, path in expected_source.items():
        if source[name]["sha256"] != sha256_file(path):
            raise RuntimeError(f"training source drift for {name}")
    expected_output = (
        receipt["outputs"]["gpu_ready_receipt"]
        if args.mode == "gpu-smoke"
        else receipt["outputs"]["run_output"]
    )
    if Path(expected_output).resolve() != args.output.resolve():
        raise RuntimeError("registered output path drift")
    return receipt, applied_protected


def model_pair(
    *,
    checkpoint: Path,
    protected_pairs: tuple[int, ...],
    rank: int,
    alpha: float,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], int, torch.Tensor]:
    teacher = load_model(checkpoint)
    student = load_model(checkpoint)
    teacher_frequency = apply_frequency(teacher, "native")
    student_frequency = apply_native_protected_evq(
        student, protected_pairs
    )
    trainable, output_mask = install_masked_qk_lora(
        student,
        protected_pairs=protected_pairs,
        rank=int(rank),
        alpha=float(alpha),
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    student.train()
    teacher.config.use_cache = False
    student.config.use_cache = False
    if (
        float(getattr(teacher.config, "attention_dropout", 0.0)) != 0.0
        or float(getattr(student.config, "attention_dropout", 0.0)) != 0.0
    ):
        raise RuntimeError("attention restoration requires zero dropout")
    teacher.gradient_checkpointing_disable()
    student.gradient_checkpointing_disable()
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
        output_mask,
    )


def one_loss(
    *,
    teacher: Any,
    student: Any,
    input_ids: torch.Tensor,
    capture: MultiLayerRelationCapture,
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
        total, components = all_layer_attention_restoration_loss(
            captures=capture,
            kernel=kernel,
            attention_weight=protocol["objective"][
                "all_layer_post_rope_causal_qk_forward_kl"
            ],
            context_weight=protocol["objective"][
                "all_layer_post_attention_context_normalized_mse"
            ],
        )
    return total, components


def _component_json(
    components: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in components.items():
        detached = value.detach().float().cpu()
        result[name] = (
            float(detached)
            if detached.ndim == 0
            else detached.tolist()
        )
    return result


def gpu_smoke(args: argparse.Namespace) -> None:
    prepared, protected = validate_bound_receipt(
        args, expected_status=PREPARED_STATUS
    )
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
        output_mask,
    ) = model_pair(
        checkpoint=args.checkpoint,
        protected_pairs=protected,
        rank=int(args.rank),
        alpha=float(args.alpha),
        device=device,
    )
    capture = MultiLayerRelationCapture()
    protocol = protocol_from_args(
        args,
        selection_receipt=args.selection_receipt.resolve(),
        protected_pairs=protected,
    )
    started = time.perf_counter()
    total, components = one_loss(
        teacher=teacher,
        student=student,
        input_ids=input_ids,
        capture=capture,
        kernel=kernel,
        protocol=protocol,
    )
    total.backward()
    torch.cuda.synchronize()
    gradient_rows = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        gradient_rows.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": bool(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                ),
                "nonzero": bool(
                    gradient is not None
                    and torch.count_nonzero(gradient)
                ),
            }
        )
    b_rows = [
        value for value in gradient_rows if value["name"].endswith(".b")
    ]
    if (
        len(gradient_rows) != 64
        or len(b_rows) != 32
        or not all(value["present"] and value["finite"] for value in gradient_rows)
        or not all(value["nonzero"] for value in b_rows)
        or not torch.isfinite(total)
    ):
        raise RuntimeError("masked Q/K full-model gradient smoke failed")
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
                "components": _component_json(components),
                "trainable_gradient_tensors": len(gradient_rows),
                "nonzero_lora_b_gradient_tensors": sum(
                    value["nonzero"] for value in b_rows
                ),
                "trainable_parameters": trainable,
                "qk_output_mask_sha256": tensor_sha256(output_mask),
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
    ready, protected = validate_bound_receipt(
        args, expected_status=READY_STATUS
    )
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    incomplete.mkdir(parents=True)
    environment = configure_cuda()
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    kernel = import_linearard_kernel(args.linearard_root)
    protocol = protocol_from_args(
        args,
        selection_receipt=args.selection_receipt.resolve(),
        protected_pairs=protected,
    )
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
        output_mask,
    ) = model_pair(
        checkpoint=args.checkpoint,
        protected_pairs=protected,
        rank=int(args.rank),
        alpha=float(args.alpha),
        device=device,
    )
    parameters = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        weight_decay=0.0,
        fused=True,
    )
    capture = MultiLayerRelationCapture()
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
        optimizer.zero_grad(set_to_none=True)
        micro_records = []
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
                raise RuntimeError("non-finite protected-EVQ loss")
            (
                total / float(args.gradient_accumulation_steps)
            ).backward()
            total_value += float(total.detach())
            micro_records.append(_component_json(components))
            del input_ids, total, components
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=float(args.maximum_gradient_norm)
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite protected-EVQ gradient norm")
        optimizer.step()
        torch.cuda.synchronize()
        append_jsonl(
            log_path,
            {
                "step": step,
                "learning_rate": lr,
                "loss": total_value
                / float(args.gradient_accumulation_steps),
                "micro_components": micro_records,
                "gradient_norm": float(gradient_norm),
                "step_seconds": time.perf_counter() - step_started,
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            },
        )
    active = student.model.rotary_emb.inv_freq.detach().cpu().float()
    expected = native_protected_evq_inv_freq(protected)
    if not torch.equal(active, expected):
        raise RuntimeError("protected-EVQ frequency changed during training")
    for module in student.modules():
        mask = getattr(module, "output_mask", None)
        if mask is not None and tensor_sha256(mask.detach().cpu().float()) != (
            tensor_sha256(output_mask)
        ):
            raise RuntimeError("Q/K output mask changed during training")
    metadata = {
        "status": RESULT_STATUS,
        "protocol": protocol,
        "base_checkpoint_sha256": composite_checkpoint_sha256(
            args.checkpoint
        ),
        "frequency": FREQUENCY_NAME,
        "frequency_sha256_float32": tensor_sha256(active),
        "adaptation": ADAPTATION_NAME,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": SEQUENCE_LENGTH,
        "qk_output_mask_sha256": tensor_sha256(output_mask),
        "protected_native_pair_indices": list(protected),
        "teacher_frequency": teacher_frequency,
        "student_frequency": student_frequency,
        "selection_receipt_sha256": sha256_file(
            args.selection_receipt
        ),
        "ready_receipt_sha256": sha256_file(args.receipt),
        "training_row_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(rows, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    adapter_sha256 = save_adapter(
        incomplete / "adapter.pt", student, None, metadata
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
        "next_gate": (
            "Fresh Native, immediate hybrid, and trained hybrid must pass "
            "the registered 4K NLL, 2Wiki, RULER13, and independent-retention "
            "gates before any 8K/16K evaluation."
        ),
    }
    atomic_json(incomplete / "results.json", results)
    incomplete.replace(output)


def main() -> None:
    args = parse_args()
    if (
        int(args.steps) <= 0
        or int(args.micro_batch_size) != 1
        or int(args.gradient_accumulation_steps) != 4
        or int(args.rank) != 512
        or not math.isclose(float(args.alpha), 1024.0)
    ):
        raise RuntimeError(
            "registered first run fixes steps>0, micro/accum=1/4, "
            "rank=512 and alpha=1024"
        )
    if args.mode == "gpu-smoke":
        gpu_smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
