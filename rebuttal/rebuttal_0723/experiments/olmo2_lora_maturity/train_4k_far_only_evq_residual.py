#!/usr/bin/env python3
"""Smoke or train the Native-preserving far-query EVQ residual."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    seed_everything,
    sha256_file,
)

from .far_only_evq_residual import (
    ADAPTATION_NAME,
    METHOD_ID,
    READY_STATUS,
    RESULT_STATUS,
    FarOnlyEVQAttention,
    FarOnlyEVQConfig,
    far_only_trainable_named_parameters,
    install_far_only_evq_residual,
    save_far_only_adapter,
    set_far_only_evq_route,
)
from .phase_adaptation import (
    LENGTH,
    PhaseAdaptationView,
    phase_batch,
    position_ids_for_offsets,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import fused_loss_module


EXTERNAL_OFFSETS = (LENGTH, LENGTH, 3 * LENGTH, 3 * LENGTH)
REGISTERED_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path, required=True)
    parser.add_argument("--gpu-ready-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=REGISTERED_STEPS)
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


def method_config(args: argparse.Namespace) -> FarOnlyEVQConfig:
    return FarOnlyEVQConfig(
        threshold_position=LENGTH,
        projection_rank=int(args.projection_rank),
        residual_head_dim=int(args.residual_head_dim),
        initial_logit_gain=float(args.initial_logit_gain),
        initialization_seed=int(args.seed),
    )


def registered_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "adaptation": ADAPTATION_NAME,
        "method_config": asdict(method_config(args)),
        "base_path": "untouched_native_rope_qkvo",
        "residual_path": "endpoint_evq_cosh_qk_low_rank",
        "attention_combination": (
            "single_augmented_qk_score_single_softmax_zero_padded_values"
        ),
        "short_request_route": "native_exact_when_total_budget_le_4096",
        "long_request_route": "evq_residual_enabled_before_prefill",
        "query_gate": "position_id_ge_4096",
        "physical_training_length_maximum": LENGTH,
        "virtual_position_offsets": list(EXTERNAL_OFFSETS),
        "supervision": "complete_answer_plus_immediate_eos",
        "steps": int(args.steps),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "optimizer": "fused_adamw",
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "precision": "bf16_autocast",
        "compile_mode": str(args.compile_mode),
        "validation_rows": int(args.validation_rows),
        "seed": int(args.seed),
    }


def bound_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = {
        "trainer": Path(__file__).resolve(),
        "method": root / "far_only_evq_residual.py",
        "phase_contract": root / "phase_adaptation.py",
        "training_primitives": root / "train_screen.py",
        "checkpoint_contract": root / "train_4k_stage_a.py",
        "model_loader": root.parent / "olmo2_lora_conversion.py",
        "evq_contract": root.parent / "olmo2_1b_evq" / "contract.py",
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _prepared_input_receipt(
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


def verify_prepared(
    *,
    args: argparse.Namespace,
    checkpoint_sha256: str,
    view: PhaseAdaptationView,
) -> dict[str, Any]:
    path = args.prepared_receipt.resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status")
        != "OLMO2_FAR_ONLY_EVQ_RESIDUAL_PREPARED_NO_GPU"
        or receipt.get("method_id") != METHOD_ID
        or receipt.get("protocol") != registered_protocol(args)
        or receipt.get("bound_code_sha256") != bound_code_sha256()
        or receipt.get("inputs")
        != _prepared_input_receipt(
            args=args,
            checkpoint_sha256=checkpoint_sha256,
            view=view,
        )
    ):
        raise RuntimeError("far-only EVQ prepared receipt drift")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": receipt["status"],
    }


def verify_gpu_ready(
    *,
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    if args.gpu_ready_receipt is None:
        raise RuntimeError("training requires --gpu-ready-receipt")
    path = args.gpu_ready_receipt.resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != READY_STATUS
        or receipt.get("method_id") != METHOD_ID
        or receipt.get("protocol") != registered_protocol(args)
        or receipt.get("bound_code_sha256") != bound_code_sha256()
        or receipt.get("prepared_receipt_sha256")
        != prepared["sha256"]
    ):
        raise RuntimeError("far-only EVQ GPU READY receipt drift")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": receipt["status"],
    }


def _selected_phase_batch(
    *,
    view: PhaseAdaptationView,
    generator: torch.Generator,
    accumulation_index: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    dict[str, int],
]:
    local = int(
        torch.randint(
            len(view.training_rows),
            (1,),
            generator=generator,
        )
    )
    row_index = int(view.training_rows[local])
    indices = np.asarray([row_index], dtype=np.int64)
    offset = int(EXTERNAL_OFFSETS[accumulation_index])
    contexts, labels, supervised = phase_batch(
        view=view, indices=indices
    )
    position_ids, receipts, _ = position_ids_for_offsets(
        view=view,
        indices=indices,
        offsets=np.asarray([offset], dtype=np.int64),
    )
    receipt = receipts[0]
    if (
        int(receipt["virtual_answer_prediction_position"]) < LENGTH
        or int(position_ids.max()) >= 4 * LENGTH
    ):
        raise RuntimeError("registered external phase exposure drift")
    return (
        contexts,
        labels,
        position_ids,
        supervised,
        row_index,
        receipt,
    )


@torch.no_grad()
def teacher_forced_validation(
    *,
    model: Any,
    view: PhaseAdaptationView,
    count: int,
) -> dict[str, Any]:
    rows = view.validation_rows[: int(count)]
    if len(rows) == 0:
        raise RuntimeError("phase view has no validation rows")
    model.eval()
    set_far_only_evq_route(model, True)
    losses: dict[str, list[float]] = {"8k": [], "16k": []}
    exact: dict[str, list[float]] = {"8k": [], "16k": []}
    for slot, row_index in enumerate(rows.tolist()):
        offset = LENGTH if slot % 2 == 0 else 3 * LENGTH
        bucket = "8k" if offset == LENGTH else "16k"
        indices = np.asarray([row_index], dtype=np.int64)
        contexts, labels, _ = phase_batch(view=view, indices=indices)
        position_ids, _, _ = position_ids_for_offsets(
            view=view,
            indices=indices,
            offsets=np.asarray([offset], dtype=np.int64),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(
                input_ids=contexts,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        mask = labels != -100
        selected_logits = model.lm_head(hidden[mask]).float()
        selected_labels = labels[mask]
        loss = torch.nn.functional.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="mean",
        )
        prediction = selected_logits.argmax(dim=-1)
        losses[bucket].append(float(loss))
        exact[bucket].append(
            float(bool(torch.equal(prediction, selected_labels)))
        )
        del contexts, labels, position_ids, hidden, selected_logits
    return {
        bucket: {
            "rows": len(losses[bucket]),
            "mean_supervised_nll": float(np.mean(losses[bucket])),
            "teacher_forced_full_sequence_exact": float(
                np.mean(exact[bucket])
            ),
        }
        for bucket in ("8k", "16k")
    }


def smoke(
    *,
    args: argparse.Namespace,
    model: Any,
    view: PhaseAdaptationView,
    install_receipt: dict[str, Any],
    prepared: dict[str, Any],
    expected_native_logits: torch.Tensor,
) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model.eval()
    model.to("cuda")
    prefix_ids = torch.from_numpy(
        np.asarray(
            view.input_ids[int(view.training_rows[0]), :128],
            dtype=np.int64,
        )
    )[None, :].to("cuda")
    set_far_only_evq_route(model, False)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        repeated_logits = model(
            input_ids=prefix_ids,
            use_cache=False,
            return_dict=True,
        ).logits
    short_bitwise = bool(
        torch.equal(expected_native_logits, repeated_logits)
    )
    if not short_bitwise:
        raise RuntimeError("short Native route is not deterministic")

    set_far_only_evq_route(model, True)
    full_indices = np.asarray(
        [int(view.training_rows[0])], dtype=np.int64
    )
    contexts, labels, _ = phase_batch(
        view=view, indices=full_indices
    )
    positions, phase_receipts, _ = position_ids_for_offsets(
        view=view,
        indices=full_indices,
        offsets=np.asarray([LENGTH], dtype=np.int64),
    )
    model.train()
    for _, parameter in far_only_trainable_named_parameters(model):
        parameter.grad = None
    torch.cuda.reset_peak_memory_stats()
    backbone = TrainingBackbone(model.model)
    loss_module = fused_loss_module()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(contexts, positions)
        loss = loss_module(
            model.lm_head.weight,
            hidden.reshape(-1, hidden.shape[-1]),
            labels.reshape(-1),
        )
        loss = loss.loss if hasattr(loss, "loss") else loss
    if not torch.isfinite(loss):
        raise RuntimeError("far-only EVQ smoke loss is non-finite")
    loss.backward()
    named = far_only_trainable_named_parameters(model)
    nonzero_gradients = [
        name
        for name, parameter in named
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
    ]
    if not nonzero_gradients:
        raise RuntimeError("far-only EVQ smoke gradients are all zero")
    for _, parameter in named:
        parameter.grad = None
    del contexts, labels, positions, hidden, loss
    torch.cuda.empty_cache()

    model.eval()
    cache_positions = (
        torch.arange(prefix_ids.shape[1], device="cuda")[None, :]
        + LENGTH
    )
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        cached = model(
            input_ids=prefix_ids[:, :32],
            position_ids=cache_positions[:, :32],
            use_cache=True,
            return_dict=True,
        )
        decoded = model(
            input_ids=prefix_ids[:, 32:33],
            position_ids=cache_positions[:, 32:33],
            cache_position=torch.tensor([32], device="cuda"),
            past_key_values=cached.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    if not torch.isfinite(decoded.logits).all():
        raise RuntimeError("far-only EVQ cached decode is non-finite")
    cache_widths = {
        int(layer.keys.shape[-1])
        for layer in cached.past_key_values.layers
    }
    expected_width = 128 + int(args.residual_head_dim)
    if cache_widths != {expected_width}:
        raise RuntimeError(
            f"augmented cache width drift: {cache_widths}"
        )
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "flash_sdp_enabled": bool(
            torch.backends.cuda.flash_sdp_enabled()
        ),
        "math_sdp_enabled": bool(
            torch.backends.cuda.math_sdp_enabled()
        ),
        "memory_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
    }
    if (
        not runtime["bf16_supported"]
        or not runtime["flash_sdp_enabled"]
        or runtime["math_sdp_enabled"]
        or runtime["memory_efficient_sdp_enabled"]
    ):
        raise RuntimeError("Flash-only BF16 runtime contract failed")
    result = {
        "status": READY_STATUS,
        "method_id": METHOD_ID,
        "classification": "GPU_READY_NOT_EXPERIMENT_RESULT",
        "protocol": registered_protocol(args),
        "bound_code_sha256": bound_code_sha256(),
        "prepared_receipt_sha256": prepared["sha256"],
        "install_receipt": install_receipt,
        "checks": {
            "short_native_route_deterministic_bitwise": short_bitwise,
            "full_4k_active_loss_finite": True,
            "full_4k_phase_receipt": phase_receipts[0],
            "nonzero_gradient_parameter_tensors": len(
                nonzero_gradients
            ),
            "cached_decode_finite": True,
            "augmented_cache_width": expected_width,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "runtime": runtime,
    }
    atomic_json(output / "gpu_ready.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def train(
    *,
    args: argparse.Namespace,
    model: Any,
    view: PhaseAdaptationView,
    install_receipt: dict[str, Any],
    checkpoint_sha256: str,
    prepared: dict[str, Any],
    gpu_ready: dict[str, Any],
) -> dict[str, Any]:
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    incomplete.mkdir(parents=True)
    named = far_only_trainable_named_parameters(model)
    parameters = [parameter for _, parameter in named]
    model.gradient_checkpointing_disable()
    set_far_only_evq_route(model, True)
    backbone_module: torch.nn.Module = TrainingBackbone(model.model)
    if args.compile_mode != "none":
        backbone_module = torch.compile(
            backbone_module,
            fullgraph=False,
            dynamic=False,
            mode=str(args.compile_mode),
        )
    loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 81_337)
    validation_before = teacher_forced_validation(
        model=model,
        view=view,
        count=int(args.validation_rows),
    )
    model.train()
    started = time.perf_counter()
    processed_tokens = 0
    supervised_tokens = 0
    last_log_time = started
    last_log_tokens = 0
    recent: list[float] = []
    offset_counts = {str(value): 0 for value in sorted(set(EXTERNAL_OFFSETS))}
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(args.steps) + 1):
        learning_rate = cosine_lr(
            step,
            int(args.steps),
            int(args.warmup_steps),
            float(args.learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for accumulation_index in range(
            int(args.gradient_accumulation_steps)
        ):
            (
                contexts,
                labels,
                position_ids,
                supervised,
                row_index,
                phase_receipt,
            ) = _selected_phase_batch(
                view=view,
                generator=generator,
                accumulation_index=accumulation_index,
            )
            offset = int(EXTERNAL_OFFSETS[accumulation_index])
            offset_counts[str(offset)] += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone_module(contexts, position_ids)
                raw_loss = loss_module(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                raw_loss = (
                    raw_loss.loss
                    if hasattr(raw_loss, "loss")
                    else raw_loss
                )
                loss = raw_loss / float(
                    args.gradient_accumulation_steps
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite far-only EVQ loss at step {step}"
                )
            loss.backward()
            step_losses.append(float(raw_loss.detach()))
            processed_tokens += int(contexts.numel())
            supervised_tokens += int(supervised)
            if step == 1:
                append_jsonl(
                    incomplete / "exposure_log.jsonl",
                    {
                        "optimizer_step": step,
                        "accumulation_index": accumulation_index,
                        "row_index": row_index,
                        "offset": offset,
                        "phase_receipt": phase_receipt,
                    },
                )
            del contexts, labels, position_ids, hidden, raw_loss, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("far-only EVQ gradient norm is non-finite")
        optimizer.step()
        mean_loss = float(np.mean(step_losses))
        recent.append(mean_loss)
        if step == 1 or step % 10 == 0 or step == int(args.steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            append_jsonl(
                incomplete / "train_log.jsonl",
                {
                    "step": step,
                    "total_steps": int(args.steps),
                    "loss": mean_loss,
                    "mean_loss_last_10": float(
                        np.mean(recent[-10:])
                    ),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "processed_input_tokens": processed_tokens,
                    "supervised_tokens": supervised_tokens,
                    "offset_counts": dict(offset_counts),
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
    model.eval()
    validation_after = teacher_forced_validation(
        model=model,
        view=view,
        count=int(args.validation_rows),
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "method_id": METHOD_ID,
        "adaptation": ADAPTATION_NAME,
        "method_config": asdict(method_config(args)),
        "base_checkpoint_sha256": checkpoint_sha256,
        "global_frequency": "native",
        "residual_frequency": "endpoint_evq_cosh",
        "training_view_manifest_sha256": sha256_file(
            view.root / "manifest.json"
        ),
        "physical_training_length_maximum": LENGTH,
        "training_position_offsets": list(EXTERNAL_OFFSETS),
        "seed": int(args.seed),
    }
    adapter_sha256 = save_far_only_adapter(
        incomplete / "adapter.pt",
        model,
        metadata=metadata,
    )
    result = {
        "status": RESULT_STATUS,
        "evidence_tier": "DESIGN_ONLY_OR_PENDING_UNTIL_EVALUATED",
        "metric_boundary": (
            "Training and teacher-forced validation are not autoregressive "
            "capability evidence.  Short-request preservation is structural; "
            "8K/16K capability requires the registered strict evaluators."
        ),
        "method_id": METHOD_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_view": {
            "path": str(view.root),
            "manifest_sha256": sha256_file(
                view.root / "manifest.json"
            ),
        },
        "prepared_receipt": prepared,
        "gpu_ready_receipt": gpu_ready,
        "protocol": registered_protocol(args),
        "bound_code_sha256": bound_code_sha256(),
        "install_receipt": install_receipt,
        "adapter_sha256": adapter_sha256,
        "adapter_metadata": metadata,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "training": {
            "steps": int(args.steps),
            "processed_input_tokens": processed_tokens,
            "supervised_tokens": supervised_tokens,
            "offset_counts": offset_counts,
            "elapsed_seconds": elapsed,
            "tokens_per_second": processed_tokens / elapsed,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
    }
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    if int(args.steps) != REGISTERED_STEPS:
        raise ValueError("registered first run requires exactly 100 steps")
    if int(args.micro_batch_size) != 1:
        raise ValueError("registered first run requires micro batch one")
    if int(args.gradient_accumulation_steps) != len(EXTERNAL_OFFSETS):
        raise ValueError("registered first run requires accumulation four")
    if int(args.projection_rank) != 64:
        raise ValueError("registered first run requires projection rank 64")
    if int(args.residual_head_dim) != 128:
        raise ValueError("registered first run requires residual head dim 128")
    seed_everything(int(args.seed))
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = ready_checkpoint_digest(
        checkpoint,
        args.checkpoint_ready_receipt.resolve(),
    )
    view = PhaseAdaptationView(args.training_view.resolve())
    prepared = verify_prepared(
        args=args,
        checkpoint_sha256=checkpoint_sha256,
        view=view,
    )
    configure_cuda()
    model = load_model(checkpoint)
    expected_native_logits = None
    if args.mode == "smoke":
        model.to("cuda")
        prefix_ids = torch.from_numpy(
            np.asarray(
                view.input_ids[int(view.training_rows[0]), :128],
                dtype=np.int64,
            )
        )[None, :].to("cuda")
        model.eval()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            expected_native_logits = model(
                input_ids=prefix_ids,
                use_cache=False,
                return_dict=True,
            ).logits.detach().clone()
    install_receipt = install_far_only_evq_residual(
        model,
        method_config(args),
    )
    model.to("cuda")
    if args.mode == "smoke":
        if expected_native_logits is None:
            raise RuntimeError("Native smoke baseline was not captured")
        smoke(
            args=args,
            model=model,
            view=view,
            install_receipt=install_receipt,
            prepared=prepared,
            expected_native_logits=expected_native_logits,
        )
        return
    gpu_ready = verify_gpu_ready(args=args, prepared=prepared)
    train(
        args=args,
        model=model,
        view=view,
        install_receipt=install_receipt,
        checkpoint_sha256=checkpoint_sha256,
        prepared=prepared,
        gpu_ready=gpu_ready,
    )


if __name__ == "__main__":
    main()
