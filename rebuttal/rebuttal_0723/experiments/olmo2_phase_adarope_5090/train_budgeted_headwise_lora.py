#!/usr/bin/env python3
"""Train budgeted headwise Q/K LoRA on generic long natural-text CLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.budgeted_headwise_lora import (
    adapter_state,
    install_budgeted_headwise_qk,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope import (
    _checkpoint_identity,
    _configure_cuda,
    _load_base,
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_adapter(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(path)


def _fused_loss() -> torch.nn.Module:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        return_z_loss=False,
        accum_dtype=torch.float32,
    )


def _natural_8k_batch(
    rows: torch.Tensor,
    *,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor]:
    first, second = generator.choice(len(rows), size=2, replace=False).tolist()
    sequence = torch.cat((rows[first], rows[second]), dim=0).unsqueeze(0)
    context = sequence[:, :-1].to("cuda", non_blocking=True)
    labels = sequence[:, 1:].clone().to("cuda", non_blocking=True)
    labels[:, 4095] = -100
    return context, labels, (int(first), int(second)), rows[second]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready", type=Path, required=True)
    parser.add_argument("--source-tensor", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--transport-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--teacher-distill", action="store_true")
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and accumulation must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    source_path = args.source_tensor.resolve()
    if (
        source_receipt.get("status") != "PURE_TEXT_ROWS_READY"
        or Path(source_receipt["tensor"]).resolve() != source_path
        or source_receipt.get("tensor_sha256") != sha256_file(source_path)
        or source_receipt.get("shape") != [1024, 4096]
    ):
        raise RuntimeError("generic natural-text source identity drift")
    rows = torch.load(source_path, map_location="cpu", weights_only=True)
    if rows.shape != (1024, 4096) or rows.dtype != torch.int64:
        raise RuntimeError("generic natural-text tensor shape drift")

    runtime = _configure_cuda(torch)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    model = _load_base(args.checkpoint, torch)
    model.config.max_position_embeddings = 8192
    model.config.use_cache = False
    method = install_budgeted_headwise_qk(
        model,
        target_table=args.target_table,
        transport_map=args.transport_map,
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.to("cuda")
    model.train()
    teacher = None
    if args.teacher_distill:
        teacher = _load_base(args.checkpoint, torch)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.config.use_cache = False
        teacher.to("cuda").eval()
    backbone = model.model
    if args.compile:
        backbone = torch.compile(
            backbone,
            fullgraph=False,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
    parameters = [value for value in model.parameters() if value.requires_grad]
    if sum(value.numel() for value in parameters) != int(method["trainable_parameters"]):
        raise RuntimeError("trainable parameter count drift")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    loss_fn = _fused_loss()
    generator = np.random.default_rng(int(args.seed))
    logs: list[dict] = []
    started = time.perf_counter()
    processed = 0
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(args.steps) + 1):
        if step <= int(args.warmup_steps):
            lr = float(args.learning_rate) * step / max(1, int(args.warmup_steps))
        else:
            progress = (step - int(args.warmup_steps)) / max(
                1, int(args.steps) - int(args.warmup_steps)
            )
            lr = float(args.learning_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        source_pairs: list[list[int]] = []
        component_losses: list[dict[str, float]] = []
        for _ in range(int(args.gradient_accumulation)):
            context, labels, pair, target_row = _natural_8k_batch(
                rows, generator=generator
            )
            teacher_hidden = None
            if teacher is not None:
                labels[:, :4096] = -100
                teacher_context = target_row[:-1].unsqueeze(0).to(
                    "cuda", non_blocking=True
                )
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    teacher_hidden = teacher.model(
                        input_ids=teacher_context,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state.detach()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(
                    input_ids=context,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                ce_loss = loss_fn(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                hidden_loss = (
                    torch.nn.functional.mse_loss(
                        hidden[:, 4096:, :].float(),
                        teacher_hidden.float(),
                    )
                    if teacher_hidden is not None
                    else hidden.new_zeros((), dtype=torch.float32)
                )
                raw_loss = ce_loss + float(args.hidden_weight) * hidden_loss
                loss = raw_loss / float(args.gradient_accumulation)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            losses.append(float(raw_loss.detach()))
            component_losses.append(
                {
                    "natural_ce": float(ce_loss.detach()),
                    "native_teacher_hidden_mse": float(hidden_loss.detach()),
                }
            )
            source_pairs.append(list(pair))
            processed += int(context.numel())
            del context, labels, hidden, raw_loss, loss, ce_loss, hidden_loss
            if teacher_hidden is not None:
                del teacher_hidden, teacher_context
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row = {
            "step": step,
            "loss": float(np.mean(losses)),
            "lr": lr,
            "grad_norm": float(grad_norm),
            "source_row_pairs": source_pairs,
            "components": {
                name: float(np.mean([item[name] for item in component_losses]))
                for name in component_losses[0]
            },
            "processed_tokens": processed,
            "tokens_per_second": processed / max(elapsed, 1e-9),
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        logs.append(row)
        _atomic_json(output / "train_steps.json", {"rows": logs})
        print(json.dumps(row, sort_keys=True), flush=True)

    adapter_path = output / "adapter.pt"
    payload = {
        "status": "BUDGETED_HEADWISE_QK_LORA_COMPLETE",
        "method": method,
        "checkpoint": checkpoint,
        "source_tensor_sha256": sha256_file(source_path),
        "target_table_file_sha256": sha256_file(args.target_table.resolve()),
        "transport_map_file_sha256": sha256_file(args.transport_map.resolve()),
        "training": {
            "objective": "generic natural-text next-token CE",
            "task_labels": False,
            "native_teacher_distillation": bool(args.teacher_distill),
            "teacher_target": (
                "final hidden state for the same natural text at Native positions 0..4094"
                if args.teacher_distill
                else None
            ),
            "student_target_positions": (
                [4096, 8190] if args.teacher_distill else None
            ),
            "hidden_weight": float(args.hidden_weight),
            "sequence_length": 8192,
            "document_boundary_labels_masked": True,
            "steps": int(args.steps),
            "gradient_accumulation": int(args.gradient_accumulation),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
        },
        "state": adapter_state(model),
    }
    _atomic_adapter(adapter_path, payload)
    receipt = {
        **{key: value for key, value in payload.items() if key != "state"},
        "adapter_sha256": sha256_file(adapter_path),
        "train_steps_sha256": sha256_file(output / "train_steps.json"),
        "runtime": runtime,
        "result": logs[-1],
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
