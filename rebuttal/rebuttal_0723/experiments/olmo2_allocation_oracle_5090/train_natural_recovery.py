#!/usr/bin/env python3
"""Recover a frozen learned allocation with dense 4K natural-LM Q/K LoRA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import TrainingBackbone
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import fused_loss_module
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    seed_everything,
)

from .evaluate_long_natural import _evaluate_view, _load_view
from .oracle import sha256_file
from .train import RESULT_STATUS, _base_model, _load_model


STATUS = "OLMO2_ALLOCATION_NATURAL_RECOVERY_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--train-tensor", type=Path, required=True)
    parser.add_argument("--train-receipt", type=Path, required=True)
    parser.add_argument("--long-view", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--training-table", choices=("learned", "native"), default="learned"
    )
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def _load_training_data(
    *, tensor_path: Path, receipt_path: Path, checkpoint: Path, excluded_rows: set[int]
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, Any]]:
    tensor_path = tensor_path.resolve()
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tokens = torch.load(tensor_path, map_location="cpu", weights_only=True)
    documents = receipt.get("documents", [])
    if (
        receipt.get("status") != "PURE_TEXT_ROWS_READY"
        or receipt.get("training_or_parameter_updates") is not False
        or Path(str(receipt.get("model", ""))).resolve() != checkpoint
        or Path(str(receipt.get("tensor", ""))).resolve() != tensor_path
        or sha256_file(tensor_path) != receipt.get("tensor_sha256")
        or tuple(tokens.shape) != (1024, 4096)
        or len(documents) != len(tokens)
    ):
        raise RuntimeError("dense 4K training-data receipt drift")
    eligible = np.asarray(
        [
            index
            for index, document in enumerate(documents)
            if int(document["parquet_row"]) not in excluded_rows
        ],
        dtype=np.int64,
    )
    if len(eligible) < 1016:
        raise RuntimeError("unexpected overlap with long natural evaluation rows")
    return tokens, eligible[:896], eligible[896:1016], receipt


@torch.inference_mode()
def _evaluate_4k(
    *, model: Any, backbone: Any, loss_module: Any, tokens: torch.Tensor, rows: np.ndarray
) -> dict[str, Any]:
    total = 0.0
    count = 0
    for start in range(0, len(rows), 4):
        ids = tokens[rows[start : start + 4]].to("cuda", non_blocking=True)
        hidden = backbone(ids[:, :-1], None)
        labels = ids[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = loss_module(
                _base_model(model).lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )
            value = value.loss if hasattr(value, "loss") else value
        batch_tokens = int(labels.numel())
        total += float(value) * batch_tokens
        count += batch_tokens
        del ids, hidden, labels, value
    return {"rows": int(len(rows)), "tokens": count, "mean_nll": total / count}


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED") != "YES":
        raise PermissionError("natural recovery requires both authorization factors")
    if int(args.steps) != 300:
        raise ValueError("the registered recovery budget is exactly 300 steps")
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)

    checkpoint = args.checkpoint.resolve()
    oracle_path = args.oracle_result.resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("status") != RESULT_STATUS or oracle.get("smoke"):
        raise RuntimeError("completed full oracle result is required")
    state_path = oracle_path.parent / "allocation_state.pt"
    if sha256_file(state_path) != oracle["artifacts"]["allocation_state_sha256"]:
        raise RuntimeError("allocation artifact receipt drift")

    long_views = {
        path.resolve(): _load_view(path, checkpoint=checkpoint)
        for path in args.long_view
    }
    if sorted(rows.shape[1] for rows, _ in long_views.values()) != [8192, 16384]:
        raise RuntimeError("exactly one physical 8K and one physical 16K view are required")
    excluded = {
        int(document["parquet_row"])
        for _, receipt in long_views.values()
        for document in receipt["documents"]
    }
    tokens, train_rows, validation_rows, train_receipt = _load_training_data(
        tensor_path=args.train_tensor,
        receipt_path=args.train_receipt,
        checkpoint=checkpoint,
        excluded_rows=excluded,
    )

    incomplete.mkdir(parents=True)
    seed_everything(int(args.seed))
    environment = configure_cuda()
    model, allocation = _load_model(checkpoint)
    saved = torch.load(state_path, map_location="cpu", weights_only=True)
    if not torch.equal(saved["native_inv_freq"], allocation.native_inv_freq.cpu()):
        raise RuntimeError("allocation Native table drift")
    training_logits = (
        saved["gap_delta_logits"]
        if args.training_table == "learned"
        else torch.zeros_like(saved["gap_delta_logits"])
    )
    with torch.no_grad():
        allocation.gap_delta_logits.copy_(training_logits)
    allocation.gap_delta_logits.requires_grad_(False)
    lora_parameters = [
        value
        for name, value in model.named_parameters()
        if value.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    if len(lora_parameters) != 64:
        raise RuntimeError("Q/K LoRA trainable scope drift")

    model.to("cuda")
    backbone = torch.compile(
        TrainingBackbone(_base_model(model).model),
        fullgraph=True,
        dynamic=False,
        mode=str(args.compile_mode),
    )
    loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        lora_parameters, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0, fused=True
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 120_001)
    started = time.perf_counter()
    processed = 0
    first_gradients: dict[str, bool] = {}
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(1, 301):
        optimizer.param_groups[0]["lr"] = 5e-5 * cosine_lr(step, 300, 20, 1.0)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(2):
            local = torch.randint(len(train_rows), (4,), generator=generator).numpy()
            ids = tokens[train_rows[local]].to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(ids[:, :-1], None)
                labels = ids[:, 1:]
                value = loss_module(
                    _base_model(model).lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                value = value.loss if hasattr(value, "loss") else value
            if not torch.isfinite(value):
                raise RuntimeError(f"non-finite loss at step {step}")
            (value / 2.0).backward()
            losses.append(float(value.detach()))
            processed += int(ids.numel())
            del ids, hidden, labels, value
        if step == 1:
            gradients = [parameter.grad for parameter in lora_parameters]
            first_gradients = {
                "present": bool(all(value is not None for value in gradients)),
                "finite": bool(all(torch.isfinite(value).all() for value in gradients if value is not None)),
                "nonzero": bool(any(torch.count_nonzero(value).item() for value in gradients if value is not None)),
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            append_jsonl(
                incomplete / "train_log.jsonl",
                {
                    "step": step,
                    "loss": float(np.mean(losses)),
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "processed_input_tokens": processed,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
    if not all(first_gradients.values()):
        raise RuntimeError(f"first-step gradient gate failed: {first_gradients}")

    final_lora = [parameter.detach().clone() for parameter in lora_parameters]
    results: dict[str, Any] = {}
    for table_name, logits in (
        ("native", torch.zeros_like(saved["gap_delta_logits"])),
        ("learned", saved["gap_delta_logits"]),
    ):
        with torch.no_grad():
            allocation.gap_delta_logits.copy_(logits.to(allocation.gap_delta_logits.device))
        for lora_name, enabled in (("off", False), ("on", True)):
            with torch.no_grad():
                for parameter, final_value in zip(lora_parameters, final_lora, strict=True):
                    parameter.copy_(final_value if enabled else torch.zeros_like(final_value))
            key = f"{table_name}_table__recovery_lora_{lora_name}"
            results[key] = {
                "heldout_4k": _evaluate_4k(
                    model=model,
                    backbone=backbone,
                    loss_module=loss_module,
                    tokens=tokens,
                    rows=validation_rows,
                ),
                "long_natural": {
                    str(rows.shape[1]): _evaluate_view(
                        model=model,
                        backbone=backbone,
                        loss_module=loss_module,
                        rows=rows,
                    )
                    for rows, _ in long_views.values()
                },
            }

    with torch.no_grad():
        allocation.gap_delta_logits.copy_(
            training_logits.to(allocation.gap_delta_logits.device)
        )
        for parameter, final_value in zip(lora_parameters, final_lora, strict=True):
            parameter.copy_(final_value)
    adapter = incomplete / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    result = {
        "status": STATUS,
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "oracle_result": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
        "protocol": {
            "steps": 300,
            "physical_length": 4096,
            "objective": "dense next-token natural-LM NLL",
            "allocation": f"frozen {args.training_table} fixed-support table",
            "trainable": "Q/K LoRA only, rank 64, alpha 128",
            "target_length_used_by_training": False,
            "train_rows": int(len(train_rows)),
            "heldout_rows": int(len(validation_rows)),
            "excluded_long_document_rows": sorted(excluded),
        },
        "environment": environment,
        "training_data": {
            "tensor_sha256": train_receipt["tensor_sha256"],
            "receipt_sha256": sha256_file(args.train_receipt.resolve()),
        },
        "first_step_gradients": first_gradients,
        "runtime_seconds": time.perf_counter() - started,
        "processed_input_tokens": processed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "results": results,
        "adapter": {
            "config_sha256": sha256_file(adapter / "adapter_config.json"),
            "model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        },
    }
    atomic_json(incomplete / "result.json", result)
    incomplete.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
