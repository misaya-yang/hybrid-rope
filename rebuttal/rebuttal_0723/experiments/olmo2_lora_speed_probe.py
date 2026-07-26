#!/usr/bin/env python3
"""Bounded 16K LoRA throughput probe for RTX 5090 execution settings."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    forward_hidden,
    install_adaptation,
    load_model,
    logits_from_hidden,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    TRAIN_DISTRACTOR_COUNTS,
    TRAIN_SOURCE_FRACTIONS,
    apply_schedule,
    build_probe_set,
    load_documents,
    one_token_values,
    select_document_rows,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)


class LastTokenBackbone(nn.Module):
    """Compile-friendly OLMo backbone returning only the supervised state."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        return hidden[:, -1, :]


def train_step(
    *,
    model: Any,
    backbone: nn.Module | None,
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    sourced: np.ndarray,
    gold: np.ndarray,
    indices: np.ndarray,
    micro_batch_size: int,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    raw_losses: list[float] = []
    micro_batches = len(indices) // int(micro_batch_size)
    for start in range(0, len(indices), int(micro_batch_size)):
        local = indices[start : start + int(micro_batch_size)]
        context = torch.from_numpy(sourced[local]).to(
            "cuda", non_blocking=True
        )
        labels = torch.from_numpy(gold[local]).to(
            "cuda", non_blocking=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = (
                forward_hidden(model, context, all_positions=False)
                if backbone is None
                else backbone(context)
            )
            logits = logits_from_hidden(model, hidden, None)
            raw_loss = F.cross_entropy(logits.float(), labels)
            loss = raw_loss / float(micro_batches)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite speed-probe loss")
        loss.backward()
        raw_losses.append(float(raw_loss.detach()))
        del context, labels, hidden, logits, raw_loss, loss
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return float(np.mean(raw_losses)), float(grad_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--documents-16k", type=Path, required=True)
    parser.add_argument(
        "--documents-16k-metadata", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, required=True
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=20_260_725)
    args = parser.parse_args()

    global_batch = (
        int(args.micro_batch_size)
        * int(args.gradient_accumulation_steps)
    )
    if global_batch != 4:
        raise RuntimeError(
            "speed probe freezes global batch at four sequences"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    train_values, _ = one_token_values(tokenizer)
    documents_path = args.documents_16k.resolve()
    metadata_path = args.documents_16k_metadata.resolve()
    documents, metadata = load_documents(
        documents_path,
        metadata_path,
        expected_length=16_384,
    )
    document_rows = select_document_rows(
        metadata, split="validation"
    )
    data = build_probe_set(
        tokenizer=tokenizer,
        documents=documents,
        document_metadata=metadata,
        document_rows=document_rows,
        values=train_values,
        length=16_384,
        count=64,
        seed=int(args.seed) + 1,
        source_fractions=TRAIN_SOURCE_FRACTIONS,
        distractor_counts=TRAIN_DISTRACTOR_COUNTS,
        phase="train",
    )
    model = load_model(checkpoint)
    frequency = apply_schedule(model, "geo")
    readout = install_adaptation(
        model, "qkvo_answer", rank=64, alpha=128.0
    )
    if readout is not None:
        raise RuntimeError("unexpected readout adapter")
    model.to("cuda")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    backbone: nn.Module | None = None
    if args.compile_mode != "none":
        backbone = LastTokenBackbone(model.model)
        backbone = torch.compile(
            backbone,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 31_337)
    total_steps = int(args.warmup_steps) + int(args.measured_steps)
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    try:
        for step in range(1, total_steps + 1):
            indices = torch.randint(
                len(data.gold),
                (global_batch,),
                generator=generator,
            ).numpy()
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, grad_norm = train_step(
                model=model,
                backbone=backbone,
                parameters=parameters,
                optimizer=optimizer,
                sourced=data.sourced,
                gold=data.gold,
                indices=indices,
                micro_batch_size=int(args.micro_batch_size),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "step": step,
                    "measured": step > int(args.warmup_steps),
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": (
                        global_batch * (16_384 - 1) / elapsed
                    ),
                    "loss": loss,
                    "grad_norm": grad_norm,
                }
            )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        receipt = {
            "status": "OLMO2_LORA_SPEED_PROBE_OOM",
            "error": str(exc),
            "configuration": {
                "micro_batch_size": int(args.micro_batch_size),
                "gradient_accumulation_steps": int(
                    args.gradient_accumulation_steps
                ),
                "gradient_checkpointing": bool(
                    args.gradient_checkpointing
                ),
            },
            "runtime": runtime,
            "rows": rows,
        }
        atomic_json(output / "results.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    measured = [row for row in rows if row["measured"]]
    receipt = {
        "status": "OLMO2_LORA_SPEED_PROBE_COMPLETE",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "documents_sha256": sha256_file(documents_path),
        "metadata_sha256": sha256_file(metadata_path),
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "data_sha256": data.digest(),
        "configuration": {
            "sequence_length": 16_384,
            "global_batch_sequences": global_batch,
            "global_batch_tokens": global_batch * 16_384,
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "gradient_checkpointing": bool(
                args.gradient_checkpointing
            ),
            "precision": "bf16_autocast",
            "attention": "flash_only",
            "optimizer": "fused_adamw",
            "compile_mode": args.compile_mode,
            "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "runtime": runtime,
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "mean_tokens_per_second": float(
            np.mean([row["tokens_per_second"] for row in measured])
        ),
        "median_tokens_per_second": float(
            np.median([row["tokens_per_second"] for row in measured])
        ),
        "mean_step_seconds": float(
            np.mean([row["elapsed_seconds"] for row in measured])
        ),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
        "rows": rows,
    }
    atomic_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
