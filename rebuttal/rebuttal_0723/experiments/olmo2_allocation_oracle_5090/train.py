#!/usr/bin/env python3
"""Train one shared fixed-support RoPE allocation with Q/K LoRA.

The run starts exactly at the released Native table.  Interior normalized gaps
and Q/K LoRA co-adapt under a deterministic mixture of Native, medium, and far
phase exposures plus promptless 4K language-model replay.  No target table,
target evaluation length, attention temperature, router, or second RoPE path is
part of the intervention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
    configure_flash_only_attention,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.phase_adaptation import (
    PhaseAdaptationView,
    phase_batch,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    fused_loss_module,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope import (
    RawReplayView,
    RetentionView,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    seed_everything,
)
from scripts.lib.rope.fixed_support_z import FixedSupportZRotaryEmbedding

from .oracle import (
    FAMILY_PATTERN,
    METHOD_ID,
    NATIVE_LENGTH,
    code_hashes,
    deterministic_offsets,
    position_ids_for_offsets,
    protocol,
    sha256_file,
    tensor_sha256,
)


READY_STATUS = "OLMO2_ALLOCATION_ORACLE_READY_V1"
RESULT_STATUS = "OLMO2_ALLOCATION_ORACLE_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--phase-view", type=Path, required=True)
    parser.add_argument("--raw-replay", type=Path, required=True)
    parser.add_argument("--retention-view", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def _base_model(model: Any) -> Any:
    value = getattr(model, "_orig_mod", model)
    return value.get_base_model() if hasattr(value, "get_base_model") else value


def _load_model(checkpoint: Path) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if (
        base.config.model_type != "olmo2"
        or int(base.config.hidden_size) != 2_048
        or int(base.config.num_hidden_layers) != 16
        or int(base.config.num_attention_heads) != 16
        or int(base.config.num_key_value_heads) != 16
        or int(base.config.max_position_embeddings) != NATIVE_LENGTH
        or float(
            getattr(base.config, "rope_theta", None)
            or (getattr(base.config, "rope_parameters", None) or {}).get("rope_theta")
        )
        != 500_000.0
    ):
        raise RuntimeError("released OLMo-2 architecture identity drift")
    configure_flash_only_attention(base)
    base.config.max_position_embeddings = 16 * NATIVE_LENGTH
    model = get_peft_model(
        base,
        LoraConfig(
            r=64,
            lora_alpha=128.0,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    rotary = _base_model(model).model.rotary_emb
    allocation = FixedSupportZRotaryEmbedding(rotary.inv_freq.detach().clone())
    _base_model(model).model.rotary_emb = allocation
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if (
        sum(name.endswith("gap_delta_logits") for name in names) != 1
        or any(
            "lora_A" not in name
            and "lora_B" not in name
            and not name.endswith("gap_delta_logits")
            for name in names
        )
    ):
        raise RuntimeError(f"oracle trainable scope drift: {names}")
    return model, allocation


def _verify_ready(args: argparse.Namespace) -> dict[str, Any]:
    ready = json.loads(args.ready_receipt.resolve().read_text(encoding="utf-8"))
    expected_protocol = protocol(steps=int(args.steps), seed=int(args.seed), smoke=bool(args.smoke))
    if (
        ready.get("status") != READY_STATUS
        or ready.get("protocol") != expected_protocol
        or ready.get("code_sha256") != code_hashes()
        or Path(ready["output"]).resolve() != args.output.resolve()
    ):
        raise RuntimeError("allocation-oracle READY receipt drift")
    for name, path in {
        "checkpoint": args.checkpoint,
        "checkpoint_ready_receipt": args.checkpoint_ready_receipt,
        "phase_view": args.phase_view,
        "raw_replay": args.raw_replay,
        "retention_view": args.retention_view,
    }.items():
        if Path(ready["inputs"][name]["path"]).resolve() != path.resolve():
            raise RuntimeError(f"READY path drift: {name}")
    return ready


def _sparse_raw_loss(
    *, model: Any, backbone: Any, rows: np.ndarray, view: RawReplayView
) -> torch.Tensor:
    input_ids = torch.as_tensor(
        np.asarray(view.input_ids[rows]), device="cuda", dtype=torch.long
    )
    targets = torch.arange(64, input_ids.shape[1], 64, device="cuda")
    hidden = backbone(input_ids[:, :-1], None)
    logits = _base_model(model).lm_head(hidden[:, targets - 1]).float()
    labels = input_ids[:, targets]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


@torch.inference_mode()
def _evaluate_phase(
    *,
    model: Any,
    backbone: Any,
    loss_module: Any,
    view: PhaseAdaptationView,
    rows: np.ndarray,
    offset: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact = 0
    for start in range(0, len(rows), 4):
        indices = rows[start : start + 4]
        contexts, labels, supervised = phase_batch(view=view, indices=indices)
        host_positions = position_ids_for_offsets(
            query_starts=np.asarray(view.query_starts[indices]),
            offsets=np.full(len(indices), int(offset), dtype=np.int64),
        )
        position_ids = torch.from_numpy(host_positions).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(contexts, position_ids)
            value = loss_module(
                _base_model(model).lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )
            value = value.loss if hasattr(value, "loss") else value
        mask = labels != -100
        logits = _base_model(model).lm_head(hidden[mask]).float()
        exact += int(logits.argmax(-1).eq(labels[mask]).sum())
        total_loss += float(value) * int(supervised)
        total_tokens += int(supervised)
        del contexts, labels, position_ids, hidden, value, logits
    mean = total_loss / max(total_tokens, 1)
    return {
        "offset": int(offset),
        "rows": int(len(rows)),
        "supervised_tokens": int(total_tokens),
        "mean_nll": float(mean),
        "perplexity": float(math.exp(min(mean, 50.0))),
        "token_exact": float(exact / max(total_tokens, 1)),
    }


@torch.inference_mode()
def _evaluate_retention(*, model: Any, backbone: Any, view: RetentionView) -> dict[str, Any]:
    model.eval()
    losses = []
    for start in range(0, len(view.input_ids), 4):
        input_ids = torch.as_tensor(
            np.asarray(view.input_ids[start : start + 4]),
            device="cuda",
            dtype=torch.long,
        )
        targets = torch.arange(64, input_ids.shape[1], 64, device="cuda")
        hidden = backbone(input_ids[:, :-1], None)
        logits = _base_model(model).lm_head(hidden[:, targets - 1]).float()
        labels = input_ids[:, targets]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        losses.append(float(loss))
        del input_ids, hidden, logits, labels, loss
    return {"rows": int(len(view.input_ids)), "stride": 64, "mean_nll": float(np.mean(losses))}


def _evaluate(
    *, model: Any, backbone: Any, loss_module: Any, phase: PhaseAdaptationView,
    retention: RetentionView, rows: np.ndarray,
) -> dict[str, Any]:
    return {
        "phase": {
            str(offset): _evaluate_phase(
                model=model,
                backbone=backbone,
                loss_module=loss_module,
                view=phase,
                rows=rows,
                offset=offset,
            )
            for offset in (0, NATIVE_LENGTH, 3 * NATIVE_LENGTH, 15 * NATIVE_LENGTH)
        },
        "retention_4k": _evaluate_retention(model=model, backbone=backbone, view=retention),
    }


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED") != "YES":
        raise PermissionError(
            "GPU execution requires --authorize and OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED=YES"
        )
    if int(args.steps) <= 0 or int(args.validation_rows) not in range(1, 129):
        raise ValueError("steps and validation rows are outside the registered bounds")
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("allocation-oracle output path must be new")
    _verify_ready(args)

    checkpoint = args.checkpoint.resolve()
    phase = PhaseAdaptationView(args.phase_view.resolve())
    raw = RawReplayView.load(args.raw_replay.resolve())
    retention = RetentionView.load(args.retention_view.resolve())
    rows = phase.validation_rows[: int(args.validation_rows)]
    if len(rows) != int(args.validation_rows):
        raise RuntimeError("phase validation row count drift")

    incomplete.mkdir(parents=True)
    seed_everything(int(args.seed))
    environment = configure_cuda()
    model, allocation = _load_model(checkpoint)
    model.to("cuda")
    backbone = torch.compile(
        TrainingBackbone(_base_model(model).model),
        fullgraph=True,
        dynamic=False,
        mode=str(args.compile_mode),
    )
    loss_module = fused_loss_module()
    baseline = _evaluate(
        model=model,
        backbone=backbone,
        loss_module=loss_module,
        phase=phase,
        retention=retention,
        rows=rows,
    )

    lora_parameters = [
        value
        for name, value in model.named_parameters()
        if value.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    allocation_parameters = [allocation.gap_delta_logits]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": 5e-5, "base_lr": 5e-5, "weight_decay": 0.0, "name": "qk_lora"},
            {"params": allocation_parameters, "lr": 1e-3, "base_lr": 1e-3, "weight_decay": 0.0, "name": "allocation"},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 91_003)
    steps = 1 if args.smoke else int(args.steps)
    started = time.perf_counter()
    processed = 0
    first_gradients: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(1, steps + 1):
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        scale = cosine_lr(step, steps, min(20, steps), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for accumulation in range(2):
            if family == "phase":
                local = torch.randint(
                    len(phase.training_rows), (4,), generator=generator
                ).numpy()
                indices = phase.training_rows[local]
                contexts, labels, _ = phase_batch(view=phase, indices=indices)
                offsets = deterministic_offsets(
                    seed=int(args.seed), step=step, accumulation=accumulation
                )
                host_positions = position_ids_for_offsets(
                    query_starts=np.asarray(phase.query_starts[indices]),
                    offsets=offsets,
                )
                position_ids = torch.from_numpy(host_positions).to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts, position_ids)
                    value = loss_module(
                        _base_model(model).lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        labels.reshape(-1),
                    )
                    value = value.loss if hasattr(value, "loss") else value
                processed += int(contexts.numel())
                del contexts, labels, position_ids, hidden
            else:
                indices = torch.randint(
                    len(raw.input_ids), (4,), generator=generator
                ).numpy()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    value = _sparse_raw_loss(
                        model=model, backbone=backbone, rows=indices, view=raw
                    )
                processed += int(4 * NATIVE_LENGTH)
            if not torch.isfinite(value):
                raise RuntimeError(f"non-finite loss at step {step}")
            (value / 2.0).backward()
            losses.append(float(value.detach()))
            del value
        if step == 1:
            for name, values in {
                "qk_lora": lora_parameters,
                "allocation": allocation_parameters,
            }.items():
                grads = [value.grad for value in values]
                first_gradients[name] = {
                    "present": bool(grads and all(grad is not None for grad in grads)),
                    "finite": bool(grads and all(torch.isfinite(grad).all() for grad in grads if grad is not None)),
                    "nonzero": bool(grads and any(torch.count_nonzero(grad).item() for grad in grads if grad is not None)),
                }
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters + allocation_parameters, 1.0
        )
        optimizer.step()
        allocation.project_()
        if step == 1 or step % 25 == 0 or step == steps:
            append_jsonl(
                incomplete / "train_log.jsonl",
                {
                    "step": step,
                    "family": family,
                    "loss": float(np.mean(losses)),
                    "grad_norm": float(grad_norm),
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "processed_input_tokens": processed,
                    "maximum_coordinate_shift": allocation.receipt()["maximum_normalized_coordinate_shift"],
                    "minimum_normalized_gap": allocation.receipt()["minimum_normalized_gap"],
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    if any(not all(value.values()) for value in first_gradients.values()):
        raise RuntimeError(f"first-step gradient gate failed: {first_gradients}")
    free_bytes = int(torch.cuda.mem_get_info()[0])
    if free_bytes < 1 << 30:
        raise RuntimeError("less than 1 GiB CUDA headroom")
    final = _evaluate(
        model=model,
        backbone=backbone,
        loss_module=loss_module,
        phase=phase,
        retention=retention,
        rows=rows,
    )

    adapter = incomplete / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    table = allocation.realized_inv_freq().detach().cpu().numpy().astype("<f4")
    np.save(incomplete / "allocation_table.npy", table, allow_pickle=False)
    torch.save(
        {
            "method": METHOD_ID,
            "gap_delta_logits": allocation.gap_delta_logits.detach().cpu(),
            "native_inv_freq": allocation.native_inv_freq.detach().cpu(),
            "realized_inv_freq": allocation.realized_inv_freq().detach().cpu(),
        },
        incomplete / "allocation_state.pt",
    )
    result = {
        "status": RESULT_STATUS,
        "smoke": bool(args.smoke),
        "protocol": protocol(steps=int(args.steps), seed=int(args.seed), smoke=bool(args.smoke)),
        "code_sha256": code_hashes(),
        "ready_receipt_sha256": sha256_file(args.ready_receipt.resolve()),
        "environment": environment,
        "baseline": baseline,
        "final": final,
        "allocation": allocation.receipt(),
        "allocation_table_float32_sha256": tensor_sha256(allocation.realized_inv_freq()),
        "first_step_gradients": first_gradients,
        "runtime_seconds": time.perf_counter() - started,
        "processed_input_tokens": processed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_memory_bytes": free_bytes,
        "artifacts": {
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "allocation_state_sha256": sha256_file(incomplete / "allocation_state.pt"),
            "allocation_table_sha256": sha256_file(incomplete / "allocation_table.npy"),
        },
    }
    atomic_json(incomplete / "result.json", result)
    incomplete.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
