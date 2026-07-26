#!/usr/bin/env python3
"""Continue a frozen Llama-3-8B adapter on the physical-8K RULER mixture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AttentionMaskInterface, AutoModelForCausalLM

from experiments.lora_evq_v2.train_evq_lora import (
    PACKED_FREE_CAUSAL_SDPA_BACKEND,
    configure_packed_free_causal_sdpa,
    inject_inv_freq,
    load_frequency_artifact,
    verify_model_inv_freq,
)

from .common import (
    READY_STATUS,
    RESULT_STATUS,
    TRAIN_LENGTH,
    VIEW_STATUS,
    append_jsonl,
    atomic_json,
    canonical_json_sha256,
    configure_cuda,
    sha256_file,
)


class TrainingBackbone(torch.nn.Module):
    """Compile only the gradient-bearing transformer backbone."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=False,
            return_dict=False,
        )[0]


def packed_free_causal_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    if attention_mask is not None:
        raise RuntimeError("physical-8K training forbids padding masks")
    return None


class FixedView:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("status") != VIEW_STATUS:
            raise RuntimeError("training-view status drift")
        storage = self.manifest["storage"]
        for name, receipt in storage["files"].items():
            path = root / name
            if (
                not path.is_file()
                or path.stat().st_size != int(receipt["size_bytes"])
                or sha256_file(path) != receipt["sha256"]
            ):
                raise RuntimeError(f"training-view file drift: {path}")
        self.input_ids = np.load(
            root / "input_ids.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.assistant_mask = np.load(
            root / "assistant_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.lengths = np.load(
            root / "lengths.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.split = np.load(
            root / "split.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            self.input_ids.shape != self.assistant_mask.shape
            or self.input_ids.shape[1] != TRAIN_LENGTH
            or len(self.lengths) != len(self.input_ids)
            or len(self.split) != len(self.input_ids)
        ):
            raise RuntimeError("training-view storage shape drift")
        self.training_rows = np.flatnonzero(self.split == 0)
        self.validation_rows = np.flatnonzero(self.split == 1)
        if (
            len(self.training_rows),
            len(self.validation_rows),
        ) != (1_376, 52):
            raise RuntimeError("registered training-view split drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("evq_cosh", "native_geo"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
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
    parser.add_argument("--discarded-steady-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_420_726)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--required-gpu-substring",
        default="RTX PRO 6000",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
) -> float:
    if step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(
        1,
        total_steps - warmup_steps,
    )
    return peak_lr * 0.1 + peak_lr * 0.9 * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def fused_loss_module() -> Any:
    from liger_kernel.transformers import (
        LigerFusedLinearCrossEntropyLoss,
    )

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100,
        reduction="mean",
        return_z_loss=False,
        accum_dtype=torch.float32,
    )


def causal_parts(model: PeftModel) -> tuple[torch.nn.Module, torch.nn.Module]:
    base = model.get_base_model()
    backbone = getattr(base, "model", None)
    head = getattr(base, "lm_head", None)
    if backbone is None or head is None:
        raise RuntimeError("expected a causal LM backbone and lm_head")
    return backbone, head


def make_batch(
    view: FixedView,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    host_ids = np.asarray(view.input_ids[indices], dtype=np.int64)
    host_mask = np.asarray(
        view.assistant_mask[indices],
        dtype=np.uint8,
    )
    context = torch.from_numpy(host_ids).to("cuda", non_blocking=True)
    labels = torch.full_like(context, -100)
    next_tokens = torch.from_numpy(host_ids[:, 1:].copy()).to(
        "cuda",
        non_blocking=True,
    )
    next_mask = torch.from_numpy(host_mask[:, 1:].copy()).to(
        "cuda",
        non_blocking=True,
    ).bool()
    labels[:, :-1] = torch.where(
        next_mask,
        next_tokens,
        torch.full_like(next_tokens, -100),
    )
    supervised = int(next_mask.sum())
    if supervised <= 0:
        raise RuntimeError("batch has no supervised answer tokens")
    return context, labels, supervised


def loss_value(
    *,
    backbone: torch.nn.Module,
    lm_head: torch.nn.Module,
    loss_module: Any,
    context: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(context)
        value = loss_module(
            lm_head.weight,
            hidden.reshape(-1, hidden.shape[-1]),
            labels.reshape(-1),
        )
        return value.loss if hasattr(value, "loss") else value


@torch.no_grad()
def evaluate_validation(
    *,
    model: PeftModel,
    view: FixedView,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    backbone_module, lm_head = causal_parts(model)
    backbone = TrainingBackbone(backbone_module)
    loss_module = fused_loss_module()
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for offset in range(0, len(view.validation_rows), batch_size):
        indices = view.validation_rows[offset : offset + batch_size]
        context, labels, supervised = make_batch(view, indices)
        value = loss_value(
            backbone=backbone,
            lm_head=lm_head,
            loss_module=loss_module,
            context=context,
            labels=labels,
        )
        if not torch.isfinite(value):
            raise RuntimeError("non-finite validation NLL")
        total_nll += float(value) * supervised
        total_tokens += supervised
        del context, labels, value
    torch.cuda.synchronize()
    mean_nll = total_nll / total_tokens
    return {
        "rows": int(len(view.validation_rows)),
        "supervised_tokens": int(total_tokens),
        "mean_nll": float(mean_nll),
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_parent(parent: Path, method: str) -> dict[str, Any]:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "experiment_meta.json",
        "custom_inv_freq.pt",
    )
    for name in required:
        if not (parent / name).is_file():
            raise FileNotFoundError(parent / name)
    metadata = json.loads(
        (parent / "experiment_meta.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (parent / "adapter_config.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("status") != "complete"
        or int(metadata.get("global_step", -1)) != 300
        or metadata.get("rope_method") != method
        or int(metadata.get("max_seq_len", -1)) != TRAIN_LENGTH
    ):
        raise RuntimeError("frozen parent adapter identity drift")
    if (
        int(config.get("r", -1)) != 64
        or int(config.get("lora_alpha", -1)) != 128
        or set(config.get("target_modules", []))
        != {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise RuntimeError("frozen parent LoRA configuration drift")
    return {
        "metadata": metadata,
        "files": {
            name: {
                "sha256": sha256_file(parent / name),
                "size_bytes": (parent / name).stat().st_size,
            }
            for name in required
        },
    }


def validate_ready(
    *,
    path: Path,
    checkpoint: Path,
    parent: Path,
    view: Path,
    method: str,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("READY receipt status drift")
    if method not in receipt["arms"]:
        raise RuntimeError("requested arm is absent from READY receipt")
    expected = {
        "checkpoint": checkpoint,
        "training_view": view,
        "parent_adapter": parent,
    }
    arm = receipt["arms"][method]
    recorded = {
        "checkpoint": receipt["inputs"]["checkpoint"]["path"],
        "training_view": receipt["inputs"]["training_view"]["path"],
        "parent_adapter": arm["parent_adapter"]["path"],
    }
    for name, actual_path in expected.items():
        if Path(recorded[name]).resolve() != actual_path.resolve():
            raise RuntimeError(f"READY {name} path drift")
    if sha256_file(view / "manifest.json") != receipt["inputs"][
        "training_view"
    ]["manifest_sha256"]:
        raise RuntimeError("READY training-view manifest drift")
    for name, record in arm["parent_adapter"]["files"].items():
        candidate = parent / name
        if (
            candidate.stat().st_size != int(record["size_bytes"])
            or sha256_file(candidate) != record["sha256"]
        ):
            raise RuntimeError(f"READY parent file drift: {candidate}")
    code_root = Path(receipt["code"]["root"])
    for relative, record in receipt["code"]["files"].items():
        candidate = code_root / relative
        if (
            candidate.stat().st_size != int(record["size_bytes"])
            or sha256_file(candidate) != record["sha256"]
        ):
            raise RuntimeError(f"READY code drift: {candidate}")
    return receipt


def trainable_parameters(model: PeftModel) -> list[torch.nn.Parameter]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named or any("lora_" not in name for name, _ in named):
        raise RuntimeError("only LoRA parameters may be trainable")
    return [parameter for _, parameter in named]


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def save_recovery(
    *,
    incomplete: Path,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    completed_step: int,
    initial_validation: dict[str, Any],
) -> None:
    destination = incomplete / "recovery"
    temporary = incomplete / "recovery.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(
        temporary / "adapter",
        safe_serialization=True,
    )
    torch.save(
        {
            "completed_step": int(completed_step),
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "initial_validation": initial_validation,
        },
        temporary / "state.pt",
    )
    atomic_json(
        temporary / "progress.json",
        {"completed_step": int(completed_step)},
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    parent = args.parent_adapter.resolve()
    view_root = args.training_view.resolve()
    ready_path = args.ready_receipt.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    global_batch = (
        int(args.micro_batch_size)
        * int(args.gradient_accumulation_steps)
    )
    if int(args.epochs) != 3:
        raise ValueError("registered protocol requires exactly three epochs")
    if global_batch != 8:
        raise ValueError("registered global batch size is exactly eight")
    if not math.isclose(float(args.learning_rate), 2e-5):
        raise ValueError("registered peak learning rate is 2e-5")
    if not math.isclose(float(args.warmup_ratio), 0.05):
        raise ValueError("registered warmup ratio is 0.05")
    if int(args.discarded_steady_steps) < 5:
        raise ValueError("at least five discarded steady steps are required")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    if output.exists():
        raise FileExistsError(output)
    if incomplete.exists() and not args.resume:
        raise FileExistsError(
            f"{incomplete} exists; pass --resume only for a valid recovery"
        )
    if args.probe_only and (output.exists() or incomplete.exists()):
        raise FileExistsError(output if output.exists() else incomplete)

    view = FixedView(view_root)
    parent_receipt = validate_parent(parent, args.method)
    ready = validate_ready(
        path=ready_path,
        checkpoint=checkpoint,
        parent=parent,
        view=view_root,
        method=args.method,
    )
    runtime = configure_cuda()
    if args.required_gpu_substring not in runtime["gpu_name"]:
        raise RuntimeError(
            f"wrong GPU: {runtime['gpu_name']!r} does not contain "
            f"{args.required_gpu_substring!r}"
        )

    recovery_state: dict[str, Any] | None = None
    adapter_source = parent
    if args.resume:
        recovery = incomplete / "recovery"
        if not (recovery / "adapter").is_dir() or not (
            recovery / "state.pt"
        ).is_file():
            raise RuntimeError("resume requested without a complete recovery")
        adapter_source = recovery / "adapter"
        recovery_state = torch.load(
            recovery / "state.pt",
            map_location="cpu",
            weights_only=False,
        )
    elif not args.probe_only:
        incomplete.mkdir(parents=True)

    seed_everything(int(args.seed))
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    if (
        base.config.model_type != "llama"
        or int(base.config.max_position_embeddings) != TRAIN_LENGTH
        or int(base.config.hidden_size) != 4_096
        or int(base.config.num_hidden_layers) != 32
    ):
        raise RuntimeError("Llama-3-8B base geometry drift")
    base.config.use_cache = False
    attention_backend = configure_packed_free_causal_sdpa(base)
    if attention_backend != PACKED_FREE_CAUSAL_SDPA_BACKEND:
        raise RuntimeError("packed-free attention backend identity drift")
    AttentionMaskInterface.register(
        PACKED_FREE_CAUSAL_SDPA_BACKEND,
        packed_free_causal_mask,
    )
    inv_freq, frequency_metadata, frequency_receipt = (
        load_frequency_artifact(
            parent / "custom_inv_freq.pt",
            expected_method=args.method,
        )
    )
    inject_inv_freq(base, inv_freq)
    verify_model_inv_freq(base, inv_freq)
    model = PeftModel.from_pretrained(
        base,
        adapter_source,
        is_trainable=True,
    )
    inject_inv_freq(model, inv_freq)
    frequency_verification = verify_model_inv_freq(model, inv_freq)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    parameters = trainable_parameters(model)
    trainable_count = sum(parameter.numel() for parameter in parameters)

    backbone_module, lm_head = causal_parts(model)
    eager_backbone = TrainingBackbone(backbone_module)
    compiled_backbone = torch.compile(
        eager_backbone,
        fullgraph=False,
        dynamic=False,
        mode=args.compile_mode,
    )
    loss_module = fused_loss_module()

    probe_indices = view.training_rows[: int(args.micro_batch_size)]
    model.train()
    torch.cuda.reset_peak_memory_stats()
    probe_durations: list[float] = []
    probe_tokens = int(args.micro_batch_size) * TRAIN_LENGTH
    for probe_step in range(1 + int(args.discarded_steady_steps)):
        for parameter in parameters:
            parameter.grad = None
        context, labels, _ = make_batch(view, probe_indices)
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = loss_value(
            backbone=compiled_backbone,
            lm_head=lm_head,
            loss_module=loss_module,
            context=context,
            labels=labels,
        )
        if not torch.isfinite(value):
            raise RuntimeError("non-finite compile/probe loss")
        value.backward()
        torch.cuda.synchronize()
        probe_durations.append(time.perf_counter() - started)
        del context, labels, value
    for parameter in parameters:
        parameter.grad = None
    probe = {
        "compile_plus_first_step_seconds": probe_durations[0],
        "steady_steps": int(args.discarded_steady_steps),
        "steady_step_seconds": probe_durations[1:],
        "steady_tokens_per_second": probe_tokens
        * int(args.discarded_steady_steps)
        / sum(probe_durations[1:]),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": global_batch,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "compile_mode": args.compile_mode,
        "compile_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
    }
    if args.probe_only:
        probe_result = {
            "status": "LLAMA8B_RULER_MIX_PROBE_COMPLETE_V1",
            "method": args.method,
            "runtime": runtime,
            "attention_backend": attention_backend,
            "frequency": frequency_receipt,
            "probe": probe,
        }
        atomic_json(output, probe_result)
        print(json.dumps(probe_result, indent=2, sort_keys=True))
        return

    steps_per_epoch = len(view.training_rows) // global_batch
    total_steps = int(args.epochs) * steps_per_epoch
    warmup_steps = max(
        1,
        int(round(float(args.warmup_ratio) * total_steps)),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 61_003)
    ordered_rows = np.concatenate(
        [
            torch.as_tensor(view.training_rows)[
                torch.randperm(len(view.training_rows), generator=generator)
            ].numpy()
            for _ in range(int(args.epochs))
        ]
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )

    if recovery_state is None:
        seed_everything(int(args.seed))
        initial_validation = evaluate_validation(
            model=model,
            view=view,
            batch_size=int(args.micro_batch_size),
        )
        start_step = 0
        seed_everything(int(args.seed))
    else:
        start_step = int(recovery_state["completed_step"])
        if not 0 < start_step < total_steps:
            raise RuntimeError("recovery step is outside the training range")
        if start_step % steps_per_epoch:
            raise RuntimeError("only epoch-boundary recovery is admitted")
        optimizer.load_state_dict(recovery_state["optimizer"])
        optimizer_to_device(optimizer, torch.device("cuda", 0))
        initial_validation = recovery_state["initial_validation"]
        random.setstate(recovery_state["python_rng_state"])
        np.random.set_state(recovery_state["numpy_rng_state"])
        torch.set_rng_state(recovery_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(recovery_state["cuda_rng_state_all"])

    log_path = incomplete / "train_log.jsonl"
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = start_step * global_batch * TRAIN_LENGTH
    processed_tokens = last_log_tokens
    supervised_tokens = 0
    recent_losses: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step + 1, total_steps + 1):
        lr = cosine_lr(
            step,
            total_steps,
            warmup_steps,
            float(args.learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_indices = ordered_rows[
            (step - 1) * global_batch : step * global_batch
        ]
        micro_batches = []
        step_supervised = 0
        for micro in range(int(args.gradient_accumulation_steps)):
            indices = batch_indices[
                micro
                * int(args.micro_batch_size) : (micro + 1)
                * int(args.micro_batch_size)
            ]
            context, labels, supervised = make_batch(view, indices)
            micro_batches.append((context, labels, supervised))
            step_supervised += supervised
        weighted_nll = 0.0
        for context, labels, supervised in micro_batches:
            value = loss_value(
                backbone=compiled_backbone,
                lm_head=lm_head,
                loss_module=loss_module,
                context=context,
                labels=labels,
            )
            weight = supervised / step_supervised
            loss = value * weight
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            weighted_nll += float(value.detach()) * weight
            del context, labels, value, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        processed_tokens += global_batch * TRAIN_LENGTH
        supervised_tokens += step_supervised
        recent_losses.append(weighted_nll)

        if step == 1 or step % 25 == 0 or step == total_steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "total_steps": total_steps,
                    "epoch": 1 + (step - 1) // steps_per_epoch,
                    "loss": weighted_nll,
                    "mean_loss_last_25": float(
                        np.mean(recent_losses[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "processed_input_tokens": processed_tokens,
                    "supervised_tokens_since_process_start": (
                        supervised_tokens
                    ),
                    "elapsed_seconds_since_process_start": now - started,
                    "interval_tokens_per_second": (
                        processed_tokens - last_log_tokens
                    )
                    / max(now - last_log_time, 1e-9),
                    "peak_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
            )
            last_log_time = now
            last_log_tokens = processed_tokens
        if step % steps_per_epoch == 0 and step < total_steps:
            save_recovery(
                incomplete=incomplete,
                model=model,
                optimizer=optimizer,
                completed_step=step,
                initial_validation=initial_validation,
            )

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    final_validation = evaluate_validation(
        model=model,
        view=view,
        batch_size=int(args.micro_batch_size),
    )
    model.save_pretrained(incomplete, safe_serialization=True)
    shutil.copy2(
        parent / "custom_inv_freq.pt",
        incomplete / "custom_inv_freq.pt",
    )
    adapter_path = incomplete / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError("PEFT did not save adapter_model.safetensors")
    result = {
        "status": RESULT_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "method": args.method,
        "scientific_contract": {
            "physical_training_length": TRAIN_LENGTH,
            "virtual_position_ids": False,
            "long_context_backward": False,
            "objective": "answer_only_ruler_family_plus_natural_replay",
            "epochs": int(args.epochs),
            "training_rows": int(len(view.training_rows)),
            "global_batch_size": global_batch,
            "processed_input_tokens": int(
                len(view.training_rows) * TRAIN_LENGTH * int(args.epochs)
            ),
            "peak_learning_rate": float(args.learning_rate),
            "warmup_steps": warmup_steps,
            "scheduler": "cosine_to_0.1x_peak",
            "optimizer": "fused_adamw",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "seed": int(args.seed),
        },
        "execution": {
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "precision": "bf16_autocast",
            "attention_backend": attention_backend,
            "flash_only": True,
            "gradient_checkpointing": False,
            "compile_mode": args.compile_mode,
            "compile_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "loss_backend": "liger_fused_linear_cross_entropy",
            "probe": probe,
            "training_seconds_since_process_start": training_seconds,
            "tokens_per_second_since_process_start": (
                (total_steps - start_step) * global_batch * TRAIN_LENGTH
                / training_seconds
            ),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "liger_kernel": importlib.metadata.version("liger-kernel"),
        },
        "frequency": {
            "artifact": frequency_receipt,
            "metadata": {
                name: value
                for name, value in frequency_metadata.items()
                if name != "inv_freq"
            },
            "verification": frequency_verification,
        },
        "parent_adapter": parent_receipt,
        "training_view": {
            "manifest_sha256": sha256_file(view_root / "manifest.json"),
            "manifest": view.manifest,
        },
        "ready_receipt_sha256": sha256_file(ready_path),
        "ready_protocol_sha256": canonical_json_sha256(
            ready["protocol"]
        ),
        "trainable_parameters": trainable_count,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "adapter_sha256": sha256_file(adapter_path),
        "frequency_artifact_sha256": sha256_file(
            incomplete / "custom_inv_freq.pt"
        ),
    }
    atomic_json(incomplete / "result.json", result)
    atomic_json(
        incomplete / "experiment_meta.json",
        {
            "status": "complete",
            "stage": "physical_8k_ruler_family_adaptation",
            "method": args.method,
            "physical_training_length": TRAIN_LENGTH,
            "parent_adapter_sha256": parent_receipt["files"][
                "adapter_model.safetensors"
            ]["sha256"],
            "adapter_sha256": result["adapter_sha256"],
            "frequency_sha256": result["frequency_artifact_sha256"],
            "training_view_manifest_sha256": result["training_view"][
                "manifest_sha256"
            ],
            "seed": int(args.seed),
        },
    )
    recovery = incomplete / "recovery"
    if recovery.exists():
        shutil.rmtree(recovery)
    incomplete.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
