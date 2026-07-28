#!/usr/bin/env python3
"""Train fresh Native or EVQ Q/K-only LoRA on matched general 4K data."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    FixedView,
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


READY_STATUS = "OLMO2_GENERAL_QK_READY_V1"
RESULT_STATUS = "OLMO2_GENERAL_QK_COMPLETE_V1"
FAMILY_PATTERN = ("longalign_full", "longalign_full", "tulu_assistant")
TRAINABLE_SCOPE = "fresh_qk_lora_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
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
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def registered_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frequency": str(args.frequency),
        "adaptation": "qk_answer",
        "trainable_scope": TRAINABLE_SCOPE,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "steps": int(args.steps),
        "family_pattern": list(FAMILY_PATTERN),
        "longalign_objective": "full_token_next_token_ce",
        "tulu_objective": "assistant_content_plus_eos_ce",
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
        "maximum_physical_training_length": 4_096,
        "position_ids": "ordinary_contiguous_0_to_4095",
        "qa_or_ruler_training_rows": 0,
        "natural_eval_rows": int(args.natural_eval_rows),
        "natural_tail_tokens": int(args.natural_tail_tokens),
        "seed": int(args.seed),
    }


def bound_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    experiments = root.parent
    paths = {
        "trainer": Path(__file__).resolve(),
        "stage_a_checkpoint_contract": root / "train_4k_stage_a.py",
        "training_primitives": root / "train_screen.py",
        "lora_conversion": experiments / "olmo2_lora_conversion.py",
        "shared_training_utils": (
            experiments / "small_model_lora_conversion.py"
        ),
        "evq_contract": experiments / "olmo2_1b_evq" / "contract.py",
        "attention_backend": experiments / "olmo2_1b_evq" / "train.py",
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _tensor_bundle_sha256(
    values: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            np.asarray(value.shape, dtype="<i8").tobytes(order="C")
        )
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _view_receipt(view: FixedView) -> dict[str, Any]:
    return {
        "path": str(view.path),
        "manifest_sha256": sha256_file(view.path / "manifest.json"),
        "source_id": str(view.manifest["source"]["id"]),
        "source_revision": str(view.manifest["source"]["revision"]),
        "shape": [int(value) for value in view.input_ids.shape],
        "training_rows": int(len(view.training_rows)),
        "maximum_active_length": int(view.lengths.max()),
    }


def verify_ready(
    *,
    args: argparse.Namespace,
    checkpoint_digest: str,
    longalign: FixedView,
    tulu: FixedView,
) -> dict[str, Any]:
    path = args.experiment_ready_receipt.resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected_inputs = {
        "checkpoint_sha256": checkpoint_digest,
        "longalign": _view_receipt(longalign),
        "tulu": _view_receipt(tulu),
        "background_manifest_sha256": sha256_file(
            args.background_dir.resolve() / "manifest.json"
        ),
    }
    if (
        receipt.get("status") != READY_STATUS
        or receipt.get("protocol") != registered_protocol(args)
        or receipt.get("bound_code_sha256") != bound_code_sha256()
        or receipt.get("inputs") != expected_inputs
        or Path(str(receipt.get("output", ""))).resolve()
        != args.output.resolve()
    ):
        raise RuntimeError("general-QK READY receipt drift")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def train_general_qk(
    *,
    model: Any,
    longalign: FixedView,
    tulu: FixedView,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    named = trainable_named_parameters(model, None)
    names = [name for name, _ in named]
    if (
        len(names) != 64
        or any(
            ".q_proj." not in name and ".k_proj." not in name
            for name in names
        )
    ):
        raise RuntimeError("general-QK trainable scope escaped Q/K LoRA")
    parameters = [parameter for _, parameter in named]
    initial_sha256 = _tensor_bundle_sha256(named)

    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
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
    pools = {
        "longalign_full": torch.from_numpy(
            longalign.training_rows.copy()
        ),
        "tulu_assistant": torch.from_numpy(tulu.training_rows.copy()),
    }
    views = {
        "longalign_full": (longalign, "full"),
        "tulu_assistant": (tulu, "assistant"),
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 70_001)
    selection_digest = hashlib.sha256()
    processed_dense_tokens = 0
    processed_active_tokens = 0
    supervised_tokens = 0
    family_steps = {name: 0 for name in views}
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    recent_losses: list[float] = []

    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(args.steps) + 1):
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        family_steps[family] += 1
        view, objective = views[family]
        pool = pools[family]
        lr = cosine_lr(
            step,
            int(args.steps),
            int(args.warmup_steps),
            float(args.learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for micro in range(int(args.gradient_accumulation_steps)):
            selection = torch.randint(
                len(pool),
                (int(args.micro_batch_size),),
                generator=generator,
            )
            indices = pool[selection].numpy()
            selection_digest.update(family.encode("ascii"))
            selection_digest.update(
                np.asarray([step, micro], dtype="<i8").tobytes()
            )
            selection_digest.update(
                np.asarray(indices, dtype="<i8").tobytes()
            )
            context, labels, local_supervised = natural_batch(
                view=view,
                indices=indices,
                objective=objective,
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
                    args.gradient_accumulation_steps
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite general-QK loss at step {step}"
                )
            loss.backward()
            step_losses.append(float(raw_loss.detach()))
            supervised_tokens += int(local_supervised)
            processed_dense_tokens += int(context.numel())
            processed_active_tokens += int(
                np.asarray(view.lengths[indices] - 1).sum()
            )
            del context, labels, hidden, raw_loss, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(step_losses))
        recent_losses.append(mean_loss)

        if step == 1 or step % 25 == 0 or step == int(args.steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            interval_tokens = processed_dense_tokens - last_log_tokens
            append_jsonl(
                log_path,
                {
                    "step": step,
                    "total_steps": int(args.steps),
                    "family": family,
                    "loss": mean_loss,
                    "mean_loss_last_25": float(
                        np.mean(recent_losses[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "supervised_tokens": supervised_tokens,
                    "processed_dense_tokens": processed_dense_tokens,
                    "processed_active_tokens": processed_active_tokens,
                    "elapsed_seconds": now - started,
                    "interval_dense_tokens_per_second": (
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
            last_log_tokens = processed_dense_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_named = trainable_named_parameters(model, None)
    return {
        "initial_adapter_sha256": initial_sha256,
        "final_adapter_tensor_sha256": _tensor_bundle_sha256(final_named),
        "trainable_parameter_names": names,
        "trainable_parameter_tensors": len(names),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "family_steps": family_steps,
        "selection_sha256": selection_digest.hexdigest(),
        "supervised_tokens": supervised_tokens,
        "processed_dense_tokens": processed_dense_tokens,
        "processed_active_tokens": processed_active_tokens,
        "elapsed_seconds": elapsed,
        "dense_tokens_per_second": processed_dense_tokens / elapsed,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if (
        int(args.steps) <= 0
        or int(args.micro_batch_size) <= 0
        or int(args.gradient_accumulation_steps) <= 0
        or int(args.rank) <= 0
        or float(args.alpha) <= 0
        or float(args.learning_rate) <= 0
        or not 0 < int(args.warmup_steps) < int(args.steps)
    ):
        raise ValueError("invalid general-QK training protocol")

    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    prepared = args.prepared_data.resolve()
    background = args.background_dir.resolve()
    longalign = load_fixed_view(prepared / "longalign_paired_L4096")
    tulu = load_fixed_view(prepared / "tulu3_replay_L4096")
    if (
        tuple(longalign.input_ids.shape)[1] != 4_096
        or tuple(tulu.input_ids.shape)[1] != 4_096
        or int(longalign.lengths.max()) > 4_096
        or int(tulu.lengths.max()) > 4_096
    ):
        raise RuntimeError("general-QK data escaped physical 4K")
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, checkpoint_ready
    )
    ready = verify_ready(
        args=args,
        checkpoint_digest=checkpoint_digest,
        longalign=longalign,
        tulu=tulu,
    )

    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    model = load_model(checkpoint)
    frequency = apply_frequency(model, str(args.frequency))
    readout = install_adaptation(
        model,
        "qk_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("general-QK training must not add a readout")
    model.to("cuda")
    training = train_general_qk(
        model=model,
        longalign=longalign,
        tulu=tulu,
        args=args,
        log_path=output / "train_log.jsonl",
    )
    model.eval()
    natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=background,
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=int(args.natural_tail_tokens),
    )
    metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": str(args.frequency),
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qk_answer",
        "adaptation_description": (
            f"fresh_qk_r{int(args.rank)}_alpha{float(args.alpha):g}"
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": 4_096,
        "longalign_manifest_sha256": sha256_file(
            longalign.path / "manifest.json"
        ),
        "tulu_manifest_sha256": sha256_file(
            tulu.path / "manifest.json"
        ),
        "family_pattern": list(FAMILY_PATTERN),
        "seed": int(args.seed),
    }
    adapter_sha256 = save_adapter(
        output / "adapter.pt",
        model,
        None,
        metadata,
    )
    result = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Matched general-data Q/K-only adaptation; downstream QA and "
            "RULER capability require separate autoregressive evaluation."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bound_code_sha256": bound_code_sha256(),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt": ready,
        "protocol": registered_protocol(args),
        "inputs": {
            "longalign": _view_receipt(longalign),
            "tulu": _view_receipt(tulu),
            "background_manifest_sha256": sha256_file(
                background / "manifest.json"
            ),
        },
        "frequency": frequency,
        "adapter_metadata": metadata,
        "adapter_sha256": adapter_sha256,
        "training": training,
        "natural_nll": natural_nll,
        "runtime": runtime,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    atomic_json(output / "results.json", result)
    print(
        json.dumps(
            {
                "status": RESULT_STATUS,
                "output": str(output),
                "adapter_sha256": adapter_sha256,
                "initial_adapter_sha256": training[
                    "initial_adapter_sha256"
                ],
                "selection_sha256": training["selection_sha256"],
                "elapsed_seconds": training["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
