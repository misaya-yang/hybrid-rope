#!/usr/bin/env python3
"""Continue a learned Native routing adapter while morphing it to full EVQ."""

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

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    save_adapter,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    seed_everything,
    sha256_file,
)

from .train_4k_counterfactual_routing import (
    FAMILY_PATTERN,
    LENGTH,
    RoutingPairView,
    evaluate_routing,
    routing_batch,
    routing_objective,
)
from .train_4k_progressive_morph import set_morph_frequency
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


READY_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_MORPH_READY"
RESULT_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_MORPH_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--stage-ready-receipt", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin-weight", type=float, default=0.5
    )
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
    parser.add_argument("--seed", type=int, default=20_260_802)
    return parser.parse_args()


def protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frequency_start": "native",
        "frequency_end": "evq",
        "morph_schedule": "linear",
        "steps": int(args.steps),
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "family_pattern": list(FAMILY_PATTERN),
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
        "warmup_steps": int(args.warmup_steps),
        "counterfactual_margin": float(args.counterfactual_margin),
        "counterfactual_margin_weight": float(
            args.counterfactual_margin_weight
        ),
        "compile_mode": str(args.compile_mode),
        "natural_eval_rows": int(args.natural_eval_rows),
        "seed": int(args.seed),
    }


def validate_ready(
    *,
    path: Path,
    args: argparse.Namespace,
    output: Path,
    parent: Path,
    routing_root: Path,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("natural multi-query morph READY status drift")
    if receipt.get("protocol") != protocol_from_args(args):
        raise RuntimeError("natural multi-query morph protocol drift")
    if Path(receipt["run_output"]).resolve() != output:
        raise RuntimeError("natural multi-query morph output drift")
    if (
        receipt["trainer"]["sha256"]
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("natural multi-query morph trainer hash drift")
    if (
        receipt["inputs"]["parent_adapter"]["sha256"]
        != sha256_file(parent)
    ):
        raise RuntimeError("natural multi-query morph parent hash drift")
    if (
        receipt["inputs"]["routing_data"]["manifest_sha256"]
        != sha256_file(routing_root / "manifest.json")
    ):
        raise RuntimeError("natural multi-query morph data hash drift")
    return receipt


def train(
    *,
    model: Any,
    routing_view: RoutingPairView,
    calibration_view: RoutingPairView,
    natural_view_path: Path,
    native_frequency: torch.Tensor,
    evq_frequency: torch.Tensor,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    if int(args.micro_batch_size) % 2:
        raise ValueError("micro-batch-size must be even")
    pair_batch_size = int(args.micro_batch_size) // 2
    natural_view = load_fixed_view(natural_view_path)
    natural_rows = torch.from_numpy(natural_view.training_rows.copy())
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("frequency morph has no trainable parameters")

    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=str(args.compile_mode),
    )
    natural_loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 71_001)
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    family_steps = {"routing": 0, "natural": 0}
    supervised_tokens = {"routing": 0, "natural": 0}
    recent_losses: list[float] = []
    frequency_checkpoints: dict[str, dict[str, Any]] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(args.steps) + 1):
        progress = float(step) / float(args.steps)
        set_morph_frequency(
            model=model,
            native=native_frequency,
            evq=evq_frequency,
            progress=progress,
        )
        if step in {1, max(1, int(args.steps) // 2), int(args.steps)}:
            frequency_checkpoints[str(step)] = {
                "progress": progress,
                "sha256_float32": tensor_sha256(
                    model.model.rotary_emb.inv_freq
                ),
            }
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        family_steps[family] += 1
        lr = cosine_lr(
            step,
            int(args.steps),
            int(args.warmup_steps),
            float(args.learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        raw_losses = []
        metric_rows: list[dict[str, float]] = []
        for _ in range(int(args.gradient_accumulation_steps)):
            if family == "routing":
                indices = torch.randint(
                    len(routing_view.input_ids),
                    (pair_batch_size,),
                    generator=generator,
                ).numpy()
                (
                    contexts,
                    labels,
                    alternate_labels,
                    supervised,
                ) = routing_batch(
                    view=routing_view,
                    row_indices=indices,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts)
                    raw_loss, metrics = routing_objective(
                        model=model,
                        hidden=hidden,
                        labels=labels,
                        alternate_labels=alternate_labels,
                        margin=float(args.counterfactual_margin),
                        margin_weight=float(
                            args.counterfactual_margin_weight
                        ),
                    )
                metric_rows.append(metrics)
            else:
                indices = natural_rows[
                    torch.randint(
                        len(natural_rows),
                        (int(args.micro_batch_size),),
                        generator=generator,
                    )
                ].numpy()
                contexts, labels, supervised = natural_batch(
                    view=natural_view,
                    indices=indices,
                    objective="full",
                )
                alternate_labels = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts)
                    raw_loss = natural_loss_module(
                        model.lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        labels.reshape(-1),
                    )
                    if hasattr(raw_loss, "loss"):
                        raw_loss = raw_loss.loss
            loss = raw_loss / float(args.gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite morph loss at step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            supervised_tokens[family] += int(supervised)
            processed_tokens += int(contexts.numel())
            del contexts, labels, hidden, raw_loss, loss
            if alternate_labels is not None:
                del alternate_labels
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(raw_losses))
        recent_losses.append(mean_loss)

        if step == 1 or step % 25 == 0 or step == int(args.steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            row: dict[str, Any] = {
                "step": step,
                "morph_progress": progress,
                "family": family,
                "loss": mean_loss,
                "mean_loss_last_25": float(
                    np.mean(recent_losses[-25:])
                ),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "processed_input_tokens": processed_tokens,
                "supervised_tokens": dict(supervised_tokens),
                "family_steps": dict(family_steps),
                "elapsed_seconds": now - started,
                "interval_tokens_per_second": (
                    (processed_tokens - last_log_tokens)
                    / max(now - last_log_time, 1e-9)
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
            }
            if metric_rows:
                for name in metric_rows[0]:
                    row[name] = float(
                        np.mean(
                            [metrics[name] for metrics in metric_rows]
                        )
                    )
            if step in {100, 200, int(args.steps)}:
                row["routing_calibration"] = evaluate_routing(
                    model=model,
                    view=calibration_view,
                    rows=8,
                    pair_batch_size=2,
                    margin=float(args.counterfactual_margin),
                )
                model.train()
            append_jsonl(log_path, row)
            last_log_time = now
            last_log_tokens = processed_tokens

    set_morph_frequency(
        model=model,
        native=native_frequency,
        evq=evq_frequency,
        progress=1.0,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": int(args.steps),
        "morph": "linear_native_to_full_evq",
        "frequency_checkpoints": frequency_checkpoints,
        "final_frequency_sha256_float32": tensor_sha256(
            model.model.rotary_emb.inv_freq
        ),
        "family_pattern": list(FAMILY_PATTERN),
        "family_steps": family_steps,
        "supervised_tokens": supervised_tokens,
        "processed_input_tokens": processed_tokens,
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "compile_mode": str(args.compile_mode),
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_tokens / elapsed,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    parent = args.parent_adapter.resolve()
    routing_root = args.routing_data.resolve()
    stage_ready = args.stage_ready_receipt.resolve()
    experiment_ready_path = args.experiment_ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, stage_ready)
    validate_ready(
        path=experiment_ready_path,
        args=args,
        output=output,
        parent=parent,
        routing_root=routing_root,
    )
    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()

    routing_view = RoutingPairView(routing_root / "train")
    calibration_view = RoutingPairView(
        routing_root / "calibration"
    )
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
        raise RuntimeError("frequency morph forbids a readout")
    parent_metadata = load_adapter(parent, model, None)
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": "native",
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if parent_metadata.get(name) != expected:
            raise RuntimeError(f"parent adapter metadata drift for {name}")
    model.to("cuda")

    initial_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=16,
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    training = train(
        model=model,
        routing_view=routing_view,
        calibration_view=calibration_view,
        natural_view_path=(
            args.prepared_data.resolve() / "longalign_paired_L4096"
        ),
        native_frequency=native_frequency,
        evq_frequency=evq_frequency,
        args=args,
        log_path=output / "train_log.jsonl",
    )
    if (
        training["final_frequency_sha256_float32"]
        != frequency["evq_sha256_float32"]
    ):
        raise RuntimeError("frequency morph did not end at full EVQ")
    final_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=len(calibration_view.input_ids),
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=args.background_dir.resolve(),
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
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
        "stage": "linear_native_to_evq_natural_multiquery_morph",
        "parent_adapter_sha256": sha256_file(parent),
        "routing_data_sha256": sha256_file(
            routing_root / "manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha = save_adapter(
        output / "adapter.pt", model, None, metadata
    )
    result = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "4K-only independent natural multi-query routing; the learned "
            "Native adapter is linearly morphed to full EVQ"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "stage_ready_receipt_sha256": sha256_file(stage_ready),
        "experiment_ready_receipt_sha256": sha256_file(
            experiment_ready_path
        ),
        "parent_adapter": str(parent),
        "parent_adapter_sha256": sha256_file(parent),
        "adapter_sha256": adapter_sha,
        "routing_data": {
            "path": str(routing_root),
            "manifest_sha256": sha256_file(
                routing_root / "manifest.json"
            ),
        },
        "runtime": runtime,
        "frequency": frequency,
        "protocol": protocol_from_args(args),
        "initial_routing_calibration": initial_calibration,
        "training": training,
        "final_routing_calibration": final_calibration,
        "natural_nll": natural_nll,
    }
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
