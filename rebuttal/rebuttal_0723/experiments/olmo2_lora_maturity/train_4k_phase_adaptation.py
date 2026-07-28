#!/usr/bin/env python3
"""Continue a Stage-A OLMo adapter with answer-plus-EOS phase supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from .phase_adaptation import (
    LENGTH,
    READY_STATUS,
    PhaseAdaptationView,
    deterministic_offset_batch,
    offset_bucket,
    phase_batch,
    position_ids_for_offsets,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


RESULT_STATUS = "OLMO2_4K_PHASE_ADAPTATION_COMPLETE_V1"
FAMILY_PATTERN = ("phase", "phase", "natural")
SHARED_FAMILY_PATTERN = (
    "phase_2wiki",
    "phase_ruler",
    "phase_2wiki",
    "phase_ruler",
    "natural",
)
TRAINABLE_SCOPE = "continue_parent_qk_lora_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--ruler-training-view", type=Path)
    parser.add_argument("--natural-view", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument(
        "--parent-adaptation",
        choices=("qkvo_answer", "qk_answer"),
        default="qkvo_answer",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def family_pattern(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        SHARED_FAMILY_PATTERN
        if args.ruler_training_view is not None
        else FAMILY_PATTERN
    )


def registered_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frequency": str(args.frequency),
        "parent_adaptation": str(args.parent_adaptation),
        "trainable_scope": TRAINABLE_SCOPE,
        "frozen_parent_projections": (
            ["v_proj", "o_proj"]
            if args.parent_adaptation == "qkvo_answer"
            else []
        ),
        "steps": int(args.steps),
        "family_pattern": list(family_pattern(args)),
        "task_views": (
            ["2wiki", "ruler13"]
            if args.ruler_training_view is not None
            else ["single"]
        ),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size * args.gradient_accumulation_steps
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "optimizer": "fused_adamw",
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "precision": "bf16_autocast",
        "compile_mode": str(args.compile_mode),
        "physical_training_length": LENGTH,
        "phase_target_maximum": 4 * LENGTH,
        "phase_micro_batch_ratio": {
            "contiguous_4k": 1,
            "phase_to_8k": 1,
            "phase_to_16k": 2,
        },
        "supervision": "complete_answer_plus_immediate_eos",
        "validation_rows": int(args.validation_rows),
        "seed": int(args.seed),
    }


def bound_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    experiments = root.parent
    paths = {
        "trainer": Path(__file__).resolve(),
        "phase_contract": root / "phase_adaptation.py",
        "training_primitives": root / "train_screen.py",
        "checkpoint_contract": root / "train_4k_stage_a.py",
        "lora_conversion": experiments / "olmo2_lora_conversion.py",
        "adapter_loader": experiments / "olmo2_lora_ood_factorial.py",
        "shared_training_utils": (
            experiments / "small_model_lora_conversion.py"
        ),
        "evq_contract": experiments / "olmo2_1b_evq" / "contract.py",
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _projection_parameters(
    model: Any,
    projections: tuple[str, ...],
) -> list[tuple[str, torch.nn.Parameter]]:
    selected = []
    for name, parameter in model.named_parameters():
        if (
            name.endswith((".a", ".b"))
            and any(
                f".{projection}." in name
                for projection in projections
            )
        ):
            selected.append((f"model.{name}", parameter))
    return selected


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


def _freeze_parent_vo_and_validate_qk(
    model: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    frozen = _projection_parameters(model, ("v_proj", "o_proj"))
    if len(frozen) != 64:
        raise RuntimeError(
            f"expected 64 parent V/O LoRA tensors, got {len(frozen)}"
        )
    for _, parameter in frozen:
        parameter.requires_grad_(False)
    trainable = trainable_named_parameters(model, None)
    names = [name for name, _ in trainable]
    if (
        len(names) != 64
        or any(
            ".q_proj." not in name and ".k_proj." not in name
            for name in names
        )
    ):
        raise RuntimeError("phase-adaptation trainable scope escaped Q/K")
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in frozen
    }
    return snapshot, {
        "scope": TRAINABLE_SCOPE,
        "parameter_names": names,
        "parameter_tensors": len(names),
        "parameters": int(
            sum(parameter.numel() for _, parameter in trainable)
        ),
        "frozen_projection_parameter_tensors": len(frozen),
        "frozen_projection_parameters": int(
            sum(parameter.numel() for _, parameter in frozen)
        ),
        "frozen_projection_sha256_before": _tensor_bundle_sha256(
            [(name, value) for name, value in snapshot.items()]
        ),
    }


def _validate_qk_parent(
    model: Any,
) -> dict[str, Any]:
    trainable = trainable_named_parameters(model, None)
    names = [name for name, _ in trainable]
    if (
        len(names) != 64
        or any(
            ".q_proj." not in name and ".k_proj." not in name
            for name in names
        )
    ):
        raise RuntimeError("Q/K-parent trainable scope escaped Q/K")
    return {
        "scope": TRAINABLE_SCOPE,
        "parameter_names": names,
        "parameter_tensors": len(names),
        "parameters": int(
            sum(parameter.numel() for _, parameter in trainable)
        ),
        "frozen_projection_parameter_tensors": 0,
        "frozen_projection_parameters": 0,
    }


def _assert_vo_unchanged_and_enable_full_save(
    model: Any,
    snapshot: dict[str, torch.Tensor],
) -> str:
    frozen = _projection_parameters(model, ("v_proj", "o_proj"))
    observed = {name: parameter for name, parameter in frozen}
    if set(observed) != set(snapshot):
        raise RuntimeError("frozen V/O tensor identity drift")
    for name, parameter in observed.items():
        if not torch.equal(parameter.detach().cpu(), snapshot[name]):
            raise RuntimeError(f"frozen parent tensor changed: {name}")
    digest = _tensor_bundle_sha256(frozen)
    for _, parameter in frozen:
        parameter.requires_grad_(True)
    if len(trainable_named_parameters(model, None)) != 128:
        raise RuntimeError("full QKVO adapter save scope is incomplete")
    return digest


def verify_ready(
    *,
    args: argparse.Namespace,
    checkpoint_digest: str,
    view: PhaseAdaptationView,
    ruler_view: PhaseAdaptationView | None,
) -> dict[str, Any]:
    path = args.ready_receipt.resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != READY_STATUS
        or receipt.get("protocol") != registered_protocol(args)
        or receipt.get("bound_code_sha256") != bound_code_sha256()
        or Path(receipt["output"]).resolve() != args.output.resolve()
        or receipt["inputs"]["checkpoint"]["composite_sha256"]
        != checkpoint_digest
        or receipt["inputs"]["parent_adapter"]["sha256"]
        != sha256_file(args.parent_adapter.resolve())
        or receipt["inputs"]["training_view"]["manifest_sha256"]
        != sha256_file(view.root / "manifest.json")
    ):
        raise RuntimeError("phase-adaptation READY receipt drift")
    ruler_receipt = receipt["inputs"].get("ruler_training_view")
    if (ruler_view is None) != (ruler_receipt is None):
        raise RuntimeError("shared-task READY view presence drift")
    if (
        ruler_view is not None
        and ruler_receipt["manifest_sha256"]
        != sha256_file(ruler_view.root / "manifest.json")
    ):
        raise RuntimeError("shared-task RULER view drift")
    natural = args.natural_view.resolve()
    for name, entry in receipt["inputs"]["natural_view"][
        "files"
    ].items():
        if sha256_file(natural / name) != entry["sha256"]:
            raise RuntimeError(f"natural replay drift: {name}")
    return receipt


@torch.inference_mode()
def evaluate_phase_view(
    *,
    model: Any,
    backbone: Any,
    loss_module: Any,
    view: PhaseAdaptationView,
    rows: np.ndarray,
    batch_size: int,
    offset: int,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    exact_tokens = 0
    for start in range(0, len(rows), int(batch_size)):
        indices = rows[start : start + int(batch_size)]
        contexts, labels, supervised = phase_batch(
            view=view, indices=indices
        )
        offsets = np.full(len(indices), int(offset), dtype=np.int64)
        position_ids, _, _ = position_ids_for_offsets(
            view=view,
            indices=indices,
            offsets=offsets,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(contexts, position_ids)
            value = loss_module(
                model.lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )
            value = value.loss if hasattr(value, "loss") else value
        if not torch.isfinite(value):
            raise RuntimeError("non-finite phase validation NLL")
        mask = labels != -100
        logits = model.lm_head(hidden[mask]).float()
        exact_tokens += int(logits.argmax(dim=-1).eq(labels[mask]).sum())
        total_nll += float(value) * int(supervised)
        total_tokens += int(supervised)
        del contexts, labels, position_ids, hidden, value, logits
    mean_nll = total_nll / total_tokens
    return {
        "rows": int(len(rows)),
        "query_offset": int(offset),
        "supervised_tokens": int(total_tokens),
        "mean_nll": float(mean_nll),
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
        "teacher_forced_token_exact": exact_tokens / total_tokens,
    }


def main() -> None:
    args = parse_args()
    if (
        int(args.steps) <= 0
        or int(args.micro_batch_size) != 4
        or int(args.gradient_accumulation_steps) != 2
        or int(args.warmup_steps) < 0
    ):
        raise ValueError("locked phase-adaptation protocol drift")
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)

    checkpoint = args.checkpoint.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    view = PhaseAdaptationView(args.training_view.resolve())
    ruler_view = (
        PhaseAdaptationView(args.ruler_training_view.resolve())
        if args.ruler_training_view is not None
        else None
    )
    if (
        ruler_view is not None
        and "RULER13" not in ruler_view.manifest["status"]
    ):
        raise RuntimeError("shared-task secondary view is not RULER13")
    natural_view = load_fixed_view(args.natural_view.resolve())
    if natural_view.input_ids.shape[1] != LENGTH:
        raise RuntimeError("natural replay violates physical-4K contract")
    ready = verify_ready(
        args=args,
        checkpoint_digest=checkpoint_digest,
        view=view,
        ruler_view=ruler_view,
    )
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    model = load_model(checkpoint)
    frequency = apply_frequency(model, args.frequency)
    readout = install_adaptation(
        model,
        str(args.parent_adaptation),
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("phase adaptation forbids a readout")
    parent_path = args.parent_adapter.resolve()
    parent_metadata = load_adapter(parent_path, model, None)
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": str(args.parent_adaptation),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if parent_metadata.get(name) != expected:
            raise RuntimeError(
                f"parent metadata drift for {name}: "
                f"{parent_metadata.get(name)!r} != {expected!r}"
            )
    frozen_vo: dict[str, torch.Tensor] | None
    if args.parent_adaptation == "qkvo_answer":
        frozen_vo, trainable_scope = _freeze_parent_vo_and_validate_qk(
            model
        )
    else:
        frozen_vo = None
        trainable_scope = _validate_qk_parent(model)

    incomplete.mkdir(parents=True)
    model.to("cuda")
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("phase adaptation has no trainable parameters")
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=args.compile_mode,
    )
    loss_module = fused_loss_module()
    validation_offsets = (0, LENGTH, 3 * LENGTH + 1)

    def validate_one(active_view: PhaseAdaptationView) -> dict[str, Any]:
        rows = active_view.validation_rows[
            : min(
                int(args.validation_rows),
                len(active_view.validation_rows),
            )
        ]
        return {
            str(offset): evaluate_phase_view(
                model=model,
                backbone=backbone,
                loss_module=loss_module,
                view=active_view,
                rows=rows,
                batch_size=int(args.micro_batch_size),
                offset=offset,
            )
            for offset in validation_offsets
        }

    validation_before = (
        {
            "2wiki": validate_one(view),
            "ruler13": validate_one(ruler_view),
        }
        if ruler_view is not None
        else validate_one(view)
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 83_001)
    natural_rows = torch.from_numpy(
        natural_view.training_rows.copy()
    )
    position_hash = hashlib.sha256()
    exposure_hash = hashlib.sha256()
    selection_hash = hashlib.sha256()
    bucket_counts = {
        "contiguous_4k": 0,
        "phase_to_8k": 0,
        "phase_to_16k": 0,
    }
    task_bucket_counts = {
        "2wiki": dict.fromkeys(bucket_counts, 0),
        "ruler13": dict.fromkeys(bucket_counts, 0),
    }
    pattern = family_pattern(args)
    family_steps = {name: 0 for name in pattern}
    family_supervised = {name: 0 for name in pattern}
    processed_tokens = 0
    recent: list[float] = []
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(args.steps) + 1):
        family = pattern[(step - 1) % len(pattern)]
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
        raw_losses: list[float] = []
        for accumulation_index in range(
            int(args.gradient_accumulation_steps)
        ):
            if family.startswith("phase"):
                active_view = (
                    ruler_view
                    if family == "phase_ruler"
                    else view
                )
                if active_view is None:
                    raise RuntimeError("phase family has no registered view")
                local = torch.randint(
                    len(active_view.training_rows),
                    (int(args.micro_batch_size),),
                    generator=generator,
                ).numpy()
                indices = active_view.training_rows[local]
                contexts, labels, supervised = phase_batch(
                    view=active_view, indices=indices
                )
                selection_hash.update(family.encode("ascii"))
                selection_hash.update(b"\0")
                selection_hash.update(
                    np.asarray(
                        [step, accumulation_index],
                        dtype="<i8",
                    ).tobytes(order="C")
                )
                selection_hash.update(
                    np.asarray(indices, dtype="<i8").tobytes(order="C")
                )
                offsets = deterministic_offset_batch(
                    seed=int(args.seed),
                    optimizer_step=step,
                    accumulation_index=accumulation_index,
                    micro_batch_size=int(args.micro_batch_size),
                )
                position_ids, receipts, payload = (
                    position_ids_for_offsets(
                        view=active_view,
                        indices=indices,
                        offsets=offsets,
                    )
                )
                position_hash.update(payload)
                for slot, receipt in enumerate(receipts):
                    bucket = offset_bucket(int(offsets[slot]))
                    bucket_counts[bucket] += 1
                    task_name = (
                        "ruler13"
                        if family == "phase_ruler"
                        else "2wiki"
                    )
                    task_bucket_counts[task_name][bucket] += 1
                    exposure_hash.update(
                        np.asarray(
                            [
                                step,
                                accumulation_index,
                                slot,
                                int(indices[slot]),
                                int(offsets[slot]),
                                int(receipt["query_start"]),
                                int(
                                    receipt[
                                        "virtual_answer_prediction_position"
                                    ]
                                ),
                            ],
                            dtype="<i8",
                        ).tobytes(order="C")
                    )
            else:
                local = torch.randint(
                    len(natural_rows),
                    (int(args.micro_batch_size),),
                    generator=generator,
                )
                indices = natural_rows[local].numpy()
                selection_hash.update(family.encode("ascii"))
                selection_hash.update(b"\0")
                selection_hash.update(
                    np.asarray(
                        [step, accumulation_index],
                        dtype="<i8",
                    ).tobytes(order="C")
                )
                selection_hash.update(
                    np.asarray(indices, dtype="<i8").tobytes(order="C")
                )
                contexts, labels, supervised = natural_batch(
                    view=natural_view,
                    indices=indices,
                    objective="full",
                )
                position_ids = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(contexts, position_ids)
                value = loss_module(
                    model.lm_head.weight,
                    hidden.reshape(-1, hidden.shape[-1]),
                    labels.reshape(-1),
                )
                value = value.loss if hasattr(value, "loss") else value
                loss = value / float(
                    args.gradient_accumulation_steps
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            raw_losses.append(float(value.detach()))
            family_supervised[family] += int(supervised)
            processed_tokens += int(contexts.numel())
            del contexts, labels, hidden, value, loss
            if position_ids is not None:
                del position_ids
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(raw_losses))
        recent.append(mean_loss)
        if step == 1 or step % 25 == 0 or step == int(args.steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            append_jsonl(
                incomplete / "train_log.jsonl",
                {
                    "step": step,
                    "family": family,
                    "loss": mean_loss,
                    "mean_loss_last_25": float(
                        np.mean(recent[-25:])
                    ),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "processed_input_tokens": processed_tokens,
                    "family_steps": dict(family_steps),
                    "family_supervised_tokens": dict(
                        family_supervised
                    ),
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
    validation_after = (
        {
            "2wiki": validate_one(view),
            "ruler13": validate_one(ruler_view),
        }
        if ruler_view is not None
        else validate_one(view)
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": str(args.parent_adaptation),
        "adaptation_description": (
            f"{str(args.parent_adaptation)}_parent_qk_only_continuation_"
            f"r{int(args.rank)}_"
            f"alpha{float(args.alpha):g}"
        ),
        "continuation_trainable_scope": TRAINABLE_SCOPE,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
        "stage": "task_realized_phase_answer_eos_continuation",
        "parent_adapter_sha256": sha256_file(parent_path),
        "training_view_manifest_sha256": sha256_file(
            view.root / "manifest.json"
        ),
        "ruler_training_view_manifest_sha256": (
            sha256_file(ruler_view.root / "manifest.json")
            if ruler_view is not None
            else None
        ),
        "ready_receipt_sha256": sha256_file(
            args.ready_receipt.resolve()
        ),
        "seed": int(args.seed),
    }
    frozen_vo_sha256_after: str | None = None
    if frozen_vo is not None:
        frozen_vo_sha256_after = (
            _assert_vo_unchanged_and_enable_full_save(
                model,
                frozen_vo,
            )
        )
        if (
            frozen_vo_sha256_after
            != trainable_scope["frozen_projection_sha256_before"]
        ):
            raise RuntimeError("frozen parent V/O digest changed")
    adapter_sha = save_adapter(
        incomplete / "adapter.pt", model, None, metadata
    )
    result = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Physical sequences are <=4K. Position IDs shift only the "
            "semantic question/answer block; capability requires separate "
            "autoregressive evaluation."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "parent_adapter": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "metadata": parent_metadata,
        },
        "training_view": {
            "path": str(view.root),
            "manifest_sha256": sha256_file(
                view.root / "manifest.json"
            ),
            "manifest": view.manifest,
        },
        "ruler_training_view": (
            {
                "path": str(ruler_view.root),
                "manifest_sha256": sha256_file(
                    ruler_view.root / "manifest.json"
                ),
                "manifest": ruler_view.manifest,
            }
            if ruler_view is not None
            else None
        ),
        "ready_receipt": {
            "path": str(args.ready_receipt.resolve()),
            "sha256": sha256_file(args.ready_receipt.resolve()),
            "status": ready["status"],
        },
        "protocol": registered_protocol(args),
        "trainable_scope": {
            **trainable_scope,
            "frozen_projection_sha256_after": (
                frozen_vo_sha256_after
            ),
            "frozen_projection_bitwise_unchanged": (
                frozen_vo is not None
            ),
        },
        "bound_code_sha256": bound_code_sha256(),
        "frequency": frequency,
        "adapter_sha256": adapter_sha,
        "adapter_metadata": metadata,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "training": {
            "steps": int(args.steps),
            "family_steps": family_steps,
            "family_supervised_tokens": family_supervised,
            "processed_input_tokens": processed_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": processed_tokens / elapsed,
            "position_stream_sha256": position_hash.hexdigest(),
            "exposure_stream_sha256": exposure_hash.hexdigest(),
            "selection_stream_sha256": selection_hash.hexdigest(),
            "position_bucket_counts": bucket_counts,
            "task_position_bucket_counts": task_bucket_counts,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
        "runtime": {
            **runtime,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "compile_cache": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
            "allocator": allocator,
        },
    }
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
