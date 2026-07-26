#!/usr/bin/env python3
"""Train the 4K-only dense source-binding curriculum after Stage A."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    logits_from_hidden,
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
    rank_of,
    seed_everything,
    sha256_file,
)

from .prepare_4k_binding_data import LENGTH, SLOTS
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


class BindingView:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        self.input_ids = np.load(
            path / "input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            path / "labels.npy", mmap_mode="r", allow_pickle=False
        )
        expected = tuple(int(value) for value in self.manifest["shape"])
        if (
            tuple(self.input_ids.shape) != expected
            or tuple(self.labels.shape) != expected
            or expected[-1] != LENGTH
        ):
            raise RuntimeError(f"binding view shape drift: {path}")
        if self.input_ids.dtype != np.uint32:
            raise RuntimeError("binding input dtype drift")
        if self.labels.dtype != np.int32:
            raise RuntimeError("binding label dtype drift")
        expected_labels = int(np.prod(expected[:-1])) * SLOTS
        if int((self.labels != -100).sum()) != expected_labels:
            raise RuntimeError("binding answer-label count drift")


def binding_batch(
    *,
    view: BindingView,
    row_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if view.input_ids.shape[1] != 2:
        raise RuntimeError("training binding view must have two variants")
    host_ids = np.asarray(
        view.input_ids[row_indices], dtype=np.int64
    ).reshape(-1, LENGTH)
    host_labels = np.asarray(
        view.labels[row_indices], dtype=np.int64
    ).reshape(-1, LENGTH)
    context = torch.from_numpy(host_ids[:, :-1].copy()).to(
        "cuda", non_blocking=True
    )
    labels = torch.from_numpy(host_labels[:, 1:].copy()).to(
        "cuda", non_blocking=True
    )
    supervised = int((labels != -100).sum())
    if supervised != len(row_indices) * 2 * SLOTS:
        raise RuntimeError("binding batch supervision drift")
    return context, labels, supervised


@torch.inference_mode()
def evaluate_binding(
    *,
    model: Any,
    view: BindingView,
    candidate_token_ids: Sequence[int],
    rows: int,
    batch_size: int,
) -> dict[str, Any]:
    if view.input_ids.shape[1] != 3:
        raise RuntimeError("evaluation binding view needs three variants")
    n = min(int(rows), len(view.input_ids))
    candidate_ids = torch.tensor(
        list(candidate_token_ids), device="cuda", dtype=torch.long
    )
    totals = {
        "n": 0,
        "exact": 0,
        "candidate_exact": 0,
        "nll_sum": 0.0,
        "rank": [],
        "deletion_gap_sum": 0.0,
        "swap_follow_positive": 0,
        "swap_follow_sum": 0.0,
    }
    model.eval()
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        source_logits_by_variant = []
        labels_by_variant = []
        for variant_index in range(3):
            host_ids = np.asarray(
                view.input_ids[start:end, variant_index],
                dtype=np.int64,
            )
            host_labels = np.asarray(
                view.labels[start:end, variant_index],
                dtype=np.int64,
            )
            positions = []
            targets = []
            for row in host_labels:
                local = np.flatnonzero(row != -100)
                if len(local) != SLOTS:
                    raise RuntimeError("evaluation slot-count drift")
                positions.append(local - 1)
                targets.append(row[local])
            context = torch.from_numpy(host_ids[:, :-1].copy()).to(
                "cuda", non_blocking=True
            )
            position_tensor = torch.from_numpy(
                np.stack(positions)
            ).to("cuda", non_blocking=True)
            target_tensor = torch.from_numpy(
                np.stack(targets)
            ).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(
                    input_ids=context,
                    use_cache=False,
                    return_dict=False,
                )[0]
                row_ids = torch.arange(
                    hidden.shape[0], device="cuda"
                )[:, None]
                selected = hidden[row_ids, position_tensor]
                logits = logits_from_hidden(
                    model, selected, None
                ).float()
            source_logits_by_variant.append(logits)
            labels_by_variant.append(target_tensor)
            del context, hidden, selected

        sourced, deleted, swapped = source_logits_by_variant
        gold, _, alternate = labels_by_variant
        flat_source = sourced.reshape(-1, sourced.shape[-1])
        flat_deleted = deleted.reshape(-1, deleted.shape[-1])
        flat_swapped = swapped.reshape(-1, swapped.shape[-1])
        flat_gold = gold.reshape(-1)
        flat_alternate = alternate.reshape(-1)
        source_nll = torch.nn.functional.cross_entropy(
            flat_source, flat_gold, reduction="none"
        )
        deleted_nll = torch.nn.functional.cross_entropy(
            flat_deleted, flat_gold, reduction="none"
        )
        ranks = rank_of(flat_source, flat_gold)
        exact = flat_source.argmax(dim=-1).eq(flat_gold)
        candidate_prediction = candidate_ids[
            flat_source[:, candidate_ids].argmax(dim=-1)
        ]
        candidate_exact = candidate_prediction.eq(flat_gold)
        source_gold = flat_source.gather(
            -1, flat_gold[:, None]
        ).squeeze(-1)
        source_alternate = flat_source.gather(
            -1, flat_alternate[:, None]
        ).squeeze(-1)
        swapped_gold = flat_swapped.gather(
            -1, flat_gold[:, None]
        ).squeeze(-1)
        swapped_alternate = flat_swapped.gather(
            -1, flat_alternate[:, None]
        ).squeeze(-1)
        swap_follow = (
            source_gold
            - source_alternate
            - swapped_gold
            + swapped_alternate
        )
        local_n = int(flat_gold.numel())
        totals["n"] += local_n
        totals["exact"] += int(exact.sum())
        totals["candidate_exact"] += int(candidate_exact.sum())
        totals["nll_sum"] += float(source_nll.sum())
        totals["rank"].extend(
            int(value) for value in ranks.detach().cpu().tolist()
        )
        totals["deletion_gap_sum"] += float(
            (deleted_nll - source_nll).sum()
        )
        totals["swap_follow_positive"] += int(
            (swap_follow > 0).sum()
        )
        totals["swap_follow_sum"] += float(swap_follow.sum())
        del source_logits_by_variant, labels_by_variant

    count = int(totals["n"])
    return {
        "answer_slots": count,
        "full_vocab_exact": totals["exact"] / count,
        "candidate_exact": totals["candidate_exact"] / count,
        "mean_nll": totals["nll_sum"] / count,
        "median_rank": float(np.median(totals["rank"])),
        "mean_source_deletion_nll_gap": (
            totals["deletion_gap_sum"] / count
        ),
        "swap_follow_positive_fraction": (
            totals["swap_follow_positive"] / count
        ),
        "mean_swap_follow_score": totals["swap_follow_sum"] / count,
    }


def train_stage(
    *,
    model: Any,
    natural_view_path: Path,
    binding_view: BindingView,
    calibration_view: BindingView,
    candidate_token_ids: Sequence[int],
    stage: str,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    compile_mode: str,
    natural_objective: str,
    natural_replay_every: int,
    log_path: Path,
) -> dict[str, Any]:
    natural = load_fixed_view(natural_view_path)
    if natural.input_ids.shape[1] != LENGTH:
        raise RuntimeError("Stage B natural replay must be 4K")
    if int(micro_batch_size) % 2 != 0:
        raise ValueError("binding micro-batch must be even")
    if int(natural_replay_every) < 1:
        raise ValueError("natural replay interval must be positive")

    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=compile_mode,
    )
    loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 50_001)
    natural_rows = torch.from_numpy(natural.training_rows.copy())
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    supervised_tokens = {"natural": 0, "binding": 0}
    family_steps = {"natural": 0, "binding": 0}
    recent: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(steps) + 1):
        family = (
            "binding"
            if (
                stage == "b1"
                or (
                    int(natural_replay_every) > 1
                    and step % int(natural_replay_every) != 0
                )
            )
            else "natural"
        )
        family_steps[family] += 1
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        raw_losses = []
        for _ in range(int(gradient_accumulation_steps)):
            if family == "binding":
                pair_count = int(micro_batch_size) // 2
                indices = torch.randint(
                    len(binding_view.input_ids),
                    (pair_count,),
                    generator=generator,
                ).numpy()
                context, labels, supervised = binding_batch(
                    view=binding_view, row_indices=indices
                )
            else:
                indices = natural_rows[
                    torch.randint(
                        len(natural_rows),
                        (int(micro_batch_size),),
                        generator=generator,
                    )
                ].numpy()
                context, labels, supervised = natural_batch(
                    view=natural,
                    indices=indices,
                    objective=natural_objective,
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
                    gradient_accumulation_steps
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite Stage-B loss at step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            supervised_tokens[family] += int(supervised)
            processed_tokens += int(context.numel())
            del context, labels, hidden, raw_loss, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(raw_losses))
        recent.append(mean_loss)

        if step == 1 or step % 25 == 0 or step == int(steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            row = {
                "step": step,
                "stage": stage,
                "family": family,
                "loss": mean_loss,
                "mean_loss_last_25": float(np.mean(recent[-25:])),
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
            if step in {100, 200, int(steps)}:
                row["binding_calibration"] = evaluate_binding(
                    model=model,
                    view=calibration_view,
                    candidate_token_ids=candidate_token_ids,
                    rows=16,
                    batch_size=4,
                )
                model.train()
            append_jsonl(log_path, row)
            last_log_time = now
            last_log_tokens = processed_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "stage": stage,
        "steps": int(steps),
        "family_steps": family_steps,
        "supervised_tokens": supervised_tokens,
        "processed_input_tokens": processed_tokens,
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": int(
            micro_batch_size * gradient_accumulation_steps
        ),
        "learning_rate": float(learning_rate),
        "warmup_steps": int(warmup_steps),
        "compile_mode": compile_mode,
        "natural_replay_objective": natural_objective,
        "natural_replay_every": int(natural_replay_every),
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_tokens / elapsed,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "loss_backend": "liger_fused_linear_cross_entropy",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--binding-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("b1", "b2"), required=True)
    parser.add_argument("--steps", type=int)
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
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument(
        "--natural-replay-every",
        type=int,
        default=2,
        help=(
            "Use one natural-replay optimizer step at this interval during "
            "B2; the remaining steps use paired binding supervision. Set 1 "
            "for a clean Tulu-only instruction-recovery control."
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    steps = int(
        args.steps
        if args.steps is not None
        else (100 if args.stage == "b1" else 300)
    )
    if steps <= 0:
        raise ValueError("steps must be positive")

    seed_everything(int(args.seed))
    runtime = configure_cuda()
    checkpoint = args.checkpoint.resolve()
    parent_adapter = args.parent_adapter.resolve()
    prepared = args.prepared_data.resolve()
    binding_root = args.binding_data.resolve()
    background = args.background_dir.resolve()
    ready_receipt = args.ready_receipt.resolve()
    collection = json.loads(
        (binding_root / "collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if int(collection["hard_maximum_training_length"]) != LENGTH:
        raise RuntimeError("binding collection violates 4K contract")

    train_name = "train_anchor" if args.stage == "b1" else "train_broad"
    calibration_name = (
        "calibration_anchor"
        if args.stage == "b1"
        else "validation_ood"
    )
    binding_view = BindingView(binding_root / train_name)
    calibration_view = BindingView(binding_root / calibration_name)
    candidate_ids = (
        collection["train_value_token_ids"][:128]
        if args.stage == "b1"
        else collection["eval_value_token_ids"]
    )

    model = load_model(checkpoint)
    frequency = apply_frequency(model, "evq")
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("Stage B must not add a readout")
    parent_metadata = load_adapter(
        parent_adapter, model, None
    )
    if parent_metadata.get("frequency") != "evq":
        raise RuntimeError("parent adapter is not an EVQ adapter")
    if int(parent_metadata.get("training_sequence_length", -1)) != LENGTH:
        raise RuntimeError("parent adapter is not from 4K Stage A/B")
    model.to("cuda")

    training = train_stage(
        model=model,
        natural_view_path=prepared / "tulu3_replay_L4096",
        binding_view=binding_view,
        calibration_view=calibration_view,
        candidate_token_ids=candidate_ids,
        stage=args.stage,
        steps=steps,
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        seed=int(args.seed),
        compile_mode=args.compile_mode,
        natural_objective="assistant",
        natural_replay_every=int(args.natural_replay_every),
        log_path=output / "train_log.jsonl",
    )
    binding_validation = evaluate_binding(
        model=model,
        view=calibration_view,
        candidate_token_ids=candidate_ids,
        rows=len(calibration_view.input_ids),
        batch_size=4,
    )
    binding_final_test = None
    if args.stage == "b2":
        final_view = BindingView(binding_root / "final_test")
        binding_final_test = evaluate_binding(
            model=model,
            view=final_view,
            candidate_token_ids=collection["eval_value_token_ids"],
            rows=len(final_view.input_ids),
            batch_size=4,
        )
    natural_nll = (
        evaluate_natural_nll(
            model=model,
            background_dir=background,
            lengths=(4_096, 8_192, 16_384),
            rows=int(args.natural_eval_rows),
            tail_tokens=1_024,
        )
        if args.stage == "b2"
        else {}
    )
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": "evq",
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "adaptation_description": (
            f"qkvo_r{int(args.rank)}_alpha{float(args.alpha):g}"
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
        "stage": args.stage,
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "parent_adapter_metadata": parent_metadata,
        "binding_collection_sha256": sha256_file(
            binding_root / "collection_manifest.json"
        ),
        "seed": int(args.seed),
    }
    adapter_sha = save_adapter(
        output / "adapter.pt", model, None, adapter_metadata
    )
    receipt = {
        "status": f"OLMO2_4K_STAGE_{args.stage.upper()}_COMPLETE",
        "metric_boundary": (
            "4K-only dense paired source-binding curriculum; "
            "longer contexts remain evaluation-only"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "parent_adapter": str(parent_adapter),
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "adapter_sha256": adapter_sha,
        "runtime": runtime,
        "frequency": frequency,
        "training": training,
        "binding_validation": binding_validation,
        "binding_final_test": binding_final_test,
        "natural_nll": natural_nll,
        "protocol": {
            "hard_maximum_training_length": LENGTH,
            "stage": args.stage,
            "train_binding_set": train_name,
            "calibration_set": calibration_name,
            "steps": steps,
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "compile_mode": args.compile_mode,
            "natural_replay_view": "tulu3_replay_L4096",
            "natural_replay_objective": "assistant",
            "natural_replay_every": int(args.natural_replay_every),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "training": training,
                "binding_validation": binding_validation,
                "binding_final_test": binding_final_test,
                "natural_nll": natural_nll,
                "adapter_sha256": adapter_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
