#!/usr/bin/env python3
"""Repair EVQ routing with paired counterfactual supervision at 4K only."""

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
import torch.nn.functional as F

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
    rank_of,
    seed_everything,
    sha256_file,
)

from .prepare_4k_routing_pairs import LENGTH
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


FAMILY_PATTERN = ("routing", "routing", "natural")
PAIR_SET_STATUSES = {
    "OLMO2_4K_COUNTERFACTUAL_ROUTING_SET_PREPARED",
    "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_SET_PREPARED",
}
PAIR_COLLECTION_STATUSES = {
    "OLMO2_4K_COUNTERFACTUAL_ROUTING_DATA_PREPARED",
    "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED",
}
NATURAL_MULTIQUERY_READY_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_READY"


class RoutingPairView:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("status") not in PAIR_SET_STATUSES:
            raise RuntimeError(f"routing set is not prepared: {path}")
        if (
            int(self.manifest["maximum_training_length"]) != LENGTH
            or int(self.manifest["maximum_training_position_id"])
            != LENGTH - 1
            or int(self.manifest["queries_per_sequence"]) < 1
        ):
            raise RuntimeError("routing set violates the 4K contract")
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
            or expected[1:] != (2, LENGTH)
        ):
            raise RuntimeError("routing set shape drift")
        if self.input_ids.dtype != np.uint32:
            raise RuntimeError("routing input dtype drift")
        if self.labels.dtype != np.int32:
            raise RuntimeError("routing label dtype drift")
        for name, expected_digest in self.manifest["files"].items():
            if sha256_file(path / name) != expected_digest:
                raise RuntimeError(f"routing data hash drift: {path / name}")
        self.rows = [
            json.loads(line)
            for line in (path / "rows.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(self.rows) != expected[0]:
            raise RuntimeError("routing metadata row-count drift")
        self.source_starts = np.empty(expected[0], dtype=np.int64)
        self.source_stops = np.empty(expected[0], dtype=np.int64)
        self.answer_starts = np.empty(expected[0], dtype=np.int64)
        self.active_lengths = np.empty(expected[0], dtype=np.int64)
        for index, row in enumerate(self.rows):
            answer_start = int(row["answer_start"])
            active_length = int(row["source_length"])
            if not 0 < answer_start < active_length <= LENGTH:
                raise RuntimeError("routing active-length metadata drift")
            starts = []
            stops = []
            for variant, token_key in (
                (0, "gold_token_ids"),
                (1, "alternate_token_ids"),
            ):
                pattern = [int(value) for value in row[token_key]]
                sequence = self.input_ids[index, variant, :answer_start]
                occurrences = [
                    start
                    for start in range(
                        0, answer_start - len(pattern) + 1
                    )
                    if sequence[
                        start : start + len(pattern)
                    ].tolist()
                    == pattern
                ]
                if len(occurrences) != 1:
                    raise RuntimeError(
                        "routing source value is not uniquely recoverable"
                    )
                starts.append(int(occurrences[0]))
                stops.append(int(occurrences[0] + len(pattern)))
            if starts[0] != starts[1] or stops[0] != stops[1]:
                raise RuntimeError(
                    "counterfactual pair source geometry drift"
                )
            if (
                not np.array_equal(
                    self.input_ids[index, 0, : starts[0]],
                    self.input_ids[index, 1, : starts[1]],
                )
                or not np.array_equal(
                    self.input_ids[index, 0, stops[0] : answer_start],
                    self.input_ids[index, 1, stops[1] : answer_start],
                )
            ):
                raise RuntimeError(
                    "counterfactual prompt differs outside source value"
                )
            if not stops[0] < answer_start:
                raise RuntimeError("source does not precede answer query")
            self.source_starts[index] = starts[0]
            self.source_stops[index] = stops[0]
            self.answer_starts[index] = answer_start
            self.active_lengths[index] = active_length


def source_gap_position_ids(
    *,
    view: RoutingPairView,
    row_indices: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, int]], bytes]:
    """Insert one virtual gap after the source while preserving local steps."""

    row_indices = np.asarray(row_indices, dtype=np.int64)
    target_lengths = np.asarray(target_lengths, dtype=np.int64)
    if row_indices.shape != target_lengths.shape:
        raise RuntimeError("virtual target-length shape drift")
    physical = np.arange(LENGTH - 1, dtype=np.int64)
    pair_positions = np.broadcast_to(
        physical, (len(row_indices), LENGTH - 1)
    ).copy()
    receipts: list[dict[str, int]] = []
    for local_index, (row_index, target_length) in enumerate(
        zip(row_indices.tolist(), target_lengths.tolist())
    ):
        active_length = int(view.active_lengths[row_index])
        active_context = active_length - 1
        source_start = int(view.source_starts[row_index])
        source_stop = int(view.source_stops[row_index])
        answer_start = int(view.answer_starts[row_index])
        if target_length not in {LENGTH, 2 * LENGTH, 4 * LENGTH}:
            raise RuntimeError("unsupported virtual target-length bucket")
        if target_length < active_length:
            raise RuntimeError("virtual target shorter than active sequence")
        extra = int(target_length - LENGTH)
        if extra:
            pair_positions[
                local_index, source_stop:active_context
            ] += extra
        final_active = int(
            pair_positions[local_index, active_context - 1]
        )
        if active_context < LENGTH - 1:
            pair_positions[
                local_index, active_context:
            ] = final_active
        active = pair_positions[local_index, :active_context]
        if (
            int(active[0]) != 0
            or int(active[-1]) != active_context - 1 + extra
            or np.any(np.diff(active) <= 0)
            or np.any(np.diff(active[:source_stop]) != 1)
            or np.any(np.diff(active[source_stop:]) != 1)
        ):
            raise RuntimeError("virtual source-gap position contract drift")
        prediction_index = answer_start - 1
        physical_gap = prediction_index - source_start
        virtual_gap = (
            int(pair_positions[local_index, prediction_index])
            - source_start
        )
        if virtual_gap != physical_gap + extra:
            raise RuntimeError("realized source-query gap drift")
        receipts.append(
            {
                "row": int(row_index),
                "target_length": int(target_length),
                "active_length": active_length,
                "source_start": source_start,
                "source_stop": source_stop,
                "answer_prediction_index": prediction_index,
                "physical_gap": physical_gap,
                "virtual_gap": virtual_gap,
                "maximum_active_position_id": final_active,
            }
        )
    flattened = np.repeat(pair_positions, 2, axis=0)
    payload = flattened.tobytes(order="C")
    return (
        torch.from_numpy(flattened).to("cuda", non_blocking=True),
        receipts,
        payload,
    )


def routing_batch(
    *,
    view: RoutingPairView,
    row_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    host_ids = np.asarray(
        view.input_ids[row_indices], dtype=np.int64
    )
    host_labels = np.asarray(
        view.labels[row_indices], dtype=np.int64
    )
    alternate = host_labels[:, ::-1, :].copy()
    contexts = torch.from_numpy(
        host_ids[:, :, :-1].reshape(-1, LENGTH - 1).copy()
    ).to("cuda", non_blocking=True)
    labels = torch.from_numpy(
        host_labels[:, :, 1:].reshape(-1, LENGTH - 1).copy()
    ).to("cuda", non_blocking=True)
    alternate_labels = torch.from_numpy(
        alternate[:, :, 1:].reshape(-1, LENGTH - 1)
    ).to("cuda", non_blocking=True)
    mask = labels != -100
    if not torch.equal(mask, alternate_labels != -100):
        raise RuntimeError("paired routing answer geometry drift")
    supervised = int(mask.sum())
    if supervised <= 0:
        raise RuntimeError("routing batch has no supervised tokens")
    return contexts, labels, alternate_labels, supervised


def routing_objective(
    *,
    model: Any,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    alternate_labels: torch.Tensor,
    margin: float,
    margin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = labels != -100
    selected_hidden = hidden[mask]
    gold = labels[mask]
    alternate = alternate_labels[mask]
    logits = F.linear(selected_hidden, model.lm_head.weight).float()
    answer_ce = F.cross_entropy(logits, gold)
    source_mask = gold != alternate
    if not torch.any(source_mask):
        raise RuntimeError("paired routing batch has no source-dependent labels")
    source_logits = logits[source_mask]
    source_gold = gold[source_mask]
    source_alternate = alternate[source_mask]
    gold_logits = source_logits.gather(
        1, source_gold[:, None]
    ).squeeze(1)
    alternate_logits = source_logits.gather(
        1, source_alternate[:, None]
    ).squeeze(1)
    preference = gold_logits - alternate_logits
    counterfactual = F.softplus(float(margin) - preference).mean()
    total = answer_ce + float(margin_weight) * counterfactual
    with torch.no_grad():
        metrics = {
            "answer_ce": float(answer_ce),
            "counterfactual_loss": float(counterfactual),
            "preference_mean": float(preference.mean()),
            "preference_positive_fraction": float(
                (preference > 0).float().mean()
            ),
            "token_exact": float(
                logits.argmax(dim=-1).eq(gold).float().mean()
            ),
            "source_token_exact": float(
                source_logits.argmax(dim=-1)
                .eq(source_gold)
                .float()
                .mean()
            ),
            "source_tokens": float(source_gold.numel()),
        }
    return total, metrics


@torch.inference_mode()
def evaluate_routing(
    *,
    model: Any,
    view: RoutingPairView,
    rows: int,
    pair_batch_size: int,
    margin: float,
    target_length: int = LENGTH,
) -> dict[str, Any]:
    n = min(int(rows), len(view.input_ids))
    totals = {
        "tokens": 0,
        "nll_sum": 0.0,
        "exact": 0,
        "ranks": [],
        "source_tokens": 0,
        "source_exact": 0,
        "preference_sum": 0.0,
        "preference_positive": 0,
        "margin_satisfied": 0,
    }
    model.eval()
    for start in range(0, n, int(pair_batch_size)):
        indices = np.arange(start, min(n, start + pair_batch_size))
        contexts, labels, alternate_labels, _ = routing_batch(
            view=view, row_indices=indices
        )
        position_ids = None
        if int(target_length) != LENGTH:
            position_ids, _, _ = source_gap_position_ids(
                view=view,
                row_indices=indices,
                target_lengths=np.full(
                    len(indices), int(target_length), dtype=np.int64
                ),
            )
        mask = labels != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(
                input_ids=contexts,
                position_ids=position_ids,
                use_cache=False,
                return_dict=False,
            )[0]
            logits = F.linear(
                hidden[mask], model.lm_head.weight
            ).float()
        gold = labels[mask]
        alternate = alternate_labels[mask]
        nll = F.cross_entropy(logits, gold, reduction="none")
        ranks = rank_of(logits, gold)
        source_mask = gold != alternate
        if not torch.any(source_mask):
            raise RuntimeError(
                "paired routing calibration has no source-dependent labels"
            )
        source_logits = logits[source_mask]
        source_gold = gold[source_mask]
        source_alternate = alternate[source_mask]
        preference = (
            source_logits.gather(
                1, source_gold[:, None]
            ).squeeze(1)
            - source_logits.gather(
                1, source_alternate[:, None]
            ).squeeze(1)
        )
        count = int(gold.numel())
        totals["tokens"] += count
        totals["nll_sum"] += float(nll.sum())
        totals["exact"] += int(logits.argmax(dim=-1).eq(gold).sum())
        totals["source_tokens"] += int(source_gold.numel())
        totals["source_exact"] += int(
            source_logits.argmax(dim=-1).eq(source_gold).sum()
        )
        totals["ranks"].extend(
            int(value) for value in ranks.detach().cpu().tolist()
        )
        totals["preference_sum"] += float(preference.sum())
        totals["preference_positive"] += int((preference > 0).sum())
        totals["margin_satisfied"] += int(
            (preference >= float(margin)).sum()
        )
        del contexts, labels, alternate_labels, hidden, logits, position_ids
    count = int(totals["tokens"])
    source_count = int(totals["source_tokens"])
    return {
        "pairs": n,
        "answer_tokens": count,
        "mean_answer_nll": totals["nll_sum"] / count,
        "token_exact": totals["exact"] / count,
        "source_answer_tokens": source_count,
        "source_token_exact": totals["source_exact"] / source_count,
        "median_full_vocab_rank": float(
            np.median(totals["ranks"])
        ),
        "mean_counterfactual_preference": (
            totals["preference_sum"] / source_count
        ),
        "preference_positive_fraction": (
            totals["preference_positive"] / source_count
        ),
        "margin_satisfied_fraction": (
            totals["margin_satisfied"] / source_count
        ),
        "position_policy": (
            "contiguous"
            if int(target_length) == LENGTH
            else "single_source_query_block_gap"
        ),
        "virtual_target_length": int(target_length),
    }


def train(
    *,
    model: Any,
    routing_view: RoutingPairView,
    calibration_view: RoutingPairView,
    natural_view_path: Path,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    margin: float,
    margin_weight: float,
    compile_mode: str,
    seed: int,
    log_path: Path,
    virtual_target_length: int = 0,
    virtual_bucket_weights: tuple[int, int, int] = (1, 1, 2),
) -> dict[str, Any]:
    if int(micro_batch_size) % 2:
        raise ValueError("micro-batch-size must be even")
    pair_batch_size = int(micro_batch_size) // 2
    natural_view = load_fixed_view(natural_view_path)
    if natural_view.input_ids.shape[1] != LENGTH:
        raise RuntimeError("natural replay violates the 4K contract")
    natural_rows = torch.from_numpy(
        natural_view.training_rows.copy()
    )
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("counterfactual routing has no parameters")
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=compile_mode,
    )
    natural_loss_module = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 71_001)
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    family_steps = {"routing": 0, "natural": 0}
    supervised_tokens = {"routing": 0, "natural": 0}
    recent_losses: list[float] = []
    position_hash = hashlib.sha256()
    position_bucket_counts = {
        str(LENGTH): 0,
        str(2 * LENGTH): 0,
        str(4 * LENGTH): 0,
    }
    virtual_gap_min: int | None = None
    virtual_gap_max: int | None = None
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(steps) + 1):
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        family_steps[family] += 1
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        raw_losses = []
        metric_rows: list[dict[str, float]] = []
        for _ in range(int(gradient_accumulation_steps)):
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
                    view=routing_view, row_indices=indices
                )
                position_ids = None
                if int(virtual_target_length):
                    weights = torch.tensor(
                        virtual_bucket_weights, dtype=torch.float64
                    )
                    buckets = torch.multinomial(
                        weights,
                        num_samples=pair_batch_size,
                        replacement=True,
                        generator=generator,
                    ).numpy()
                    target_lengths = np.asarray(
                        (LENGTH, 2 * LENGTH, 4 * LENGTH),
                        dtype=np.int64,
                    )[buckets]
                    position_ids, exposures, payload = (
                        source_gap_position_ids(
                            view=routing_view,
                            row_indices=indices,
                            target_lengths=target_lengths,
                        )
                    )
                    position_hash.update(payload)
                    for exposure in exposures:
                        bucket = str(exposure["target_length"])
                        position_bucket_counts[bucket] += 1
                        gap = int(exposure["virtual_gap"])
                        virtual_gap_min = (
                            gap
                            if virtual_gap_min is None
                            else min(virtual_gap_min, gap)
                        )
                        virtual_gap_max = (
                            gap
                            if virtual_gap_max is None
                            else max(virtual_gap_max, gap)
                        )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts, position_ids)
                    raw_loss, metrics = routing_objective(
                        model=model,
                        hidden=hidden,
                        labels=labels,
                        alternate_labels=alternate_labels,
                        margin=float(margin),
                        margin_weight=float(margin_weight),
                    )
                metric_rows.append(metrics)
            else:
                indices = natural_rows[
                    torch.randint(
                        len(natural_rows),
                        (int(micro_batch_size),),
                        generator=generator,
                    )
                ].numpy()
                contexts, labels, supervised = natural_batch(
                    view=natural_view,
                    indices=indices,
                    objective="full",
                )
                alternate_labels = None
                position_ids = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts, None)
                    raw_loss = natural_loss_module(
                        model.lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        labels.reshape(-1),
                    )
                    if hasattr(raw_loss, "loss"):
                        raw_loss = raw_loss.loss
            loss = raw_loss / float(gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite routing loss at step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            supervised_tokens[family] += int(supervised)
            processed_tokens += int(contexts.numel())
            del contexts, labels, hidden, raw_loss, loss
            if alternate_labels is not None:
                del alternate_labels
            if position_ids is not None:
                del position_ids
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(raw_losses))
        recent_losses.append(mean_loss)

        if step == 1 or step % 25 == 0 or step == int(steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            row: dict[str, Any] = {
                "step": step,
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
            if step in {100, 200, int(steps)}:
                row["routing_calibration"] = evaluate_routing(
                    model=model,
                    view=calibration_view,
                    rows=8,
                    pair_batch_size=2,
                    margin=float(margin),
                )
                model.train()
            append_jsonl(log_path, row)
            last_log_time = now
            last_log_tokens = processed_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": int(steps),
        "family_pattern": list(FAMILY_PATTERN),
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
        "counterfactual_margin": float(margin),
        "counterfactual_margin_weight": float(margin_weight),
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_tokens / elapsed,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "natural_loss_backend": "liger_fused_linear_cross_entropy",
        "position_policy": (
            "contiguous"
            if not int(virtual_target_length)
            else "single_source_query_block_gap"
        ),
        "virtual_target_length": int(virtual_target_length),
        "virtual_bucket_weights": list(virtual_bucket_weights),
        "position_bucket_counts": position_bucket_counts,
        "virtual_gap_min": virtual_gap_min,
        "virtual_gap_max": virtual_gap_max,
        "realized_position_stream_sha256": position_hash.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
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
    parser.add_argument("--virtual-target-length", type=int, default=0)
    parser.add_argument(
        "--virtual-bucket-weights",
        type=int,
        nargs=3,
        default=(1, 1, 2),
        metavar=("W4K", "W8K", "W16K"),
    )
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    if float(args.counterfactual_margin) <= 0:
        raise ValueError("counterfactual margin must be positive")
    if float(args.counterfactual_margin_weight) <= 0:
        raise ValueError("counterfactual margin weight must be positive")
    if int(args.virtual_target_length) not in {0, 4 * LENGTH}:
        raise ValueError("virtual target length must be 0 or 16384")
    if (
        len(args.virtual_bucket_weights) != 3
        or any(int(value) < 0 for value in args.virtual_bucket_weights)
        or sum(int(value) for value in args.virtual_bucket_weights) <= 0
    ):
        raise ValueError("invalid virtual bucket weights")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    seed_everything(int(args.seed))
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    routing_root = args.routing_data.resolve()
    routing_manifest = json.loads(
        (routing_root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        routing_manifest.get("status") not in PAIR_COLLECTION_STATUSES
        or int(routing_manifest["hard_maximum_training_length"])
        != LENGTH
        or int(routing_manifest["hard_maximum_training_position_id"])
        != LENGTH - 1
    ):
        raise RuntimeError("routing collection violates the 4K contract")
    experiment_ready = None
    experiment_ready_path = None
    if (
        routing_manifest.get("status")
        == "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED"
    ):
        if args.experiment_ready_receipt is None:
            raise RuntimeError(
                "natural multi-query training requires an experiment READY "
                "receipt"
            )
        experiment_ready_path = args.experiment_ready_receipt.resolve()
        experiment_ready = json.loads(
            experiment_ready_path.read_text(encoding="utf-8")
        )
        if experiment_ready.get("status") != NATURAL_MULTIQUERY_READY_STATUS:
            raise RuntimeError("experiment READY status drift")
        if Path(experiment_ready["run_output"]).resolve() != output:
            raise RuntimeError("experiment READY output path drift")
        if (
            experiment_ready["trainer"]["sha256"]
            != sha256_file(Path(__file__).resolve())
        ):
            raise RuntimeError("experiment READY trainer hash drift")
        if (
            experiment_ready["inputs"]["routing_data"]["manifest"]["sha256"]
            != sha256_file(routing_root / "manifest.json")
        ):
            raise RuntimeError("experiment READY routing-data hash drift")
    if (
        sha256_file(checkpoint / "tokenizer.json")
        != routing_manifest["tokenizer_sha256"]
    ):
        raise RuntimeError("routing collection tokenizer drift")
    for name in ("train", "calibration"):
        entry = routing_manifest["sets"][name]
        if (
            sha256_file(routing_root / name / "manifest.json")
            != entry["manifest_sha256"]
        ):
            raise RuntimeError(
                f"routing collection manifest drift: {name}"
            )
    routing_view = RoutingPairView(routing_root / "train")
    calibration_view = RoutingPairView(
        routing_root / "calibration"
    )

    model = load_model(checkpoint)
    frequency = apply_frequency(model, args.frequency)
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("counterfactual routing forbids a readout")
    parent_adapter = args.parent_adapter.resolve()
    if (
        experiment_ready is not None
        and experiment_ready["inputs"]["parent_adapter"]["sha256"]
        != sha256_file(parent_adapter)
    ):
        raise RuntimeError("experiment READY parent-adapter hash drift")
    parent_metadata = load_adapter(parent_adapter, model, None)
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if parent_metadata.get(name) != expected:
            raise RuntimeError(
                f"parent adapter metadata drift for {name}"
            )
    if experiment_ready is not None:
        expected_protocol = {
            "frequency": args.frequency,
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
            "counterfactual_margin": float(
                args.counterfactual_margin
            ),
            "counterfactual_margin_weight": float(
                args.counterfactual_margin_weight
            ),
            "compile_mode": args.compile_mode,
            "natural_eval_rows": int(args.natural_eval_rows),
            "seed": int(args.seed),
        }
        if experiment_ready["protocol"] != expected_protocol:
            raise RuntimeError("experiment READY protocol drift")
    output.mkdir(parents=True)
    runtime = configure_cuda()
    model.to("cuda")
    initial_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=16,
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    initial_virtual_calibration = None
    if int(args.virtual_target_length):
        initial_virtual_calibration = {
            str(target): evaluate_routing(
                model=model,
                view=calibration_view,
                rows=16,
                pair_batch_size=2,
                margin=float(args.counterfactual_margin),
                target_length=target,
            )
            for target in (2 * LENGTH, 4 * LENGTH)
        }
    training = train(
        model=model,
        routing_view=routing_view,
        calibration_view=calibration_view,
        natural_view_path=(
            args.prepared_data.resolve() / "longalign_paired_L4096"
        ),
        steps=int(args.steps),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        margin=float(args.counterfactual_margin),
        margin_weight=float(args.counterfactual_margin_weight),
        compile_mode=args.compile_mode,
        seed=int(args.seed),
        log_path=output / "train_log.jsonl",
        virtual_target_length=int(args.virtual_target_length),
        virtual_bucket_weights=tuple(
            int(value) for value in args.virtual_bucket_weights
        ),
    )
    final_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=len(calibration_view.input_ids),
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    final_virtual_calibration = None
    if int(args.virtual_target_length):
        final_virtual_calibration = {
            str(target): evaluate_routing(
                model=model,
                view=calibration_view,
                rows=len(calibration_view.input_ids),
                pair_batch_size=2,
                margin=float(args.counterfactual_margin),
                target_length=target,
            )
            for target in (2 * LENGTH, 4 * LENGTH)
        }
    natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=args.background_dir.resolve(),
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
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
        "stage": (
            "counterfactual_source_routing_virtual_gap_16k"
            if int(args.virtual_target_length)
            else "counterfactual_source_routing_4k"
        ),
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "routing_data_sha256": sha256_file(
            routing_root / "manifest.json"
        ),
        "seed": int(args.seed),
        "position_policy": (
            "contiguous"
            if not int(args.virtual_target_length)
            else "single_source_query_block_gap"
        ),
        "virtual_target_length": int(args.virtual_target_length),
    }
    adapter_sha = save_adapter(
        output / "adapter.pt", model, None, adapter_metadata
    )
    receipt = {
        "status": "OLMO2_4K_COUNTERFACTUAL_ROUTING_COMPLETE",
        "metric_boundary": (
            "paired source-content swaps use one source/query block gap "
            "with 4K/8K/16K virtual position buckets while natural replay "
            "uses contiguous 4K positions; strict 8K/16K evaluation uses "
            "real physical sequences"
            if int(args.virtual_target_length)
            else "paired source-content swaps and natural replay use only "
            "positions 0..4095; 8K/16K remain evaluation-only"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": (
            None
            if experiment_ready_path is None
            else sha256_file(experiment_ready_path)
        ),
        "frequency": frequency,
        "parent_adapter": str(parent_adapter),
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "adapter_sha256": adapter_sha,
        "routing_data": {
            "path": str(routing_root),
            "manifest_sha256": sha256_file(
                routing_root / "manifest.json"
            ),
        },
        "runtime": runtime,
        "initial_routing_calibration": initial_calibration,
        "initial_virtual_routing_calibration": (
            initial_virtual_calibration
        ),
        "training": training,
        "final_routing_calibration": final_calibration,
        "final_virtual_routing_calibration": (
            final_virtual_calibration
        ),
        "natural_nll": natural_nll,
        "protocol": {
            "frequency": args.frequency,
            "hard_maximum_training_length": LENGTH,
            "hard_maximum_training_position_id": (
                4 * LENGTH - 2
                if int(args.virtual_target_length)
                else LENGTH - 1
            ),
            "steps": int(args.steps),
            "family_pattern": list(FAMILY_PATTERN),
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "counterfactual_margin": float(
                args.counterfactual_margin
            ),
            "counterfactual_margin_weight": float(
                args.counterfactual_margin_weight
            ),
            "compile_mode": args.compile_mode,
            "position_policy": (
                "contiguous"
                if not int(args.virtual_target_length)
                else "single_source_query_block_gap"
            ),
            "virtual_target_length": int(args.virtual_target_length),
            "virtual_bucket_weights": [
                int(value)
                for value in args.virtual_bucket_weights
            ],
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "initial_routing_calibration": initial_calibration,
                "training": training,
                "final_routing_calibration": final_calibration,
                "natural_nll": natural_nll,
                "adapter_sha256": adapter_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
