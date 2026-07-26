#!/usr/bin/env python3
"""Fast OLMo-2 maturity screen for native RoPE versus EVQ plus QKVO LoRA."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    TAU,
    assert_frequency_contract,
    endpoint_geo_inv_freq,
    patch_endpoint_evq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    causal_margin_losses,
    forward_hidden,
    install_adaptation,
    load_model,
    logits_from_hidden,
    quick_source_metrics,
    save_adapter,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    FormalProbeSet,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    aggregate_rows,
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    rank_of,
    scalar_alpha_interval,
    score_logits,
    seed_everything,
    sha256_file,
)

from .causal_data import load_probe_set
from .prepare_data import sha256_file as sha256_large_file


MIXTURE_PATTERN = (
    "longalign_full",
    "causal_margin",
    "longalign_full",
    "causal_margin",
    "tulu_assistant",
)


@dataclass
class FixedView:
    path: Path
    input_ids: np.ndarray
    assistant_mask: np.ndarray
    lengths: np.ndarray
    split: np.ndarray
    manifest: dict[str, Any]

    @property
    def training_rows(self) -> np.ndarray:
        return np.flatnonzero(self.split == 0)


def load_fixed_view(path: Path) -> FixedView:
    manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    view = FixedView(
        path=path,
        input_ids=np.load(
            path / "input_ids.npy", mmap_mode="r", allow_pickle=False
        ),
        assistant_mask=np.load(
            path / "assistant_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        lengths=np.load(
            path / "lengths.npy", mmap_mode="r", allow_pickle=False
        ),
        split=np.load(
            path / "split.npy", mmap_mode="r", allow_pickle=False
        ),
        manifest=manifest,
    )
    expected = tuple(int(value) for value in manifest["shape"])
    if (
        tuple(view.input_ids.shape) != expected
        or tuple(view.assistant_mask.shape) != expected
        or view.lengths.shape != (expected[0],)
        or view.split.shape != (expected[0],)
    ):
        raise RuntimeError(f"fixed-view shape drift: {path}")
    if len(view.training_rows) == 0:
        raise RuntimeError(f"fixed view has no training rows: {path}")
    return view


def apply_frequency(model: Any, frequency: str) -> dict[str, Any]:
    receipt = assert_frequency_contract()
    native = model.model.rotary_emb.inv_freq.detach().cpu().float()
    expected_native = endpoint_geo_inv_freq()
    if not torch.equal(native, expected_native):
        raise RuntimeError("released checkpoint native RoPE frequency drift")
    if frequency == "native":
        receipt.update(
            {
                "active_frequency": "native_endpoint_rope",
                "active_sha256_float32": tensor_sha256(native),
                "tau": 0.0,
            }
        )
        return receipt
    if frequency != "evq":
        raise ValueError(f"unknown frequency {frequency!r}")
    receipt = patch_endpoint_evq(model, tau=TAU)
    receipt.update(
        {
            "active_frequency": "evq_endpoint_cosh",
            "active_sha256_float32": receipt["evq_sha256_float32"],
        }
    )
    return receipt


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


def _loss_value(value: Any) -> torch.Tensor:
    return value.loss if hasattr(value, "loss") else value


def natural_batch(
    *,
    view: FixedView,
    indices: np.ndarray,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    host_ids = np.asarray(view.input_ids[indices], dtype=np.int64)
    host_lengths = np.asarray(view.lengths[indices], dtype=np.int64)
    context = torch.from_numpy(host_ids[:, :-1]).to(
        "cuda", non_blocking=True
    )
    labels = torch.from_numpy(host_ids[:, 1:].copy()).to(
        "cuda", non_blocking=True
    )
    positions = torch.arange(
        labels.shape[1], device="cuda"
    )[None, :]
    valid = positions < torch.from_numpy(
        (host_lengths - 1)[:, None]
    ).to("cuda")
    if objective == "assistant":
        host_mask = np.asarray(
            view.assistant_mask[indices, 1:], dtype=np.uint8
        )
        valid &= torch.from_numpy(host_mask).to(
            "cuda", non_blocking=True
        ).bool()
    elif objective != "full":
        raise ValueError(f"unknown natural objective {objective!r}")
    labels.masked_fill_(~valid, -100)
    supervised = int(valid.sum())
    if supervised <= 0:
        raise RuntimeError("natural batch has no supervised tokens")
    return context, labels, supervised


def train_adapter(
    *,
    model: Any,
    longalign: FixedView,
    tulu: FixedView,
    causal: FormalProbeSet,
    canary: FormalProbeSet,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    compile_mode: str,
    margin_loss_weight: float,
    log_path: Path,
) -> dict[str, Any]:
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("training mode has no trainable adapter parameters")
    if int(micro_batch_size) != 1:
        raise RuntimeError("verified RTX 5090 recipe requires micro-batch one")
    if int(gradient_accumulation_steps) != 4:
        raise RuntimeError(
            "verified RTX 5090 recipe requires four accumulation steps"
        )
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=compile_mode,
    )
    fused_loss = fused_loss_module()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 31_337)
    train_rows = {
        "longalign_full": torch.from_numpy(
            longalign.training_rows.copy()
        ),
        "tulu_assistant": torch.from_numpy(tulu.training_rows.copy()),
    }
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    recent_losses: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(steps) + 1):
        family = MIXTURE_PATTERN[(step - 1) % len(MIXTURE_PATTERN)]
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        raw_losses: list[float] = []
        component_losses: dict[str, list[float]] = {}
        for micro_step in range(int(gradient_accumulation_steps)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if family in {"longalign_full", "tulu_assistant"}:
                    view = (
                        longalign
                        if family == "longalign_full"
                        else tulu
                    )
                    row_pool = train_rows[family]
                    selection = torch.randint(
                        len(row_pool),
                        (int(micro_batch_size),),
                        generator=generator,
                    )
                    indices = row_pool[selection].numpy()
                    context, labels, _ = natural_batch(
                        view=view,
                        indices=indices,
                        objective=(
                            "full"
                            if family == "longalign_full"
                            else "assistant"
                        ),
                    )
                    hidden = backbone(context)
                    raw_loss = _loss_value(
                        fused_loss(
                            model.lm_head.weight,
                            hidden.reshape(-1, hidden.shape[-1]),
                            labels.reshape(-1),
                        )
                    )
                    component_losses.setdefault(
                        "natural_ce", []
                    ).append(float(raw_loss.detach()))
                else:
                    index = int(
                        torch.randint(
                            len(causal.gold),
                            (1,),
                            generator=generator,
                        )
                    )
                    use_swapped = micro_step % 2 == 1
                    source_array = (
                        causal.swapped
                        if use_swapped
                        else causal.sourced
                    )
                    target_array = (
                        causal.alternate
                        if use_swapped
                        else causal.gold
                    )
                    counterfactual_array = (
                        causal.gold
                        if use_swapped
                        else causal.alternate
                    )
                    context = torch.from_numpy(
                        np.asarray(
                            source_array[index : index + 1],
                            dtype=np.int64,
                        )
                    ).to("cuda", non_blocking=True)
                    labels = torch.from_numpy(
                        np.asarray(
                            target_array[index : index + 1],
                            dtype=np.int64,
                        )
                    ).to("cuda", non_blocking=True)
                    counterfactual = torch.from_numpy(
                        np.asarray(
                            counterfactual_array[index : index + 1],
                            dtype=np.int64,
                        )
                    ).to("cuda", non_blocking=True)
                    hidden = backbone(context)[:, -1, :]
                    logits = logits_from_hidden(model, hidden, None)
                    ce_loss = F.cross_entropy(logits.float(), labels)
                    top1_loss, counterfactual_loss = causal_margin_losses(
                        logits,
                        labels,
                        counterfactual,
                        top1_margin=1.0,
                        counterfactual_margin=1.0,
                    )
                    raw_loss = ce_loss + float(margin_loss_weight) * (
                        top1_loss + counterfactual_loss
                    )
                    component_losses.setdefault("causal_ce", []).append(
                        float(ce_loss.detach())
                    )
                    component_losses.setdefault(
                        "top1_margin", []
                    ).append(float(top1_loss.detach()))
                    component_losses.setdefault(
                        "counterfactual_margin", []
                    ).append(float(counterfactual_loss.detach()))
                loss = raw_loss / float(gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss at optimizer step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            processed_tokens += int(context.numel())
            del context, labels, hidden, raw_loss, loss
            if family == "causal_margin":
                del logits, ce_loss, top1_loss
                del counterfactual_loss, counterfactual
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(raw_losses))
        recent_losses.append(mean_loss)
        should_log = (
            step == 1 or step % 10 == 0 or step == int(steps)
        )
        if should_log:
            torch.cuda.synchronize()
            now = time.perf_counter()
            interval_tokens = processed_tokens - last_log_tokens
            row = {
                "step": step,
                "family": family,
                "loss": mean_loss,
                "mean_loss_last_10": float(
                    np.mean(recent_losses[-10:])
                ),
                "components": {
                    name: float(np.mean(values))
                    for name, values in component_losses.items()
                },
                "lr": lr,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": now - started,
                "processed_tokens": processed_tokens,
                "interval_tokens_per_second": (
                    interval_tokens / max(now - last_log_time, 1e-9)
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            }
            if step in {1, 25, 50, 100, 200, 300, int(steps)}:
                row["heldout_canary"] = quick_source_metrics(
                    model=model,
                    readout=None,
                    data=canary,
                    count=min(8, len(canary.gold)),
                )
                model.train()
            append_jsonl(log_path, row)
            last_log_time = now
            last_log_tokens = processed_tokens
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": int(steps),
        "mixture_pattern": list(MIXTURE_PATTERN),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": int(
            micro_batch_size * gradient_accumulation_steps
        ),
        "gradient_checkpointing": False,
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "processed_tokens": processed_tokens,
        "elapsed_seconds": elapsed,
        "overall_tokens_per_second": processed_tokens / elapsed,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
        "optimizer": "fused_adamw",
        "precision": "bf16_autocast",
        "loss_backend": "liger_fused_linear_cross_entropy+native_margin",
    }


@torch.inference_mode()
def evaluate_natural_nll(
    *,
    model: Any,
    background_dir: Path,
    lengths: Sequence[int],
    rows: int,
    tail_tokens: int,
) -> dict[str, Any]:
    documents = np.load(
        background_dir / "documents_L16384.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = json.loads(
        (background_dir / "documents_L16384.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    test_rows = [
        int(row["row"]) for row in metadata if row["split"] == "test"
    ][: int(rows)]
    if len(test_rows) != int(rows):
        raise RuntimeError("insufficient natural-NLL evaluation rows")
    loss_module = fused_loss_module()
    model.eval()
    output: dict[str, Any] = {}
    for length in sorted(int(value) for value in lengths):
        total_loss = 0.0
        total_tokens = 0
        tail_loss = 0.0
        tail_count = 0
        for row_index in test_rows:
            token_ids = np.asarray(
                documents[row_index, :length], dtype=np.int64
            )
            context = torch.from_numpy(
                token_ids[:-1][None, :].copy()
            ).to("cuda", non_blocking=True)
            targets = torch.from_numpy(
                token_ids[1:].copy()
            ).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = forward_hidden(
                    model, context, all_positions=True
                )
                full = _loss_value(
                    loss_module(
                        model.lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        targets.reshape(-1),
                    )
                )
                local_tail = min(int(tail_tokens), targets.numel())
                tail = _loss_value(
                    loss_module(
                        model.lm_head.weight,
                        hidden[:, -local_tail:, :].reshape(
                            -1, hidden.shape[-1]
                        ),
                        targets[-local_tail:],
                    )
                )
            count = int(targets.numel())
            total_loss += float(full) * count
            total_tokens += count
            tail_loss += float(tail) * local_tail
            tail_count += local_tail
            del context, targets, hidden, full, tail
        mean_nll = total_loss / total_tokens
        mean_tail_nll = tail_loss / tail_count
        output[f"L{length}"] = {
            "rows": len(test_rows),
            "tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": float(math.exp(min(mean_nll, 50.0))),
            "tail_tokens_per_row": int(tail_tokens),
            "tail_mean_nll": mean_tail_nll,
            "tail_perplexity": float(
                math.exp(min(mean_tail_nll, 50.0))
            ),
        }
    return output


@torch.inference_mode()
def evaluate_causal(
    *,
    model: Any,
    data: FormalProbeSet,
    answer_token_ids: Sequence[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    candidate_ids = torch.tensor(
        list(answer_token_ids), device="cuda", dtype=torch.long
    )
    rows: list[dict[str, Any]] = []
    for start in range(0, len(data.gold), int(batch_size)):
        end = min(len(data.gold), start + int(batch_size))
        arrays = (data.sourced, data.deleted, data.swapped)
        combined = np.concatenate(
            [
                np.asarray(value[start:end], dtype=np.int64)
                for value in arrays
            ],
            axis=0,
        )
        context = torch.from_numpy(combined).to(
            "cuda", non_blocking=True
        )
        gold = torch.from_numpy(
            np.asarray(data.gold[start:end], dtype=np.int64)
        ).to("cuda", non_blocking=True)
        alternate = torch.from_numpy(
            np.asarray(data.alternate[start:end], dtype=np.int64)
        ).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = forward_hidden(model, context, all_positions=False)
            logits = logits_from_hidden(model, hidden, None).float()
        local = end - start
        source_logits, deleted_logits, swapped_logits = logits.split(
            local, dim=0
        )
        source_nll, source_rank, source_exact = score_logits(
            source_logits, gold
        )
        deleted_nll, _, deleted_exact = score_logits(
            deleted_logits, gold
        )
        delta = source_logits - deleted_logits
        delta_rank = rank_of(delta, gold)
        gold_delta = delta.gather(-1, gold[:, None]).squeeze(-1)
        source_gold = source_logits.gather(
            -1, gold[:, None]
        ).squeeze(-1)
        source_alternate = source_logits.gather(
            -1, alternate[:, None]
        ).squeeze(-1)
        swapped_gold = swapped_logits.gather(
            -1, gold[:, None]
        ).squeeze(-1)
        swapped_alternate = swapped_logits.gather(
            -1, alternate[:, None]
        ).squeeze(-1)
        swap_follow = (
            source_gold
            - source_alternate
            - swapped_gold
            + swapped_alternate
        )
        answer_logits = source_logits[:, candidate_ids]
        candidate_other = answer_logits.masked_fill(
            candidate_ids[None, :] == gold[:, None],
            float("-inf"),
        ).max(dim=-1).values
        for offset in range(local):
            interval = scalar_alpha_interval(
                deleted_logits[offset],
                delta[offset],
                int(gold[offset]),
            )
            fraction = float(data.source_fraction[start + offset])
            rows.append(
                {
                    "row": int(start + offset),
                    "key": data.keys[start + offset],
                    "source_fraction": fraction,
                    "position_decile": min(9, int(fraction * 10.0)),
                    "distractor_count": int(
                        data.distractor_count[start + offset]
                    ),
                    "template_id": data.template_ids[start + offset],
                    "gold_token_id": int(gold[offset]),
                    "alternate_token_id": int(alternate[offset]),
                    "answer_nll": float(source_nll[offset]),
                    "answer_rank": int(source_rank[offset]),
                    "exact": int(source_exact[offset]),
                    "deleted_exact": int(deleted_exact[offset]),
                    "source_deletion_nll_gap": float(
                        deleted_nll[offset] - source_nll[offset]
                    ),
                    "source_delta_rank": int(delta_rank[offset]),
                    "gold_source_delta_logit": float(
                        gold_delta[offset]
                    ),
                    "gold_minus_answer_candidate": float(
                        source_gold[offset] - candidate_other[offset]
                    ),
                    "swap_follow_score": float(swap_follow[offset]),
                    "scalar_alpha_feasible": interval is not None,
                    "scalar_alpha_lower": (
                        None if interval is None else interval[0]
                    ),
                    "scalar_alpha_upper": (
                        None
                        if interval is None
                        or not math.isfinite(interval[1])
                        else interval[1]
                    ),
                }
            )
        del context, hidden, logits, delta
    summary = {"overall": aggregate_rows(rows)}
    for decile in range(10):
        local_rows = [
            row for row in rows if row["position_decile"] == decile
        ]
        if local_rows:
            summary[f"position_decile_{decile}"] = aggregate_rows(
                local_rows
            )
    return rows, summary


def verified_checkpoint_digest(
    receipt_path: Path,
    checkpoint: Path,
) -> tuple[str, str]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "OLMO2_MATURITY_ASSETS_VERIFIED":
        raise RuntimeError("asset receipt is not a verified READY receipt")
    entry = receipt["checkpoints"].get(checkpoint.name)
    if entry is None:
        raise RuntimeError(
            f"checkpoint {checkpoint.name!r} missing from asset receipt"
        )
    if entry.get("status") != "verified":
        raise RuntimeError("checkpoint is not verified in asset receipt")
    return str(entry["composite_sha256"]), sha256_large_file(receipt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--causal-data", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("base", "train"), required=True
    )
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
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
    parser.add_argument("--margin-loss-weight", type=float, default=0.25)
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--natural-tail-tokens", type=int, default=1_024)
    parser.add_argument("--causal-eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.mode == "base" and int(args.steps) != 0:
        raise ValueError("base mode requires --steps 0")
    if args.mode == "train" and args.frequency not in {"native", "evq"}:
        raise ValueError("training requires native or EVQ frequency")
    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    checkpoint = args.checkpoint.resolve()
    prepared = args.prepared_data.resolve()
    background = args.background_dir.resolve()
    causal_root = args.causal_data.resolve()
    asset_receipt = args.asset_receipt.resolve()
    longalign = load_fixed_view(
        prepared / "longalign_paired_L16384"
    )
    tulu = load_fixed_view(prepared / "tulu3_replay_L4096")
    train_causal = load_probe_set(
        causal_root / "train_continuous_compositional"
    )
    canary = load_probe_set(causal_root / "canary_full_heldout")

    model = load_model(checkpoint)
    frequency = apply_frequency(model, args.frequency)
    if args.mode == "train":
        readout = install_adaptation(
            model,
            "qkvo_causal_margin",
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        if readout is not None:
            raise RuntimeError("first maturity candidate must not add readout")
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.to("cuda")

    training: dict[str, Any] | None = None
    if args.mode == "train":
        training = train_adapter(
            model=model,
            longalign=longalign,
            tulu=tulu,
            causal=train_causal,
            canary=canary,
            steps=int(args.steps),
            micro_batch_size=int(args.micro_batch_size),
            gradient_accumulation_steps=int(
                args.gradient_accumulation_steps
            ),
            learning_rate=float(args.learning_rate),
            warmup_steps=int(args.warmup_steps),
            seed=int(args.seed),
            compile_mode=args.compile_mode,
            margin_loss_weight=float(args.margin_loss_weight),
            log_path=output / "train_log.jsonl",
        )
    gc.collect()
    torch.cuda.empty_cache()

    natural = evaluate_natural_nll(
        model=model,
        background_dir=background,
        lengths=(4_096, 8_192, 16_384),
        rows=int(args.natural_eval_rows),
        tail_tokens=int(args.natural_tail_tokens),
    )
    causal_manifest = json.loads(
        (causal_root / "collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    answer_ids = [
        int(value)
        for value in causal_manifest[
            "registered_eval_value_token_ids"
        ]
    ]
    causal_results: dict[str, Any] = {}
    for entry in causal_manifest["sets"]:
        if entry["purpose"] not in {
            "context_factorial_evaluation",
            "dense_position_evaluation",
        }:
            continue
        data = load_probe_set(causal_root / entry["relative_path"])
        rows, summary = evaluate_causal(
            model=model,
            data=data,
            answer_token_ids=answer_ids,
            batch_size=int(args.causal_eval_batch_size),
        )
        causal_results[entry["name"]] = {
            "dataset_sha256": entry["dataset_sha256"],
            "summary": summary,
            "rows": rows,
        }

    checkpoint_hash, asset_receipt_hash = verified_checkpoint_digest(
        asset_receipt, checkpoint
    )
    adapter_metadata = {
        "base_checkpoint_digest": checkpoint_hash,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": (
            "none" if args.mode == "base" else "qkvo_causal_margin"
        ),
        "rank": int(args.rank) if args.mode == "train" else None,
        "alpha": float(args.alpha) if args.mode == "train" else None,
        "seed": int(args.seed),
        "causal_collection_sha256": sha256_large_file(
            causal_root / "collection_manifest.json"
        ),
        "prepared_collection_sha256": sha256_large_file(
            prepared / "collection_manifest.json"
        ),
        "background_manifest_sha256": sha256_large_file(
            background / "manifest.json"
        ),
        "asset_receipt_sha256": asset_receipt_hash,
    }
    adapter_sha = (
        save_adapter(
            output / "adapter.pt",
            model,
            None,
            adapter_metadata,
        )
        if args.mode == "train"
        else None
    )
    receipt = {
        "status": "OLMO2_LORA_MATURITY_SCREEN_COMPLETE",
        "metric_boundary": (
            "natural next-token NLL plus held-out one-token source-causal "
            "retrieval; not yet broad downstream-task evidence"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": adapter_metadata[
            "base_checkpoint_digest"
        ],
        "mode": args.mode,
        "frequency": frequency,
        "adaptation": adapter_metadata["adaptation"],
        "adapter_sha256": adapter_sha,
        "seed": int(args.seed),
        "runtime": runtime,
        "protocol": {
            "steps": int(args.steps),
            "rank": (
                int(args.rank) if args.mode == "train" else None
            ),
            "alpha": (
                float(args.alpha) if args.mode == "train" else None
            ),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "compile_mode": args.compile_mode,
            "mixture_pattern": list(MIXTURE_PATTERN),
            "margin_loss_weight": float(args.margin_loss_weight),
            "natural_eval_rows": int(args.natural_eval_rows),
            "natural_tail_tokens": int(args.natural_tail_tokens),
        },
        "assets": adapter_metadata,
        "training": training,
        "natural_nll": natural,
        "causal": causal_results,
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "mode": args.mode,
                "frequency": args.frequency,
                "training": training,
                "natural_nll": natural,
                "causal_summary": {
                    name: value["summary"]["overall"]
                    for name, value in causal_results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
