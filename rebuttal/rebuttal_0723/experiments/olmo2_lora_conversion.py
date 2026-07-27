#!/usr/bin/env python3
"""LoRA routing/readout conversion probe for released OLMo-2 checkpoints.

The source checkpoint is loaded read-only.  Outputs contain only diagnostics
and trainable adapter tensors, never a copied base model or optimizer state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_retrieval_data import (
    load_filler,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
    configure_flash_only_attention,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
    ProbeSet,
    ReadoutAdapter,
    aggregate_rows,
    append_jsonl,
    atomic_json,
    build_probe_set,
    configure_cuda,
    cosine_lr,
    one_token_answers,
    rank_of,
    scalar_alpha_interval,
    score_logits,
    seed_everything,
    sha256_file,
)


ADAPTATIONS = (
    "baseline",
    "qk_answer",
    "qv_answer",
    "qkvo_answer",
    "qkvo_causal_margin",
    "qkvo_full",
    "readout_answer",
    "qkvo_readout_answer",
)


def load_model(checkpoint: Path, *, config: Any | None = None) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "olmo2":
        raise RuntimeError(f"expected olmo2, got {model.config.model_type}")
    if (
        int(model.config.hidden_size) != 2_048
        or int(model.config.num_hidden_layers) != 16
        or int(model.config.vocab_size) != 100_352
    ):
        raise RuntimeError("released OLMo-2 1B architecture drift")
    model.config.use_cache = False
    configure_flash_only_attention(model)
    return model


def install_adaptation(
    model: Any,
    adaptation: str,
    rank: int,
    alpha: float,
    *,
    qk_output_mask: torch.Tensor | None = None,
) -> ReadoutAdapter | None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    use_qv = adaptation.startswith("qv_")
    use_qk = adaptation.startswith("qk_")
    use_qkvo = adaptation.startswith("qkvo_")
    if use_qv or use_qk or use_qkvo:
        for layer in model.model.layers:
            attention = layer.self_attn
            if use_qv:
                names = ("q_proj", "v_proj")
            elif use_qk:
                names = ("q_proj", "k_proj")
            else:
                names = ("q_proj", "k_proj", "v_proj", "o_proj")
            for name in names:
                setattr(
                    attention,
                    name,
                    LoRALinear(
                        getattr(attention, name),
                        rank=rank,
                        alpha=alpha,
                        output_mask=(
                            qk_output_mask
                            if name in {"q_proj", "k_proj"}
                            else None
                        ),
                    ),
                )
    readout = None
    if "readout" in adaptation:
        readout = ReadoutAdapter(
            hidden_size=int(model.config.hidden_size),
            vocab_size=int(model.config.vocab_size),
            rank=rank,
            alpha=alpha,
        )
    return readout


def trainable_named_parameters(
    model: Any,
    readout: ReadoutAdapter | None,
) -> list[tuple[str, nn.Parameter]]:
    values = [
        (f"model.{name}", parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if readout is not None:
        values.extend(
            (f"readout.{name}", parameter)
            for name, parameter in readout.named_parameters()
            if parameter.requires_grad
        )
    return values


def forward_hidden(
    model: Any,
    input_ids: torch.Tensor,
    *,
    all_positions: bool,
) -> torch.Tensor:
    outputs = model.model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    return hidden if all_positions else hidden[:, -1, :]


class TrainingBackbone(nn.Module):
    """Compile-friendly OLMo backbone returning all hidden positions."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def logits_from_hidden(
    model: Any,
    hidden: torch.Tensor,
    readout: ReadoutAdapter | None,
) -> torch.Tensor:
    logits = model.lm_head(hidden)
    if readout is not None:
        logits = logits + readout(hidden)
    return logits


def causal_margin_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    counterfactual_labels: torch.Tensor,
    *,
    top1_margin: float,
    counterfactual_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize both the hardest vocabulary rival and the paired answer."""

    scores = logits.float()
    target = scores.gather(1, labels[:, None]).squeeze(1)
    counterfactual = scores.gather(
        1, counterfactual_labels[:, None]
    ).squeeze(1)
    top_values, top_indices = scores.topk(k=2, dim=-1)
    hardest_other = torch.where(
        top_indices[:, 0] == labels,
        top_values[:, 1],
        top_values[:, 0],
    )
    top1_loss = F.relu(
        float(top1_margin) - (target - hardest_other)
    ).mean()
    counterfactual_loss = F.relu(
        float(counterfactual_margin) - (target - counterfactual)
    ).mean()
    return top1_loss, counterfactual_loss


@torch.no_grad()
def quick_source_metrics(
    *,
    model: Any,
    readout: ReadoutAdapter | None,
    data: ProbeSet,
    count: int,
) -> dict[str, Any]:
    model.eval()
    if readout is not None:
        readout.eval()
    n = min(int(count), len(data.gold))
    context = torch.from_numpy(data.sourced[:n]).to("cuda")
    labels = torch.from_numpy(data.gold[:n]).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = forward_hidden(model, context, all_positions=False)
        logits = logits_from_hidden(model, hidden, readout).float()
    nll, ranks, exact = score_logits(logits, labels)
    return {
        "n": n,
        "mean_nll": float(nll.mean()),
        "median_rank": float(torch.median(ranks.float())),
        "exact_match": float(exact.float().mean()),
    }


def save_adapter(
    path: Path,
    model: Any,
    readout: ReadoutAdapter | None,
    metadata: dict[str, Any],
) -> str | None:
    named = trainable_named_parameters(model, readout)
    if not named:
        return None
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in named
    }
    temporary = path.with_name(path.name + ".incomplete")
    torch.save({"state": state, "metadata": metadata}, temporary)
    temporary.replace(path)
    return sha256_file(path)


def train(
    *,
    model: Any,
    readout: ReadoutAdapter | None,
    data: ProbeSet,
    canary_data: ProbeSet,
    adaptation: str,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    log_path: Path,
    canary_count: int,
    gradient_checkpointing: bool = True,
    compile_mode: str = "none",
    margin_loss_weight: float = 0.25,
    top1_margin: float = 1.0,
    counterfactual_margin: float = 1.0,
) -> dict[str, Any]:
    named_parameters = trainable_named_parameters(model, readout)
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        return {
            "steps": 0,
            "micro_steps": 0,
            "trainable_parameters": 0,
            "objective": "none",
        }
    if adaptation.endswith("_full"):
        objective = "full_token"
    elif adaptation.endswith("_causal_margin"):
        objective = "causal_margin"
    else:
        objective = "answer_only"
    if (
        objective == "causal_margin"
        and int(gradient_accumulation_steps) % 2 != 0
    ):
        raise ValueError(
            "causal-margin training requires even gradient accumulation "
            "so each sampled row contributes sourced and swapped contexts"
        )
    full_token_loss: nn.Module | None = None
    if objective == "full_token":
        if readout is not None:
            raise RuntimeError(
                "fused full-token loss does not support a readout adapter"
            )
        from liger_kernel.transformers import (
            LigerFusedLinearCrossEntropyLoss,
        )

        full_token_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
            return_z_loss=False,
            accum_dtype=torch.float32,
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    training_backbone: nn.Module | None = None
    if compile_mode != "none":
        training_backbone = torch.compile(
            TrainingBackbone(model.model),
            fullgraph=True,
            dynamic=False,
            mode=compile_mode,
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 31_337)
    started = time.perf_counter()
    recent: list[float] = []
    model.train()
    if readout is not None:
        readout.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(steps) + 1):
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        step_ce_losses: list[float] = []
        step_top1_margin_losses: list[float] = []
        step_counterfactual_margin_losses: list[float] = []
        last_answer_logits: torch.Tensor | None = None
        last_labels: torch.Tensor | None = None
        paired_indices: np.ndarray | None = None
        for micro_step in range(int(gradient_accumulation_steps)):
            if objective == "causal_margin":
                if micro_step % 2 == 0:
                    paired_indices = torch.randint(
                        len(data.gold),
                        (int(micro_batch_size),),
                        generator=generator,
                    ).numpy()
                assert paired_indices is not None
                indices = paired_indices
                use_swapped = micro_step % 2 == 1
                context_values = (
                    data.swapped if use_swapped else data.sourced
                )
                label_values = (
                    data.alternate if use_swapped else data.gold
                )
                counterfactual_values = (
                    data.gold if use_swapped else data.alternate
                )
            else:
                indices = torch.randint(
                    len(data.gold),
                    (int(micro_batch_size),),
                    generator=generator,
                ).numpy()
                context_values = data.sourced
                label_values = data.gold
                counterfactual_values = None
            context = torch.from_numpy(context_values[indices]).to(
                "cuda", non_blocking=True
            )
            labels = torch.from_numpy(label_values[indices]).to(
                "cuda", non_blocking=True
            )
            counterfactual_labels = (
                None
                if counterfactual_values is None
                else torch.from_numpy(
                    counterfactual_values[indices]
                ).to("cuda", non_blocking=True)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if training_backbone is None:
                    hidden = forward_hidden(
                        model,
                        context,
                        all_positions=(objective == "full_token"),
                    )
                else:
                    hidden = training_backbone(context)
                    if objective != "full_token":
                        hidden = hidden[:, -1, :]
                logits: torch.Tensor | None = None
                if objective in {"answer_only", "causal_margin"}:
                    logits = logits_from_hidden(model, hidden, readout)
                    ce_loss = F.cross_entropy(logits.float(), labels)
                    if objective == "causal_margin":
                        assert counterfactual_labels is not None
                        (
                            top1_margin_loss,
                            counterfactual_margin_loss,
                        ) = causal_margin_losses(
                            logits,
                            labels,
                            counterfactual_labels,
                            top1_margin=float(top1_margin),
                            counterfactual_margin=float(
                                counterfactual_margin
                            ),
                        )
                        raw_loss = ce_loss + float(
                            margin_loss_weight
                        ) * (
                            top1_margin_loss
                            + counterfactual_margin_loss
                        )
                    else:
                        top1_margin_loss = None
                        counterfactual_margin_loss = None
                        raw_loss = ce_loss
                    answer_logits = logits
                else:
                    assert full_token_loss is not None
                    full = torch.cat((context, labels[:, None]), dim=1)
                    targets = full[:, 1:]
                    fused_output = full_token_loss(
                        model.lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        targets.reshape(-1),
                    )
                    raw_loss = (
                        fused_output.loss
                        if hasattr(fused_output, "loss")
                        else fused_output
                    )
                    answer_logits = logits_from_hidden(
                        model, hidden[:, -1, :], readout
                    )
                    ce_loss = None
                    top1_margin_loss = None
                    counterfactual_margin_loss = None
                loss = raw_loss / float(gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss at optimizer step {step}"
                )
            loss.backward()
            step_losses.append(float(raw_loss.detach()))
            if ce_loss is not None:
                step_ce_losses.append(float(ce_loss.detach()))
            if top1_margin_loss is not None:
                step_top1_margin_losses.append(
                    float(top1_margin_loss.detach())
                )
            if counterfactual_margin_loss is not None:
                step_counterfactual_margin_losses.append(
                    float(counterfactual_margin_loss.detach())
                )
            last_answer_logits = answer_logits.detach()
            last_labels = labels.detach()
            del (
                context,
                labels,
                counterfactual_labels,
                hidden,
                logits,
                loss,
                raw_loss,
                ce_loss,
                top1_margin_loss,
                counterfactual_margin_loss,
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        mean_loss = float(np.mean(step_losses))
        recent.append(mean_loss)
        should_log = (
            step == 1 or step % 10 == 0 or step == int(steps)
        )
        if should_log:
            assert last_answer_logits is not None and last_labels is not None
            ranks = rank_of(last_answer_logits.float(), last_labels)
            row = {
                "step": step,
                "loss": mean_loss,
                "mean_loss_last_10": float(np.mean(recent[-10:])),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "batch_exact": float((ranks == 1).float().mean()),
                "batch_median_rank": float(torch.median(ranks.float())),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
            if step_ce_losses:
                row["ce_loss"] = float(np.mean(step_ce_losses))
            if step_top1_margin_losses:
                row["top1_margin_loss"] = float(
                    np.mean(step_top1_margin_losses)
                )
            if step_counterfactual_margin_losses:
                row["counterfactual_margin_loss"] = float(
                    np.mean(step_counterfactual_margin_losses)
                )
            if step in {1, 10, 25, 50, 100, 200, 300, int(steps)}:
                row["heldout_canary"] = quick_source_metrics(
                    model=model,
                    readout=readout,
                    data=canary_data,
                    count=int(canary_count),
                )
                model.train()
                if readout is not None:
                    readout.train()
            append_jsonl(log_path, row)
    torch.cuda.synchronize()
    return {
        "steps": int(steps),
        "micro_steps": int(steps) * int(gradient_accumulation_steps),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "objective": objective,
        "margin_loss_weight": (
            float(margin_loss_weight)
            if objective == "causal_margin"
            else None
        ),
        "top1_margin": (
            float(top1_margin)
            if objective == "causal_margin"
            else None
        ),
        "counterfactual_margin": (
            float(counterfactual_margin)
            if objective == "causal_margin"
            else None
        ),
        "loss_backend": (
            "liger_fused_linear_cross_entropy"
            if objective == "full_token"
            else "native_cross_entropy"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


@torch.no_grad()
def evaluate(
    *,
    model: Any,
    readout: ReadoutAdapter | None,
    data_by_length: dict[int, ProbeSet],
    answer_token_ids: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    model.gradient_checkpointing_disable()
    if readout is not None:
        readout.eval()
    rows: list[dict[str, Any]] = []
    candidate_ids = torch.tensor(answer_token_ids, device="cuda")
    for length, data in sorted(data_by_length.items()):
        for start in range(0, len(data.gold), int(batch_size)):
            end = min(len(data.gold), start + int(batch_size))
            source = torch.from_numpy(data.sourced[start:end]).to("cuda")
            deleted = torch.from_numpy(data.deleted[start:end]).to("cuda")
            swapped = torch.from_numpy(data.swapped[start:end]).to("cuda")
            gold = torch.from_numpy(data.gold[start:end]).to("cuda")
            alternate = torch.from_numpy(data.alternate[start:end]).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_logits = logits_from_hidden(
                    model,
                    forward_hidden(model, source, all_positions=False),
                    readout,
                ).float()
                deleted_logits = logits_from_hidden(
                    model,
                    forward_hidden(model, deleted, all_positions=False),
                    readout,
                ).float()
                swapped_logits = logits_from_hidden(
                    model,
                    forward_hidden(model, swapped, all_positions=False),
                    readout,
                ).float()
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
            for offset in range(end - start):
                interval = scalar_alpha_interval(
                    deleted_logits[offset],
                    delta[offset],
                    int(gold[offset]),
                )
                rows.append(
                    {
                        "length": int(length),
                        "row": int(start + offset),
                        "key": data.keys[start + offset],
                        "source_fraction": float(
                            data.source_fraction[start + offset]
                        ),
                        "distractor_count": int(
                            data.distractor_count[start + offset]
                        ),
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
            del source, deleted, swapped
            del source_logits, deleted_logits, swapped_logits, delta
            gc.collect()
            torch.cuda.empty_cache()
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        keys = (
            f"L{row['length']}",
            (
                f"L{row['length']}_P{row['source_fraction']:.1f}_"
                f"D{row['distractor_count']}"
            ),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    return rows, {
        key: aggregate_rows(local)
        for key, local in sorted(groups.items())
    }


def captured_last_hidden(
    model: Any,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    captured: list[torch.Tensor | None] = [
        None for _ in range(len(model.model.layers))
    ]
    handles = []

    def make_hook(index: int):
        def hook(
            _: nn.Module,
            __: tuple[Any, ...],
            output: Any,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach()

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final = forward_hidden(model, input_ids, all_positions=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("layer hook did not capture every decoder layer")
    values = [
        model.model.norm(value)  # type: ignore[arg-type]
        for value in captured
    ]
    values.append(final)
    return values


@torch.no_grad()
def layer_trace(
    *,
    model: Any,
    readout: ReadoutAdapter | None,
    data_by_length: dict[int, ProbeSet],
    examples: int,
) -> list[dict[str, Any]]:
    if examples <= 0:
        return []
    model.eval()
    model.gradient_checkpointing_disable()
    if readout is not None:
        readout.eval()
    output: list[dict[str, Any]] = []
    for length, data in sorted(data_by_length.items()):
        count = min(int(examples), len(data.gold))
        source = torch.from_numpy(data.sourced[:count]).to("cuda")
        deleted = torch.from_numpy(data.deleted[:count]).to("cuda")
        labels = torch.from_numpy(data.gold[:count]).to("cuda")
        source_layers = captured_last_hidden(model, source)
        deleted_layers = captured_last_hidden(model, deleted)
        for layer, (source_hidden, deleted_hidden) in enumerate(
            zip(source_layers, deleted_layers), start=1
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_logits = logits_from_hidden(
                    model, source_hidden, readout
                ).float()
                deleted_logits = logits_from_hidden(
                    model, deleted_hidden, readout
                ).float()
            delta = source_logits - deleted_logits
            ranks = rank_of(source_logits, labels).float()
            delta_ranks = rank_of(delta, labels).float()
            gold_delta = delta.gather(
                -1, labels[:, None]
            ).squeeze(-1)
            output.append(
                {
                    "length": int(length),
                    "layer": layer,
                    "n": count,
                    "median_answer_rank": float(torch.median(ranks)),
                    "median_source_delta_rank": float(
                        torch.median(delta_ranks)
                    ),
                    "mean_gold_source_delta_logit": float(
                        gold_delta.mean()
                    ),
                }
            )
        del source_layers, deleted_layers, source, deleted
        gc.collect()
        torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    filler_group = parser.add_mutually_exclusive_group(required=True)
    filler_group.add_argument("--eval-manifest", type=Path)
    filler_group.add_argument("--filler-array", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation", choices=ADAPTATIONS, required=True)
    parser.add_argument("--train-length", type=int, default=8_192)
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 16_384],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples-per-cell", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--layer-trace-examples", type=int, default=1)
    parser.add_argument("--canary-count", type=int, default=8)
    parser.add_argument("--margin-loss-weight", type=float, default=0.25)
    parser.add_argument("--top1-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin", type=float, default=1.0
    )
    args = parser.parse_args()

    if (
        args.compile_mode != "none"
        and bool(args.gradient_checkpointing)
    ):
        raise RuntimeError(
            "compiled conversion probe requires checkpointing off"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    answers = one_token_answers(tokenizer)
    if args.filler_array is not None:
        filler_path = args.filler_array.resolve()
        filler = np.load(
            filler_path, mmap_mode="r", allow_pickle=False
        ).reshape(-1)
        if filler.dtype != np.uint32:
            raise RuntimeError("OLMo filler array must be uint32")
        filler_source_sha = sha256_file(filler_path)
    else:
        assert args.eval_manifest is not None
        filler, filler_source_sha = load_filler(
            args.eval_manifest.resolve()
        )
    train_data = build_probe_set(
        tokenizer=tokenizer,
        filler=filler,
        answers=answers,
        length=int(args.train_length),
        count=int(args.train_examples),
        seed=int(args.seed) + 1,
        source_fractions=(0.1, 0.5, 0.9),
        distractor_counts=(0, 4, 8),
    )
    cell_count = 3 * 2 * int(args.eval_examples_per_cell)
    eval_data = {
        int(length): build_probe_set(
            tokenizer=tokenizer,
            filler=filler,
            answers=answers,
            length=int(length),
            count=cell_count,
            seed=int(args.seed) + 100_000 + int(length),
            source_fractions=(0.1, 0.5, 0.9),
            distractor_counts=(0, 8),
        )
        for length in args.eval_lengths
    }
    if any(
        set(train_data.keys) & set(local.keys)
        for local in eval_data.values()
    ):
        raise RuntimeError("train/evaluation key overlap")
    if int(args.train_length) not in eval_data:
        raise RuntimeError(
            "train-length must also be present in eval-lengths "
            "for the held-out training canary"
        )

    model = load_model(checkpoint)
    readout = install_adaptation(
        model,
        args.adaptation,
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    model.to("cuda")
    if readout is not None:
        readout.to("cuda")
    effective_steps = 0 if args.adaptation == "baseline" else int(args.steps)
    training = train(
        model=model,
        readout=readout,
        data=train_data,
        canary_data=eval_data[int(args.train_length)],
        adaptation=args.adaptation,
        steps=effective_steps,
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        seed=int(args.seed),
        log_path=output / "train_log.jsonl",
        canary_count=int(args.canary_count),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        compile_mode=args.compile_mode,
        margin_loss_weight=float(args.margin_loss_weight),
        top1_margin=float(args.top1_margin),
        counterfactual_margin=float(args.counterfactual_margin),
    )
    rows, summary = evaluate(
        model=model,
        readout=readout,
        data_by_length=eval_data,
        answer_token_ids=[token_id for _, token_id in answers],
        batch_size=int(args.eval_batch_size),
    )
    trace = layer_trace(
        model=model,
        readout=readout,
        data_by_length=eval_data,
        examples=int(args.layer_trace_examples),
    )
    adapter_metadata = {
        "base_checkpoint_sha256": ":".join(
            sha256_file(path)
            for path in sorted(checkpoint.glob("model-*.safetensors"))
        ),
        "adaptation": args.adaptation,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "train_length": int(args.train_length),
        "train_dataset_sha256": train_data.digest(),
    }
    adapter_sha = save_adapter(
        output / "adapter.pt",
        model,
        readout,
        adapter_metadata,
    )
    receipt = {
        "status": "OLMO2_LORA_CONVERSION_COMPLETE",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metric_boundary": (
            "held-out teacher-forced one-token associative retrieval; "
            "not general downstream capability"
        ),
        "checkpoint": str(checkpoint),
        "base_checkpoint_sha256": adapter_metadata[
            "base_checkpoint_sha256"
        ],
        "filler_source_sha256": filler_source_sha,
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "adaptation": args.adaptation,
        "adapter_sha256": adapter_sha,
        "seed": int(args.seed),
        "runtime": runtime,
        "protocol": {
            "train_length": int(args.train_length),
            "train_examples": int(args.train_examples),
            "eval_lengths": [int(value) for value in args.eval_lengths],
            "eval_examples_per_cell": int(args.eval_examples_per_cell),
            "source_fractions": [0.1, 0.5, 0.9],
            "train_distractor_counts": [0, 4, 8],
            "eval_distractor_counts": [0, 8],
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "learning_rate": float(args.learning_rate),
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "gradient_checkpointing": bool(
                args.gradient_checkpointing
            ),
            "compile_mode": args.compile_mode,
            "margin_loss_weight": float(args.margin_loss_weight),
            "top1_margin": float(args.top1_margin),
            "counterfactual_margin": float(
                args.counterfactual_margin
            ),
            "warmup_steps": int(args.warmup_steps),
            "train_dataset_sha256": train_data.digest(),
            "eval_dataset_sha256": {
                str(length): data.digest()
                for length, data in sorted(eval_data.items())
            },
        },
        "training": training,
        "summary": summary,
        "layer_trace": trace,
        "rows": rows,
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "adaptation": args.adaptation,
                "training": training,
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key in {
                        f"L{int(length)}" for length in args.eval_lengths
                    }
                },
                "adapter_sha256": adapter_sha,
                "output": str(output / "results.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
