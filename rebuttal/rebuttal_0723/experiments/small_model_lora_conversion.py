#!/usr/bin/env python3
"""Causal LoRA-to-readout conversion probe for the repository's small GPTs.

This is an exploratory diagnostic.  It does not modify source checkpoints.
The task uses held-out random code/value associations, source deletion, and
source-value swaps to separate long-range routing from vocabulary readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.native_rope_evq_150m.model import GPT


ANSWER_WORDS = (
    " blue",
    " green",
    " red",
    " black",
    " white",
    " orange",
    " purple",
    " yellow",
    " seven",
    " nine",
    " four",
    " six",
)
ADAPTATIONS = (
    "baseline",
    "qv_answer",
    "qkvo_answer",
    "qkvo_full",
    "readout_answer",
    "qkvo_readout_answer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(values: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    capability = torch.cuda.get_device_capability(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": list(torch.cuda.get_arch_list()),
        "active_arch": f"sm_{capability[0]}{capability[1]}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cudnn_sdp_enabled": bool(
            torch.backends.cuda.cudnn_sdp_enabled()
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
            else False
        ),
    }


def one_token_answers(tokenizer: Any) -> list[tuple[str, int]]:
    answers: list[tuple[str, int]] = []
    for word in ANSWER_WORDS:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1:
            answers.append((word, int(token_ids[0])))
    if len(answers) < 8:
        raise RuntimeError(
            f"only {len(answers)} registered answers are single-token"
        )
    return answers


def encode(tokenizer: Any, text: str) -> np.ndarray:
    return np.asarray(
        tokenizer.encode(text, add_special_tokens=False), dtype=np.int64
    )


def random_key(rng: random.Random) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(alphabet) for _ in range(8))


def place_phrase(
    output: np.ndarray,
    occupied: np.ndarray,
    phrase: np.ndarray,
    preferred: int,
) -> int:
    limit = len(output) - len(phrase)
    if limit < 0:
        raise RuntimeError("phrase is longer than the available prefix")
    preferred = min(max(0, int(preferred)), limit)
    for radius in range(0, len(output) + 1):
        candidates = (
            (preferred,)
            if radius == 0
            else (preferred + radius, preferred - radius)
        )
        for start in candidates:
            if start < 0 or start > limit:
                continue
            end = start + len(phrase)
            if occupied[start:end].any():
                continue
            output[start:end] = phrase
            occupied[start:end] = True
            return start
    raise RuntimeError("could not place a non-overlapping phrase")


@dataclass
class ProbeSet:
    sourced: np.ndarray
    deleted: np.ndarray
    swapped: np.ndarray
    gold: np.ndarray
    alternate: np.ndarray
    source_fraction: np.ndarray
    distractor_count: np.ndarray
    keys: list[str]

    def digest(self) -> str:
        return sha256_arrays(
            (
                self.sourced,
                self.deleted,
                self.swapped,
                self.gold,
                self.alternate,
                self.source_fraction,
                self.distractor_count,
            )
        )


def build_probe_set(
    *,
    tokenizer: Any,
    filler: np.ndarray,
    answers: list[tuple[str, int]],
    length: int,
    count: int,
    seed: int,
    source_fractions: tuple[float, ...],
    distractor_counts: tuple[int, ...],
) -> ProbeSet:
    if length < 128:
        raise ValueError("probe length must be at least 128")
    rng = random.Random(seed)
    sourced_rows: list[np.ndarray] = []
    deleted_rows: list[np.ndarray] = []
    swapped_rows: list[np.ndarray] = []
    gold_ids: list[int] = []
    alternate_ids: list[int] = []
    fractions: list[float] = []
    densities: list[int] = []
    keys: list[str] = []
    context_length = int(length) - 1

    for row_index in range(int(count)):
        fraction = source_fractions[row_index % len(source_fractions)]
        density = distractor_counts[
            (row_index // len(source_fractions)) % len(distractor_counts)
        ]
        gold_index = rng.randrange(len(answers))
        alternate_index = rng.randrange(len(answers) - 1)
        if alternate_index >= gold_index:
            alternate_index += 1
        gold_word, gold_id = answers[gold_index]
        alternate_word, alternate_id = answers[alternate_index]
        key = random_key(rng)
        query = encode(
            tokenizer, f"\nQuestion: The value for code {key} is"
        )
        source = encode(
            tokenizer,
            f"\nMemory: The value for code {key} is{gold_word}.\n",
        )
        swapped_source = encode(
            tokenizer,
            f"\nMemory: The value for code {key} is{alternate_word}.\n",
        )
        if len(source) != len(swapped_source):
            raise RuntimeError("source-value swap changed phrase length")
        usable = context_length - len(query)
        if usable <= len(source) + 64:
            raise RuntimeError("probe is too short for source and query")
        start_max = len(filler) - usable
        if start_max <= 0:
            raise RuntimeError("filler corpus is too short")
        filler_start = rng.randrange(start_max)
        neutral = np.asarray(
            filler[filler_start : filler_start + usable],
            dtype=np.int64,
        ).copy()
        deleted_prefix = neutral.copy()
        occupied = np.zeros(usable, dtype=np.bool_)
        source_preferred = min(
            max(24, int(usable * float(fraction))),
            usable - len(source) - 24,
        )
        source_start = place_phrase(
            neutral.copy(),
            np.zeros_like(occupied),
            source,
            source_preferred,
        )

        distractor_answers = [
            row for index, row in enumerate(answers) if index != gold_index
        ]
        rng.shuffle(distractor_answers)
        for distractor_index in range(int(density)):
            word, _ = distractor_answers[
                distractor_index % len(distractor_answers)
            ]
            distractor_key = random_key(rng)
            phrase = encode(
                tokenizer,
                (
                    f"\nMemory: The value for code {distractor_key} "
                    f"is{word}.\n"
                ),
            )
            preferred = int(
                (distractor_index + 1) * usable / (int(density) + 1)
            )
            if (
                preferred < source_start + len(source)
                and preferred + len(phrase) > source_start
            ):
                preferred = source_start + len(source) + 8
            place_phrase(deleted_prefix, occupied, phrase, preferred)

        sourced_prefix = deleted_prefix.copy()
        swapped_prefix = deleted_prefix.copy()
        if occupied[source_start : source_start + len(source)].any():
            # Re-place source in the nearest free region while preserving the
            # requested depth as closely as possible.
            source_occupied = occupied.copy()
            source_start = place_phrase(
                sourced_prefix,
                source_occupied,
                source,
                source_preferred,
            )
            swapped_prefix[:] = deleted_prefix
            swapped_prefix[
                source_start : source_start + len(swapped_source)
            ] = swapped_source
        else:
            sourced_prefix[
                source_start : source_start + len(source)
            ] = source
            swapped_prefix[
                source_start : source_start + len(swapped_source)
            ] = swapped_source

        sourced = np.concatenate((sourced_prefix, query))
        deleted = np.concatenate((deleted_prefix, query))
        swapped = np.concatenate((swapped_prefix, query))
        expected = (context_length,)
        if (
            sourced.shape != expected
            or deleted.shape != expected
            or swapped.shape != expected
        ):
            raise RuntimeError("probe context length drift")
        sourced_rows.append(sourced)
        deleted_rows.append(deleted)
        swapped_rows.append(swapped)
        gold_ids.append(gold_id)
        alternate_ids.append(alternate_id)
        fractions.append(float(fraction))
        densities.append(int(density))
        keys.append(key)

    return ProbeSet(
        sourced=np.stack(sourced_rows),
        deleted=np.stack(deleted_rows),
        swapped=np.stack(swapped_rows),
        gold=np.asarray(gold_ids, dtype=np.int64),
        alternate=np.asarray(alternate_ids, dtype=np.int64),
        source_fraction=np.asarray(fractions, dtype=np.float64),
        distractor_count=np.asarray(densities, dtype=np.int64),
        keys=keys,
    )


def checkpoint_inv_freq(
    state: dict[str, torch.Tensor],
    checkpoint: Path,
) -> torch.Tensor:
    values = [
        value.detach().cpu().float().contiguous()
        for name, value in state.items()
        if name.endswith(".rope.inv_freq")
    ]
    if values:
        reference = values[0]
        if any(not torch.equal(reference, value) for value in values[1:]):
            raise RuntimeError("checkpoint contains inconsistent inv_freq")
        return reference
    sidecar = checkpoint.parent / "inv_freq.npy"
    if not sidecar.is_file():
        raise RuntimeError("checkpoint and sidecar both lack inv_freq")
    return torch.from_numpy(
        np.load(sidecar, allow_pickle=False)
    ).float().contiguous()


def load_checkpoint(checkpoint: Path) -> tuple[GPT, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is not a mapping")
    state = payload.get("model")
    metadata = payload.get("metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise RuntimeError("checkpoint lacks model state or metadata")
    config = metadata.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint metadata lacks model_config")
    inv_freq = checkpoint_inv_freq(state, checkpoint)
    model = GPT(config, inv_freq)
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [
        name for name in missing if not name.endswith(".rope.inv_freq")
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    model.lm_head.weight = model.embedding.weight
    return model, metadata


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        output_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("the small-model probe expects bias-free linears")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        if output_mask is None:
            self.register_buffer("output_mask", None)
        else:
            if tuple(output_mask.shape) != (base.out_features,):
                raise ValueError("LoRA output mask shape mismatch")
            self.register_buffer(
                "output_mask", output_mask.to(dtype=torch.float32)
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(value, self.a), self.b)
        if self.output_mask is not None:
            update = update * self.output_mask.to(
                device=update.device, dtype=update.dtype
            )
        return self.base(value) + update * self.scale


class ReadoutAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.scale = float(alpha) / float(rank)
        self.a = nn.Linear(hidden_size, rank, bias=False)
        self.b = nn.Linear(rank, vocab_size, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(hidden)) * self.scale


def install_adaptation(
    model: GPT,
    adaptation: str,
    rank: int,
    alpha: float,
) -> ReadoutAdapter | None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    use_qv = adaptation.startswith("qv_")
    use_qkvo = adaptation.startswith("qkvo_")
    if use_qv or use_qkvo:
        hidden = int(model.config["hidden_size"])
        for block in model.blocks:
            mask = None
            if use_qv:
                mask = torch.ones(3 * hidden)
                mask[hidden : 2 * hidden] = 0
            block.attention.qkv = LoRALinear(
                block.attention.qkv,
                rank=rank,
                alpha=alpha,
                output_mask=mask,
            )
            if use_qkvo:
                block.attention.output = LoRALinear(
                    block.attention.output,
                    rank=rank,
                    alpha=alpha,
                )
    readout = None
    if "readout" in adaptation:
        readout = ReadoutAdapter(
            hidden_size=int(model.config["hidden_size"]),
            vocab_size=int(model.config["vocab_size"]),
            rank=rank,
            alpha=alpha,
        )
    return readout


def trainable_parameters(
    model: GPT,
    readout: ReadoutAdapter | None,
) -> list[nn.Parameter]:
    values = [p for p in model.parameters() if p.requires_grad]
    if readout is not None:
        values.extend(p for p in readout.parameters() if p.requires_grad)
    return values


def forward_hidden(
    model: GPT,
    token_ids: torch.Tensor,
    *,
    all_positions: bool,
) -> torch.Tensor:
    hidden = model.embedding(token_ids)
    if hidden.is_cuda and torch.is_autocast_enabled():
        hidden = hidden.to(dtype=torch.get_autocast_dtype("cuda"))
    for block in model.blocks:
        hidden = block(hidden)
    if not all_positions:
        hidden = hidden[:, -1, :]
    return model.final_norm(hidden)


def logits_from_hidden(
    model: GPT,
    hidden: torch.Tensor,
    readout: ReadoutAdapter | None,
) -> torch.Tensor:
    logits = F.linear(hidden, model.lm_head.weight)
    if readout is not None:
        logits = logits + readout(hidden)
    return logits


def forward_layer_last_hidden(
    model: GPT,
    token_ids: torch.Tensor,
) -> list[torch.Tensor]:
    hidden = model.embedding(token_ids)
    if hidden.is_cuda and torch.is_autocast_enabled():
        hidden = hidden.to(dtype=torch.get_autocast_dtype("cuda"))
    layers = [model.final_norm(hidden[:, -1, :])]
    for block in model.blocks:
        hidden = block(hidden)
        layers.append(model.final_norm(hidden[:, -1, :]))
    return layers


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
) -> float:
    if step <= warmup_steps:
        return peak_lr * float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.1 + peak_lr * 0.9 * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def rank_of(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = logits.gather(-1, labels[:, None])
    return 1 + (logits > target).sum(dim=-1)


def train(
    *,
    model: GPT,
    readout: ReadoutAdapter | None,
    data: ProbeSet,
    adaptation: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    parameters = trainable_parameters(model, readout)
    if not parameters:
        return {
            "steps": 0,
            "trainable_parameters": 0,
            "objective": "none",
        }
    objective = "full_token" if adaptation.endswith("_full") else "answer_only"
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 7_919)
    model.train()
    if readout is not None:
        readout.train()
    started = time.perf_counter()
    recent: list[float] = []
    for step in range(1, int(steps) + 1):
        indices = torch.randint(
            len(data.gold),
            (int(batch_size),),
            generator=generator,
        ).numpy()
        context = torch.from_numpy(data.sourced[indices]).to(
            "cuda", non_blocking=True
        )
        labels = torch.from_numpy(data.gold[indices]).to(
            "cuda", non_blocking=True
        )
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = forward_hidden(
                model,
                context,
                all_positions=(objective == "full_token"),
            )
            logits = logits_from_hidden(model, hidden, readout)
            if objective == "answer_only":
                loss = F.cross_entropy(logits.float(), labels)
                answer_logits = logits
            else:
                full = torch.cat((context, labels[:, None]), dim=1)
                targets = full[:, 1:]
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                answer_logits = logits[:, -1, :]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        recent.append(float(loss.detach()))
        if step == 1 or step % 25 == 0 or step == int(steps):
            with torch.no_grad():
                ranks = rank_of(answer_logits.float(), labels)
                exact = (ranks == 1).float().mean()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "mean_loss_last_25": float(np.mean(recent[-25:])),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "batch_exact": float(exact),
                "batch_median_rank": float(torch.median(ranks.float())),
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(log_path, row)
    torch.cuda.synchronize()
    return {
        "steps": int(steps),
        "trainable_parameters": int(sum(p.numel() for p in parameters)),
        "objective": objective,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def scalar_alpha_interval(
    logits: torch.Tensor,
    delta: torch.Tensor,
    gold: int,
) -> tuple[float, float] | None:
    z = logits.float()
    d = delta.float()
    slopes = d[gold] - d
    gaps = z - z[gold]
    competitor = torch.ones_like(slopes, dtype=torch.bool)
    competitor[gold] = False
    zero = competitor & (slopes == 0)
    if bool((zero & (gaps > 0)).any()):
        return None
    positive = competitor & (slopes > 0)
    negative = competitor & (slopes < 0)
    lower = 0.0
    upper = float("inf")
    if bool(positive.any()):
        lower = max(lower, float((gaps[positive] / slopes[positive]).max()))
    if bool(negative.any()):
        upper = float((gaps[negative] / slopes[negative]).min())
    lower = max(0.0, lower)
    return (lower, upper) if lower <= upper else None


def score_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = logits.gather(-1, labels[:, None]).squeeze(-1)
    nll = torch.logsumexp(logits, dim=-1) - target
    rank = 1 + (logits > target[:, None]).sum(dim=-1)
    exact = rank == 1
    return nll, rank, exact


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_mean(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return float(np.mean(values))

    feasible = [row for row in rows if row["scalar_alpha_feasible"]]
    return {
        "n": len(rows),
        "mean_answer_nll": finite_mean("answer_nll"),
        "median_answer_rank": float(
            np.median([row["answer_rank"] for row in rows])
        ),
        "next_token_exact_match": finite_mean("exact"),
        "mean_source_deletion_nll_gap": finite_mean(
            "source_deletion_nll_gap"
        ),
        "median_source_delta_rank": float(
            np.median([row["source_delta_rank"] for row in rows])
        ),
        "mean_gold_source_delta_logit": finite_mean(
            "gold_source_delta_logit"
        ),
        "mean_swap_follow_score": finite_mean("swap_follow_score"),
        "swap_follow_positive_fraction": float(
            np.mean([row["swap_follow_score"] > 0 for row in rows])
        ),
        "scalar_alpha_feasible_fraction": len(feasible) / len(rows),
        "median_required_alpha": (
            float(
                np.median(
                    [row["scalar_alpha_lower"] for row in feasible]
                )
            )
            if feasible
            else None
        ),
        "deleted_exact_match": finite_mean("deleted_exact"),
    }


@torch.no_grad()
def evaluate(
    *,
    model: GPT,
    readout: ReadoutAdapter | None,
    data_by_length: dict[int, ProbeSet],
    answer_token_ids: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
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
                source_hidden = forward_hidden(
                    model, source, all_positions=False
                )
                deleted_hidden = forward_hidden(
                    model, deleted, all_positions=False
                )
                swapped_hidden = forward_hidden(
                    model, swapped, all_positions=False
                )
                source_logits = logits_from_hidden(
                    model, source_hidden, readout
                ).float()
                deleted_logits = logits_from_hidden(
                    model, deleted_hidden, readout
                ).float()
                swapped_logits = logits_from_hidden(
                    model, swapped_hidden, readout
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
            candidate_target = source_gold
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
                            candidate_target[offset]
                            - candidate_other[offset]
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


@torch.no_grad()
def layer_trace(
    *,
    model: GPT,
    readout: ReadoutAdapter | None,
    data_by_length: dict[int, ProbeSet],
    examples: int,
) -> list[dict[str, Any]]:
    if examples <= 0:
        return []
    model.eval()
    if readout is not None:
        readout.eval()
    output: list[dict[str, Any]] = []
    for length, data in sorted(data_by_length.items()):
        count = min(int(examples), len(data.gold))
        source = torch.from_numpy(data.sourced[:count]).to("cuda")
        deleted = torch.from_numpy(data.deleted[:count]).to("cuda")
        labels = torch.from_numpy(data.gold[:count]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            source_layers = forward_layer_last_hidden(model, source)
            deleted_layers = forward_layer_last_hidden(model, deleted)
            for layer, (source_hidden, deleted_hidden) in enumerate(
                zip(source_layers, deleted_layers)
            ):
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
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation", choices=ADAPTATIONS, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-examples", type=int, default=2_048)
    parser.add_argument("--train-length", type=int, default=256)
    parser.add_argument("--eval-examples-per-cell", type=int, default=8)
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--layer-trace-examples", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer_path.resolve()
    validation_path = args.validation.resolve()
    seed_everything(int(args.seed))
    runtime = configure_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    answers = one_token_answers(tokenizer)
    validation = np.load(
        validation_path, mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    if validation.dtype != np.int64:
        raise RuntimeError("validation filler must be int64")

    train_data = build_probe_set(
        tokenizer=tokenizer,
        filler=validation,
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
            filler=validation,
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
        set(train_data.keys) & set(local.keys) for local in eval_data.values()
    ):
        raise RuntimeError("train/evaluation key overlap")

    model, metadata = load_checkpoint(checkpoint)
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
    train_receipt = train(
        model=model,
        readout=readout,
        data=train_data,
        adaptation=args.adaptation,
        steps=effective_steps,
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        seed=int(args.seed),
        log_path=output / "train_log.jsonl",
    )
    rows, summary = evaluate(
        model=model,
        readout=readout,
        data_by_length=eval_data,
        answer_token_ids=[token_id for _, token_id in answers],
        batch_size=int(args.batch_size),
    )
    trace = layer_trace(
        model=model,
        readout=readout,
        data_by_length=eval_data,
        examples=int(args.layer_trace_examples),
    )
    receipt = {
        "status": "SMALL_MODEL_LORA_CONVERSION_COMPLETE",
        "metric_boundary": (
            "held-out teacher-forced one-token associative retrieval; "
            "not general downstream capability"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "validation_sha256": sha256_file(validation_path),
        "adaptation": args.adaptation,
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
            "batch_size": int(args.batch_size),
            "warmup_steps": int(args.warmup_steps),
            "train_dataset_sha256": train_data.digest(),
            "eval_dataset_sha256": {
                str(length): data.digest()
                for length, data in sorted(eval_data.items())
            },
        },
        "training": train_receipt,
        "summary": summary,
        "layer_trace": trace,
        "rows": rows,
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checkpoint_arm": metadata.get("arm"),
                "adaptation": args.adaptation,
                "training": train_receipt,
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key in {f"L{length}" for length in args.eval_lengths}
                },
                "output": str(output / "results.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
