#!/usr/bin/env python3
"""Matched 50M Native-RoPE versus EVQ-LeRoPE experiment.

The experiment is deliberately narrow:

* train from scratch on 15M FineWeb-Edu tokens at physical length 128;
* compare standard endpoint Native RoPE against paper-grid EVQ-Cosh (tau=5);
* in the EVQ arm, learn one log-scale for every frequency band;
* evaluate paired PPL at 128, 256, and 512 tokens.

The 32 LeRoPE-style scalars are shared across all layers and heads,
represented in log space, receive no weight decay, and are clipped
independently from the model parameters. EVQ-Cosh is their exact
initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.rope.schedules import (
    evq_cosh_inv_freq,
    geometric_inv_freq,
)


METHOD_ID = "50m_native_vs_evq_lerope_full_band_v2"
PREPARED_STATUS = "50M_NATIVE_EVQ_LEROPE_PREPARED_NO_GPU_V1"
READY_STATUS = "50M_NATIVE_EVQ_LEROPE_RUNTIME_READY_V1"
RESULT_STATUS = "50M_NATIVE_EVQ_LEROPE_COMPLETE_V1"
ARMS = ("native", "evq_lerope")


@dataclass(frozen=True)
class Protocol:
    vocab_size: int = 50_304
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2_048
    train_length: int = 128
    train_tokens_requested: int = 15_000_000
    global_batch_sequences: int = 128
    micro_batch_sequences: int = 128
    learning_rate: float = 6e-4
    minimum_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    frequency_gradient_clip: float = 1.0
    frequency_lr_multiplier: float = 1.0
    rope_base: float = 500_000.0
    evq_tau: float = 5.0
    dominant_wavelength_ratio: float = 2.205
    eval_lengths: tuple[int, ...] = (128, 256, 512)
    eval_chunks: int = 32
    seed: int = 42
    eval_seed: int = 9_999

    def validate(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden size must equal heads times head dimension")
        if self.head_dim % 2:
            raise ValueError("head dimension must be even")
        if self.global_batch_sequences % self.micro_batch_sequences:
            raise ValueError("global batch must divide into whole micro batches")
        if self.train_length != 128:
            raise ValueError("registered experiment requires L_train=128")
        if tuple(self.eval_lengths) != (128, 256, 512):
            raise ValueError("registered evaluation lengths are 128/256/512")
        if self.evq_tau != 5.0 or self.rope_base != 500_000.0:
            raise ValueError("registered EVQ identity requires tau=5/base=500K")

    @property
    def train_rows(self) -> int:
        available = self.train_tokens_requested // self.train_length
        return (
            available // self.global_batch_sequences
        ) * self.global_batch_sequences

    @property
    def optimizer_steps(self) -> int:
        return self.train_rows // self.global_batch_sequences

    @property
    def accumulation_steps(self) -> int:
        return self.global_batch_sequences // self.micro_batch_sequences


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_inv_freq(protocol: Protocol) -> torch.Tensor:
    return geometric_inv_freq(
        head_dim=protocol.head_dim,
        base=protocol.rope_base,
        dtype=torch.float64,
    ).float()


def evq_inv_freq(protocol: Protocol) -> torch.Tensor:
    return evq_cosh_inv_freq(
        head_dim=protocol.head_dim,
        tau=protocol.evq_tau,
        base=protocol.rope_base,
        midpoint=True,
        dtype=torch.float64,
    ).float()


def selected_lerope_band(
    protocol: Protocol,
    frequencies: torch.Tensor | None = None,
) -> dict[str, float | int]:
    values = (
        evq_inv_freq(protocol)
        if frequencies is None
        else frequencies.detach().cpu().float()
    )
    target = (
        protocol.dominant_wavelength_ratio * protocol.train_length
    )
    wavelengths = (2.0 * math.pi) / values.double()
    index = int(
        torch.argmin(torch.abs(torch.log(wavelengths / target)))
    )
    return {
        "index": index,
        "target_wavelength": float(target),
        "initial_wavelength": float(wavelengths[index]),
        "initial_to_target_ratio": float(wavelengths[index] / target),
    }


class SelectiveRotaryEmbedding(nn.Module):
    """Native fixed RoPE or EVQ with shared per-band log scalars."""

    def __init__(self, protocol: Protocol, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.protocol = protocol
        self.arm = arm
        base = (
            native_inv_freq(protocol)
            if arm == "native"
            else evq_inv_freq(protocol)
        )
        self.register_buffer("base_inv_freq", base, persistent=True)
        selection = selected_lerope_band(protocol, base)
        self.selected_band = int(selection["index"])
        mask = (
            torch.ones_like(base)
            if arm == "evq_lerope"
            else torch.zeros_like(base)
        )
        self.register_buffer("learnable_mask", mask, persistent=True)
        self.log_frequency_scale = nn.Parameter(
            torch.zeros_like(base, dtype=torch.float32),
            requires_grad=(arm == "evq_lerope"),
        )

    def current_inv_freq(self) -> torch.Tensor:
        return self.base_inv_freq * torch.exp(
            self.learnable_mask * self.log_frequency_scale
        )

    def forward(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            int(sequence_length),
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(
            positions,
            self.current_inv_freq().to(device=device, dtype=torch.float32),
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    return value * cosine + rotate_half(value) * sine


class RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + 1e-6
        ).to(value.dtype)
        return normalized * self.weight


class Attention(nn.Module):
    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.protocol = protocol
        self.qkv = nn.Linear(
            protocol.hidden_size,
            3 * protocol.hidden_size,
            bias=False,
        )
        self.output = nn.Linear(
            protocol.hidden_size,
            protocol.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        packed = self.qkv(hidden).view(
            batch,
            length,
            3,
            self.protocol.num_heads,
            self.protocol.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = packed.unbind(0)
        cosine = cosine[None, None]
        sine = sine[None, None]
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        return self.output(
            attended.transpose(1, 2).reshape(batch, length, -1)
        )


class MLP(nn.Module):
    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.gate = nn.Linear(
            protocol.hidden_size,
            protocol.intermediate_size,
            bias=False,
        )
        self.up = nn.Linear(
            protocol.hidden_size,
            protocol.intermediate_size,
            bias=False,
        )
        self.down = nn.Linear(
            protocol.intermediate_size,
            protocol.hidden_size,
            bias=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class Block(nn.Module):
    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.input_norm = RMSNorm(protocol.hidden_size)
        self.attention = Attention(protocol)
        self.post_attention_norm = RMSNorm(protocol.hidden_size)
        self.mlp = MLP(protocol)

    def forward(
        self,
        hidden: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.input_norm(hidden), cosine, sine
        )
        return hidden + self.mlp(self.post_attention_norm(hidden))


class GPT(nn.Module):
    def __init__(self, protocol: Protocol, arm: str) -> None:
        super().__init__()
        self.protocol = protocol
        self.arm = arm
        self.embedding = nn.Embedding(
            protocol.vocab_size, protocol.hidden_size
        )
        self.rope = SelectiveRotaryEmbedding(protocol, arm)
        self.blocks = nn.ModuleList(
            [Block(protocol) for _ in range(protocol.num_layers)]
        )
        self.final_norm = RMSNorm(protocol.hidden_size)
        self.lm_head = nn.Linear(
            protocol.hidden_size,
            protocol.vocab_size,
            bias=False,
        )
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)
        residual_scale = 1.0 / math.sqrt(2 * protocol.num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output.weight,
                std=0.02 * residual_scale,
            )
            nn.init.normal_(
                block.mlp.down.weight,
                std=0.02 * residual_scale,
            )

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        cosine, sine = self.rope(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=hidden.dtype,
        )
        for block in self.blocks:
            hidden = block(hidden, cosine, sine)
        return self.lm_head(self.final_norm(hidden))


def trainable_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name == "rope.log_frequency_scale":
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_rows(path: Path, protocol: Protocol) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("training artifact must be a tensor")
    value = value.to(dtype=torch.long)
    if value.ndim == 1:
        usable = value.numel() // protocol.train_length
        value = value[: usable * protocol.train_length].view(
            usable, protocol.train_length
        )
    if (
        value.ndim != 2
        or value.shape[1] != protocol.train_length
        or value.shape[0] < protocol.train_rows
    ):
        raise RuntimeError(
            "training tensor must contain enough [rows, 128] sequences"
        )
    return value[: protocol.train_rows].contiguous()


def load_validation_tokens(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("validation artifact must be a tensor")
    return value.to(dtype=torch.long).reshape(-1).contiguous()


def evaluation_offsets(
    token_count: int,
    protocol: Protocol,
) -> dict[int, list[int]]:
    generator = np.random.RandomState(protocol.eval_seed)
    result: dict[int, list[int]] = {}
    for length in protocol.eval_lengths:
        maximum = int(token_count) - int(length)
        if maximum <= protocol.eval_chunks:
            raise RuntimeError("validation artifact is too small")
        result[int(length)] = sorted(
            int(value)
            for value in generator.choice(
                maximum,
                size=protocol.eval_chunks,
                replace=False,
            )
        )
    return result


def runtime_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def optimizer_for(
    model: GPT,
    protocol: Protocol,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], list[nn.Parameter]]:
    frequency = [model.rope.log_frequency_scale]
    frequency_ids = {id(parameter) for parameter in frequency}
    ordinary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in frequency_ids
    ]
    groups: list[dict[str, Any]] = [{
        "params": ordinary,
        "weight_decay": protocol.weight_decay,
        "lr_scale": 1.0,
    }]
    if model.arm == "evq_lerope":
        groups.append({
            "params": frequency,
            "weight_decay": 0.0,
            "lr_scale": protocol.frequency_lr_multiplier,
        })
    optimizer = torch.optim.AdamW(
        groups,
        lr=protocol.learning_rate,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )
    return optimizer, ordinary, (
        frequency if model.arm == "evq_lerope" else []
    )


def learning_rate_at(step: int, protocol: Protocol) -> float:
    warmup = max(
        1, int(protocol.optimizer_steps * protocol.warmup_ratio)
    )
    if step <= warmup:
        return protocol.learning_rate * step / warmup
    progress = (step - warmup) / max(
        protocol.optimizer_steps - warmup, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = protocol.learning_rate * protocol.minimum_lr_ratio
    return floor + (protocol.learning_rate - floor) * cosine


def train_arm(
    *,
    arm: str,
    protocol: Protocol,
    train_rows: torch.Tensor,
    device: torch.device,
    output: Path,
    save_checkpoint: bool = True,
) -> tuple[GPT, dict[str, Any]]:
    set_seed(protocol.seed)
    model = GPT(protocol, arm).to(device)
    initial_weight_hash = trainable_state_sha256(model)
    initial_frequency = model.rope.current_inv_freq().detach().cpu()
    optimizer, ordinary, frequency = optimizer_for(
        model, protocol, device
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(protocol.seed + 1_000_003)
    permutation = torch.randperm(
        protocol.train_rows,
        generator=permutation_generator,
    )
    started = time.perf_counter()
    losses: list[float] = []
    first_frequency_gradient = None
    model.train()
    for step in range(1, protocol.optimizer_steps + 1):
        lr = learning_rate_at(step, protocol)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group["lr_scale"])
        optimizer.zero_grad(set_to_none=True)
        global_indices = permutation[
            (step - 1) * protocol.global_batch_sequences :
            step * protocol.global_batch_sequences
        ]
        step_losses: list[float] = []
        for slot in range(protocol.accumulation_steps):
            indices = global_indices[
                slot * protocol.micro_batch_sequences :
                (slot + 1) * protocol.micro_batch_sequences
            ]
            batch = train_rows[indices].to(device)
            with autocast_context(device):
                logits = model(batch[:, :-1])
                raw_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1),
                )
                loss = raw_loss / protocol.accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss in {arm} at step {step}")
            loss.backward()
            step_losses.append(float(raw_loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(
            ordinary, protocol.gradient_clip
        )
        if frequency:
            gradient = frequency[0].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("LeRoPE frequency gradient is invalid")
            if step == 1:
                first_frequency_gradient = {
                    "l2_norm": float(
                        torch.linalg.vector_norm(gradient.detach()).cpu()
                    ),
                    "max_abs": float(
                        gradient.detach().abs().max().cpu()
                    ),
                    "per_band": gradient.detach().cpu().tolist(),
                }
                if first_frequency_gradient["l2_norm"] == 0.0:
                    raise RuntimeError(
                        "LeRoPE frequency-gradient norm is zero on the first step"
                    )
            torch.nn.utils.clip_grad_norm_(
                frequency, protocol.frequency_gradient_clip
            )
        optimizer.step()
        mean_loss = float(np.mean(step_losses))
        losses.append(mean_loss)
        if step == 1 or step % 50 == 0 or step == protocol.optimizer_steps:
            elapsed = time.perf_counter() - started
            record = {
                "arm": arm,
                "step": step,
                "steps": protocol.optimizer_steps,
                "loss": mean_loss,
                "mean_loss_last_20": float(np.mean(losses[-20:])),
                "learning_rate": lr,
                "frequency_learning_rate": (
                    lr * protocol.frequency_lr_multiplier
                ),
                "elapsed_seconds": elapsed,
                "log_frequency_scale_l2": float(
                    torch.linalg.vector_norm(
                        model.rope.log_frequency_scale.detach()
                    ).cpu()
                ),
                "log_frequency_scale_max_abs": float(
                    model.rope.log_frequency_scale.detach().abs().max().cpu()
                ),
                "selected_log_frequency_scale": float(
                    model.rope.log_frequency_scale[
                        model.rope.selected_band
                    ].detach().cpu()
                ),
            }
            with (output / "train_log.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    elapsed = time.perf_counter() - started
    final_frequency = model.rope.current_inv_freq().detach().cpu()
    receipt = {
        "arm": arm,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer": {
            "name": "AdamW",
            "fused": device.type == "cuda",
            "betas": [0.9, 0.95],
        },
        "initial_weight_sha256": initial_weight_hash,
        "initial_inv_freq_sha256_float32": tensor_sha256(
            initial_frequency.float()
        ),
        "final_inv_freq_sha256_float32": tensor_sha256(
            final_frequency.float()
        ),
        "selected_band": model.rope.selected_band,
        "selected_initial_wavelength": float(
            2.0 * math.pi / initial_frequency[model.rope.selected_band]
        ),
        "selected_final_wavelength": float(
            2.0 * math.pi / final_frequency[model.rope.selected_band]
        ),
        "log_frequency_scale": (
            model.rope.log_frequency_scale.detach().cpu().tolist()
        ),
        "log_frequency_scale_l2": float(
            torch.linalg.vector_norm(
                model.rope.log_frequency_scale.detach()
            ).cpu()
        ),
        "log_frequency_scale_max_abs": float(
            model.rope.log_frequency_scale.detach().abs().max().cpu()
        ),
        "selected_log_frequency_scale": float(
            model.rope.log_frequency_scale[
                model.rope.selected_band
            ].detach().cpu()
        ),
        "final_frequency_strictly_decreasing": bool(
            torch.all(final_frequency[:-1] > final_frequency[1:])
        ),
        "first_frequency_gradient": first_frequency_gradient,
        "final_mean_loss_last_20": float(np.mean(losses[-20:])),
        "elapsed_seconds": elapsed,
        "checkpoint": None,
        "checkpoint_sha256": None,
    }
    if save_checkpoint:
        checkpoint = output / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_sha256"] = file_sha256(checkpoint)
    return model, receipt


@torch.no_grad()
def evaluate_arm(
    *,
    model: GPT,
    validation_tokens: torch.Tensor,
    offsets: dict[int, list[int]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for length, starts in offsets.items():
        losses: list[float] = []
        for start in starts:
            row = validation_tokens[start : start + length].to(device)
            with autocast_context(device):
                logits = model(row[:-1][None, :])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    row[1:].reshape(-1),
                )
            losses.append(float(loss.detach().cpu()))
        mean_nll = float(np.mean(losses))
        result[str(length)] = {
            "chunks": len(losses),
            "offsets": starts,
            "per_chunk_nll": losses,
            "mean_nll": mean_nll,
            "ppl": float(math.exp(mean_nll)),
        }
    return result


def protocol_receipt(
    *,
    protocol: Protocol,
    train_data: Path,
    val_data: Path,
) -> dict[str, Any]:
    protocol.validate()
    native = native_inv_freq(protocol)
    evq = evq_inv_freq(protocol)
    selection = selected_lerope_band(protocol, evq)
    protocol_values = asdict(protocol)
    protocol_values["eval_lengths"] = list(protocol.eval_lengths)
    schedule_path = (
        Path(__file__).resolve().parents[3]
        / "scripts/lib/rope/schedules.py"
    ).resolve()
    return {
        "method_id": METHOD_ID,
        "protocol": protocol_values,
        "derived": {
            "train_rows": protocol.train_rows,
            "optimizer_steps": protocol.optimizer_steps,
            "accumulation_steps": protocol.accumulation_steps,
            "processed_storage_tokens": (
                protocol.train_rows * protocol.train_length
            ),
            "supervised_next_tokens": (
                protocol.train_rows * (protocol.train_length - 1)
            ),
            "selected_lerope_band": selection,
        },
        "frequency_identity": {
            "native_definition": "standard_endpoint_geometric_rope",
            "native_sha256_float32": tensor_sha256(native),
            "evq_definition": "paper_midpoint_evq_cosh",
            "evq_tau": protocol.evq_tau,
            "evq_sha256_float32": tensor_sha256(evq),
            "evq_is_non_geometric": bool(
                not torch.allclose(
                    torch.diff(torch.log(evq.double())),
                    torch.diff(torch.log(evq.double()))[:1].expand(
                        evq.numel() - 1
                    ),
                    atol=1e-12,
                    rtol=0.0,
                )
            ),
        },
        "data": {
            "train_path": str(train_data.resolve()),
            "train_sha256": file_sha256(train_data.resolve()),
            "validation_path": str(val_data.resolve()),
            "validation_sha256": file_sha256(val_data.resolve()),
        },
        "code": {
            "trainer_path": str(Path(__file__).resolve()),
            "trainer_sha256": file_sha256(Path(__file__).resolve()),
            "schedule_path": str(
                schedule_path
            ),
            "schedule_sha256": file_sha256(schedule_path),
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare", "smoke", "train", "train-arm", "aggregate"),
        required=True,
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path, required=True)
    parser.add_argument("--runtime-ready-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    return parser.parse_args()


def verify_prepared(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if (
        observed.get("status") != PREPARED_STATUS
        or observed.get("contract") != expected
    ):
        raise RuntimeError("prepared receipt drift")
    return observed


def paired_result(
    *,
    contract: dict[str, Any],
    prepared: dict[str, Any],
    runtime_ready_receipt: Path,
    device: str,
    results: dict[str, Any],
) -> dict[str, Any]:
    initial_hashes = {
        result["training"]["initial_weight_sha256"]
        for result in results.values()
    }
    if len(initial_hashes) != 1:
        raise RuntimeError("matched model initialization drift")
    deltas = {}
    all_three_win = True
    for length in contract["protocol"]["eval_lengths"]:
        key = str(length)
        native = results["native"]["evaluation"][key]
        candidate = results["evq_lerope"]["evaluation"][key]
        delta = candidate["mean_nll"] - native["mean_nll"]
        ppl_delta = candidate["ppl"] - native["ppl"]
        wins = delta < 0.0
        all_three_win = all_three_win and wins
        deltas[key] = {
            "mean_nll_delta_evq_lerope_minus_native": delta,
            "ppl_delta_evq_lerope_minus_native": ppl_delta,
            "evq_lerope_wins": wins,
        }
    return {
        "status": RESULT_STATUS,
        "method_id": METHOD_ID,
        "contract": contract,
        "prepared_receipt": prepared,
        "runtime_ready_receipt_sha256": file_sha256(
            runtime_ready_receipt
        ),
        "device": device,
        "arms": results,
        "paired_deltas": deltas,
        "decision": {
            "gate": "EVQ-LeRoPE PPL lower than Native at 128/256/512",
            "all_three_lengths_win": all_three_win,
            "verdict": (
                "PASS_RESEARCH_LARGER_MODEL_APPLICATION"
                if all_three_win
                else "FAIL_STOP_DO_NOT_PROMOTE"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    protocol = Protocol(seed=args.seed)
    protocol.validate()
    train_path = args.train_data.resolve()
    validation_path = args.validation_data.resolve()
    contract = protocol_receipt(
        protocol=protocol,
        train_data=train_path,
        val_data=validation_path,
    )
    if args.mode == "prepare":
        load_training_rows(train_path, protocol)
        validation = load_validation_tokens(validation_path)
        evaluation_offsets(validation.numel(), protocol)
        receipt = {
            "status": PREPARED_STATUS,
            "classification": "OFFLINE_PREPARATION_NOT_RESULT",
            "contract": contract,
        }
        if args.prepared_receipt.exists():
            raise FileExistsError(args.prepared_receipt)
        atomic_json(args.prepared_receipt.resolve(), receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    prepared = verify_prepared(
        args.prepared_receipt.resolve(), contract
    )
    if args.mode == "aggregate":
        if args.arm is not None:
            raise RuntimeError("aggregate mode does not accept --arm")
        if args.runtime_ready_receipt is None:
            raise RuntimeError(
                "aggregate mode requires --runtime-ready-receipt"
            )
        ready = json.loads(
            args.runtime_ready_receipt.resolve().read_text(
                encoding="utf-8"
            )
        )
        if (
            ready.get("status") != READY_STATUS
            or ready.get("contract") != contract
            or ready.get("prepared_receipt_sha256")
            != file_sha256(args.prepared_receipt.resolve())
        ):
            raise RuntimeError("runtime READY receipt drift")
        output = args.output.resolve()
        results: dict[str, Any] = {}
        for arm in ARMS:
            arm_result_path = output / arm / "results.json"
            arm_result = json.loads(
                arm_result_path.read_text(encoding="utf-8")
            )
            if (
                arm_result.get("status") != RESULT_STATUS
                or arm_result.get("arm") != arm
                or arm_result.get("contract") != contract
            ):
                raise RuntimeError(f"{arm} result contract drift")
            results[arm] = {
                "training": arm_result["training"],
                "evaluation": arm_result["evaluation"],
            }
        final = paired_result(
            contract=contract,
            prepared=prepared,
            runtime_ready_receipt=args.runtime_ready_receipt.resolve(),
            device=str(ready["device"]),
            results=results,
        )
        final_path = output / "results.json"
        if final_path.exists():
            raise FileExistsError(final_path)
        atomic_json(final_path, final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return

    device = runtime_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    train_rows = load_training_rows(train_path, protocol)
    validation = load_validation_tokens(validation_path)
    offsets = evaluation_offsets(validation.numel(), protocol)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    if args.mode == "smoke":
        checks: dict[str, Any] = {}
        initial_hashes = {}
        for arm in ARMS:
            arm_output = output / arm
            arm_output.mkdir()
            small_protocol = replace(
                protocol,
                train_tokens_requested=(
                    protocol.global_batch_sequences
                    * protocol.train_length
                ),
            )
            model, training = train_arm(
                arm=arm,
                protocol=small_protocol,
                train_rows=train_rows[: small_protocol.train_rows],
                device=device,
                output=arm_output,
                save_checkpoint=False,
            )
            evaluation = evaluate_arm(
                model=model,
                validation_tokens=validation,
                offsets={
                    length: values[:1]
                    for length, values in offsets.items()
                },
                device=device,
            )
            initial_hashes[arm] = training["initial_weight_sha256"]
            checks[arm] = {
                "training": training,
                "evaluation_finite": all(
                    math.isfinite(cell["mean_nll"])
                    for cell in evaluation.values()
                ),
            }
            del model
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError("matched model initialization drift")
        receipt = {
            "status": READY_STATUS,
            "classification": "RUNTIME_SMOKE_NOT_RESULT",
            "contract": contract,
            "prepared_receipt_sha256": file_sha256(
                args.prepared_receipt.resolve()
            ),
            "device": str(device),
            "runtime": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_capability": list(
                    torch.cuda.get_device_capability()
                ),
                "gpu_name": torch.cuda.get_device_name(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "flash_sdp_enabled": (
                    torch.backends.cuda.flash_sdp_enabled()
                ),
                "mem_efficient_sdp_enabled": (
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "math_sdp_enabled": (
                    torch.backends.cuda.math_sdp_enabled()
                ),
                "cudnn_sdp_enabled": (
                    torch.backends.cuda.cudnn_sdp_enabled()
                    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
                    else None
                ),
                "tf32_matmul": (
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "allocator_config": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF"
                ),
            },
            "checks": checks,
        }
        atomic_json(output / "runtime_ready.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    if args.runtime_ready_receipt is None:
        raise RuntimeError("train mode requires --runtime-ready-receipt")
    ready = json.loads(
        args.runtime_ready_receipt.resolve().read_text(encoding="utf-8")
    )
    if (
        ready.get("status") != READY_STATUS
        or ready.get("contract") != contract
        or ready.get("prepared_receipt_sha256")
        != file_sha256(args.prepared_receipt.resolve())
    ):
        raise RuntimeError("runtime READY receipt drift")

    if args.mode == "train-arm":
        if args.arm is None:
            raise RuntimeError("train-arm mode requires --arm")
        model, training = train_arm(
            arm=args.arm,
            protocol=protocol,
            train_rows=train_rows,
            device=device,
            output=output,
        )
        evaluation = evaluate_arm(
            model=model,
            validation_tokens=validation,
            offsets=offsets,
            device=device,
        )
        result = {
            "status": RESULT_STATUS,
            "arm": args.arm,
            "contract": contract,
            "runtime_ready_receipt_sha256": file_sha256(
                args.runtime_ready_receipt.resolve()
            ),
            "training": training,
            "evaluation": evaluation,
        }
        atomic_json(output / "results.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.arm is not None:
        raise RuntimeError("--arm is only valid with train-arm mode")

    results: dict[str, Any] = {}
    initial_hashes: dict[str, str] = {}
    for arm in ARMS:
        arm_output = output / arm
        arm_output.mkdir()
        model, training = train_arm(
            arm=arm,
            protocol=protocol,
            train_rows=train_rows,
            device=device,
            output=arm_output,
        )
        evaluation = evaluate_arm(
            model=model,
            validation_tokens=validation,
            offsets=offsets,
            device=device,
        )
        initial_hashes[arm] = training["initial_weight_sha256"]
        results[arm] = {
            "training": training,
            "evaluation": evaluation,
        }
        atomic_json(
            arm_output / "results.json",
            {
                "status": RESULT_STATUS,
                "arm": arm,
                "contract": contract,
                "training": training,
                "evaluation": evaluation,
            },
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    final = paired_result(
        contract=contract,
        prepared=prepared,
        runtime_ready_receipt=args.runtime_ready_receipt.resolve(),
        device=str(device),
        results=results,
    )
    atomic_json(output / "results.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
