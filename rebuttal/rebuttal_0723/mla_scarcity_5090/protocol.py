#!/usr/bin/env python3
"""Frozen protocol for the MLA scarce-channel range-vs-shape experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from scripts.lib.rope.schedules import evq_cosh_phi


ARMS = ("native_geo", "range_matched_uniform", "evq_cosh")
FREQUENCY_PAIRS = (8, 32)
SEEDS = (42, 43, 88)
GATE_SEED = 42
CONFIRMATORY_SEEDS = (43, 88)
TAU = 1.414
BASE = 500_000.0
CHECKPOINT_LABELS = ("100m", "200m", "300m")


@dataclass(frozen=True)
class MLAScarcitySpec:
    train_length: int = 4_096
    requested_train_tokens: int = 300_000_000
    global_batch_size: int = 32
    micro_batch_size: int = 32
    vocab_size: int = 50_304
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2_048
    kv_lora_rank: int = 128
    v_head_dim: int = 64
    rotary_pair_capacity: int = 32
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    eval_lengths: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768)
    eval_tail_tokens: int = 4_096
    selection_anchor_count: int = 16
    test_anchor_count: int = 32
    anchor_seed: int = 20_260_725

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.global_batch_size * self.train_length

    @property
    def optimizer_steps(self) -> int:
        return self.requested_train_tokens // self.tokens_per_optimizer_step

    @property
    def train_tokens(self) -> int:
        return self.optimizer_steps * self.tokens_per_optimizer_step

    @property
    def train_rows(self) -> int:
        return self.optimizer_steps * self.global_batch_size

    @property
    def grad_accum_steps(self) -> int:
        if self.global_batch_size % self.micro_batch_size:
            raise RuntimeError("global batch must be divisible by micro batch")
        return self.global_batch_size // self.micro_batch_size

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.optimizer_steps * self.warmup_fraction))

    @property
    def checkpoint_steps(self) -> dict[str, int]:
        targets = {"100m": 100_000_000, "200m": 200_000_000}
        steps = {
            label: int(round(tokens / self.tokens_per_optimizer_step))
            for label, tokens in targets.items()
        }
        steps["300m"] = self.optimizer_steps
        if not (0 < steps["100m"] < steps["200m"] < steps["300m"]):
            raise RuntimeError("checkpoint stages are not strictly ordered")
        return steps

    @property
    def checkpoint_tokens(self) -> dict[str, int]:
        return {
            label: step * self.tokens_per_optimizer_step
            for label, step in self.checkpoint_steps.items()
        }

    def model_config(self, frequency_pairs: int) -> dict[str, Any]:
        pairs = int(frequency_pairs)
        if pairs not in FREQUENCY_PAIRS:
            raise ValueError(f"unregistered frequency-pair budget: {pairs}")
        if pairs > self.rotary_pair_capacity:
            raise ValueError("active pairs exceed rotary-pair capacity")
        # Keep the MLA architecture and parameter count fixed. Scarcity is
        # induced only by setting unused rotary-pair frequencies to zero,
        # which makes their rotation exactly the identity.
        d_rope = 2 * self.rotary_pair_capacity
        d_nope = self.head_dim - d_rope
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.train_length,
            "seq_len": self.train_length,
            "batch_size": self.global_batch_size,
            "train_tokens": self.train_tokens,
            "lr": self.learning_rate,
            "eval_lengths": list(self.eval_lengths),
            "attn_type": "mla",
            "d_rope": d_rope,
            "d_nope": d_nope,
            "active_frequency_pairs": pairs,
            "v_head_dim": self.v_head_dim,
            "kv_lora_rank": self.kv_lora_rank,
            "passkey_mix_ratio": 0.0,
        }

    def fingerprint(self) -> str:
        payload = {
            **asdict(self),
            "arms": ARMS,
            "frequency_pairs": FREQUENCY_PAIRS,
            "seeds": SEEDS,
            "tau": TAU,
            "base": BASE,
            "checkpoint_steps": self.checkpoint_steps,
            "checkpoint_tokens": self.checkpoint_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


SPEC = MLAScarcitySpec()


def schedule_phi(
    arm: str,
    frequency_pairs: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unregistered arm: {arm}")
    pairs = int(frequency_pairs)
    if pairs not in FREQUENCY_PAIRS:
        raise ValueError(f"unregistered frequency-pair budget: {pairs}")
    native = torch.arange(pairs, dtype=dtype) / float(pairs)
    evq = evq_cosh_phi(pairs, tau=TAU, midpoint=True, dtype=dtype)
    if arm == "native_geo":
        phi = native
        metadata = {
            "family": "geometric",
            "grid": "native endpoint u=k/K",
            "role": "native Std-RoPE reference",
        }
    elif arm == "range_matched_uniform":
        phi = torch.linspace(
            float(evq[0]), float(evq[-1]), pairs, dtype=dtype
        )
        metadata = {
            "family": "uniform_log_frequency",
            "grid": "uniform interior with EVQ endpoints",
            "role": "pure range control",
            "matched_to": "evq_cosh",
        }
    else:
        phi = evq
        metadata = {
            "family": "evq_cosh",
            "grid": "midpoint u=(k+0.5)/K",
            "tau": TAU,
            "role": "range plus allocation-shape intervention",
        }
    return phi.contiguous(), {
        **metadata,
        "frequency_pairs": pairs,
        "active_rotary_dimensions": 2 * pairs,
        "model_d_rope": 2 * SPEC.rotary_pair_capacity,
        "phi_min": float(phi[0]),
        "phi_max": float(phi[-1]),
        "phi_span": float(phi[-1] - phi[0]),
    }


def training_inv_freq(
    arm: str,
    frequency_pairs: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    phi, metadata = schedule_phi(
        arm, frequency_pairs, dtype=torch.float64
    )
    active = torch.pow(torch.tensor(BASE, dtype=torch.float64), -phi)
    capacity = SPEC.rotary_pair_capacity
    inv = torch.zeros(capacity, dtype=torch.float64)
    inv[: int(frequency_pairs)] = active
    return inv.to(dtype=dtype).contiguous(), {
        **metadata,
        "rotary_pair_capacity": capacity,
        "inactive_identity_pairs": capacity - int(frequency_pairs),
        "inactive_frequency_value": 0.0,
        "architecture_control": (
            "d_rope=64 and d_nope=0 fixed; inactive pairs have zero "
            "frequency and therefore identity rotation"
        ),
    }


def learning_rate_for_step(step: int) -> float:
    current = int(step)
    if current < 0 or current >= SPEC.optimizer_steps:
        raise ValueError(f"optimizer step out of range: {current}")
    if current < SPEC.warmup_steps:
        return SPEC.learning_rate * (current + 1) / SPEC.warmup_steps
    progress = (current - SPEC.warmup_steps) / max(
        SPEC.optimizer_steps - SPEC.warmup_steps - 1, 1
    )
    return SPEC.min_learning_rate + (
        SPEC.learning_rate - SPEC.min_learning_rate
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
