"""Locked construction and training constants for the 151.9M S1 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


ARMS = ("geo", "cosh", "full_z")
SUPPORTS = (2_048, 500_000)
SEEDS = (42, 137, 256)
TAU = math.sqrt(2.0)


@dataclass(frozen=True)
class S1Spec:
    vocab_size: int = 50_304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3_072
    train_length: int = 2_048
    global_batch_size: int = 32
    micro_batch_size: int = 8
    source_tokens_per_epoch: int = 499_974_144
    epochs: int = 2
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 1_525
    weight_decay: float = 0.01
    allocation_lr_multiplier: float = 1.0
    allocation_weight_decay: float = 0.0
    allocation_projection: str = "none; subtract-mean gauge fixing only"
    max_grad_norm: float = 1.0

    @property
    def rows_per_epoch(self) -> int:
        return self.source_tokens_per_epoch // self.train_length

    @property
    def optimizer_steps(self) -> int:
        return self.rows_per_epoch * self.epochs // self.global_batch_size

    @property
    def train_tokens(self) -> int:
        return self.optimizer_steps * self.global_batch_size * self.train_length

    @property
    def midpoint_step(self) -> int:
        return self.optimizer_steps // 2

    def model_config(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.train_length,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


SPEC = S1Spec()


def geo_table(support: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if int(support) not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    pairs = SPEC.head_dim // 2
    z = torch.linspace(0.0, 1.0, pairs, dtype=torch.float64)
    log_span = (pairs - 1) / pairs * math.log(float(support))
    return torch.exp(-log_span * z).to(dtype=dtype)


def cosh_table(support: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Endpoint-anchored midpoint Cosh quantiles with the Geo sampled support."""
    if int(support) not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    pairs = SPEC.head_dim // 2
    u = (np.arange(pairs, dtype=np.float64) + 0.5) / pairs
    q = 1.0 - np.arcsinh((1.0 - u) * np.sinh(TAU)) / TAU
    z = (q - q[0]) / (q[-1] - q[0])
    log_span = (pairs - 1) / pairs * math.log(float(support))
    return torch.from_numpy(np.exp(-log_span * z)).to(dtype=dtype)


def table_for(arm: str, support: int) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return cosh_table(support) if arm == "cosh" else geo_table(support)


def learning_rate(completed_steps: int) -> float:
    step = int(completed_steps)
    if step < 0 or step >= SPEC.optimizer_steps:
        raise ValueError("step is outside the locked trajectory")
    if step < SPEC.warmup_steps:
        return SPEC.learning_rate * (step + 1) / SPEC.warmup_steps
    progress = (step - SPEC.warmup_steps) / max(
        1, SPEC.optimizer_steps - SPEC.warmup_steps - 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return SPEC.min_learning_rate + (
        SPEC.learning_rate - SPEC.min_learning_rate
    ) * cosine

