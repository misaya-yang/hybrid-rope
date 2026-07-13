#!/usr/bin/env python3
"""Immutable protocol identities for the 151.9M paired scratch experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from scripts.lib.rope.schedules import evq_cosh_inv_freq, geometric_inv_freq


LEGACY_PASSKEY_HASH_MULTIPLIER = 6364136223846793005
TRAIN_NPY_SHA256 = "115c5dca5c9023e5595fade1251cbc8d023edb8e0117abe5632aeb4218bcc2bf"
FORBIDDEN_LEAKED_VAL_SHA256 = (
    "85bfe9af77642d5e8995283e12645544c50b6233a33931fa2b1b37ec122e1b7e"
)
OFFICIAL_YARN_COMMIT = "995db5b575e75230b3384d658f8b944c9662f775"


@dataclass(frozen=True)
class ExperimentSpec:
    vocab_size: int = 50_304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3_072
    seq_len: int = 2_048
    train_tokens_requested: int = 500_000_000
    batch_size: int = 60
    micro_batch_size: int = 12
    seed: int = 42
    rope_base: float = 500_000.0
    evq_tau: float = 1.5
    passkey_mix_ratio: float = 0.02
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02

    @property
    def train_rows(self) -> int:
        return self.train_tokens_requested // self.seq_len

    @property
    def train_tokens(self) -> int:
        return self.train_rows * self.seq_len

    @property
    def optimizer_steps(self) -> int:
        if self.train_rows % self.batch_size:
            raise RuntimeError(
                f"train_rows={self.train_rows} must divide batch_size={self.batch_size}"
            )
        return self.train_rows // self.batch_size

    @property
    def grad_accum_steps(self) -> int:
        if self.batch_size % self.micro_batch_size:
            raise RuntimeError(
                f"batch_size={self.batch_size} must divide micro_batch_size="
                f"{self.micro_batch_size}"
            )
        return self.batch_size // self.micro_batch_size

    @property
    def micro_steps(self) -> int:
        if self.train_rows % self.micro_batch_size:
            raise RuntimeError(
                f"train_rows={self.train_rows} must divide micro_batch_size="
                f"{self.micro_batch_size}"
            )
        return self.train_rows // self.micro_batch_size

    @property
    def warmup_steps(self) -> int:
        return int(self.optimizer_steps * self.warmup_fraction)

    @property
    def train_npy_sha256(self) -> str:
        return TRAIN_NPY_SHA256

    @property
    def forbidden_leaked_val_sha256(self) -> str:
        return FORBIDDEN_LEAKED_VAL_SHA256

    def model_config(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.seq_len,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "train_tokens": self.train_tokens,
            "lr": self.learning_rate,
            "eval_lengths": [2_048, 4_096, 8_192, 16_384],
            "eval_chunks": 8,
        }


SPEC = ExperimentSpec()
ARMS = ("native_rope", "endpoint_evq_tau1p5")


def estimate_parameter_count(spec: ExperimentSpec = SPEC) -> int:
    """Return the exact tied-embedding GPT parameter count."""
    embedding = spec.vocab_size * spec.hidden_size
    attention = 4 * spec.hidden_size * spec.hidden_size
    mlp = 3 * spec.hidden_size * spec.intermediate_size
    norms = 2 * spec.hidden_size
    final_norm = spec.hidden_size
    return embedding + spec.num_layers * (attention + mlp + norms) + final_norm


def legacy_passkey_indices(
    n_rows: int, ratio: float = SPEC.passkey_mix_ratio
) -> tuple[int, ...]:
    """Match the prior ``MixedDataset`` deterministic row-selection rule."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"ratio must be within [0, 1], got {ratio}")
    return tuple(
        index
        for index in range(int(n_rows))
        if random.Random(index * LEGACY_PASSKEY_HASH_MULTIPLIER + 1).random()
        < float(ratio)
    )


def get_arm_inv_freq(
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
    tau_override: float | None = None,
) -> torch.Tensor:
    """Build one float64 endpoint frequency tensor for a registered arm."""
    if arm == "native_rope":
        if tau_override not in (None, 0.0):
            raise ValueError("native_rope does not accept a non-zero tau_override")
        return geometric_inv_freq(
            head_dim=spec.head_dim,
            base=spec.rope_base,
            dtype=torch.float64,
        )
    if arm == "endpoint_evq_tau1p5":
        tau = spec.evq_tau if tau_override is None else float(tau_override)
        return evq_cosh_inv_freq(
            head_dim=spec.head_dim,
            tau=tau,
            base=spec.rope_base,
            midpoint=False,
            dtype=torch.float64,
        )
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
