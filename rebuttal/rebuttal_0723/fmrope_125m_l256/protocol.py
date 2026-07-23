#!/usr/bin/env python3
"""Frozen protocol for the matched FMRoPE / EVQ L=256 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rebuttal.rebuttal_0723.geo_rope_contract import (
    EVQ_COSH,
    PAPER_GEO,
    assert_historical_paper_geo,
    build_training_inv_freq,
    frequency_receipt,
    std_geo_inv_freq,
)
from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq


ARMS = (
    "paper_geo_base500k",
    "fmrope_base256",
    "evq_cosh_tau4_paper_grid_base500k",
)

ARM_CONDITIONS = {
    "paper_geo_base500k": ("raw", "yarn_derived_inference_only"),
    "fmrope_base256": ("fixed_train_base", "target_matched_base"),
    "evq_cosh_tau4_paper_grid_base500k": ("raw",),
}


@dataclass(frozen=True)
class ExperimentSpec:
    # This is the repository's historical "125M" configuration. The exact
    # tied-embedding parameter count is 151,898,880 and is always reported.
    vocab_size: int = 50_304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3_072

    train_length: int = 256
    requested_train_tokens: int = 100_000_000
    global_batch_size: int = 256
    micro_batch_size: int = 64
    seed: int = 42

    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.01
    warmup_steps: int = 152

    geo_base: float = 500_000.0
    fmrope_train_base: float = 256.0
    evq_base: float = 500_000.0
    evq_tau: float = 4.0

    eval_lengths: tuple[int, ...] = (256, 512, 1_024, 2_048)
    eval_tail_tokens: int = 128
    eval_anchor_count: int = 32
    eval_anchor_seed: int = 20_260_723

    @property
    def optimizer_steps(self) -> int:
        return self.requested_train_tokens // (
            self.global_batch_size * self.train_length
        )

    @property
    def train_rows(self) -> int:
        return self.optimizer_steps * self.global_batch_size

    @property
    def train_tokens(self) -> int:
        return self.train_rows * self.train_length

    @property
    def prediction_tokens(self) -> int:
        return self.train_rows * (self.train_length - 1)

    @property
    def grad_accum_steps(self) -> int:
        if self.global_batch_size % self.micro_batch_size:
            raise RuntimeError("global batch must be divisible by micro batch")
        return self.global_batch_size // self.micro_batch_size

    @property
    def micro_steps(self) -> int:
        return self.optimizer_steps * self.grad_accum_steps

    def model_config(self) -> dict[str, int | float]:
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
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "train_tokens": self.train_tokens,
            "lr": self.learning_rate,
            "eval_lengths": list(self.eval_lengths),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SPEC = ExperimentSpec()


def estimate_parameter_count(spec: ExperimentSpec = SPEC) -> int:
    embedding = spec.vocab_size * spec.hidden_size
    attention = 4 * spec.hidden_size * spec.hidden_size
    mlp = 3 * spec.hidden_size * spec.intermediate_size
    norms = 2 * spec.hidden_size
    final_norm = spec.hidden_size
    return embedding + spec.num_layers * (attention + mlp + norms) + final_norm


def training_inv_freq(
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact training-time frequency tensor for one arm."""
    if arm == "paper_geo_base500k":
        if (
            spec.head_dim == 64
            and math.isclose(spec.geo_base, 500_000.0)
        ):
            assert_historical_paper_geo()
        return build_training_inv_freq(
            PAPER_GEO,
            head_dim=spec.head_dim,
            base=spec.geo_base,
            dtype=dtype,
        )
    if arm == "fmrope_base256":
        # Paper-faithful FMRoPE: theta_train = L_train.
        return std_geo_inv_freq(
            spec.head_dim,
            spec.fmrope_train_base,
            dtype=dtype,
        )
    if arm == "evq_cosh_tau4_paper_grid_base500k":
        return build_training_inv_freq(
            EVQ_COSH,
            head_dim=spec.head_dim,
            tau=spec.evq_tau,
            base=spec.evq_base,
            dtype=dtype,
        )
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def runtime_frequency(
    arm: str,
    condition: str,
    length: int,
    *,
    spec: ExperimentSpec = SPEC,
    checkpoint_inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    """Build one registered inference-time frequency condition."""
    length_i = int(length)
    if length_i not in spec.eval_lengths:
        raise ValueError(
            f"length must be one of {spec.eval_lengths}, got {length_i}"
        )
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if condition not in ARM_CONDITIONS[arm]:
        raise ValueError(
            f"condition {condition!r} is not registered for arm {arm!r}"
        )

    training_inv = (
        training_inv_freq(arm, spec=spec)
        if checkpoint_inv_freq is None
        else checkpoint_inv_freq.detach().cpu().float().contiguous()
    )
    if tuple(training_inv.shape) != (spec.head_dim // 2,):
        raise ValueError(
            f"checkpoint inv_freq has shape {tuple(training_inv.shape)}, "
            f"expected {(spec.head_dim // 2,)}"
        )
    if not torch.isfinite(training_inv).all() or not torch.all(
        training_inv > 0
    ):
        raise ValueError("checkpoint inv_freq must be finite and positive")

    if arm == "paper_geo_base500k":
        if condition == "raw":
            return training_inv.clone(), 1.0, {
                "identity": "Paper-Geo midpoint geometric RoPE",
                "operator": "raw",
                "base": spec.geo_base,
                "target_length": length_i,
            }
        scale = float(length_i) / float(spec.train_length)
        inv, mscale, meta = official_yarn_on_inv_freq(
            training_inv,
            head_dim=spec.head_dim,
            base=spec.geo_base,
            scale=scale,
            original_max_position_embeddings=spec.train_length,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        return inv.to(training_inv.dtype), float(mscale), {
            **meta,
            "operator": condition,
            "target_length": length_i,
            "claim_boundary": (
                "YaRN-derived virtual-coordinate generalization applied to "
                "checkpoint Paper-Geo; not official native-grid YaRN and not "
                "the full continuation-training protocol"
            ),
        }

    if arm == "fmrope_base256":
        if condition == "fixed_train_base":
            base = spec.fmrope_train_base
            inv = training_inv.clone()
        else:
            base = float(length_i)
            inv = std_geo_inv_freq(spec.head_dim, base)
        return inv, 1.0, {
            "identity": "paper-faithful local FMRoPE formula",
            "operator": condition,
            "training_base": spec.fmrope_train_base,
            "inference_base": float(base),
            "target_length": length_i,
            "requires_declared_target_length": (
                condition == "target_matched_base"
            ),
            "formula": "theta_train=L_train; theta_infer=L_target",
        }

    return training_inv.clone(), 1.0, {
        "identity": "EVQ-Cosh on the Paper-Geo midpoint quantizer",
        "operator": "raw",
        "base": spec.evq_base,
        "tau": spec.evq_tau,
        "midpoint": True,
        "target_length": length_i,
    }


def frequency_contract(spec: ExperimentSpec = SPEC) -> dict[str, Any]:
    """Return the exact schedules used by the main Paper-Geo/EVQ pair."""
    return {
        "main_comparison": [
            "paper_geo_base500k",
            "evq_cosh_tau4_paper_grid_base500k",
        ],
        "paper_geo": frequency_receipt(
            PAPER_GEO,
            head_dim=spec.head_dim,
            base=spec.geo_base,
        ),
        "evq_cosh": frequency_receipt(
            EVQ_COSH,
            head_dim=spec.head_dim,
            base=spec.evq_base,
            tau=spec.evq_tau,
        ),
        "std_geo_ablation_location": (
            "reviewer27be_shape_base/shape_l128:std_geo"
        ),
    }


def learning_rate_for_step(
    step: int, spec: ExperimentSpec = SPEC
) -> float:
    """Warm up then cosine-decay to the registered floor."""
    step_i = int(step)
    if step_i < 0 or step_i >= spec.optimizer_steps:
        raise ValueError(
            f"step must be in [0, {spec.optimizer_steps}), got {step_i}"
        )
    if step_i < spec.warmup_steps:
        return spec.learning_rate * float(step_i + 1) / float(spec.warmup_steps)
    decay_steps = max(spec.optimizer_steps - spec.warmup_steps - 1, 1)
    progress = (step_i - spec.warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.min_learning_rate + (
        spec.learning_rate - spec.min_learning_rate
    ) * cosine
