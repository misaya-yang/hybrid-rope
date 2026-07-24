#!/usr/bin/env python3
"""Pre-registered protocols and frequency schedules for Reviewer 27bE."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rebuttal.rebuttal_0723.experiments.geo_rope_contract import (
    EVQ_COSH,
    PAPER_GEO,
    STD_GEO,
    build_training_inv_freq,
    frequency_receipt,
)
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.real_rope_schedules import (
    DERIVATION_METRICS,
    SCHEDULES as REAL_ROPE_SCHEDULES,
    SOURCE as REAL_ROPE_SOURCE,
)
from scripts.lib.rope.schedules import evq_cosh_phi


SEEDS = (42, 137, 256)
TAU_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.656854249492381, 6.0, 7.0)

# Seed 42 supplies the pre-registered tau scan.  The core shape arms and the
# native-endpoint Std-Geo reference are repeated at three seeds.  Paper-Geo
# versus EVQ-Cosh remains the paper-lineage midpoint comparison; Std-Geo is the
# standard-RoPE deployment reference.
SHAPE_TAU_ARMS = (
    "paper_geo",
    "evq_tau1",
    "evq_tau2",
    "evq_tau3",
    "evq_tau4",
    "evq_tau5",
    "evq_rule",
    "evq_tau6",
    "evq_tau7",
)
SHAPE_CORE_ARMS = (
    "paper_geo",
    "uniform_span_matched",
    "evq_rule",
    "power_matched",
    "exp_matched",
)
SHAPE_REFERENCE_ARMS = ("std_geo",)
SHAPE_REAL_ARMS = (
    "native_evq_span_rule",
    "native_exp_span_matched",
    "exact_kernel_uniform_span_matched",
    "attention_kernel_stdgeo42_span_matched",
)
SHAPE_ARMS = tuple(
    dict.fromkeys(
        SHAPE_REFERENCE_ARMS
        + SHAPE_TAU_ARMS
        + SHAPE_CORE_ARMS
        + SHAPE_REAL_ARMS
    )
)
HELDOUT_ARMS = ("paper_geo", "evq_rule")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    train_length: int
    requested_train_tokens: int
    global_batch_size: int
    micro_batch_size: int
    head_dim: int
    num_heads: int
    rope_base: float
    eval_lengths: tuple[int, ...]
    eval_tail_tokens: int
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    vocab_size: int = 50_304
    hidden_size: int = 768
    num_layers: int = 12
    intermediate_size: int = 3_072
    selection_anchor_count: int = 16
    test_anchor_count: int = 32
    anchor_seed: int = 20_260_724

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
    def grad_accum_steps(self) -> int:
        if self.global_batch_size % self.micro_batch_size:
            raise RuntimeError("global batch must be divisible by micro batch")
        return self.global_batch_size // self.micro_batch_size

    @property
    def micro_steps(self) -> int:
        return self.optimizer_steps * self.grad_accum_steps

    @property
    def warmup_steps(self) -> int:
        return max(1, int(self.optimizer_steps * self.warmup_fraction))

    @property
    def rule_tau(self) -> float:
        return self.head_dim / math.sqrt(self.train_length)

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
        return hashlib.sha256(payload.encode()).hexdigest()


SPECS = {
    # The historical Primary-II model/data scale and optimizer batch are kept.
    # micro=128 x accumulation=2 is an execution optimization for a 32-GiB 5090.
    "shape_l128": ExperimentSpec(
        name="shape_l128",
        train_length=128,
        requested_train_tokens=15_000_000,
        global_batch_size=256,
        micro_batch_size=128,
        head_dim=64,
        num_heads=12,
        rope_base=500_000.0,
        eval_lengths=(128, 256, 512, 1_024, 2_048, 4_096, 8_192),
        eval_tail_tokens=128,
    ),
    # Held-out from the submitted base=500K, d_head=64 configuration.
    "heldout_b1m_d128": ExperimentSpec(
        name="heldout_b1m_d128",
        train_length=512,
        requested_train_tokens=50_000_000,
        global_batch_size=16,
        micro_batch_size=16,
        head_dim=128,
        num_heads=6,
        rope_base=1_000_000.0,
        eval_lengths=(512, 1_024, 2_048, 4_096, 8_192, 16_384),
        eval_tail_tokens=128,
    ),
}


def arms_for_suite(suite: str) -> tuple[str, ...]:
    if suite == "shape_l128":
        return SHAPE_ARMS
    if suite == "heldout_b1m_d128":
        return HELDOUT_ARMS
    raise ValueError(f"unknown suite {suite!r}")


def seeds_for_arm(suite: str, arm: str) -> tuple[int, ...]:
    if suite == "heldout_b1m_d128":
        return SEEDS
    if (
        arm in SHAPE_CORE_ARMS
        or arm in SHAPE_REFERENCE_ARMS
        or arm in SHAPE_REAL_ARMS
    ):
        return SEEDS
    return (42,)


def estimate_parameter_count(spec: ExperimentSpec) -> int:
    embedding = spec.vocab_size * spec.hidden_size
    attention = 4 * spec.hidden_size * spec.hidden_size
    mlp = 3 * spec.hidden_size * spec.intermediate_size
    norms = 2 * spec.hidden_size
    final_norm = spec.hidden_size
    return embedding + spec.num_layers * (attention + mlp + norms) + final_norm


def midpoint_grid(n_freqs: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return (torch.arange(n_freqs, dtype=dtype) + 0.5) / float(n_freqs)


def _span_uniform(target: torch.Tensor) -> torch.Tensor:
    t = torch.linspace(
        0.0, 1.0, target.numel(), dtype=target.dtype, device=target.device
    )
    return target[0] + (target[-1] - target[0]) * t


def _matched_shape(
    target: torch.Tensor, family: str
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Match target endpoints and RMS interior deformation, not PPL."""
    uniform = _span_uniform(target)
    target_rms = torch.sqrt(torch.mean((target - uniform).square())).item()
    t = torch.linspace(
        0.0, 1.0, target.numel(), dtype=target.dtype, device=target.device
    )

    def candidate(parameter: float) -> torch.Tensor:
        if family == "power":
            warped = t.pow(parameter)
        elif family == "exp":
            if abs(parameter) < 1e-10:
                warped = t
            else:
                warped = torch.expm1(parameter * t) / math.expm1(parameter)
        else:
            raise ValueError(f"unknown matched family {family!r}")
        return target[0] + (target[-1] - target[0]) * warped

    def rms(parameter: float) -> float:
        value = candidate(parameter)
        return torch.sqrt(torch.mean((value - uniform).square())).item()

    low, high = (1.0, 128.0) if family == "power" else (0.0, 128.0)
    if rms(high) < target_rms:
        raise RuntimeError(f"{family} family cannot match target deformation")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if rms(middle) < target_rms:
            low = middle
        else:
            high = middle
    parameter = 0.5 * (low + high)
    value = candidate(parameter)
    achieved = rms(parameter)
    return value, parameter, {
        "target_deformation_rms": target_rms,
        "achieved_deformation_rms": achieved,
        "absolute_match_error": abs(achieved - target_rms),
    }


def _arm_tau(arm: str, spec: ExperimentSpec) -> float | None:
    values = {
        "paper_geo": 0.0,
        "evq_tau1": 1.0,
        "evq_tau2": 2.0,
        "evq_tau3": 3.0,
        "evq_tau4": 4.0,
        "evq_tau5": 5.0,
        "evq_rule": spec.rule_tau,
        "evq_tau6": 6.0,
        "evq_tau7": 7.0,
    }
    return values.get(arm)


def schedule_phi(
    suite: str,
    arm: str,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = SPECS[suite]
    if arm not in arms_for_suite(suite):
        raise ValueError(f"arm {arm!r} is not registered for {suite!r}")
    n_freqs = spec.head_dim // 2
    if arm == "std_geo":
        phi = torch.arange(n_freqs, dtype=dtype) / float(n_freqs)
        return phi, {
            "family": "geometric",
            "grid": "native endpoint u=k/K",
            "role": "small Std-Geo ablation",
            "method_identity": "Std-Geo",
        }
    if arm in SHAPE_REAL_ARMS:
        if (
            spec.head_dim != REAL_ROPE_SOURCE["head_dim"]
            or spec.train_length != REAL_ROPE_SOURCE["train_length"]
            or spec.rope_base != REAL_ROPE_SOURCE["base"]
        ):
            raise ValueError("frozen real-RoPE schedule contract mismatch")
        phi = torch.tensor(REAL_ROPE_SCHEDULES[arm], dtype=dtype)
        source = (
            "closed-form native-grid EVQ rule"
            if arm == "native_evq_span_rule"
            else (
                "deformation-matched exponential control"
                if arm == "native_exp_span_matched"
                else (
                    "exact cosine-kernel collision optimization"
                    if arm == "exact_kernel_uniform_span_matched"
                    else "Std-RoPE seed-42 selection-attention prior"
                )
            )
        )
        return phi, {
            "family": arm,
            "grid": "native Std-RoPE endpoint/span matched",
            "matched_to": "native_evq_span_rule",
            "derivation": source,
            "derivation_metrics": DERIVATION_METRICS[arm],
            "attention_prior_sha256": REAL_ROPE_SOURCE[
                "attention_prior_sha256"
            ],
            "selection_only": (
                arm == "attention_kernel_stdgeo42_span_matched"
            ),
        }

    tau = _arm_tau(arm, spec)
    if tau is not None:
        phi = evq_cosh_phi(n_freqs, tau=tau, midpoint=True, dtype=dtype)
        return phi, {
            "family": "geometric" if tau == 0.0 else "evq_cosh",
            "grid": "midpoint u=(k+0.5)/K",
            "tau": tau,
            "method_identity": (
                "Paper-Geo" if tau == 0.0 else "EVQ-Cosh"
            ),
            "tau_source": "d/sqrt(L)" if arm == "evq_rule" else "fixed grid",
        }

    target = evq_cosh_phi(
        n_freqs, tau=spec.rule_tau, midpoint=True, dtype=dtype
    )
    if arm == "uniform_span_matched":
        phi = _span_uniform(target)
        return phi, {
            "family": "uniform_log_grid",
            "grid": "EVQ-rule endpoint/span matched",
            "matched_to": "evq_rule",
        }
    family = "power" if arm == "power_matched" else "exp"
    phi, parameter, match = _matched_shape(target, family)
    return phi, {
        "family": family,
        "grid": "EVQ-rule endpoint/span/deformation-RMS matched",
        "matched_to": "evq_rule",
        "shape_parameter": parameter,
        **match,
    }


def training_inv_freq(
    suite: str,
    arm: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = SPECS[suite]
    if arm == "std_geo":
        inv = build_training_inv_freq(
            STD_GEO,
            head_dim=spec.head_dim,
            base=spec.rope_base,
            dtype=dtype,
        )
        _, metadata = schedule_phi(suite, arm, dtype=dtype)
        return inv, metadata
    if arm == "paper_geo":
        inv = build_training_inv_freq(
            PAPER_GEO,
            head_dim=spec.head_dim,
            base=spec.rope_base,
            dtype=dtype,
        )
        _, metadata = schedule_phi(suite, arm, dtype=dtype)
        return inv, metadata
    tau = _arm_tau(arm, spec)
    if tau is not None:
        inv = build_training_inv_freq(
            EVQ_COSH,
            head_dim=spec.head_dim,
            base=spec.rope_base,
            tau=tau,
            dtype=dtype,
        )
        _, metadata = schedule_phi(suite, arm, dtype=dtype)
        return inv, metadata
    phi, metadata = schedule_phi(suite, arm, dtype=torch.float64)
    value = torch.pow(
        torch.tensor(spec.rope_base, dtype=torch.float64),
        -phi.to(torch.float64),
    )
    return value.to(dtype=dtype).contiguous(), metadata


def frequency_contract(
    suite: str, *, spec: ExperimentSpec | None = None
) -> dict[str, Any]:
    """Return exact named-method receipts for a registered suite."""
    spec = SPECS[suite] if spec is None else spec
    return {
        "main_comparison": ["paper_geo", "evq_rule"],
        "std_geo_ablation": (
            "std_geo" if suite == "shape_l128" else None
        ),
        "std_geo": frequency_receipt(
            STD_GEO,
            head_dim=spec.head_dim,
            base=spec.rope_base,
        ),
        "paper_geo": frequency_receipt(
            PAPER_GEO,
            head_dim=spec.head_dim,
            base=spec.rope_base,
        ),
        "evq_cosh": frequency_receipt(
            EVQ_COSH,
            head_dim=spec.head_dim,
            base=spec.rope_base,
            tau=spec.rule_tau,
        ),
    }


def learning_rate_for_step(step: int, spec: ExperimentSpec) -> float:
    if step < 0 or step >= spec.optimizer_steps:
        raise ValueError(f"step outside [0,{spec.optimizer_steps}): {step}")
    if step < spec.warmup_steps:
        return spec.learning_rate * float(step + 1) / spec.warmup_steps
    decay = max(spec.optimizer_steps - spec.warmup_steps - 1, 1)
    progress = min(max((step - spec.warmup_steps) / decay, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.min_learning_rate + (
        spec.learning_rate - spec.min_learning_rate
    ) * cosine
