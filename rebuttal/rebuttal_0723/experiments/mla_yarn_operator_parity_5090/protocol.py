#!/usr/bin/env python3
"""Frozen protocol for the MLA shared-index YaRN factorial."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


TRAINING_ARMS = ("native_geo", "evq_cosh")
FREQUENCY_PAIRS = (8, 32)
SEEDS = (42, 43, 88)
GATE_SEED = 42
CONFIRMATORY_SEEDS = (43, 88)
OPERATORS = (
    "raw",
    "position_interpolation",
    "shared_index_freq_only",
    "mscale_only",
    "shared_index_full",
    "virtual_coordinate_full",
)
PRIMARY_LENGTHS = (16_384, 32_768)

# The underlying trainer is the completed MLA-scarcity trainer. This is its
# scientific protocol fingerprint, not a code fingerprint.
BASE_TRAINING_PROTOCOL_SHA256 = (
    "f64ba972e97b7af4759bab9ad49751dd26c92b7440e5fdb08b853fec6cf5e0a3"
)


@dataclass(frozen=True)
class OperatorParitySpec:
    train_length: int = 4_096
    train_tokens: int = 299_892_736
    checkpoint_labels: tuple[str, ...] = ("200m", "300m")
    eval_lengths: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768)
    eval_batch_sizes: tuple[int, ...] = (8, 4, 2, 1)
    eval_tail_tokens: int = 4_096
    selection_anchor_count: int = 16
    test_anchor_count: int = 32
    anchor_seed: int = 2_026_072_402
    base: float = 500_000.0
    tau: float = 1.414
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    minimum_mean_scaled_advantage_nll: float = 0.05
    minimum_mean_interaction_nll: float = 0.05
    maximum_in_domain_cost_nll: float = 0.02
    maximum_native_scaler_cost_nll: float = 0.05

    @property
    def eval_batch_size_by_length(self) -> dict[int, int]:
        if len(self.eval_lengths) != len(self.eval_batch_sizes):
            raise RuntimeError("evaluation batch-size table is malformed")
        return dict(zip(self.eval_lengths, self.eval_batch_sizes))

    def fingerprint(self) -> str:
        payload = {
            **asdict(self),
            "training_arms": TRAINING_ARMS,
            "frequency_pairs": FREQUENCY_PAIRS,
            "seeds": SEEDS,
            "operators": OPERATORS,
            "primary_lengths": PRIMARY_LENGTHS,
            "base_training_protocol_sha256": (
                BASE_TRAINING_PROTOCOL_SHA256
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


SPEC = OperatorParitySpec()


def operators_for_stage(stage: str) -> tuple[str, ...]:
    current = str(stage)
    if current not in SPEC.checkpoint_labels:
        raise ValueError(f"unregistered checkpoint stage: {stage}")
    if current == "200m":
        return ("raw", "shared_index_full")
    return OPERATORS
