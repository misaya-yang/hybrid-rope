"""Low-dimensional slow-band residual contract.

This is a shape/preflight implementation only.  It does not load OLMo,
construct a transformer module, or claim that a residual improves capability.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .protocol import ContractError, PHYSICAL_TRAIN_LENGTH, ROTARY_PAIRS


def route_for_budget(
    total_tokens: int,
    *,
    threshold: int = PHYSICAL_TRAIN_LENGTH,
) -> str:
    if int(total_tokens) < 0:
        raise ContractError("total token budget must be non-negative")
    if int(threshold) != PHYSICAL_TRAIN_LENGTH:
        raise ContractError("registered residual threshold is 4096")
    return "native" if int(total_tokens) <= int(threshold) else "augmented"


def select_slow_pairs(
    inv_freq: Sequence[float],
    *,
    minimum_wavelength_tokens: float = 500_000.0,
    max_pairs: int = 16,
) -> tuple[int, ...]:
    """Select the slowest available pairs whose wavelength clears the bound."""

    values = tuple(float(value) for value in inv_freq)
    if len(values) != ROTARY_PAIRS:
        raise ContractError(f"expected {ROTARY_PAIRS} inverse frequencies")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ContractError("inverse frequencies must be finite and positive")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ContractError("inverse frequencies must be strictly decreasing")
    if max_pairs < 1 or max_pairs > 16:
        raise ContractError("slow residual pair budget must be in [1, 16]")
    if (
        not math.isfinite(minimum_wavelength_tokens)
        or minimum_wavelength_tokens <= 0.0
    ):
        raise ContractError("minimum wavelength must be finite and positive")
    eligible = [
        index
        for index, omega in enumerate(values)
        if (2.0 * math.pi / omega) >= float(minimum_wavelength_tokens)
    ]
    if not eligible:
        raise ContractError("no EVQ pair satisfies the slow-band wavelength bound")
    # Inverse frequencies decrease with pair index; preserve table order while
    # taking the slowest eligible channels.
    return tuple(eligible[-int(max_pairs) :])


@dataclass(frozen=True)
class ResidualShape:
    pair_indices: tuple[int, ...]
    residual_head_dim: int
    qk_output_shape: tuple[str, ...]
    padded_value_shape: tuple[str, ...]
    augmented_cache_head_dim: int

    def validate(self) -> None:
        if not self.pair_indices:
            raise ContractError("residual must carry at least one pair")
        if len(set(self.pair_indices)) != len(self.pair_indices):
            raise ContractError("residual pair indices must be unique")
        if any(index < 0 or index >= ROTARY_PAIRS for index in self.pair_indices):
            raise ContractError("residual pair index is out of range")
        if self.residual_head_dim != 2 * len(self.pair_indices):
            raise ContractError("residual width must be two coordinates per pair")
        if self.residual_head_dim > 32:
            raise ContractError("residual route is not low-dimensional")
        if self.augmented_cache_head_dim != 128 + self.residual_head_dim:
            raise ContractError("augmented cache width drift")
        if self.qk_output_shape != ("batch", "heads", "query", "residual_dim"):
            raise ContractError("Q/K residual shape contract drift")
        if self.padded_value_shape != (
            "batch",
            "heads",
            "sequence",
            "residual_dim",
        ):
            raise ContractError("zero-padded value shape contract drift")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "pair_indices": list(self.pair_indices),
            "residual_head_dim": self.residual_head_dim,
            "qk_output_shape": list(self.qk_output_shape),
            "padded_value_shape": list(self.padded_value_shape),
            "augmented_cache_head_dim": self.augmented_cache_head_dim,
        }


def build_residual_shape(pair_indices: Iterable[int]) -> ResidualShape:
    pairs = tuple(sorted(int(index) for index in pair_indices))
    shape = ResidualShape(
        pair_indices=pairs,
        residual_head_dim=2 * len(pairs),
        qk_output_shape=("batch", "heads", "query", "residual_dim"),
        padded_value_shape=(
            "batch",
            "heads",
            "sequence",
            "residual_dim",
        ),
        augmented_cache_head_dim=128 + 2 * len(pairs),
    )
    shape.validate()
    return shape


def bitwise_short_route_gate(
    native_bytes: bytes,
    candidate_bytes: bytes,
    *,
    route: str,
) -> dict[str, object]:
    """Check the mandatory short-route identity without a model forward."""

    if route != "native":
        raise ContractError("bitwise short-route gate must use the Native route")
    native_digest = hashlib.sha256(native_bytes).hexdigest()
    candidate_digest = hashlib.sha256(candidate_bytes).hexdigest()
    passed = native_bytes == candidate_bytes
    return {
        "route": route,
        "passed": passed,
        "native_sha256": native_digest,
        "candidate_sha256": candidate_digest,
        "max_abs_logit_diff": 0.0 if passed else None,
        "claim_boundary": (
            "byte equality of supplied outputs only; no model capability result"
        ),
    }


def build_shape_receipt(
    *,
    inv_freq: Sequence[float],
    minimum_wavelength_tokens: float = 500_000.0,
    max_pairs: int = 16,
) -> dict[str, object]:
    pairs = select_slow_pairs(
        inv_freq,
        minimum_wavelength_tokens=minimum_wavelength_tokens,
        max_pairs=max_pairs,
    )
    shape = build_residual_shape(pairs)
    return {
        "route_threshold": PHYSICAL_TRAIN_LENGTH,
        "minimum_wavelength_tokens": float(minimum_wavelength_tokens),
        "selected_pair_indices": list(pairs),
        "shape": shape.as_dict(),
        "global_native_path_frozen": True,
        "short_route_bitwise_gate_required": True,
        "status": "SHAPE_ONLY_NO_GPU",
    }
