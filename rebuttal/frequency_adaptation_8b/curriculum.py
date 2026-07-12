"""Pure contracts for the matched Geo/EVQ frequency-adaptation curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class PhaseSpec:
    """One fixed-token-budget training phase."""

    name: str
    seq_len: int
    min_distance: int
    max_distance: int
    steps: int
    effective_batch: int
    learning_rate: float
    transition: bool = False

    def __post_init__(self) -> None:
        if self.seq_len <= 0 or self.steps <= 0 or self.effective_batch <= 0:
            raise ValueError("phase length, steps, and batch must be positive")
        if not 0 < self.min_distance <= self.max_distance < self.seq_len:
            raise ValueError("phase distance bounds must fit inside the sequence")
        if self.learning_rate <= 0:
            raise ValueError("phase learning rate must be positive")

    @property
    def tokens_per_step(self) -> int:
        return self.seq_len * self.effective_batch

    @property
    def training_examples(self) -> int:
        return self.steps * self.effective_batch

    @property
    def training_tokens(self) -> int:
        return self.tokens_per_step * self.steps


_PHASES: Dict[str, PhaseSpec] = {
    "warmup": PhaseSpec(
        name="warmup",
        seq_len=4096,
        min_distance=256,
        max_distance=2048,
        steps=64,
        effective_batch=8,
        learning_rate=1e-4,
    ),
    "transition": PhaseSpec(
        name="transition",
        seq_len=4096,
        min_distance=512,
        max_distance=2048,
        steps=128,
        effective_batch=8,
        learning_rate=2e-5,
        transition=True,
    ),
    "exact_8k": PhaseSpec(
        name="exact_8k",
        seq_len=8192,
        min_distance=2048,
        max_distance=6144,
        steps=128,
        effective_batch=4,
        learning_rate=2e-5,
    ),
    "exact_16k": PhaseSpec(
        name="exact_16k",
        seq_len=16384,
        min_distance=6144,
        max_distance=14336,
        steps=96,
        effective_batch=2,
        learning_rate=2e-5,
    ),
}


def get_phase(name: str) -> PhaseSpec:
    """Return a registered phase or fail with the allowed names."""
    try:
        return _PHASES[name]
    except KeyError as exc:
        raise ValueError(f"unknown phase {name!r}; expected one of {sorted(_PHASES)}") from exc


def smoothstep(progress: float) -> float:
    """Cubic endpoint-flat interpolation weight on [0, 1]."""
    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    return progress * progress * (3.0 - 2.0 * progress)


def log_frequency_homotopy(
    native_inv_freq: torch.Tensor,
    target_inv_freq: torch.Tensor,
    progress: float,
) -> torch.Tensor:
    """Interpolate positive frequencies in log space with exact endpoints."""
    if native_inv_freq.ndim != 1 or target_inv_freq.ndim != 1:
        raise ValueError("frequency tensors must be one-dimensional")
    if native_inv_freq.shape != target_inv_freq.shape:
        raise ValueError("frequency tensors must have the same shape")
    if not torch.isfinite(native_inv_freq).all() or not torch.isfinite(target_inv_freq).all():
        raise ValueError("frequency tensors must be finite")
    if not torch.all(native_inv_freq > 0) or not torch.all(target_inv_freq > 0):
        raise ValueError("frequency tensors must be strictly positive")

    progress = float(progress)
    if progress == 0.0:
        return native_inv_freq.clone()
    if progress == 1.0:
        return target_inv_freq.clone()
    weight = smoothstep(progress)
    native_log = native_inv_freq.log()
    target_log = target_inv_freq.to(
        device=native_inv_freq.device,
        dtype=native_inv_freq.dtype,
    ).log()
    return torch.exp(native_log.lerp(target_log, weight))


def answer_only_labels(
    input_ids: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> torch.Tensor:
    """Mask the prompt and supervise only ``[answer_start, answer_end)``."""
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")
    answer_start = int(answer_start)
    answer_end = int(answer_end)
    if not 0 <= answer_start < answer_end <= input_ids.numel():
        raise ValueError("answer span must be non-empty and inside input_ids")
    labels = torch.full_like(input_ids, -100, dtype=torch.long)
    labels[answer_start:answer_end] = input_ids[answer_start:answer_end].to(torch.long)
    return labels


def half_split_pair_energy(row_energy: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Aggregate output-row energy using LLaMA ``rotate_half`` channel pairs.

    For each head the rotary pairs are ``(i, i + head_dim/2)``. Energies are
    summed across q heads or k/v heads so the result has ``head_dim/2`` bins.
    """
    if row_energy.ndim != 1:
        raise ValueError("row_energy must be one-dimensional")
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if row_energy.numel() == 0 or row_energy.numel() % head_dim:
        raise ValueError("row_energy length must be a positive multiple of head_dim")
    per_head = row_energy.reshape(-1, head_dim)
    half = head_dim // 2
    return (per_head[:, :half] + per_head[:, half:]).sum(dim=0)
