"""Shared full-z RoPE for scratch joint training with fixed sampled support."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy().astype("<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


class JointFixedSupportRotaryEmbedding(nn.Module):
    """Positive normalized gaps, exact endpoints, and no shape projection."""

    def __init__(self, initial_inv_freq: torch.Tensor) -> None:
        super().__init__()
        initial = torch.as_tensor(initial_inv_freq).detach().float().reshape(-1)
        if (
            initial.numel() < 4
            or not bool(torch.isfinite(initial).all())
            or not bool((initial > 0).all())
            or not bool((initial[:-1] > initial[1:]).all())
        ):
            raise ValueError("initial frequencies must be finite, positive, decreasing")
        x = -initial.double().log()
        span = x[-1] - x[0]
        z = (x - x[0]) / span
        gaps = torch.diff(z)
        self.register_buffer("initial_inv_freq", initial.contiguous())
        self.register_buffer("initial_gap_log", gaps.log().contiguous())
        self.register_buffer("log_fast", x[0].contiguous())
        self.register_buffer("log_span", span.contiguous())
        self.gap_delta_logits = nn.Parameter(torch.zeros_like(gaps, dtype=torch.float32))
        self.attention_scaling = 1.0

    @property
    def pair_count(self) -> int:
        return int(self.initial_inv_freq.numel())

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.realized_inv_freq()

    def normalized_coordinates(self) -> torch.Tensor:
        gaps = torch.softmax(
            self.initial_gap_log + self.gap_delta_logits.double(), dim=0
        )
        return torch.cat((gaps.new_zeros(1), torch.cumsum(gaps, dim=0)))

    def realized_inv_freq(self) -> torch.Tensor:
        z = self.normalized_coordinates()
        computed = torch.exp(-(self.log_fast + self.log_span * z)).float()
        computed = torch.cat(
            (
                self.initial_inv_freq[:1],
                computed[1:-1],
                self.initial_inv_freq[-1:],
            )
        )
        exact_initial_with_grad = self.initial_inv_freq + (
            computed - computed.detach()
        )
        at_initial = torch.eq(self.gap_delta_logits, 0.0).all()
        return torch.where(at_initial, exact_initial_with_grad, computed)

    def fix_gauge_(self) -> None:
        """Remove the softmax-null common offset without restricting a shape."""
        with torch.no_grad():
            self.gap_delta_logits.sub_(self.gap_delta_logits.mean())

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            int(length), device=self.gap_delta_logits.device, dtype=torch.float32
        )
        inv = self.realized_inv_freq().float()
        with torch.autocast(device_type=inv.device.type, enabled=False):
            phase = torch.outer(positions, inv)
            embedding = torch.cat((phase, phase), dim=-1)
            return embedding.cos(), embedding.sin()

    def receipt(self) -> dict[str, Any]:
        z = self.normalized_coordinates().detach().cpu()
        active = self.realized_inv_freq().detach().cpu()
        return {
            "parameterization": "positive normalized gaps with exact endpoints",
            "pair_count": self.pair_count,
            "gap_parameters": int(self.gap_delta_logits.numel()),
            "effective_degrees_of_freedom": self.pair_count - 2,
            "projection": "none",
            "gauge_fix": "subtract mean after each optimizer update",
            "initial_sha256_float32": _tensor_sha256(self.initial_inv_freq),
            "active_sha256_float32": _tensor_sha256(active),
            "minimum_gap": float(torch.diff(z).min()),
            "maximum_abs_gap_delta_logit": float(
                self.gap_delta_logits.detach().abs().max()
            ),
        }


def install_joint_full_z(model: nn.Module) -> JointFixedSupportRotaryEmbedding:
    first = model.blocks[0].attention.rope
    initial = first.inv_freq.detach().clone()
    if any(block.attention.rope is not first for block in model.blocks):
        raise RuntimeError("GPT blocks do not share the original rotary module")
    replacement = JointFixedSupportRotaryEmbedding(initial)
    for block in model.blocks:
        block.attention.rope = replacement
    named = [name for name, _ in model.named_parameters() if name.endswith("gap_delta_logits")]
    if len(named) != 1:
        raise RuntimeError(f"expected one shared full-z parameter, found {named}")
    return replacement


__all__ = ("JointFixedSupportRotaryEmbedding", "install_joint_full_z")
