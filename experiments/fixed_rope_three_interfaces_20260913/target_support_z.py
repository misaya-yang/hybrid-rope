"""Differentiable full-z allocation with arbitrary fixed frequency endpoints."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class TargetSupportZAllocation(nn.Module):
    """Positive log-frequency gaps initialized from one deployed fixed table."""

    def __init__(self, native_rotary: nn.Module, *, initial_inv_freq: np.ndarray, gain: float) -> None:
        super().__init__()
        native = getattr(native_rotary, "inv_freq", None)
        if not isinstance(native, torch.Tensor) or native.ndim != 1:
            raise ValueError("rotary module must expose a one-dimensional Native inv_freq")
        initial = np.asarray(initial_inv_freq, dtype=np.float32)
        if (
            initial.shape != tuple(native.shape) or not np.isfinite(initial).all()
            or np.any(initial <= 0.0) or np.any(initial[:-1] <= initial[1:])
            or not math.isfinite(gain) or gain <= 0.0
        ):
            raise ValueError("invalid target-support initialization")
        initial_tensor = torch.as_tensor(initial, dtype=torch.float32, device=native.device)
        log_frequency = -initial_tensor.double().log()
        span = log_frequency[-1] - log_frequency[0]
        coordinates = (log_frequency - log_frequency[0]) / span
        gaps = torch.diff(coordinates)
        if not bool((gaps > 0.0).all()) or not bool(torch.isfinite(gaps).all()):
            raise ValueError("initial target-support gaps must be positive")
        self.register_buffer("native_inv_freq", native.detach().float().clone())
        self.register_buffer("initial_inv_freq", initial_tensor.clone())
        self.register_buffer("base_gap_log", gaps.log())
        self.register_buffer("log_fast", log_frequency[0])
        self.register_buffer("log_span", span)
        self.gap_delta_logits = nn.Parameter(torch.zeros(len(gaps), dtype=torch.float32, device=native.device))
        self.attention_scaling = float(gain)
        self.rope_type = getattr(native_rotary, "rope_type", "default")

    @property
    def original_inv_freq(self) -> torch.Tensor:
        return self.initial_inv_freq

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.realized_inv_freq()

    def normalized_coordinates(self) -> torch.Tensor:
        gaps = torch.softmax(self.base_gap_log + self.gap_delta_logits.double(), dim=0)
        return torch.cat((gaps.new_zeros(1), torch.cumsum(gaps, dim=0)))

    def realized_inv_freq(self) -> torch.Tensor:
        z = self.normalized_coordinates()
        computed = torch.exp(-(self.log_fast + self.log_span * z)).float()
        computed = torch.cat((self.initial_inv_freq[:1], computed[1:-1], self.initial_inv_freq[-1:]))
        exact_initial_with_grad = self.initial_inv_freq + (computed - computed.detach())
        return torch.where(torch.eq(self.gap_delta_logits, 0.0).all(), exact_initial_with_grad, computed)

    def set_gap_delta_(self, value: np.ndarray | torch.Tensor) -> None:
        tensor = torch.as_tensor(value, dtype=self.gap_delta_logits.dtype, device=self.gap_delta_logits.device)
        if tensor.shape != self.gap_delta_logits.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("invalid full-z gap delta")
        with torch.no_grad():
            self.gap_delta_logits.copy_(tensor - tensor.mean())

    def forward(self, value: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        positions = position_ids[:, None, :].float().to(value.device)
        inv = self.realized_inv_freq()[None, :, None].expand(position_ids.shape[0], -1, 1)
        with torch.autocast(device_type=value.device.type, enabled=False):
            phase = (inv.float() @ positions).transpose(1, 2)
            embedding = torch.cat((phase, phase), dim=-1)
            return embedding.cos() * self.attention_scaling, embedding.sin() * self.attention_scaling

    def receipt(self) -> dict[str, Any]:
        initial = self.initial_inv_freq.detach().cpu().numpy()
        active = self.realized_inv_freq().detach().cpu().numpy()
        return {
            "method": "target_support_full_z_gap_allocation_v1",
            "positive_log_frequency_gaps": True,
            "fixed_fast_endpoint": bool(active[0] == initial[0]),
            "fixed_slow_endpoint": bool(active[-1] == initial[-1]),
            "gap_parameters": int(self.gap_delta_logits.numel()),
            "effective_gap_degrees_of_freedom": int(self.gap_delta_logits.numel() - 1),
            "gain": self.attention_scaling,
            "same_table_all_layers_and_lengths": True,
            "model_weight_updates": 0,
        }


def install_target_support_z(model: nn.Module, *, initial_inv_freq: np.ndarray, gain: float) -> TargetSupportZAllocation:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise ValueError("model has no shared rotary module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = TargetSupportZAllocation(
        rotary, initial_inv_freq=initial_inv_freq, gain=gain,
    ).to(next(model.parameters()).device)
    model.model.rotary_emb = replacement
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if len(trainable) != 1 or not trainable[0][0].endswith("gap_delta_logits"):
        raise RuntimeError(f"unexpected target-support parameters: {[name for name, _ in trainable]}")
    return replacement


__all__ = ("TargetSupportZAllocation", "install_target_support_z")
