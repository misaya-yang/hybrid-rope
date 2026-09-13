"""Differentiable Native-relative static RoPE allocation for frozen models."""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class NativeRelativeAllocation(nn.Module):
    """One shared fixed-table family; only transition increments and gain vary."""

    def __init__(
        self,
        native_rotary: nn.Module,
        *,
        low: int,
        high: int,
        scale: float,
        initial_exponents: np.ndarray,
        initial_gain: float,
        initial_inv_freq: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        native = getattr(native_rotary, "inv_freq", None)
        if not isinstance(native, torch.Tensor) or native.ndim != 1:
            raise ValueError("Native rotary module must expose one inverse-frequency vector")
        count = int(native.numel())
        if not 0 <= low < high < count or not math.isfinite(scale) or scale <= 1.0:
            raise ValueError("invalid Native-relative band or scale")
        exponents = np.asarray(initial_exponents, dtype=np.float64)
        if (
            exponents.shape != (count,)
            or not np.isfinite(exponents).all()
            or np.any(np.diff(exponents) < -1e-12)
            or np.max(np.abs(exponents[: low + 1])) > 1e-10
            or np.max(np.abs(exponents[high:] - 1.0)) > 1e-10
        ):
            raise ValueError("initial exponents violate the fixed three-band contract")
        increments = np.diff(exponents[low : high + 1])
        if np.any(increments <= 0.0) or not np.isclose(increments.sum(), 1.0, atol=1e-10):
            raise ValueError("transition increments must be positive and sum to one")
        if not math.isfinite(initial_gain) or initial_gain <= 0.0:
            raise ValueError("initial gain must be finite and positive")
        self.low = int(low)
        self.high = int(high)
        self.scale = float(scale)
        self.register_buffer("native_inv_freq", native.detach().to(torch.float32).clone())
        logits = np.log(increments)
        logits -= logits.mean()
        self.increment_logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32, device=native.device))
        self.log_gain = nn.Parameter(torch.tensor(math.log(initial_gain), dtype=torch.float32, device=native.device))
        initial_values = (
            self.native_inv_freq.detach().cpu().numpy() * np.power(self.scale, -exponents)
            if initial_inv_freq is None
            else np.asarray(initial_inv_freq, dtype=np.float32)
        )
        if initial_values.shape != (count,) or not np.isfinite(initial_values).all():
            raise ValueError("initial inverse-frequency table is invalid")
        self.register_buffer("initial_inv_freq", torch.as_tensor(initial_values, dtype=torch.float32, device=native.device))
        self.register_buffer("initial_increment_logits", self.increment_logits.detach().clone())
        self.register_buffer("initial_log_gain", self.log_gain.detach().clone())
        self.register_buffer("initial_gain", torch.tensor(initial_gain, dtype=torch.float32, device=native.device))
        self.rope_type = getattr(native_rotary, "rope_type", "default")

    @property
    def original_inv_freq(self) -> torch.Tensor:
        return self.native_inv_freq

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.realized_inv_freq()

    @property
    def attention_scaling(self) -> torch.Tensor:
        computed = self.log_gain.exp()
        exact_initial_with_grad = self.initial_gain + (computed - computed.detach())
        return torch.where(torch.eq(self.log_gain, self.initial_log_gain), exact_initial_with_grad, computed)

    def normalized_logits(self) -> torch.Tensor:
        return self.increment_logits - self.increment_logits.mean()

    def exponents(self) -> torch.Tensor:
        increments = torch.softmax(self.normalized_logits(), dim=0)
        middle = torch.cumsum(increments, dim=0)
        prefix = torch.zeros(self.low + 1, dtype=middle.dtype, device=middle.device)
        suffix = torch.ones(
            self.native_inv_freq.numel() - self.high - 1,
            dtype=middle.dtype,
            device=middle.device,
        )
        return torch.cat((prefix, middle, suffix))

    def realized_inv_freq(self) -> torch.Tensor:
        computed = self.native_inv_freq * torch.exp(-math.log(self.scale) * self.exponents())
        exact_initial_with_grad = self.initial_inv_freq + (computed - computed.detach())
        at_initial = torch.eq(self.increment_logits, self.initial_increment_logits).all()
        return torch.where(at_initial, exact_initial_with_grad, computed)

    def set_state_(self, logits: torch.Tensor, log_gain: torch.Tensor | float) -> None:
        logits = torch.as_tensor(logits, dtype=self.increment_logits.dtype, device=self.increment_logits.device)
        gain = torch.as_tensor(log_gain, dtype=self.log_gain.dtype, device=self.log_gain.device)
        if logits.shape != self.increment_logits.shape or gain.numel() != 1:
            raise ValueError("proposal state shape mismatch")
        if not torch.isfinite(logits).all() or not torch.isfinite(gain).all():
            raise ValueError("proposal state is non-finite")
        with torch.no_grad():
            self.increment_logits.copy_(logits - logits.mean())
            self.log_gain.copy_(gain.reshape(()))

    def forward(self, value: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        inv = self.realized_inv_freq()[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float().to(value.device)
        with torch.autocast(device_type=value.device.type, enabled=False):
            phase = (inv.float() @ positions.float()).transpose(1, 2)
            embedding = torch.cat((phase, phase), dim=-1)
            gain = self.attention_scaling.float()
            return embedding.cos() * gain, embedding.sin() * gain

    def receipt(self) -> dict[str, Any]:
        exponents = self.exponents().detach().cpu().double().numpy()
        return {
            "method": "model_conditioned_native_relative_range_allocation_v1",
            "scale": self.scale,
            "low": self.low,
            "high": self.high,
            "increment_parameters": int(self.increment_logits.numel()),
            "effective_shape_degrees_of_freedom": int(self.increment_logits.numel() - 1),
            "gain_degrees_of_freedom": 1,
            "exponents": exponents.tolist(),
            "gain": float(self.attention_scaling.detach()),
            "same_table_all_layers_and_lengths": True,
            "model_weight_updates": 0,
        }


def install_native_relative_allocation(
    model: nn.Module,
    *,
    low: int,
    high: int,
    scale: float,
    initial_exponents: np.ndarray,
    initial_gain: float,
    initial_inv_freq: np.ndarray | None = None,
) -> NativeRelativeAllocation:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise ValueError("model has no shared rotary module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = NativeRelativeAllocation(
        copy.deepcopy(rotary),
        low=low,
        high=high,
        scale=scale,
        initial_exponents=initial_exponents,
        initial_gain=initial_gain,
        initial_inv_freq=initial_inv_freq,
    ).to(next(model.parameters()).device)
    model.model.rotary_emb = replacement
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != 2 or {name.rsplit(".", 1)[-1] for name, _ in trainable} != {"increment_logits", "log_gain"}:
        raise RuntimeError(f"unexpected trainable parameters: {[name for name, _ in trainable]}")
    return replacement


__all__ = ("NativeRelativeAllocation", "install_native_relative_allocation")
