"""Differentiable full-profile Native-relative exponent allocation.

The optimization coordinates are the interior exponents themselves.  Candidate
construction, rather than the module, is responsible for enforcing the closed
convex family ``0 <= m_0 <= ... <= m_{K-1} <= 1``.  This makes the constraints
seen by the calibration solver identical to the constraints declared for the
installed fixed table.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ExponentBoxZAllocation(nn.Module):
    """One fixed table with directly differentiable interior exponents."""

    def __init__(
        self,
        native_rotary: nn.Module,
        *,
        scale: float,
        initial_exponents: np.ndarray,
        initial_inv_freq: np.ndarray,
        gain: float,
    ) -> None:
        super().__init__()
        native = getattr(native_rotary, "inv_freq", None)
        if not isinstance(native, torch.Tensor) or native.ndim != 1:
            raise ValueError("rotary module must expose one-dimensional Native inv_freq")
        count = int(native.numel())
        exponents = np.asarray(initial_exponents, dtype=np.float64)
        values = np.asarray(initial_inv_freq, dtype=np.float32)
        if (
            count < 3 or exponents.shape != (count,) or values.shape != (count,)
            or not np.isfinite(exponents).all() or not np.isfinite(values).all()
            or np.any(values <= 0.0) or np.any(values[:-1] <= values[1:])
            or exponents.min() < -2e-6 or exponents.max() > 1.0 + 2e-6
            or np.any(np.diff(exponents) < -2e-6)
            or abs(float(exponents[0])) > 2e-6 or abs(float(exponents[-1]) - 1.0) > 2e-6
            or not math.isfinite(scale) or scale <= 1.0
            or not math.isfinite(gain) or gain <= 0.0
        ):
            raise ValueError("invalid Native-relative full-z initialization")
        exponents = np.clip(exponents, 0.0, 1.0)
        self.scale = float(scale)
        self.attention_scaling = float(gain)
        self.rope_type = getattr(native_rotary, "rope_type", "default")
        self.register_buffer("native_inv_freq", native.detach().float().clone())
        self.register_buffer(
            "initial_exponents",
            torch.as_tensor(exponents, dtype=torch.float64, device=native.device),
        )
        self.register_buffer(
            "initial_inv_freq",
            torch.as_tensor(values, dtype=torch.float32, device=native.device),
        )
        self.delta_interior = nn.Parameter(
            torch.zeros(count - 2, dtype=torch.float32, device=native.device),
        )

    @property
    def original_inv_freq(self) -> torch.Tensor:
        return self.native_inv_freq

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.realized_inv_freq()

    def exponents(self) -> torch.Tensor:
        middle = self.initial_exponents[1:-1] + self.delta_interior.double()
        return torch.cat((self.initial_exponents[:1], middle, self.initial_exponents[-1:]))

    def realized_inv_freq(self) -> torch.Tensor:
        computed = self.native_inv_freq * torch.exp(-math.log(self.scale) * self.exponents()).float()
        exact_initial_with_grad = self.initial_inv_freq + (computed - computed.detach())
        return torch.where(
            torch.eq(self.delta_interior, 0.0).all(), exact_initial_with_grad, computed,
        )

    def set_delta_(self, value: np.ndarray | torch.Tensor) -> None:
        tensor = torch.as_tensor(value, dtype=self.delta_interior.dtype, device=self.delta_interior.device)
        if tensor.shape != self.delta_interior.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("invalid exponent delta")
        with torch.no_grad():
            self.delta_interior.copy_(tensor)

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
        active = self.exponents().detach().cpu().double().numpy()
        return {
            "method": "native_relative_exponent_box_full_z_v1",
            "scale": self.scale,
            "interior_exponent_parameters": int(self.delta_interior.numel()),
            "fixed_fast_exponent": float(active[0]),
            "fixed_slow_exponent": float(active[-1]),
            "gain": self.attention_scaling,
            "same_table_all_layers_and_lengths": True,
            "model_weight_updates": 0,
        }


def install_exponent_box_z(
    model: nn.Module,
    *,
    scale: float,
    initial_exponents: np.ndarray,
    initial_inv_freq: np.ndarray,
    gain: float,
) -> ExponentBoxZAllocation:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise ValueError("model has no shared rotary module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = ExponentBoxZAllocation(
        copy.deepcopy(rotary), scale=scale, initial_exponents=initial_exponents,
        initial_inv_freq=initial_inv_freq, gain=gain,
    ).to(next(model.parameters()).device)
    model.model.rotary_emb = replacement
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if len(trainable) != 1 or not trainable[0][0].endswith("delta_interior"):
        raise RuntimeError(f"unexpected exponent-box parameters: {[name for name, _ in trainable]}")
    return replacement


__all__ = ("ExponentBoxZAllocation", "install_exponent_box_z")
