"""Differentiable fixed-support RoPE allocation for frozen checkpoints.

The model's sampled frequency endpoints are immutable.  Positive normalized
gaps parameterize the interior coordinate ``z`` in
``x_k = a + R z_k``.  Model weights are frozen; only the gap logits may be
calibrated against an explicit ID/OOD objective.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


METHOD_ID = "frozen_checkpoint_fixed_support_direct_z_v1"
MAX_GAP_LOGIT_DELTA = 2.0


def float32_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_native(value: torch.Tensor) -> torch.Tensor:
    native = torch.as_tensor(value).detach().to(torch.float32).reshape(-1)
    if (
        native.numel() < 4
        or not bool(torch.isfinite(native).all())
        or not bool((native > 0.0).all())
        or not bool((native[:-1] > native[1:]).all())
    ):
        raise ValueError("Native inverse frequencies must be positive and decreasing")
    return native.contiguous()


class FixedSupportZRotaryEmbedding(nn.Module):
    """Shared RoPE module with exact endpoints and positive interior gaps."""

    def __init__(self, native_inv_freq: torch.Tensor) -> None:
        super().__init__()
        native = _validate_native(native_inv_freq)
        native_x = -native.double().log()
        span = native_x[-1] - native_x[0]
        if not math.isfinite(float(span)) or float(span) <= 0.0:
            raise ValueError("Native log-frequency span is invalid")
        native_z = (native_x - native_x[0]) / span
        native_gaps = torch.diff(native_z)
        if not bool((native_gaps > 0.0).all()):
            raise ValueError("Native normalized gaps must be positive")
        self.register_buffer("native_inv_freq", native)
        self.register_buffer("native_z", native_z.to(torch.float64))
        self.register_buffer("native_gap_log", native_gaps.log().to(torch.float64))
        self.register_buffer("log_fast", native_x[0].to(torch.float64))
        self.register_buffer("log_span", span.to(torch.float64))
        self.gap_delta_logits = nn.Parameter(
            torch.zeros(
                native.numel() - 1,
                dtype=torch.float32,
                device=native.device,
            )
        )
        self.attention_scaling = 1.0

    @property
    def pair_count(self) -> int:
        return int(self.native_inv_freq.numel())

    @property
    def original_inv_freq(self) -> torch.Tensor:
        return self.native_inv_freq

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.realized_inv_freq()

    def normalized_coordinates(self) -> torch.Tensor:
        logits = self.native_gap_log + self.gap_delta_logits.double()
        gaps = torch.softmax(logits, dim=0)
        return torch.cat(
            (
                torch.zeros(1, dtype=gaps.dtype, device=gaps.device),
                torch.cumsum(gaps, dim=0),
            )
        )

    def realized_inv_freq(self) -> torch.Tensor:
        z = self.normalized_coordinates()
        computed = torch.exp(-(self.log_fast + self.log_span * z)).to(torch.float32)
        computed = torch.cat(
            (
                self.native_inv_freq[:1],
                computed[1:-1],
                self.native_inv_freq[-1:],
            )
        )
        # Start from the exact released tensor while retaining a nonzero
        # straight-through derivative for the first optimizer step.  Keep the
        # decision in tensor space so this module remains fullgraph-compilable.
        native_forward_with_grad = self.native_inv_freq + (computed - computed.detach())
        at_native = torch.eq(self.gap_delta_logits, 0.0).all()
        return torch.where(at_native, native_forward_with_grad, computed)

    def project_(self) -> None:
        with torch.no_grad():
            self.gap_delta_logits.clamp_(
                -MAX_GAP_LOGIT_DELTA,
                MAX_GAP_LOGIT_DELTA,
            )

    def reset_native_(self) -> None:
        with torch.no_grad():
            self.gap_delta_logits.zero_()

    def set_gap_delta_(self, value: torch.Tensor) -> None:
        tensor = torch.as_tensor(value, dtype=self.gap_delta_logits.dtype)
        if tensor.shape != self.gap_delta_logits.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("gap delta state has an invalid identity")
        with torch.no_grad():
            self.gap_delta_logits.copy_(tensor.to(self.gap_delta_logits))
        self.project_()

    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim != 2 or position_ids.shape[0] not in {1, value.shape[0]}:
            raise ValueError("position_ids must have shape [B,T]")
        positions = position_ids[:, None, :].float().to(value.device)
        inv = (
            self.realized_inv_freq()[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(value.device)
        )
        with torch.autocast(device_type=value.device.type, enabled=False):
            phase = (inv.float() @ positions.float()).transpose(1, 2)
            embedding = torch.cat((phase, phase), dim=-1)
            cos = embedding.cos()
            sin = embedding.sin()
        return cos, sin

    def receipt(self) -> dict[str, Any]:
        native = self.native_inv_freq.detach().cpu().numpy()
        active = self.realized_inv_freq().detach().cpu().numpy()
        z = self.normalized_coordinates().detach().cpu().numpy()
        native_z = self.native_z.detach().cpu().numpy()
        return {
            "method": METHOD_ID,
            "parameterization": "positive normalized gaps with exact Native endpoints",
            "pair_count": self.pair_count,
            "trainable_table_parameters": int(self.gap_delta_logits.numel()),
            "effective_z_degrees_of_freedom": self.pair_count - 2,
            "model_weight_updates": 0,
            "native_sha256_float32": float32_sha256(native),
            "active_sha256_float32": float32_sha256(active),
            "fixed_support": bool(
                np.array_equal(
                    np.asarray(active, dtype=np.float32)[[0, -1]],
                    np.asarray(native, dtype=np.float32)[[0, -1]],
                )
            ),
            "candidate_differs_from_native": bool(
                not np.array_equal(
                    np.asarray(active, dtype=np.float32),
                    np.asarray(native, dtype=np.float32),
                )
            ),
            "maximum_normalized_coordinate_shift": float(np.max(np.abs(z - native_z))),
            "minimum_normalized_gap": float(np.diff(z).min()),
            "gap_logit_delta_bound": MAX_GAP_LOGIT_DELTA,
            "attention_scaling": 1.0,
        }


def install_fixed_support_z(model: nn.Module) -> tuple[FixedSupportZRotaryEmbedding, dict[str, Any]]:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    native = getattr(rotary, "inv_freq", None)
    if not isinstance(native, torch.Tensor):
        raise ValueError("model has no shared Native inverse-frequency buffer")
    if float(getattr(rotary, "attention_scaling", 1.0)) != 1.0:
        raise ValueError("direct-z pilot requires Native attention scaling one")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = FixedSupportZRotaryEmbedding(native)
    model.model.rotary_emb = replacement
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != 1 or not trainable[0][0].endswith("gap_delta_logits"):
        raise RuntimeError(f"direct-z trainable scope drift: {[name for name, _ in trainable]}")
    return replacement, {
        **replacement.receipt(),
        "trainable_parameter_name": trainable[0][0],
        "model_parameters_frozen": True,
        "standard_shared_rotary_path": True,
    }


__all__ = (
    "FixedSupportZRotaryEmbedding",
    "METHOD_ID",
    "float32_sha256",
    "install_fixed_support_z",
)
