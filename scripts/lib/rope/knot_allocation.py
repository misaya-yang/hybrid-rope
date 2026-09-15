"""Five-degree knot allocation for frozen checkpoints (families F3/F4).

Seven fixed normalized pair-index knots pin the two endpoints and expose five
ordered interior values.  Positive knot gaps whose softmax-normalised cumulative
sum spans [0,1] parameterise the interior coordinates, so every realised table
is strictly ordered on one support.  A single model-relative ``support_factor``
stretches the slow endpoint while the fast endpoint, pair count, gain and all
model weights stay fixed.

All model weights remain frozen; only the gap logits may be calibrated.  This is
a *behavioural* allocation selected on position-resolved checkpoint loss, not a
static score: the module is the parameterisation and the optimiser, and both are
frozen before development opens (see the success-first preflight, family F3/F4).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


METHOD_ID = "frozen_checkpoint_z5_knot_allocation_v1"
MAX_GAP_LOGIT_DELTA = 4.0
INTERIOR_KNOTS = 5
TOTAL_KNOTS = INTERIOR_KNOTS + 2  # seven: two pinned endpoints + five interior


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


def knot_positions(pair_count: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Fixed normalized pair-index coordinates ``phi_k = k/(K-1)``."""

    return torch.linspace(0.0, 1.0, pair_count, dtype=dtype)


def interior_knot_u() -> torch.Tensor:
    """Fixed normalized knot abscissae ``u_j = j/6`` for the five interior knots."""

    return torch.linspace(0.0, 1.0, TOTAL_KNOTS, dtype=torch.float64)[1:-1].contiguous()


class Z5KnotRotaryEmbedding(nn.Module):
    """Shared RoPE module with pinned endpoints and five interior knot values."""

    def __init__(self, native_inv_freq: torch.Tensor, support_factor: float = 1.0) -> None:
        super().__init__()
        native = _validate_native(native_inv_freq)
        factor = float(support_factor)
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError("support_factor must be finite and >= 1")
        native_x = -native.double().log()
        log_fast = native_x[0]
        native_span = native_x[-1] - native_x[0]
        support_span = native_span + math.log(factor)
        if not math.isfinite(float(support_span)) or float(support_span) <= 0.0:
            raise ValueError("support span is invalid")
        self.register_buffer("native_inv_freq", native)
        self.register_buffer("log_fast", log_fast.to(torch.float64))
        self.register_buffer("native_span", native_span.to(torch.float64))
        self.register_buffer("support_span", support_span.to(torch.float64))
        self.register_buffer("support_factor", torch.tensor(factor, dtype=torch.float64))
        self.register_buffer("pair_phi", knot_positions(native.numel()))
        self.register_buffer("knot_u", interior_knot_u())
        # Six positive gaps between seven knots; uniform logits recover Native.
        self.gap_logits = nn.Parameter(
            torch.zeros(TOTAL_KNOTS - 1, dtype=torch.float32, device=native.device)
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

    def knot_values(self) -> torch.Tensor:
        """Seven knot values ``z_j`` with pinned ``z_0=0`` and ``z_{last}=1``."""

        gaps = torch.softmax(self.gap_logits.double(), dim=0)
        interior = torch.cumsum(gaps, dim=0)
        zero = torch.zeros(1, dtype=interior.dtype, device=interior.device)
        return torch.cat((zero, interior))

    def normalized_coordinates(self) -> torch.Tensor:
        """Piecewise-linear interpolation of the knot values at all pair positions."""

        z_knot = self.knot_values()
        u = torch.cat(
            (
                torch.zeros(1, dtype=z_knot.dtype, device=z_knot.device),
                self.knot_u.to(device=z_knot.device, dtype=z_knot.dtype),
                torch.ones(1, dtype=z_knot.dtype, device=z_knot.device),
            )
        )
        phi = self.pair_phi.to(device=z_knot.device, dtype=z_knot.dtype)
        return _piecewise_linear(u, z_knot, phi)

    def realized_inv_freq(self) -> torch.Tensor:
        z = self.normalized_coordinates()
        span = self.support_span
        computed = torch.exp(-(self.log_fast + span * z)).to(torch.float32)
        slow_endpoint = self.native_inv_freq[-1] / self.support_factor.to(torch.float32)
        realized = torch.cat(
            (
                self.native_inv_freq[:1],
                computed[1:-1],
                slow_endpoint.reshape(1),
            )
        )
        native_forward_with_grad = self.native_inv_freq + (realized - realized.detach())
        if float(self.support_factor) == 1.0:
            at_native = torch.eq(self.gap_logits, 0.0).all()
            return torch.where(at_native, native_forward_with_grad, realized)
        return realized

    def project_(self) -> None:
        with torch.no_grad():
            self.gap_logits.clamp_(-MAX_GAP_LOGIT_DELTA, MAX_GAP_LOGIT_DELTA)

    def reset_native_(self) -> None:
        with torch.no_grad():
            self.gap_logits.zero_()

    def set_gap_logits_(self, value: torch.Tensor) -> None:
        tensor = torch.as_tensor(value, dtype=self.gap_logits.dtype)
        if tensor.shape != self.gap_logits.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("gap-logit state has an invalid identity")
        with torch.no_grad():
            self.gap_logits.copy_(tensor.to(self.gap_logits))
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
        return {
            "method": METHOD_ID,
            "parameterization": (
                "seven fixed knots; two pinned endpoints; five ordered interior "
                "values from softmax-positive gaps"
            ),
            "pair_count": self.pair_count,
            "trainable_table_parameters": int(self.gap_logits.numel()),
            "effective_z_degrees_of_freedom": INTERIOR_KNOTS,
            "support_factor": float(self.support_factor),
            "model_weight_updates": 0,
            "native_sha256_float32": float32_sha256(native),
            "active_sha256_float32": float32_sha256(active),
            "fast_endpoint_fixed": bool(active[0] == native[0]),
            "strictly_decreasing": bool(np.all(active[:-1] > active[1:])),
            "minimum_normalized_gap": float(np.diff(z).min()),
            "gap_logit_delta_bound": MAX_GAP_LOGIT_DELTA,
            "attention_scaling": 1.0,
        }


def _piecewise_linear(
    u: torch.Tensor, z_knot: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of knot values ``z_knot`` at abscissae ``phi``.

    ``u`` and ``z_knot`` share length ``TOTAL_KNOTS`` and are strictly ordered.
    """

    # Locate each phi in [u_j, u_{j+1}) with a searchsorted on the interior.
    idx = torch.searchsorted(u, phi, right=True) - 1
    idx = idx.clamp(0, u.numel() - 2)
    left = u[idx]
    right = u[idx + 1]
    frac = (phi - left) / (right - left)
    frac = frac.clamp(0.0, 1.0)
    return z_knot[idx] + frac * (z_knot[idx + 1] - z_knot[idx])


def install_z5_knot(
    model: nn.Module, *, support_factor: float = 1.0
) -> tuple[Z5KnotRotaryEmbedding, dict[str, Any]]:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    native = getattr(rotary, "inv_freq", None)
    if not isinstance(native, torch.Tensor):
        raise ValueError("model has no shared Native inverse-frequency buffer")
    if float(getattr(rotary, "attention_scaling", 1.0)) != 1.0:
        raise ValueError("F3/F4 requires Native attention scaling one")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = Z5KnotRotaryEmbedding(native, support_factor=support_factor)
    model.model.rotary_emb = replacement
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != 1 or not trainable[0][0].endswith("gap_logits"):
        raise RuntimeError(f"z5-knot trainable scope drift: {[name for name, _ in trainable]}")
    return replacement, {
        **replacement.receipt(),
        "trainable_parameter_name": trainable[0][0],
        "model_parameters_frozen": True,
        "standard_shared_rotary_path": True,
    }


def init_gap_logits_from_table(
    table: Any, native: torch.Tensor, *, support_factor: float = 1.0
) -> torch.Tensor:
    """Recover gap logits whose piecewise-linear knots best match a realised table.

    Used to seed the four predeclared F3/F4 initialisations (Native, the F1
    family winner, the learned-direction teacher, and the coarse budgeted
    direction) inside the frozen knot parameterisation.
    """

    table_t = torch.as_tensor(np.asarray(table, dtype=np.float64)).reshape(-1)
    native_t = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    if table_t.numel() != native_t.numel():
        raise ValueError("initialisation table shape does not match Native")
    log_fast = -native_t[0].log()
    span = (-native_t[-1].log() - log_fast) + math.log(float(support_factor))
    z = (-table_t.log() - log_fast) / span
    z = z.clamp(0.0, 1.0)
    u = interior_knot_u()
    knot_values = torch.cat(
        (torch.zeros(1, dtype=z.dtype), _interp_at(z, u), torch.ones(1, dtype=z.dtype))
    )
    gaps = torch.diff(knot_values).clamp_min(1e-8)
    logits = gaps.log()
    return (logits - logits.mean()).to(torch.float32)


def _interp_at(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Sample a per-pair coordinate curve ``z`` at the interior knot abscissae."""

    pair_phi = knot_positions(z.numel(), dtype=torch.float64)
    idx = torch.searchsorted(pair_phi, u, right=True) - 1
    idx = idx.clamp(0, pair_phi.numel() - 2)
    left = pair_phi[idx]
    right = pair_phi[idx + 1]
    frac = ((u - left) / (right - left)).clamp(0.0, 1.0)
    return z[idx] + frac * (z[idx + 1] - z[idx])


__all__ = (
    "METHOD_ID",
    "INTERIOR_KNOTS",
    "TOTAL_KNOTS",
    "MAX_GAP_LOGIT_DELTA",
    "Z5KnotRotaryEmbedding",
    "install_z5_knot",
    "init_gap_logits_from_table",
    "float32_sha256",
)
