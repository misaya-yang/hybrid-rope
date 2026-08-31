"""Five-dimensional hat-basis phase-chord projection (family F2).

Interior log-frequency displacements are represented in a piecewise-linear hat
basis with five interior knots and exact zero displacement at both endpoints.
The F1 phase-chord direction is projected onto this basis by least squares, and
a single calibrated step ``alpha`` moves the frozen table along that registered
direction.  Construction reads only Native-prefix behaviour; the long-range test
remains a genuine out-of-construction outcome.

All model weights stay frozen; only the five hat coefficients are calibrated, so
this is ``GRADIENT_CALIBRATED_Z``, not a zero-learned-parameter construction.
The local Gauss--Newton metric constrains one declared direction; it is not a
standalone finite-table selector.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn


METHOD_ID = "frozen_checkpoint_pc_retention_project_v1"
INTERIOR_KNOTS = 5


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


def hat_basis(pair_count: int, interior_knots: int = INTERIOR_KNOTS) -> torch.Tensor:
    """Return the ``(pair_count, interior_knots)`` hat basis in normalized space.

    Knots sit at ``u_j = j/(interior_knots+1)`` with bandwidth equal to the knot
    spacing, so every hat vanishes exactly at both endpoints.
    """

    phi = torch.linspace(0.0, 1.0, pair_count, dtype=torch.float64)
    knots = torch.linspace(0.0, 1.0, interior_knots + 2, dtype=torch.float64)[1:-1]
    bandwidth = float(knots[0])
    basis = torch.zeros(pair_count, interior_knots, dtype=torch.float64)
    for j in range(interior_knots):
        basis[:, j] = (1.0 - (phi - knots[j]).abs() / bandwidth).clamp_min(0.0)
    return basis


def project_direction_onto_basis(direction: Any, basis: torch.Tensor) -> torch.Tensor:
    """Least-squares projection of a full log-frequency direction onto the basis."""

    d = torch.as_tensor(np.asarray(direction, dtype=np.float64)).reshape(-1)
    if d.numel() != basis.shape[0]:
        raise ValueError("direction length does not match the basis rows")
    gram = basis.t() @ basis
    rhs = basis.t() @ d
    coefficients = torch.linalg.solve(gram, rhs)
    return coefficients


def phase_chord_direction(
    native: torch.Tensor, target: Any
) -> np.ndarray:
    """Negative-log-frequency displacement from Native to the target table."""

    native_t = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    target_t = torch.as_tensor(np.asarray(target, dtype=np.float64)).reshape(-1)
    if target_t.numel() != native_t.numel():
        raise ValueError("target table shape does not match Native")
    return (native_t.log() - target_t.log()).numpy().astype(np.float64)


class HatBasisRotaryEmbedding(nn.Module):
    """Shared RoPE module displaced along a five-dimensional hat basis."""

    def __init__(self, native_inv_freq: torch.Tensor) -> None:
        super().__init__()
        native = _validate_native(native_inv_freq)
        self.register_buffer("native_inv_freq", native)
        self.register_buffer("log_native", (-native.double().log()).contiguous())
        self.register_buffer("basis", hat_basis(native.numel()))
        self.coefficients = nn.Parameter(
            torch.zeros(INTERIOR_KNOTS, dtype=torch.float32, device=native.device)
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

    def realized_inv_freq(self) -> torch.Tensor:
        displacement = self.basis.to(self.coefficients.dtype) @ self.coefficients.double()
        log_inv = self.log_native + displacement.to(torch.float64)
        computed = torch.exp(-log_inv).to(torch.float32)
        realized = torch.cat(
            (
                self.native_inv_freq[:1],
                computed[1:-1],
                self.native_inv_freq[-1:],
            )
        )
        native_forward_with_grad = self.native_inv_freq + (realized - realized.detach())
        at_native = torch.eq(self.coefficients, 0.0).all()
        return torch.where(at_native, native_forward_with_grad, realized)

    def set_coefficients_(self, value: torch.Tensor) -> None:
        tensor = torch.as_tensor(value, dtype=self.coefficients.dtype)
        if tensor.shape != self.coefficients.shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError("hat coefficients have an invalid identity")
        with torch.no_grad():
            self.coefficients.copy_(tensor.to(self.coefficients))

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
        return {
            "method": METHOD_ID,
            "parameterization": (
                "five-dimensional hat-basis log-frequency displacement with exact "
                "zero endpoint displacement"
            ),
            "pair_count": self.pair_count,
            "trainable_table_parameters": int(self.coefficients.numel()),
            "model_weight_updates": 0,
            "native_sha256_float32": float32_sha256(native),
            "active_sha256_float32": float32_sha256(active),
            "fast_endpoint_fixed": bool(active[0] == native[0]),
            "slow_endpoint_fixed": bool(active[-1] == native[-1]),
            "strictly_decreasing": bool(np.all(active[:-1] > active[1:])),
            "attention_scaling": 1.0,
        }


def install_hat_basis(model: nn.Module) -> tuple[HatBasisRotaryEmbedding, dict[str, Any]]:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    native = getattr(rotary, "inv_freq", None)
    if not isinstance(native, torch.Tensor):
        raise ValueError("model has no shared Native inverse-frequency buffer")
    if float(getattr(rotary, "attention_scaling", 1.0)) != 1.0:
        raise ValueError("F2 requires Native attention scaling one")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = HatBasisRotaryEmbedding(native)
    model.model.rotary_emb = replacement
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != 1 or not trainable[0][0].endswith("coefficients"):
        raise RuntimeError(f"hat-basis trainable scope drift: {[name for name, _ in trainable]}")
    return replacement, {
        **replacement.receipt(),
        "trainable_parameter_name": trainable[0][0],
        "model_parameters_frozen": True,
        "standard_shared_rotary_path": True,
    }


def table_from_coefficients(
    native: torch.Tensor, coefficients: Any
) -> np.ndarray:
    """Realise a float32 table from hat coefficients, pinning the endpoints."""

    native_t = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    coeff = torch.as_tensor(np.asarray(coefficients, dtype=np.float64)).reshape(-1)
    basis = hat_basis(native_t.numel()).to(coeff.dtype)
    log_inv = -native_t.log() + (basis @ coeff)
    table = torch.exp(-log_inv).to(torch.float32).numpy().astype("<f4")
    table[0] = np.float32(float(native_t[0]))
    table[-1] = np.float32(float(native_t[-1]))
    if not np.all(table[:-1] > table[1:]):
        raise ValueError("realised hat-basis table is not strictly decreasing")
    if not np.isfinite(table).all():
        raise ValueError("realised hat-basis table is not finite")
    return table


__all__ = (
    "METHOD_ID",
    "INTERIOR_KNOTS",
    "HatBasisRotaryEmbedding",
    "install_hat_basis",
    "hat_basis",
    "project_direction_onto_basis",
    "phase_chord_direction",
    "table_from_coefficients",
    "float32_sha256",
)
