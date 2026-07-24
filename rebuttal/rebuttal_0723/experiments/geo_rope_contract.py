#!/usr/bin/env python3
"""Canonical Geo/EVQ frequency identities for the 2026-07-23 rebuttal.

The historical submission code computed schedules in float64 and stored the
training buffer in float32.  All constructors below preserve that convention
so receipts can be compared byte-for-byte with historical ``inv_freq`` data.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from scripts.lib.rope.schedules import evq_cosh_inv_freq, geometric_inv_freq


STD_GEO = "std_geo"
PAPER_GEO = "paper_geo"
EVQ_COSH = "evq_cosh"
METHODS = (STD_GEO, PAPER_GEO, EVQ_COSH)

HISTORICAL_HEAD_DIM = 64
HISTORICAL_BASE = 500_000.0
HISTORICAL_PAPER_GEO_SHA256_FLOAT32 = (
    "88654f1fe2a414d38b1cc7e5a1c0e119e168eaa1118934d865b5b5004b858139"
)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash the tensor's contiguous raw bytes without changing its dtype."""
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _historical_cast(
    value: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Match historical float64 construction followed by buffer casting."""
    return value.to(dtype=dtype).contiguous()


def std_geo_inv_freq(
    head_dim: int,
    base: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard RoPE: u_k=k/K and omega_k=b**(-u_k)."""
    value = geometric_inv_freq(
        head_dim=int(head_dim),
        base=float(base),
        dtype=torch.float64,
    )
    return _historical_cast(value, dtype)


def paper_geo_inv_freq(
    head_dim: int,
    base: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Submission Geo: u_k=(k+1/2)/K and omega_k=b**(-u_k)."""
    value = evq_cosh_inv_freq(
        head_dim=int(head_dim),
        tau=0.0,
        base=float(base),
        midpoint=True,
        dtype=torch.float64,
    )
    return _historical_cast(value, dtype)


def paper_grid_evq_cosh_inv_freq(
    head_dim: int,
    tau: float,
    base: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """EVQ-Cosh on the same midpoint quantiles as Paper-Geo."""
    value = evq_cosh_inv_freq(
        head_dim=int(head_dim),
        tau=float(tau),
        base=float(base),
        midpoint=True,
        dtype=torch.float64,
    )
    return _historical_cast(value, dtype)


def build_training_inv_freq(
    method: str,
    *,
    head_dim: int,
    base: float,
    tau: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build one of the three explicitly named training schedules."""
    if method == STD_GEO:
        if tau is not None:
            raise ValueError("Std-Geo does not accept tau")
        return std_geo_inv_freq(head_dim, base, dtype=dtype)
    if method == PAPER_GEO:
        if tau is not None:
            raise ValueError("Paper-Geo does not accept tau")
        return paper_geo_inv_freq(head_dim, base, dtype=dtype)
    if method == EVQ_COSH:
        if tau is None:
            raise ValueError("EVQ-Cosh requires an explicit tau")
        return paper_grid_evq_cosh_inv_freq(
            head_dim,
            tau,
            base,
            dtype=dtype,
        )
    raise ValueError(f"unknown frequency method {method!r}; expected {METHODS}")


def paper_to_std_ratios(head_dim: int, base: float) -> dict[str, float]:
    """Return the exact global frequency and wavelength ratios."""
    d = int(head_dim)
    if d <= 0 or d % 2:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    b = float(base)
    if b <= 1.0:
        raise ValueError(f"base must exceed one, got {base}")
    frequency = b ** (-1.0 / d)
    return {
        "paper_over_std_frequency": frequency,
        "paper_over_std_wavelength": 1.0 / frequency,
    }


def frequency_receipt(
    method: str,
    *,
    head_dim: int,
    base: float,
    tau: float | None = None,
    preview_channels: int = 4,
) -> dict[str, Any]:
    """Return a reviewer-readable float32 schedule receipt."""
    inv = build_training_inv_freq(
        method,
        head_dim=head_dim,
        base=base,
        tau=tau,
        dtype=torch.float32,
    )
    count = min(max(int(preview_channels), 1), int(inv.numel()))
    formulas = {
        STD_GEO: {
            "quantiles": "u_k = k/K",
            "frequency": "omega_k = b^(-k/K)",
            "grid": "endpoint",
        },
        PAPER_GEO: {
            "quantiles": "u_k = (k+1/2)/K",
            "frequency": "omega_k = b^(-(k+1/2)/K)",
            "grid": "midpoint",
        },
        EVQ_COSH: {
            "quantiles": "u_k = (k+1/2)/K",
            "phi": (
                "phi_k(tau) = 1 - asinh((1-u_k)sinh(tau))/tau"
            ),
            "frequency": "omega_k = b^(-phi_k(tau))",
            "grid": "midpoint",
        },
    }
    receipt: dict[str, Any] = {
        "method": method,
        "head_dim": int(head_dim),
        "K": int(head_dim) // 2,
        "base": float(base),
        "dtype": str(inv.numpy().dtype),
        "sha256": tensor_sha256(inv),
        "first_channels": [float(value) for value in inv[:count]],
        "last_channels": [float(value) for value in inv[-count:]],
        "minimum": float(inv.min()),
        "maximum": float(inv.max()),
        "formula": formulas[method],
    }
    if method == EVQ_COSH:
        receipt["tau"] = float(tau) if tau is not None else None
        receipt["tau_zero_limit"] = PAPER_GEO
    if method == PAPER_GEO:
        receipt["relative_to_std_geo"] = paper_to_std_ratios(
            head_dim, base
        )
        receipt["single_base_equivalence"] = (
            "none: endpoint Std-Geo always has omega_0=1"
        )
        receipt["channelwise_effective_base"] = (
            "b_eff(k)=b^(1+1/(2k)) for k>0; no solution at k=0"
        )
    return receipt


def assert_historical_paper_geo() -> None:
    """Fail if the canonical paper setting drifts from its recorded bytes."""
    actual = paper_geo_inv_freq(HISTORICAL_HEAD_DIM, HISTORICAL_BASE)
    digest = tensor_sha256(actual)
    if digest != HISTORICAL_PAPER_GEO_SHA256_FLOAT32:
        raise RuntimeError(
            "Paper-Geo drift: "
            f"{digest} != {HISTORICAL_PAPER_GEO_SHA256_FLOAT32}"
        )
    ratio = paper_to_std_ratios(HISTORICAL_HEAD_DIM, HISTORICAL_BASE)
    if not math.isclose(
        ratio["paper_over_std_frequency"],
        float(actual[0]),
        rel_tol=1e-7,
        abs_tol=0.0,
    ):
        raise RuntimeError("Paper-Geo ratio does not match channel zero")
