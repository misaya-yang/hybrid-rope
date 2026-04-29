#!/usr/bin/env python3
"""RoPE schedule construction and metadata helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


METHOD_ALIASES = {
    "baseline": "baseline",
    "baseline_native": "baseline",
    "base": "baseline",
    "a_baseline": "baseline",
    "c_base_only": "baseline",
    "pi": "pi",
    "yarn": "yarn",
    "hybrid": "anchored_sigmoid",
    "b_shape_only": "anchored_sigmoid",
    "d_full_hybrid": "anchored_sigmoid",
    "anchored_hybrid": "anchored_hybrid",
    "sigmoid": "sigmoid",
    "anchored_sigmoid": "anchored_sigmoid",
    "evq_cosh": "evq_cosh",
    "exact_cosh": "evq_cosh",
    "cosh": "evq_cosh",
    "evq_exp": "evq_exp",
    "exact_exp": "evq_exp",
    "exp_cdf": "evq_exp",
}


def canonical_method(method: str) -> str:
    key = method.strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unsupported method alias: {method}")
    return METHOD_ALIASES[key]


def infer_shape_name(method: str) -> str:
    m = canonical_method(method)
    if m == "baseline":
        return "geometric"
    if m in {"pi", "yarn"}:
        return m
    if m == "anchored_hybrid":
        return "anchored_hybrid"
    if m == "sigmoid":
        return "sigmoid"
    if m == "anchored_sigmoid":
        return "anchored_sigmoid"
    if m == "evq_cosh":
        return "evq_cosh"
    if m == "evq_exp":
        return "evq_exp"
    return m


def infer_rope_base_from_config(model_path: str, fallback: float = 500000.0) -> float:
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return float(fallback)
    try:
        data = json.loads(cfg.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return float(fallback)
    val = data.get("rope_theta")
    if val is None:
        # transformers 5.x may keep rope_theta under rope_scaling.
        rs = data.get("rope_scaling")
        if isinstance(rs, dict):
            val = rs.get("rope_theta")
    try:
        out = float(val)
    except (TypeError, ValueError):
        out = float(fallback)
    if out <= 0:
        out = float(fallback)
    return out


def geometric_inv_freq(head_dim: int, base: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    k = head_dim // 2
    idx = torch.arange(k, dtype=dtype)
    return 1.0 / (float(base) ** (2.0 * idx / float(head_dim)))


def evq_cosh_phi(
    n_freqs: int,
    tau: float,
    midpoint: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the EVQ-Cosh log-frequency quantiles phi_k(tau).

    ``midpoint=True`` matches the grid used in the paper experiments:
    u_k = (k + 0.5) / K.  With tau=0 this recovers the midpoint-discretized
    geometric schedule, while ``geometric_inv_freq`` above keeps the standard
    RoPE endpoint grid u_k = k / K.
    """
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")

    idx = torch.arange(n_freqs, dtype=dtype)
    if midpoint:
        u = (idx + 0.5) / float(n_freqs)
    else:
        u = idx / float(n_freqs)

    tau_f = float(tau)
    if abs(tau_f) < 1e-8:
        return u

    sinh_tau = math.sinh(tau_f)
    return 1.0 - (1.0 / tau_f) * torch.asinh((1.0 - u) * sinh_tau)


def evq_cosh_inv_freq(
    head_dim: int,
    tau: float,
    base: float = 500000.0,
    midpoint: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return EVQ-Cosh inverse RoPE frequencies for one attention head."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    phi = evq_cosh_phi(
        head_dim // 2,
        tau=tau,
        midpoint=midpoint,
        dtype=dtype,
    )
    return torch.pow(torch.tensor(float(base), dtype=dtype), -phi)


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def build_inv_freq(
    method: str,
    head_dim: int,
    base: float,
    max_seq_len: int,
    rigid_j0: int = 12,
    anchor_factor: float = 0.0,
    tau: Optional[float] = None,
    midpoint: bool = True,
) -> torch.Tensor:
    m = canonical_method(method)
    base_inv = geometric_inv_freq(head_dim=head_dim, base=base, dtype=torch.float64)

    if m == "baseline":
        return base_inv

    scale = max(float(max_seq_len) / 8192.0, 1.0)

    if m == "pi":
        return base_inv / scale

    if m == "yarn":
        k = head_dim // 2
        idx = torch.arange(k, dtype=torch.float64)
        start = int(0.20 * k)
        end = int(0.90 * k)
        if end <= start:
            end = min(k - 1, start + 1)
        ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
        ramp = smoothstep(ramp)
        temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
        yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
        return base_inv / yarn_scale

    if m == "anchored_hybrid":
        k = head_dim // 2
        rigid_j0 = min(max(int(rigid_j0), 0), k)
        if scale <= 1.0:
            return base_inv

        tail_base = float(base) * (scale ** 2)
        tail_base = max(tail_base, float(base) * 4.0)
        tail_inv = geometric_inv_freq(head_dim=head_dim, base=tail_base, dtype=torch.float64)

        out = base_inv.clone()
        if rigid_j0 < k:
            t = torch.arange(k - rigid_j0, dtype=torch.float64)
            if t.numel() == 1:
                ramp = torch.ones_like(t)
            else:
                t = t / float(t.numel() - 1)
                ramp = 0.5 - 0.5 * torch.cos(math.pi * t)
            alpha = min(0.40, max(0.08, 0.16 * math.log2(scale)))
            blend = alpha * ramp
            out[rigid_j0:] = (1.0 - blend) * base_inv[rigid_j0:] + blend * tail_inv[rigid_j0:]
        out[:rigid_j0] = base_inv[:rigid_j0]
        return out

    if m == "sigmoid":
        n = head_dim // 2
        slope = 16.05 / float(head_dim)
        center = 0.47 * float(n)
        idx = torch.arange(n, dtype=torch.float64)
        sig = 1.0 / (1.0 + torch.exp(-slope * (idx - center)))
        denom = sig[-1] - sig[0]
        if torch.abs(denom) < 1e-18:
            raise RuntimeError("sigmoid normalization collapsed.")
        s_norm = (sig - sig[0]) / denom
        return 1.0 / (float(base) ** s_norm)

    if m == "anchored_sigmoid":
        n = head_dim // 2
        slope = 16.05 / float(head_dim)
        j0 = 0.47 * float(n)

        eff_anchor = float(anchor_factor)
        if eff_anchor <= 0:
            eff_anchor = min(max(2.0, 2.5 * scale), 30.0)

        idx = torch.arange(n, dtype=torch.float64)
        sig = 1.0 / (1.0 + torch.exp(-slope * (idx - j0)))
        scale_factor = 1.0 + (eff_anchor - 1.0) * sig
        return base_inv / scale_factor

    if m == "evq_cosh":
        if tau is None:
            raise ValueError("evq_cosh requires explicit tau; pass tau=... to build_inv_freq")
        return evq_cosh_inv_freq(
            head_dim=head_dim,
            tau=tau,
            base=base,
            midpoint=midpoint,
            dtype=torch.float64,
        )

    if m == "evq_exp":
        n = head_dim // 2
        beta = 3.0
        idx = torch.arange(n, dtype=torch.float64)
        u = idx / float(n)
        if beta <= 1e-6:
            phi = u
        else:
            phi = (torch.pow(1.0 + beta, u) - 1.0) / beta
        return torch.pow(float(base), -phi)

    raise ValueError(f"Unsupported method: {method}")


def default_shape_params(method: str, base: float, max_seq_len: int) -> Dict[str, object]:
    m = canonical_method(method)
    if m == "anchored_sigmoid":
        scale = max(float(max_seq_len) / 8192.0, 1.0)
        eff_anchor = min(max(2.0, 2.5 * scale), 30.0)
        return {
            "schedule": "anchored_sigmoid",
            "anchor_factor": float(eff_anchor),
            "slope": 16.05,
            "center_ratio": 0.47,
            "base": float(base),
        }
    if m == "sigmoid":
        return {
            "schedule": "sigmoid",
            "slope": 16.05,
            "center_ratio": 0.47,
            "base": float(base),
        }
    if m == "evq_cosh":
        return {
            "schedule": "evq_cosh",
            "requires_explicit_tau": True,
            "base": float(base),
        }
    if m == "evq_exp":
        return {
            "schedule": "evq_exp",
            "beta": 3.0,
            "base": float(base),
        }
    if m == "yarn":
        return {"schedule": "yarn", "base": float(base)}
    if m == "pi":
        return {"schedule": "pi", "base": float(base)}
    if m == "anchored_hybrid":
        return {"schedule": "anchored_hybrid", "base": float(base)}
    return {"schedule": "geometric", "base": float(base)}


def infer_case(method: str, base: float, model_base: float) -> Tuple[str, str]:
    shape = infer_shape_name(method)
    if abs(float(base) - float(model_base)) <= 1e-6:
        return "A_shape_only", shape
    return "B_base_plus_shape", shape
