#!/usr/bin/env python3
"""Official YaRN-equation operators (pinned: jquesnelle/yarn@995db5b).

Source of truth: ``scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`` in that commit.

Identity contract
-----------------
* **Native endpoint geometric grid** (ω_i = base^{-2i/d}): element-wise parity
  with the official ``yarn()`` method is required. Only this arm may be called
  "official YaRN equations" (still zero-shot if no YaRN continuation training).
* **Midpoint-Geo / EVQ** (non-native grids): apply the same equations after
  mapping ω → virtual channel coordinate
  ``j_v = -d · log(ω) / (2 · log(b))``. Label: **YaRN-derived generalization**.
* ``mscale = 1 + 0.1·ln(s)`` multiplies cos/sin **amplitude**; never fold into
  the phase divisor.
* Repo ``schedules.py`` method name ``yarn`` is a *different* fixed-index
  smoothstep scaler (20%–90%, temperature in divisor, no mscale). Keep that
  name only as a legacy alias; new code must call it ``repo_fixed_ramp``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Official helpers (equation-faithful to yarn@995db5b; evaluated in float64)
# ---------------------------------------------------------------------------


def find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Inverse dim formula → continuous channel index (official)."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (
        2.0 * math.log(base)
    )


def find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """Floor/ceil correction bounds; clamp to [0, dim-1] as in official code."""
    low = math.floor(
        find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_idx: float, max_idx: float, dim: int) -> torch.Tensor:
    """Official linear-ramp equation over ``dim`` bins, evaluated in float64."""
    if min_idx == max_idx:
        max_idx = max_idx + 0.001
    linear_func = (torch.arange(dim, dtype=torch.float64) - min_idx) / (max_idx - min_idx)
    return torch.clamp(linear_func, 0.0, 1.0)


def get_mscale(scale: float) -> float:
    """Official ``get_mscale``: 1 + 0.1·ln(scale) for scale > 1."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def yarn_mscale(scale: float, attn_factor: float = 1.0) -> float:
    """Official final amplitude: get_mscale(scale) * attn_factor."""
    return float(get_mscale(scale) * attn_factor)


def native_endpoint_inv_freq(
    head_dim: int,
    base: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Standard RoPE inv_freq: 1 / base^{2i/d}, i = 0,2,...,d-2 → length d/2."""
    # Match official: pos_freqs = base ** (arange(0, dim, 2).float() / dim)
    #                 inv_freq  = 1.0 / pos_freqs
    idx = torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2]
    return 1.0 / (float(base) ** (idx / float(head_dim)))


def official_yarn_on_native_grid(
    *,
    head_dim: int,
    base: float,
    scale: float,
    original_max_position_embeddings: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    extrapolation_factor: float = 1.0,
    attn_factor: float = 1.0,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Exact official ``LlamaYaRNScaledRotaryEmbedding.yarn`` for native Geo."""
    if scale <= 1.0:
        inv = native_endpoint_inv_freq(head_dim, base)
        return inv, 1.0, {
            "mode": "identity",
            "scale": float(scale),
            "mscale": 1.0,
            "low": 0,
            "high": 0,
            "label": "identity (scale<=1)",
        }

    # Official:
    #   pos_freqs = base ** (arange(0, dim, 2).float() / dim)
    #   inv_freq_extrapolation = 1.0 / pos_freqs
    #   inv_freq_interpolation = 1.0 / (scale * pos_freqs)
    pos_freqs = float(base) ** (
        torch.arange(0, head_dim, 2, dtype=torch.float64)[: head_dim // 2] / float(head_dim)
    )
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (float(scale) * pos_freqs)

    low, high = find_correction_range(
        beta_fast,
        beta_slow,
        head_dim,
        base,
        original_max_position_embeddings,
    )
    n_freq = head_dim // 2
    ramp = linear_ramp_mask(float(low), float(high), n_freq)
    inv_freq_mask = (1.0 - ramp) * float(extrapolation_factor)
    inv = inv_inter * (1.0 - inv_freq_mask) + inv_extra * inv_freq_mask
    mscale = yarn_mscale(float(scale), attn_factor=attn_factor)
    meta = {
        "mode": "official_yarn_native",
        "label": "official YaRN equations (native endpoint geometric grid)",
        "scale": float(scale),
        "mscale": float(mscale),
        "low": int(low),
        "high": int(high),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "original_max_position_embeddings": int(original_max_position_embeddings),
        "extrapolation_factor": float(extrapolation_factor),
        "attn_factor": float(attn_factor),
        "n_freq": int(n_freq),
    }
    return inv, mscale, meta


def virtual_dim_from_inv_freq(
    inv_freq: torch.Tensor,
    head_dim: int,
    base: float,
) -> torch.Tensor:
    """j_v = -d · log(ω) / (2 · log(b)); recovers 0..K-1 on native grid."""
    omega = inv_freq.to(dtype=torch.float64).view(-1).clamp_min(1e-300)
    return -float(head_dim) * torch.log(omega) / (2.0 * math.log(float(base)))


def official_yarn_on_inv_freq(
    inv_freq: torch.Tensor,
    *,
    head_dim: int,
    base: float,
    scale: float,
    original_max_position_embeddings: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    extrapolation_factor: float = 1.0,
    attn_factor: float = 1.0,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Apply official YaRN blend to an arbitrary inv_freq table.

    On native geometric frequencies this matches ``official_yarn_on_native_grid``.
    On midpoint/EVQ tables the ramp uses virtual coordinates → YaRN-derived.
    """
    inv = inv_freq.to(dtype=torch.float64).view(-1)
    if scale <= 1.0:
        return inv.clone(), 1.0, {
            "mode": "identity",
            "scale": float(scale),
            "mscale": 1.0,
            "label": "identity (scale<=1)",
        }

    inv_extra = inv.clone()
    inv_inter = inv / float(scale)

    low, high = find_correction_range(
        beta_fast,
        beta_slow,
        head_dim,
        base,
        original_max_position_embeddings,
    )
    j_v = virtual_dim_from_inv_freq(inv, head_dim=head_dim, base=base)
    # Same linear ramp formula as official, evaluated at continuous j_v.
    if high == low:
        high_f = float(high) + 0.001
    else:
        high_f = float(high)
    ramp_v = torch.clamp((j_v - float(low)) / (high_f - float(low)), 0.0, 1.0)
    inv_freq_mask = (1.0 - ramp_v) * float(extrapolation_factor)
    out = inv_inter * (1.0 - inv_freq_mask) + inv_extra * inv_freq_mask
    mscale = yarn_mscale(float(scale), attn_factor=attn_factor)

    # Detect near-native grid for labeling
    native = native_endpoint_inv_freq(head_dim, base)
    near_native = (
        inv.numel() == native.numel()
        and bool(torch.allclose(inv, native, rtol=1e-5, atol=1e-8))
    )
    meta = {
        "mode": "official_yarn_native" if near_native else "yarn_derived_virtual_dim",
        "label": (
            "official YaRN equations (native endpoint geometric grid)"
            if near_native
            else "YaRN-derived generalization (virtual-dim ramp on non-native grid)"
        ),
        "scale": float(scale),
        "mscale": float(mscale),
        "low": int(low),
        "high": int(high),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "original_max_position_embeddings": int(original_max_position_embeddings),
        "near_native_grid": near_native,
        "n_freq": int(inv.numel()),
    }
    return out, mscale, meta


def shared_index_yarn_control_on_inv_freq(
    inv_freq: torch.Tensor,
    *,
    head_dim: int,
    base: float,
    scale: float,
    original_max_position_embeddings: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    extrapolation_factor: float = 1.0,
    attn_factor: float = 1.0,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Apply one fixed official-index YaRN mask to any frequency table.

    This is an attribution control, not a new official-YaRN definition. The
    correction bounds and linear mask are computed once from the native
    ``head_dim/base`` channel indices, then the same per-index interpolation
    coefficients and the same official ``mscale`` are applied to ``inv_freq``.

    On a native endpoint geometric table this is exactly official YaRN. On
    EVQ or another non-native table it must be labeled a shared-index
    YaRN-component control, never official YaRN.
    """
    inv = inv_freq.to(dtype=torch.float64).view(-1)
    n_freq = int(head_dim) // 2
    if int(head_dim) <= 0 or int(head_dim) % 2:
        raise ValueError("head_dim must be a positive even integer")
    if inv.numel() != n_freq:
        raise ValueError(
            f"inv_freq has {inv.numel()} entries; expected {n_freq}"
        )
    if scale <= 1.0:
        native = native_endpoint_inv_freq(head_dim, base)
        return inv.clone(), 1.0, {
            "mode": "identity",
            "label": "identity (scale<=1)",
            "scale": float(scale),
            "mscale": 1.0,
            "official_on_input": bool(torch.equal(inv, native)),
            "shared_index_mask": True,
            "index_extrapolation_weights": [1.0] * n_freq,
        }

    low, high = find_correction_range(
        beta_fast,
        beta_slow,
        head_dim,
        base,
        original_max_position_embeddings,
    )
    ramp = linear_ramp_mask(float(low), float(high), n_freq)
    inv_freq_mask = (1.0 - ramp) * float(extrapolation_factor)
    inv_inter = inv / float(scale)
    out = inv_inter * (1.0 - inv_freq_mask) + inv * inv_freq_mask
    mscale = yarn_mscale(float(scale), attn_factor=attn_factor)
    native = native_endpoint_inv_freq(head_dim, base)
    official_on_input = bool(torch.equal(inv, native))
    return out, mscale, {
        "mode": (
            "official_yarn_native"
            if official_on_input
            else "shared_index_yarn_component_control"
        ),
        "label": (
            "official YaRN equations (native endpoint geometric grid)"
            if official_on_input
            else (
                "shared-index YaRN-component control "
                "(NOT official YaRN on this input)"
            )
        ),
        "scale": float(scale),
        "mscale": float(mscale),
        "low": int(low),
        "high": int(high),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "original_max_position_embeddings": int(
            original_max_position_embeddings
        ),
        "extrapolation_factor": float(extrapolation_factor),
        "attn_factor": float(attn_factor),
        "n_freq": int(n_freq),
        "official_on_input": official_on_input,
        "shared_index_mask": True,
        "index_extrapolation_weights": [
            float(value) for value in inv_freq_mask
        ],
    }


def repo_fixed_ramp_inv_freq(
    inv_freq: torch.Tensor,
    *,
    scale: float,
    temperature_coeff: float = 0.07,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Repo-defined fixed-index smoothstep ramp (legacy Table 2 'YaRN').

    20%–90% channel-index boundaries; temperature folded into the divisor;
    **no** attention mscale. Must not be called 'official YaRN'.
    """
    inv = inv_freq.to(dtype=torch.float64).view(-1)
    if scale <= 1.0:
        return inv.clone(), 1.0, {
            "mode": "identity",
            "scale": float(scale),
            "mscale": 1.0,
            "label": "identity",
        }
    k = inv.numel()
    idx = torch.arange(k, dtype=torch.float64)
    start = int(0.20 * k)
    end = int(0.90 * k)
    if end <= start:
        end = min(k - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + temperature_coeff * math.log2(scale)
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    out = inv / yarn_scale
    meta = {
        "mode": "repo_fixed_ramp",
        "label": "repo-defined fixed-ramp scaler (NOT official YaRN)",
        "scale": float(scale),
        "mscale": 1.0,
        "start_frac": 0.20,
        "end_frac": 0.90,
        "temperature_coeff": float(temperature_coeff),
        "temperature": float(temperature),
        "channel_start": int(start),
        "channel_end": int(end),
    }
    return out, 1.0, meta


def parity_vs_official_source(
    *,
    head_dim: int = 64,
    base: float = 500000.0,
    scale: float = 8.0,
    original_max_position_embeddings: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Dict[str, Any]:
    """Re-implement official yarn() inline and compare to our native path."""
    # --- inline official yarn() ---
    pos_freqs = float(base) ** (
        torch.arange(0, head_dim, 2, dtype=torch.float64)[: head_dim // 2] / float(head_dim)
    )
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (float(scale) * pos_freqs)
    low, high = find_correction_range(
        beta_fast, beta_slow, head_dim, base, original_max_position_embeddings
    )
    ramp = linear_ramp_mask(float(low), float(high), head_dim // 2)
    inv_freq_mask = (1.0 - ramp) * 1.0
    expected = inv_inter * (1.0 - inv_freq_mask) + inv_extra * inv_freq_mask
    expected_mscale = yarn_mscale(scale)

    got, got_mscale, meta = official_yarn_on_native_grid(
        head_dim=head_dim,
        base=base,
        scale=scale,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    # Also: applying on native inv_freq table must match
    native = native_endpoint_inv_freq(head_dim, base)
    got2, m2, meta2 = official_yarn_on_inv_freq(
        native,
        head_dim=head_dim,
        base=base,
        scale=scale,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )

    j_v = virtual_dim_from_inv_freq(native, head_dim=head_dim, base=base)
    j_err = float((j_v - torch.arange(head_dim // 2, dtype=torch.float64)).abs().max())

    ok = (
        bool(torch.allclose(got, expected, rtol=rtol, atol=atol))
        and bool(torch.allclose(got2, expected, rtol=rtol, atol=atol))
        and abs(got_mscale - expected_mscale) < 1e-12
        and abs(m2 - expected_mscale) < 1e-12
        and j_err < 1e-6
        and meta2.get("near_native_grid") is True
    )
    return {
        "parity_ok": ok,
        "max_abs_diff_native_path": float((got - expected).abs().max()),
        "max_abs_diff_invfreq_path": float((got2 - expected).abs().max()),
        "mscale": got_mscale,
        "expected_mscale": expected_mscale,
        "low": low,
        "high": high,
        "virtual_j_max_err": j_err,
        "transition_channels": f"{low}-{high}",
        "meta": meta,
    }


# Back-compat aliases used by earlier prep scripts
def official_yarn_inv_freq(inv_freq, **kwargs):
    out, mscale, meta = official_yarn_on_inv_freq(inv_freq, **kwargs)
    meta = dict(meta)
    meta["mscale"] = mscale
    return out, meta


if __name__ == "__main__":
    report = parity_vs_official_source()
    print(report)
    assert report["parity_ok"], report
    print("PASS official_yarn parity")
