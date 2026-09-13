"""Exact CPU replay objective on captured pre-RoPE Q/K.

The objective is a checkpoint-conditioned transport diagnostic.  It preserves
full key competition and GQA, but it does not model hidden-state feedback,
new distractors, values, decoding, or task correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from scripts.analysis.native_attention_kl import native_attention_kl


@dataclass(frozen=True)
class ReplayCapture:
    """One layer/row capture with all causally visible keys.

    ``q`` has shape ``[Hq,Q,D]`` and ``k`` has shape ``[Hkv,T,D]``.  Query
    heads are in model order so contiguous GQA repetition is well-defined.
    """

    q: np.ndarray
    k: np.ndarray
    query_positions: np.ndarray
    native_inv_freq: np.ndarray
    attention_scale: float
    reference_gain: float
    group: str
    row_id: str
    layer: int


def validate_capture(capture: ReplayCapture) -> ReplayCapture:
    q = np.asarray(capture.q)
    k = np.asarray(capture.k)
    positions = np.asarray(capture.query_positions)
    inv = np.asarray(capture.native_inv_freq)
    if q.ndim != 3 or k.ndim != 3 or q.shape[-1] != k.shape[-1]:
        raise ValueError("capture q/k must be matching rank-three arrays")
    if q.shape[-1] % 2 or q.shape[0] % k.shape[0]:
        raise ValueError("capture must have an even head dimension and divisible GQA heads")
    if positions.shape != (q.shape[1],) or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("capture query positions must be integer [Q]")
    if np.any(positions < 0) or np.any(positions >= k.shape[1]):
        raise ValueError("capture query positions must lie in the complete key sequence")
    if inv.shape != (q.shape[-1] // 2,) or np.any(inv <= 0):
        raise ValueError("capture native frequencies must be positive [D/2]")
    if not all(np.isfinite(value).all() for value in (q, k, inv)):
        raise ValueError("capture arrays must be finite")
    if (not math.isfinite(float(capture.attention_scale)) or capture.attention_scale <= 0
            or not math.isfinite(float(capture.reference_gain)) or capture.reference_gain < 0):
        raise ValueError("capture attention scale/gain is invalid")
    if not capture.group or not capture.row_id or capture.layer < 0:
        raise ValueError("capture group, row identity, and layer are required")
    return capture


def increments_to_exponents(increments: Iterable[float], *, pairs: int | None = None) -> np.ndarray:
    """Map K-1 nonnegative gap increments to K monotone exponents.

    Exact zeros are legal and expose plateaus or disjoint active regions.
    Their total is the tail depth and may be anywhere in ``[0,1]``.
    """
    values = np.asarray(list(increments), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("increments must be a nonempty finite vector")
    if np.any(values < 0) or values.sum() > 1.0 + 1e-12:
        raise ValueError("increments must be nonnegative with total depth at most one")
    if pairs is not None and len(values) != pairs - 1:
        raise ValueError("K rotary pairs require K-1 increments")
    result = np.concatenate(([0.0], np.cumsum(values)))
    if np.any(np.diff(result) < 0) or result[-1] > 1.0 + 1e-12:
        raise AssertionError("increment-to-exponent invariant failed")
    return result


def profile_inv_freq(native_inv_freq: Iterable[float], increments: Iterable[float], scale: float) -> np.ndarray:
    native = np.asarray(list(native_inv_freq), dtype=np.float64)
    if (native.ndim != 1 or not len(native) or not np.isfinite(native).all()
            or np.any(native <= 0) or not math.isfinite(scale) or scale <= 1):
        raise ValueError("native frequencies and extension scale are invalid")
    exponents = increments_to_exponents(increments, pairs=len(native))
    active = native * np.power(float(scale), -exponents)
    if np.any(active[:-1] <= active[1:]):
        raise ValueError("profile produces a non-descending frequency table")
    return active


def finite_rho_grid(scale: float, *, points_per_doubling: int = 2) -> np.ndarray:
    """Return the declared finite log-length grid, including 1 and ``scale``."""
    if not math.isfinite(scale) or scale <= 1 or points_per_doubling < 1:
        raise ValueError("scale and points_per_doubling are invalid")
    steps = max(1, int(math.ceil(math.log2(scale) * points_per_doubling)))
    grid = np.geomspace(1.0, scale, steps + 1, dtype=np.float64)
    grid[0], grid[-1] = 1.0, float(scale)
    return grid


def _cvar(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(flat) or not 0 < fraction <= 1:
        raise ValueError("CVaR requires values and a fraction in (0,1]")
    count = max(1, int(math.ceil(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def evaluate_replay_objective(
    captures: Iterable[ReplayCapture],
    increments: Iterable[float],
    *,
    scale: float,
    gain: float,
    rhos: Iterable[float],
    cvar_fraction: float = 0.1,
    native_mean_limit: float | None = None,
    native_group_cvar_limit: float | None = None,
    active_inv_freq: Iterable[float] | None = None,
) -> dict:
    """Evaluate the exact finite-grid KL objective and Native constraints.

    The candidate table is shared by every capture.  Overall means weight each
    captured head/query equally.  Group statistics remain separate so a global
    mean cannot hide one row/layer group.
    """
    records = [validate_capture(item) for item in captures]
    rho_values = np.asarray(list(rhos), dtype=np.float64)
    if not records or rho_values.ndim != 1 or not len(rho_values):
        raise ValueError("captures and a finite rho grid are required")
    if (not np.isfinite(rho_values).all() or np.any(rho_values < 1)
            or np.any(np.diff(rho_values) <= 0) or not np.any(rho_values == 1.0)):
        raise ValueError("rho grid must be finite, increasing, and include exact rho=1")
    pair_count = len(records[0].native_inv_freq)
    if any(len(item.native_inv_freq) != pair_count for item in records):
        raise ValueError("all captures must use the same rotary pair count")
    increments_array = np.asarray(list(increments), dtype=np.float64)
    exponents = increments_to_exponents(increments_array, pairs=pair_count)
    depth = float(increments_array.sum())
    exact_active = None
    if active_inv_freq is not None:
        exact_active = np.asarray(list(active_inv_freq), dtype=np.float64)
        if (
            exact_active.shape != (pair_count,) or not np.isfinite(exact_active).all()
            or np.any(exact_active <= 0) or np.any(exact_active[:-1] <= exact_active[1:])
        ):
            raise ValueError("explicit active frequency table is invalid")
    by_rho = {}
    for rho in rho_values:
        grouped: dict[str, list[np.ndarray]] = {}
        all_kl = []
        for record in records:
            active = (
                exact_active if exact_active is not None
                else record.native_inv_freq.astype(np.float64) * np.power(scale, -exponents)
            )
            result = native_attention_kl(
                record.q,
                record.k,
                record.query_positions,
                record.native_inv_freq,
                active,
                gain=gain,
                native_gain=record.reference_gain,
                attention_scale=record.attention_scale,
                position_scale=float(rho),
            )
            values = np.asarray(result["kl"], dtype=np.float64).reshape(-1)
            all_kl.append(values)
            grouped.setdefault(record.group, []).append(values)
        flat = np.concatenate(all_kl)
        group_stats = {}
        for name, chunks in sorted(grouped.items()):
            values = np.concatenate(chunks)
            group_stats[name] = {
                "count": int(len(values)),
                "mean_kl": float(values.mean()),
                "cvar_kl": _cvar(values, cvar_fraction),
                "max_kl": float(values.max()),
            }
        by_rho[format(float(rho), ".12g")] = {
            "count": int(len(flat)),
            "mean_kl": float(flat.mean()),
            "cvar_kl": _cvar(flat, cvar_fraction),
            "worst_group_mean_kl": max(value["mean_kl"] for value in group_stats.values()),
            "groups": group_stats,
        }
    native = by_rho["1"]
    native_worst_group_cvar = max(value["cvar_kl"] for value in native["groups"].values())
    mean_holds = native_mean_limit is None or native["mean_kl"] <= native_mean_limit
    group_holds = (
        native_group_cvar_limit is None
        or native_worst_group_cvar <= native_group_cvar_limit
    )
    return {
        "status": "CHECKPOINT_ATTENTION_REPLAY_COMPLETE_V1",
        "rhos": rho_values.tolist(),
        "risk_by_rho": by_rho,
        "finite_grid_worst_mean_kl": max(value["mean_kl"] for value in by_rho.values()),
        "finite_grid_worst_group_mean_kl": max(
            value["worst_group_mean_kl"] for value in by_rho.values()
        ),
        "native_constraints": {
            "mean_kl": native["mean_kl"],
            "mean_limit": native_mean_limit,
            "worst_group_cvar_kl": native_worst_group_cvar,
            "group_cvar_limit": native_group_cvar_limit,
            "holds": bool(mean_holds and group_holds),
        },
        "profile": {
            "increments": increments_array.tolist(),
            "exponents": exponents.tolist(),
            "tail_depth": depth,
            "zero_increment_indices": np.flatnonzero(increments_array == 0.0).tolist(),
            "scale": float(scale),
            "gain": float(gain),
            "frequency_source": "explicit_frozen_table" if exact_active is not None else "reconstructed_from_profile",
        },
        "cvar_fraction": float(cvar_fraction),
        "scope": "finite declared rho grid; detached checkpoint Q/K transport diagnostic",
    }
