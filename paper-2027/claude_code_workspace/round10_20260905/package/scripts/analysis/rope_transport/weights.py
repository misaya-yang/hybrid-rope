"""Relative-distance supports and weights.

The transport residual is an expectation over relative distance. The weight is
part of the scientific contract, never an implementation detail, so every
supported family is named, deterministic, and recorded in the receipt.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


FAMILIES = ("causal", "uniform", "powerlaw", "empirical")


def _subsample(support: np.ndarray, weight: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic stride subsample that preserves total weight per bin."""
    n = int(support.size)
    if max_points <= 0 or n <= max_points:
        return support, weight
    stride = int(np.ceil(n / float(max_points)))
    edges = np.arange(0, n + stride, stride, dtype=np.int64)
    edges[-1] = n
    keep_idx = []
    keep_w = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        block = weight[lo:hi]
        total = float(block.sum())
        if total <= 0.0:
            continue
        # Representative point is the weighted centroid rounded to an integer
        # distance actually present in the block.
        centroid = float((support[lo:hi] * block).sum() / total)
        pick = int(np.clip(np.round(centroid), support[lo], support[hi - 1]))
        keep_idx.append(pick)
        keep_w.append(total)
    return np.asarray(keep_idx, dtype=np.float64), np.asarray(keep_w, dtype=np.float64)


def distance_weight(
    family: str,
    *,
    length: int,
    min_distance: int = 0,
    alpha: float = 1.0,
    max_points: int = 4096,
    empirical: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(support, weight, meta)`` with ``weight`` summing to one.

    ``causal``   -- exact pair count at distance d in one causal sequence of
                    ``length`` tokens: ``length - d``. This is the content-free
                    ground truth for a packed training sequence.
    ``uniform``  -- flat over the support.
    ``powerlaw`` -- ``(1 + d) ** -alpha``; a registered stand-in for a measured
                    attention-distance profile.
    ``empirical``-- caller-supplied histogram over ``0..length-1``.
    """
    if family not in FAMILIES:
        raise ValueError(f"unsupported weight family: {family}")
    if length <= 1:
        raise ValueError(f"length must exceed 1, got {length}")
    if not 0 <= min_distance < length:
        raise ValueError(f"min_distance out of range: {min_distance}")

    support = np.arange(int(min_distance), int(length), dtype=np.float64)
    if family == "causal":
        weight = float(length) - support
    elif family == "uniform":
        weight = np.ones_like(support)
    elif family == "powerlaw":
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        weight = (1.0 + support) ** (-float(alpha))
    else:
        if empirical is None:
            raise ValueError("family 'empirical' requires an explicit histogram")
        hist = np.asarray(empirical, dtype=np.float64).reshape(-1)
        if hist.size < int(length):
            raise ValueError(f"empirical histogram is shorter than length: {hist.size} < {length}")
        weight = hist[int(min_distance) : int(length)].copy()

    if not np.isfinite(weight).all() or float(weight.sum()) <= 0.0:
        raise ValueError("weight vector must be finite and positive in total")
    weight = np.clip(weight, 0.0, None)

    n_full = int(support.size)
    support, weight = _subsample(support, weight, int(max_points))
    weight = weight / float(weight.sum())

    meta = {
        "family": str(family),
        "length": int(length),
        "min_distance": int(min_distance),
        "alpha": float(alpha) if family == "powerlaw" else None,
        "support_points_full": n_full,
        "support_points_used": int(support.size),
        "max_points": int(max_points),
        "max_distance": float(support.max()),
    }
    return support, weight, meta
