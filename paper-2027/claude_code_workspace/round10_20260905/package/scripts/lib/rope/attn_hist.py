#!/usr/bin/env python3
"""Online distance-histogram accumulation and power-law fitting."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import torch


def accumulate_distance_histogram(
    q: torch.Tensor,
    k: torch.Tensor,
    query_positions: torch.Tensor,
    max_distance: int,
    hist: torch.Tensor,
    block_q: int = 128,
) -> None:
    """
    Accumulate attention mass by distance.

    q: [H, Q, D]
    k: [H, K, D]
    query_positions: [Q] absolute positions
    hist: [max_distance + 1] float tensor
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"Expected q/k rank-3, got q={q.shape} k={k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"Head/dim mismatch: q={q.shape} k={k.shape}")

    h, qn, d = q.shape
    kn = k.shape[1]
    device = q.device
    k_pos = torch.arange(kn, device=device, dtype=torch.long)
    scale = 1.0 / math.sqrt(float(d))

    for s in range(0, qn, block_q):
        e = min(qn, s + block_q)
        q_blk = q[:, s:e, :]  # [H, B, D]
        pos_blk = query_positions[s:e]  # [B]
        logits = torch.einsum("hbd,hkd->hbk", q_blk, k) * scale

        # Causal masking: key position must be <= query position.
        causal = k_pos.unsqueeze(0) <= pos_blk.unsqueeze(1)
        logits = logits.masked_fill(~causal.unsqueeze(0), float("-inf"))
        attn = torch.softmax(logits, dim=-1)  # [H, B, K]

        dists = torch.abs(pos_blk.unsqueeze(1) - k_pos.unsqueeze(0))  # [B, K]
        dists = torch.clamp(dists, min=0, max=max_distance)

        dist_flat = dists.unsqueeze(0).expand(h, -1, -1).reshape(-1)
        weight_flat = attn.reshape(-1).to(hist.dtype)
        hist.scatter_add_(0, dist_flat, weight_flat)


def fit_power_law(
    hist: np.ndarray,
    d_min: int = 8,
    d_max: int | None = None,
) -> Dict[str, float | int | None]:
    if hist.ndim != 1:
        raise ValueError("hist must be 1D")
    n = hist.shape[0]
    if d_max is None:
        d_max = max(d_min + 16, int(0.5 * n))
    d_max = min(d_max, n - 1)

    dist = np.arange(d_min, d_max + 1, dtype=np.float64)
    vals = hist[d_min : d_max + 1].astype(np.float64)
    mask = np.isfinite(vals) & (vals > 0)
    if int(mask.sum()) < 8:
        return {"alpha": None, "r2": None, "n_points": int(mask.sum())}

    x = np.log(dist[mask])
    y = np.log(vals[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(-slope), "r2": float(r2), "n_points": int(mask.sum())}


def bootstrap_alpha_ci(
    per_sample_hists: Iterable[np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
    d_min: int = 8,
    d_max: int | None = None,
) -> Dict[str, float | None]:
    hists = [np.asarray(h, dtype=np.float64) for h in per_sample_hists]
    if not hists:
        return {"alpha_mean": None, "alpha_ci_low": None, "alpha_ci_high": None}
    if len(hists) == 1:
        fit = fit_power_law(hists[0], d_min=d_min, d_max=d_max)
        a = fit["alpha"]
        return {"alpha_mean": a, "alpha_ci_low": a, "alpha_ci_high": a}

    rng = np.random.default_rng(seed)
    n = len(hists)
    alphas: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        h = np.mean([hists[i] for i in idx], axis=0)
        fit = fit_power_law(h, d_min=d_min, d_max=d_max)
        a = fit["alpha"]
        if isinstance(a, float) and np.isfinite(a):
            alphas.append(float(a))

    if not alphas:
        return {"alpha_mean": None, "alpha_ci_low": None, "alpha_ci_high": None}
    arr = np.asarray(alphas, dtype=np.float64)
    return {
        "alpha_mean": float(np.mean(arr)),
        "alpha_ci_low": float(np.percentile(arr, 2.5)),
        "alpha_ci_high": float(np.percentile(arr, 97.5)),
    }
