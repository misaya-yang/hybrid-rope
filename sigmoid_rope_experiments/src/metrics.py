from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


def _sample_log_distances(L: int, num_samples: int, device: torch.device) -> torch.Tensor:
    d = torch.unique(torch.logspace(0, math.log10(float(L)), num_samples, device=device).long())
    d = d[d > 0]
    return d


def phase_collision_score(
    freqs: torch.Tensor,
    L: int,
    d: int,
    num_samples: int = 5000,
    device: torch.device | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute phase collision score for one frequency vector.
    Lower is better.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distances = _sample_log_distances(L=L, num_samples=num_samples, device=device)
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(dtype=torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    inner = torch.cos(angles).mean(dim=-1)  # (D,)

    short_mask = distances <= 100
    mid_mask = (distances > 100) & (distances <= 10000)
    long_mask = distances > 10000

    score_short = inner[short_mask].abs().mean().item() if short_mask.any() else 0.0
    score_mid = inner[mid_mask].abs().mean().item() if mid_mask.any() else 0.0
    score_long = inner[long_mask].abs().mean().item() if long_mask.any() else 0.0
    total = 0.2 * score_short + 0.3 * score_mid + 0.5 * score_long

    del distances, freqs_f32, angles, inner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(total), {
        "short": float(score_short),
        "mid": float(score_mid),
        "long": float(score_long),
        "total": float(total),
    }


def phase_collision_score_batch(
    freqs_batch: torch.Tensor,
    L: int,
    num_samples: int = 2000,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Vectorized phase collision score for batch of frequency vectors.

    Args:
        freqs_batch: (B, N), float64 preferred
    Returns:
        total_scores: (B,)
        breakdown: dict with short/mid/long/total tensors (B,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if freqs_batch.ndim != 2:
        raise ValueError(f"freqs_batch must be (B,N), got {tuple(freqs_batch.shape)}")

    distances = _sample_log_distances(L=L, num_samples=num_samples, device=device).to(torch.float32)
    fb = freqs_batch.to(device=device, dtype=torch.float32)
    # angles: (D, B, N)
    angles = distances[:, None, None] * fb[None, :, :]
    inner = torch.cos(angles).mean(dim=-1)  # (D, B)
    inner_abs = inner.abs()

    short_mask = distances <= 100
    mid_mask = (distances > 100) & (distances <= 10000)
    long_mask = distances > 10000

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return inner_abs[mask].mean(dim=0)
        return torch.zeros(fb.shape[0], device=device, dtype=torch.float32)

    score_short = masked_mean(short_mask)
    score_mid = masked_mean(mid_mask)
    score_long = masked_mean(long_mask)
    total = 0.2 * score_short + 0.3 * score_mid + 0.5 * score_long

    breakdown = {
        "short": score_short.detach().cpu(),
        "mid": score_mid.detach().cpu(),
        "long": score_long.detach().cpu(),
        "total": total.detach().cpu(),
    }
    total_cpu = total.detach().cpu()

    del distances, fb, angles, inner, inner_abs, score_short, score_mid, score_long, total
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return total_cpu, breakdown


def compute_phase_collision_curve(
    freqs: torch.Tensor,
    L: int,
    num_points: int = 2000,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distances = torch.unique(torch.logspace(math.log10(1.0), math.log10(float(L)), num_points, device=device).long())
    distances = distances[distances > 0]
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    collisions = torch.cos(angles).mean(dim=-1)

    out_d = distances.detach().cpu().numpy()
    out_c = collisions.detach().cpu().numpy()

    del distances, freqs_f32, angles, collisions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_d, out_c


def compute_attention_score_decay(
    freqs: torch.Tensor,
    L: int,
    d: int,
    max_distance: int = 50000,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upper = int(min(L, max_distance))
    distances = torch.arange(0, upper, dtype=torch.long, device=device)
    freqs_f32 = freqs.to(device=device, dtype=torch.float32)
    angles = distances.to(torch.float32).unsqueeze(-1) * freqs_f32.unsqueeze(0)
    scores = torch.cos(angles).mean(dim=-1)
    return distances.detach().cpu().numpy(), scores.detach().cpu().numpy()

