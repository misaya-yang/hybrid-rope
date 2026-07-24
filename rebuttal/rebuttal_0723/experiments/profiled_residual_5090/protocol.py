#!/usr/bin/env python3
"""Frozen schedules for the target-aware profiled-residual diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    ARMS as SOURCE_ARMS,
    SPEC as SOURCE_SPEC,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.real_rope_schedules import (
    SCHEDULES as REAL_ROPE_SCHEDULES,
    SOURCE as REAL_ROPE_SOURCE,
)


GATE_ARM = "fmrope_base256"
EVAL_LENGTHS = (256, 1_024, 2_048, 4_096, 8_192)
BASE_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
LAMBDA_RATIOS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
PROFILE_FAMILIES = ("base", "raw_evq", "cosh_residual", "band_residual")
TAIL_TOKENS = 128


@dataclass(frozen=True)
class ProfileSpec:
    source_protocol_sha256: str = SOURCE_SPEC.fingerprint()
    head_dim: int = SOURCE_SPEC.head_dim
    train_length: int = SOURCE_SPEC.train_length
    evq_base: float = SOURCE_SPEC.evq_base
    evq_tau: float = SOURCE_SPEC.evq_tau
    eval_lengths: tuple[int, ...] = EVAL_LENGTHS
    base_multipliers: tuple[float, ...] = BASE_MULTIPLIERS
    lambda_ratios: tuple[float, ...] = LAMBDA_RATIOS
    tail_tokens: int = TAIL_TOKENS
    selection_anchors: int = 16
    test_anchors: int = 32
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_724

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


SPEC = ProfileSpec()


def _affine_projection(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a channel vector onto span{1, channel index}."""
    value = values.detach().to(dtype=torch.float64, device="cpu").contiguous()
    if value.ndim != 1 or value.numel() < 3 or not torch.isfinite(value).all():
        raise ValueError("affine projection requires a finite one-dimensional vector")
    coordinate = torch.linspace(0.0, 1.0, value.numel(), dtype=torch.float64)
    design = torch.stack((torch.ones_like(coordinate), coordinate), dim=1)
    coefficients = torch.linalg.lstsq(design, value[:, None]).solution[:, 0]
    fitted = design @ coefficients
    residual = value - fitted
    if not torch.allclose(value, fitted + residual, atol=1e-12, rtol=0.0):
        raise RuntimeError("affine decomposition does not reconstruct the input")
    if not torch.allclose(
        design.T @ residual,
        torch.zeros(2, dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("residual is not orthogonal to the affine subspace")
    return coefficients, residual


def frequency_components() -> dict[str, torch.Tensor | dict[str, float]]:
    """Return the actual EVQ affine/raw/residual components in log-frequency."""
    geo_inv = training_inv_freq("paper_geo_base500k").double()
    evq_inv = training_inv_freq("evq_cosh_tau4_paper_grid_base500k").double()
    q_geo = -torch.log(geo_inv)
    q_full = -torch.log(evq_inv)
    coefficients, cosh_residual = _affine_projection(q_full)
    coordinate = torch.linspace(0.0, 1.0, q_full.numel(), dtype=torch.float64)
    q_affine = coefficients[0] + coefficients[1] * coordinate

    if (
        REAL_ROPE_SOURCE["head_dim"] != SPEC.head_dim
        or not math.isclose(float(REAL_ROPE_SOURCE["base"]), SPEC.evq_base)
    ):
        raise ValueError("frozen two-band schedule does not match the profile contract")
    band_phi = torch.tensor(
        REAL_ROPE_SCHEDULES["attention_kernel_stdgeo42_span_matched"],
        dtype=torch.float64,
    )
    _, band_residual = _affine_projection(math.log(SPEC.evq_base) * band_phi)
    band_norm = torch.linalg.vector_norm(band_residual)
    cosh_norm = torch.linalg.vector_norm(cosh_residual)
    if float(band_norm) <= 0.0 or float(cosh_norm) <= 0.0:
        raise RuntimeError("nonlinear residual norm must be positive")
    band_residual = band_residual * (cosh_norm / band_norm)

    return {
        "q_geo": q_geo,
        "q_affine": q_affine,
        "q_full": q_full,
        "raw_evq_deformation": q_full - q_geo,
        "cosh_residual": cosh_residual,
        "band_residual": band_residual,
        "affine": {
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
        },
    }


def native_geometric_q(base: float) -> torch.Tensor:
    """Native endpoint geometric log-frequency; recovers FMR training RoPE."""
    base_f = float(base)
    if not math.isfinite(base_f) or base_f <= 1.0:
        raise ValueError(f"base must be finite and >1, got {base!r}")
    k = torch.arange(SPEC.head_dim // 2, dtype=torch.float64)
    return math.log(base_f) * k / float(SPEC.head_dim // 2)


def candidate_q(
    *, family: str, length: int, base_multiplier: float, lambda_ratio: float = 0.0
) -> torch.Tensor:
    """Build one nested BR or BR-plus-deformation candidate."""
    if family not in PROFILE_FAMILIES:
        raise ValueError(f"unknown profile family {family!r}")
    length_i = int(length)
    multiplier = float(base_multiplier)
    if length_i not in SPEC.eval_lengths:
        raise ValueError(f"length must be one of {SPEC.eval_lengths}")
    if multiplier not in SPEC.base_multipliers:
        raise ValueError(f"base multiplier must be one of {SPEC.base_multipliers}")
    q = native_geometric_q(length_i * multiplier)
    components = frequency_components()
    if family == "raw_evq":
        if float(lambda_ratio) != 1.0:
            raise ValueError("raw EVQ deformation is registered only at strength 1")
        q = q + components["raw_evq_deformation"]  # type: ignore[operator]
    elif family == "cosh_residual":
        q = q + float(lambda_ratio) * components["cosh_residual"]  # type: ignore[operator]
    elif family == "band_residual":
        q = q + float(lambda_ratio) * components["band_residual"]  # type: ignore[operator]
    elif float(lambda_ratio) != 0.0:
        raise ValueError("base family requires lambda_ratio=0")
    return q.contiguous()


def candidate_inv_freq(**kwargs: Any) -> torch.Tensor:
    q = candidate_q(**kwargs)
    if not torch.isfinite(q).all():
        raise ValueError("candidate log-frequency table is not finite")
    inv = torch.exp(-q).float().contiguous()
    if not torch.isfinite(inv).all() or torch.any(inv <= 0):
        raise ValueError("candidate inverse frequency is not finite and positive")
    return inv


def candidate_grid(length: int) -> list[dict[str, float | str]]:
    """Return every valid candidate; invalid negative residuals stay auditable."""
    rows: list[dict[str, float | str]] = []
    for multiplier in SPEC.base_multipliers:
        candidates = [("base", 0.0), ("raw_evq", 1.0)]
        candidates.extend(
            (family, ratio)
            for family in ("cosh_residual", "band_residual")
            for ratio in SPEC.lambda_ratios
        )
        for family, ratio in candidates:
            try:
                inv = candidate_inv_freq(
                    family=family,
                    length=length,
                    base_multiplier=multiplier,
                    lambda_ratio=ratio,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "family": family,
                    "base_multiplier": float(multiplier),
                    "lambda_ratio": float(ratio),
                    "inv_freq_sha256": hashlib.sha256(inv.numpy().tobytes()).hexdigest(),
                    "strictly_monotonic": bool(torch.all(torch.diff(-torch.log(inv.double())) > 0)),
                }
            )
    return rows


def _exact_collision(inv_freq: torch.Tensor, length: int) -> float:
    omega = inv_freq.detach().cpu().double().numpy()
    distance = np.arange(int(length), dtype=np.float64)
    features = np.cos(np.outer(distance, omega))
    gram = features.T @ features / float(len(distance))
    diagonal = np.diag(gram)
    normalized = gram / np.sqrt(np.outer(diagonal, diagonal))
    return float(np.square(np.triu(normalized, k=1)).sum())


def decomposition_report() -> dict[str, Any]:
    components = frequency_components()
    tables = {
        "paper_geo": components["q_geo"],
        "affine_evq": components["q_affine"],
        "full_evq": components["q_full"],
    }
    rows: dict[str, Any] = {}
    for name, raw_q in tables.items():
        q = raw_q  # type: ignore[assignment]
        inv = torch.exp(-q)
        rows[name] = {
            "min_frequency": float(inv.min()),
            "max_frequency": float(inv.max()),
            "log_frequency_span": float(q[-1] - q[0]),
            "inv_freq_sha256_float64": hashlib.sha256(
                inv.numpy().tobytes()
            ).hexdigest(),
            "channels": [
                {"index": index, "q": float(q[index]), "omega": float(inv[index])}
                for index in range(q.numel())
            ],
            "by_length": {
                str(length): {
                    "exact_cosine_collision": _exact_collision(inv, length),
                    "alive_channels_omega_L_ge_1": int((inv * length >= 1.0).sum()),
                }
                for length in SPEC.eval_lengths
            },
        }
    cosh_residual = components["cosh_residual"]
    band_residual = components["band_residual"]
    return {
        "schema_version": 1,
        "protocol_sha256": SPEC.fingerprint(),
        "decomposition": "least-squares orthogonal projection onto span{1, channel_index}",
        "affine": components["affine"],
        "cosh_residual_l2": float(torch.linalg.vector_norm(cosh_residual)),
        "cosh_residual_rms": float(torch.sqrt(torch.mean(cosh_residual.square()))),
        "cosh_residual_max_abs": float(cosh_residual.abs().max()),
        "band_residual_l2_after_norm_match": float(torch.linalg.vector_norm(band_residual)),
        "tables": rows,
        "warning": "frequency-table norm fractions are not performance attribution",
    }
