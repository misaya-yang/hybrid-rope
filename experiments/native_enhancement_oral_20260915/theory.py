#!/usr/bin/env python3
"""Reproduce native/extension theory checks without torch or model execution.

The response convention is q.T @ R(theta) @ k, where R is the standard
two-dimensional counterclockwise rotation. This module does not extract model
activations: an actual model adapter must verify its rotary sign and layout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from experiments.native_contrastive_proximal_20260915.tables import (
    MAX_LOG_SHIFT,
    PROXIMAL_CURVATURE,
    reference_fourier,
    reference_log_gradient,
)


def _finite(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("inputs must be finite")
    return result


def rotation(phase: Any) -> np.ndarray:
    """Return full 2-by-2 rotary blocks, retaining sine/cosine cross terms."""
    phase = _finite(phase)
    cosine, sine = np.cos(phase), np.sin(phase)
    return np.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
        phase.shape + (2, 2),
    )


def pair_coefficients(query: Any, key: Any) -> tuple[np.ndarray, ...]:
    """Return C, D, A for C cos(theta) + D sin(theta), with pairs last."""
    query, key = np.broadcast_arrays(_finite(query), _finite(key))
    if query.ndim < 2 or query.shape[-1] != 2:
        raise ValueError("query and key must end in (pairs, 2)")
    cosine = np.sum(query * key, axis=-1)
    sine = query[..., 1] * key[..., 0] - query[..., 0] * key[..., 1]
    return cosine, sine, np.hypot(cosine, sine)


def signed_margin_change(
    query: Any, correct_key: Any, distractor_key: Any,
    correct_distance: Any, distractor_distance: Any,
    native_frequencies: Any, candidate_frequencies: Any, *, scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Exact fixed-content margin change and a finite phase Taylor certificate.

    Q/K arrays end in (pairs, 2); distances are scalars or have the shared batch
    shape. Frequencies are positive one-dimensional tables of that pair count.
    Scale is the shared positive logit scale, including any shared rotary gain.
    All calculations hold the supplied Q/K fixed; no network guarantee follows.
    """
    native, candidate = _finite(native_frequencies), _finite(candidate_frequencies)
    if (native.ndim != 1 or not native.size or candidate.shape != native.shape
            or np.any(native <= 0) or np.any(candidate <= 0)):
        raise ValueError("frequency tables must have matching positive pair entries")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("shared scale must be finite and positive")
    q, positive, negative = np.broadcast_arrays(
        _finite(query), _finite(correct_key), _finite(distractor_key),
    )
    if q.ndim < 2 or q.shape[-2:] != (native.size, 2):
        raise ValueError("query and keys must end in (table pair count, 2)")
    batch = q.shape[:-2]

    def response(key: np.ndarray, distances: Any) -> tuple[np.ndarray, ...]:
        distance = np.broadcast_to(_finite(distances), batch)[..., None]
        theta, delta = distance * native, distance * (candidate - native)
        c, d, amplitude = pair_coefficients(q, key)
        old = c * np.cos(theta) + d * np.sin(theta)
        new = c * np.cos(theta + delta) + d * np.sin(theta + delta)
        linear = (-c * np.sin(theta) + d * np.cos(theta)) * delta
        remainder = 0.5 * amplitude * delta**2
        return old, new, linear, remainder

    p, n = response(positive, correct_distance), response(negative, distractor_distance)
    old = scale * np.sum(p[0] - n[0], axis=-1)
    new = scale * np.sum(p[1] - n[1], axis=-1)
    pair_change = scale * ((p[1] - p[0]) - (n[1] - n[0]))
    exact = np.sum(pair_change, axis=-1)
    linear = scale * np.sum(p[2] - n[2], axis=-1)
    bound = scale * np.sum(p[3] + n[3], axis=-1)
    return {
        "native_margin": old, "candidate_margin": new,
        "pair_exact_change": pair_change, "exact_change": exact,
        "linear_change": linear, "remainder_bound": bound,
        "lower": linear - bound, "upper": linear + bound,
    }


def opposite_content_counterexample() -> dict[str, Any]:
    """One fixed-support slowdown helps one content pair and harms another.

    Only the middle of three frequencies changes; both endpoint frequencies and
    slot assignments stay fixed. All active query/key vectors have unit norm,
    the distractor is the negative correct key, and both Native margins are
    positive. This is an algebraic construction, not a model/task result.
    """
    native, candidate = np.array([1.0, 0.5, 0.1]), np.array([1.0, 0.25, 0.1])
    query = np.zeros((3, 2))
    query[1, 0] = 1.0
    aligned = query.copy()
    learned_phase = np.zeros_like(query)
    learned_phase[1] = [math.cos(1.0), -math.sin(1.0)]
    examples = {}
    for name, correct in (("pre_rotation_aligned", aligned),
                          ("native_phase_matched", learned_phase)):
        result = signed_margin_change(query, correct, -correct, 2.0, 2.0, native, candidate)
        examples[name] = {key: float(result[key]) for key in (
            "native_margin", "candidate_margin", "exact_change",
        )}
    return {"native": native.tolist(), "candidate": candidate.tolist(),
            "distance": 2.0, "examples": examples}


def _scalar_shift(phase: float, coefficients: np.ndarray) -> float:
    """Independent Brent solve for the inactive-gap NCP coordinate problem."""
    def gradient(u: float) -> float:
        return PROXIMAL_CURVATURE * u - float(reference_log_gradient(
            phase * math.exp(-u), coefficients=coefficients,
        ))
    return 0.0 if gradient(0.0) >= 0 else brentq(
        gradient, 0.0, MAX_LOG_SHIFT, xtol=1e-15,
    )


def audit_theory() -> dict[str, Any]:
    """Return deterministic numerical evidence, with mathematical scope intact."""
    points = 2048
    angles = 2 * np.pi * np.arange(points) / points
    cosine = np.cos(angles)
    risk = np.logaddexp(0, cosine[None, :] - cosine[:, None]).mean(axis=1)
    spectrum = 2 * np.fft.rfft(risk).real / points
    _, coefficients = reference_fourier(grid_points=points)
    reflection = risk - risk[(points // 2 - np.arange(points)) % points] + cosine
    m = float(expit(cosine - 1).mean())
    slow = []
    for phase in (0.1, 0.01, 0.001):
        shift = _scalar_shift(phase, coefficients)
        slow.append({"phase": phase, "u": shift, "u_over_phase_squared": shift / phase**2})
    fast = []
    for j in (100, 1000, 10000):
        phase = 1.5 * math.pi + 2 * math.pi * j
        shift = _scalar_shift(phase, coefficients)
        movement = phase * -math.expm1(-shift)
        fast.append({"j": j, "phase": phase, "u": shift,
                     "full_window_phase_change": movement,
                     "rotation_operator_difference": 2 * abs(math.sin(movement / 2))})
    mathematical_native = 500000.0 ** (-np.arange(64) / 64)
    shifts = np.array([_scalar_shift(float(4095 * value), coefficients)
                       for value in mathematical_native])
    shifts[[0, -1]] = 0

    rng = np.random.default_rng(20260915)
    rows, pairs = 10000, 8
    queries = rng.normal(size=(rows, pairs, 2))
    correct, distractor = rng.normal(size=(2, rows, pairs, 2))
    native = np.geomspace(1.0, 0.01, pairs)
    candidate = native * np.exp(-np.linspace(0.0, 0.15, pairs))
    candidate[[0, -1]] = native[[0, -1]]
    distances = rng.uniform(0, 50, size=(2, rows))
    result = signed_margin_change(queries, correct, distractor, *distances, native, candidate)
    residual = np.abs(result["exact_change"] - result["linear_change"])
    bound = result["remainder_bound"]
    phase, delta = rng.normal(size=(2, 5000))
    op = np.linalg.svd(rotation(phase + delta) - rotation(phase), compute_uv=False)[..., 0]
    certificate = result["lower"] > 0
    checks = {
        "bound_violations": int(np.count_nonzero(residual > bound + 1e-12)),
        "false_positive_certificates": int(np.count_nonzero(certificate & (result["exact_change"] <= 0))),
    }
    if any(checks.values()):
        raise RuntimeError(f"finite-margin identity failed: {checks}")
    return {
        "status": "NATIVE_EXTENSION_THEORY_CPU_V1", "model_execution": False,
        "seed": 20260915, "fourier_points": points, "fourier_modes": len(coefficients),
        "reflection_identity_max_error": float(np.abs(reflection).max()),
        "first_harmonic": float(spectrum[1]),
        "higher_odd_harmonics_max_abs": float(np.abs(spectrum[3:33:2]).max()),
        "slow_phase_coefficient": 2 * m / 27, "slow_phase_examples": slow,
        "high_phase_examples": fast,
        "mathematical_fp64_olmo_grid": {
            "maximum_relative_slowdown": float(np.max(-np.expm1(-shifts))),
            "peak_pair": int(np.argmax(shifts)),
            "runtime_table_identity_claimed": False,
        },
        "margin_audit": {"rows": rows, "pairs": pairs, **checks,
                         "maximum_remainder_bound_ratio": float((residual / bound).max()),
                         "positive_certificates": int(np.count_nonzero(certificate))},
        "rotation_identity_max_error": float(np.abs(op - 2 * np.abs(np.sin(delta / 2))).max()),
        "counterexample": opposite_content_counterexample(),
        "claim_boundary": (
            "Reference-risk asymptotics, explicit full-pair algebra, and synthetic "
            "fixed-content certificates; not Transformer NLL, task results, a proof "
            "of asymptotic convergence from finite samples, or a universal gain guarantee."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional new JSON receipt; existing files are not overwritten")
    args = parser.parse_args()
    payload = json.dumps(audit_theory(), indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
