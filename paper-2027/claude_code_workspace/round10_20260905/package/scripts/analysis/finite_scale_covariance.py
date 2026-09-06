#!/usr/bin/env python3
"""Validate finite scale-covariance bounds without a model or GPU.

This is a theorem-audit harness, not a frequency selector.  It covers exact
orbit growth, coherent approximate matching, continuous/discrete Fourier-orbit
rank, the real-orthogonal signed-spectrum extension, approximate discrete
aliasing, and sharp/equality constructions.  Optional frozen RoPE tables are
read from an existing scale-orbit manifest; no table is fitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.scale_orbit_validation import (  # noqa: E402
    continuous_gram,
    discrete_gram,
    ky_fan_lower_bound,
)


STATUS = "FINITE_SCALE_COVARIANCE_AUDIT_COMPLETE"
METHOD_ID = "finite_scale_covariance_audit_v2"
EXECUTABLE_COVERAGE = (
    "exact_positive_scaling_trilemma_and_s_minus_one_exception",
    "theorem3_exact_orbit_growth",
    "theorem3_one_and_two_sided_boundary",
    "theorem3_equality_and_packed_chain_sharpness",
    "theorem4_multilevel_coherent_partial_matching",
    "theorem4_two_sided_coherent_partial_matching",
    "theorem4_dimension_span_inversion",
    "continuous_operator_error_to_log_tolerance",
    "theorem5_complex_continuous_and_discrete_ky_fan_bound",
    "theorem5_standard_rope_real_orthogonal_signed_spectrum_specialization",
    "theorem5_coherence_and_separated_frequency_corollaries",
    "theorem5_montgomery_vaughan_dimension_free_separation_corollary",
    "theorem5_orbit_growth_coherence_combination",
    "theorem5_arbitrary_invertible_mixing_l2_diagnostic",
    "theorem5_orthogonal_character_sharpness",
    "exact_discrete_periodicity_and_coprimality",
    "theorem6_approximate_alias",
    "theorem6_anti_alias_corollary",
    "commensurate_and_incommensurate_uniform_log_allocation",
    "multi_generator_uniform_log_allocation",
    "optional_hash_bound_frozen_rope_table_panels",
)
NON_EXECUTABLE_GATES = {
    "supplied_source_scope": "Only Pro sections 5--12 were supplied; sections 1--4 cannot be audited.",
    "formal_proof": "Numerical checks do not prove Theorems 3--6.",
    "general_real_orthogonal_proof": (
        "The numerical signed-spectrum panel covers standard RoPE blocks; the general proof is non-executable."
    ),
    "separation_best_constant": (
        "The Montgomery--Vaughan corollary has sharp inverse-(L delta) order, not a claimed optimal constant."
    ),
    "matching_upper_construction": (
        "The PCA/Ky-Fan equality control is not an operator-similarity upper construction."
    ),
    "literature_novelty": "Requires a primary-source literature audit.",
    "behaviour_bridge": "Requires readback of canonical experiment owners, not a synthetic theorem run.",
    "paper_potential": "Requires proof, novelty, and behavioural-evidence synthesis.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require_cpu_only() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise RuntimeError("CPU theorem audit requires CUDA_VISIBLE_DEVICES=-1 or unset")
    if "torch" in sys.modules:
        raise RuntimeError("the CPU theorem audit must not import torch")
    return {"cuda_visible_devices": visible, "torch_imported": False}


def _fraction(value: int | str | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _residue(value: Fraction, alpha: Fraction) -> Fraction:
    quotient = math.floor(value / alpha)
    return value - quotient * alpha


def _residue_chains(
    points: Sequence[Fraction], alpha: Fraction
) -> list[dict[str, Any]]:
    groups: dict[Fraction, list[int]] = {}
    for point in points:
        residue = _residue(point, alpha)
        groups.setdefault(residue, []).append(int((point - residue) / alpha))
    chains = []
    for residue, indices in sorted(groups.items()):
        indices.sort()
        chains.append({
            "residue": str(residue),
            "size": len(indices),
            "index_min": indices[0],
            "index_max": indices[-1],
            "holes": indices[-1] - indices[0] + 1 - len(indices),
        })
    return chains


def exact_orbit_report(
    values: Iterable[int | str | Fraction],
    alpha: int | str | Fraction,
    levels: int,
) -> dict[str, Any]:
    """Theorem 3 and its one-/two-sided boundary consequences, exactly."""

    points = sorted({_fraction(value) for value in values})
    step = _fraction(alpha)
    if not points or step <= 0 or levels < 0:
        raise ValueError("nonempty distinct points, positive alpha, and levels >= 0 required")
    span = points[-1] - points[0]
    classes = len({_residue(point, step) for point in points})
    capacity = min(len(points), math.floor(span / step) + 1)
    orbit = {
        point - level * step
        for point in points
        for level in range(levels + 1)
    }
    two_sided = {
        point - level * step
        for point in points
        for level in range(-levels, levels + 1)
    }
    boundary = {point - step for point in points} - set(points)
    chains = _residue_chains(points, step)
    first_bound = len(points) + levels * classes
    global_bound = len(points) + levels * math.ceil(len(points) / capacity)
    return {
        "K": len(points),
        "span": str(span),
        "alpha": str(step),
        "levels": levels,
        "residue_classes": classes,
        "capacity": capacity,
        "orbit_modes": len(orbit),
        "first_lower_bound": first_bound,
        "global_lower_bound": global_bound,
        "first_bound_holds": len(orbit) >= first_bound,
        "global_bound_holds": len(orbit) >= global_bound,
        "boundary_count": len(boundary),
        "boundary_bound_holds": len(boundary) >= classes,
        "two_sided_modes": len(two_sided),
        "two_sided_bound": len(points) + 2 * levels * classes,
        "two_sided_bound_holds": len(two_sided) >= len(points) + 2 * levels * classes,
        "first_bound_equality": len(orbit) == first_bound,
        "all_residue_classes_are_consecutive": all(chain["holes"] == 0 for chain in chains),
        "chains": chains,
    }


def packed_survivors(total: int, capacity: int, levels: int) -> int:
    if total < 0 or capacity <= 0 or levels < 0:
        raise ValueError("total >= 0, capacity > 0, and levels >= 0 required")
    full, remainder = divmod(total, capacity)
    return full * max(capacity - levels, 0) + max(remainder - levels, 0)


def packed_chain_lengths(total: int, capacity: int) -> list[int]:
    full, remainder = divmod(total, capacity)
    return [capacity] * full + ([remainder] if remainder else [])


def _partitions_bounded(total: int, maximum: int) -> Iterable[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    for first in range(min(total, maximum), 0, -1):
        for rest in _partitions_bounded(total - first, first):
            yield (first, *rest)


def verify_packing_formula(max_total: int = 14) -> dict[str, Any]:
    cases = 0
    equality_cases = 0
    for total in range(1, max_total + 1):
        for capacity in range(1, total + 1):
            for levels in range(0, capacity + 3):
                observed = max(
                    sum(max(length - levels, 0) for length in partition)
                    for partition in _partitions_bounded(total, capacity)
                )
                expected = packed_survivors(total, capacity, levels)
                if observed != expected:
                    raise AssertionError((total, capacity, levels, observed, expected))
                cases += 1
                equality_cases += int(
                    sum(
                        max(length - levels, 0)
                        for length in packed_chain_lengths(total, capacity)
                    )
                    == expected
                )
    return {
        "status": "PASS",
        "cases": cases,
        "packed_construction_equality_cases": equality_cases,
        "max_total": max_total,
    }


def _survivors(mapping: Sequence[int | None], levels: int) -> int:
    count = 0
    for start in range(len(mapping)):
        current: int | None = start
        for _ in range(levels):
            current = None if current is None else mapping[current]
            if current is None:
                break
        else:
            count += 1
    return count


def _two_sided_survivors(mapping: Sequence[int | None], levels: int) -> int:
    inverse = {target: source for source, target in enumerate(mapping) if target is not None}
    return sum(
        _survivors_from(mapping, start, levels)
        and _survivors_from(inverse, start, levels)
        for start in range(len(mapping))
    )


def _survivors_from(
    mapping: Sequence[int | None] | dict[int, int], start: int, levels: int
) -> bool:
    current: int | None = start
    for _ in range(levels):
        current = mapping.get(current) if isinstance(mapping, dict) else mapping[current]
        if current is None:
            return False
    return True


def best_coherent_matching(
    points: Sequence[float], alpha: float, tau: float, levels: int, *, two_sided: bool = False
) -> dict[str, Any]:
    """Brute-force the coherent partial injection for small theorem tests."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 1 or not 0 < len(values) <= 9:
        raise ValueError("coherent brute force requires one to nine points")
    if not np.all(values[:-1] < values[1:]) or not 0 <= tau < alpha:
        raise ValueError("points must increase and 0 <= tau < alpha")
    candidates = [
        [
            target
            for target in range(len(values))
            if abs(values[target] - (values[source] - alpha)) <= tau + 1e-12
        ]
        for source in range(len(values))
    ]
    best_count = -1
    best_map: list[int | None] = [None] * len(values)
    mapping: list[int | None] = [None] * len(values)

    def visit(source: int, used_targets: set[int]) -> None:
        nonlocal best_count, best_map
        if source == len(values):
            score = (
                _two_sided_survivors(mapping, levels)
                if two_sided
                else _survivors(mapping, levels)
            )
            if score > best_count:
                best_count, best_map = score, list(mapping)
            return
        mapping[source] = None
        visit(source + 1, used_targets)
        for target in candidates[source]:
            if target in used_targets:
                continue
            mapping[source] = target
            used_targets.add(target)
            visit(source + 1, used_targets)
            used_targets.remove(target)
        mapping[source] = None

    visit(0, set())
    beta = alpha - tau
    span = float(values[-1] - values[0])
    capacity = min(len(values), math.floor(span / beta) + 1)
    effective_levels = levels * (2 if two_sided else 1)
    theorem_bound = packed_survivors(len(values), capacity, effective_levels)
    if best_count > theorem_bound:
        raise AssertionError("Theorem 4 survivor bound violated")
    return {
        "K": len(values),
        "span": span,
        "alpha": alpha,
        "tau": tau,
        "beta": beta,
        "levels": levels,
        "directions": 2 if two_sided else 1,
        "effective_chain_steps": effective_levels,
        "capacity": capacity,
        "best_survivors": best_count,
        "theorem_survivor_bound": theorem_bound,
        "leakage": 1.0 - best_count / len(values),
        "simple_leakage_lower_bound": min(1.0, effective_levels / capacity),
        "mapping": best_map,
    }


def dimension_span_requirements(
    levels: int, leakage: float, alpha: float, tau: float, *, directions: int = 1
) -> dict[str, Any]:
    if (
        levels <= 0
        or directions not in (1, 2)
        or not 0 < leakage < 1
        or not 0 <= tau < alpha
    ):
        raise ValueError(
            "levels > 0, directions in {1,2}, 0 < leakage < 1, and 0 <= tau < alpha required"
        )
    effective_levels = directions * levels
    capacity = math.ceil(effective_levels / leakage)
    return {
        "directions": directions,
        "effective_chain_steps": effective_levels,
        "minimum_dimension": capacity,
        "minimum_span": (alpha - tau) * (capacity - 1),
    }


def phase_character_sup_error(source: float, target: float, horizon: float) -> float:
    phase = horizon * abs(source - target)
    return 2.0 if phase >= math.pi else 2.0 * math.sin(phase / 2.0)


def operator_error_to_log_tau(
    epsilon: float, horizon: float, scale: float, omega_min: float
) -> dict[str, float]:
    if not 0 <= epsilon < 2 or horizon <= 0 or scale <= 0 or omega_min <= 0:
        raise ValueError("epsilon in [0,2), positive horizon/scale/omega_min required")
    delta = 2.0 / horizon * math.asin(epsilon / 2.0)
    if delta >= scale * omega_min:
        raise ValueError("frequency error does not preserve a positive log-frequency anchor")
    return {
        "frequency_error_bound": delta,
        "log_mismatch_bound": -math.log1p(-delta / (scale * omega_min)),
    }


def _matrix_exponential_nilpotent(generator: np.ndarray, time: float) -> np.ndarray:
    result = np.eye(generator.shape[0])
    power = np.eye(generator.shape[0])
    for order in range(1, generator.shape[0]):
        power = power @ generator
        result = result + time**order / math.factorial(order) * power
    return result


def continuous_scaling_trilemma_report(dimension: int, scale: float) -> dict[str, Any]:
    """Exact nilpotent witness, bounded-generator obstruction, and s=-1 exception."""

    if dimension < 2 or scale <= 1:
        raise ValueError("dimension >= 2 and scale > 1 required")
    nilpotent = np.zeros((dimension, dimension), dtype=np.float64)
    nilpotent[np.arange(dimension - 1), np.arange(1, dimension)] = 1.0
    dilation = np.diag(scale ** (-np.arange(dimension, dtype=np.float64)))
    nilpotent_residual = float(
        np.linalg.norm(dilation @ nilpotent @ np.linalg.inv(dilation) - scale * nilpotent)
    )
    nilpotent_index = next(
        power
        for power in range(1, dimension + 1)
        if np.linalg.norm(np.linalg.matrix_power(nilpotent, power)) <= 1e-14
    )
    growth = {
        str(time): float(np.linalg.norm(_matrix_exponential_nilpotent(nilpotent, time), 2))
        for time in (1.0, 4.0, 16.0)
    }
    skew = np.array(((0.0, -1.0), (1.0, 0.0)))
    reflection = np.diag((1.0, -1.0))
    negative_scale_residual = float(
        np.linalg.norm(reflection @ skew @ reflection - (-1.0) * skew)
    )
    return {
        "dimension": dimension,
        "positive_scale": scale,
        "nilpotent_similarity_residual": nilpotent_residual,
        "nilpotent_index": nilpotent_index,
        "nilpotent_exponential_operator_norm": growth,
        "bounded_nonzero_skew_spectrum": [
            [float(value.real), float(value.imag)] for value in np.linalg.eigvals(skew)
        ],
        "positive_scaling_changes_nonzero_spectral_radius": True,
        "negative_one_nontrivial_residual": negative_scale_residual,
        "claim": (
            "Positive exact scaling admits nilpotent generators, but their exponentials "
            "are unbounded unless the generator is zero; s=-1 has a bounded nontrivial exception."
        ),
    }


def _unique(values: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        return ordered
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.diff(ordered) > tolerance
    return ordered[keep]


def scaled_fourier_orbit(
    frequencies: Sequence[float], scale: float, levels: int, *, real: bool
) -> np.ndarray:
    base = np.asarray(frequencies, dtype=np.float64)
    if base.ndim != 1 or not len(base) or not np.all(base > 0):
        raise ValueError("positive one-dimensional frequencies required")
    if scale <= 1 or levels < 0:
        raise ValueError("scale > 1 and levels >= 0 required")
    signs = (-1.0, 1.0) if real else (1.0,)
    return _unique(np.concatenate([
        sign * base * scale**level
        for level in range(levels + 1)
        for sign in signs
    ]))


def _continuous_cross(left: np.ndarray, right: np.ndarray, horizon: int) -> np.ndarray:
    return np.sinc(float(horizon) * (left[:, None] - right[None, :]) / np.pi)


def _discrete_cross(left: np.ndarray, right: np.ndarray, horizon: int) -> np.ndarray:
    delta = (left[:, None] - right[None, :] + np.pi) % (2.0 * np.pi) - np.pi
    denominator = np.sin(delta / 2.0)
    numerator = np.sin((2 * horizon + 1) * delta / 2.0)
    return np.divide(
        numerator,
        (2 * horizon + 1) * denominator,
        out=np.ones_like(delta),
        where=np.abs(denominator) > 1e-15,
    )


def _gram(values: np.ndarray, horizon: int, discrete: bool) -> np.ndarray:
    return discrete_gram(values, horizon) if discrete else continuous_gram(values, horizon)


def _ky_fan(gram: np.ndarray, dimension: int) -> dict[str, float]:
    result = ky_fan_lower_bound(gram, dimension)
    return {
        "captured_energy": result["ky_fan_captured_energy"],
        "epsilon_squared_lower_bound": result["epsilon_squared_lower_bound"],
        "epsilon_lower_bound": result["epsilon_lower_bound"],
        "coherence": result["coherence"],
    }


def montgomery_vaughan_separation_bound(
    modes: Sequence[float], horizon: float, dimension: int
) -> dict[str, float]:
    """Dimension-free Gram bound from the weighted Hilbert inequality.

    The safe Montgomery--Vaughan constant 3*pi/2 is used.  The inverse
    ``horizon * separation`` order is sharp, but this constant is not claimed
    optimal.
    """

    values = _unique(np.asarray(modes, dtype=np.float64))
    if len(values) < 2 or horizon <= 0 or not 0 < dimension <= len(values):
        raise ValueError("at least two modes, positive horizon, and 0 < dimension <= M required")
    separation = float(np.min(np.diff(values)))
    slack = 3.0 * math.pi / (2.0 * horizon * separation)
    lambda_max_upper = 1.0 + slack
    epsilon_squared = max(0.0, 1.0 - dimension * lambda_max_upper / len(values))
    return {
        "minimum_separation": separation,
        "lambda_max_upper_bound": lambda_max_upper,
        "epsilon_squared_lower_bound": epsilon_squared,
        "epsilon_lower_bound": math.sqrt(epsilon_squared),
        "inverse_horizon_separation_slack": slack,
    }


def fourier_orbit_rank_report(
    frequencies: Sequence[float],
    scale: float,
    levels: int,
    horizon: int,
    *,
    real: bool,
    discrete: bool,
) -> dict[str, Any]:
    base_positive = np.asarray(frequencies, dtype=np.float64)
    base = _unique(np.concatenate((-base_positive, base_positive))) if real else base_positive
    orbit = scaled_fourier_orbit(base_positive, scale, levels, real=real)
    dimension = len(base)
    gram = _gram(orbit, horizon, discrete)
    bound = _ky_fan(gram, dimension)
    mv_bound = (
        None
        if discrete or len(orbit) < 2
        else montgomery_vaughan_separation_bound(orbit, horizon, dimension)
    )
    if mv_bound is not None:
        actual_lambda_max = float(np.linalg.eigvalsh(gram)[-1])
        if actual_lambda_max > mv_bound["lambda_max_upper_bound"] + 1e-9:
            raise AssertionError("Montgomery--Vaughan Gram bound violated")
    cross = (
        _discrete_cross(orbit, base, horizon)
        if discrete
        else _continuous_cross(orbit, base, horizon)
    )
    base_gram = _gram(base, horizon, discrete)
    projection_energy = np.einsum(
        "ij,jk,ik->i", cross, np.linalg.pinv(base_gram, rcond=1e-12), cross
    )
    projection_residual = np.clip(1.0 - projection_energy, 0.0, 1.0)
    if len(orbit) > 1:
        separation = float(np.min(np.diff(orbit)))
        analytic_coherence = (
            None
            if discrete
            else min(1.0, 1.0 / (horizon * separation))
        )
    else:
        separation, analytic_coherence = math.inf, 0.0
    coherence_squared_bound = max(
        0.0,
        1.0 - dimension * (1.0 + (len(orbit) - 1) * bound["coherence"]) / len(orbit),
    )
    separation_squared_bound = (
        None
        if analytic_coherence is None
        else max(
            0.0,
            1.0
            - dimension * (1.0 + (len(orbit) - 1) * analytic_coherence) / len(orbit),
        )
    )
    return {
        "representation": "real_orthogonal_complexification" if real else "complex_diagonal",
        "position_domain": "discrete" if discrete else "continuous",
        "base_dimension": dimension,
        "positive_frequency_pairs": len(base_positive),
        "orbit_modes": len(orbit),
        "scale": scale,
        "levels": levels,
        "horizon": horizon,
        "ky_fan": bound,
        "coherence_epsilon_squared_lower_bound": coherence_squared_bound,
        "separation_epsilon_squared_lower_bound": separation_squared_bound,
        "montgomery_vaughan_separation_bound": mv_bound,
        "minimum_orbit_separation": separation,
        "separation_only_coherence_upper_bound": analytic_coherence,
        "fixed_native_subspace_mean_squared_residual": float(projection_residual.mean()),
        "fixed_native_subspace_max_squared_residual": float(projection_residual.max()),
    }


def orbit_growth_coherence_lower_bound(
    dimension: int, residue_classes: int, levels: int, rho: float
) -> dict[str, Any]:
    """Combine Theorem 3's orbit count with Theorem 5's Gram bound."""

    if (
        dimension <= 0
        or not 1 <= residue_classes <= dimension
        or levels < 0
        or rho < 0
    ):
        raise ValueError("positive dimension, 1 <= q <= K, levels >= 0, and rho >= 0 required")
    orbit_lower_bound = dimension + levels * residue_classes
    epsilon_squared = max(
        0.0,
        1.0 - dimension * (1.0 + rho) / orbit_lower_bound,
    )
    return {
        "dimension": dimension,
        "residue_classes": residue_classes,
        "levels": levels,
        "lambda_max_upper_slack": rho,
        "orbit_modes_lower_bound": orbit_lower_bound,
        "epsilon_squared_lower_bound": epsilon_squared,
        "epsilon_lower_bound": math.sqrt(epsilon_squared),
    }


def arbitrary_mixing_l2_report(
    frequencies: Sequence[float],
    scale: float,
    levels: int,
    horizon: int,
    *,
    discrete: bool,
    seed: int,
) -> dict[str, Any]:
    """Exact L2 diagnostic for non-permutation invertible similarity maps."""

    base = np.asarray(frequencies, dtype=np.float64)
    if base.ndim != 1 or not len(base) or not np.all(base > 0):
        raise ValueError("positive one-dimensional frequencies required")
    if scale <= 1 or levels < 0 or horizon <= 0:
        raise ValueError("scale > 1, levels >= 0, and positive horizon required")
    targets = np.concatenate([base * scale**level for level in range(levels + 1)])
    orbit = _unique(targets)
    if len(orbit) != len(targets):
        raise ValueError("diagnostic requires a collision-free scaled orbit")
    base_gram = _gram(base, horizon, discrete)
    cross = (
        _discrete_cross(targets, base, horizon)
        if discrete
        else _continuous_cross(targets, base, horizon)
    )
    inverse_gram = np.linalg.pinv(base_gram, rcond=1e-12)
    optimal_residuals = np.clip(
        1.0 - np.einsum("ij,jk,ik->i", cross, inverse_gram, cross),
        0.0,
        None,
    )
    rng = np.random.default_rng(seed)
    actual_residuals: list[float] = []
    condition_numbers = []
    for level in range(levels + 1):
        if level == 0:
            dilation = np.eye(len(base), dtype=np.complex128)
        else:
            dilation = np.eye(len(base), dtype=np.complex128) + 0.2 * (
                rng.normal(size=(len(base), len(base)))
                + 1j * rng.normal(size=(len(base), len(base)))
            ) / math.sqrt(len(base))
        condition_numbers.append(float(np.linalg.cond(dilation)))
        coefficients = dilation * np.linalg.inv(dilation).T
        for index, coefficients_for_coordinate in enumerate(coefficients):
            target_index = level * len(base) + index
            residual = (
                1.0
                + np.vdot(
                    coefficients_for_coordinate,
                    base_gram @ coefficients_for_coordinate,
                ).real
                - 2.0
                * np.vdot(
                    coefficients_for_coordinate,
                    cross[target_index],
                ).real
            )
            actual_residuals.append(max(0.0, float(residual)))
    ky_fan = _ky_fan(_gram(orbit, horizon, discrete), len(base))
    actual = np.asarray(actual_residuals)
    if np.min(actual - optimal_residuals) < -1e-9:
        raise AssertionError("an invertible-mixing diagonal beat its fixed-subspace projection")
    if float(optimal_residuals.mean()) + 1e-9 < ky_fan["epsilon_squared_lower_bound"]:
        raise AssertionError("fixed native subspace beat the Ky-Fan optimum")
    return {
        "position_domain": "discrete" if discrete else "continuous",
        "dimension": len(base),
        "orbit_modes": len(orbit),
        "condition_numbers": condition_numbers,
        "selected_mixing_mean_squared_diagonal_error": float(actual.mean()),
        "selected_mixing_operator_epsilon_squared_lower_bound": float(actual.max()),
        "fixed_native_optimal_mean_squared_residual": float(optimal_residuals.mean()),
        "ky_fan_epsilon_squared_lower_bound": ky_fan["epsilon_squared_lower_bound"],
        "minimum_selected_vs_projection_slack": float(np.min(actual - optimal_residuals)),
        "claim_limit": (
            "This checks the exact L2 implication for sampled invertible D_j; it does not "
            "optimize the continuous operator supremum or prove the universal theorem."
        ),
    }


def _chord(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    delta = (np.asarray(left) - np.asarray(right) + np.pi) % (2.0 * np.pi) - np.pi
    return 2.0 * np.abs(np.sin(delta / 2.0))


def _matching_for_threshold(cost: np.ndarray, threshold: float) -> list[int] | None:
    owner = [-1] * cost.shape[0]

    def augment(source: int, seen: list[bool]) -> bool:
        for target in np.flatnonzero(cost[:, source] <= threshold + 1e-15):
            target = int(target)
            if seen[target]:
                continue
            seen[target] = True
            if owner[target] < 0 or augment(owner[target], seen):
                owner[target] = source
                return True
        return False

    for source in range(cost.shape[1]):
        if not augment(source, [False] * cost.shape[0]):
            return None
    permutation = [-1] * cost.shape[1]
    for target, source in enumerate(owner):
        permutation[source] = target
    return permutation


def bottleneck_scaling_match(phases: Sequence[float], scale: int) -> dict[str, Any]:
    angles = np.asarray(phases, dtype=np.float64) % (2.0 * np.pi)
    if angles.ndim != 1 or not len(angles) or scale < 2:
        raise ValueError("nonempty phases and integer scale >= 2 required")
    cost = _chord(angles[:, None], scale * angles[None, :])
    candidates = np.unique(cost)
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        middle = (lo + hi) // 2
        if _matching_for_threshold(cost, float(candidates[middle])) is None:
            lo = middle + 1
        else:
            hi = middle
    defect = float(candidates[lo])
    permutation = _matching_for_threshold(cost, defect)
    if permutation is None:
        raise AssertionError("bottleneck matching reconstruction failed")
    return {"defect": defect, "permutation": permutation}


def _cycle_lengths(permutation: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    lengths = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return lengths


def anti_alias_margin(phases: Sequence[float], horizon: int) -> float:
    angles = np.asarray(phases, dtype=np.float64)
    if horizon <= 0:
        raise ValueError("positive discrete horizon required")
    return min(float(np.max(_chord(multiple * angles, 0.0))) for multiple in range(1, horizon + 1))


def approximate_alias_report(phases: Sequence[float], scale: int) -> dict[str, Any]:
    """Theorem 6, including the realized common approximate period."""

    angles = np.asarray(phases, dtype=np.float64) % (2.0 * np.pi)
    match = bottleneck_scaling_match(angles, scale)
    lengths = _cycle_lengths(match["permutation"])
    period = math.lcm(*(scale**length - 1 for length in lengths))
    dimension_bound = scale ** len(angles) - 1
    if period > dimension_bound or math.gcd(period, scale) != 1:
        raise AssertionError("discrete-period arithmetic bound violated")
    actual_alias = float(np.max(_chord(period * angles, 0.0)))
    upper = 2.0 * math.sin(min(
        math.pi / 2.0,
        period / (scale - 1.0) * math.asin(min(1.0, match["defect"] / 2.0)),
    ))
    if actual_alias > upper + 5e-10:
        raise AssertionError((actual_alias, upper))
    return {
        "dimension": len(angles),
        "scale": scale,
        "scaling_defect": match["defect"],
        "matching_permutation": match["permutation"],
        "cycle_lengths": lengths,
        "common_period": period,
        "dimension_period_bound": dimension_bound,
        "coprime_to_scale": math.gcd(period, scale) == 1,
        "actual_alias_error": actual_alias,
        "alias_error_upper_bound": upper,
    }


def sharp_periodic_phases(dimension: int, scale: int) -> np.ndarray:
    if dimension <= 0 or scale < 2:
        raise ValueError("positive dimension and scale >= 2 required")
    period = scale**dimension - 1
    residues = np.asarray([pow(scale, index, period) for index in range(dimension)])
    return 2.0 * np.pi * residues / period


def anti_alias_corollary_lower_bound(
    gamma: float, dimension: int, scale: int
) -> float:
    if not 0 <= gamma <= 2 or dimension <= 0 or scale < 2:
        raise ValueError("gamma in [0,2], positive dimension, scale >= 2 required")
    return 2.0 * math.sin(
        (scale - 1.0) / (scale**dimension - 1.0) * math.asin(gamma / 2.0)
    )


def uniform_log_lattice_report(
    pairs: int, spacing: float, alpha: float, levels: int
) -> dict[str, Any]:
    if pairs < 2 or spacing <= 0 or alpha <= 0 or levels < 0:
        raise ValueError("pairs >= 2, positive spacing/alpha, and levels >= 0 required")
    shift = int(round(alpha / spacing))
    mismatch = alpha - shift * spacing
    commensurate = abs(mismatch) <= 1e-12 * max(1.0, alpha)
    return {
        "pairs": pairs,
        "spacing": spacing,
        "alpha": alpha,
        "nearest_index_shift": shift,
        "single_step_log_mismatch": abs(mismatch),
        "level_log_mismatch": [level * abs(mismatch) for level in range(levels + 1)],
        "commensurate": commensurate,
        "exact_survivors": max(0, pairs - levels * shift) if commensurate else None,
        "exact_orbit_modes": (
            pairs + levels * min(shift, pairs) if commensurate else None
        ),
        "overlap_regime": (
            "overlapping_or_touching_shifted_intervals"
            if commensurate and shift <= pairs
            else "disjoint_shifted_intervals"
            if commensurate
            else None
        ),
    }


def uniform_log_multi_generator_report(
    pairs: int, spacing: float, alphas: Sequence[float], levels: int
) -> dict[str, Any]:
    """Finite multi-generator lattice diagnostic, not a universal optimizer."""

    if not alphas:
        raise ValueError("at least one log-scale generator is required")
    generators = [uniform_log_lattice_report(pairs, spacing, alpha, levels) for alpha in alphas]
    all_exact = all(row["commensurate"] for row in generators)
    shifts = [row["nearest_index_shift"] for row in generators]
    reachable_shifts: list[int] | None = None
    orbit_modes: int | None = None
    if all_exact:
        reachable = {
            sum(count * shift for count, shift in zip(counts, shifts))
            for counts in itertools.product(range(levels + 1), repeat=len(shifts))
            if sum(counts) <= levels
        }
        reachable_shifts = sorted(reachable)
        orbit_modes = len({
            pair - shift
            for shift in reachable
            for pair in range(pairs)
        })
    return {
        "pairs": pairs,
        "spacing": spacing,
        "levels": levels,
        "generators": generators,
        "all_generators_are_exact_lattice_shifts": all_exact,
        "reachable_index_shifts": reachable_shifts,
        "joint_orbit_modes": orbit_modes,
        "claim_limit": (
            "Exact joint shifts require every alpha/spacing ratio to be integral; "
            "nonuniform multi-generator optimization remains unresolved."
        ),
    }


def _load_manifest_tables(path: Path) -> dict[str, np.ndarray]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("tables")
    if not isinstance(records, dict):
        raise ValueError("scale-orbit manifest requires a tables object")
    tables: dict[str, np.ndarray] = {}
    for name, record in records.items():
        table_path = path.parent / str(record["path"])
        values = np.load(table_path, allow_pickle=False)
        digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
        if digest != record["float32_sha256"]:
            raise RuntimeError(f"table hash drift: {name}")
        tables[str(name)] = np.asarray(values, dtype=np.float64)
    return tables


def _theorem3_exhaustive() -> dict[str, Any]:
    cases = equality = 0
    for denominator in (1, 2, 3):
        grid = [Fraction(value, denominator) for value in range(10)]
        for size in range(2, 6):
            for points in itertools.combinations(grid, size):
                for alpha in (Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)):
                    for levels in (0, 1, 2, 3):
                        report = exact_orbit_report(points, alpha, levels)
                        if not all(report[key] for key in (
                            "first_bound_holds", "global_bound_holds",
                            "boundary_bound_holds", "two_sided_bound_holds",
                        )):
                            raise AssertionError(report)
                        if levels > 0 and report["first_bound_equality"] != report[
                            "all_residue_classes_are_consecutive"
                        ]:
                            raise AssertionError("Theorem 3 equality characterization drift")
                        cases += 1
                        equality += int(report["first_bound_equality"])
    return {"status": "PASS", "cases": cases, "equality_cases": equality}


def _theorem4_random(seed: int, cases: int, *, two_sided: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tight = 0
    reports = []
    for index in range(cases):
        size = int(rng.integers(3, 9))
        points = np.sort(rng.choice(np.arange(0, 25), size=size, replace=False) / 4.0)
        alpha = float(rng.choice((0.5, 0.75, 1.0, 1.25)))
        tau = float(rng.choice((0.0, 0.1, 0.2)))
        if tau >= alpha:
            continue
        levels = int(rng.integers(1, 4))
        report = best_coherent_matching(
            points, alpha, tau, levels, two_sided=two_sided
        )
        tight += int(report["best_survivors"] == report["theorem_survivor_bound"])
        if index < 8:
            reports.append(report)
    return {
        "status": "PASS",
        "cases": cases,
        "directions": 2 if two_sided else 1,
        "tight_cases": tight,
        "examples": reports,
    }


def _operator_tau_random(seed: int, cases: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    checked = 0
    worst_slack = math.inf
    for _ in range(cases):
        scale = float(rng.choice((2.0, 4.0)))
        omega = float(np.exp(rng.uniform(-4.0, 0.0)))
        horizon = float(rng.choice((16, 64, 256, 4096)))
        relative = float(rng.uniform(-0.2, 0.2))
        target = scale * omega * math.exp(relative)
        epsilon = phase_character_sup_error(target, scale * omega, horizon)
        if epsilon >= 2.0 - 1e-12:
            continue
        bound = operator_error_to_log_tau(epsilon, horizon, scale, omega)
        slack = bound["log_mismatch_bound"] - abs(relative)
        if slack < -1e-10:
            raise AssertionError((relative, epsilon, bound))
        worst_slack = min(worst_slack, slack)
        checked += 1
    return {"status": "PASS", "attempted": cases, "checked": checked, "minimum_slack": worst_slack}


def _theorem5_panels(
    factors: Sequence[float], levels: Sequence[int], horizons: Sequence[int]
) -> list[dict[str, Any]]:
    frequencies = np.exp(-np.linspace(0.0, 6.0, 12))
    return [
        fourier_orbit_rank_report(
            frequencies, factor, level, horizon, real=real, discrete=discrete
        )
        for factor in factors
        for level in levels
        for horizon in horizons
        for real in (False, True)
        for discrete in (False, True)
    ]


def _theorem5_orthogonal_sharpness() -> dict[str, Any]:
    pairs = 4
    scale = float(pairs + 1)
    report = fourier_orbit_rank_report(
        np.arange(1, pairs + 1, dtype=np.float64),
        scale,
        1,
        math.pi,
        real=False,
        discrete=False,
    )
    expected = 1.0 - pairs / (2 * pairs)
    if abs(report["ky_fan"]["epsilon_squared_lower_bound"] - expected) > 1e-12:
        raise AssertionError(report)
    return {
        "status": "PASS",
        "construction": "integer Fourier characters on [-pi,pi]",
        "expected_epsilon_squared": expected,
        "report": report,
    }


def _theorem6_panels(seed: int, max_dimension: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    exact, approximate = [], []
    for scale in (2, 3):
        for dimension in range(1, max_dimension + 1):
            phases = sharp_periodic_phases(dimension, scale)
            row = approximate_alias_report(phases, scale)
            if row["scaling_defect"] > 1e-10 or row["common_period"] > scale**dimension - 1:
                raise AssertionError(row)
            exact.append(row)
            for noise in (1e-8, 1e-5, 1e-3):
                perturbed = phases + rng.normal(0.0, noise, size=dimension)
                alias = approximate_alias_report(perturbed, scale)
                horizon = scale**dimension - 1
                gamma = anti_alias_margin(perturbed, horizon)
                lower = anti_alias_corollary_lower_bound(gamma, dimension, scale)
                if alias["scaling_defect"] + 1e-10 < lower:
                    raise AssertionError((alias, gamma, lower))
                approximate.append({
                    "noise": noise,
                    "anti_alias_horizon": horizon,
                    "anti_alias_margin": gamma,
                    "anti_alias_defect_lower_bound": lower,
                    **alias,
                })
    return {"status": "PASS", "exact": exact, "approximate": approximate}


def _dimension_span_grid(
    factors: Sequence[float], levels: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {
            "factor": factor,
            "levels": level,
            "directions": directions,
            "leakage_target": leakage,
            "tau_fraction_of_log_scale": tau_fraction,
            **dimension_span_requirements(
                level,
                leakage,
                math.log(factor),
                tau_fraction * math.log(factor),
                directions=directions,
            ),
        }
        for factor in factors
        for level in levels
        if level > 0
        for directions in (1, 2)
        for leakage in (0.05, 0.1, 0.25, 0.5)
        for tau_fraction in (0.0, 0.01, 0.1)
    ]


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    runtime = require_cpu_only()
    payload: dict[str, Any] = {
        "status": STATUS,
        "method_id": METHOD_ID,
        "protocol": {
            "seed": args.seed,
            "random_cases": args.random_cases,
            "factors": args.factors,
            "levels": args.levels,
            "horizons": args.horizons,
            "max_discrete_dimension": args.max_discrete_dimension,
            "executable_coverage": list(EXECUTABLE_COVERAGE),
            "non_executable_or_unresolved_gates": NON_EXECUTABLE_GATES,
            "content_boundary": (
                "Theorems are checked under their declared representation, matching, "
                "window, and alias assumptions. No quantity is an LM selector."
            ),
        },
        "exact_scaling_trilemma": continuous_scaling_trilemma_report(6, 2.0),
        "theorem3_exact_orbit": _theorem3_exhaustive(),
        "theorem4_abstract_packing": verify_packing_formula(),
        "theorem4_coherent_matching": _theorem4_random(args.seed, args.random_cases),
        "theorem4_two_sided_coherent_matching": _theorem4_random(
            args.seed + 1, max(8, args.random_cases // 2), two_sided=True
        ),
        "theorem4_dimension_span": _dimension_span_grid(args.factors, args.levels),
        "continuous_operator_error_to_tau": _operator_tau_random(
            args.seed + 2, args.random_cases * 4
        ),
        "theorem5_fourier_orbit_rank": _theorem5_panels(
            args.factors, args.levels, args.horizons
        ),
        "theorem5_orthogonal_character_sharpness": _theorem5_orthogonal_sharpness(),
        "theorem5_orbit_growth_coherence_combination": [
            orbit_growth_coherence_lower_bound(12, residue_classes, level, rho)
            for residue_classes in (1, 3, 12)
            for level in args.levels
            for rho in (0.0, 0.05, 0.25)
        ],
        "theorem5_arbitrary_invertible_mixing_l2": [
            arbitrary_mixing_l2_report(
                np.exp(-np.linspace(0.1, 2.7, 6)),
                factor,
                level,
                horizon,
                discrete=discrete,
                seed=args.seed + 3,
            )
            for factor in args.factors
            for level in args.levels
            for horizon in args.horizons
            for discrete in (False, True)
        ],
        "theorem6_discrete_alias": _theorem6_panels(
            args.seed + 4, args.max_discrete_dimension
        ),
        "allocation_panels": {
            "commensurate": uniform_log_lattice_report(64, 0.25, 1.0, 4),
            "incommensurate": uniform_log_lattice_report(64, 0.25, 1.1, 4),
            "multi_generator_commensurate": uniform_log_multi_generator_report(
                64, 0.25, (0.5, 0.75), 4
            ),
            "multi_generator_incommensurate": uniform_log_multi_generator_report(
                64, 0.25, (0.5, math.sqrt(2.0)), 4
            ),
        },
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "claim_limit": (
            "Numerical verification and counterexample search do not replace a proof, "
            "a literature novelty audit, or a behavioural experiment."
        ),
    }
    if args.table_manifest:
        tables = _load_manifest_tables(args.table_manifest.resolve())
        payload["frozen_table_manifest_sha256"] = sha256_file(args.table_manifest.resolve())
        payload["frozen_table_panels"] = {
            name: [
                fourier_orbit_rank_report(
                    values, factor, level, horizon, real=real, discrete=discrete
                )
                for factor in args.factors
                for level in args.levels
                for horizon in args.horizons
                for real in (False, True)
                for discrete in (False, True)
            ]
            for name, values in tables.items()
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--factors", type=float, nargs="+", default=(2.0, 4.0))
    parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--horizons", type=int, nargs="+", default=(32, 256, 4096))
    parser.add_argument("--max-discrete-dimension", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.random_cases <= 0 or args.max_discrete_dimension <= 0:
        parser.error("random-cases and max-discrete-dimension must be positive")
    if any(value <= 1.0 for value in args.factors):
        parser.error("all factors must exceed one")
    if any(value < 0 for value in args.levels) or any(value <= 0 for value in args.horizons):
        parser.error("levels must be nonnegative and horizons positive")
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    return args


def main() -> int:
    args = parse_args()
    runtime = require_cpu_only()
    if args.table_manifest and not args.table_manifest.is_file():
        raise FileNotFoundError(args.table_manifest)
    if args.preflight_only:
        frozen_tables = (
            sorted(_load_manifest_tables(args.table_manifest.resolve()))
            if args.table_manifest
            else []
        )
        print(json.dumps({
            "status": "FINITE_SCALE_COVARIANCE_PREFLIGHT_COMPLETE",
            "method_id": METHOD_ID,
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "cuda_initialized": False,
            "torch_imported": runtime["torch_imported"],
            "planned": {
                "components": list(EXECUTABLE_COVERAGE),
                "non_executable_or_unresolved_gates": NON_EXECUTABLE_GATES,
                "frozen_tables": frozen_tables,
            },
            "frozen_table_manifest_sha256": (
                sha256_file(args.table_manifest.resolve()) if args.table_manifest else None
            ),
        }, indent=2, sort_keys=True))
        return 0
    assert args.output is not None
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = run_audit(args)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
