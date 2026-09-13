"""CPU-only mechanism probes.

These quantities describe a frozen RoPE operator.  They are deliberately kept
separate from language-model or generation results: a better value here is a
candidate-construction signal, not evidence of task improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable

import numpy as np


DELTA_GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


class MechanismNotLocallySeparable(RuntimeError):
    """The declared local projections are numerically undefined or degenerate."""


def _float64_vector(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def phase_unseen_fraction(
    omega: np.ndarray | Iterable[float],
    train_length: int,
    target_length: int,
) -> np.ndarray:
    """Continuous unseen phase-arc fraction for each positive frequency.

    ``train_length`` and ``target_length`` are token-window sizes, so their
    largest causal relative distances are length minus one.  This corrects the
    Web proposal's inconsistent use of ``[0, L]`` beside a discrete
    ``0,...,L-1`` collision domain.  The quantity still measures phase-set
    coverage only; it does not weight channels by a checkpoint's learned use.
    """
    values = _float64_vector(omega, name="omega")
    if np.any(values <= 0.0):
        raise ValueError("omega must be positive")
    if train_length < 2 or target_length < train_length:
        raise ValueError("target_length must be at least train_length >= 2")
    tau = 2.0 * math.pi
    train_extent = float(train_length - 1)
    target_extent = float(target_length - 1)
    return (
        np.minimum(tau, values * target_extent)
        - np.minimum(tau, values * train_extent)
    ) / tau


def table_phase_ood(
    omega: np.ndarray | Iterable[float],
    train_length: int,
    target_factors: tuple[int, ...] = (2, 4, 8),
    factor_weights: tuple[float, ...] | None = None,
) -> tuple[float, dict[int, float]]:
    """Return weighted table mean and per-factor phase-arc burdens."""
    values = _float64_vector(omega, name="omega")
    if not target_factors or any(factor < 1 for factor in target_factors):
        raise ValueError("target_factors must contain positive integers")
    if len(set(target_factors)) != len(target_factors):
        raise ValueError("target_factors must be unique")
    weights = (
        np.full(len(target_factors), 1.0 / len(target_factors), dtype=np.float64)
        if factor_weights is None
        else _float64_vector(factor_weights, name="factor_weights")
    )
    if weights.size != len(target_factors) or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("factor_weights must be nonnegative and sum to one")
    per_factor = {
        int(factor): float(phase_unseen_fraction(values, train_length, factor * train_length).mean())
        for factor in target_factors
    }
    aggregate = float(sum(weight * per_factor[int(factor)] for weight, factor in zip(weights, target_factors)))
    return aggregate, per_factor


def causal_separation_weights(length: int) -> np.ndarray:
    """Normalized causal-pair distance weights for d=0,...,length-1."""
    if length <= 0:
        raise ValueError("length must be positive")
    distances = np.arange(length, dtype=np.float64)
    weights = 2.0 * (float(length) - distances) / (float(length) * float(length + 1))
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2e-15):
        raise RuntimeError("causal weights lost normalization")
    return weights


def _inverse_sqrt(matrix: np.ndarray, eig_floor: float) -> tuple[np.ndarray, bool, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues[-1] <= 0.0 or not np.isfinite(eigenvalues).all():
        raise FloatingPointError("self Gram is not positive semidefinite")
    floor = max(float(eig_floor), float(eigenvalues[-1]) * float(eig_floor))
    clipped = np.maximum(eigenvalues, floor)
    return (
        (eigenvectors * np.power(clipped, -0.5)) @ eigenvectors.T,
        bool(np.any(eigenvalues < floor)),
        float(eigenvalues[0]),
    )


def canonical_pair_overlap(
    omega_i: float,
    omega_j: float,
    distances: np.ndarray | Iterable[float],
    weights: np.ndarray | Iterable[float],
    *,
    eig_floor: float = 1e-12,
) -> float:
    """Full sine-cosine canonical overlap of two pair subspaces."""
    d = _float64_vector(distances, name="distances")
    w = _float64_vector(weights, name="weights")
    if d.size != w.size or np.any(w < 0.0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("distances and normalized nonnegative weights must align")
    if omega_i <= 0.0 or omega_j <= 0.0 or eig_floor <= 0.0:
        raise ValueError("frequencies and eig_floor must be positive")
    first = np.stack((np.cos(float(omega_i) * d), np.sin(float(omega_i) * d)), axis=1)
    second = np.stack((np.cos(float(omega_j) * d), np.sin(float(omega_j) * d)), axis=1)
    first *= np.sqrt(w)[:, None]
    second *= np.sqrt(w)[:, None]
    first_whitener, _, _ = _inverse_sqrt(first.T @ first, eig_floor)
    second_whitener, _, _ = _inverse_sqrt(second.T @ second, eig_floor)
    cross = np.einsum("li,lj->ij", first, second, optimize=False)
    canonical = np.einsum(
        "ab,bc,cd->ad", first_whitener, cross, second_whitener, optimize=False
    )
    raw = 0.5 * float(np.square(canonical).sum())
    if raw < -1e-9 or raw > 1.0 + 1e-7:
        raise FloatingPointError(f"canonical overlap outside [0,1]: {raw}")
    return float(np.clip(raw, 0.0, 1.0))


def table_collision_details(
    omega: np.ndarray | Iterable[float],
    train_length: int,
    *,
    eig_floor: float = 1e-12,
) -> dict:
    """Vectorized full-pair collision plus numerical diagnostics."""
    values = _float64_vector(omega, name="omega")
    if values.size < 2 or np.any(values <= 0.0) or eig_floor <= 0.0:
        raise ValueError("need at least two positive frequencies and a positive eig_floor")
    distances = np.arange(train_length, dtype=np.float64)
    weights = causal_separation_weights(train_length)
    phases = distances[:, None] * values[None, :]
    features = np.stack((np.cos(phases), np.sin(phases)), axis=2)
    weighted = features * np.sqrt(weights)[:, None, None]
    blocks = weighted.reshape(train_length, 2 * values.size)
    # macOS Accelerate emits spurious overflow/underflow warnings for this
    # bounded 2048x64 GEMM.  The explicit contraction is deterministic and
    # returns the same block Gram without hiding numerical warnings globally.
    gram = np.einsum("li,lj->ij", blocks, blocks, optimize=False)
    whiteners: list[np.ndarray] = []
    floor_count = 0
    minimum_eigenvalue = math.inf
    for index in range(values.size):
        block = gram[2 * index : 2 * index + 2, 2 * index : 2 * index + 2]
        whitener, used_floor, smallest = _inverse_sqrt(block, eig_floor)
        whiteners.append(whitener)
        floor_count += int(used_floor)
        minimum_eigenvalue = min(minimum_eigenvalue, smallest)
    overlaps = np.eye(values.size, dtype=np.float64)
    raw_min, raw_max = 1.0, 0.0
    for left in range(values.size):
        for right in range(left + 1, values.size):
            cross = gram[2 * left : 2 * left + 2, 2 * right : 2 * right + 2]
            canonical = whiteners[left] @ cross @ whiteners[right]
            raw = 0.5 * float(np.square(canonical).sum())
            raw_min, raw_max = min(raw_min, raw), max(raw_max, raw)
            if raw < -1e-9 or raw > 1.0 + 1e-7:
                raise FloatingPointError(f"canonical overlap outside [0,1]: {raw}")
            overlaps[left, right] = overlaps[right, left] = np.clip(raw, 0.0, 1.0)
    upper = overlaps[np.triu_indices(values.size, 1)]
    collision = float(upper.mean())
    effective_rank = float(2.0 * values.size / (1.0 + (values.size - 1.0) * collision))
    return {
        "collision": collision,
        "effective_rank": effective_rank,
        "overlap_matrix": overlaps,
        "eig_floor": float(eig_floor),
        "self_gram_floor_count": floor_count,
        "minimum_self_gram_eigenvalue": minimum_eigenvalue,
        "raw_overlap_min": raw_min,
        "raw_overlap_max": raw_max,
    }


def table_collision(
    omega: np.ndarray | Iterable[float],
    train_length: int,
    *,
    eig_floor: float = 1e-12,
) -> tuple[float, float]:
    details = table_collision_details(omega, train_length, eig_floor=eig_floor)
    return details["collision"], details["effective_rank"]


def _central_gradient(function: Callable[[np.ndarray], float], point: np.ndarray, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    gradient = np.empty_like(point, dtype=np.float64)
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        gradient[index] = (function(point + offset) - function(point - offset)) / (2.0 * step)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("mechanism gradient is non-finite")
    return gradient


@dataclass(frozen=True)
class MechanismDirections:
    d_ood: np.ndarray
    d_collision: np.ndarray
    grad_log_ood: np.ndarray
    grad_log_collision: np.ndarray
    cosine_between_gradients: float
    projected_ood_norm: float
    projected_collision_norm: float
    finite_difference_step: float
    gradient_stability_max_abs: float
    gradient_stability_relative_l2: float


def compute_mechanism_directions(
    z_geo: np.ndarray | Iterable[float],
    support_a: float,
    support_span: float,
    train_length: int,
    target_factors: tuple[int, ...] = (2, 4, 8),
    *,
    finite_difference_step: float = 1e-6,
    minimum_projected_norm: float = 1e-8,
) -> MechanismDirections:
    """Compute the two declared local projected directions around Geo."""
    z = _float64_vector(z_geo, name="z_geo")
    if z.size < 3 or support_span <= 0.0 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z_geo must be a strictly increasing fixed-support grid")
    if z[0] != 0.0 or z[-1] != 1.0:
        raise ValueError("z_geo endpoints must be exactly 0 and 1")
    interior = z[1:-1].copy()

    def frequencies(candidate: np.ndarray) -> np.ndarray:
        full = np.concatenate(([0.0], candidate, [1.0]))
        return np.exp(-(float(support_a) + float(support_span) * full))

    def log_ood(candidate: np.ndarray) -> float:
        value, _ = table_phase_ood(frequencies(candidate), train_length, target_factors)
        if value <= 0.0:
            raise MechanismNotLocallySeparable("phase OOD is zero at the declared support")
        return math.log(value)

    def log_collision(candidate: np.ndarray) -> float:
        value, _ = table_collision(frequencies(candidate), train_length)
        if value <= 0.0:
            raise MechanismNotLocallySeparable("collision is zero at the declared support")
        return math.log(value)

    grad_ood = _central_gradient(log_ood, interior, finite_difference_step)
    grad_collision = _central_gradient(log_collision, interior, finite_difference_step)
    refined_ood = _central_gradient(log_ood, interior, finite_difference_step / 2.0)
    refined_collision = _central_gradient(log_collision, interior, finite_difference_step / 2.0)
    stacked = np.concatenate((grad_ood, grad_collision))
    refined = np.concatenate((refined_ood, refined_collision))
    gradient_stability_max_abs = float(np.max(np.abs(stacked - refined)))
    gradient_stability_relative_l2 = float(
        np.linalg.norm(stacked - refined) / max(np.linalg.norm(refined), np.finfo(np.float64).tiny)
    )
    norm_ood = float(np.linalg.norm(grad_ood))
    norm_collision = float(np.linalg.norm(grad_collision))
    if min(norm_ood, norm_collision) <= minimum_projected_norm:
        raise MechanismNotLocallySeparable("MECHANISMS_NOT_LOCALLY_SEPARABLE: zero mechanism gradient")
    projected_ood = grad_ood - np.dot(grad_ood, grad_collision) / np.dot(grad_collision, grad_collision) * grad_collision
    projected_collision = grad_collision - np.dot(grad_collision, grad_ood) / np.dot(grad_ood, grad_ood) * grad_ood
    projected_ood_norm = float(np.linalg.norm(projected_ood))
    projected_collision_norm = float(np.linalg.norm(projected_collision))
    if min(projected_ood_norm, projected_collision_norm) < minimum_projected_norm:
        raise MechanismNotLocallySeparable("MECHANISMS_NOT_LOCALLY_SEPARABLE: projected gradient is degenerate")
    return MechanismDirections(
        d_ood=-projected_ood / projected_ood_norm,
        d_collision=-projected_collision / projected_collision_norm,
        grad_log_ood=grad_ood,
        grad_log_collision=grad_collision,
        cosine_between_gradients=float(np.dot(grad_ood, grad_collision) / (norm_ood * norm_collision)),
        projected_ood_norm=projected_ood_norm,
        projected_collision_norm=projected_collision_norm,
        finite_difference_step=float(finite_difference_step),
        gradient_stability_max_abs=gradient_stability_max_abs,
        gradient_stability_relative_l2=gradient_stability_relative_l2,
    )


def _table_record(z: np.ndarray, geo_frequencies: np.ndarray, support_a: float, support_span: float) -> dict:
    frequencies = np.exp(-(support_a + support_span * z))
    frequencies[0] = geo_frequencies[0]
    frequencies[-1] = geo_frequencies[-1]
    fp32 = frequencies.astype("<f4")
    fp32[0] = np.float32(geo_frequencies[0])
    fp32[-1] = np.float32(geo_frequencies[-1])
    return {
        "z": z.tolist(),
        "log_frequency": (-np.log(frequencies)).tolist(),
        "values_float64": frequencies.tolist(),
        "values_float32": fp32.astype(np.float64).tolist(),
        "sha256_float32": hashlib.sha256(fp32.tobytes()).hexdigest(),
        "strict_order_float64": bool(np.all(frequencies[:-1] > frequencies[1:])),
        "strict_order_float32": bool(np.all(fp32[:-1] > fp32[1:])),
        "endpoint_equal_float64": bool(frequencies[0] == geo_frequencies[0] and frequencies[-1] == geo_frequencies[-1]),
        "endpoint_equal_float32": bool(fp32[0] == np.float32(geo_frequencies[0]) and fp32[-1] == np.float32(geo_frequencies[-1])),
    }


def _pareto_frontier(rows: list[dict]) -> list[float]:
    candidates = [row for row in rows if row["valid_ordering"] and row["correct_direction_signs"]]
    result = []
    for row in candidates:
        dominated = any(
            other is not row
            and other["minimum_target_log_separation"] >= row["minimum_target_log_separation"] - 1e-15
            and other["maximum_leakage_ratio"] <= row["maximum_leakage_ratio"] + 1e-15
            and (
                other["minimum_target_log_separation"] > row["minimum_target_log_separation"] + 1e-15
                or other["maximum_leakage_ratio"] < row["maximum_leakage_ratio"] - 1e-15
            )
            for other in candidates
        )
        if not dominated:
            result.append(float(row["delta"]))
    return result


def audit_mechanism_deltas(
    geo_frequencies: np.ndarray | Iterable[float],
    train_length: int,
    target_factors: tuple[int, ...] = (2, 4, 8),
    delta_grid: tuple[float, ...] = DELTA_GRID,
    *,
    target_log_separation: float = 0.10,
    maximum_leakage_ratio: float = 0.20,
    ordering_fraction_of_geo_gap: float = 0.25,
) -> dict:
    """Audit every declared delta and retain strict, Pareto, or unresolved arms."""
    geo = _float64_vector(geo_frequencies, name="geo_frequencies")
    if geo.size < 3 or np.any(geo <= 0.0) or np.any(geo[:-1] <= geo[1:]):
        raise ValueError("Geo frequencies must be positive and strictly decreasing")
    if not delta_grid or any(delta <= 0.0 for delta in delta_grid) or len(set(delta_grid)) != len(delta_grid):
        raise ValueError("delta_grid must contain unique positive values")
    log_values = -np.log(geo)
    support_a = float(log_values[0])
    support_span = float(log_values[-1] - log_values[0])
    if support_span <= 0.0:
        raise ValueError("Geo support span must be positive")
    z_geo = (log_values - support_a) / support_span
    z_geo[0], z_geo[-1] = 0.0, 1.0
    expected_geo = np.linspace(0.0, 1.0, geo.size)
    if not np.allclose(z_geo, expected_geo, rtol=0.0, atol=2e-6):
        raise ValueError("the supplied center is not a geometric fixed-support table")
    try:
        directions = compute_mechanism_directions(
            z_geo, support_a, support_span, train_length, target_factors
        )
    except MechanismNotLocallySeparable as error:
        return {
            "status": "unresolved",
            "receipt_kind": "CPU_MECHANISM_FAILURE",
            "failure_class": "SCIENTIFIC_UNRESOLVED",
            "reason": str(error),
            "proxy_scope": "operator geometry only; no model or task conclusion",
            "geo": _table_record(z_geo, geo, support_a, support_span),
            "delta_records": [],
            "pareto_deltas": [],
            "selected_delta": None,
        }

    center_ood, center_ood_by_factor = table_phase_ood(geo, train_length, target_factors)
    center_collision_details = table_collision_details(geo, train_length)
    center_collision = center_collision_details["collision"]
    distances = np.arange(train_length, dtype=np.float64)
    weights = causal_separation_weights(train_length)
    overlap_crosscheck = max(
        abs(
            canonical_pair_overlap(geo[left], geo[right], distances, weights)
            - center_collision_details["overlap_matrix"][left, right]
        )
        for left in range(geo.size)
        for right in range(left + 1, geo.size)
    )
    geo_gap = 1.0 / float(geo.size - 1)
    records = []
    for delta in delta_grid:
        arms_z = {
            "geo": z_geo.copy(),
            "ood_plus": np.concatenate(([0.0], z_geo[1:-1] + delta * directions.d_ood, [1.0])),
            "ood_minus": np.concatenate(([0.0], z_geo[1:-1] - delta * directions.d_ood, [1.0])),
            "col_plus": np.concatenate(([0.0], z_geo[1:-1] + delta * directions.d_collision, [1.0])),
            "col_minus": np.concatenate(([0.0], z_geo[1:-1] - delta * directions.d_collision, [1.0])),
        }
        arms = {}
        for name, z in arms_z.items():
            table = _table_record(z, geo, support_a, support_span)
            frequencies = np.asarray(table["values_float64"], dtype=np.float64)
            ood, per_factor = table_phase_ood(frequencies, train_length, target_factors)
            collision, effective_rank = table_collision(frequencies, train_length)
            displacement = float(np.sqrt(np.mean(np.square(support_span * (z[1:-1] - z_geo[1:-1])))))
            arms[name] = {
                **table,
                "phase_ood": ood,
                "phase_ood_by_factor": {str(key): value for key, value in per_factor.items()},
                "collision": collision,
                "effective_rank": effective_rank,
                "rms_log_frequency_displacement": displacement,
                "minimum_z_gap": float(np.min(np.diff(z))),
            }
        ood_target = abs(math.log(arms["ood_plus"]["phase_ood"] / arms["ood_minus"]["phase_ood"]))
        ood_leakage = abs(math.log(arms["ood_plus"]["collision"] / arms["ood_minus"]["collision"]))
        col_target = abs(math.log(arms["col_plus"]["collision"] / arms["col_minus"]["collision"]))
        col_leakage = abs(math.log(arms["col_plus"]["phase_ood"] / arms["col_minus"]["phase_ood"]))
        ood_ratio = ood_leakage / ood_target if ood_target > 0.0 else math.inf
        col_ratio = col_leakage / col_target if col_target > 0.0 else math.inf
        valid_ordering = all(
            arm["minimum_z_gap"] >= ordering_fraction_of_geo_gap * geo_gap - 1e-15
            and arm["strict_order_float64"]
            and arm["strict_order_float32"]
            and arm["endpoint_equal_float64"]
            and arm["endpoint_equal_float32"]
            for arm in arms.values()
        )
        equal_displacement = (
            abs(arms["ood_plus"]["rms_log_frequency_displacement"] - arms["ood_minus"]["rms_log_frequency_displacement"]) <= 1e-12
            and abs(arms["col_plus"]["rms_log_frequency_displacement"] - arms["col_minus"]["rms_log_frequency_displacement"]) <= 1e-12
            and abs(arms["ood_plus"]["rms_log_frequency_displacement"] - arms["col_plus"]["rms_log_frequency_displacement"]) <= 1e-12
        )
        correct_signs = (
            arms["ood_plus"]["phase_ood"] < arms["ood_minus"]["phase_ood"]
            and arms["col_plus"]["collision"] < arms["col_minus"]["collision"]
        )
        strict = (
            valid_ordering
            and equal_displacement
            and correct_signs
            and ood_target >= target_log_separation
            and col_target >= target_log_separation
            and ood_ratio <= maximum_leakage_ratio
            and col_ratio <= maximum_leakage_ratio
        )
        records.append({
            "delta": float(delta),
            "arms": arms,
            "valid_ordering": valid_ordering,
            "equal_displacement": equal_displacement,
            "correct_direction_signs": correct_signs,
            "ood_target_log_separation": ood_target,
            "ood_nuisance_log_separation": ood_leakage,
            "ood_leakage_ratio": ood_ratio,
            "collision_target_log_separation": col_target,
            "collision_nuisance_log_separation": col_leakage,
            "collision_leakage_ratio": col_ratio,
            "minimum_target_log_separation": min(ood_target, col_target),
            "maximum_leakage_ratio": max(ood_ratio, col_ratio),
            "strict_contract_passed": strict,
        })
    strict_deltas = [row["delta"] for row in records if row["strict_contract_passed"]]
    pareto_deltas = _pareto_frontier(records)
    if strict_deltas:
        status = "strict-feasible"
        receipt_kind = "CPU_MECHANISM_ARMS_READY"
        selected_delta = max(strict_deltas)
        failure_class = None
        reason = None
    elif pareto_deltas:
        status = "pareto"
        receipt_kind = "CPU_MECHANISM_FAILURE"
        selected_delta = None
        failure_class = "SCIENTIFIC_UNRESOLVED"
        reason = "no common delta satisfies the declared purity contract; Pareto arms are retained without a clean-isolation claim"
    else:
        status = "unresolved"
        receipt_kind = "CPU_MECHANISM_FAILURE"
        selected_delta = None
        failure_class = "SCIENTIFIC_UNRESOLVED"
        reason = "no ordered delta preserves the declared directional signs"
    return {
        "status": status,
        "receipt_kind": receipt_kind,
        "failure_class": failure_class,
        "reason": reason,
        "proxy_scope": "operator geometry only; no model or task conclusion",
        "contract": {
            "train_length": int(train_length),
            "target_factors": list(target_factors),
            "delta_grid": [float(value) for value in delta_grid],
            "target_log_separation": float(target_log_separation),
            "maximum_leakage_ratio": float(maximum_leakage_ratio),
            "ordering_fraction_of_geo_gap": float(ordering_fraction_of_geo_gap),
            "support_a": support_a,
            "support_span": support_span,
            "pairs": int(geo.size),
        },
        "center_metrics": {
            "phase_ood": center_ood,
            "phase_ood_by_factor": {str(key): value for key, value in center_ood_by_factor.items()},
            "collision": center_collision,
            "effective_rank": center_collision_details["effective_rank"],
            "collision_numerics": {
                key: center_collision_details[key]
                for key in (
                    "eig_floor",
                    "self_gram_floor_count",
                    "minimum_self_gram_eigenvalue",
                    "raw_overlap_min",
                    "raw_overlap_max",
                )
            } | {"discrete_vectorized_crosscheck_max_abs": overlap_crosscheck},
        },
        "directions": {
            "d_ood": directions.d_ood.tolist(),
            "d_collision": directions.d_collision.tolist(),
            "grad_log_ood": directions.grad_log_ood.tolist(),
            "grad_log_collision": directions.grad_log_collision.tolist(),
            "cosine_between_gradients": directions.cosine_between_gradients,
            "projected_ood_norm": directions.projected_ood_norm,
            "projected_collision_norm": directions.projected_collision_norm,
            "finite_difference_step": directions.finite_difference_step,
            "gradient_stability_max_abs": directions.gradient_stability_max_abs,
            "gradient_stability_relative_l2": directions.gradient_stability_relative_l2,
        },
        "geo": _table_record(z_geo, geo, support_a, support_span),
        "delta_records": records,
        "strict_deltas": strict_deltas,
        "pareto_deltas": pareto_deltas,
        "selected_delta": selected_delta,
        "gpu_authorization": "not granted by this CPU receipt",
    }
