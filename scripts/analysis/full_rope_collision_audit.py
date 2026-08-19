#!/usr/bin/env python3
"""CPU-only audit of finite-K RoPE frequency-subspace redundancy.

The measure is uniform on Delta in [0, L].  Every compared schedule has the
same physical endpoints omega in [1/base, 1].  Numerical optima are explicitly
candidate-grid optima, not certified continuous global optima.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(np.asarray(x, dtype=np.float64) / math.pi)


def _j(x: np.ndarray) -> np.ndarray:
    """(1-cos(x))/x, evaluated without cancellation near zero."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = xs / 2.0 - xs**3 / 24.0 + xs**5 / 720.0
    out[~small] = 2.0 * np.sin(x[~small] / 2.0) ** 2 / x[~small]
    return out


def gram_blocks(omega: np.ndarray, length: float) -> np.ndarray:
    """E [cos(w_i D), sin(w_i D)]^T [cos(w_j D), sin(w_j D)]."""
    omega = np.asarray(omega, dtype=np.float64)
    left = omega[:, None]
    right = omega[None, :]
    difference = (left - right) * float(length)
    total = (left + right) * float(length)
    blocks = np.empty((len(omega), len(omega), 2, 2), dtype=np.float64)
    blocks[:, :, 0, 0] = 0.5 * (_sinc(difference) + _sinc(total))
    blocks[:, :, 1, 1] = 0.5 * (_sinc(difference) - _sinc(total))
    blocks[:, :, 0, 1] = 0.5 * (_j(total) - _j(difference))
    blocks[:, :, 1, 0] = 0.5 * (_j(total) + _j(difference))
    return blocks


def _block_matrix(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    return blocks.transpose(0, 2, 1, 3).reshape(2 * n, 2 * n)


def whitened_blocks(blocks: np.ndarray) -> np.ndarray:
    diagonal = blocks[np.arange(len(blocks)), np.arange(len(blocks))]
    values, vectors = np.linalg.eigh(diagonal)
    if float(values.min()) <= 0.0:
        raise FloatingPointError("non-positive self Gram")
    inverse_roots = np.einsum(
        "...ik,...k,...jk->...ij", vectors, values ** -0.5, vectors
    )
    return np.einsum(
        "aik,abkl,blj->abij", inverse_roots, blocks, inverse_roots,
        optimize=True,
    )


def _entropy_rank(matrix: np.ndarray) -> float:
    values = np.clip(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)), 0.0, None)
    values = values[values > max(float(values.max()) * 1e-14, 1e-15)]
    probability = values / values.sum()
    return float(np.exp(-np.sum(probability * np.log(probability))))


def schedule_metrics(phi: np.ndarray, length: int, base: float) -> dict[str, float]:
    omega = np.power(float(base), -np.asarray(phi, dtype=np.float64))
    blocks = gram_blocks(omega, length)
    white = whitened_blocks(blocks)
    pair_index = np.triu_indices(len(phi), 1)

    cosine = blocks[:, :, 0, 0]
    cosine /= np.sqrt(np.outer(np.diag(cosine), np.diag(cosine)))
    cosine_collision = float(np.mean(cosine[pair_index] ** 2))

    affinity = 0.5 * np.sum(white**2, axis=(2, 3))
    full_collision = float(np.mean(affinity[pair_index]))
    stable_rank = float(
        2.0 * len(phi) / (1.0 + (len(phi) - 1.0) * full_collision)
    )
    full_gram = _block_matrix(blocks)
    white_gram = _block_matrix(white)
    direct_stable_rank = float(np.trace(white_gram) ** 2 / np.sum(white_gram**2))
    if not math.isclose(stable_rank, direct_stable_rank, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError("pairwise collision/stable-rank identity failed")
    eigenvalues = np.linalg.eigvalsh(0.5 * (white_gram + white_gram.T))
    logdet = float(np.mean(np.log(np.clip(eigenvalues, 1e-14, None))))
    return {
        "cosine_collision_mean": cosine_collision,
        "full_subspace_collision_mean": full_collision,
        "full_whitened_stable_rank": stable_rank,
        "full_whitened_entropy_rank": _entropy_rank(white_gram),
        "raw_full_entropy_rank": _entropy_rank(full_gram),
        "whitened_logdet_per_dimension": logdet,
    }


def geometric_phi(pairs: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(pairs), dtype=np.float64)


def evq_phi(pairs: int, tau: float) -> np.ndarray:
    u = geometric_phi(pairs)
    if abs(float(tau)) < 1e-8:
        return u
    return 1.0 - np.arcsinh((1.0 - u) * math.sinh(float(tau))) / float(tau)


def _candidate_phi(length: int, base: float, *, cosine: bool, grid: int) -> np.ndarray:
    spacing = math.pi if cosine else 2.0 * math.pi
    highest = int(math.floor(float(length) / spacing))
    harmonics = spacing * np.arange(1, highest + 1, dtype=np.float64) / length
    harmonics = harmonics[(harmonics >= 1.0 / base) & (harmonics <= 1.0)]
    harmonic_phi = -np.log(harmonics) / math.log(base)
    values = np.concatenate((np.linspace(0.0, 1.0, grid), harmonic_phi, [0.0, 1.0]))
    return np.unique(np.round(np.clip(values, 0.0, 1.0), 14))


def _pairwise_exchange(
    cost: np.ndarray, pairs: int, preferred: np.ndarray
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    greedy = [0, len(cost) - 1]
    available = np.ones(len(cost), dtype=bool)
    available[greedy] = False
    while len(greedy) < pairs:
        incremental = cost[:, greedy].sum(axis=1)
        incremental[~available] = np.inf
        chosen = int(np.argmin(incremental))
        greedy.append(chosen)
        available[chosen] = False

    starts: list[tuple[str, list[int]]] = [("greedy", greedy)]
    if len(preferred) >= pairs - 2:
        endpoint_cost = cost[preferred, 0] + cost[preferred, -1]
        harmonic = preferred[np.argsort(endpoint_cost)[: pairs - 2]].tolist()
        starts.append(("harmonic", [0, len(cost) - 1, *harmonic]))

    candidates = []
    for start_name, start in starts:
        selected = start.copy()
        exchanges = 0
        for _ in range(8):
            changed = False
            for position in range(2, len(selected)):
                old = selected[position]
                remainder = selected[:position] + selected[position + 1 :]
                incremental = cost[:, remainder].sum(axis=1)
                incremental[remainder] = np.inf
                chosen = int(np.argmin(incremental))
                if incremental[chosen] + 1e-12 < incremental[old]:
                    selected[position] = chosen
                    exchanges += 1
                    changed = True
            if not changed:
                break
        chosen = np.array(sorted(selected), dtype=np.int64)
        objective = float(np.sum(np.triu(cost[np.ix_(chosen, chosen)], 1)))
        candidates.append((objective, chosen, exchanges, start_name))
    objective, chosen, exchanges, start_name = min(candidates, key=lambda row: row[0])
    return chosen, {
        "exchanges": exchanges,
        "pairwise_sum": objective,
        "selected_start": start_name,
    }


def _d_optimal_greedy(white: np.ndarray, pairs: int) -> tuple[np.ndarray, dict[str, float | int]]:
    selected = [0, len(white) - 1]
    all_indices = np.arange(len(white))
    while len(selected) < pairs:
        current = _block_matrix(white[np.ix_(selected, selected)])
        inverse = np.linalg.inv(current)
        cross = white[np.asarray(selected)[:, None], all_indices[None, :]]
        cross = cross.transpose(1, 0, 2, 3).reshape(len(white), 2 * len(selected), 2)
        conditional = np.eye(2)[None] - np.einsum(
            "mki,kl,mlj->mij", cross, inverse, cross, optimize=True
        )
        determinant = (
            conditional[:, 0, 0] * conditional[:, 1, 1]
            - conditional[:, 0, 1] * conditional[:, 1, 0]
        )
        determinant[np.asarray(selected)] = -np.inf
        determinant[determinant <= 1e-14] = -np.inf
        chosen = int(np.argmax(determinant))
        if not np.isfinite(determinant[chosen]):
            raise RuntimeError("D-optimal candidate pool became singular")
        selected.append(chosen)
    chosen = np.array(sorted(selected), dtype=np.int64)
    matrix = _block_matrix(white[np.ix_(chosen, chosen)])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise RuntimeError("D-optimal selected Gram is not positive definite")
    return chosen, {"greedy_logdet": float(logdet), "exchanges": 0}


def _best_logdet_start(
    white: np.ndarray, starts: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, float | str]]:
    candidates = []
    for name, selected in starts.items():
        matrix = _block_matrix(white[np.ix_(selected, selected)])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sign, logdet = np.linalg.slogdet(matrix)
        if sign > 0:
            candidates.append((float(logdet), name, selected))
    if not candidates:
        raise RuntimeError("all logdet starts are singular")
    logdet, name, selected = max(candidates, key=lambda row: row[0])
    return selected, {
        "selected_start": name,
        "logdet": logdet,
        "gap_to_correlation_logdet_upper_bound_zero": -logdet,
    }


def _pool_geometry(phi: np.ndarray, length: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    blocks = gram_blocks(np.power(base, -phi), length)
    white = whitened_blocks(blocks)
    affinity = 0.5 * np.sum(white**2, axis=(2, 3))
    return white, affinity


def _symbolic_collapse_coefficient() -> Fraction:
    """Exact HS graph coefficient using polynomial L2[0,1] inner products."""
    def inner(left: list[Fraction], right: list[Fraction]) -> Fraction:
        return sum(
            a * b / Fraction(i + j + 1)
            for i, a in enumerate(left)
            for j, b in enumerate(right)
        )

    basis = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    perturbation = [
        [Fraction(0), Fraction(0), Fraction(-1, 2)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(-1, 6)],
    ]
    gram = [[inner(a, b) for b in basis] for a in basis]
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    inverse = [
        [gram[1][1] / determinant, -gram[0][1] / determinant],
        [-gram[1][0] / determinant, gram[0][0] / determinant],
    ]
    cross = [[inner(a, b) for b in perturbation] for a in basis]
    raw = [[inner(a, b) for b in perturbation] for a in perturbation]
    residual = [[raw[i][j] for j in range(2)] for i in range(2)]
    for i in range(2):
        for j in range(2):
            residual[i][j] -= sum(
                cross[a][i] * inverse[a][b] * cross[b][j]
                for a in range(2) for b in range(2)
            )
    return sum(inverse[i][j] * residual[j][i] for i in range(2) for j in range(2))


def _self_checks() -> dict[str, float | str]:
    coefficient = _symbolic_collapse_coefficient()
    assert coefficient == Fraction(19, 12600)

    omega = np.array([0.17, 0.41])
    length = 3.7
    analytic = gram_blocks(omega, length)
    delta = np.linspace(0.0, length, 200_001)
    features = np.stack((np.cos(delta[:, None] * omega), np.sin(delta[:, None] * omega)), axis=2)
    numerical = np.einsum("nia,njb->ijab", features, features) / len(delta)
    quadrature_error = float(np.max(np.abs(analytic - numerical)))
    assert quadrature_error < 3e-6

    blocks = gram_blocks(np.array([0.11, 0.27]), 1.0)
    before = np.linalg.svd(whitened_blocks(blocks)[0, 1], compute_uv=False)
    angles = (0.37, -0.91)
    rotations = np.array([
        [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]] for a in angles
    ])
    rotated = np.einsum("aki,abkl,blj->abij", rotations, blocks, rotations, optimize=True)
    after = np.linalg.svd(whitened_blocks(rotated)[0, 1], compute_uv=False)
    rotation_error = float(np.max(np.abs(before - after)))
    assert rotation_error < 1e-11

    x, y = 0.05, 0.10
    small = gram_blocks(np.array([x, y]), 1.0)
    q = whitened_blocks(small)[0, 1]
    observed = 2.0 - float(np.sum(q**2))
    predicted = float(coefficient) * (x * x - y * y) ** 2
    ratio = observed / predicted
    assert abs(ratio - 1.0) < 0.01
    return {
        "analytic_vs_quadrature_max_abs": quadrature_error,
        "phase_rotation_canonical_correlation_max_abs": rotation_error,
        "low_frequency_symbolic_coefficient": f"{coefficient.numerator}/{coefficient.denominator}",
        "low_frequency_numeric_over_leading": ratio,
    }


def _low_frequency_report(pairs: int, length: int, base: float) -> dict[str, float | int]:
    phi = geometric_phi(pairs)
    omega = np.power(base, -phi)
    mask = omega * length <= 1.0
    selected = phi[mask]
    metrics = schedule_metrics(selected, length, base)
    return {
        "pairs_total": pairs,
        "pairs_with_omega_L_le_1": int(mask.sum()),
        "available_dimensions_in_low_band": int(2 * mask.sum()),
        "low_band_whitened_stable_rank": metrics["full_whitened_stable_rank"],
        "low_band_raw_entropy_rank": metrics["raw_full_entropy_rank"],
        "low_band_collision_mean": metrics["full_subspace_collision_mean"],
        "stable_dimension_loss_fraction": 1.0
        - metrics["full_whitened_stable_rank"] / (2.0 * mask.sum()),
    }


def _hash_phi(phi: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(phi, dtype=np.float64).tobytes()).hexdigest()


def _discrete_full_collision(phi: list[float], length: int, base: float) -> float:
    omega = np.power(base, -np.asarray(phi, dtype=np.float64))
    phase = np.outer(np.arange(length, dtype=np.float64), omega)
    features = np.stack((np.cos(phase), np.sin(phase)), axis=2)
    blocks = np.einsum("nia,njb->ijab", features, features, optimize=True) / length
    affinity = 0.5 * np.sum(whitened_blocks(blocks) ** 2, axis=(2, 3))
    return float(np.mean(affinity[np.triu_indices(len(phi), 1)]))


def _random_schedules(count: int, pairs: int, rng: np.random.Generator) -> list[np.ndarray]:
    schedules = []
    while len(schedules) < count:
        concentration = 10.0 ** rng.uniform(-0.25, 0.45)
        gaps = rng.dirichlet(np.full(pairs - 1, concentration))
        if float(gaps.min()) < 1e-5:
            continue
        schedules.append(np.concatenate(([0.0], np.cumsum(gaps))))
    return schedules


def _counterexamples(
    schedules: list[np.ndarray], length: int, base: float, rng: np.random.Generator
) -> dict[str, dict[str, object]]:
    rows = []
    for phi in schedules:
        at_l = schedule_metrics(phi, length, base)
        at_2l = schedule_metrics(phi, 2 * length, base)
        at_4l = schedule_metrics(phi, 4 * length, base)
        rows.append((phi, at_l, at_2l, at_4l))
    cosine = np.array([row[1]["cosine_collision_mean"] for row in rows])
    raw_rank = np.array([row[1]["raw_full_entropy_rank"] for row in rows])
    white_rank = np.array([row[1]["full_whitened_stable_rank"] for row in rows])
    collision_l = np.array([row[1]["full_subspace_collision_mean"] for row in rows])
    collision_2l = np.array([row[2]["full_subspace_collision_mean"] for row in rows])
    collision_4l = np.array([row[3]["full_subspace_collision_mean"] for row in rows])

    left = rng.integers(0, len(rows), size=1_000_000)
    right = rng.integers(0, len(rows), size=1_000_000)
    # The first five rows are the named K=16 methods; prefer the clean
    # cosine-grid-optimum versus full-subspace-grid-optimum counterexample.
    a, b = 2, 3
    if not (
        cosine[a] < cosine[b]
        and raw_rank[a] < raw_rank[b]
        and white_rank[a] < white_rank[b]
    ):
        mask = (
            (cosine[left] < cosine[right])
            & (raw_rank[left] < raw_rank[right])
            & (white_rank[left] < white_rank[right])
        )
        score = np.where(
            mask,
            (cosine[right] - cosine[left])
            * (raw_rank[right] - raw_rank[left])
            * (white_rank[right] - white_rank[left]),
            -np.inf,
        )
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            raise RuntimeError("no cosine/full-rank counterexample found")
        a, b = int(left[best]), int(right[best])

    reversal_mask = (collision_l[left] < collision_l[right]) & (
        collision_4l[left] > collision_4l[right]
    )
    reversal_score = np.where(
        reversal_mask,
        (collision_l[right] - collision_l[left])
        * (collision_4l[left] - collision_4l[right]),
        -np.inf,
    )
    best_reversal = int(np.argmax(reversal_score))
    if not np.isfinite(reversal_score[best_reversal]):
        raise RuntimeError("no length-reversal counterexample found")
    c, d = int(left[best_reversal]), int(right[best_reversal])

    def record(index: int) -> dict[str, object]:
        phi, at_l, at_2l, at_4l = rows[index]
        return {
            "phi": phi.tolist(),
            "phi_sha256_float64": _hash_phi(phi),
            "metrics_L": at_l,
            "metrics_2L": at_2l,
            "metrics_4L": at_4l,
        }

    return {
        "cosine_lower_but_full_rank_worse": {"A": record(a), "B": record(b)},
        "full_collision_ranking_reverses_by_4L": {"A": record(c), "B": record(d)},
    }


def run(length: int, base: float, candidate_grid: int, random_count: int) -> dict[str, object]:
    if length < 2 or base <= 1.0 or candidate_grid < 33 or random_count < 100:
        raise ValueError("require length>=2, base>1, candidate-grid>=33, random-schedules>=100")
    started = time.time()
    checks = _self_checks()
    cosine_phi = _candidate_phi(length, base, cosine=True, grid=candidate_grid)
    full_phi = _candidate_phi(length, base, cosine=False, grid=candidate_grid)

    cosine_blocks = gram_blocks(np.power(base, -cosine_phi), length)
    cosine_matrix = cosine_blocks[:, :, 0, 0]
    cosine_matrix /= np.sqrt(
        np.outer(np.diag(cosine_matrix), np.diag(cosine_matrix))
    )
    cosine_cost = cosine_matrix**2
    full_white, full_cost = _pool_geometry(full_phi, length, base)
    cosine_harmonics = np.flatnonzero(
        np.abs(np.power(base, -cosine_phi) * length / math.pi
               - np.rint(np.power(base, -cosine_phi) * length / math.pi)) < 1e-8
    )
    full_harmonics = np.flatnonzero(
        np.abs(np.power(base, -full_phi) * length / (2.0 * math.pi)
               - np.rint(np.power(base, -full_phi) * length / (2.0 * math.pi))) < 1e-8
    )
    cosine_harmonics = cosine_harmonics[
        (cosine_harmonics != 0) & (cosine_harmonics != len(cosine_phi) - 1)
    ]
    full_harmonics = full_harmonics[
        (full_harmonics != 0) & (full_harmonics != len(full_phi) - 1)
    ]

    methods: dict[str, object] = {}
    named_for_counterexample = []
    for pairs in (16, 32, 64):
        cosine_selected, cosine_receipt = _pairwise_exchange(
            cosine_cost, pairs, cosine_harmonics
        )
        full_selected, full_receipt = _pairwise_exchange(
            full_cost, pairs, full_harmonics
        )
        d_greedy, _ = _d_optimal_greedy(full_white, pairs)
        d_selected, d_receipt = _best_logdet_start(
            full_white,
            {"d_optimal_greedy": d_greedy, "full_collision_exchange": full_selected},
        )
        tau = 2.0 * pairs / math.sqrt(length)
        schedules = {
            "geometric": geometric_phi(pairs),
            "evq_cosh": evq_phi(pairs, tau),
            "cosine_collision_grid_opt": cosine_phi[cosine_selected],
            "full_rope_collision_grid_opt": full_phi[full_selected],
            "whitened_logdet_grid_opt": full_phi[d_selected],
        }
        receipts = {
            "geometric": {"fixed_endpoints": True},
            "evq_cosh": {"fixed_endpoints": True, "tau": tau},
            "cosine_collision_grid_opt": cosine_receipt,
            "full_rope_collision_grid_opt": full_receipt,
            "whitened_logdet_grid_opt": d_receipt,
        }
        records = {}
        for name, phi in schedules.items():
            assert phi[0] == 0.0 and phi[-1] == 1.0 and np.all(np.diff(phi) > 0.0)
            records[name] = {
                "phi": phi.tolist(),
                "phi_sha256_float64": _hash_phi(phi),
                "receipt": receipts[name],
                "metrics_L": schedule_metrics(phi, length, base),
                "metrics_2L": schedule_metrics(phi, 2 * length, base),
                "metrics_4L": schedule_metrics(phi, 4 * length, base),
            }
        methods[str(pairs)] = records
        if pairs == 16:
            named_for_counterexample.extend(schedules.values())

    rng = np.random.default_rng(20260819)
    random_schedules = _random_schedules(random_count, 16, rng)
    counterexamples = _counterexamples(
        named_for_counterexample + random_schedules, length, base, rng
    )
    counterexamples["cosine_lower_but_full_rank_worse"] = {
        "K": 64,
        "A_method": "cosine_collision_grid_opt",
        "B_method": "full_rope_collision_grid_opt",
        "A": methods["64"]["cosine_collision_grid_opt"],
        "B": methods["64"]["full_rope_collision_grid_opt"],
    }
    reversal = counterexamples["full_collision_ranking_reverses_by_4L"]
    reversal["discrete_uniform_confirmation"] = {
        f"{multiple}L": {
            side: _discrete_full_collision(
                reversal[side]["phi"], multiple * length, base
            )
            for side in ("A", "B")
        }
        for multiple in (1, 2, 4)
    }
    return {
        "status": "CPU_ONLY_COMPLETE",
        "scope": {
            "distance_measure": "continuous uniform Delta in [0,L]",
            "length": length,
            "base": base,
            "physical_frequency_endpoints": [1.0 / base, 1.0],
            "candidate_grid_points": candidate_grid,
            "continuous_global_optimum_certified": False,
            "random_counterexample_schedules": random_count,
        },
        "self_checks": checks,
        "candidate_pool_sizes": {
            "cosine": len(cosine_phi),
            "full_rope": len(full_phi),
        },
        "finite_K": methods,
        "low_frequency_collapse": {
            str(pairs): _low_frequency_report(pairs, length, base)
            for pairs in (16, 32, 64)
        },
        "counterexamples": counterexamples,
        "runtime_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--base", type=float, default=500_000.0)
    parser.add_argument("--candidate-grid", type=int, default=1025)
    parser.add_argument("--random-schedules", type=int, default=3000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/full_rope_collision_audit_20260819.json"),
    )
    args = parser.parse_args()
    result = run(args.length, args.base, args.candidate_grid, args.random_schedules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
