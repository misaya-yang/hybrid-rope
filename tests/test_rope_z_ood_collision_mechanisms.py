import math

import numpy as np

from experiments.rope_z_ood_collision_20260913.mechanisms import (
    audit_mechanism_deltas,
    canonical_pair_overlap,
    causal_separation_weights,
    phase_unseen_fraction,
    table_collision,
    table_collision_details,
    table_phase_ood,
)


def test_phase_unseen_fraction_has_declared_piecewise_limits_and_peak():
    length = 128
    target = 4 * length
    frequencies = np.asarray([
        2.0 * math.pi / (length - 1),
        2.0 * math.pi / (target - 1),
        math.pi / (target - 1),
    ])
    observed = phase_unseen_fraction(frequencies, length, target)
    assert observed[0] == 0.0
    assert math.isclose(
        observed[1], 1.0 - (length - 1) / (target - 1), rel_tol=0.0, abs_tol=1e-15
    )
    assert math.isclose(observed[2], frequencies[2] * (target - length) / (2.0 * math.pi))
    assert observed[1] > observed[2]


def test_table_phase_ood_keeps_per_factor_values_separate():
    aggregate, factors = table_phase_ood(np.asarray([0.1, 0.01]), 32, (2, 4), (0.25, 0.75))
    assert set(factors) == {2, 4}
    assert math.isclose(aggregate, 0.25 * factors[2] + 0.75 * factors[4])


def test_full_sine_cosine_overlap_is_symmetric_and_self_is_one():
    distances = np.arange(64, dtype=np.float64)
    weights = causal_separation_weights(64)
    assert math.isclose(canonical_pair_overlap(0.1, 0.1, distances, weights), 1.0, abs_tol=2e-12)
    left = canonical_pair_overlap(0.1, 0.03, distances, weights)
    right = canonical_pair_overlap(0.03, 0.1, distances, weights)
    assert 0.0 <= left <= 1.0
    assert math.isclose(left, right, rel_tol=0.0, abs_tol=2e-12)


def test_table_collision_effective_rank_identity():
    frequencies = np.asarray([0.4, 0.1, 0.02])
    collision, rank = table_collision(frequencies, 64)
    assert 0.0 <= collision <= 1.0
    assert math.isclose(rank, 6.0 / (1.0 + 2.0 * collision), rel_tol=1e-13)
    details = table_collision_details(frequencies, 64)
    direct = canonical_pair_overlap(0.4, 0.02, np.arange(64), causal_separation_weights(64))
    assert math.isclose(direct, details["overlap_matrix"][0, 2], rel_tol=0.0, abs_tol=2e-12)


def test_delta_audit_retains_all_pareto_arms_when_strict_contract_fails():
    frequencies = np.power(500_000.0, -np.arange(32, dtype=np.float64) / 32.0)
    result = audit_mechanism_deltas(
        frequencies,
        2048,
        delta_grid=(0.005, 0.01, 0.02),
    )
    assert result["status"] == "pareto"
    assert result["receipt_kind"] == "CPU_MECHANISM_FAILURE"
    assert result["selected_delta"] is None
    assert len(result["delta_records"]) == 3
    assert result["pareto_deltas"]
    for record in result["delta_records"]:
        assert set(record["arms"]) == {"geo", "ood_plus", "ood_minus", "col_plus", "col_minus"}
        assert record["correct_direction_signs"]
    assert result["proxy_scope"].startswith("operator geometry only")
    assert result["center_metrics"]["collision_numerics"]["discrete_vectorized_crosscheck_max_abs"] < 1e-12
    assert result["directions"]["gradient_stability_relative_l2"] < 1e-5
