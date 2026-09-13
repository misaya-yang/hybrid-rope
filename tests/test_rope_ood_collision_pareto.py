import numpy as np

from scripts.analysis.audit_rope_ood_collision_pareto import (
    dominance,
    nearest_native_distances,
    position_codes,
    separation_curve,
)


def test_separation_curve_matches_direct_code_distance():
    frequencies = np.asarray([1.0, 0.25])
    observed = separation_curve(frequencies, 5)
    origin = position_codes(frequencies, np.asarray([0]))[0]
    expected = np.linalg.norm(position_codes(frequencies, np.arange(1, 6)) - origin, axis=1)
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)


def test_native_discrete_codes_have_zero_ood_on_the_same_integer_domain():
    frequencies = np.asarray([1.0, 0.25])
    native = position_codes(frequencies, np.arange(9))
    observed = nearest_native_distances(native, frequencies, 8, block_size=3)
    np.testing.assert_allclose(observed, 0.0, atol=3e-8)


def test_pareto_dominance_records_task_reversal():
    rows = [
        {"name": "geometry", "ood_max": 0.1, "sep_min": 0.3, "task_score": 0.2},
        {"name": "task", "ood_max": 0.2, "sep_min": 0.2, "task_score": 0.8},
    ]
    frontier, contradictions = dominance(rows, "ood_max")
    assert frontier == ["geometry"]
    assert contradictions[0]["geometric_dominator"] == "geometry"
    assert contradictions[0]["task_better_profile"] == "task"
