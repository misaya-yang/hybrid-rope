import pytest

from experiments.olmo_recovery_20260912.bootstrap_task_equal_contrast import bootstrap_task_equal


def test_task_equal_bootstrap_preserves_task_weights():
    candidate = {
        "a": {"task": "large", "score": 1.0},
        "b": {"task": "large", "score": 1.0},
        "c": {"task": "small", "score": 0.0},
    }
    baseline = {
        "a": {"task": "large", "score": 0.0},
        "b": {"task": "large", "score": 0.0},
        "c": {"task": "small", "score": 1.0},
    }
    result = bootstrap_task_equal(
        candidate, baseline, candidate_metric="score", baseline_metric="score", samples=50, seed=7
    )
    assert result["point_delta"] == pytest.approx(0.0)
    assert result["row_stratified_percentile_95"] == pytest.approx([0.0, 0.0])


def test_task_or_row_drift_is_rejected():
    with pytest.raises(ValueError, match="row ids differ"):
        bootstrap_task_equal(
            {"a": {"task": "x", "score": 1}},
            {"b": {"task": "x", "score": 1}},
            candidate_metric="score", baseline_metric="score", samples=1, seed=1,
        )
    with pytest.raises(ValueError, match="task identity differs"):
        bootstrap_task_equal(
            {"a": {"task": "x", "score": 1}},
            {"a": {"task": "y", "score": 1}},
            candidate_metric="score", baseline_metric="score", samples=1, seed=1,
        )
