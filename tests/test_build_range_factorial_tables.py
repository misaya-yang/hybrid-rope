import pytest

from experiments.olmo_recovery_20260912.build_range_factorial_tables import build_factorial_tables


def table(values, gain):
    return {"table": {"values_float32": values, "gain": gain, "construction": {"old": True}}}


def test_builds_only_missing_shape_gain_cells():
    result = build_factorial_tables(table([1.0, 0.5, 0.2], 1.2), table([1.0, 0.4, 0.1], 1.1))
    assert result["BaseShape_SolverGain"]["values_float32"] == pytest.approx([1.0, 0.5, 0.2])
    assert result["BaseShape_SolverGain"]["gain"] == 1.1
    assert result["SolverShape_BaseGain"]["values_float32"] == pytest.approx([1.0, 0.4, 0.1])
    assert result["SolverShape_BaseGain"]["gain"] == 1.2
    assert all(cell["construction"]["same_table_all_layers_and_lengths"] for cell in result.values())


def test_rejects_dimension_or_order_drift():
    with pytest.raises(ValueError, match="dimensions differ"):
        build_factorial_tables(table([1.0, 0.5], 1.2), table([1.0, 0.4, 0.1], 1.1))
    with pytest.raises(ValueError, match="invalid table"):
        build_factorial_tables(table([1.0, 0.5, 0.6], 1.2), table([1.0, 0.4, 0.1], 1.1))
