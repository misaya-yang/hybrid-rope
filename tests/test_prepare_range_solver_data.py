import pytest

from experiments.olmo_recovery_20260912.prepare_range_solver_data import split_for_index


def test_frozen_eight_four_four_split():
    assert [split_for_index(index) for index in range(16)] == [
        *("fit" for _ in range(8)),
        *("select" for _ in range(4)),
        *("internal_confirm" for _ in range(4)),
    ]
    with pytest.raises(ValueError):
        split_for_index(16)
