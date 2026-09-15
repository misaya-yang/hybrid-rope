"""CPU contract checks for the no-selection Native-Z5 all50 refit."""

import pytest

from experiments.native_z_enhancement_20260914.refit import calibration_indices


def records():
    values = []
    for split, rows in (
        ("design", range(0, 16)),
        ("selection", range(16, 32)),
        ("internal_confirm", range(32, 50)),
    ):
        values.extend({"split": split, "row": row} for row in rows)
    return values


def test_calibration_indices_requires_exact_original_union():
    assert calibration_indices({"optimization": {"records": records()}}) == list(range(50))


def test_calibration_indices_rejects_duplicate_or_missing_row():
    broken = records()
    broken[-1] = {"split": "internal_confirm", "row": 48}
    with pytest.raises(ValueError, match="exact rows"):
        calibration_indices({"optimization": {"records": broken}})
