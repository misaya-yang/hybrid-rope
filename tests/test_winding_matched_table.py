import math

import numpy as np
import pytest

from experiments.olmo_recovery_20260912.winding_matched_table import winding_matched


def test_exact_endpoint_and_maximal_nonaccelerating_winding():
    native = np.power(500_000.0, -np.arange(64, dtype=np.float64) / 64.0)
    result = winding_matched(native, native_length=8192, scale=8.0)
    construction = result["construction"]
    assert construction["max_endpoint_circular_residual_float64"] < 1e-9
    assert all(a > b for a, b in zip(result["values_float32"], result["values_float32"][1:]))
    values = np.asarray(result["values_float32"], dtype=np.float64)
    assert np.all(values >= native / 8.0 - 1e-7)
    assert np.all(values <= native + 1e-7)


def test_low_turn_slots_reduce_to_full_interpolation():
    native = np.array([1e-5, 5e-6], dtype=np.float64)
    result = winding_matched(native, native_length=4096, scale=4.0)
    assert result["construction"]["winding_numbers"] == [0, 0]
    assert result["values_float32"] == pytest.approx((native / 4.0).tolist())


def test_rejects_invalid_frequency_order():
    with pytest.raises(ValueError, match="invalid Native"):
        winding_matched([1.0, 0.5, 0.6], native_length=4096, scale=4.0)
