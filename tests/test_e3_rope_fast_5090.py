from fractions import Fraction

import numpy as np

from experiments.rope_fast_5090_20260912.e3_tables import exact_exponents, exact_increments, tables, tensor_sha


def test_exact_c42_pair_contract():
    for arm in ("C42", "C42V24"):
        increments = exact_increments(arm)
        exponents = exact_exponents(arm)
        assert sum(increments) == 1
        assert sum(Fraction(r) * value for r, value in enumerate(increments, 1)) == 8
        assert sum(exponents) == 42
        assert exponents[:15] == [0] * 15
        assert exponents[32:] == [1] * 32
        assert all(value > 0 for value in increments)


def test_fp32_table_receipts_are_distinct_and_stable():
    payload = tables()
    assert payload["C42"]["gain"] == payload["C42V24"]["gain"]
    assert payload["C42"]["tensor_sha256"] != payload["C42V24"]["tensor_sha256"]
    for arm, receipt in payload.items():
        values = np.asarray(receipt["values_float32"], dtype=np.float32)
        assert values.shape == (64,)
        assert np.isfinite(values).all() and (values > 0).all()
        assert (values[:-1] > values[1:]).all()
        assert tensor_sha(values) == receipt["tensor_sha256"]


def test_pair_has_identical_fixed_regions_and_endpoints():
    payload = tables()
    a = np.asarray(payload["C42"]["values_float32"], dtype=np.float32)
    b = np.asarray(payload["C42V24"]["values_float32"], dtype=np.float32)
    assert np.array_equal(a[:15], b[:15])
    assert np.array_equal(a[32:], b[32:])
    assert np.array_equal(a[[0, -1]], b[[0, -1]])
    assert not np.array_equal(a[15:32], b[15:32])
