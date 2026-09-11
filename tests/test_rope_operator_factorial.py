import numpy as np
import pytest

from experiments.rope_decision_20260911.operator_factorial import split_tables
from experiments.rope_decision_20260911.tables import build_tables


def test_finite_interventions_are_disjoint_and_reconstruct_full_change():
    # Hand-calculated example, including an exactly unchanged plateau.
    y = dict(values_float32=[8., 4., 2., 1.], gain=1.1)
    m = dict(values_float32=[8., 5., 1.5, 1.], gain=1.1)
    r = split_tables(y, m)
    a = np.array(r['tables']['only_faster']['values_float32'])
    b = np.array(r['tables']['only_slower']['values_float32'])
    np.testing.assert_array_equal(a, [8., 5., 2., 1.])
    np.testing.assert_array_equal(b, [8., 4., 1.5, 1.])
    np.testing.assert_array_equal(a + b - y['values_float32'], m['values_float32'])
    assert r['faster_slots'] == [1] and r['slower_slots'] == [2]


def test_qwen_split_preserves_plateaus_and_uses_both_interventions():
    t = build_tables(1e6, 32768, scale=4.)
    r = split_tables(t['yarn_index'], t['mrpro'])
    assert r['faster_slots'] == list(range(24, 38))
    assert r['slower_slots'] == [38, 39]
    y = np.array(t['yarn_index']['values_float32'])
    m = np.array(t['mrpro']['values_float32'])
    for arm in r['tables'].values():
        v = np.array(arm['values_float32'])
        assert np.all(np.diff(v) < 0)
        np.testing.assert_array_equal(v[:24], y[:24])
        np.testing.assert_array_equal(v[40:], y[40:])
        assert np.all((v == y) | (v == m))
        assert arm['gain'] == t['mrpro']['gain']


def test_mismatched_gain_is_not_a_frequency_factorial():
    with pytest.raises(ValueError, match='same gain'):
        split_tables(dict(values_float32=[2., 1.], gain=1.),
                     dict(values_float32=[2., 1.], gain=1.1))
