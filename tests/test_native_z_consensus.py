"""CPU checks for the Native-Z5 split-consensus direction solver."""

import numpy as np

from experiments.native_z_enhancement_20260914.consensus_direction import (
    maximin_direction,
    minimum_norm_convex_combination,
    zero_sum_basis,
)


def test_minimum_norm_convex_combination_finds_origin_between_opposites():
    weights, point = minimum_norm_convex_combination(
        np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    )
    assert np.linalg.norm(point) < 1e-10
    np.testing.assert_allclose(weights[:2], [0.5, 0.5], atol=1e-10)


def test_maximin_direction_descends_on_two_orthogonal_blocks():
    basis = zero_sum_basis(3)
    reduced_gradients = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gradients = reduced_gradients @ basis.T
    result = maximin_direction(gradients, basis.T)
    assert result["advance"] is True
    np.testing.assert_allclose(result["margin"], 1.0 / np.sqrt(2.0), atol=1e-10)
    assert max(result["unit_metric_block_slopes"]) < 0.0


def test_maximin_direction_stops_for_conflicting_blocks():
    basis = zero_sum_basis(3)
    reduced_gradients = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    gradients = reduced_gradients @ basis.T
    result = maximin_direction(gradients, basis.T)
    assert result["advance"] is False
    assert result["margin"] < 1e-10
