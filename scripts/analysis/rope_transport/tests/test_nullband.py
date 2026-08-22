"""Contracts for the null-band operators and the graded risk axis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.rope_transport import nullband, tables  # noqa: E402

BASE = 500000.0
PAIRS = 64
NATIVE_LENGTH = 4096


def native_table() -> tables.Table:
    arr = BASE ** (-np.arange(PAIRS) / PAIRS)
    return tables.Table("native", "test", arr, tables.float32_sha256(arr), {})


def test_phi_coordinate_is_the_channel_index():
    native = native_table()
    phi = nullband.phi_coordinate(native.inv_freq, BASE)
    assert np.allclose(phi, np.arange(PAIRS) / PAIRS, atol=1e-12)


def test_no_extension_carries_no_risk():
    native = native_table()
    risk = nullband.phase_excess_risk(
        native.inv_freq, native.inv_freq,
        native_length=NATIVE_LENGTH, target_length=NATIVE_LENGTH,
    )
    assert risk["mean_turns"] == pytest.approx(0.0, abs=1e-15)


def test_a_channel_that_wrapped_in_training_is_never_at_risk():
    """The whole training-free argument rests on this clause, so pin it."""
    native = native_table()
    moved = native.inv_freq.copy()
    wrapped = native.inv_freq * NATIVE_LENGTH >= 2.0 * math.pi
    moved[wrapped] *= 0.37  # arbitrary displacement of the wrapped block only
    risk = nullband.phase_excess_risk(
        native.inv_freq, moved, native_length=NATIVE_LENGTH, target_length=8 * NATIVE_LENGTH
    )
    per_channel = risk["per_channel_turns"]
    assert np.all(per_channel[wrapped] == 0.0)


def test_beta_one_turn_budget_is_exactly_the_safety_floor():
    native = native_table()
    beta_one = nullband.turn_budget_table(
        native, scale=4.0, beta=1.0, native_length=NATIVE_LENGTH, rope_base=BASE
    )
    floor = nullband.floor_table(
        native, scale=4.0, native_length=NATIVE_LENGTH,
        target_length=4 * NATIVE_LENGTH, rope_base=BASE,
    )
    assert np.allclose(beta_one.inv_freq, floor.inv_freq, rtol=1e-12)
    risk = nullband.phase_excess_risk(
        native.inv_freq, beta_one.inv_freq,
        native_length=NATIVE_LENGTH, target_length=4 * NATIVE_LENGTH,
    )
    assert risk["mean_turns"] == pytest.approx(0.0, abs=1e-15)


def test_turn_budget_reproduces_the_official_yarn_boundary():
    """beta = 32 turns is exactly YaRN's beta_fast, so the kept block must match."""
    native = native_table()
    yarn = tables.official_yarn(
        native, scale=4.0, head_dim=2 * PAIRS, rope_base=BASE,
        original_max_position_embeddings=NATIVE_LENGTH,
    )
    budget = nullband.turn_budget_table(
        native, scale=4.0, beta=32.0, native_length=NATIVE_LENGTH, rope_base=BASE
    )
    assert budget.meta["untouched_pairs"] == yarn.meta["untouched_fast_pairs"]


def test_zero_tau_band_warp_is_the_linear_stretch():
    native = native_table()
    linear = nullband.band_warp_table(
        native, warp="identity", param=0.0, scale=4.0, band_start=22, rope_base=BASE
    )
    tiny_tau = nullband.band_warp_table(
        native, warp="evq_cosh", param=1e-10, scale=4.0, band_start=22, rope_base=BASE
    )
    assert np.allclose(linear.inv_freq, tiny_tau.inv_freq, rtol=1e-9)


def test_band_warp_pins_the_resolvable_block():
    native = native_table()
    for tau in (-3.0, -1.0, 1.0, 3.0):
        warped = nullband.band_warp_table(
            native, warp="evq_cosh", param=tau, scale=4.0, band_start=22, rope_base=BASE
        )
        assert np.allclose(warped.inv_freq[:23], native.inv_freq[:23], rtol=1e-12)


def test_encode_band_round_trips_operators_that_fit_the_band():
    native = native_table()
    cases = [
        (0, tables.position_interpolation(native, 4.0)),
        (0, tables.official_yarn(native, scale=4.0, head_dim=2 * PAIRS, rope_base=BASE,
                                 original_max_position_embeddings=NATIVE_LENGTH)),
        (22, nullband.turn_budget_table(native, scale=4.0, beta=1.0,
                                        native_length=NATIVE_LENGTH, rope_base=BASE)),
        (22, nullband.band_warp_table(native, warp="power", param=1.7, scale=2.0,
                                      band_start=22, rope_base=BASE)),
    ]
    for band_start, table in cases:
        z, delta, start_delta = nullband.encode_band(
            native, table, band_start=band_start, rope_base=BASE)
        rebuilt = nullband.free_band_delta_table(
            native, z, delta, band_start=band_start, rope_base=BASE,
            start_delta=start_delta, name="rebuilt"
        )
        assert np.allclose(rebuilt.inv_freq, table.inv_freq, rtol=1e-10), table.name


def test_encode_band_refuses_a_table_that_moved_the_resolvable_block():
    """A silent mis-encode would seed the search with the wrong operator."""
    native = native_table()
    pi = tables.position_interpolation(native, 4.0)
    with pytest.raises(ValueError, match="below band_start"):
        nullband.encode_band(native, pi, band_start=22, rope_base=BASE)


def test_free_band_never_violates_the_supplied_floor():
    native = native_table()
    floor = nullband.phi_floor(
        native.inv_freq, native_length=NATIVE_LENGTH,
        target_length=4 * NATIVE_LENGTH, rope_base=BASE,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.normal(0.0, 3.0, size=PAIRS - 22 - 1)
        table = nullband.free_band_table(
            native, z, scale=4.0, band_start=22, rope_base=BASE, floor=floor
        )
        risk = nullband.phase_excess_risk(
            native.inv_freq, table.inv_freq,
            native_length=NATIVE_LENGTH, target_length=4 * NATIVE_LENGTH,
        )
        assert risk["mean_turns"] == pytest.approx(0.0, abs=1e-12)
