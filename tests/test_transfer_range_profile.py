import math

import pytest

from experiments.olmo_recovery_20260912.transfer_range_profile import transfer_profile


def source():
    return {
        "allocation": {
            "exponents": [0.0, 0.5, 1.0],
            "scale": 4.0,
            "gain": 1.12,
            "low": 0,
            "high": 2,
        }
    }


def test_transfers_normalized_exponents_not_absolute_frequencies():
    table = transfer_profile(source(), [1.0, 0.1, 0.01], target_scale=8.0)
    assert table["values_float32"] == pytest.approx([1.0, 0.1 / math.sqrt(8.0), 0.01 / 8.0])
    assert table["gain"] == pytest.approx(1.0 + 0.1 * math.log(8.0))
    assert table["construction"]["same_table_all_layers_and_lengths"] is True


def test_relative_gain_preserves_source_deviation_from_yarn():
    table = transfer_profile(source(), [1.0, 0.1, 0.01], target_scale=8.0, gain_policy="relative_solver")
    expected = (1.0 + 0.1 * math.log(8.0)) * 1.12 / (1.0 + 0.1 * math.log(4.0))
    assert table["gain"] == pytest.approx(expected)


def test_remaps_shape_into_target_native_transition_band():
    value = {
        "allocation": {
            "exponents": [0.0, 0.0, 0.25, 1.0, 1.0, 1.0],
            "scale": 4.0,
            "gain": 1.12,
            "low": 1,
            "high": 3,
        }
    }
    table = transfer_profile(
        value,
        [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        target_scale=8.0,
        target_low=2,
        target_high=5,
    )
    exponent = table["construction"]["exponents"]
    assert exponent[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert exponent[3:] == pytest.approx([1 / 6, 0.5, 1.0])
    assert table["construction"]["band_policy"] == "normalized_transition_band_remap"


def test_rejects_nonmonotone_allocation():
    bad = source()
    bad["allocation"]["exponents"] = [0.0, 0.8, 0.7]
    with pytest.raises(ValueError, match="invalid source"):
        transfer_profile(bad, [1.0, 0.1, 0.01], target_scale=8.0)


def test_rejects_half_specified_target_band():
    with pytest.raises(ValueError, match="supplied together"):
        transfer_profile(source(), [1.0, 0.1, 0.01], target_scale=8.0, target_low=0)


def test_default_tail_depth_is_numerically_unchanged():
    implicit = transfer_profile(source(), [1.0, 0.1, 0.01], target_scale=8.0)
    explicit = transfer_profile(source(), [1.0, 0.1, 0.01], target_scale=8.0, tail_depth=1.0)
    assert implicit["values_float32"] == explicit["values_float32"]
    assert implicit["construction"]["exponents"] == explicit["construction"]["exponents"]
    assert implicit["construction"]["tail_depth"] == 1.0


def test_tail_depth_scales_full_remapped_profile_and_frequency_formula():
    value = {
        "allocation": {
            "exponents": [0.0, 0.0, 0.25, 1.0, 1.0, 1.0],
            "scale": 4.0,
            "gain": 1.12,
            "low": 1,
            "high": 3,
        }
    }
    native = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    table = transfer_profile(
        value, native, target_scale=8.0, target_low=2, target_high=5, tail_depth=0.5,
    )
    construction = table["construction"]
    full = [0.0, 0.0, 0.0, 1 / 6, 0.5, 1.0]
    final = [exponent * 0.5 for exponent in full]
    assert construction["full_depth_exponents"] == pytest.approx(full)
    assert construction["exponents"] == pytest.approx(final)
    assert construction["tail_depth"] == 0.5
    assert construction["final_exponent_sum"] == pytest.approx(sum(final))
    assert construction["final_exponent_max"] == pytest.approx(max(final))
    assert table["values_float32"] == pytest.approx([
        frequency * 8.0 ** (-exponent) for frequency, exponent in zip(native, final)
    ])


@pytest.mark.parametrize("tail_depth", [0.0, -0.1, 1.01, math.inf, math.nan])
def test_rejects_invalid_tail_depth(tail_depth):
    with pytest.raises(ValueError, match="tail_depth"):
        transfer_profile(
            source(), [1.0, 0.1, 0.01], target_scale=8.0, tail_depth=tail_depth,
        )
