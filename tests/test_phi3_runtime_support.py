from types import SimpleNamespace

import pytest

from experiments.fixed_rope_three_interfaces_20260913.tables import build_analytic
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config


def phi3_config():
    return SimpleNamespace(
        model_type="phi3",
        hidden_size=3072,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        partial_rotary_factor=None,
    )


def test_phi3_4k_native_runtime_geometry():
    table = table_for_config(phi3_config(), "Native")
    assert len(table["values_float32"]) == 48
    assert table["gain"] == 1.0


def test_phi3_requires_explicit_non_native_table():
    with pytest.raises(ValueError, match="explicit static table"):
        table_for_config(phi3_config(), "MrPro_g4")


def test_phi3_tailspline_s32_is_valid_static_geometry():
    config = vars(phi3_config())
    values, gain, construction = build_analytic(
        config, method="tailspline", scale=32.0,
        low=None, high=None, depth=1.0, gain=None,
    )
    assert values.shape == (48,)
    assert gain > 1.0
    assert construction["method"] == "tailspline_exact_finite_grid"
