from types import SimpleNamespace

import numpy as np

from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config
from experiments.olmo_recovery_20260912.score_fixed_table_interval import log_auc


def llama_config():
    return SimpleNamespace(
        model_type="llama",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        max_position_embeddings=8192,
        rope_theta=500000.0,
    )


def test_llama_gamma3_g8_is_one_valid_fixed_target_table():
    table = table_for_config(llama_config(), "BetaSym_gamma3_g8")
    values = np.asarray(table["values_float32"], dtype=np.float32)
    construction = table["construction"]
    assert values.shape == (64,)
    assert np.isfinite(values).all() and np.all(values[:-1] > values[1:])
    assert construction["scale"] == 8.0
    assert construction["reference_length"] == 8192
    assert (construction["low"], construction["high"], construction["N"]) == (18, 35, 17)
    assert construction["gamma"] == 3.0


def test_llama_gamma3_g8_is_distinct_from_both_strong_controls():
    candidate = np.asarray(table_for_config(llama_config(), "BetaSym_gamma3_g8")["values_float32"])
    bm = np.asarray(table_for_config(llama_config(), "BM_g8")["values_float32"])
    mrpro = np.asarray(table_for_config(llama_config(), "MrPro_g8")["values_float32"])
    assert not np.array_equal(candidate, bm)
    assert not np.array_equal(candidate, mrpro)


def test_log_auc_weights_each_context_doubling_equally():
    curve = {8192: 1.0, 16384: 0.5, 32768: 0.5, 65536: 0.0}
    assert log_auc(curve) == 0.5


def test_range_bridge_is_exact_midpoint_in_exponent_space():
    config = llama_config()
    candidate = np.asarray(table_for_config(config, "RangeBridge50_g8")["values_float32"], dtype=np.float64)
    bm = np.asarray(table_for_config(config, "BM_g8")["values_float32"], dtype=np.float64)
    mrpro = np.asarray(table_for_config(config, "MrPro_g8")["values_float32"], dtype=np.float64)
    native = np.power(config.rope_theta, -np.arange(64, dtype=np.float64) / 64)
    denominator = np.log(8.0)
    candidate_m = -np.log(candidate / native) / denominator
    midpoint_m = 0.5 * (-np.log(bm / native) / denominator - np.log(mrpro / native) / denominator)
    np.testing.assert_allclose(candidate_m, midpoint_m, rtol=0.0, atol=2e-7)


def test_range_gain_changes_only_the_fixed_bm_gain():
    config = llama_config()
    endpoint = table_for_config(config, "BM_g8")
    ranged = table_for_config(config, "BM_g8_RangeGain")
    np.testing.assert_array_equal(ranged["values_float32"], endpoint["values_float32"])
    assert ranged["gain"] == 1.0 + 0.05 * np.log(8.0)
    assert endpoint["gain"] == 1.0 + 0.1 * np.log(8.0)
