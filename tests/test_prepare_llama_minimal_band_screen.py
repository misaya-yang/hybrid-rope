import numpy as np
from types import SimpleNamespace

from experiments.olmo_recovery_20260912.prepare_llama_minimal_band_screen import (
    build_tables,
    native_for_config,
)
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config


def test_builds_matched_s2_baselines_and_shifted_bands():
    native = np.power(500_000.0, -np.arange(64, dtype=np.float64) / 64).astype(np.float32)
    tables = build_tables(
        native, native_length=8192, base=500_000.0, scale=2.0,
        bands=[(12, 30), (14, 32), (16, 34), (18, 36)],
    )
    assert set(tables) == {
        "BM_s2", "MrPro_s2", "C42Band12_30_s2", "C42Band14_32_s2",
        "C42Band16_34_s2", "C42Band18_36_s2",
    }
    for table in tables.values():
        values = np.asarray(table["values_float32"])
        assert values.shape == (64,)
        assert np.all(values[:-1] > values[1:])
        assert table["gain"] == 1.0 + 0.1 * np.log(2.0)


def test_reads_qwen_standard_rope_geometry_without_architecture_whitelist():
    config = SimpleNamespace(
        model_type="qwen2",
        hidden_size=1536,
        num_attention_heads=12,
        head_dim=None,
        rope_theta=1_000_000.0,
        max_position_embeddings=32768,
    )
    native, base, head_dim = native_for_config(config)
    assert native.shape == (64,)
    assert base == 1_000_000.0
    assert head_dim == 128
    assert len(table_for_config(config, "Native")["values_float32"]) == 64
