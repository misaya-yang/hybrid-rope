import argparse
import json

import numpy as np
import pytest

from scripts.analysis.export_frozen_coupling_transport import (
    build_tables,
    config_identity,
    export,
)


def gemma_config() -> dict:
    return {
        "model_type": "gemma",
        "head_dim": 256,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "max_position_embeddings": 8192,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
    }


def test_k128_export_is_ordered_and_zero_refit(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(gemma_config()) + "\n")
    identity = config_identity(config)
    tables, movements, c_orth = build_tables(
        identity,
        scale=2.0,
        x_high=0.7382780681078285,
        x_low=0.366403835112904,
        include_wrong_c_orth=True,
        native_override=None,
    )
    assert identity["pairs"] == 128
    assert c_orth["target"] > c_orth["wrong_source"]
    assert set(tables) == {
        "dimensionless_x",
        "normalized_raw_index",
        "wrong_source_c_orth",
    }
    assert np.flatnonzero(
        (movements["dimensionless_x"] > 1e-10)
        & (movements["dimensionless_x"] < 1 - 1e-10)
    ).tolist() == [53, 54, 55, 56, 57]
    for name, table in tables.items():
        assert table.dtype == np.dtype("float32")
        assert table.shape == (128,)
        assert np.all(table[:-1] > table[1:]), name
        assert movements[name].shape == (128,)


def test_export_manifest_and_reject_scaled_checkpoint(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(gemma_config()) + "\n")
    output = tmp_path / "out"
    receipt = export(argparse.Namespace(
        config=config,
        native_length=None,
        scale=2.0,
        x_high=0.7382780681078285,
        x_low=0.366403835112904,
        gain_coefficient=0.074,
        include_wrong_c_orth=False,
        native_inv=None,
        output=output,
    ))
    assert receipt["benchmark_scores_used"] is False
    assert receipt["checkpoint"]["pairs"] == 128
    assert len(receipt["checkpoint"]["native_sha256_float32"]) == 64
    assert (output / "manifest.json").is_file()
    assert set(receipt["tables"]) == {"dimensionless_x", "normalized_raw_index"}

    scaled = gemma_config()
    scaled["rope_scaling"] = {"rope_type": "linear", "factor": 2.0}
    config.write_text(json.dumps(scaled) + "\n")
    with pytest.raises(ValueError, match="already has RoPE scaling"):
        config_identity(config)
