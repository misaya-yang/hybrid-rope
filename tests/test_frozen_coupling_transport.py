import argparse
import json

import numpy as np
import pytest

from scripts.analysis.export_frozen_coupling_transport import (
    DEFAULT_GAIN_COEFFICIENT,
    DEFAULT_X_HIGH,
    DEFAULT_X_LOW,
    build_tables,
    config_identity,
    export,
    sha256_bytes,
    sha256_file,
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


@pytest.fixture
def confirmed_args(tmp_path):
    """Synthetic receipt/tensor fixture only; not a real calibration result."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps(gemma_config()) + "\n")
    native = np.ascontiguousarray(10000.0 ** (-np.arange(128) / 128), dtype="<f4")
    native_path = tmp_path / "native.npy"
    np.save(native_path, native)
    reference = tmp_path / "confirmed.json"
    reference.write_text(json.dumps({
        "status": "NATIVE_REFERENCE_CONFIRMED", "reference_length": 4096,
        "checkpoint_weight_sha256": "a" * 64,
        "config_sha256": sha256_file(config),
        "native_sha256_float32": sha256_bytes(native.tobytes()),
        "confirmation_decision_sha256": "b" * 64, "data_manifest_sha256": "c" * 64,
    }))
    return argparse.Namespace(
        config=config, native_length=None, native_inv=native_path,
        scale=2.0, x_high=DEFAULT_X_HIGH, x_low=DEFAULT_X_LOW,
        gain_coefficient=DEFAULT_GAIN_COEFFICIENT, include_wrong_c_orth=True,
        reference_receipt=reference, target_length=8192, output=tmp_path / "export",
    )


def test_confirmed_reference_preserves_config_and_binds_provenance(confirmed_args):
    args = confirmed_args
    original_config = args.config.read_bytes()
    receipt = export(args)
    identity = receipt["checkpoint"]
    assert args.config.read_bytes() == original_config
    assert identity["native_length"] == 8192
    assert identity["reference_length"] == 4096
    assert identity["reference_calibration"]["receipt_sha256"] == sha256_file(args.reference_receipt)
    assert identity["reference_calibration"]["checkpoint_weight_sha256"] == "a" * 64
    assert identity["reference_calibration"]["confirmation_decision_sha256"] == "b" * 64
    assert identity["reference_calibration"]["data_manifest_sha256"] == "c" * 64
    assert receipt["reference_scope"]["L_config"] == 8192
    assert receipt["reference_scope"]["L_ref"] == 4096
    assert receipt["reference_scope"]["target_length"] == 8192
    assert receipt["reference_scope"]["s"] == 2.0
    assert receipt["reference_scope"]["table_parameters_refit"] is False
    assert receipt["source_grid_for_normalized_index"]["native_length"] == 8192
    assert receipt["source_grid_for_normalized_index"]["coordinate_length"] == 4096


@pytest.mark.parametrize("runtime_native", [False, True])
def test_reference_coordinates_apply_to_physical_index_and_wrong_control(confirmed_args, runtime_native):
    args = confirmed_args
    identity = config_identity(args.config)
    reference_identity = {**identity, "reference_length": 4096}
    # Independent oracle: same analytical grid at L=4096; never write config.
    oracle_identity = {**identity, "native_length": 4096}
    native = np.load(args.native_inv) if runtime_native else None
    kwargs = dict(scale=2.0, x_high=DEFAULT_X_HIGH, x_low=DEFAULT_X_LOW,
                  include_wrong_c_orth=True, native_override=native)
    actual, movements, c_orth = build_tables(reference_identity, **kwargs)
    expected, expected_movement, expected_c = build_tables(oracle_identity, **kwargs)
    _, default_movement, _ = build_tables(identity, **kwargs)
    assert c_orth == expected_c
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(movements[name], expected_movement[name])
        assert not np.array_equal(movements[name], default_movement[name])
    assert identity["native_length"] == reference_identity["native_length"] == 8192


def test_no_reference_retains_original_semantics(confirmed_args):
    args = confirmed_args
    args.reference_receipt = None
    args.target_length = None
    receipt = export(args)
    assert receipt["checkpoint"]["native_length"] == 8192
    assert "reference_length" not in receipt["checkpoint"]
    assert "reference_scope" not in receipt
    assert "coordinate_length" not in receipt["source_grid_for_normalized_index"]
    assert receipt["source_grid_for_normalized_index"]["native_length"] == 8192


@pytest.mark.parametrize("field,value,match", [
    ("status", "NATIVE_REFERENCE_PROVISIONAL", "NATIVE_REFERENCE_CONFIRMED"),
    ("config_sha256", "d" * 64, "config_sha256 mismatch"),
    ("native_sha256_float32", "d" * 64, "native_sha256_float32 mismatch"),
    ("reference_length", 0, "positive power of two"),
    ("reference_length", 3072, "positive power of two"),
    ("reference_length", 16384, "positive power of two"),
    ("reference_length", 4096.0, "positive power of two"),
    ("reference_length", True, "positive power of two"),
    ("checkpoint_weight_sha256", "missing", "checkpoint_weight_sha256"),
    ("confirmation_decision_sha256", None, "confirmation_decision_sha256"),
    ("data_manifest_sha256", "", "data_manifest_sha256"),
])
def test_invalid_reference_fails_before_export(confirmed_args, field, value, match):
    args = confirmed_args
    receipt = json.loads(args.reference_receipt.read_text())
    receipt[field] = value
    args.reference_receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        export(args)
    assert not args.output.exists()


@pytest.mark.parametrize("field,value,match", [
    ("native_inv", None, "requires runtime --native-inv"),
    ("target_length", None, "requires integer target_length"),
    ("target_length", 4096, "requires integer target_length"),
    ("target_length", 16384, "scale must equal"),
    ("scale", 4.0, "scale must equal"),
    ("x_high", DEFAULT_X_HIGH + 0.01, "locks x_high"),
    ("x_low", DEFAULT_X_LOW + 0.01, "locks x_high"),
    ("gain_coefficient", 0.1, "locks x_high"),
    ("reference_receipt", None, "target_length requires"),
])
def test_reference_requires_fixed_ratio_native_and_frozen_parameters(confirmed_args, field, value, match):
    args = confirmed_args
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        export(args)
    assert not args.output.exists()
