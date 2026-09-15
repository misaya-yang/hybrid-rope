"""CPU coverage of public-input NCP across geometries and active constraints."""

import copy
import subprocess
import sys

import numpy as np
import pytest

from experiments.native_contrastive_proximal_20260915.tables import build_ncp_arrays
from experiments.native_enhancement_oral_20260915.multimodel import (
    build_matrix,
    build_public_ncp,
    config_native,
)


@pytest.mark.parametrize("pairs,base,length", [
    (16, 10_000, 2048), (32, 10_000, 32768), (64, 500_000, 4096),
    (64, 500_000, 8192), (128, 1_000_000, 131072),
])
def test_public_geometries_preserve_constraints_and_reference_descent(pairs, base, length):
    native = (base ** (-np.arange(pairs) / pairs)).astype(np.float32)
    table = build_public_ncp(native, native_length=length, model_id="synthetic", input_label="synthetic_test_geometry")
    audit = table["cpu_audit"]
    assert table["gain"] == 1.0
    assert audit["endpoints_bit_exact"]
    assert audit["minimum_gap_slack_precast"] >= -1e-9
    assert audit["minimum_gap_slack_float32"] >= -2e-7
    assert audit["reference_descent_inequality_margin_precast"] >= -2e-13
    assert audit["maximum_kkt_stationarity_residual"] < 2e-7
    assert audit["model_execution"] is False


def test_general_independent_solver_preserves_existing_frozen_builder_output():
    native = (500_000.0 ** (-np.arange(64) / 64)).astype(np.float32)
    existing = build_ncp_arrays(native, native_length=4096)
    general = build_public_ncp(native, native_length=4096, model_id="synthetic")
    assert np.array_equal(general["values_float32"], existing["candidate"])
    assert not general["cpu_audit"]["coupled_gap_solver_used"]


def test_coupled_solution_matches_analytically_active_gap_chain():
    # Every free gradient is negative over this narrow feasible polytope, so
    # the optimum is its componentwise upper envelope from the slow endpoint.
    native = (np.asarray([5, 4.9, 4.8, 4.7]) / 4095).astype(np.float32)
    table = build_public_ncp(native, native_length=4096, model_id="active_gap")
    audit = table["cpu_audit"]
    gaps = np.diff(-np.log(native.astype(float)))
    expected = np.asarray([0, 0.5 * gaps[1:].sum(), 0.5 * gaps[-1], 0])
    np.testing.assert_allclose(table["construction"]["log_shifts_precast"], expected, atol=2e-9)
    assert audit["coupled_gap_solver_used"]
    assert audit["independent_minimum_gap_slack"] < 0
    assert audit["maximum_kkt_stationarity_residual"] < 1e-8


def test_two_fixed_endpoints_are_identity_and_need_no_interior_solution():
    native = np.asarray([1.0, 0.001], dtype=np.float32)
    table = build_public_ncp(native, native_length=4096, model_id="two_pairs")
    assert np.array_equal(native, table["values_float32"])
    assert table["cpu_audit"]["changed_pair_count"] == 0


def default_config():
    return {"model_type": "llama", "hidden_size": 4096, "num_attention_heads": 32,
            "rope_theta": 500_000, "rope_scaling": None, "max_position_embeddings": 131072}


def test_config_adapter_requires_explicit_length_and_labels_rounding_identity():
    config = default_config()
    before = copy.deepcopy(config)
    case = config_native(config, native_length=8192)
    assert case["native_length"] == 8192
    assert case["config_max_position_embeddings_informational_only"] == 131072
    assert case["input_label"] == "config_derived_fp32_not_runtime_verified"
    assert len(case["native_values"]) == 64
    assert config == before
    with pytest.raises(TypeError):
        config_native(config)


@pytest.mark.parametrize("change", [
    {"rope_scaling": {"rope_type": "llama3", "factor": 8}},
    {"rope_parameters": {"rope_type": "yarn", "factor": 4}},
    {"rope_scaling": {"rope_type": "default", "type": "linear"}},
    {"rope_type": "linear"}, {"rotary_dim": 64},
    {"partial_rotary_factor": 0.5}, {"rotary_pct": 0.25},
    {"model_type": "unverified"}, {"head_dim": 127},
    {"rope_theta": None}, {"rope_theta": float("nan")},
])
def test_unsupported_config_requires_explicit_native_table(change):
    config = default_config()
    config.update(change)
    with pytest.raises(ValueError):
        config_native(config, native_length=8192)


@pytest.mark.parametrize("values,length", [
    ([1.0], 4096), ([1.0, 1.0], 4096), ([1.0, -1.0], 4096),
    ([1.0, float("nan")], 4096), ([1.0, 0.01], 1), ([1.0, 0.01], 8192.5),
])
def test_invalid_public_inputs_are_rejected(values, length):
    with pytest.raises(ValueError):
        build_public_ncp(values, native_length=length, model_id="invalid")


def test_matrix_preserves_evidence_labels_and_rejects_duplicate_identity():
    case = {"model_id": "synthetic", "native_values": [1, 0.1, 0.01, 0.001],
            "native_length": 4096, "input_label": "synthetic_test_geometry"}
    matrix = build_matrix([case])
    assert matrix["rows"][0]["input_label"] == "synthetic_test_geometry"
    assert not matrix["parameters_selected_using_model_or_task_outputs"]
    assert "not cross-model performance" in matrix["claim_boundary"]
    with pytest.raises(ValueError, match="unique"):
        build_matrix([case, case])


def test_general_module_does_not_import_torch_or_cuda():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import experiments.native_enhancement_oral_20260915.multimodel; "
        "assert 'torch' not in sys.modules; assert 'cupy' not in sys.modules",
    ], check=True)
