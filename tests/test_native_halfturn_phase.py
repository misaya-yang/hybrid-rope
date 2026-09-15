import math

import numpy as np
import pytest

from experiments.native_halfturn_phase_20260915.tables import (
    build_audit,
    build_halfturn_arrays,
    compare_v1,
    tensor_sha256,
)


def native_table(pairs=64, base=500_000.0):
    return np.power(base, -np.arange(pairs, dtype=np.float64) / pairs).astype(np.float32)


def test_halfturn_preserves_support_order_and_equal_phase_dose():
    native = native_table()
    result = build_halfturn_arrays(native, native_length=4096)
    active = result["active_indices"]
    assert (int(active[0]), int(active[-1]), len(active)) == (35, 62, 28)
    assert np.array_equal(result["contract"][[0, -1]], native[[0, -1]])
    assert np.array_equal(result["reverse"][[0, -1]], native[[0, -1]])
    assert np.all(result["contract"][active] < native[active])
    assert np.all(result["reverse"][active] > native[active])
    assert np.all(result["contract"][:-1] > result["contract"][1:])
    assert np.all(result["reverse"][:-1] > result["reverse"][1:])
    assert result["fp32_phase_dose_max_abs_error"] < 2e-6
    assert all(result["checks"].values())


def test_halfturn_returns_native_when_slowest_phase_reaches_cap():
    native = np.asarray([1.0, 0.8, 0.6, 0.5], dtype=np.float32)
    result = build_halfturn_arrays(native, native_length=16, phase_cap=1.0)
    assert result["active_indices"].size == 0
    assert np.array_equal(result["contract"], native)
    assert np.array_equal(result["reverse"], native)


def test_continuous_phase_bound_and_short_distance_bound():
    result = build_halfturn_arrays(native_table(), native_length=4096)
    maximum = float(result["displacement_precast"].max())
    expected = 0.25 * (math.pi - result["lower_phase"]) / 2.0
    assert maximum <= expected
    assert maximum * 256 / 4095 < 0.023


def test_cpu_audit_does_not_upgrade_math_to_model_evidence():
    result = build_halfturn_arrays(native_table(), native_length=4096)
    audit = build_audit(
        result,
        model_id="olmo2_1b",
        eta=0.25,
        phase_cap=math.pi,
        config_sha256="a" * 64,
    )
    assert audit["model_execution"] is False
    assert audit["public_inputs_only"] is True
    assert "does not predict" in audit["claim_boundary"]
    assert max(audit["rotation_operator_distance_equality_errors"].values()) < 2e-6


def test_v1_comparison_rejects_another_native_identity():
    native = native_table()
    result = build_halfturn_arrays(native, native_length=4096)
    payload = {
        "values_float32": (native * np.linspace(1.0, 0.99, len(native))).tolist(),
        "gain": 1.0,
        "native_table_sha256_float32": "wrong",
    }
    with pytest.raises(ValueError, match="another Native table"):
        compare_v1(payload, result)


def test_v1_comparison_is_descriptive_not_equal_dose():
    native = native_table()
    result = build_halfturn_arrays(native, native_length=4096)
    v1 = native.copy()
    v1[10:-1] *= np.linspace(0.999, 0.95, len(v1) - 11).astype(np.float32)
    assert np.all(v1[:-1] > v1[1:])
    payload = {
        "candidate_id": "historical-v1",
        "values_float32": v1.tolist(),
        "gain": 1.0,
        "native_table_sha256_float32": tensor_sha256(native),
    }
    comparison = compare_v1(payload, result)
    assert comparison["changed_pairs"] > 0
    assert comparison["contract_to_v1_l2_phase_ratio"] > 0.0
    assert "not an equal-dose control" in comparison["interpretation"]


@pytest.mark.parametrize("eta", [-1.0, 0.0, 1.0, float("nan")])
def test_invalid_eta_is_rejected(eta):
    with pytest.raises(ValueError, match="eta"):
        build_halfturn_arrays(native_table(), native_length=4096, eta=eta)
