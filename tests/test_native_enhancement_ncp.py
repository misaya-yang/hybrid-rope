"""Independent CPU reference oracle and equal-dose attribution-control tests."""

import math
import subprocess
import sys

import numpy as np
import pytest

from experiments.native_contrastive_proximal_20260915.tables import build_ncp_arrays
from experiments.native_enhancement_oral_20260915.ncp import (
    MAX_LOG_SHIFT,
    audit_reference,
    build_dose_control,
    direct_reference,
)


def synthetic_native():
    # CPU-only synthetic geometry, not a replacement for runtime Torch FP32.
    return (500_000.0 ** (-np.arange(64) / 64)).astype(np.float32)


def test_direct_original_loss_agrees_with_fourier_risk_and_derivatives():
    audit = audit_reference()
    assert max(audit["maximum_absolute_error"].values()) < 1e-10
    assert audit["model_execution"] is False
    assert "not Transformer NLL" in audit["claim_boundary"]


def test_direct_risk_zero_phase_and_deep_slow_asymptotic():
    zero = direct_reference(0.0)
    first, second = direct_reference(0.002), direct_reference(0.004)
    assert np.array_equal(zero[1:], np.zeros(2))
    assert (second[0] - zero[0]) / (first[0] - zero[0]) == pytest.approx(4, rel=2e-6)
    assert second[1] / first[1] == pytest.approx(4, rel=2e-6)


def test_control_matches_installed_source_log_dose_and_preserves_contract():
    native = synthetic_native()
    source = build_ncp_arrays(native, native_length=4096)["candidate"]
    receipt = build_dose_control(native, source, native_length=4096, model_id="synthetic")
    control = np.asarray(receipt["values_float32"], dtype=np.float32)
    construction = receipt["construction"]
    assert receipt["gain"] == 1.0 and receipt["role"] == "control"
    assert np.array_equal(control[[0, -1]], native[[0, -1]])
    assert np.all(control <= native)
    assert np.all(control[:-1] > control[1:])
    assert construction["control_log_dose_precast"] == pytest.approx(construction["source_log_dose"], abs=1e-12)
    assert abs(construction["float32_log_dose_error"]) < 3e-6
    assert construction["maximum_kkt_stationarity_residual"] < 1e-10
    assert construction["minimum_log_gap_slack_precast"] >= 0
    assert construction["solver"] == "equality_closed_form"
    assert "does not match phase displacement" in receipt["claim_boundary"]


def test_equal_gap_unconstrained_shape_is_discrete_parabola():
    native = np.exp(-np.arange(8, dtype=float)).astype(np.float32)
    source = native.copy()
    source[3:5] *= np.float32(math.exp(-0.08))
    control = build_dose_control(native, source, native_length=4096)
    actual_u = np.log(native / np.asarray(control["values_float32"], dtype=np.float32))
    expected = np.arange(8) * (7 - np.arange(8))
    expected = expected * control["construction"]["source_log_dose"] / expected.sum()
    np.testing.assert_allclose(actual_u, expected, atol=9e-8, rtol=0)


def test_constraint_active_control_has_convex_optimality_certificate():
    native = np.exp(-np.arange(12, dtype=float)).astype(np.float32)
    source = native.copy()
    source[1:-1] *= np.float32(math.exp(-0.205))
    receipt = build_dose_control(native, source, native_length=4)
    c = receipt["construction"]
    assert c["solver"] == "constrained_quadratic_slsqp"
    actual_u = np.log(native.astype(float) / receipt["values_float32"])
    assert np.max(actual_u) <= MAX_LOG_SHIFT + 2e-7
    assert c["maximum_kkt_stationarity_residual"] < 2e-6
    assert abs(c["float32_log_dose_error"]) < 2e-6


def test_half_gap_active_control_is_solved_instead_of_ignoring_coupling():
    native = np.exp(-0.01 * np.arange(6, dtype=float)).astype(np.float32)
    source_u = np.asarray([0.0, 0.016, 0.012, 0.008, 0.004, 0.0])
    source = (native.astype(float) * np.exp(-source_u)).astype(np.float32)
    receipt = build_dose_control(native, source, native_length=4)
    c = receipt["construction"]
    assert c["solver"] == "constrained_quadratic_slsqp"
    assert abs(c["minimum_log_gap_slack_precast"]) < 1e-9
    assert c["minimum_log_gap_slack_float32"] >= -2e-7
    assert c["maximum_kkt_stationarity_residual"] < 2e-6


def test_zero_dose_control_is_identity():
    native = synthetic_native()
    result = build_dose_control(native, native, native_length=4096)
    assert np.array_equal(result["values_float32"], native)
    assert result["construction"]["source_log_dose"] == 0


def test_invalid_source_identity_is_rejected():
    native = synthetic_native()
    other = native.copy()
    other[-1] *= 0.99
    with pytest.raises(ValueError, match="endpoints"):
        build_dose_control(native, other, native_length=4096)


def test_module_import_does_not_load_torch_or_cuda():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import experiments.native_enhancement_oral_20260915.ncp; "
        "assert 'torch' not in sys.modules; assert 'cupy' not in sys.modules",
    ], check=True)
