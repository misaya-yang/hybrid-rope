"""Independent matrix checks of finite full-pair theory and its limitations."""

import subprocess
import sys

import numpy as np
import pytest

from experiments.native_enhancement_oral_20260915.theory import (
    audit_theory,
    opposite_content_counterexample,
    pair_coefficients,
    rotation,
    signed_margin_change,
)


def test_pair_response_matches_explicit_rotation_including_sine_sign():
    rng = np.random.default_rng(47)
    q, k = rng.normal(size=(2, 12, 7, 2))
    theta = rng.normal(size=(12, 7))
    c, d, amplitude = pair_coefficients(q, k)
    direct = np.einsum("...i,...ij,...j->...", q, rotation(theta), k)
    np.testing.assert_allclose(direct, c * np.cos(theta) + d * np.sin(theta), atol=2e-15)
    np.testing.assert_allclose(amplitude, np.linalg.norm(q, axis=-1) * np.linalg.norm(k, axis=-1))


def test_margin_and_certificate_match_matrix_bilinear_forms_at_distinct_distances():
    rng = np.random.default_rng(53)
    q, positive, negative = rng.normal(size=(3, 128, 5, 2))
    dp, dn = rng.uniform(-20, 20, size=(2, 128))
    native = np.geomspace(1, 0.01, 5)
    candidate = native * np.array([1, 0.9, 0.99, 0.85, 1])
    result = signed_margin_change(q, positive, negative, dp, dn, native, candidate, scale=0.125)

    def direct(frequencies):
        pos = np.einsum("...ki,...kij,...kj->...k", q, rotation(dp[:, None] * frequencies), positive)
        neg = np.einsum("...ki,...kij,...kj->...k", q, rotation(dn[:, None] * frequencies), negative)
        return 0.125 * np.sum(pos - neg, axis=-1)

    np.testing.assert_allclose(result["native_margin"], direct(native), atol=2e-15)
    np.testing.assert_allclose(result["candidate_margin"], direct(candidate), atol=2e-15)
    np.testing.assert_allclose(result["exact_change"], direct(candidate) - direct(native), atol=2e-15)
    assert np.all(result["exact_change"] >= result["lower"] - 1e-14)
    assert np.all(result["exact_change"] <= result["upper"] + 1e-14)


def test_noop_has_exactly_zero_movement_bound_and_per_pair_change():
    q = np.array([[1.0, 2.0], [-1.0, 3.0]])
    result = signed_margin_change(q, q, -q, 17, 3, [1, 0.1], [1, 0.1])
    for key in ("exact_change", "linear_change", "remainder_bound", "pair_exact_change"):
        assert np.all(result[key] == 0)


def test_phase_taylor_remainder_scales_quadratically_for_known_content():
    q = np.array([[1.0, 0.0]])
    coarse = signed_margin_change(q, q, -q, 1.0, 1.0, [1.0], [0.999])
    fine = signed_margin_change(q, q, -q, 1.0, 1.0, [1.0], [0.9995])
    remainder = lambda r: abs(float(r["exact_change"] - r["linear_change"]))
    assert remainder(coarse) / remainder(fine) == pytest.approx(4.0, rel=0.001)
    assert float(coarse["remainder_bound"] / fine["remainder_bound"]) == pytest.approx(4)


def test_fixed_support_counterexample_has_positive_native_margins_and_opposite_effects():
    result = opposite_content_counterexample()
    assert result["native"][0] == result["candidate"][0]
    assert result["native"][-1] == result["candidate"][-1]
    assert np.all(np.diff(result["candidate"]) < 0)
    good, bad = result["examples"].values()
    assert good["native_margin"] > 0 and bad["native_margin"] > 0
    assert good["exact_change"] > 0 > bad["exact_change"]
    assert bad["exact_change"] == pytest.approx(2 * (np.cos(0.5) - 1))


def test_fullpair_operator_identity_includes_large_phase_changes():
    phase = np.array([0.0, 0.1, 3.0, 10.0])
    delta = np.array([0.0, -0.7, np.pi, 4 * np.pi])
    norm = np.linalg.svd(rotation(phase + delta) - rotation(phase), compute_uv=False)[:, 0]
    np.testing.assert_allclose(norm, 2 * np.abs(np.sin(delta / 2)), atol=2e-15)


def test_phase_reflection_matches_norm_and_remainder_but_reverses_signed_certificate():
    q = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    native = np.array([1.0, 0.5, 0.1])
    candidate = np.array([1.0, 0.45, 0.1])
    reflected = 2 * native - candidate
    forward = signed_margin_change(q, q, -q, 2, 2, native, candidate)
    reverse = signed_margin_change(q, q, -q, 2, 2, native, reflected)
    assert float(forward["linear_change"]) == pytest.approx(-float(reverse["linear_change"]))
    assert float(forward["remainder_bound"]) == pytest.approx(float(reverse["remainder_bound"]))
    assert float(forward["lower"]) > 0 > float(reverse["upper"])
    assert float(forward["exact_change"]) > 0 > float(reverse["exact_change"])
    norm = lambda table: np.linalg.svd(rotation(2 * table) - rotation(2 * native), compute_uv=False)[:, 0]
    np.testing.assert_allclose(norm(candidate), norm(reflected), atol=2e-15)


def test_distance_and_frequency_rescaling_preserves_every_margin_quantity():
    q = np.array([[0.5, 0.2], [0.2, 0.7]])
    native, candidate = np.array([1.0, 0.1]), np.array([0.9, 0.1])
    original = signed_margin_change(q, q, -q, 5, 8, native, candidate)
    scaled = signed_margin_change(q, q, -q, 50, 80, native / 10, candidate / 10)
    for field in original:
        np.testing.assert_allclose(original[field], scaled[field], atol=2e-15)


def test_cpu_audit_reproduces_symmetry_asymptotics_and_bound_without_claiming_model_gain():
    result = audit_theory()
    assert result["reflection_identity_max_error"] < 1e-14
    assert result["first_harmonic"] == pytest.approx(-0.5, abs=1e-15)
    assert result["higher_odd_harmonics_max_abs"] < 1e-14
    slow = result["slow_phase_examples"][-1]
    assert slow["u_over_phase_squared"] == pytest.approx(result["slow_phase_coefficient"], rel=1e-6)
    fast = result["high_phase_examples"]
    assert fast[-1]["u"] < fast[0]["u"] / 90
    assert fast[-1]["full_window_phase_change"] > 0.4
    assert result["margin_audit"]["bound_violations"] == 0
    assert result["margin_audit"]["false_positive_certificates"] == 0
    assert result["mathematical_fp64_olmo_grid"]["maximum_relative_slowdown"] == pytest.approx(0.1671638685, abs=1e-9)
    assert result["model_execution"] is False
    assert "not Transformer NLL" in result["claim_boundary"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        rotation(bad)


def test_shape_and_scale_mismatches_are_rejected():
    q = np.ones((3, 2))
    with pytest.raises(ValueError, match="pair count"):
        signed_margin_change(q, q, q, 1, 1, [1, 0.1], [1, 0.1])
    with pytest.raises(ValueError, match="scale"):
        signed_margin_change(q, q, q, 1, 1, [1, 0.5, 0.1], [1, 0.5, 0.1], scale=-1)


def test_import_does_not_load_torch_or_gpu_runtime():
    subprocess.run([
        sys.executable, "-c", "import sys; "
        "import experiments.native_enhancement_oral_20260915.theory; "
        "assert 'torch' not in sys.modules; assert 'cupy' not in sys.modules",
    ], check=True)
