"""CPU-only tests for scripts.analysis.demand_companding_numeric."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import demand_companding_numeric as dcn


def test_high_rate_cube_root_and_d_star_identity():
    x = np.linspace(0.0, 1.0, 1001)
    demand = 0.3 + 1.7 * x**2
    result = dcn.high_rate_density(x, demand)
    positive = result["m"] > 0
    ratios = result["rho"][positive] ** 3 / result["m"][positive]
    assert np.max(ratios) - np.min(ratios) < 1.0e-10
    assert 0.0 < result["H"] < 1.0
    assert result["H_differential"] != result["H"]
    np.testing.assert_allclose(dcn.rho_m13(x, demand), result["rho"], atol=1.0e-14)
    assert dcn.D_star(x, demand) == pytest.approx(result["H_m13"] ** 3, abs=1.0e-14)
    assert dcn.H(x, demand) == pytest.approx(result["H"], abs=1.0e-14)
    expected = result["H_m13"] ** 3 / (12.0 * 24**2)
    np.testing.assert_allclose(dcn.high_rate_distortion(x, demand, 24), expected, rtol=0.0, atol=1.0e-14)


def test_lambda_mixes_with_uniform_without_changing_normalization():
    x = np.linspace(-2.0, 3.0, 1001)
    demand = 0.2 + np.exp(-0.5 * ((x - 0.7) / 0.4) ** 2)
    uniform = dcn.uniform_density(x)
    assert np.testing.assert_allclose(dcn.mix_demand(x, demand, 1.0), uniform, atol=2.0e-12) is None
    mixed = dcn.mix_demand(x, demand, 0.35)
    np.testing.assert_allclose(mixed, 0.65 * demand + 0.35 * uniform, atol=2.0e-12)
    _, normalized_mixed = dcn.normalize_density(x, mixed)
    assert abs(dcn._trapz(normalized_mixed, x) - 1.0) < 2.0e-12


def test_capp_alpha_term_has_quartic_small_tau_coefficient():
    x = np.linspace(0.0, 1.0, 4001)
    alpha = 3.2
    tau = 2.0e-3
    at_tau = dcn.capp_components(x, dcn.cosh_density(x, tau), alpha=alpha, beta=0.0)["value"]
    at_zero = dcn.capp_components(x, np.ones_like(x), alpha=alpha, beta=0.0)["value"]
    observed = (at_tau - at_zero) / tau**4
    np.testing.assert_allclose(observed, dcn.capp_quartic_alpha_coefficient(alpha), rtol=0.0, atol=2.0e-4)


def test_quartic_balance_uses_monotone_root_and_reports_no_finite_case():
    result = dcn.solve_quartic_balance(alpha=0.7, beta=0.4, gain=1.3)
    assert result["status"] == "CONVERGED"
    assert result["derivative_monotone_on_domain"] is True
    assert abs(result["residual"]) < 1.0e-10
    assert dcn.solve_quartic_balance(0.0, 0.0, 1.0)["status"] == "NO_FINITE_ROOT"


def test_pointwise_alpha_quartic_is_monotone_and_normalizes_after_solving():
    x = np.linspace(0.0, 1.0, 301)
    demand = 0.1 + 2.0 * x**2
    fixed = dcn.quartic_density_solution(x, demand, alpha=0.8, nu=1.2, epsilon=0.01)
    assert fixed["status"] == "BLOCKED_DIAGNOSTIC"
    assert fixed["rho"] is None
    assert fixed["reason"] == "fixed_nu_does_not_satisfy_integral_constraint"
    assert fixed["max_abs_residual"] < 1.0e-9
    uniform = dcn.quartic_density_solution(x, np.ones_like(x), alpha=1.0, nu=1.0)
    assert uniform["status"] == "CONVERGED_FIXED_NU_NORMALIZED"
    np.testing.assert_allclose(dcn._trapz(uniform["rho"], x), 1.0, atol=1.0e-12)
    outer = dcn.quartic_density_solution(x, np.ones_like(x), alpha=1.0, nu=None)
    assert outer["status"] == "CONVERGED_OUTER_NU"
    np.testing.assert_allclose(dcn._trapz(outer["rho"], x), 1.0, atol=1.0e-12)


def test_endpoint_quantiles_and_collision_spacing_are_explicit():
    x = np.linspace(0.0, 1.0, 501)
    demand = 0.05 + np.exp(-0.5 * ((x - 0.6) / 0.1) ** 2)
    q = dcn.quantile_from_density(x, demand, 17, endpoint=True)
    assert q[0] == 0.0
    assert q[-1] == 1.0
    assert np.all(np.diff(q) >= 0.0)
    assert dcn.minimum_spacing([0.0, 0.0, 0.4]) == 0.0
    theta = dcn.frequency_from_coordinate(q)
    distances = np.linspace(1.0, 20.0, 200)
    score = dcn.phase_collision_score(theta, distances)
    assert np.isfinite(score)
    assert score >= 0.0
    assert dcn.coordinate_collision_score(q) >= 0.0
    assert dcn.coordinate_collision_score([0.0, 0.0, 1.0]) > dcn.coordinate_collision_score([0.0, 0.5, 1.0])


def test_bimodal_and_trimodal_cases_run_without_stiffness_p_claims():
    for name in ("bimodal", "trimodal"):
        case = dcn.synthetic_case(name, n=301)
        result = dcn.analyze_case(case, K=14, lambda_values=(0.0, 0.5, 1.0))
        assert result["name"] == name
        assert len(result["records"]) == 3
        for row in result["records"]:
            assert row["collision_status"] == "COMPUTED_PHASE_COHERENCE_PROXY"
            assert row["endpoint_left"] == 0.0
            assert row["endpoint_right"] == 1.0
            assert row["rho_integral"] == pytest.approx(1.0, abs=2.0e-10)
        text = json.dumps(result)
        assert "stiffness" not in text.lower()


def test_conditional_capp_keeps_protected_points_and_makes_no_global_claim():
    result = dcn.conditional_capp_optimize(12, [0.22, 0.74], alpha=1.0, beta=1.0)
    assert result["status"] in {"NUMERICAL_SMALL_SCALE_ONLY", "BLOCKED_NONCONVEX_OR_SOLVER"}
    values = np.asarray(result["values"], dtype=float)
    assert values.size == 12
    assert np.all(np.diff(values) > 0.0)
    assert np.min(np.abs(values - 0.22)) < 1.0e-12
    assert np.min(np.abs(values - 0.74)) < 1.0e-12
    assert result["global_optimum_claim"] is False
    assert "not_established" in result["convexity"]


def test_conditional_indexed_protection_is_fixed():
    result = dcn.conditional_capp_optimize(
        10,
        [0.2, 0.8],
        protected_indices=[2, 7],
        alpha=1.0,
        beta=1.0,
    )
    values = np.asarray(result["values"], dtype=float)
    np.testing.assert_allclose(values[[2, 7]], [0.2, 0.8], atol=1.0e-12)
    assert result["global_optimum_claim"] is False


def test_native_frequency_protection_is_converted_to_phi():
    native = dcn.frequency_from_coordinate([0.2, 0.8])
    result = dcn.conditional_capp_from_native_frequencies(10, native, alpha=1.0, beta=1.0)
    np.testing.assert_allclose(result["protected_values"], [0.2, 0.8], atol=1.0e-12)
    values = np.asarray(result["values"], dtype=float)
    assert np.min(np.abs(values - 0.2)) < 1.0e-12
    assert np.min(np.abs(values - 0.8)) < 1.0e-12


def test_r0_loader_and_cli_writes_json_and_csv(tmp_path: Path):
    x = np.linspace(0.0, 8.0, 101).tolist()
    payload = {
        "name": "r0_fixture",
        "delta": x,
        "m": (0.2 + np.asarray(x) ** 2).tolist(),
        "distances": np.linspace(1.0, 8.0, 32).tolist(),
        "distance_weights": np.ones(32).tolist(),
        "protected_values": [2.0, 6.0],
    }
    r0 = tmp_path / "R0.json"
    r0.write_text(json.dumps(payload), encoding="utf-8")
    loaded = dcn.load_r0_json(r0)
    assert loaded["name"] == "r0_fixture"
    assert loaded["x"][0] == 0.0 and loaded["x"][-1] == 1.0
    assert loaded["source"]["sha256"]
    assert loaded["protected_values"] == [0.25, 0.75]

    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    assert dcn.main(
        [
            "--r0-json",
            str(r0),
            "--synthetic",
            "none",
            "--K",
            "8",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    ) == 0
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["gpu_used"] is False
    assert result["cases"][0]["name"] == "r0_fixture"
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["collision_status"] == "COMPUTED_PHASE_COHERENCE_PROXY"


def test_self_checks_pass():
    assert dcn.run_self_checks()["status"] == "PASS"
