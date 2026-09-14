import hashlib
import json
import math

import numpy as np
import pytest

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913 import tables as table_module
from experiments.fixed_rope_three_interfaces_20260913.causal_intervention import (
    finite_attention_intervention,
    first_divergence,
    rotate_split_half,
    select_cases,
)
from experiments.fixed_rope_three_interfaces_20260913.factorial_report import (
    ARMS,
    build_report,
    filter_rows,
    row_effects,
)
from experiments.fixed_rope_three_interfaces_20260913.math_and_transport import (
    analytic_profiles,
    build_capsule,
    run_verification,
    transport_fixed_u,
)
from experiments.fixed_rope_three_interfaces_20260913.tailspline_verification import (
    run_verification as run_tailspline_verification,
)


def prompt_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def test_today_math_suite_and_fixed_u_reproducible_values():
    verification = run_verification()
    assert verification["passed"] == 31
    assert all(record["passed"] for record in verification["checks"])
    native = np.power(500_000.0, -np.arange(64, dtype=np.float64) / 64.0)
    profile = analytic_profiles(64, low=16, high=34)["mix075"]
    parent = native * np.power(4.0, -profile)
    result = transport_fixed_u(
        native, parent, scale_from=4.0, scale_to=8.0, low=16, high=34,
    )
    assert result["alpha"] == pytest.approx(0.757685348286866, abs=2e-15)
    assert result["target_exponents"][25] == pytest.approx(0.634588845, abs=6e-10)
    assert result["max_normalized_u_residual"] < 3e-16
    assert np.all(result["fixed_u_inv_freq"][17:34] > result["fixed_m_inv_freq"][17:34])


def test_runner_mix075_matches_the_registered_exponent_formula():
    expected = analytic_profiles(64, low=16, high=34)["mix075"]
    actual = table_module.analytic_exponents("mix075", 64, low=16, high=34)
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("n", [17, 18])
def test_tailspline_is_exact_finite_grid_solution_and_not_mix075(n):
    pairs, low, high = n + 5, 2, n + 2
    actual = table_module.analytic_exponents(
        "tailspline", pairs, low=low, high=high,
    )
    q = np.clip(np.arange(pairs, dtype=np.float64) - low, 0, n)
    expected = q * (3 * n * n + 3 * n + 1 - q * q)
    expected /= n * (n + 1) * (2 * n + 1)
    np.testing.assert_array_equal(actual, expected)
    epsilon = np.diff(actual[low : high + 1])
    q_increment = np.arange(1, n + 1, dtype=np.float64)
    expected_epsilon = 3 * (n + q_increment) * (n - q_increment + 1)
    expected_epsilon /= n * (n + 1) * (2 * n + 1)
    np.testing.assert_allclose(epsilon, expected_epsilon, atol=2e-16, rtol=2e-15)
    assert actual[low] == 0.0 and actual[high] == 1.0
    assert np.all(epsilon > 0.0) and np.all(np.diff(epsilon) < 0.0)
    assert epsilon.sum() == pytest.approx(1.0, abs=2e-16)

    bm = table_module.analytic_exponents("bm", pairs, low=low, high=high)
    front = table_module.analytic_exponents(
        "mrpro_frontloaded", pairs, low=low, high=high,
    )
    finite_weight = 3 * n / (2 * (2 * n + 1))
    np.testing.assert_allclose(
        actual, (1 - finite_weight) * bm + finite_weight * front,
        atol=2e-16, rtol=2e-15,
    )
    mix075 = table_module.analytic_exponents(
        "mix075", pairs, low=low, high=high,
    )
    assert np.max(np.abs(actual - mix075)) > 0.005
    with pytest.raises(ValueError, match="depth=1"):
        table_module.analytic_exponents(
            "tailspline", pairs, low=low, high=high, depth=0.9,
        )


@pytest.mark.parametrize("n", [3, 17, 18, 31])
def test_tailspline_dose_control_matches_sum_and_endpoints(n):
    pairs = n + 5
    low, high = 2, 2 + n
    tailspline = table_module.analytic_exponents(
        "tailspline", pairs, low=low, high=high,
    )
    control = table_module.analytic_exponents(
        "tailspline_dose_control", pairs, low=low, high=high,
    )
    assert control[low] == 0.0
    assert control[high] == 1.0
    assert np.all(np.diff(control) >= 0.0)
    assert np.sum(control) == pytest.approx(np.sum(tailspline), abs=2e-14)
    assert not np.array_equal(control, tailspline)


def test_tailspline_builder_enforces_shared_mrpro_band_and_gain(monkeypatch):
    config = {
        "model_type": "llama", "hidden_size": 16, "num_attention_heads": 2,
        "rope_theta": 10_000.0, "max_position_embeddings": 128,
    }
    native = np.geomspace(1.0, 0.001, 4).astype(np.float32)
    monkeypatch.setattr(table_module, "runtime_native_inv_freq", lambda _geometry: native.copy())
    monkeypatch.setattr(table_module, "default_band", lambda _geometry: (1, 3))
    expected_gain = 1.0 + 0.1 * math.log(4.0)
    values, gain, meta = table_module.build_analytic(
        config, method="tailspline", scale=4.0, low=1, high=3,
        depth=1.0, gain=expected_gain,
    )
    assert gain == expected_gain
    assert meta["method"] == "tailspline_exact_finite_grid"
    assert meta["finite_grid_front_weight"] == pytest.approx(0.6)
    assert meta["fitted_coefficients"] == 0
    expected_m = table_module.analytic_exponents("tailspline", 4, low=1, high=3)
    np.testing.assert_array_equal(
        values, (native.astype(np.float64) * np.power(4.0, -expected_m)).astype(np.float32),
    )
    with pytest.raises(ValueError, match="canonical MrRoPE"):
        table_module.build_analytic(
            config, method="tailspline", scale=4.0, low=0, high=3,
            depth=1.0, gain=expected_gain,
        )
    with pytest.raises(ValueError, match="shared YaRN/MrRoPE gain"):
        table_module.build_analytic(
            config, method="tailspline", scale=4.0, low=1, high=3,
            depth=1.0, gain=1.0,
        )
    with pytest.raises(ValueError, match="depth=1"):
        table_module.build_analytic(
            config, method="tailspline", scale=4.0, low=1, high=3,
            depth=0.9, gain=expected_gain,
        )


def test_tailspline_cpu_verification_matches_documented_widths_and_scope():
    verification = run_tailspline_verification()
    assert verification["status"] == "TAILSPLINE_CPU_VERIFICATION_V2"
    assert verification["checkpoint_loaded"] is False
    assert verification["real_model_evaluations"] == 0
    assert verification["benchmark_advantage_demonstrated"] is False
    assert verification["boundary_condition_audit"] == {
        "one_sided_entry_penalty": 0.0,
        "one_sided_tail_penalty": 1.0,
        "one_sided_unique_minimizer": "tailspline",
        "symmetric_entry_penalty": 1.0,
        "symmetric_tail_penalty": 1.0,
        "symmetric_unique_minimizer": "bm",
        "nonnegative_entry_penalty_has_unique_closed_form_continuum": True,
        "intermediate_penalties_are_gpu_candidates": False,
        "cpu_selects_boundary_condition": False,
    }
    records = {record["n"]: record for record in verification["widths"]}
    assert set(records) == {17, 18}
    assert all(all(record["checks"].values()) for record in records.values())
    assert records[17]["finite_grid_front_weight"] == pytest.approx(0.7285714285714285)
    assert records[18]["finite_grid_front_weight"] == pytest.approx(0.7297297297297297)
    assert records[17]["objective"] == pytest.approx(0.0005602240896358543)
    assert records[18]["objective"] == pytest.approx(0.000474158368895211)
    assert records[17]["symmetric_objective"] == pytest.approx(12 / (17 * 18 * 19))
    assert records[18]["symmetric_objective"] == pytest.approx(12 / (18 * 19 * 20))
    assert records[18]["transport_geometry"]["tailspline"]["increment_centroid"] < (
        records[18]["transport_geometry"]["bm"]["increment_centroid"]
    )
    assert records[18]["transport_geometry"]["bm"]["increment_centroid"] < (
        records[18]["transport_geometry"]["mrpro"]["increment_centroid"]
    )
    assert records[18]["maximum_relative_frequency_difference_from_bm"]["S4"] > 0.24
    assert records[17]["maximum_relative_frequency_difference_from_mix075"]["S8"] == pytest.approx(
        0.012214028458268356,
    )


def test_transport_capsule_keeps_actual_arrays_and_is_table_wrappable():
    native = np.geomspace(1.0, 0.001, 8).astype(np.float32)
    exponent = np.array([0, 0, 0.2, 0.5, 0.8, 1, 1, 1], dtype=np.float64)
    parent = (native.astype(np.float64) * np.power(4.0, -exponent)).astype(np.float32)
    receipt, arrays = build_capsule({
        "model_id": "tiny", "native_inv_freq": native.tolist(),
        "parent_inv_freq": parent.tolist(), "native_length": 128,
        "scale_from": 4.0, "scale_to": 8.0, "low": 1, "high": 5,
        "gain": 1.13,
    })
    assert receipt["status"] == "FIXED_U_TRANSPORT_CAPSULE_V1"
    assert receipt["scope"].startswith("CPU construction")
    np.testing.assert_array_equal(receipt["table"]["values_float32"], arrays["fixed_u_inv_freq"])
    assert receipt["table"]["construction"]["model_weight_updates"] == 0
    with pytest.raises(ValueError, match="declared band"):
        transport_fixed_u(native, parent, scale_from=4.0, scale_to=8.0, low=2, high=5)


def test_runner_receipt_transport_builds_fixed_m_and_fixed_u_without_retuning(monkeypatch):
    config = {
        "model_type": "llama", "hidden_size": 16, "num_attention_heads": 2,
        "rope_theta": 10_000.0, "max_position_embeddings": 128,
    }
    geometry = table_module.model_geometry(config)
    native = np.array([1.0, 0.2, 0.04, 0.008], dtype=np.float32)
    monkeypatch.setattr(table_module, "runtime_native_inv_freq", lambda _geometry: native.copy())
    exponents = np.array([0.0, 0.0, 0.7, 1.0])
    values = (native.astype(np.float64) * np.power(4.0, -exponents)).astype(np.float32)
    parent = {
        "status": TABLE_FORMAT, "candidate_id": "parent", "model_geometry": geometry,
        "scale": 4.0, "table_sha256_float32": table_module.tensor_sha256(values),
        "table": {"values_float32": values.tolist(), "gain": 1.07},
    }
    fixed_m, m_gain, m_meta = table_module.build_scale_transport_control(
        config, parent, scale_from=4.0, scale_to=8.0, low=1, high=3, mode="fixed_m",
    )
    fixed_u, u_gain, u_meta = table_module.build_scale_transport_control(
        config, parent, scale_from=4.0, scale_to=8.0, low=1, high=3, mode="fixed_u",
    )
    assert m_gain == u_gain == 1.07
    assert m_meta["target_exponents"] == pytest.approx(exponents)
    assert u_meta["max_normalized_u_residual"] < 3e-16
    assert fixed_u[2] > fixed_m[2]
    assert fixed_u[[0, 1, 3]].tolist() == fixed_m[[0, 1, 3]].tolist()
    _, common_gain, gain_meta = table_module.build_scale_transport_control(
        config, parent, scale_from=4.0, scale_to=8.0, low=1, high=3,
        mode="fixed_u", gain=1.09,
    )
    assert common_gain == 1.09
    assert gain_meta["same_gain_as_parent"] is False


def test_factorial_report_uses_paired_row_effects_and_task_equal_cells():
    tasks, lengths = ["fwe", "vt"], [8192, 32768]
    arms = {name: {} for name in ARMS}
    score_offsets = {"Y00": 0.0, "Y01": 0.1, "Y10": 0.2, "Y11": 0.5}
    for task_index, task in enumerate(tasks):
        for length_index, length in enumerate(lengths):
            for row_index in range(2):
                prompt = f"{task}-{length}-{row_index}"
                base = 0.05 * (task_index + length_index + row_index)
                for arm in ARMS:
                    arms[arm][prompt] = {
                        "prompt_sha256": prompt, "task": task, "length_cap": length,
                        "official_score": base + score_offsets[arm],
                    }
    report = build_report(
        arms, tasks=tasks, lengths=lengths, rows_per_cell=2, draws=200, seed=14,
    )
    effect = report["task_equal_over_declared_cells"]
    assert effect["F_g0"] == pytest.approx(0.2)
    assert effect["G_T0"] == pytest.approx(0.1)
    assert effect["interaction"] == pytest.approx(0.2)
    assert effect["delta_F"] + effect["delta_G"] == pytest.approx(effect["combined"])
    assert report["interpretation_contract"]["mechanism_panel_not_core6_auc"] is True
    assert row_effects(0.0, 0.1, 0.2, 0.5)["combined"] == pytest.approx(0.5)


def test_factorial_report_filters_broader_reused_sources_to_declared_cells():
    rows = {
        "keep": {"task": "fwe", "length_cap": 8192},
        "other_task": {"task": "qa_1", "length_cap": 8192},
        "other_length": {"task": "fwe", "length_cap": 16384},
    }
    assert filter_rows(rows, tasks={"fwe"}, lengths={8192}) == {
        "keep": rows["keep"],
    }


def test_case_selection_is_bounded_deterministic_and_uses_existing_divergences():
    panel, recipient, donor = [], {}, {}
    specs = [
        ("fwe", 0.0, 1.0), ("fwe", 0.0, 1.0), ("fwe", 0.0, 1.0),
        ("fwe", 0.0, 1.0), ("fwe", 0.0, 1.0),
        ("vt", 1.0, 0.0), ("vt", 1.0, 0.0), ("vt", 1.0, 0.0),
        ("niah_single_2", 1.0, 1.0), ("qa_1", 0.0, 0.0),
    ]
    for index, (task, left_score, right_score) in enumerate(specs):
        ids = [10, index + 20]
        identity = prompt_hash(ids)
        panel.append({
            "row_id": f"r{index:02d}", "task": task, "length_cap": 8192,
            "prompt_ids": ids, "prompt_sha256": identity, "references": ["answer"],
            "max_new_tokens": 4,
        })
        recipient[identity] = {
            "task": task, "generated_ids": [5, 100 + index, 2],
            "official_score": left_score,
        }
        donor[identity] = {
            "task": task, "generated_ids": [5, 200 + index, 2],
            "official_score": right_score,
        }
    selected = select_cases(panel, recipient, donor, length=8192)
    assert [row["category"] for row in selected] == [
        "fwe_damage", "fwe_damage", "fwe_damage", "fwe_damage",
        "vt_benefit", "vt_benefit", "concordant", "concordant",
    ]
    assert [row["row_id"] for row in selected[:4]] == ["r00", "r01", "r02", "r03"]
    assert all(row["selection_uses_logits"] is False for row in selected)
    assert first_divergence([1, 2], [1, 3]) == (1, 2, 3)
    assert first_divergence([1, 2], [1, 2]) is None


def test_finite_attention_intervention_matches_direct_rotary_and_signed_formula():
    rng = np.random.default_rng(914)
    q = rng.normal(size=(4, 4))
    k = rng.normal(size=(2, 6, 4))
    v = rng.normal(size=(2, 6, 4))
    projection = rng.normal(size=(16, 16))
    recipient_inv = np.array([0.7, 0.08])
    donor_inv = np.array([0.55, 0.03])
    result = finite_attention_intervention(
        q, k, v, query_position=5,
        recipient_inv_freq=recipient_inv, donor_inv_freq=donor_inv,
        recipient_gain=1.1, donor_gain=1.1,
        attention_scale=1 / math.sqrt(4), output_projection=projection,
    )
    assert result["recipient"]["signed_formula_max_abs_error"] < 2e-15
    assert result["phase_only"]["signed_formula_max_abs_error"] < 2e-15
    repeated_k = np.repeat(k, 2, axis=0)
    repeated_v = np.repeat(v, 2, axis=0)
    q_rot = 1.1 * rotate_split_half(q, 5, donor_inv)
    k_rot = 1.1 * rotate_split_half(repeated_k, np.arange(6)[None, :], donor_inv)
    logits = np.einsum("hd,htd->ht", q_rot, k_rot) / math.sqrt(4)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = projection @ np.einsum("ht,htd->hd", probabilities, repeated_v).reshape(-1)
    np.testing.assert_allclose(result["phase_only"]["attention_output"], expected, atol=2e-14)
