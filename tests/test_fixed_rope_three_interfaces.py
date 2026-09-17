import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913.pipeline import (
    DRAFT_FORMAT,
    bootstrap_range_contrast,
    file_sha256,
    freeze_contract,
    plan_queue,
    read_jsonl,
    summarize_point,
)
from experiments.fixed_rope_three_interfaces_20260913.margin_direction import (
    exponents_from_increments,
    feasible_step,
    select_direction,
)
from experiments.fixed_rope_three_interfaces_20260913.full_z_fwe_repair import (
    minimum_norm_margin_step,
)
from experiments.fixed_rope_three_interfaces_20260913.constrained_full_z_fwe_repair import (
    exponent_feasibility,
    maximum_feasible_line_scale,
    minimum_norm_exponent_step,
)
from experiments.fixed_rope_three_interfaces_20260913.target_support_z import (
    TargetSupportZAllocation,
)
from experiments.fixed_rope_three_interfaces_20260913.tables import (
    analytic_exponents,
    build_depth_control,
    build_exact_c42,
    build_bm_skew_control,
    build_exponent_mix_control,
    build_gain_control,
    build_tail_cap_control,
    build_tail_denominator_control,
    make_receipt,
    model_geometry,
    profile_metadata,
    runtime_native_inv_freq,
)
from experiments.fixed_rope_three_interfaces_20260913.scale_transfer_audit import (
    audit_scale_transfer,
)
from experiments.fixed_rope_three_interfaces_20260913.matched_point_report import (
    build_point_report,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def tiny_config() -> dict:
    return {
        "model_type": "llama",
        "hidden_size": 128,
        "num_attention_heads": 1,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 128,
    }


def c42_config() -> dict:
    return {
        "model_type": "olmo2",
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "rope_theta": 500_000.0,
        "max_position_embeddings": 4096,
    }


def test_analytic_profiles_are_monotone_and_have_exact_endpoints():
    for method in ("mrpro", "mrpro_frontloaded", "bm", "uni"):
        values = analytic_exponents(method, 64, low=10, high=30)
        assert values[10] == 0.0
        assert values[30] == 1.0
        assert np.all(np.diff(values) >= 0.0)
        assert profile_metadata(values)["band_envelope"] == [10, 30]


def test_depth_control_scales_the_full_profile_not_only_the_tail():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    full = analytic_exponents("bm", 64, low=10, high=30)
    values = (native * np.power(4.0, -full)).astype(np.float32)
    parent = make_receipt(
        candidate_id="parent", model_id="model", role="candidate", scale=4.0,
        geometry=geometry, values=values, gain=1.2, construction={"method": "test"},
        source="test",
    )
    installed, gain, construction = build_depth_control(
        config, parent, scale=4.0, depth=0.6, gain=None,
    )
    recovered = -np.log(installed.astype(np.float64) / native) / np.log(4.0)
    assert recovered == pytest.approx(0.6 * full, abs=2e-7)
    assert gain == 1.2
    assert construction["tail_depth"] == 0.6
    full_installed, _, _ = build_depth_control(
        config, parent, scale=4.0, depth=1.0, gain=None,
    )
    np.testing.assert_array_equal(full_installed, values)


def test_replanned_queue_keeps_live_output_as_prefix_of_resident_input(tmp_path: Path):
    model_dir = tmp_path / "model"
    config = tiny_config()
    write_json(model_dir / "config.json", config)
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    candidate_values = (native * np.power(2.0, -np.linspace(0.0, 1.0, 64))).astype(np.float32)
    tables = {}
    for table_id, values, role in (
        ("candidate", candidate_values, "candidate"),
        ("baseline", native, "native"),
    ):
        receipt = make_receipt(
            candidate_id=table_id, model_id="model", role=role, scale=2.0,
            geometry=geometry, values=values, gain=1.0,
            construction={"method": "test"}, source="test",
        )
        path = tmp_path / "tables" / f"{table_id}.json"
        write_json(path, receipt)
        tables[table_id] = str(path)

    rows = []
    for index, length in enumerate((64, 128)):
        prompt = [1, 2, index + 3]
        prompt_hash = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
        rows.append({
            "row_id": str(index), "task": "niah_single_2", "length_cap": length,
            "prompt_ids": prompt, "prompt_sha256": prompt_hash,
            "references": ["answer"], "max_new_tokens": 4,
        })
    panel_path = tmp_path / "panel.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(panel_path, rows)
    write_json(manifest_path, {
        "status": "FROZEN", "rows": 2, "panel_sha256": file_sha256(panel_path),
    })
    candidate_output = tmp_path / "runs/candidate"
    write_jsonl(candidate_output / "generations.jsonl", [{
        **rows[0], "arm": "candidate", "output_text": "answer", "generated_ids": [7],
        "ended_eos": True,
    }])
    draft = {
        "status": DRAFT_FORMAT,
        "stage_order": ["paper_confirm"],
        "models": {"model": {
            "model_path": str(model_dir), "revision": "test",
            "tokenizer_template": "test", "precision_arithmetic": "test",
            "prefill_chunk_size": 0,
        }},
        "panels": {"panel": {
            "model_id": "model", "path": str(panel_path),
            "manifest_path": str(manifest_path),
        }},
        "tables": tables,
        "jobs": [
            {
                "job_id": table_id, "stage": "paper_confirm", "priority": index,
                "model_id": "model", "panel_id": "panel", "table_id": table_id,
                "output_dir": str(candidate_output if table_id == "candidate" else tmp_path / "runs/baseline"),
                "result_sources": [], "reuse_receipts": [], "source_arm_labels": [],
                "decoder": "greedy", "scorer": "official",
            }
            for index, table_id in enumerate(("candidate", "baseline"))
        ],
        "comparisons": [{
            "comparison_id": "comparison", "panel_id": "panel",
            "candidate": "candidate", "baselines": ["baseline"],
        }],
    }
    draft_path = tmp_path / "draft.json"
    contract_path = tmp_path / "contract.json"
    write_json(draft_path, draft)
    contract = freeze_contract(draft_path, contract_path)
    write_json(candidate_output / "contract.json", {
        "status": "FIXED_ROPE_RESIDENT_RUN_V1",
        "pipeline_contract_sha256": file_sha256(contract_path),
        "job_id": "candidate", "stage": "paper_confirm",
        "model_id": "model", "model_config_sha256": contract["models"]["model"]["config_sha256"],
        "panel_id": "panel", "panel_sha256": contract["panels"]["panel"]["panel_sha256"],
        "table_id": "candidate",
        "table_receipt_sha256": contract["tables"]["candidate"]["receipt_sha256"],
        "table_sha256_float32": contract["tables"]["candidate"]["table_sha256_float32"],
        "gain": contract["tables"]["candidate"]["gain"],
        "decoder": "greedy", "scorer": "official", "precision_arithmetic": "test",
        "row_prompt_sha256": [row["prompt_sha256"] for row in rows],
        "same_table_all_layers_and_lengths": True, "runtime_table_switching": False,
    })
    existing = read_jsonl(candidate_output / "generations.jsonl")
    existing[0]["table_sha256_float32"] = contract["tables"]["candidate"]["table_sha256_float32"]
    existing[0]["gain"] = contract["tables"]["candidate"]["gain"]
    write_jsonl(candidate_output / "generations.jsonl", existing)
    planned = plan_queue(contract_path, tmp_path / "queue")
    candidate_job = next(job for job in planned["queue"] if job["job_id"] == "candidate")
    assert candidate_job["remaining_rows"] == 1
    assert candidate_job["saved_prefix_rows"] == 1
    resident_rows = read_jsonl(Path(candidate_job["missing_panel"]))
    assert [row["prompt_sha256"] for row in resident_rows] == [row["prompt_sha256"] for row in rows]
    assert [row["source_row_id"] for row in resident_rows] == ["0", "1"]


def test_point_summary_is_task_equal_not_row_equal():
    rows = [
        {"task": "a", "length_cap": 8, "official_score": 1.0, "ended_eos": True, "hit_cap": False},
        {"task": "a", "length_cap": 8, "official_score": 1.0, "ended_eos": True, "hit_cap": False},
        {"task": "b", "length_cap": 8, "official_score": 0.0, "ended_eos": False, "hit_cap": True},
    ]
    summary = summarize_point(rows, ["a", "b"], 8)
    assert summary["task_macro_official"] == 0.5
    assert summary["by_length"]["8"]["tasks"]["a"]["rows"] == 2


def test_point_report_supports_a_single_task_paired_slice():
    arms = {}
    for arm, scores in (("tailspline", (1.0, 0.5)), ("mrpro", (0.0, 0.5))):
        arms[arm] = [
            {
                "task": "niah_single_1",
                "length_cap": 131072,
                "prompt_sha256": f"prompt-{index}",
                "official_score": score,
                "ended_eos": True,
                "hit_cap": False,
            }
            for index, score in enumerate(scores)
        ]
    result = build_point_report(
        arms, candidate="tailspline", baselines=["mrpro"], draws=20, seed=7,
    )
    assert result["tasks"] == ["niah_single_1"]
    assert result["paired_prompts"] == 2
    assert result["contrasts"]["mrpro"]["delta_task_macro_official"] == 0.5


def test_receipt_format_is_explicit():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    receipt = make_receipt(
        candidate_id="native", model_id="model", role="native", scale=2.0,
        geometry=geometry, values=native, gain=1.0,
        construction={"method": "identity"}, source="test",
    )
    assert receipt["status"] == TABLE_FORMAT
    assert receipt["exponent_reconstruction"]["exact"] is True
    assert receipt["table"]["construction"]["runtime_table_switching"] is False
    with pytest.raises(ValueError, match="exact FP32 Native"):
        make_receipt(
            candidate_id="false-native", model_id="model", role="native", scale=2.0,
            geometry=geometry, values=(native * 0.9).astype(np.float32), gain=1.0,
            construction={"method": "invalid"}, source="test",
        )


def test_range_bootstrap_recomputes_worst_length_inside_every_draw():
    candidate = []
    baseline = []
    for task in ("a", "b"):
        for length in (8, 16, 32):
            for case in range(2):
                common = {
                    "task": task, "length_cap": length,
                    "prompt_sha256": f"{task}-{length}-{case}",
                    "mini_semantic_id": f"{task}-{case}",
                }
                candidate.append({**common, "official_score": 1.0})
                baseline.append({**common, "official_score": 0.0})
    result = bootstrap_range_contrast(
        candidate, baseline, tasks=["a", "b"], lengths=[8, 16, 32],
        task_families={"a": "family_a", "b": "family_b"}, draws=20, seed=7,
    )
    assert result["delta_worst_length_score"]["interval95"] == pytest.approx([1.0, 1.0])
    assert result["candidate_worst_length_score"]["recomputed_inside_each_draw"] is True
    assert result["delta_curve_simultaneous95_halfwidth"] == 0.0


def test_range_bootstrap_treats_single_length_auc_as_point_score():
    candidate = []
    baseline = []
    for task in ("a", "b"):
        for case in range(3):
            common = {
                "task": task, "length_cap": 32,
                "prompt_sha256": f"{task}-32-{case}",
                "mini_semantic_id": f"{task}-{case}",
            }
            candidate.append({**common, "official_score": 0.75})
            baseline.append({**common, "official_score": 0.25})
    result = bootstrap_range_contrast(
        candidate, baseline, tasks=["a", "b"], lengths=[32],
        task_families={"a": "family_a", "b": "family_b"}, draws=20, seed=7,
    )
    assert result["delta_log_auc"]["mean"] == pytest.approx(0.5)
    assert result["delta_log_auc"]["interval95"] == pytest.approx([0.5, 0.5])
    assert result["delta_worst_length_score"]["mean"] == pytest.approx(0.5)


def test_constrained_full_z_step_stays_in_monotone_exponent_box():
    initial = np.asarray([0.0, 0.0, 0.2, 0.7, 1.0, 1.0])
    margins = np.asarray([-0.2, -0.1])
    jacobian = np.asarray([
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, -3.0, 0.0],
    ])
    step = minimum_norm_exponent_step(
        margins, jacobian, initial, target_margin=0.0, trust_bound=0.1,
    )
    valid, active = exponent_feasibility(initial, step)
    assert valid
    assert margins + jacobian @ step == pytest.approx([0.0, 0.0], abs=2e-6)
    assert np.all(np.diff(active) >= -1e-8)
    assert active[0] == 0.0 and active[-1] == 1.0


def test_constrained_full_z_step_reports_infeasible_margin_contract():
    initial = np.asarray([0.0, 0.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="infeasible"):
        minimum_norm_exponent_step(
            np.asarray([-1.0]), np.asarray([[1.0, 1.0]]), initial,
            target_margin=0.5, trust_bound=0.01,
        )


def test_constrained_full_z_ray_limit_respects_trust_and_monotonicity():
    initial = np.asarray([0.0, 0.0, 0.2, 0.7, 1.0, 1.0])
    step = np.asarray([0.0, 0.05, -0.05, 0.0])
    alpha = maximum_feasible_line_scale(initial, step, trust_bound=0.1)
    assert alpha == pytest.approx(2.0)
    valid, _ = exponent_feasibility(initial, alpha * step)
    assert valid


def test_scale_transfer_audit_separates_common_length_and_endpoint_ratios():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    exponents = analytic_exponents("bm", 64, low=10, high=30)
    receipts = []
    for scale, gain in ((2.0, 1.1), (4.0, 1.2)):
        values = (native * np.power(scale, -exponents)).astype(np.float32)
        receipts.append(make_receipt(
            candidate_id=f"s{int(scale)}", model_id="model", role="candidate",
            scale=scale, geometry=geometry, values=values, gain=gain,
            construction={"method": "test"}, source="test",
        ))
    result = audit_scale_transfer(*receipts)
    assert result["same_exponent_allocation"] is True
    assert result["max_abs_exponent_difference"] < 2e-6
    assert result["max_relative_frequency_identity_error"] < 2e-6
    ratios = np.asarray(result["corresponding_endpoint_phase_ratio"])
    assert ratios[0] == pytest.approx(2.0, rel=2e-6)
    assert ratios[-1] == pytest.approx(1.0, rel=2e-6)


def test_matched_point_report_is_task_equal_and_paired():
    rows = {}
    for arm, delta in (("candidate", 1.0), ("baseline", 0.0)):
        rows[arm] = []
        for task, count in (("a", 2), ("b", 1)):
            for index in range(count):
                rows[arm].append({
                    "task": task, "length_cap": 32,
                    "prompt_sha256": f"{task}-{index}",
                    "official_score": delta,
                    "ended_eos": True, "hit_cap": False,
                })
    report = build_point_report(
        rows, candidate="candidate", baselines=["baseline"], draws=20, seed=3,
    )
    assert report["summaries"]["candidate"]["task_macro_official"] == 1.0
    assert report["contrasts"]["baseline"]["delta_task_macro_official"] == 1.0
    assert report["contrasts"]["baseline"]["bootstrap"]["delta_task_macro_interval95"] == [1.0, 1.0]


def test_tail_denominator_control_leaves_faster_slots_exact():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    exponents = analytic_exponents("bm", 64, low=10, high=30)
    values = (native.astype(np.float64) * np.power(4.0, -exponents)).astype(np.float32)
    parent = make_receipt(
        candidate_id="parent", model_id="model", role="candidate", scale=4.0,
        geometry=geometry, values=values, gain=1.1,
        construction={"method": "test"}, source="test",
    )
    changed, gain, construction = build_tail_denominator_control(
        config, parent, scale=4.0, slow_start=31, denominator_fraction=0.9,
    )
    np.testing.assert_array_equal(changed[:31], values[:31])
    np.testing.assert_allclose(changed[31:], native[31:] / 3.6, rtol=2e-7, atol=0)
    assert np.all(changed[:-1] > changed[1:])
    assert gain == 1.1
    assert construction["denominator"] == 3.6
    receipt = make_receipt(
        candidate_id="tail", model_id="model", role="control", scale=4.0,
        geometry=geometry, values=changed, gain=gain, construction=construction,
        source="test", allow_nonmonotone_exponents=True,
    )
    assert receipt["monotone_exponents"] is False


def test_tail_cap_is_monotone_and_preserves_its_prefix():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    exponents = analytic_exponents("bm", 64, low=10, high=30)
    values = (native.astype(np.float64) * np.power(4.0, -exponents)).astype(np.float32)
    parent = make_receipt(
        candidate_id="parent", model_id="model", role="candidate", scale=4.0,
        geometry=geometry, values=values, gain=1.1,
        construction={"method": "test"}, source="test",
    )
    changed, gain, construction = build_tail_cap_control(
        config, parent, scale=4.0, denominator_fraction=0.9,
    )
    first = construction["first_changed_slot"]
    np.testing.assert_array_equal(changed[:first], values[:first])
    recovered = -np.log(changed.astype(np.float64) / native) / np.log(4.0)
    assert np.all(np.diff(recovered) >= -2e-7)
    assert recovered[-1] == pytest.approx(np.log(3.6) / np.log(4.0), abs=2e-7)
    assert gain == 1.1


def test_exponent_mix_interpolates_in_m_coordinates():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    parents = []
    parent_exponents = []
    for method in ("bm", "mrpro_frontloaded"):
        exponents = analytic_exponents(method, 64, low=10, high=30)
        values = (native.astype(np.float64) * np.power(4.0, -exponents)).astype(np.float32)
        parents.append(make_receipt(
            candidate_id=method, model_id="model", role="control", scale=4.0,
            geometry=geometry, values=values, gain=1.1,
            construction={"method": method}, source="test",
        ))
        parent_exponents.append(exponents)
    values, gain, construction = build_exponent_mix_control(
        config, parents[0], parents[1], scale=4.0, right_weight=0.25,
    )
    recovered = -np.log(values.astype(np.float64) / native) / np.log(4.0)
    expected = 0.75 * parent_exponents[0] + 0.25 * parent_exponents[1]
    assert recovered == pytest.approx(expected, abs=2e-7)
    assert gain == 1.1
    assert construction["right_weight"] == 0.25


def test_gain_control_keeps_frequency_table_bit_exact():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    exponents = analytic_exponents("bm", 64, low=10, high=30)
    values = (native.astype(np.float64) * np.power(4.0, -exponents)).astype(np.float32)
    parent = make_receipt(
        candidate_id="parent", model_id="model", role="candidate", scale=4.0,
        geometry=geometry, values=values, gain=1.2,
        construction={"method": "test"}, source="test",
    )
    changed, gain, construction = build_gain_control(
        config, parent, scale=4.0, gain=1.0,
    )
    np.testing.assert_array_equal(changed, values)
    assert gain == 1.0
    assert construction["parent_gain"] == 1.2
    assert construction["frequency_values_bit_exact"] is True


def test_bm_skew_preserves_endpoints_monotonicity_and_sum():
    config = tiny_config()
    geometry = model_geometry(config)
    native = runtime_native_inv_freq(geometry)
    base = analytic_exponents("bm", 64, low=10, high=30)
    for skew in (-2.0, 2.0, 4.0, 8.0):
        values, gain, construction = build_bm_skew_control(
            config, scale=4.0, low=10, high=30, skew=skew, gain=1.1,
        )
        recovered = -np.log(values.astype(np.float64) / native) / np.log(4.0)
        assert recovered[10] == pytest.approx(0.0, abs=2e-7)
        assert recovered[30] == pytest.approx(1.0, abs=2e-7)
        assert np.all(np.diff(recovered) >= -2e-7)
        assert recovered.sum() == pytest.approx(base.sum(), abs=2e-6)
        assert gain == 1.1
        assert construction["skew"] == skew


def test_exact_c42_order_control_preserves_multiset_mass_moment_and_table_contract():
    config = c42_config()
    parent_values, parent_gain, parent = build_exact_c42(config, scale=4.0)
    values, gain, construction = build_exact_c42(
        config, scale=4.0, increment_swaps=[(3, 15), (1, 9)],
    )
    assert not np.array_equal(values, parent_values)
    assert gain == parent_gain
    assert construction["increment_mass_exact"] == parent["increment_mass_exact"] == "1"
    assert construction["increment_centroid_r_exact"] == parent["increment_centroid_r_exact"] == "8"
    assert construction["sum_m_exact"] == parent["sum_m_exact"] == "42"
    assert construction["multiset_preserved_exact"] is True
    assert construction["increment_swaps_one_based"] == [[3, 15], [1, 9]]
    geometry = model_geometry(config)
    receipt = make_receipt(
        candidate_id="c42-order", model_id="olmo", role="control", scale=4.0,
        geometry=geometry, values=values, gain=gain, construction=construction,
        source="test",
    )
    assert receipt["band_envelope"] == [14, 32]
    assert receipt["depth"] == 1.0


def test_margin_direction_selection_and_feasible_step_preserve_contracts():
    increments = np.asarray([0.2, 0.3, 0.3, 0.2])
    gradients = {
        4: np.asarray([2.0, 1.0, 0.0, -1.0]),
        8: np.asarray([1.0, 0.5, 0.0, -0.5]),
        16: np.asarray([3.0, 1.0, -1.0, -2.0]),
    }
    selected = select_direction(
        family="moment", increments=increments,
        gradients_by_length=gradients, lengths=(4, 8, 16),
    )
    direction = np.asarray(selected["vector"])
    assert direction.sum() == pytest.approx(0.0)
    assert np.dot(np.arange(1, 5), direction) == pytest.approx(0.0)
    step = feasible_step(
        increments, direction, fraction=0.25, max_exponent_shift=0.03,
    )
    changed = increments + step * direction
    assert np.all(changed > 0.0)
    assert changed.sum() == pytest.approx(1.0)
    exponents = exponents_from_increments(
        pairs=8, low=1, high=5, increments=changed,
    )
    assert exponents[1] == 0.0 and exponents[5] == pytest.approx(1.0)
    assert np.all(exponents[6:] == 1.0)
    point = select_direction(
        family="gap", increments=increments,
        gradients_by_length={8: gradients[8]}, lengths=(8,),
    )
    assert set(point["derivative_by_length"]) == {"8"}


def test_full_z_minimum_norm_step_satisfies_linear_margin_constraints():
    margins = np.asarray([-0.2, -0.1])
    jacobian = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    step = minimum_norm_margin_step(
        margins, jacobian, target_margin=0.1, bound=1.0,
    )
    assert margins + jacobian @ step == pytest.approx([0.1, 0.1], abs=1e-6)


def test_target_support_z_keeps_endpoints_and_positive_frequency_gaps():
    import torch

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.tensor([1.0, 0.5, 0.25, 0.125]))
            self.attention_scaling = 1.0

    initial = np.asarray([1.0, 0.4, 0.15, 0.05], dtype=np.float32)
    allocation = TargetSupportZAllocation(Rotary(), initial_inv_freq=initial, gain=1.1)
    np.testing.assert_array_equal(allocation.realized_inv_freq().detach().numpy(), initial)
    allocation.set_gap_delta_(np.asarray([0.5, -0.2, -0.3]))
    changed = allocation.realized_inv_freq().detach().numpy()
    assert changed[0] == initial[0] and changed[-1] == initial[-1]
    assert np.all(changed[:-1] > changed[1:])
