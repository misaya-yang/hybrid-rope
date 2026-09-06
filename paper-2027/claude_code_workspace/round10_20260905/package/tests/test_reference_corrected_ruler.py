"""CPU-only fake RULER panels; no real model results are read."""

import copy
import json

import numpy as np
import pytest

from scripts.analysis import summarize_reference_corrected_ruler as summary


def panel():
    protocol = {"data_manifest_sha256": "a" * 64, "tasks": list(summary.TASKS),
                "lengths": [4096, 8192], "limit_per_cell": 20,
                "profile_target_length": 8192, "table_factor": 2,
                "native_context_length": 8192, "checkpoint_sha256": "b" * 64}
    result = {}
    for arm in summary.ARMS:
        rows = []
        for length in protocol["lengths"]:
            for task in summary.TASKS:
                successes = {"native": 20 if length == 4096 else 0,
                             "physical_x": 18, "normalized_index": 17, "official_yarn": 12}[arm]
                for index in range(20):
                    rows.append({"task": task, "nominal_length": length, "local_index": index,
                                 "official_task_score": float(index < successes), "prediction": "complete fake answer"})
        result[arm] = {"result": {"status": summary.TERMINAL_STATUS, "protocol": copy.deepcopy(protocol),
                                  "results": {"examples": len(rows)}}, "rows": rows}
    return result


def test_paired_summary_gates_and_intervals_are_reproducible():
    first = summary.summarize_panel(panel())
    assert first == summary.summarize_panel(panel())
    assert first["scores"]["4096"]["physical_x"]["macro"] == .9
    assert first["paired_comparisons"]["8192"]["physical_minus_index"]["delta"] == pytest.approx(.05)
    interval = first["paired_comparisons"]["8192"]["physical_minus_index"]["paired_stratified_ci95"]
    assert interval[0] <= .05 <= interval[1]
    family_interval = first["paired_comparisons"]["8192"]["physical_minus_index"]["bonferroni_length_family_sensitivity_ci95"]
    assert family_interval[0] <= interval[0] <= interval[1] <= family_interval[1]
    assert first["stage2_entrance"]["status"] == "PASS"
    assert first["reference_retention"]["physical_x"]["status"] == "PASS"
    assert first["bootstrap"]["seed"] == 202609024
    assert first["profile_selection_performed"] is False
    json.dumps(first, allow_nan=False)


def test_zero_native_reference_is_undefined_not_pass():
    data = panel()
    for row in data["native"]["rows"]:
        row["official_task_score"] = 0
    result = summary.summarize_panel(data)
    assert result["stage2_entrance"]["status"] == "PASS"
    assert result["reference_retention"]["physical_x"] == {"ratio": None, "threshold": .875, "status": "UNDEFINED"}


def test_s4_panel_does_not_automatically_authorize_another_scale():
    data = panel()
    for item in data.values():
        item["result"]["protocol"].update(lengths=[4096, 16384], profile_target_length=16384, table_factor=4)
        for row in item["rows"]:
            if row["nominal_length"] == 8192:
                row["nominal_length"] = 16384
    result = summary.summarize_panel(data, reference_length=4096, target_length=16384)
    assert result["stage2_entrance"]["status"] == "NOT_APPLICABLE"
    assert result["stage2_entrance"]["resolver_thresholds_met"] is True


def test_entrance_pass_does_not_hide_native_retention_failure():
    data = panel()
    for row in data["physical_x"]["rows"]:
        if row["nominal_length"] == 4096:
            row["official_task_score"] = 0
    result = summary.summarize_panel(data)
    assert result["stage2_entrance"]["status"] == "PASS"
    assert result["reference_retention"]["physical_x"]["status"] == "FAIL"


def test_single_key_entrance_is_a_separate_conjunction():
    data = panel()
    for row in data["physical_x"]["rows"]:
        if row["nominal_length"] == 8192 and row["task"] == "niah_single_1":
            row["official_task_score"] = float(row["local_index"] < 15)
    result = summary.summarize_panel(data)
    assert result["stage2_entrance"]["physical_macro_minus_native"] > .1
    assert result["stage2_entrance"]["status"] == "FAIL"


def test_exact_inclusive_macro_margin_is_not_lost_to_float_subtraction():
    data = panel()
    for row in data["native"]["rows"]:
        if row["nominal_length"] == 8192:
            row["official_task_score"] = float(row["local_index"] < 16)
    result = summary.summarize_panel(data)
    assert result["stage2_entrance"]["physical_macro_minus_native"] == pytest.approx(.1)
    assert result["stage2_entrance"]["status"] == "PASS"
    assert not summary.at_least(.09999, .1)


def test_row_count_is_read_from_manifest_not_hardcoded():
    data = panel()
    for item in data.values():
        item["result"]["protocol"]["limit_per_cell"] = 10
        item["rows"] = [row for row in item["rows"] if row["local_index"] < 10]
        item["result"]["results"]["examples"] = len(item["rows"])
    result = summary.summarize_panel(data)
    assert result["rows_per_task"] == 10
    assert result["stage2_entrance"]["single_rows"] == 10


@pytest.mark.parametrize("mutation", ["partial", "missing_arm", "missing_row", "duplicate", "data", "tasks", "lengths", "count", "checkpoint", "nonfinite", "index", "prediction", "aggregate", "run_manifest", "factor"])
def test_invalid_or_unpaired_panels_fail_closed(mutation):
    data = panel()
    item = data["physical_x"]
    if mutation == "partial":
        item["result"]["status"] = "PARTIAL"
    elif mutation == "missing_arm":
        data.pop("official_yarn")
    elif mutation == "missing_row":
        item["rows"].pop()
    elif mutation == "duplicate":
        item["rows"].append(copy.deepcopy(item["rows"][0]))
    elif mutation == "data":
        item["result"]["protocol"]["data_manifest_sha256"] = "c" * 64
    elif mutation == "tasks":
        item["result"]["protocol"]["tasks"].pop()
    elif mutation == "lengths":
        item["result"]["protocol"]["lengths"] = [4096]
    elif mutation == "count":
        item["result"]["protocol"]["limit_per_cell"] = 19
    elif mutation == "checkpoint":
        item["result"]["protocol"]["checkpoint_sha256"] = "c" * 64
    elif mutation == "nonfinite":
        item["rows"][0]["official_task_score"] = float("nan")
    elif mutation == "index":
        item["rows"][0]["local_index"] = 20
    elif mutation == "prediction":
        item["rows"][0].pop("prediction")
    elif mutation == "aggregate":
        item["result"]["results"]["cells"] = {"vt": {"4096": {"rows": 20, "official_task_score": 0}}}
    elif mutation == "run_manifest":
        item["run_manifest"] = {}
    else:
        item["result"]["protocol"]["table_factor"] = 4
    with pytest.raises(ValueError):
        summary.summarize_panel(data)


def test_bootstrap_preserves_pairing_not_independent_arm_resampling():
    values = np.zeros((4, 20, 4))
    values[:, :, :] = np.arange(20)[None, :, None] / 20
    draws = summary.paired_bootstrap_cell(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    assert np.all(draws[:, 0] == draws[:, 1])
    assert np.std(draws[:, 0]) > 0


def test_loader_binds_hashes_and_ignores_preflight_directory(tmp_path):
    for arm, item in panel().items():
        directory = tmp_path / arm
        directory.mkdir()
        examples = directory / "examples.jsonl"
        examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
        item["result"]["results"]["examples_sha256"] = summary.file_hash(examples)
        (directory / "results.json").write_text(json.dumps(item["result"]))
        (directory / "run_manifest.json").write_text(json.dumps(item["result"]["protocol"]))
    (tmp_path / "preflight_native").mkdir()
    (tmp_path / "preflight_native" / "results.json").write_text("not a completed panel")
    loaded = summary.load_panel(tmp_path)
    result = summary.summarize_panel(loaded)
    assert set(result["raw_hashes"]) == set(summary.ARMS)
    assert result["raw_hashes"]["native"]["examples_sha256"]
    path = tmp_path / "native" / "examples.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash differs"):
        summary.load_panel(tmp_path)


def test_result_can_use_bound_run_manifest_when_protocol_not_embedded():
    data = panel()
    for item in data.values():
        item["run_manifest"] = item["result"].pop("protocol")
    assert summary.summarize_panel(data)["status"] == "REFERENCE_CORRECTED_RULER_PANEL_COMPLETE"
