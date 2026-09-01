"""Synthetic paired panels only; no observed scores determine analysis rules."""

import copy
import json
import math

import numpy as np
import pytest

from scripts.analysis import summarize_qwen_s2_baseline_completion as summary


def fixture_panel(model="k32"):
    panel = {}
    for arm_index, arm in enumerate(summary.ARMS[model]):
        panel[arm] = {}
        active = ("a" if arm == "native" else str(arm_index)) * 64
        gain = 1 + (.1 if arm == "official_yarn" else .074) * math.log(2)
        for length in summary.LENGTHS:
            protocol = {
                "tasks": list(summary.TASKS), "lengths": [length], "limit_per_cell": 20,
                "checkpoint_sha256": "b" * 64, "script_sha256": str(arm_index + 4) * 64,
                "data_manifest_sha256": str(arm_index + 1) * 64,
                "native_context_length": 32768, "model_type": "qwen2",
                "method": "native" if arm == "native" else "external_table_static",
                "table_factor": 4.0 if arm == "native" else 2.0,
                "table_sha256_float32": None if arm == "native" else active,
                "table_file_sha256": None if arm == "native" else active,
                "long_attention_scaling": None if arm == "native" else gain,
            }
            method = {"method": protocol["method"], "model_type": "qwen2", "native_context_length": 32768,
                      "active_sha256_float32": active}
            if arm == "native":
                method["attention_scaling"] = 1.0
            else:
                method.update(native_sha256_float32="a" * 64, table_factor=2.0, long_attention_scaling=gain)
            rows = [{"task": task, "nominal_length": length, "local_index": i,
                     "official_metric": "string_match_all", "official_task_score": float(i < 12 + arm_index),
                     "references": [f"answer-{task}-{i}"], "prediction": "synthetic full prediction"}
                    for task in summary.TASKS for i in range(20)]
            data = {"manifest_sha256": protocol["data_manifest_sha256"], "tokenizer_sha256": "c" * 64,
                    "cells": {task: {str(length): {"rows": 20, "selected_rows": 20,
                              "sha256": str(i + 1) * 64, "path": "/private/machine/input.jsonl"}}
                              for i, task in enumerate(summary.TASKS)}}
            item = {"result": {"status": summary.TERMINAL_STATUS, "protocol": protocol,
                               "method": method, "data": data, "results": {}},
                    "manifest": copy.deepcopy(protocol), "rows": rows,
                    "hashes": {"results_sha256": "d" * 64, "examples_sha256": "e" * 64,
                               "run_manifest_sha256": "f" * 64}}
            refresh_aggregates(item)
            panel[arm][length] = item
    return panel


def refresh_aggregates(item):
    length = item["manifest"]["lengths"][0]
    cells = {}
    for task in summary.TASKS:
        scores = [row["official_task_score"] for row in item["rows"] if row["task"] == task]
        cells[task] = {str(length): {"rows": len(scores), "official_task_score": sum(scores) / len(scores)}}
    item["result"]["results"] = {"examples": len(item["rows"]),
        "examples_sha256": item["hashes"]["examples_sha256"], "cells": cells,
        "macro_official_task_score": sum(c[str(length)]["official_task_score"] for c in cells.values()) / 4}


@pytest.mark.parametrize("model,arm_count,contrast_count", [("k32", 4, 3), ("k64", 3, 2)])
def test_fixed_panels_preserve_different_manifests_and_no_private_paths(model, arm_count, contrast_count):
    result = summary.summarize_panel(fixture_panel(model), model)
    assert len(result["arms"]) == arm_count
    assert result["task_vector_order"] == ["single1", "mk2", "mk3", "vt"]
    assert result["bootstrap"]["seed"] == 202609025
    assert result["bootstrap"]["replicates"] == 10000
    assert result["resolver"]["status"] == "PASS"
    assert result["profile_selection_performed"] is False
    assert "/private/" not in json.dumps(result)
    assert len(result["paired_comparisons"]["65536"]) == contrast_count
    assert len({identity["32768"]["data_manifest_sha256"] for identity in result["arm_identities"].values()}) == arm_count
    if model == "k64":
        assert "normalized_index" not in result["arms"]
        assert all("index" not in key for key in result["paired_comparisons"]["32768"])
    for cell in result["paired_comparisons"].values():
        for contrast in cell.values():
            lo, hi = contrast["paired_stratified_ci95"]
            blo, bhi = contrast["bonferroni_two_lengths_sensitivity_ci95"]
            assert blo <= lo <= hi <= bhi


def test_bootstrap_is_paired_and_deterministic():
    values = np.tile(np.arange(20)[None, :, None] / 20, (4, 1, 3))
    first = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    second = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], first[:, 2])
    assert np.std(first[:, 0]) > 0


def test_known_constant_contrast_has_exact_intervals():
    panel = fixture_panel("k64")
    for arm in panel:
        for item in panel[arm].values():
            for row in item["rows"]:
                row["official_task_score"] = .5 if arm != "c2" else .75
            refresh_aggregates(item)
    result = summary.summarize_panel(panel, "k64")
    contrast = result["paired_comparisons"]["65536"]["c2_minus_native"]
    assert contrast["delta"] == .25
    assert contrast["paired_stratified_ci95"] == [.25, .25]


def test_retention_margin_is_inclusive_and_separate_from_resolver():
    panel = fixture_panel("k64")
    for arm, value in (("native", .8), ("c2", .7)):
        item = panel[arm][32768]
        for row in item["rows"]:
            row["official_task_score"] = value
        refresh_aggregates(item)
    result = summary.summarize_panel(panel, "k64")
    assert result["native_reference_retention"]["arms"]["c2"]["ratio"] == pytest.approx(.875)
    assert result["native_reference_retention"]["arms"]["c2"]["status"] == "PASS"
    item = panel["official_yarn"][65536]
    for row in item["rows"]:
        row["official_task_score"] = 0.0
    refresh_aggregates(item)
    result = summary.summarize_panel(panel, "k64")
    assert result["resolver"]["status"] == "STOP_INTERPRETATION"
    assert result["native_reference_retention"]["arms"]["c2"]["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["status", "runner_manifest", "checkpoint", "native", "tokenizer",
    "input_cell", "references", "official_metric", "duplicate", "missing_row", "index", "nonfinite",
    "aggregate", "count", "data_manifest", "active_drift", "factor", "examples_hash"])
def test_unmatched_or_incomplete_panels_fail_closed(mutation):
    panel = fixture_panel("k32")
    item = panel["physical_x"][65536]
    result, protocol = item["result"], item["result"]["protocol"]
    if mutation == "status": result["status"] = "PARTIAL"
    elif mutation == "runner_manifest": item["manifest"] = {}
    elif mutation == "checkpoint": protocol["checkpoint_sha256"] = "f" * 64
    elif mutation == "native": result["method"]["native_sha256_float32"] = "f" * 64
    elif mutation == "tokenizer": result["data"]["tokenizer_sha256"] = "f" * 64
    elif mutation == "input_cell": result["data"]["cells"]["vt"]["65536"]["sha256"] = "f" * 64
    elif mutation == "references": item["rows"][0]["references"] = ["different answer"]
    elif mutation == "official_metric": item["rows"][0]["official_metric"] = "string_match_part"
    elif mutation == "duplicate": item["rows"].append(copy.deepcopy(item["rows"][0]))
    elif mutation == "missing_row": item["rows"].pop()
    elif mutation == "index": item["rows"][0]["local_index"] = 20
    elif mutation == "nonfinite": item["rows"][0]["official_task_score"] = float("nan")
    elif mutation == "aggregate": result["results"]["macro_official_task_score"] = .123
    elif mutation == "count": protocol["limit_per_cell"] = 19
    elif mutation == "data_manifest": result["data"]["manifest_sha256"] = "f" * 64
    elif mutation == "active_drift":
        protocol["table_sha256_float32"] = "f" * 64
        result["method"]["active_sha256_float32"] = "f" * 64
    elif mutation == "factor": protocol["table_factor"] = 4.0
    elif mutation == "examples_hash": result["results"]["examples_sha256"] = "f" * 64
    if mutation != "runner_manifest": item["manifest"] = copy.deepcopy(protocol)
    with pytest.raises(ValueError): summary.summarize_panel(panel, "k32")


def test_k64_cannot_accept_a_fabricated_index_arm():
    panel = fixture_panel("k64")
    panel["normalized_index"] = copy.deepcopy(panel["c2"])
    with pytest.raises(ValueError, match="exactly its registered arms"):
        summary.summarize_panel(panel, "k64")


@pytest.mark.parametrize("arm", ["physical_x", "normalized_index", "official_yarn"])
def test_registered_amplitudes_are_not_free_parameters(arm):
    panel = fixture_panel("k32")
    for item in panel[arm].values():
        item["result"]["method"]["long_attention_scaling"] = 1.2
        item["result"]["protocol"]["long_attention_scaling"] = 1.2
        item["manifest"]["long_attention_scaling"] = 1.2
    with pytest.raises(ValueError, match="fixed s2 gain"):
        summary.summarize_panel(panel, "k32")


def test_checkpoint_bound_manifests_can_differ_across_lengths():
    panel = fixture_panel("k64")
    for item in (cells[65536] for cells in panel.values()):
        item["result"]["protocol"]["data_manifest_sha256"] = "9" * 64
        item["manifest"]["data_manifest_sha256"] = "9" * 64
        item["result"]["data"]["manifest_sha256"] = "9" * 64
    _, identities = summary.validate_panel(panel, "k64")
    assert identities["native"]["32768"]["data_manifest_sha256"] != identities["native"]["65536"]["data_manifest_sha256"]
    assert identities["native"]["32768"]["table_factor"] is None


def test_loader_requires_terminal_examples_hash_and_fixed_directories(tmp_path):
    model = "k64"
    panel = fixture_panel(model)
    for arm, cells in summary.directories(model).items():
        for length, relative in cells.items():
            directory = tmp_path / relative
            directory.mkdir(parents=True)
            item = panel[arm][length]
            examples = directory / "examples.jsonl"
            examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
            item["result"]["results"]["examples_sha256"] = summary.file_hash(examples)
            (directory / "results.json").write_text(json.dumps(item["result"]))
            (directory / "run_manifest.json").write_text(json.dumps(item["manifest"]))
    loaded = summary.load_panel(tmp_path, model)
    summary.validate_panel(loaded, model)
    examples = tmp_path / summary.directories(model)["native"][32768] / "examples.jsonl"
    with examples.open("a") as handle: handle.write("\n")
    with pytest.raises(ValueError, match="examples hash mismatch"):
        summary.load_panel(tmp_path, model)
