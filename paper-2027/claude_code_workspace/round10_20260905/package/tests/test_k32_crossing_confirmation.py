"""Synthetic preregistered crossing panels; no real scores used for rule selection."""

import copy
import json

import numpy as np
import pytest

from scripts.analysis import summarize_k32_crossing_confirmation as summary


def fixture_panel():
    panel = {}
    for arm in summary.ARMS:
        native = arm == "native"
        protocol = {
            "tasks": list(summary.TASKS), "lengths": list(summary.LENGTHS), "limit_per_cell": 80,
            "model_type": "qwen2", "native_context_length": 32768, "profile_target_length": 65536,
            "checkpoint_sha256": summary.WEIGHT_SHA256, "script_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64, "expected_data_manifest_sha256": "b" * 64,
            "method": "native" if native else "external_table_static", "table_factor": 2.0,
            "expected_native_sha256": summary.TENSOR_SHA256["native"],
            "expected_active_sha256": None if native else summary.TENSOR_SHA256[arm],
            "table_sha256_float32": None if native else summary.TENSOR_SHA256[arm],
            "table_file_sha256": None if native else "c" * 64,
            "table_support": None if native else "native_div_factor",
            "long_attention_scaling": None if native else summary.GAIN,
        }
        method = {"method": protocol["method"], "model_type": "qwen2", "native_context_length": 32768,
                  "active_sha256_float32": summary.TENSOR_SHA256[arm]}
        if native:
            method["attention_scaling"] = 1.0
        else:
            method.update(native_sha256_float32=summary.TENSOR_SHA256["native"], table_factor=2.0,
                          long_attention_scaling=summary.GAIN)
        rows = []
        for task in summary.TASKS:
            for length in summary.LENGTHS:
                value = .5 if native or arm == "normalized_index" else (.25 if length == 32768 else .75)
                rows.extend({"task": task, "nominal_length": length, "local_index": i,
                             "official_metric": "string_match_all", "official_task_score": value,
                             "references": [f"{task}-{length}-{i}"], "prediction": "full synthetic prediction",
                             "generated_token_ids": [123, 151645], "ended_with_eos": True,
                             "prompt_tokens": length - 128, "generation_budget": 128} for i in range(80))
        data = {"seed": summary.DATA_SEED, "ruler_commit": summary.RULER_COMMIT,
                "manifest_sha256": "b" * 64, "tokenizer_sha256": "d" * 64,
                "cells": {task: {str(length): {"rows": 80, "selected_rows": 80,
                               "sha256": str(i + 1) * 64, "path": "/private/machine/input.jsonl"}
                                  for length in summary.LENGTHS} for i, task in enumerate(summary.TASKS)}}
        item = {"result": {"status": summary.TERMINAL_STATUS, "protocol": protocol,
                           "method": method, "data": data, "results": {}},
                "manifest": copy.deepcopy(protocol), "rows": rows,
                "hashes": {"results_sha256": "e" * 64, "examples_sha256": "f" * 64,
                           "run_manifest_sha256": "1" * 64}}
        refresh(item)
        panel[arm] = item
    return panel


def refresh(item):
    cells = {task: {} for task in summary.TASKS}
    for task in summary.TASKS:
        for length in summary.LENGTHS:
            values = [r["official_task_score"] for r in item["rows"] if r["task"] == task and r["nominal_length"] == length]
            cells[task][str(length)] = {"rows": len(values), "official_task_score": sum(values) / len(values)}
    item["result"]["results"] = {"examples": len(item["rows"]), "examples_sha256": item["hashes"]["examples_sha256"],
        "cells": cells, "macro_official_task_score": sum(cell["official_task_score"] for lengths in cells.values() for cell in lengths.values()) / 8}


def test_three_arm_two_length_confirmation_and_portable_identity():
    panel = fixture_panel()
    original = copy.deepcopy(panel)
    result = summary.summarize_panel(panel)
    assert result["decision"] == "CONFIRMED_CROSSING"
    assert result["rows_per_arm"] == 640
    assert result["arms"] == ["native", "physical_x", "normalized_index"]
    assert result["data_seed"] == 202609026
    assert result["bootstrap"]["seed"] == 202609027
    assert result["bootstrap"]["replicates"] == 10000
    assert result["bootstrap"]["primary_marginal_confidence"] == .975
    assert result["primary_physical_minus_index"]["32768"]["paired_stratified_ci975"] == [-.25, -.25]
    assert result["primary_physical_minus_index"]["65536"]["paired_stratified_ci975"] == [.25, .25]
    assert result["old_pilot_pooled"] is False
    assert "/private/" not in json.dumps(result)
    assert panel == original  # Never split/rewrite manifests to fit another summarizer.
    assert result["native_retention"]["arms"]["physical_x"]["status"] == "FAIL"


@pytest.mark.parametrize("native,short,long,expected", [
    (.5, [-.2, -.1], [.1, .2], "CONFIRMED_CROSSING"),
    (.5, [.1, .2], [-.2, -.1], "REVERSED"),
    (.5, [-.2, 0], [.1, .2], "UNRESOLVED"),
    (.5, [-.2, -.1], [0, .2], "UNRESOLVED"),
    (.5, [.1, .2], [.1, .2], "UNRESOLVED"),
    (.5, [-.2, -.1], [-.2, -.1], "UNRESOLVED"),
    (0, [-.2, -.1], [.1, .2], "STOP_INSTRUMENT"),
])
def test_conservative_decision_requires_both_corrected_signs(native, short, long, expected):
    assert summary.crossing_decision(native, short, long) == expected


def test_native_zero_overrides_crossing_and_retention_is_undefined():
    panel = fixture_panel()
    for row in panel["native"]["rows"]:
        if row["nominal_length"] == 32768:
            row["official_task_score"] = 0.0
    refresh(panel["native"])
    result = summary.summarize_panel(panel)
    assert result["decision"] == "STOP_INSTRUMENT"
    assert result["native_retention"]["arms"]["physical_x"]["status"] == "UNDEFINED"


def test_paired_bootstrap_keeps_identical_arms_identical_and_repeats_seed():
    values = np.tile(np.arange(80)[None, :, None] / 80, (4, 1, 3))
    first = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    second = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 1], first[:, 2])
    assert np.std(first[:, 0]) > 0


@pytest.mark.parametrize("mutation", ["partial", "fourth_arm", "pilot_seed", "count", "one_length", "weight",
    "active", "native", "factor", "gain", "data_hash", "tokenizer", "input_cell", "runner", "references",
    "metric", "tokens", "eos_field", "prompt_tokens", "duplicate", "missing_row", "nonfinite", "aggregate", "example_hash"])
def test_fail_closed_without_mixing_pilot_or_mutating_protocol(mutation):
    panel = fixture_panel()
    item = panel["physical_x"]
    result, protocol = item["result"], item["result"]["protocol"]
    if mutation == "partial": result["status"] = "PARTIAL"
    elif mutation == "fourth_arm": panel["official_yarn"] = copy.deepcopy(item)
    elif mutation == "pilot_seed": result["data"]["seed"] = 20260822
    elif mutation == "count": protocol["limit_per_cell"] = 20
    elif mutation == "one_length": protocol["lengths"] = [32768]
    elif mutation == "weight": protocol["checkpoint_sha256"] = "0" * 64
    elif mutation == "active": result["method"]["active_sha256_float32"] = "0" * 64
    elif mutation == "native": protocol["expected_native_sha256"] = "0" * 64
    elif mutation == "factor": protocol["table_factor"] = 4.0
    elif mutation == "gain":
        protocol["long_attention_scaling"] = 1.2
        result["method"]["long_attention_scaling"] = 1.2
    elif mutation == "data_hash":
        protocol["data_manifest_sha256"] = protocol["expected_data_manifest_sha256"] = "0" * 64
        result["data"]["manifest_sha256"] = "0" * 64
    elif mutation == "tokenizer": result["data"]["tokenizer_sha256"] = "0" * 64
    elif mutation == "input_cell": result["data"]["cells"]["vt"]["65536"]["sha256"] = "0" * 64
    elif mutation == "runner": protocol["script_sha256"] = "0" * 64
    elif mutation == "references": item["rows"][0]["references"] = ["other"]
    elif mutation == "metric": item["rows"][0]["official_metric"] = "string_match_part"
    elif mutation == "tokens": item["rows"][0].pop("generated_token_ids")
    elif mutation == "eos_field": item["rows"][0].pop("ended_with_eos")
    elif mutation == "prompt_tokens": item["rows"][0]["prompt_tokens"] -= 1
    elif mutation == "duplicate": item["rows"].append(copy.deepcopy(item["rows"][0]))
    elif mutation == "missing_row": item["rows"].pop()
    elif mutation == "nonfinite": item["rows"][0]["official_task_score"] = float("nan")
    elif mutation == "aggregate": result["results"]["macro_official_task_score"] = .123
    elif mutation == "example_hash": result["results"]["examples_sha256"] = "0" * 64
    item["manifest"] = copy.deepcopy(protocol)
    with pytest.raises(ValueError):
        summary.validate_panel(panel)


def test_loader_requires_complete_hashed_combined_files(tmp_path):
    panel = fixture_panel()
    for arm, item in panel.items():
        directory = tmp_path / arm
        directory.mkdir()
        examples = directory / "examples.jsonl"
        examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
        item["result"]["results"]["examples_sha256"] = summary.common.file_hash(examples)
        (directory / "results.json").write_text(json.dumps(item["result"]))
        (directory / "run_manifest.json").write_text(json.dumps(item["manifest"]))
    loaded = summary.load_panel(tmp_path)
    summary.validate_panel(loaded)
    assert loaded["native"]["manifest"]["lengths"] == [32768, 65536]
    with (tmp_path / "native" / "examples.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="examples SHA mismatch"):
        summary.load_panel(tmp_path)
