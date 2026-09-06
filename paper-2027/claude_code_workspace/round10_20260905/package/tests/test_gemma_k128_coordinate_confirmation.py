"""Synthetic tests for the fixed Gemma K128 physical/index confirmation."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.analysis import summarize_gemma_k128_coordinate_confirmation as summary


def refresh(item):
    cells = {task: {} for task in summary.TASKS}
    for task in summary.TASKS:
        values = [row["official_task_score"] for row in item["rows"] if row["task"] == task]
        cells[task][str(summary.LENGTH)] = {
            "rows": len(values), "official_task_score": sum(values) / len(values)}
    item["result"]["results"] = {
        "examples": len(item["rows"]), "examples_sha256": item["raw_hashes"]["examples_sha256"],
        "cells": cells,
        "macro_official_task_score": sum(value["official_task_score"]
            for task_cells in cells.values() for value in task_cells.values()) / len(summary.TASKS),
    }


def fixture_panel(physical_passes=32, index_passes=48):
    panel = {}
    for arm, passes in (("physical_x", physical_passes), ("normalized_index", index_passes)):
        protocol = {
            "status": summary.FROZEN_STATUS, "tasks": list(summary.TASKS),
            "lengths": [summary.LENGTH], "limit_per_cell": summary.ROWS_PER_TASK,
            "model_type": "gemma", "native_context_length": 8192,
            "profile_target_length": summary.LENGTH, "checkpoint_sha256": summary.WEIGHT_SHA256,
            "script_sha256": "a" * 64, "data_manifest_sha256": "b" * 64,
            "expected_data_manifest_sha256": "b" * 64, "method": "external_table_static",
            "table_factor": 4.0, "expected_native_sha256": summary.NATIVE_SHA256,
            "expected_active_sha256": summary.TABLE_SHA256[arm],
            "table_sha256_float32": summary.TABLE_SHA256[arm], "table_file_sha256": "c" * 64,
            "table_support": "native_div_factor", "long_attention_scaling": summary.GAIN,
        }
        method = {
            "method": "external_table_static", "model_type": "gemma", "native_context_length": 8192,
            "profile_target_length": summary.LENGTH, "native_sha256_float32": summary.NATIVE_SHA256,
            "active_sha256_float32": summary.TABLE_SHA256[arm], "table_factor": 4.0,
            "table_support": "native_div_factor", "initial_branch": "external_long",
            "long_attention_scaling": summary.GAIN,
        }
        rows = []
        for task in summary.TASKS:
            for index in range(summary.ROWS_PER_TASK):
                passed = index < passes
                budget = summary.GENERATION_BUDGET[task]
                rows.append({
                    "task": task, "nominal_length": summary.LENGTH, "local_index": index,
                    "official_metric": "string_match_all", "official_task_score": float(passed),
                    "reference_recall": float(passed), "references": [f"answer-{task}-{index}"],
                    "prediction": f"answer-{task}-{index}" if passed else "complete synthetic miss",
                    "generated_token_ids": [123, summary.EOS_TOKEN_ID], "ended_with_eos": True,
                    "prompt_tokens": summary.LENGTH - budget, "generation_budget": budget,
                })
        data = {
            "seed": summary.DATA_SEED, "ruler_commit": summary.RULER_COMMIT,
            "manifest_sha256": "b" * 64, "tokenizer_sha256": "d" * 64,
            "cells": {task: {str(summary.LENGTH): {
                "rows": summary.ROWS_PER_TASK, "selected_rows": summary.ROWS_PER_TASK,
                "sha256": str(task_index + 1) * 64, "path": "/private/input.jsonl"}}
                for task_index, task in enumerate(summary.TASKS)},
        }
        item = {
            "result": {"status": summary.TERMINAL_STATUS, "protocol": copy.deepcopy(protocol),
                       "method": method, "data": data,
                       "runtime": {"torch": "2.8.0+cu128", "cuda": "12.8"}, "results": {}},
            "manifest": protocol, "rows": rows,
            "raw_hashes": {"results_sha256": "e" * 64, "examples_sha256": "f" * 64,
                           "run_manifest_sha256": "1" * 64},
        }
        refresh(item)
        panel[arm] = item
    return panel


def test_complete_n80_panel_reports_vectors_macro_and_above_zero():
    panel = fixture_panel()
    original = copy.deepcopy(panel)
    result = summary.summarize_panel(panel)
    assert result["decision"] == "ABOVE_ZERO"
    assert result["rows_per_arm"] == 320
    assert result["scores"]["physical_x"]["task_vector"] == [.4] * 4
    assert result["scores"]["normalized_index"]["task_vector"] == [.6] * 4
    assert result["primary_index_minus_physical"]["delta"] == pytest.approx(.2)
    assert result["primary_index_minus_physical"]["paired_task_stratified_ci95"][0] > 0
    assert result["bootstrap"]["replicates"] == 10000
    assert result["bootstrap"]["seed"] == 202609029
    assert result["old_pilot_pooled"] is False
    assert "/private/" not in json.dumps(result)
    assert panel == original


@pytest.mark.parametrize("physical,index,expected", [
    (48, 32, "BELOW_ZERO"), (40, 40, "UNRESOLVED"),
])
def test_summary_realizes_other_registered_decisions(physical, index, expected):
    assert summary.summarize_panel(fixture_panel(physical, index))["decision"] == expected


@pytest.mark.parametrize("interval,expected", [
    ([.01, .2], "ABOVE_ZERO"), ([-.2, -.01], "BELOW_ZERO"),
    ([-.2, 0], "UNRESOLVED"), ([0, .2], "UNRESOLVED"), ([-.1, .1], "UNRESOLVED"),
])
def test_decision_rule(interval, expected):
    assert summary.decision(interval) == expected


def test_paired_task_stratified_bootstrap_is_deterministic():
    values = np.tile(np.arange(80)[None, :, None] / 80, (4, 1, 2))
    values[:, :, 1] += .1
    one = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    two = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    np.testing.assert_array_equal(one, two)
    np.testing.assert_allclose(one[:, 1] - one[:, 0], .1)


def test_shared_zero_collapse_is_invalid_not_a_ranking():
    with pytest.raises(ValueError, match="collapsed"):
        summary.summarize_panel(fixture_panel(0, 0))


@pytest.mark.parametrize("mutation", [
    "partial", "seed", "length", "count", "weight", "native", "active", "gain", "factor",
    "data_hash", "tokenizer", "input_cell", "runner", "references", "metric", "prediction",
    "tokens", "token_value", "eos", "eos_value", "termination", "prompt", "budget", "duplicate", "missing",
    "nonfinite", "score", "recall", "aggregate", "example_hash", "runtime",
])
def test_fail_closed_on_identity_or_full_row_drift(mutation):
    panel = fixture_panel()
    item = panel["physical_x"]
    result, protocol, row = item["result"], item["result"]["protocol"], item["rows"][0]
    if mutation == "partial": result["status"] = "PARTIAL"
    elif mutation == "seed": result["data"]["seed"] = 20260822
    elif mutation == "length": protocol["lengths"] = [8192]
    elif mutation == "count": protocol["limit_per_cell"] = 20
    elif mutation == "weight": protocol["checkpoint_sha256"] = "0" * 64
    elif mutation == "native": result["method"]["native_sha256_float32"] = "0" * 64
    elif mutation == "active": result["method"]["active_sha256_float32"] = "0" * 64
    elif mutation == "gain": result["method"]["long_attention_scaling"] = 1.0
    elif mutation == "factor": result["method"]["table_factor"] = 2.0
    elif mutation == "data_hash":
        protocol["data_manifest_sha256"] = protocol["expected_data_manifest_sha256"] = "0" * 64
        result["data"]["manifest_sha256"] = "0" * 64
    elif mutation == "tokenizer": result["data"]["tokenizer_sha256"] = "0" * 64
    elif mutation == "input_cell": result["data"]["cells"][summary.TASKS[0]][str(summary.LENGTH)]["sha256"] = "0" * 64
    elif mutation == "runner": protocol["script_sha256"] = "0" * 64
    elif mutation == "references": row["references"] = ["different"]
    elif mutation == "metric": row["official_metric"] = "string_match_part"
    elif mutation == "prediction": row.pop("prediction")
    elif mutation == "tokens": row.pop("generated_token_ids")
    elif mutation == "token_value": row["generated_token_ids"] = [1.5]
    elif mutation == "eos": row.pop("ended_with_eos")
    elif mutation == "eos_value": row["generated_token_ids"] = [123, 2]
    elif mutation == "termination":
        row["ended_with_eos"] = False
        row["generated_token_ids"] = [1]
    elif mutation == "prompt": row["prompt_tokens"] = 0
    elif mutation == "budget": row["generation_budget"] = 0
    elif mutation == "duplicate": item["rows"].append(copy.deepcopy(row))
    elif mutation == "missing": item["rows"].pop()
    elif mutation == "nonfinite": row["official_task_score"] = float("nan")
    elif mutation == "score": row["official_task_score"] = 0.0
    elif mutation == "recall": row["reference_recall"] = 0.0
    elif mutation == "aggregate": result["results"]["macro_official_task_score"] = .123
    elif mutation == "example_hash": result["results"]["examples_sha256"] = "0" * 64
    elif mutation == "runtime": result["runtime"]["torch"] = "different"
    item["manifest"] = copy.deepcopy(protocol)
    with pytest.raises((ValueError, KeyError)):
        summary.validate_panel(panel)


def test_loader_requires_terminal_hash_bound_files(tmp_path):
    panel = fixture_panel()
    for arm, item in panel.items():
        directory = tmp_path / arm
        directory.mkdir()
        examples = directory / "examples.jsonl"
        examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
        item["result"]["results"]["examples_sha256"] = summary.file_hash(examples)
        (directory / "results.json").write_text(json.dumps(item["result"]))
        (directory / "run_manifest.json").write_text(json.dumps(item["manifest"]))
    loaded = summary.load_panel(tmp_path)
    summary.validate_panel(loaded)
    with (tmp_path / "physical_x" / "examples.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        summary.load_panel(tmp_path)


def test_cli_writes_invalid_status_without_partial_summary(tmp_path):
    output = tmp_path / "summary.json"
    assert summary.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 2
    result = json.loads(output.read_text())
    assert result["status"] == "INVALID_OR_INCOMPLETE_PANEL"
    assert "decision" not in result
