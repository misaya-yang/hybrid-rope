"""Synthetic new-seed full-13 panels only; no GPU or real outcomes."""

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.analysis import summarize_k32_full13_confirmation as summary


def refresh(item):
    cells, means = {task: {} for task in summary.TASKS}, []
    for task in summary.TASKS:
        for length in summary.LENGTHS:
            values = [row["official_task_score"] for row in item["rows"]
                      if row["task"] == task and row["nominal_length"] == length]
            mean = sum(values) / len(values)
            cells[task][str(length)] = {"rows": len(values), "official_task_score": mean}
            means.append(mean)
    item["result"]["results"] = {"examples": len(item["rows"]),
        "examples_sha256": item["hashes"]["examples_sha256"], "cells": cells,
        "macro_official_task_score": sum(means) / len(means)}


def fixture_panel():
    panel = {}
    for arm in summary.ARMS:
        native = arm == "native"
        protocol = {"tasks": list(summary.TASKS), "lengths": list(summary.LENGTHS),
            "limit_per_cell": 20, "model_type": "qwen2", "native_context_length": 32768,
            "profile_target_length": 65536, "checkpoint_sha256": summary.WEIGHT_SHA256,
            "script_sha256": "a" * 64, "data_manifest_sha256": "b" * 64,
            "expected_data_manifest_sha256": "b" * 64, "method": "native" if native else "external_table_static",
            "table_factor": 2.0, "expected_native_sha256": summary.TENSOR_SHA256["native"],
            "expected_active_sha256": None if native else summary.TENSOR_SHA256[arm],
            "table_sha256_float32": None if native else summary.TENSOR_SHA256[arm],
            "table_file_sha256": None if native else "c" * 64,
            "table_support": None if native else "native_div_factor",
            "long_attention_scaling": None if native else summary.GAIN[arm]}
        method = {"method": protocol["method"], "model_type": "qwen2", "native_context_length": 32768,
                  "active_sha256_float32": summary.TENSOR_SHA256[arm]}
        if native:
            method["attention_scaling"] = 1.0
        else:
            method.update(native_sha256_float32=summary.TENSOR_SHA256["native"], table_factor=2.0,
                          long_attention_scaling=summary.GAIN[arm])
        rows = []
        for task in summary.TASKS:
            for length in summary.LENGTHS:
                value = {"native": .8 if length == 32768 else .2,
                         "normalized_index": .75 if length == 32768 else .6,
                         "official_yarn": .7 if length == 32768 else .4}[arm]
                rows.extend({"task": task, "nominal_length": length, "local_index": i,
                    "official_metric": summary.METRICS[task], "official_task_score": value,
                    "references": [f"{task}-{length}-{i}"], "prediction": "full synthetic output",
                    "generated_token_ids": [123, 151645], "ended_with_eos": True,
                    "prompt_tokens": length - 128, "generation_budget": 128} for i in range(20))
        data = {"seed": summary.DATA_SEED, "ruler_commit": summary.RULER_COMMIT,
            "manifest_sha256": "b" * 64, "tokenizer_sha256": "d" * 64,
            "cells": {task: {str(length): {"rows": 20, "selected_rows": 20,
                "sha256": hashlib.sha256(task.encode()).hexdigest(), "path": "/private/input.jsonl"}
                for length in summary.LENGTHS} for task_index, task in enumerate(summary.TASKS)}}
        item = {"result": {"status": summary.TERMINAL_STATUS, "protocol": protocol,
                           "method": method, "data": data, "results": {}},
                "manifest": copy.deepcopy(protocol), "rows": rows,
                "hashes": {"results_sha256": "e" * 64, "examples_sha256": "f" * 64,
                           "run_manifest_sha256": "1" * 64}}
        refresh(item)
        panel[arm] = item
    return panel


def test_full13_clear_advance_scores_metrics_retention_and_ci():
    result = summary.summarize_panel(fixture_panel())
    assert result["decision"] == "CLEAR_ADVANCE"
    assert result["rows_per_arm"] == 520 and len(result["tasks"]) == 13
    assert result["official_metrics"]["qa_1"] == "string_match_part"
    assert result["official_metrics"]["fwe"] == "string_match_all"
    assert result["scores"]["32768"]["normalized_index"]["macro"] == pytest.approx(.75)
    assert result["native_retention"]["ratio"] == pytest.approx(.9375)
    assert result["paired_comparisons"]["65536"]["normalized_index_minus_native"]["paired_stratified_ci95"] == pytest.approx([.4, .4])
    assert result["paired_comparisons"]["65536"]["normalized_index_minus_official_yarn"]["paired_stratified_ci95"] == pytest.approx([.2, .2])
    assert result["data_seed"] == 202609027


@pytest.mark.parametrize("retention,native_ci,yarn_ci,expected", [
    (True, [.1, .2], [.01, .2], "CLEAR_ADVANCE"),
    (True, [.1, .2], [-.1, .1], "COMPETITIVE_UNRESOLVED"),
    (False, [.1, .2], [.01, .2], "BASELINE_LOSS"),
    (True, [-.1, .1], [.01, .2], "BASELINE_LOSS"),
    (True, [.1, .2], [-.2, -.01], "BASELINE_LOSS"),
])
def test_preregistered_decision_categories(retention, native_ci, yarn_ci, expected):
    assert summary.decision(retention, native_ci, yarn_ci) == expected


def test_competitive_panel_keeps_exact_decision_and_is_reproducible():
    panel = fixture_panel()
    for row in panel["official_yarn"]["rows"]:
        if row["nominal_length"] == 65536:
            row["official_task_score"] = .6
    refresh(panel["official_yarn"])
    first = summary.summarize_panel(panel)
    second = summary.summarize_panel(panel)
    assert first["decision"] == "COMPETITIVE_UNRESOLVED"
    assert first["paired_comparisons"] == second["paired_comparisons"]
    assert first["bootstrap"]["replicates"] == 10000
    assert first["profile_selection_performed"] is False


@pytest.mark.parametrize("mutation", ["arm", "partial", "task", "length", "count", "seed", "commit", "weight",
    "native", "active", "gain", "factor", "data", "cell", "tokenizer", "runner", "metric", "qa_metric",
    "references", "prediction", "tokens", "eos", "prompt", "duplicate", "missing", "nonfinite", "aggregate", "hash"])
def test_identity_metric_and_pairing_fail_closed(mutation):
    panel = fixture_panel()
    item = panel["normalized_index"]
    result, protocol, data = item["result"], item["result"]["protocol"], item["result"]["data"]
    if mutation == "arm": panel["other"] = copy.deepcopy(item)
    elif mutation == "partial": result["status"] = "PARTIAL"
    elif mutation == "task": protocol["tasks"] = protocol["tasks"][:-1]
    elif mutation == "length": protocol["lengths"] = [32768]
    elif mutation == "count": protocol["limit_per_cell"] = 19
    elif mutation == "seed": data["seed"] = 202609026
    elif mutation == "commit": data["ruler_commit"] = "0" * 40
    elif mutation == "weight": protocol["checkpoint_sha256"] = "0" * 64
    elif mutation == "native": protocol["expected_native_sha256"] = "0" * 64
    elif mutation == "active": result["method"]["active_sha256_float32"] = "0" * 64
    elif mutation == "gain":
        protocol["long_attention_scaling"] = result["method"]["long_attention_scaling"] = 1.2
    elif mutation == "factor": protocol["table_factor"] = 4
    elif mutation == "data": data["manifest_sha256"] = protocol["data_manifest_sha256"] = protocol["expected_data_manifest_sha256"] = "0" * 64
    elif mutation == "cell": data["cells"]["qa_2"]["65536"]["sha256"] = "0" * 64
    elif mutation == "tokenizer": data["tokenizer_sha256"] = "0" * 64
    elif mutation == "runner": protocol["script_sha256"] = "0" * 64
    elif mutation == "metric": item["rows"][0]["official_metric"] = "string_match_part"
    elif mutation == "qa_metric": next(row for row in item["rows"] if row["task"] == "qa_1")["official_metric"] = "string_match_all"
    elif mutation == "references": item["rows"][0]["references"] = ["other"]
    elif mutation == "prediction": item["rows"][0].pop("prediction")
    elif mutation == "tokens": item["rows"][0].pop("generated_token_ids")
    elif mutation == "eos": item["rows"][0].pop("ended_with_eos")
    elif mutation == "prompt": item["rows"][0]["prompt_tokens"] -= 1
    elif mutation == "duplicate": item["rows"].append(copy.deepcopy(item["rows"][0]))
    elif mutation == "missing": item["rows"].pop()
    elif mutation == "nonfinite": item["rows"][0]["official_task_score"] = float("nan")
    elif mutation == "aggregate": result["results"]["macro_official_task_score"] = .123
    else: result["results"]["examples_sha256"] = "0" * 64
    item["manifest"] = copy.deepcopy(protocol)
    with pytest.raises(ValueError):
        summary.summarize_panel(panel)


def test_loader_hashes_only_three_named_arms_and_output_is_portable(tmp_path):
    panel = fixture_panel()
    for arm, item in panel.items():
        directory = tmp_path / arm
        directory.mkdir()
        examples = directory / "examples.jsonl"
        examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
        item["result"]["results"]["examples_sha256"] = summary.common.file_hash(examples)
        (directory / "results.json").write_text(json.dumps(item["result"]))
        (directory / "run_manifest.json").write_text(json.dumps(item["manifest"]))
    forbidden = tmp_path / "old_pilot"
    forbidden.mkdir()
    (forbidden / "results.json").write_text("malformed and ignored")
    loaded = summary.load_panel(tmp_path)
    result = summary.summarize_panel(loaded)
    assert set(loaded) == set(summary.ARMS)
    assert "/private/" not in json.dumps(result)
    with (tmp_path / "native" / "examples.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="examples SHA mismatch"):
        summary.load_panel(tmp_path)


def test_bootstrap_preserves_all_arm_pairings_within_tasks():
    values = np.tile(np.arange(20)[None, :, None] / 20, (13, 1, 3))
    draws = summary.paired_bootstrap(values, np.random.default_rng(summary.BOOTSTRAP_SEED))
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_array_equal(draws[:, 1], draws[:, 2])
    assert np.std(draws[:, 0]) > 0
