"""Synthetic fresh-N80 completion panels only; no pilot or model outputs."""

import copy
import json
import math

import numpy as np
import pytest

from scripts.analysis import summarize_k32_crossing_confirmation as crossing
from scripts.analysis import summarize_k32_fresh_yarn_completion as summary
from tests.test_k32_crossing_confirmation import fixture_panel, refresh


def four_arm_panel():
    panel = fixture_panel()
    yarn = copy.deepcopy(panel["normalized_index"])
    protocol = yarn["result"]["protocol"]
    protocol.update(expected_active_sha256=summary.YARN_TENSOR_SHA256,
                    table_sha256_float32=summary.YARN_TENSOR_SHA256,
                    long_attention_scaling=summary.YARN_GAIN,
                    table_file_sha256="9" * 64)
    method = yarn["result"]["method"]
    method.update(active_sha256_float32=summary.YARN_TENSOR_SHA256,
                  long_attention_scaling=summary.YARN_GAIN)
    yarn["manifest"] = copy.deepcopy(protocol)
    for row in yarn["rows"]:
        row["official_task_score"] = .4
    refresh(yarn)
    panel["official_yarn"] = yarn
    return panel


def test_four_arm_scores_ci_retention_and_original_decision_are_separate():
    panel = four_arm_panel()
    original = crossing.summarize_panel({arm: panel[arm] for arm in crossing.ARMS})
    result = summary.summarize_panel(panel)
    assert result["three_arm_crossing_decision"] == original["decision"] == "CONFIRMED_CROSSING"
    assert result["crossing_decision_changed_by_yarn"] is False
    assert result["arms"] == list(summary.ARMS)
    assert result["scores"]["32768"]["official_yarn"]["macro"] == pytest.approx(.4)
    assert result["index_minus_yarn"]["65536"]["delta"] == pytest.approx(.1)
    assert result["index_minus_yarn"]["65536"]["paired_stratified_ci95"] == pytest.approx([.1, .1])
    assert result["long_index_minus_yarn_verdict"]["status"] == "INDEX_FAVORED"
    assert result["native_retention"]["arms"]["official_yarn"]["status"] == "FAIL"
    assert result["old_pilot_read_or_pooled"] is False
    assert result["profile_selection_performed"] is False


def test_yarn_comparison_can_be_unresolved_without_rewriting_crossing():
    panel = four_arm_panel()
    for row in panel["official_yarn"]["rows"]:
        row["official_task_score"] = .5
    refresh(panel["official_yarn"])
    result = summary.summarize_panel(panel)
    assert result["long_index_minus_yarn_verdict"]["status"] == "UNRESOLVED"
    assert result["three_arm_crossing_decision"] == "CONFIRMED_CROSSING"


def test_native_zero_makes_all_retention_undefined_and_preserves_stop():
    panel = four_arm_panel()
    for row in panel["native"]["rows"]:
        if row["nominal_length"] == 32768:
            row["official_task_score"] = 0
    refresh(panel["native"])
    result = summary.summarize_panel(panel)
    assert result["three_arm_crossing_decision"] == "STOP_INSTRUMENT"
    assert {value["status"] for value in result["native_retention"]["arms"].values()} == {"UNDEFINED"}


def test_bootstrap_reuses_paired_rows_task_stratified_and_seed():
    panel = four_arm_panel()
    first = summary.summarize_panel(panel)
    second = summary.summarize_panel(panel)
    assert first["index_minus_yarn"] == second["index_minus_yarn"]
    assert first["bootstrap"]["seed"] == crossing.BOOTSTRAP_SEED
    assert first["bootstrap"]["replicates"] == 10000


@pytest.mark.parametrize("mutation", ["arm", "partial", "tensor", "gain", "factor", "support", "weight",
    "seed", "data", "cell", "runner", "references", "prompt", "metric", "tokens", "duplicate", "missing", "aggregate", "hash"])
def test_yarn_identity_and_pairing_fail_closed(mutation):
    panel = four_arm_panel()
    yarn = panel["official_yarn"]
    protocol, method, data = yarn["result"]["protocol"], yarn["result"]["method"], yarn["result"]["data"]
    if mutation == "arm": panel["another"] = copy.deepcopy(yarn)
    elif mutation == "partial": yarn["result"]["status"] = "PARTIAL"
    elif mutation == "tensor": method["active_sha256_float32"] = "0" * 64
    elif mutation == "gain":
        protocol["long_attention_scaling"] = method["long_attention_scaling"] = 1.1
    elif mutation == "factor": protocol["table_factor"] = 4
    elif mutation == "support": protocol["table_support"] = "other"
    elif mutation == "weight": protocol["checkpoint_sha256"] = "0" * 64
    elif mutation == "seed": data["seed"] = 20260822
    elif mutation == "data": data["manifest_sha256"] = protocol["data_manifest_sha256"] = protocol["expected_data_manifest_sha256"] = "0" * 64
    elif mutation == "cell": data["cells"]["vt"]["65536"]["sha256"] = "0" * 64
    elif mutation == "runner": protocol["script_sha256"] = "0" * 64
    elif mutation == "references": yarn["rows"][0]["references"] = ["other"]
    elif mutation == "prompt": yarn["rows"][0]["prompt_tokens"] -= 1
    elif mutation == "metric": yarn["rows"][0]["official_metric"] = "proxy"
    elif mutation == "tokens": yarn["rows"][0].pop("generated_token_ids")
    elif mutation == "duplicate": yarn["rows"].append(copy.deepcopy(yarn["rows"][0]))
    elif mutation == "missing": yarn["rows"].pop()
    elif mutation == "aggregate": yarn["result"]["results"]["macro_official_task_score"] = .123
    else: yarn["result"]["results"]["examples_sha256"] = "0" * 64
    yarn["manifest"] = copy.deepcopy(protocol)
    with pytest.raises(ValueError):
        summary.summarize_panel(panel)


def test_loader_reads_only_four_named_fresh_directories(tmp_path):
    panel = four_arm_panel()
    for arm, item in panel.items():
        directory = tmp_path / arm
        directory.mkdir()
        examples = directory / "examples.jsonl"
        examples.write_text("".join(json.dumps(row) + "\n" for row in item["rows"]))
        item["result"]["results"]["examples_sha256"] = crossing.common.file_hash(examples)
        (directory / "results.json").write_text(json.dumps(item["result"]))
        (directory / "run_manifest.json").write_text(json.dumps(item["manifest"]))
    old = tmp_path / "old_pilot"
    old.mkdir()
    (old / "results.json").write_text("forbidden and malformed")
    loaded = summary.load_panel(tmp_path)
    assert set(loaded) == set(summary.ARMS)
    summary.summarize_panel(loaded)
    with (tmp_path / "official_yarn" / "examples.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="examples SHA mismatch"):
        summary.load_panel(tmp_path)


def test_output_is_portable_and_does_not_copy_raw_private_paths():
    panel = four_arm_panel()
    panel["official_yarn"]["result"]["data"]["cells"]["vt"]["65536"]["path"] = "/private/yarn.jsonl"
    result = summary.summarize_panel(panel)
    assert "/private/" not in json.dumps(result)
    assert result["arm_identities"]["official_yarn"]["active_sha256_float32"] == summary.YARN_TENSOR_SHA256
