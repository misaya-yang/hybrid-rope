import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.native_enhancement_oral_20260915 import report
from experiments.native_enhancement_oral_20260915.run import normalize_static_table


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def panel_rows(world_counts=None):
    rows = []
    for task in report.TASKS:
        axis, first, second = report.CONDITIONS[task]
        for cap in report.LENGTHS:
            for world in range((world_counts or {}).get(task, 2)):
                group = f"{task}:{cap}:{world}"
                for condition in (first, second):
                    for query in "ab":
                        row_id = f"{group}:{condition}:{query}"
                        refs = {"a": "ABCDE", "b": "FGHIJ"}
                        if task == "native_chain" and condition == second:
                            refs = {"a": "FGHIJ", "b": "ABCDE"}
                        rows.append({"row_id": row_id, "task": task, "length_cap": cap,
                                     "prompt_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
                                     "references": [refs[query]], "group_id": group,
                                     "intervention": {axis: condition, "query": query},
                                     "context_id": f"{group}:{condition}", "max_new_tokens": 32,
                                     "input_tokens": cap - 40})
    return rows


def run_rows(panel, arm, correct=False):
    return [{**{key: row[key] for key in report.PAIR_FIELDS}, "row_id": row["row_id"],
             "eval_id": "extra_panel:" + row["row_id"], "arm": arm,
             "generated_ids": [123, 0], "output_text": row["references"][0] if correct else "WRONG",
             "ended_eos": True, "empty": False, "hit_cap": False} for row in panel]


def make_contract(rows, arm):
    return {"arm": arm, "base_arm": "Native", "split": "confirm", "unadapted": True,
            "checkpoint_arm": None, "generation_length_caps": list(report.LENGTHS),
            "lm_enabled": False, "limit_per_cell": 0, "prefill_chunk_size": 0,
            "generation_prefill_strategy": "direct_generate_v1", "batch_size": 1,
            "runtime_versions": {"torch": "test", "transformers": "test", "model_dtype": "bfloat16",
                                 "attention_backend": "torch_sdpa_flash_only"},
            "row_split": None, "static_table": None,
            "row_ids": [row["eval_id"] for row in rows]}


def make_artifacts(tmp_path, *, world_counts=None):
    panel = panel_rows(world_counts)
    panel_path = tmp_path / "inputs.jsonl"
    write_jsonl(panel_path, panel)
    runs = {}
    for arm in ("native", "ncp"):
        directory = tmp_path / arm
        directory.mkdir()
        runs[arm] = directory
        rows = run_rows(panel, arm, correct=arm == "ncp")
        write_jsonl(directory / "generations.jsonl", rows)
        write_json(directory / "status.json", {"status": "COMPLETE", "rows": len(rows), "lm_rows": 0})
        write_json(directory / "contract.json", make_contract(rows, arm))
        write_json(directory / "preparation_run_identity.json",
                   {"model_path": "/existing/model", "model_config_sha256": "a" * 64,
                    "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                    "arm": arm, "static_table": None})
    return panel_path, panel, runs


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def mutate_rows(path, change):
    rows = report.read_jsonl(path)
    change(rows)
    write_jsonl(path, rows)


def test_perfect_candidate_has_exact_effect_and_zero_width_world_interval(tmp_path):
    panel, _, paths = make_artifacts(tmp_path)
    result = report.build_report(panel, paths, draws=200)
    metric = result["task_equal_by_length"]["4096"]["metrics"]["exact_match"]
    assert metric == {"native": 0.0, "ncp": 1.0, "effect": 1.0,
                      "paired_world_bootstrap_ci95": [1.0, 1.0]}
    assert "paired_world_bootstrap_ci95" not in result["task_and_length_equal_descriptive"]["exact_match"]


def test_score_is_complete_response_equality_with_outer_whitespace_only():
    panel = panel_rows()
    outputs = run_rows(panel, "ncp")
    for index, text in enumerate((" \nABCDE\t", "FGHIJ.", "answer: ABCDE", "fghij")):
        outputs[index]["output_text"] = text
    values = report._world_metrics({r["row_id"]: r for r in panel}, {r["row_id"]: r for r in outputs})
    assert values["native_binding", 1024][0, :3].tolist() == [0.25, 0.0, 0.0]


def test_all_four_query_pair_and_difference_in_differences_are_distinct(tmp_path):
    panel_path, panel, paths = make_artifacts(tmp_path)
    rows = run_rows(panel, "ncp")
    for row, expected in zip(rows, panel):
        if expected["task"] == "native_binding" and expected["intervention"]["layout"] == "far":
            row["output_text"] = expected["references"][0]
    write_jsonl(paths["ncp"] / "generations.jsonl", rows)
    result = report.build_report(panel_path, paths, draws=200)
    stats = result["strata"]["native_binding:1024"]["metrics"]
    assert stats["exact_match"]["effect"] == 0.5
    assert stats["all_four_correct"]["effect"] == 0
    assert stats["query_pair_both_correct"]["effect"] == 0.5
    assert stats["condition_1_minus_0"]["effect"] == 1


def test_fixed_task_weighting_survives_unequal_world_counts(tmp_path):
    panel_path, panel, paths = make_artifacts(tmp_path, world_counts={"native_binding": 4, "native_chain": 2})
    rows = run_rows(panel, "ncp", correct=True)
    for row in rows:
        if row["task"] == "native_chain":
            row["output_text"] = "WRONG"
    write_jsonl(paths["ncp"] / "generations.jsonl", rows)
    result = report.build_report(panel_path, paths, draws=200)
    assert result["task_equal_by_length"]["1024"]["metrics"]["exact_match"]["effect"] == 0.5


def test_cluster_bootstrap_keeps_four_correlated_rows_together(tmp_path):
    panel_path, panel, paths = make_artifacts(tmp_path)
    rows = run_rows(panel, "ncp", correct=True)
    for row in rows:
        if row["group_id"].endswith(":1"):
            row["output_text"] = "WRONG"
    write_jsonl(paths["ncp"] / "generations.jsonl", rows)
    result = report.build_report(panel_path, paths, draws=1000, seed=42)
    stats = result["strata"]["native_binding:1024"]["metrics"]
    assert stats["exact_match"]["effect"] == 0.5
    assert stats["exact_match"]["paired_world_bootstrap_ci95"] == [0.0, 1.0]
    assert stats["all_four_correct"]["paired_world_bootstrap_ci95"] == [0.0, 1.0]


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.pop(), "missing or extra"),
    (lambda rows: rows.__setitem__(-1, copy.deepcopy(rows[0])), "duplicate row_id"),
    (lambda rows: rows[0].update(prompt_sha256="f" * 64), "identity drift"),
    (lambda rows: rows[0].update(input_tokens=1), "identity drift"),
    (lambda rows: rows[0].update(context_id="wrong"), "context drift"),
    (lambda rows: rows[0].update(output_text=float("nan")), "nonfinite"),
    (lambda rows: rows[0].update(hit_cap=True), "hit_cap"),
    (lambda rows: rows[0].update(empty=True), "empty diagnostic"),
])
def test_rejects_unpaired_or_corrupted_results(tmp_path, change, match):
    panel_path, _, paths = make_artifacts(tmp_path)
    mutate_rows(paths["ncp"] / "generations.jsonl", change)
    with pytest.raises(ValueError, match=match):
        report.build_report(panel_path, paths, draws=100)


@pytest.mark.parametrize("file,change,match", [
    ("status.json", lambda v: v.update(status="RUNNING"), "not COMPLETE"),
    ("contract.json", lambda v: v.update(batch_size=2), "runtime identity drift"),
    ("contract.json", lambda v: v.update(row_ids=v["row_ids"][::-1]), "generation order"),
    ("preparation_run_identity.json", lambda v: v.update(model_path="/another/model"), "model identity drift"),
    ("preparation_run_identity.json", lambda v: v.update(panel_sha256="b" * 64), "panel or arm"),
    ("preparation_run_identity.json", lambda v: v.update(static_table={"gain": 1, "values_float32": [1, .1]}), "table identity drift"),
])
def test_rejects_status_and_contract_drift(tmp_path, file, change, match):
    panel_path, _, paths = make_artifacts(tmp_path)
    mutate_json(paths["ncp"] / file, change)
    with pytest.raises(ValueError, match=match):
        report.build_report(panel_path, paths, draws=100)


def test_rejects_incomplete_world_and_within_pair_context_drift(tmp_path):
    panel_path, panel, _ = make_artifacts(tmp_path)
    broken = copy.deepcopy(panel)
    broken[1]["context_id"] = "different-context"
    write_jsonl(panel_path, broken)
    with pytest.raises(ValueError, match="context drift"):
        report.load_panel(panel_path)
    write_jsonl(panel_path, panel[1:])
    with pytest.raises(ValueError, match="exactly four"):
        report.load_panel(panel_path)


def test_rejects_duplicate_content_given_distinct_world_ids(tmp_path):
    panel_path, panel, _ = make_artifacts(tmp_path)
    panel[4]["prompt_sha256"] = panel[0]["prompt_sha256"]
    write_jsonl(panel_path, panel)
    with pytest.raises(ValueError, match="duplicate prompt"):
        report.load_panel(panel_path)


def test_bootstrap_is_reproducible_and_import_has_no_torch_dependency(tmp_path):
    # Importing the reporter only depends on NumPy, never GPU/model runtime code.
    assert "torch" not in report.__dict__
    panel_path, _, paths = make_artifacts(tmp_path)
    first = report.build_report(panel_path, paths, draws=200, seed=7)
    second = report.build_report(panel_path, paths, draws=200, seed=7)
    assert first == second
    assert np.isfinite(first["task_and_length_equal_descriptive"]["exact_match"]["effect"])


def test_prepared_table_records_exact_float32_values_and_checks_gain():
    values = [float(0.8 ** index) for index in range(64)]
    result = normalize_static_table({"table": {"values_float32": values, "gain": 1}})
    assert result["values_float32"] == np.asarray(values, dtype=np.float32).tolist()
    assert result["gain"] == 1.0
    assert normalize_static_table(None) is None
    with pytest.raises(ValueError, match="gain=1"):
        normalize_static_table({"values_float32": values, "gain": 4})
