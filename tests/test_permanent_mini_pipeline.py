import json
import math

import pytest

from experiments.olmo_recovery_20260912.permanent_mini_pipeline import (
    bootstrap_contrast,
    compare_arms,
    freeze_panel,
    log_auc,
    load_frozen_panel,
    merge_arm,
    sha256_file,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def panel_row(row_id, task, length, prompt, *, semantic=None):
    row = {
        "row_id": row_id,
        "task": task,
        "family": "retrieval",
        "length_cap": length,
        "prompt_ids": [1, 2, 3],
        "prompt_sha256": prompt,
        "references": ["gold"],
        "max_new_tokens": 4,
    }
    if semantic:
        row["semantic_group_id"] = semantic
    return row


def identity():
    return {
        "model_revision": "model-r1",
        "tokenizer_template": "tokenizer-r1",
        "table": "table-r1",
        "decoder": "greedy-4",
        "scorer": "ruler-r1",
        "precision_arithmetic": "bf16-sdpa",
    }


def test_freeze_is_source_ordered_and_deduplicates_prompt_and_semantic(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(first, [
        panel_row("a", "task", 4, "p1", semantic="s1"),
        panel_row("b", "task", 4, "p2", semantic="s2"),
    ])
    write_jsonl(second, [
        panel_row("c", "task", 4, "p1", semantic="s3"),
        panel_row("d", "task", 4, "p3", semantic="s2"),
        panel_row("e", "task", 4, "p4", semantic="s4"),
    ])
    rows, manifest = freeze_panel(
        [("first", first), ("second", second)], tasks=["task"], lengths=[4],
        rows_per_cell=3, panel_id="panel",
    )
    assert [row["row_id"] for row in rows] == ["first:a", "first:b", "second:e"]
    assert manifest["deduplication"]["duplicate_prompts_skipped"] == 1
    assert manifest["deduplication"]["duplicate_semantics_skipped"] == 1


def test_merge_reuses_old_and_new_schema_and_reports_real_gap(tmp_path, monkeypatch):
    panel = [
        {**panel_row("src:a", "task", 4, "p1"), "source_panel": "src", "source_row_id": "a", "mini_semantic_id": "p1"},
        {**panel_row("src:b", "task", 4, "p2"), "source_panel": "src", "source_row_id": "b", "mini_semantic_id": "p2"},
        {**panel_row("src:c", "task", 4, "p3"), "source_panel": "src", "source_row_id": "c", "mini_semantic_id": "p3"},
    ]
    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    write_jsonl(old, [{
        "row_id": "a", "task": "task", "length_cap": 4, "prompt_sha256": "p1",
        "references": ["gold"], "output_text": "gold", "generated_ids": [1], "ended_eos": True,
    }])
    write_jsonl(new, [{
        "eval_id": "x:b", "row_id": "b", "task": "task", "length_cap": 4,
        "prompt_sha256": "p2", "references": ["gold"], "output_text": "bad",
        "generated_ids": [2, 3, 4, 5], "ended_eos": False, "ruler_official_score": 1.0,
    }])
    monkeypatch.setattr(
        "experiments.olmo_recovery_20260912.permanent_mini_pipeline.score_output",
        lambda source, output: float("gold" in output),
    )
    merged, missing, receipt = merge_arm(
        panel, [str(old), str(new)], arm_label="BM", identity=identity(),
    )
    assert [row["prompt_sha256"] for row in merged] == ["p1", "p2"]
    assert [row["prompt_sha256"] for row in missing] == ["p3"]
    assert merged[1]["official_score"] == 0.0  # raw text is rescored; stored score is ignored
    assert receipt["covered_rows"] == 2
    assert receipt["missing_rows"] == 1


def test_conflicting_duplicate_results_are_rejected(tmp_path, monkeypatch):
    panel = [{**panel_row("src:a", "task", 4, "p1"), "source_panel": "src", "source_row_id": "a", "mini_semantic_id": "p1"}]
    path = tmp_path / "duplicates.jsonl"
    common = {"row_id": "a", "task": "task", "length_cap": 4, "prompt_sha256": "p1", "references": ["gold"], "generated_ids": [], "ended_eos": True}
    write_jsonl(path, [{**common, "output_text": "gold"}, {**common, "output_text": "bad"}])
    monkeypatch.setattr(
        "experiments.olmo_recovery_20260912.permanent_mini_pipeline.score_output",
        lambda source, output: float("gold" in output),
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge_arm(panel, [str(path)], arm_label="BM", identity=identity())


def test_frozen_panel_bytes_are_bound_by_manifest(tmp_path):
    panel = tmp_path / "screen.jsonl"
    manifest = tmp_path / "manifest.json"
    row = {
        **panel_row("src:a", "task", 4, "p1"),
        "source_panel": "src", "source_row_id": "a", "mini_semantic_id": "p1",
    }
    write_jsonl(panel, [row])
    manifest.write_text(json.dumps({
        "status": "FROZEN", "panel_id": "p", "rows": 1, "tasks": ["task"],
        "lengths": [4], "rows_per_cell": 1, "panel_sha256": sha256_file(panel),
    }))
    assert load_frozen_panel(panel, manifest)[0] == [row]
    panel.write_text(panel.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes differ"):
        load_frozen_panel(panel, manifest)


def complete_arm(panel, label, scores):
    return [
        {
            "arm": label,
            "row_id": row["row_id"],
            "source_panel": row["source_panel"],
            "source_row_id": row["source_row_id"],
            "task": row["task"],
            "family": row["family"],
            "length_cap": row["length_cap"],
            "prompt_sha256": row["prompt_sha256"],
            "mini_semantic_id": row["mini_semantic_id"],
            "references": row["references"],
            "output_text": "",
            "generated_ids": [],
            "ended_eos": True,
            "hit_cap": False,
            "official_score": scores[(row["task"], row["length_cap"])],
        }
        for row in panel
    ]


def test_task_equal_log_auc_and_paired_bootstrap_are_reported():
    tasks = ["a", "b"]
    lengths = [4, 8, 16]
    panel = []
    for task in tasks:
        for length in lengths:
            for index in range(2):
                panel.append({
                    **panel_row(f"src:{task}-{length}-{index}", task, length, f"{task}-{length}-{index}"),
                    "source_panel": "src", "source_row_id": f"{task}-{length}-{index}",
                    "mini_semantic_id": f"{task}-{length}-{index}",
                })
    baseline_scores = {(task, length): 0.25 for task in tasks for length in lengths}
    candidate_scores = {
        ("a", 4): 0.5, ("a", 8): 0.5, ("a", 16): 0.5,
        ("b", 4): 0.25, ("b", 8): 0.25, ("b", 16): 0.25,
    }
    arms = {
        "candidate": complete_arm(panel, "candidate", candidate_scores),
        "baseline": complete_arm(panel, "baseline", baseline_scores),
    }
    result = compare_arms(
        panel, arms, candidate="candidate", baselines=["baseline"], tasks=tasks,
        lengths=lengths, draws=200, seed=7,
    )
    assert result["summaries"]["candidate"]["log_length_auc"] == pytest.approx(0.375)
    contrast = result["contrasts"]["candidate_minus_baseline"]
    assert contrast["delta_log_length_auc"] == pytest.approx(0.125)
    assert contrast["bootstrap"]["probability_delta_gt_zero"] == 1.0
    assert contrast["bootstrap"]["cell_paired_tasks"] == tasks
    assert log_auc({4: 0.25, 8: 0.5, 16: 0.75}) == pytest.approx(0.5)


def test_bootstrap_binds_complete_semantic_cases_across_lengths():
    candidate = []
    baseline = []
    for length in (4, 8):
        for semantic, delta in (("case-a", 1.0), ("case-b", 0.0)):
            prompt = f"{semantic}-{length}"
            common = {
                "task": "task", "length_cap": length, "prompt_sha256": prompt,
                "mini_semantic_id": semantic,
            }
            candidate.append({**common, "official_score": delta})
            baseline.append({**common, "official_score": 0.0})
    result = bootstrap_contrast(
        candidate, baseline, tasks=["task"], lengths=[4, 8], draws=100, seed=11,
    )
    assert result["joint_semantic_tasks"] == ["task"]
    assert result["cell_paired_tasks"] == []
