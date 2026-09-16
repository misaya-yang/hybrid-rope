from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.iclr2027_strong_evidence_20260915 import run_natural_long as runner


def test_gpu_lock_reuses_an_inherited_outer_queue_descriptor(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    outer = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    inherited = os.dup(outer)
    try:
        fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with runner.acquire_gpu_lock(lock_path, inherited_fd=inherited):
            pass
    finally:
        os.close(inherited)
        os.close(outer)


def test_longbench_v2_direct_answer_parser_rejects_unknown_empty_and_truncated():
    row = {
        "score_contract": "longbench_v2_mc_direct_answer_v1",
        "references": ["B"],
    }
    assert runner.score_natural_output(row, "The correct answer is (B).") == {
        "official_score": 1.0,
        "official_metric": "accuracy",
        "official_score_contract": "longbench_v2_mc_direct_answer_v1",
        "parsed_answer": "B",
    }
    assert runner.score_natural_output(row, "The correct answer is B")["official_score"] == 1
    for output in ("", "B", "unknown", "The correct answer is (", "The correct answer is (E)"):
        scored = runner.score_natural_output(row, output)
        assert scored["official_score"] == 0
        assert scored["parsed_answer"] is None
    assert runner.classify_answer("", parsed=None, hit_cap=False) == "EMPTY"
    assert runner.classify_answer(
        "The correct answer is (", parsed=None, hit_cap=True,
    ) == "TRUNCATED_OR_UNPARSEABLE"


def test_infinitebench_en_dia_uses_official_case_insensitive_name_accuracy():
    row = {
        "score_contract": "infinitebench_en_dia_accuracy_v1",
        "references": ["Phoebe Buffay", "Phoebe"],
    }
    assert runner.score_natural_output(row, "I think it was PHOEBE.")["official_score"] == 1
    assert runner.score_natural_output(row, "Monica")["official_score"] == 0
    assert runner.score_natural_output(row, "")["official_score"] == 0


def test_infinitebench_en_qa_matches_official_word_f1_normalization():
    row = {
        "score_contract": "infinitebench_en_qa_rouge_f1_v1",
        "references": ["The red fox"],
    }
    assert runner.score_natural_output(row, "a red fox!")["official_score"] == 1
    assert runner.score_natural_output(row, "red wolf")["official_score"] == pytest.approx(0.5)
    assert runner.score_natural_output(row, "")["official_score"] == 0


def test_synthetic_contains_field_cannot_become_a_natural_score():
    row = {
        "score_contract": "longbench_v2_mc_direct_answer_v1",
        "references": ["A"],
    }
    synthetic_generation = {"output_text": "The correct answer is (D)", "ruler_official_score": 1.0}
    official = runner.score_natural_output(row, synthetic_generation["output_text"])
    assert official["official_score"] == 0


def test_default_cli_is_plan_only_and_does_not_create_output(tmp_path):
    module = "experiments.iclr2027_strong_evidence_20260915.run_natural_long"
    out = tmp_path / "must-not-exist"
    result = subprocess.run([
        sys.executable, "-m", module,
        "--model", str(tmp_path / "model"),
        "--model-id", "llama3_8b",
        "--data-root", str(tmp_path / "data"),
        "--out", str(out),
        "--scale", "4",
        "--lengths", "16384,32768",
        "--rows-per-task", "0",
        "--data-manifest", str(tmp_path / "manifest.json"),
        "--python", sys.executable,
    ], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY"
    assert plan["forward_passes_started"] == 0
    assert plan["arm_order"] == ["tailspline", "mrpro"]
    assert plan["ruler_contains_used"] is False
    assert not out.exists()


def prepared_row(row_id="row-0", contract="longbench_v2_mc_direct_answer_v1"):
    prompt = [1, 2, 3]
    if contract == "longbench_v2_mc_direct_answer_v1":
        benchmark, task, references = "longbench_v2", "Single-Document QA", ["A"]
    elif contract == "infinitebench_en_dia_accuracy_v1":
        benchmark, task, references = "infinitebench", "longdialogue_qa_eng", ["Rachel"]
    else:
        benchmark, task, references = "infinitebench", "longbook_qa_eng", ["red fox"]
    return {
        "row_id": row_id,
        "benchmark": benchmark,
        "task": task,
        "source_id": "source-0",
        "source_cluster_id": "cluster-0",
        "prompt_ids": prompt,
        "input_tokens": len(prompt),
        "max_new_tokens": 4,
        "length_cap": 16,
        "length_bucket": "8K-16K",
        "references": references,
        "score_contract": contract,
        "prompt_sha256": runner._prompt_digest(prompt),
    }


def test_prepared_contract_rejects_wrong_task_and_prompt_identity():
    manifest = {
        "status": "COMPLETE", "model_id": "model", "scale": 4,
        "lengths": [16], "rows": 1,
    }
    row = prepared_row()
    runner.validate_prepared_data(
        manifest, [row], model_id="model", scale=4, lengths=(16,),
        rows_per_task=0, benchmark="longbench_v2",
    )
    wrong_task = dict(row, benchmark="infinitebench")
    with pytest.raises(ValueError, match="score contract"):
        runner.validate_prepared_data(
            manifest, [wrong_task], model_id="model", scale=4, lengths=(16,),
            rows_per_task=0, benchmark=None,
        )
    wrong_prompt = dict(row, prompt_sha256="0" * 64)
    with pytest.raises(ValueError, match="prompt SHA256"):
        runner.validate_prepared_data(
            manifest, [wrong_prompt], model_id="model", scale=4, lengths=(16,),
            rows_per_task=0, benchmark=None,
        )
    with pytest.raises(ValueError, match="uncapped LongBench-v2"):
        runner.validate_prepared_data(
            {**manifest, "benchmark": "longbench_v2", "rows_per_task": None},
            [row], model_id="model", scale=4, lengths=(16,),
            rows_per_task=100, benchmark="longbench_v2",
        )


def test_runner_reads_the_cpu_preparer_manifest_shape(tmp_path):
    row = prepared_row()
    data = tmp_path / "frozen"
    data.mkdir()
    inputs = data / "inputs.jsonl"
    inputs.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "status": "COMPLETE",
        "benchmark": "longbench_v2",
        "model_id": "model",
        "scale": 4,
        "lengths": [16],
        "rows_per_task": None,
        "summary": {"selected_rows": 1},
        "inputs_sha256": runner.sha256_file(inputs),
    }
    path, digest = runner.resolve_inputs(data, manifest)
    assert path == inputs
    assert digest == manifest["inputs_sha256"]
    runner.validate_prepared_data(
        manifest, runner.read_jsonl(path), model_id="model", scale=4,
        lengths=(16,), rows_per_task=0, benchmark="longbench_v2",
    )


def generation(row, arm, output, *, hit_cap=False):
    return {
        "eval_id": "extra_assets:" + row["row_id"],
        "row_id": row["row_id"],
        "arm": arm,
        "generated_ids": [7],
        "output_text": output,
        "hit_cap": hit_cap,
        "ended_eos": not hit_cap,
        "ruler_official_score": 1.0,
    }


def write_complete_run(path: Path, rows: list[dict], arm: str, outputs: list[str]):
    path.mkdir(parents=True)
    (path / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "rows": len(rows), "lm_rows": 0,
    }))
    (path / "generations.jsonl").write_text("".join(
        json.dumps(generation(row, arm, output)) + "\n"
        for row, output in zip(rows, outputs)
    ))


def test_strict_resume_refuses_reordered_or_incomplete_output(tmp_path):
    rows = [prepared_row("row-0"), prepared_row("row-1")]
    run = tmp_path / "run"
    write_complete_run(run, rows, "tailspline", [
        "The correct answer is (A)", "The correct answer is (A)",
    ])
    assert len(runner._validate_complete_run(run, rows)) == 2
    reversed_rows = list(reversed(rows))
    with pytest.raises(ValueError, match="exact prefix"):
        runner._validate_complete_run(run, reversed_rows)
    (run / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "rows": 1, "lm_rows": 0,
    }))
    with pytest.raises(ValueError, match="status differs"):
        runner._validate_complete_run(run, rows)


def test_strict_resume_never_stamps_a_contract_onto_unowned_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "generations.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="lack the frozen wrapper contract"):
        runner._ensure_contract(run / "wrapper_contract.json", {"status": "expected"})
    assert not (run / "wrapper_contract.json").exists()


def test_scored_output_preserves_full_generation_and_report_does_not_pool_mixed_metrics(tmp_path):
    rows = [
        prepared_row("mc"),
        prepared_row("qa", "infinitebench_en_qa_rouge_f1_v1"),
    ]
    tail_path = tmp_path / "tailspline"
    pro_path = tmp_path / "mrpro"
    write_complete_run(tail_path, rows, "tailspline", [
        "The correct answer is (A)", "red fox",
    ])
    write_complete_run(pro_path, rows, "mrpro", [
        "The correct answer is (B)", "red wolf",
    ])
    tail = runner.score_arm(tail_path, rows, "tailspline")
    pro = runner.score_arm(pro_path, rows, "mrpro")
    assert tail[0]["generated_ids"] == [7]
    assert tail[0]["ruler_official_score"] == 1.0
    assert tail[0]["ruler_contains_used_for_natural_score"] is False
    args = type("Args", (), {
        "model_id": "model", "scale": 4.0, "lengths": (16,),
        "rows_per_task": 0,
    })()
    report = runner.build_report(
        args, rows, {"tailspline": tail, "mrpro": pro},
        data_sha="a" * 64, inputs_sha="b" * 64,
    )
    assert report["overall"]["pooled_score"] is None
    assert report["scoring"]["ruler_contains_used"] is False
    assert report["by_task"]["Single-Document QA"]["delta_tailspline_minus_mrpro"] == 1
    assert report["by_task"]["longbook_qa_eng"]["delta_tailspline_minus_mrpro"] == pytest.approx(0.5)
