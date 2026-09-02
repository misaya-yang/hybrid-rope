"""CPU-only checks for the Qwen K32 far-evidence follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import estimate_qwen_k32_qa_cost as cost
from scripts.analysis import summarize_qwen_k32_far_evidence_qa as summary
from scripts.analysis import summarize_qwen_k32_natural_nll as nll_summary
from scripts.data import prepare_qwen_k32_far_evidence_qa as prepare
from scripts.eval import eval_qwen_k32_far_evidence_qa as evaluator


def hashed(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class CharTokenizer:
    all_special_ids = [2, 151644, 151645]
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt and not tokenize
        return (chr(151644) + "system" + chr(151645) + chr(151644)
                + messages[0]["content"] + chr(151645) + chr(151644))

    def decode(self, values, **_):
        return "".join(chr(int(value)) for value in values)


def test_far_evidence_assembly_places_query_after_32k_distractor():
    tokenizer = CharTokenizer()
    source = {"context": "Evidence says zebracode is correct.", "input": "What is correct?",
              "answers": ["zebracode"], "all_classes": []}
    filler = {"sample_id": "qwen-k32-natural-000", "input_ids": [2] + [ord("x")] * 65536,
              "prompt_ids_sha256": hashed("filler")}
    row = prepare.assemble_row(
        tokenizer, task="2wikimqa", source_index=0, source=source,
        template="Context: {context}\nQuestion: {input}", generation_tokens=32,
        filler_row=filler)
    assert row is not None
    assert row["input_tokens"] + row["generation_tokens"] == prepare.TARGET_LENGTH
    assert row["evidence_end_token_exclusive"] <= prepare.EVIDENCE_END_CEILING
    assert row["evidence_to_query_lower_bound"] >= prepare.MIN_EVIDENCE_TO_QUERY_TOKENS
    assert row["filler_source_special_tokens_replaced"] == 1
    assert row["filler_replacement_token_id"] == ord("\n")
    assert row["filler_selected_special_tokens"] == 0
    leaking = dict(filler, input_ids=[ord(char) for char in ("zebracode " * 7000)])
    assert prepare.assemble_row(
        tokenizer, task="2wikimqa", source_index=0, source=source,
        template="Context: {context}\nQuestion: {input}", generation_tokens=32,
        filler_row=leaking) is None


def passing_nll_receipt():
    return {
        "status": "QWEN_K32_PACKED_NATURAL_NLL_STAGED_SUMMARIZED",
        "classification": {"resolver": "PASS", "index_vs_yarn": "UNRESOLVED"},
        "identity": {
            "checkpoint_weight_sha256": evaluator.EXPECTED_WEIGHT_SHA256,
            "config_sha256": evaluator.EXPECTED_CONFIG_SHA256,
            "profiles": {name: {"tensor_sha256": values[0], "file_sha256": values[1],
                                "attention_scaling": float(values[2])}
                         for name, values in nll_summary.PROFILE.items()},
        },
    }


def test_far_qa_requires_passing_non_dominated_nll(tmp_path):
    path = tmp_path / "nll.json"
    receipt = passing_nll_receipt(); path.write_text(json.dumps(receipt))
    assert evaluator.load_nll_receipt(path)["classification"]["resolver"] == "PASS"
    receipt["classification"]["index_vs_yarn"] = "YARN_FAVORED"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="does not authorize"):
        evaluator.load_nll_receipt(path)


def test_combined_cost_gate_respects_ten_rmb_cap():
    canary = {"status": cost.STATUS, "metrics_exposed": False, "rows": 2,
              "batch_seconds": {"32": 1.0, "128": 2.0},
              "elapsed_seconds": 3.0, "process_elapsed_seconds": 4.0}
    nll = {"status": "STAGED_NLL_COST_ADMITTED", "decision": "RUN",
           "estimated_cost_rmb": 1.0}
    report = cost.estimate(canary, nll, rmb_per_hour=1.0, quantum_seconds=1)
    assert report["decision"] == "RUN"
    nll["estimated_cost_rmb"] = 9.99
    assert cost.estimate(canary, nll, rmb_per_hour=1.0, quantum_seconds=1)["decision"] == "DO_NOT_RUN"


def test_far_qa_summary_recomputes_paired_grid(tmp_path):
    root = tmp_path / "full"; root.mkdir()
    profiles = [{"name": arm, "tensor_sha256": values[0], "file_sha256": values[1],
                 "attention_scaling": values[2]} for arm, values in nll_summary.PROFILE.items()]
    manifest = {
        "status": "QWEN_K32_FAR_EVIDENCE_QA_FROZEN", "stage": "full",
        "checkpoint_weight_sha256": nll_summary.WEIGHT_SHA256,
        "config_sha256": nll_summary.CONFIG_SHA256,
        "data_manifest_sha256": hashed("data-manifest"), "data_rows_sha256": hashed("data-rows"),
        "nll_receipt_sha256": hashed("nll"),
        "script_sha256": nll_summary.sha256(Path("scripts/eval/eval_qwen_k32_far_evidence_qa.py")),
        "model_source_sha256": hashed("model"), "attention_source_sha256": hashed("attention"),
        "tasks": list(prepare.TASKS), "rows_per_task": prepare.ROWS_PER_TASK,
        "arm_order": list(nll_summary.ARMS), "declared_arm_order": list(nll_summary.ARMS),
        "profiles": profiles, "use_cache": True, "compile": False, "model_updates": 0,
        "profile_selection": False, "all_profiles_loaded_before_inference": True,
    }
    manifest_path = root / "run_manifest.json"; manifest_path.write_text(json.dumps(manifest))
    means = {arm: {} for arm in nll_summary.ARMS}; rows = []
    arm_scores = {"Native": .2, "normalized_raw_index": .6, "official_equation_yarn": .4}
    for task in prepare.TASKS:
        for arm in nll_summary.ARMS:
            means[arm][task] = arm_scores[arm]
            for index in range(prepare.ROWS_PER_TASK):
                rows.append({"arm": arm, "task": task, "row_sha256": hashed(f"{task}-{index}"),
                             "table_sha256_float32": nll_summary.PROFILE[arm][0],
                             "attention_scaling": nll_summary.PROFILE[arm][2],
                             "generation_tokens_budget": 32, "generated_tokens": 3,
                             "prediction": "answer", "score": arm_scores[arm]})
    examples = root / "examples.jsonl"
    examples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = {"status": summary.STATUS, "stage": "full", "rows": len(rows), "means": means,
              "examples_sha256": nll_summary.sha256(examples),
              "run_manifest_sha256": nll_summary.sha256(manifest_path)}
    (root / "results.json").write_text(json.dumps(result))
    report = summary.summarize(root)
    assert report["classification"]["natural_far_evidence_utilization"] == "PASS"
    assert report["classification"]["index_vs_yarn"] == "INDEX_FAVORED"
