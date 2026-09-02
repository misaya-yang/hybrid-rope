"""Synthetic decision-rule check for frozen-candidate reranking."""

from scripts.eval.eval_qwen_k32_existing_candidate_rerank import summarize


def test_candidate_rerank_pass_requires_recovery_and_native_advantage():
    records, baselines = [], {}
    for task in ("2wikimqa", "qasper", "hotpotqa"):
        for index in range(10):
            row = f"{task}-{index}"
            records.append({"task": task, "row_sha256": row, "score": .8,
                            "oracle_f1_at_candidates": .8})
            baselines["index_unit", row] = {"score": .1}
            baselines["native_unit", row] = {"score": .2}
    assert summarize(records, baselines)["classification"] == "PASS"
