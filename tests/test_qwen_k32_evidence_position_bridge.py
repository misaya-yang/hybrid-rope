"""Synthetic evidence-position summary check."""

from scripts.eval.eval_qwen_k32_evidence_position_bridge import summarize


def test_bridge_detects_index_specific_position_penalty():
    rows = []
    for task in ("2wikimqa", "qasper", "hotpotqa"):
        for index in range(10):
            row_id = f"{task}-{index}"
            for profile, far in (("native_unit", 1.0), ("index_unit", 1.5)):
                for condition, nll in (("near", .5), ("far", far), ("ablated", 2.0)):
                    rows.append({"profile": profile, "task": task, "row_sha256": row_id,
                                 "condition": condition, "answer_nll": nll})
    result = summarize(rows)
    assert result["diagnosis"] == "INDEX_SPECIFIC_DISTANCE_PENALTY"
    assert result["contrasts"]["index_minus_native_position_penalty"]["mean_macro"] > 0
