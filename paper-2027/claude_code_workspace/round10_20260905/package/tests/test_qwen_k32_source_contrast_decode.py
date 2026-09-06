"""Synthetic decision-rule check for source-contrast decoding."""

from scripts.eval.eval_qwen_k32_source_contrast_decode import summarize


def test_source_contrast_pass_requires_index_recovery_and_advantage():
    contrast, greedy = [], []
    for task in ("2wikimqa", "qasper", "hotpotqa"):
        for index in range(10):
            row = f"{task}-{index}"
            for profile, old, new in (("native_unit", .2, .2), ("index_unit", .1, .8)):
                greedy.append({"profile": profile, "task": task, "row_sha256": row, "score": old})
                contrast.append({"profile": profile, "task": task, "row_sha256": row, "score": new})
    result = summarize(contrast, greedy)
    assert result["classification"] == "PASS"
    assert result["metrics"]["index_minus_native_contrast"]["mean_macro"] > 0
