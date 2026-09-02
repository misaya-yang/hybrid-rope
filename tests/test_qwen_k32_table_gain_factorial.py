"""Synthetic factorial summary check."""

from scripts.eval.eval_qwen_k32_table_gain_factorial import summarize


def test_frequency_only_can_be_identified_from_missing_cells():
    nll_new, qa_new, nll_old, qa_old = [], [], [], []
    for index in range(32):
        sid = f"qwen-k32-natural-{index:03d}"
        for length in (32768, 65536):
            native = 2.0 if length == 32768 else 3.0
            nll_new += [
                {"arm": "native_table_unit_gain", "sample_id": sid, "length": length, "nll": native},
                {"arm": "index_table_index_gain", "sample_id": sid, "length": length,
                 "nll": native + (.01 if length == 32768 else -.2)},
            ]
            nll_new += [
                {"arm": "native_table_index_gain", "sample_id": sid, "length": length,
                 "nll": native},
                {"arm": "index_table_unit_gain", "sample_id": sid, "length": length,
                 "nll": native + (.01 if length == 32768 else -.15)},
            ]
    for task in ("2wikimqa", "qasper", "hotpotqa"):
        for index in range(10):
            row_id = f"{task}-{index}"
            qa_new += [
                {"arm": "native_table_unit_gain", "task": task, "row_sha256": row_id, "score": .2},
                {"arm": "index_table_index_gain", "task": task, "row_sha256": row_id, "score": .1},
            ]
            qa_new += [
                {"arm": "native_table_index_gain", "task": task, "row_sha256": row_id, "score": .2},
                {"arm": "index_table_unit_gain", "task": task, "row_sha256": row_id, "score": .3},
            ]
    result = summarize(nll_new, qa_new, nll_old, qa_old)
    assert result["frequency_only_index"]["joint_operating_point_point_gate"] is True
    assert result["frequency_only_index"]["nll_64k_minus_native"]["mean"] < 0
    assert result["frequency_only_index"]["qa_minus_native"]["mean_macro"] > 0
