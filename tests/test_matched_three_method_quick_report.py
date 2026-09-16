from __future__ import annotations

import argparse
import json

from experiments.iclr2027_strong_evidence_20260915 import matched_three_method_quick_report as report


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_three_method_report_uses_exact_prompt_and_document_pairs(tmp_path):
    panel = tmp_path / "panel.jsonl"
    source = [
        {"row_id": "a", "task": "one", "length_cap": 16, "prompt_sha256": "a" * 64},
        {"row_id": "b", "task": "two", "length_cap": 16, "prompt_sha256": "b" * 64},
    ]
    write_jsonl(panel, source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    arms = []
    scores = {"tailspline": (1.0, 4.0), "mrpro": (0.0, 6.0), "yarn": (0.5, 5.0)}
    for label, (score, loss) in scores.items():
        generations = tmp_path / f"{label}.jsonl"
        write_jsonl(generations, [
            {**row, "ruler_official_score": score} for row in source
        ])
        lm = tmp_path / f"{label}.lm.jsonl"
        write_jsonl(lm, [{"document": 0, "length": 16, "whole_loss_sum": loss, "whole_target_count": 2}])
        generation_contract = tmp_path / f"{label}.generation.contract.json"
        lm_contract = tmp_path / f"{label}.lm.contract.json"
        generation_contract.write_text("{}\n")
        lm_contract.write_text("{}\n")
        arms.append([label, str(generations), str(lm), str(generation_contract), str(lm_contract)])
    args = argparse.Namespace(
        condition="test", target_length=16, task=["one", "two"], rows_per_task=1,
        ppl_documents=1, panel=panel, ppl_manifest=manifest, arm=arms,
    )
    result = report.build(args)
    assert result["niah"]["task_equal_macro"] == {"tailspline": 1.0, "mrpro": 0.0, "yarn": 0.5}
    assert result["ppl"]["pooled"]["tailspline"]["nll"] == 2.0
    assert result["comparisons"]["tailspline_minus_mrpro"]["niah_macro_delta"] == 1.0

