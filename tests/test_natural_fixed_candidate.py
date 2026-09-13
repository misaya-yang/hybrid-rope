import json
import sys

from experiments.olmo_recovery_20260912.compare_natural_fixed_candidate import main


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_natural_candidate_uses_row_matched_whole_response_f1(tmp_path, monkeypatch):
    tasks = ["hotpotqa", "2wikimqa", "qasper", "multifieldqa_en", "narrativeqa"]
    source, candidate, left, right = [], [], [], []
    for index in range(391):
        task = tasks[index % len(tasks)]
        row_id = f"row_{index}"
        common = {
            "row_id": row_id,
            "task": task,
            "length_cap": 16384,
            "references": ["gold"],
            "prompt_sha256": row_id,
        }
        source.append(common)
        candidate.append({
            **common,
            "output_text": "gold",
            "whole_response_f1": 1.0,
            "exact_plus_eos": True,
            "ended_eos": True,
            "hit_cap": False,
        })
        baseline = {**common, "output_text": "wrong", "correct": 0.0, "ended_eos": True}
        (left if index < 200 else right).append(baseline)
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    write_jsonl(prepared / "screen.jsonl", source)
    write_jsonl(run / "generations.jsonl", candidate)
    left_path, right_path = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    write_jsonl(left_path, left)
    write_jsonl(right_path, right)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "compare",
        "--prepared", str(prepared),
        "--candidate-run", str(run),
        "--baseline", f"old={left_path}",
        "--baseline", f"old={right_path}",
        "--out", str(output),
    ])
    main()
    result = json.loads(output.read_text())
    assert result["summaries"]["candidate"]["task_equal_whole_response_f1"] == 1.0
    assert result["contrasts"]["candidate_minus_old"]["paired_row_wins"] == 391

