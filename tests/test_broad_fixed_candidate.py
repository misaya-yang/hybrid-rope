import json
import sys

from experiments.olmo_recovery_20260912.compare_broad_fixed_candidate import main


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_broad_candidate_uses_row_matched_official_scores(tmp_path, monkeypatch):
    tasks = ["niah_single_1", "niah_multikey_1", "niah_multivalue", "vt", "cwe", "fwe", "qa_2"]
    source = []
    candidate = []
    baseline = []
    for index in range(350):
        task = tasks[index % len(tasks)]
        row_id = f"row_{index}"
        source.append({
            "row_id": row_id,
            "task": task,
            "length_cap": 16384,
            "references": ["gold"],
            "prompt_sha256": row_id,
        })
        candidate.append({
            "row_id": row_id,
            "task": task,
            "length_cap": 16384,
            "references": ["gold"],
            "prompt_sha256": row_id,
            "output_text": "gold",
            "ruler_official_score": 1.0,
            "exact_plus_eos": True,
            "ended_eos": True,
            "hit_cap": False,
        })
        baseline.append({
            "row_id": row_id,
            "task": task,
            "length_cap": 16384,
            "output_text": "wrong",
            "correct": 0.0,
            "ended_eos": True,
        })
    prepared = tmp_path / "prepared"
    run = tmp_path / "run"
    write_jsonl(prepared / "screen.jsonl", source)
    write_jsonl(run / "generations.jsonl", candidate)
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, baseline)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "compare",
        "--prepared", str(prepared),
        "--candidate-run", str(run),
        "--baseline", f"old={baseline_path}",
        "--out", str(output),
    ])
    main()
    result = json.loads(output.read_text())
    assert result["summaries"]["candidate"]["task_equal_official"] == 1.0
    assert result["contrasts"]["candidate_minus_old"]["paired_row_wins"] == 350

