import json
import sys

from experiments.olmo_recovery_20260912.compare_llama_64k8task import main


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_recomputes_runner_matched_64k_task_macro(tmp_path, monkeypatch):
    panel, candidate, baseline = [], [], []
    for task_index in range(8):
        task = f"task_{task_index}"
        for row_index in range(4):
            row_id = f"{task}_{row_index}"
            source = {
                "row_id": row_id, "task": task, "length_cap": 65536,
                "references": ["gold"], "prompt_sha256": row_id,
            }
            panel.append(source)
            candidate.append({**source, "output": "gold", "strict_score": 1.0, "ended_eos": True})
            baseline.append({**source, "output": "wrong", "strict_score": 0.0, "ended_eos": True})
    panel_path, candidate_path, baseline_path = tmp_path / "panel.jsonl", tmp_path / "candidate.jsonl", tmp_path / "baseline.jsonl"
    write_jsonl(panel_path, panel)
    write_jsonl(candidate_path, candidate)
    write_jsonl(baseline_path, baseline)
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "compare", "--panel", str(panel_path), "--candidate", str(candidate_path),
        "--baseline", f"BM={baseline_path}", "--out", str(out),
    ])
    main()
    result = json.loads(out.read_text())
    assert result["summaries"]["candidate"]["task_equal_official"] == 1.0
    assert result["contrasts"]["candidate_minus_BM"]["paired_row_wins"] == 32
