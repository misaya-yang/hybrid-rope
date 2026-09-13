import json
import sys

from experiments.olmo_recovery_20260912.compare_llama_fixed_candidate import main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_compares_one_static_candidate_to_reused_llama_rows(tmp_path, monkeypatch):
    root = tmp_path / "fixed"
    base = root / "prepared_llama_g8_r0"
    supplement = root / "supplement_48k" / "prepared"
    contract = {
        "one_table_per_arm_for_entire_session": True,
        "same_table_at_every_runtime_length": True,
        "same_table_in_every_layer": True,
        "runtime_table_switching": False,
    }
    write_json(base / "manifest.json", {
        "fixed_table_contract": contract, "runtime_lengths": [8192, 16384, 32768, 65536],
        "tasks": ["a", "b", "c"],
    })
    write_json(supplement / "manifest.json", {"fixed_table_contract": contract, "runtime_lengths": [49152]})
    source, candidate, baseline = [], [], []
    for cap in (8192, 16384, 32768, 65536, 49152):
        for task in ("a", "b", "c"):
            for index in range(4):
                row_id = f"{cap}_{task}_{index}"
                source.append({
                    "row_id": row_id, "task": task, "length_cap": cap,
                    "references": ["gold"], "prompt_sha256": row_id,
                })
                common = {
                    "eval_id": f"panel:{row_id}", "row_id": row_id, "task": task,
                    "length_cap": cap, "references": ["gold"], "prompt_sha256": row_id,
                    "ruler_official_score": 1.0, "exact_plus_eos": True,
                    "ended_eos": True, "hit_cap": False,
                }
                candidate.append(common)
                baseline.append({**common, "ruler_official_score": 0.0})
    write_jsonl(base / "screen.jsonl", [row for row in source if row["length_cap"] != 49152])
    write_jsonl(supplement / "screen.jsonl", [row for row in source if row["length_cap"] == 49152])
    write_jsonl(root / "candidate" / "generations.jsonl", candidate)
    for arm in ("BM_g8", "MrPro_g8"):
        write_jsonl(root / arm / "generations.jsonl", [row for row in baseline if row["length_cap"] != 49152])
        write_jsonl(root / "supplement_48k" / arm / "generations.jsonl", [row for row in baseline if row["length_cap"] == 49152])
    write_jsonl(root / "Native_8k" / "generations.jsonl", [row for row in candidate if row["length_cap"] == 8192])
    out = root / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare", "--root", str(root), "--candidate-run", str(root / "candidate"), "--out", str(out),
    ])
    main()
    result = json.loads(out.read_text())
    assert result["summaries"]["candidate"]["interval_min"] == 1.0
    assert result["contrasts"]["candidate_minus_BM_g8"]["five_point_log_auc"] == 1.0
