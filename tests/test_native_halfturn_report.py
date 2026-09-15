import json

import pytest

from experiments.native_halfturn_phase_20260915 import report_ruler


def make_rows(arm, delta=0.0):
    rows = []
    for task_index, task in enumerate(report_ruler.TASKS):
        for index in range(60):
            row_id = f"{task}-{index}"
            base = float((index + task_index) % 2)
            rows.append({
                "row_id": row_id,
                "task": task,
                "prompt_sha256": f"sha-{row_id}",
                "references": ["answer"],
                "input_tokens": 4000,
                "max_new_tokens": 32,
                "ruler_official_score": min(1.0, max(0.0, base + delta)),
                "arm": arm,
            })
    return rows


def write_run(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "status.json").write_text(json.dumps({"status": "COMPLETE", "rows": 780, "lm_rows": 0}))
    (path / "generations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_run_requires_complete_full13_panel(tmp_path):
    rows = make_rows("native")
    write_run(tmp_path, rows)
    loaded = report_ruler.load_run(tmp_path, expected_rows=780)
    assert len(loaded) == 780


def test_load_run_rejects_duplicate_identity(tmp_path):
    rows = make_rows("native")
    rows[-1]["row_id"] = rows[0]["row_id"]
    write_run(tmp_path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        report_ruler.load_run(tmp_path, expected_rows=780)


def test_paired_task_bootstrap_preserves_task_equal_weighting():
    native = {row["row_id"]: row for row in make_rows("native")}
    contract = {row["row_id"]: row for row in make_rows("contract", delta=0.1)}
    result = report_ruler.paired_task_bootstrap(contract, native, draws=1000)
    assert result["delta"] == pytest.approx(0.05, abs=0.005)
    assert result["paired_task_equal_bootstrap_ci95"][0] > 0.0
