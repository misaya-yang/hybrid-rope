import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913.tables import tensor_sha256
from experiments.iclr2027_strong_evidence_20260915 import four_model_yarn_full13 as four


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def panel_rows(*, count: int, target: int = 1024) -> list[dict]:
    rows = []
    for task_index, task in enumerate(four.TASKS):
        for index in range(count):
            prompt = [task_index + 1, index + 100]
            prompt_hash = hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest()
            rows.append({
                "row_id": f"{task}:{index}", "task": task, "length_cap": target,
                "prompt_ids": prompt, "prompt_sha256": prompt_hash,
                "input_tokens": len(prompt), "max_new_tokens": 8,
                "references": [f"answer-{index}"],
            })
    return rows


def condition(tmp_path: Path, *, mode: str, source: Path, current: Path | None = None):
    tables = {arm: tmp_path / f"{arm}.json" for arm in ("tailspline", "mrpro")}
    return four.Condition(
        name="glm4_9b_s4_128k" if mode == "glm_assets10" else "test",
        model=tmp_path / "model", model_id="test_model", target=1024,
        prefill_chunk=0, data_manifest=tmp_path / "data.json",
        source_panel=source, panel_mode=mode, tp_tables=tables,
        existing_tp={arm: () for arm in tables}, yarn_table=tmp_path / "yarn.json",
        current_glm_panel=current,
    )


def table_receipt(path: Path, *, scale: float) -> dict:
    values = (np.geomspace(1.0, 0.001, 4) * scale).astype(np.float32)
    payload = {
        "status": TABLE_FORMAT, "table_sha256_float32": tensor_sha256(values),
        "gain": 1.1,
        "table": {"values_float32": values.tolist(), "gain": 1.1},
    }
    write_json(path, payload)
    return payload


def run_source(
    root: Path, table: Path, rows: list[dict], *, source_arm: str,
    score: float, lm_rows: int = 0,
) -> four.RunSource:
    receipt = json.loads(table.read_text())
    generated = []
    row_ids = []
    for index, row in enumerate(rows):
        eval_id = f"extra_panel:{row['row_id']}"
        row_ids.append(eval_id)
        generated.append({
            "eval_id": eval_id, "row_id": row["row_id"], "task": row["task"],
            "length_cap": row["length_cap"], "prompt_sha256": row["prompt_sha256"],
            "references": row["references"], "arm": source_arm,
            "generated_ids": [7, 2], "output_text": "answer",
            "ruler_official_score": score, "ended_eos": True, "hit_cap": False,
        })
    write_json(root / "contract.json", {
        "arm": source_arm, "base_arm": "Native", "unadapted": True,
        "row_ids": row_ids, "generation_length_caps": [1024],
        "static_table": receipt["table"],
    })
    write_jsonl(root / "generations.jsonl", generated)
    write_json(root / "status.json", {
        "status": "COMPLETE", "rows": len(rows), "lm_rows": lm_rows,
    })
    return four.RunSource(root, table)


def test_plan_is_four_fixed_models_and_excludes_qwen15(tmp_path):
    plan = four.build_plan(tmp_path)
    assert plan["gpu_started"] is False
    assert plan["models"] == ["llama3_8b", "qwen25_3b", "olmo2_1b", "glm4_9b_0414"]
    assert plan["excluded_models"] == ["qwen25_1p5b"]
    qwen = next(item for item in plan["conditions"] if item["model_id"] == "qwen25_3b")
    assert qwen["actions"][:2] == ["resume_tailspline_130", "run_mrpro_130"]
    qwen_condition = next(item for item in four.conditions(tmp_path) if item.model_id == "qwen25_3b")
    assert "tailspline_qwen25_s4_128k_ruler10_clean" in str(qwen_condition.source_panel)
    assert "64k128k" not in str(qwen_condition.source_panel)


def test_freezes_first10_from_large_panel_in_source_order(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl(source, panel_rows(count=20))
    cond = condition(tmp_path, mode="first10", source=source)
    panel, back, rows = four.freeze_panel(cond, tmp_path / "out")
    assert back is None
    assert len(rows) == 130
    assert rows == four.first_rows_per_task(panel_rows(count=20), 10)
    assert four.read_jsonl(panel) == rows


def test_glm_front5_must_equal_assets10_and_only_back5_is_frozen(tmp_path):
    full = panel_rows(count=10)
    source = tmp_path / "assets10.jsonl"
    current = tmp_path / "current.jsonl"
    write_jsonl(source, full)
    front = four.first_rows_per_task(full, 5)
    write_jsonl(current, front)
    cond = condition(tmp_path, mode="glm_assets10", source=source, current=current)
    panel, back, rows = four.freeze_panel(cond, tmp_path / "out")
    assert rows == full and four.read_jsonl(panel) == full
    back_rows = four.read_jsonl(back)
    assert len(back_rows) == 65
    assert all(int(row["row_id"].rsplit(":", 1)[1]) >= 5 for row in back_rows)

    broken = [dict(row) for row in front]
    broken[0]["prompt_sha256"] = full[5]["prompt_sha256"]
    write_jsonl(current, broken)
    with pytest.raises(ValueError):
        four.freeze_panel(cond, tmp_path / "other")


def test_strict_three_arm_report_merges_shards_in_assets10_order(tmp_path):
    panel = panel_rows(count=10)
    panel_path = tmp_path / "panel.jsonl"
    write_jsonl(panel_path, panel)
    tables = {}
    for offset, arm in enumerate(four.ARMS, 1):
        path = tmp_path / "tables" / f"{arm}.json"
        table_receipt(path, scale=1.0 + offset * 0.01)
        tables[arm] = path
    front = four.first_rows_per_task(panel, 5)
    back_prompts = set(row["prompt_sha256"] for row in panel) - set(
        row["prompt_sha256"] for row in front
    )
    back = [row for row in panel if row["prompt_sha256"] in back_prompts]
    sources = {
        "tailspline": (
            run_source(tmp_path / "runs/t_front", tables["tailspline"], front,
                       source_arm="tail-front", score=0.8, lm_rows=5),
            run_source(tmp_path / "runs/t_back", tables["tailspline"], back,
                       source_arm="tail-back", score=0.8),
        ),
        "mrpro": (
            run_source(tmp_path / "runs/p_front", tables["mrpro"], front,
                       source_arm="mr-front", score=0.6, lm_rows=5),
            run_source(tmp_path / "runs/p_back", tables["mrpro"], back,
                       source_arm="mr-back", score=0.6),
        ),
        "yarn": (
            run_source(tmp_path / "runs/yarn", tables["yarn"], panel,
                       source_arm="official-yarn", score=0.5),
        ),
    }
    cond = condition(tmp_path, mode="exact10", source=panel_path)
    cond = four.Condition(
        **{**cond.__dict__, "tp_tables": {arm: tables[arm] for arm in ("tailspline", "mrpro")},
           "yarn_table": tables["yarn"]}
    )
    report = four.build_report(
        condition=cond, panel_path=panel_path, sources=sources,
        out=tmp_path / "report", draws=100, seed=16,
    )
    assert report["paired_prompts"] == 130
    assert report["identity_checks"] == [
        "row_id", "prompt_sha256", "task", "length_cap", "table", "gain",
    ]
    assert report["contrasts"]["mrpro"]["delta_task_macro_official"] == pytest.approx(0.2)
    assert report["contrasts"]["yarn"]["delta_task_macro_official"] == pytest.approx(0.3)
    merged = four.read_jsonl(tmp_path / "report/merged/tailspline/generations.jsonl")
    assert [row["prompt_sha256"] for row in merged] == [row["prompt_sha256"] for row in panel]
    assert all(row["gain"] == 1.1 and row["table_sha256_float32"] for row in merged)


def test_report_rejects_contract_gain_drift(tmp_path):
    rows = panel_rows(count=10)
    table = tmp_path / "table.json"
    table_receipt(table, scale=1.0)
    source = run_source(tmp_path / "run", table, rows, source_arm="tail", score=1.0)
    contract = json.loads((source.run / "contract.json").read_text())
    contract["static_table"]["gain"] = 1.2
    write_json(source.run / "contract.json", contract)
    with pytest.raises(ValueError, match="table/gain"):
        four.validate_run_source(source)
