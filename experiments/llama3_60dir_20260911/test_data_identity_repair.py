"""Offline fixture for the metadata-only source identity migration."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import repair_panel_source_ids as R


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_metadata_only_migration():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        for i in range(2):
            rows.append({
                "row_id": f"P_niah_multivalue_8192_{i:04d}",
                "task": "niah_multivalue", "length_cap": 8192,
                "generator_seed": 7, "prompt_sha256": f"prompt-{i}",
                "source_id": "duplicate", "source_document_id": "duplicate",
                "group_id": "duplicate", "semantic_group_id": "duplicate",
                "prompt_ids": [1, 2, i], "references": [str(i)],
            })
        data = root / "data" / "P"
        dump(data / "rows.jsonl", rows)
        (data / "manifest.json").write_text(json.dumps({
            "status": "COMPLETE", "rows_sha256": R.sha_file(data / "rows.jsonl")}),
        )
        result_rows = [{
            **{k: row[k] for k in ("row_id", "task", "length_cap", "prompt_sha256",
                                   "references", "source_id", "source_document_id",
                                   "group_id", "semantic_group_id")},
            "output": f"answer-{i}", "partial_score": float(i),
        } for i, row in enumerate(rows)]
        result_dir = root / "results" / "P_l0"
        dump(result_dir / "MR.jsonl", result_rows)
        (result_dir / "run_summary.json").write_text(json.dumps({"status": "COMPLETE"}))

        report = R.migrate_stage(root, "P", repair_results=True)
        assert report["status"] == "MIGRATED"
        migrated = [json.loads(x) for x in (data / "rows.jsonl").read_text().splitlines()]
        assert len({r["source_document_id"] for r in migrated}) == 2
        assert [r["prompt_ids"] for r in migrated] == [[1, 2, 0], [1, 2, 1]]
        outputs = [json.loads(x) for x in (result_dir / "MR.jsonl").read_text().splitlines()]
        assert [r["output"] for r in outputs] == ["answer-0", "answer-1"]
        assert [r["partial_score"] for r in outputs] == [0.0, 1.0]
        assert all(r["source_document_id"] == migrated[i]["source_document_id"]
                   for i, r in enumerate(outputs))
        assert list(data.glob("rows.jsonl.pre_*"))
        assert R.migrate_stage(root, "P", repair_results=True)["status"] == "ALREADY_MIGRATED"


if __name__ == "__main__":
    test_metadata_only_migration()
    print("data identity repair test passes")
