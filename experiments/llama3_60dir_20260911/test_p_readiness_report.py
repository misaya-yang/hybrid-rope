"""Offline readiness fixtures; none of these values is a model result."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import p_readiness_report as R


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def fixture(root, score):
    panel = []
    for task in R.TASKS:
        for cap in R.CAPS:
            for i in range(4):
                rid = f"{task}_{cap}_{i}"
                panel.append({
                    "row_id": rid, "task": task, "length_cap": cap,
                    "source_document_id": "s_" + rid,
                    "semantic_group_id": "g_" + rid,
                    "prompt_sha256": "p_" + rid, "actual_length": cap - 128,
                    "evidence_positions": [10], "distractor_positions": [],
                    "gold": ["x"], "references": ["x"],
                    "scorer_revision": "v1", "source_identity_revision": "v2",
                })
    dump(root / "data" / "P" / "rows.jsonl", panel)
    for arm in R.CONTROLS:
        rows = [{**row, "partial_score": score, "correct": score} for row in panel]
        dump(root / "results" / "P_l0" / f"{arm}.jsonl", rows)
    native = [{**row, "partial_score": score, "correct": score}
              for row in panel if row["length_cap"] == 8192]
    dump(root / "results" / "P_native" / "Native.jsonl", native)
    for directory in (root / "results" / "P_l0", root / "results" / "P_native"):
        (directory / "run_summary.json").write_text(json.dumps({"status": "COMPLETE"}))


def test_readiness_and_floor_refusal():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture(root, 0.5)
        out = root / "ready.json"
        assert R.main(["--root", str(root), "--out", str(out)]) == 0
        assert json.loads(out.read_text())["status"] == "READY_FOR_S"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture(root, 0.0)
        out = root / "floor.json"
        assert R.main(["--root", str(root), "--out", str(out)]) == 3
        assert json.loads(out.read_text())["status"] == "INSTRUMENT_LIMITED"


if __name__ == "__main__":
    test_readiness_and_floor_refusal()
    print("P readiness tests pass")
