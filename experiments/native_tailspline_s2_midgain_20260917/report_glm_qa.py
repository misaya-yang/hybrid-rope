#!/usr/bin/env python3
"""Report paired GLM-4-9B Native versus NTS2 Natural-QA scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from scripts.eval.longbench_metrics import qa_f1_score


TASKS = ("2wikimqa", "hotpotqa", "qasper")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def output_map(panel, path):
    values = rows(path)
    mapped = {str(row["row_id"]): row for row in values}
    if len(mapped) != len(values) or set(mapped) != set(panel):
        raise ValueError(f"unpaired GLM QA outputs: {path}")
    for row_id, row in mapped.items():
        for field in ("task", "prompt_sha256", "references", "input_tokens"):
            if row.get(field) != panel[row_id].get(field):
                raise ValueError(f"GLM QA input identity drift: {row_id}/{field}")
    return mapped


def score(panel, output):
    grouped = defaultdict(list)
    for row_id, prompt in panel.items():
        grouped[prompt["task"]].append(qa_f1_score(output[row_id]["output_text"], prompt["references"]))
    by_task = {task: sum(values) / len(values) for task, values in sorted(grouped.items())}
    return {"score": sum(by_task.values()) / len(by_task), "by_task": by_task}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    panel_path = args.root / "assets/qa/inputs.jsonl"
    panel = {str(row["row_id"]): row for row in rows(panel_path)}
    if Counter(row["task"] for row in panel.values()) != Counter({task: 20 for task in TASKS}):
        raise ValueError("GLM QA panel is not three tasks x 20")
    native_path = args.root / "runs/native/generations.jsonl"
    nts2_path = args.root / "runs/nts2/generations.jsonl"
    native = score(panel, output_map(panel, native_path))
    nts2 = score(panel, output_map(panel, nts2_path))
    report = {
        "status": "GLM_NTS2_NATIVE_QA_TRANSFER_COMPLETE_V1",
        "model": "GLM-4-9B-0414",
        "method": "native_tailspline_s2_midgain_v1",
        "length": 32768,
        "rows_per_arm": 60,
        "native": native,
        "nts2": nts2,
        "nts2_minus_native": nts2["score"] - native["score"],
        "input_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
    }
    path = args.root / "reports/decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "delta": report["nts2_minus_native"]}))


if __name__ == "__main__":
    main()
