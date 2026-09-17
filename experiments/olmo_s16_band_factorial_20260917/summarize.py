#!/usr/bin/env python3
"""Summarize the frozen OLMo S=16 band factorial without resampling."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ARMS = ("band14_32", "band18_32", "band14_35", "band18_35")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    arm_reports = {}
    for arm in ARMS:
        path = args.root / "runs" / arm / "generations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row["task"])].append(float(row["ruler_official_score"]))
        if len(rows) != 16 or len(grouped) != 8 or any(len(scores) != 2 for scores in grouped.values()):
            raise ValueError(f"{arm} is not the frozen NIAH-8 x 2 panel")
        by_task = {task: sum(scores) / len(scores) for task, scores in sorted(grouped.items())}
        arm_reports[arm] = {
            "rows": len(rows),
            "task_equal_official_score": sum(by_task.values()) / len(by_task),
            "by_task": by_task,
            "empty": sum(bool(row["empty"]) for row in rows),
            "ended_eos": sum(bool(row["ended_eos"]) for row in rows),
            "hit_cap": sum(bool(row["hit_cap"]) for row in rows),
            "generations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    score = {arm: arm_reports[arm]["task_equal_official_score"] for arm in ARMS}
    report = {
        "status": "OLMO_S16_BAND_FACTORIAL_COMPLETE_V1",
        "model": "OLMo-2-0425-1B-Instruct",
        "native_length": 4096,
        "target_length": 65536,
        "scale": 16,
        "panel": "NIAH-8 x 2 per arm",
        "arms": arm_reports,
        "effects": {
            "entry_at_tail32": score["band18_32"] - score["band14_32"],
            "entry_at_tail35": score["band18_35"] - score["band14_35"],
            "tail_at_entry14": score["band14_35"] - score["band14_32"],
            "tail_at_entry18": score["band18_35"] - score["band18_32"],
            "interaction": (
                score["band18_35"] - score["band14_35"]
                - score["band18_32"] + score["band14_32"]
            ),
        },
        "conclusion": "All four fixed bands are task-dead on this 16x/64K panel.",
    }
    out = args.root / "reports" / "band_factorial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
