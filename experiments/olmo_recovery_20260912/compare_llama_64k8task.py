#!/usr/bin/env python3
"""Compare one Llama 64K eight-task candidate with archived runner-matched baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def output_text(row: dict) -> str:
    return row.get("output_text", row.get("output", row.get("raw_text")))


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def validate_and_score(rows: list[dict], expected: dict[str, dict]) -> dict[str, dict]:
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError("generation rows are duplicated or differ from the panel")
    scored = {}
    for row_id, row in by_id.items():
        source = expected[row_id]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"row identity differs at {field}: {row_id}")
        scored[row_id] = {
            **row,
            "recomputed_official": float(official_score(source, output_text(row))),
        }
    return scored


def summarize(rows: dict[str, dict], tasks: list[str]) -> dict:
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["task"]].append(row)
    by_task = {}
    for task in tasks:
        items = groups[task]
        if len(items) != 4:
            raise ValueError(f"expected four rows for {task}")
        by_task[task] = {
            "rows": len(items),
            "official": mean(row["recomputed_official"] for row in items),
            "strict": mean(float(row.get("strict_score", 0.0)) for row in items),
            "eos_rate": mean(float(row.get("ended_eos", row.get("eos_seen", False))) for row in items),
        }
    return {
        "by_task": by_task,
        "task_equal_official": mean(value["official"] for value in by_task.values()),
        "task_equal_strict": mean(value["strict"] for value in by_task.values()),
        "rows": len(rows),
    }


def parse_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_baseline, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel_rows = read_rows(args.panel)
    expected = {row["row_id"]: row for row in panel_rows}
    tasks = sorted({row["task"] for row in panel_rows})
    if len(expected) != 32 or len(tasks) != 8 or {row["length_cap"] for row in panel_rows} != {65536}:
        raise ValueError("expected the frozen 32-row, eight-task 64K panel")
    candidate = validate_and_score(read_rows(args.candidate), expected)
    summaries = {"candidate": summarize(candidate, tasks)}
    contrasts = {}
    for label, path in args.baseline:
        if label in summaries:
            raise ValueError(f"duplicate baseline: {label}")
        baseline = validate_and_score(read_rows(path), expected)
        summaries[label] = summarize(baseline, tasks)
        contrasts[f"candidate_minus_{label}"] = {
            "task_equal_official": summaries["candidate"]["task_equal_official"]
            - summaries[label]["task_equal_official"],
            "by_task": {
                task: summaries["candidate"]["by_task"][task]["official"]
                - summaries[label]["by_task"][task]["official"]
                for task in tasks
            },
            "paired_row_wins": sum(candidate[row_id]["recomputed_official"] > baseline[row_id]["recomputed_official"] for row_id in expected),
            "paired_row_losses": sum(candidate[row_id]["recomputed_official"] < baseline[row_id]["recomputed_official"] for row_id in expected),
            "paired_row_ties": sum(candidate[row_id]["recomputed_official"] == baseline[row_id]["recomputed_official"] for row_id in expected),
        }
    result = {
        "status": "COMPLETE",
        "panel": "archived runner-matched Llama 64K eight-task compact panel",
        "tasks": tasks,
        "summaries": summaries,
        "contrasts": contrasts,
        "scope": "endpoint task-breadth development evidence; eight tasks x four rows, not full-interval confirmation",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_task_equal_official": summaries["candidate"]["task_equal_official"],
        "contrasts": {key: value["task_equal_official"] for key, value in contrasts.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
