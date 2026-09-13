#!/usr/bin/env python3
"""Compare one frozen table with archived row-matched natural-QA baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.eval.longbench_metrics import qa_f1_score


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    return label, Path(path)


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def summarize(rows: dict[str, dict], tasks: list[str], *, candidate: bool) -> dict:
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["task"]].append(row)
    by_task = {}
    for task in tasks:
        items = groups[task]
        if not items:
            raise ValueError(f"missing task: {task}")
        by_task[task] = {
            "rows": len(items),
            "whole_response_f1": mean(
                row["whole_response_f1"] if candidate else row["correct"] for row in items
            ),
            "eos_rate": mean(row["ended_eos"] for row in items),
        }
        if candidate:
            by_task[task].update(
                exact_plus_eos=mean(row["exact_plus_eos"] for row in items),
                cap_rate=mean(row["hit_cap"] for row in items),
            )
    return {
        "rows": sum(value["rows"] for value in by_task.values()),
        "by_task": by_task,
        "task_equal_whole_response_f1": mean(value["whole_response_f1"] for value in by_task.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_baseline, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected_rows = read_rows(args.prepared / "screen.jsonl")
    expected = {row["row_id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows) or len(expected) != 391:
        raise ValueError("expected the archived 391-row natural-QA union")
    tasks = sorted({row["task"] for row in expected.values()})
    if len(tasks) != 5 or {row["length_cap"] for row in expected.values()} != {16384}:
        raise ValueError("natural panel task or length identity differs")

    candidate_rows = read_rows(args.candidate_run / "generations.jsonl")
    candidate = {row["row_id"]: row for row in candidate_rows}
    if len(candidate) != len(candidate_rows) or set(candidate) != set(expected):
        raise ValueError("candidate generations do not match the natural panel")
    for row_id, row in candidate.items():
        source = expected[row_id]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"candidate row differs at {field}: {row_id}")
        recomputed = qa_f1_score(row["output_text"], source["references"])
        if abs(row["whole_response_f1"] - recomputed) > 1e-12:
            raise ValueError(f"candidate F1 drift: {row_id}")

    assembled = defaultdict(list)
    for label, path in args.baseline:
        assembled[label].extend(read_rows(path))
    baseline_rows = {}
    summaries = {"candidate": summarize(candidate, tasks, candidate=True)}
    for label, rows in assembled.items():
        data = {row["row_id"]: row for row in rows}
        if len(data) != len(rows) or set(data) != set(expected):
            raise ValueError(f"baseline rows are duplicated or unmatched: {label}")
        for row_id, row in data.items():
            source = expected[row_id]
            for field in ("task", "length_cap", "references", "prompt_sha256"):
                if row[field] != source[field]:
                    raise ValueError(f"baseline row differs at {field}: {label}/{row_id}")
            recomputed = qa_f1_score(row["output_text"], source["references"])
            if abs(row["correct"] - recomputed) > 1e-12:
                raise ValueError(f"baseline F1 drift: {label}/{row_id}")
        baseline_rows[label] = data
        summaries[label] = summarize(data, tasks, candidate=False)

    contrasts = {}
    for label, data in baseline_rows.items():
        contrasts[f"candidate_minus_{label}"] = {
            "task_equal_whole_response_f1": (
                summaries["candidate"]["task_equal_whole_response_f1"]
                - summaries[label]["task_equal_whole_response_f1"]
            ),
            "by_task": {
                task: (
                    summaries["candidate"]["by_task"][task]["whole_response_f1"]
                    - summaries[label]["by_task"][task]["whole_response_f1"]
                )
                for task in tasks
            },
            "paired_row_wins": sum(candidate[row_id]["whole_response_f1"] > data[row_id]["correct"] for row_id in expected),
            "paired_row_losses": sum(candidate[row_id]["whole_response_f1"] < data[row_id]["correct"] for row_id in expected),
            "paired_row_ties": sum(candidate[row_id]["whole_response_f1"] == data[row_id]["correct"] for row_id in expected),
        }
    result = {
        "status": "COMPLETE",
        "panel": "archived OLMo 391-row five-task 16K natural-QA union",
        "tasks": tasks,
        "summaries": summaries,
        "contrasts": contrasts,
        "scope": "row-matched natural-QA development evidence; this panel has a known high floor rate and is not independent confirmation",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_task_equal_whole_response_f1": summaries["candidate"]["task_equal_whole_response_f1"],
        "contrasts": {key: value["task_equal_whole_response_f1"] for key, value in contrasts.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
