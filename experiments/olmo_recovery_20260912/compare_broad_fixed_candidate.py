#!/usr/bin/env python3
"""Compare one frozen candidate with archived row-matched broad-panel baselines."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score


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
        if candidate:
            by_task[task] = {
                "rows": len(items),
                "official": mean(row["ruler_official_score"] for row in items),
                "exact_plus_eos": mean(row["exact_plus_eos"] for row in items),
                "eos_rate": mean(row["ended_eos"] for row in items),
                "cap_rate": mean(row["hit_cap"] for row in items),
            }
        else:
            by_task[task] = {
                "rows": len(items),
                "official": mean(row["correct"] for row in items),
                "eos_rate": mean(row["ended_eos"] for row in items),
            }
    return {
        "by_task": by_task,
        "task_equal_official": mean(value["official"] for value in by_task.values()),
        "rows": sum(value["rows"] for value in by_task.values()),
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
    if len(expected) != len(expected_rows) or len(expected) != 350:
        raise ValueError("expected the archived 350-row broad panel")
    tasks = sorted({row["task"] for row in expected.values()})
    if len(tasks) != 7 or {row["length_cap"] for row in expected.values()} != {16384}:
        raise ValueError("broad panel task or length identity differs")

    candidate_rows = read_rows(args.candidate_run / "generations.jsonl")
    candidate = {row["row_id"]: row for row in candidate_rows}
    if len(candidate) != len(candidate_rows) or set(candidate) != set(expected):
        raise ValueError("candidate generations do not match the archived panel")
    for row_id, row in candidate.items():
        source = expected[row_id]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"candidate row differs at {field}: {row_id}")
        recomputed = official_score(source, row["output_text"])
        if row["ruler_official_score"] != recomputed:
            raise ValueError(f"candidate official score drift: {row_id}")
    summaries = {"candidate": summarize(candidate, tasks, candidate=True)}
    baseline_rows = {}
    for label, path in args.baseline:
        rows = read_rows(path)
        data = {row["row_id"]: row for row in rows}
        if label in baseline_rows or len(data) != len(rows) or set(data) != set(expected):
            raise ValueError(f"baseline rows are duplicated or unmatched: {label}")
        for row_id, row in data.items():
            source = expected[row_id]
            if row["task"] != source["task"] or row["length_cap"] != source["length_cap"]:
                raise ValueError(f"baseline row identity differs: {label}/{row_id}")
            recomputed = official_score(source, row["output_text"])
            if row["correct"] != recomputed:
                raise ValueError(f"baseline official score drift: {label}/{row_id}")
        baseline_rows[label] = data
        summaries[label] = summarize(data, tasks, candidate=False)

    contrasts = {}
    for label, data in baseline_rows.items():
        task_deltas = {
            task: (
                summaries["candidate"]["by_task"][task]["official"]
                - summaries[label]["by_task"][task]["official"]
            )
            for task in tasks
        }
        contrasts[f"candidate_minus_{label}"] = {
            "task_equal_official": summaries["candidate"]["task_equal_official"] - summaries[label]["task_equal_official"],
            "by_task": task_deltas,
            "paired_row_wins": sum(candidate[row_id]["ruler_official_score"] > data[row_id]["correct"] for row_id in expected),
            "paired_row_losses": sum(candidate[row_id]["ruler_official_score"] < data[row_id]["correct"] for row_id in expected),
            "paired_row_ties": sum(candidate[row_id]["ruler_official_score"] == data[row_id]["correct"] for row_id in expected),
        }
    result = {
        "status": "COMPLETE",
        "panel": "archived OLMo 350-row seven-task 16K development panel",
        "tasks": tasks,
        "summaries": summaries,
        "contrasts": contrasts,
        "scope": "row-matched broad endpoint development evidence; no 4K Native guard, natural QA, or independent confirmation",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_task_equal_official": summaries["candidate"]["task_equal_official"],
        "contrasts": {key: value["task_equal_official"] for key, value in contrasts.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
