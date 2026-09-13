#!/usr/bin/env python3
"""Validate and compare the four completed E3 frozen-table arms."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path

from experiments.rope_fast_5090_20260912.e3_postprocess import percentile
from experiments.rope_fast_5090_20260912.e3_validate import COUNTS, TASKS, load_rows
from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score


ARM_FILES = {
    "C42": ("original", "C42.jsonl"),
    "C42V24": ("original", "C42V24.jsonl"),
    "BM_g4": ("strong", "BM_g4.jsonl"),
    "MrPro_g4": ("strong", "MrPro_g4.jsonl"),
}
CONTRASTS = (
    ("C42V24", "C42"),
    ("BM_g4", "C42"),
    ("BM_g4", "C42V24"),
    ("MrPro_g4", "C42"),
    ("MrPro_g4", "C42V24"),
    ("BM_g4", "MrPro_g4"),
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_091_301


def task_stratified_bootstrap(
    differences: dict[str, list[float]], *, draws: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        task_means = []
        for task in TASKS:
            cell = differences[task]
            task_means.append(sum(cell[rng.randrange(len(cell))] for _ in cell) / len(cell))
        estimates.append(sum(task_means) / len(task_means))
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--strong-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prepared_rows = load_rows(args.prepared / "screen.jsonl")
    prepared = {row["row_id"]: row for row in prepared_rows}
    if len(prepared_rows) != 980 or len(prepared) != 980:
        raise ValueError("prepared E3 panel is not exactly 980 unique rows")
    expected_counts = Counter(
        {(cap, task): count for cap, count in COUNTS.items() for task in TASKS}
    )
    generation = json.loads((args.prepared / "generation_config.json").read_text())
    eos_value = generation["eos_token_id"]
    eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])

    arm_rows: dict[str, dict[str, dict]] = {}
    arm_summary = {}
    roots = {"original": args.original_run, "strong": args.strong_run}
    for arm, (root_name, filename) in ARM_FILES.items():
        records = load_rows(roots[root_name] / filename)
        ids = [record["row_id"] for record in records]
        if len(records) != 980 or len(set(ids)) != 980 or set(ids) != set(prepared):
            raise ValueError(f"{arm} is not one complete duplicate-free 980-row panel")
        counts = Counter((prepared[row_id]["length_cap"], prepared[row_id]["task"]) for row_id in ids)
        if counts != expected_counts:
            raise ValueError(f"{arm} task/cap grid is incomplete")

        cells = defaultdict(list)
        by_id = {}
        for record in records:
            source = prepared[record["row_id"]]
            for field in ("task", "length_cap", "references", "prompt_sha256", "max_new_tokens"):
                if record.get(field) != source[field]:
                    raise ValueError(f"{arm}/{record['row_id']} input identity drift: {field}")
            generated = record.get("generated_ids")
            if not isinstance(generated, list) or not generated or len(generated) > source["max_new_tokens"]:
                raise ValueError(f"{arm}/{record['row_id']} invalid generated token record")
            ended_eos = generated[-1] in eos_ids
            if record.get("ended_eos") != ended_eos:
                raise ValueError(f"{arm}/{record['row_id']} EOS flag differs")
            score = official_score(source, record["output_text"])
            if abs(float(record["correct"]) - score) > 1e-12:
                raise ValueError(f"{arm}/{record['row_id']} official score drift")
            item = {
                "score": score,
                "ended_eos": ended_eos,
                "cap_hit": len(generated) == source["max_new_tokens"] and not ended_eos,
            }
            by_id[record["row_id"]] = item
            cells[(source["length_cap"], source["task"])].append(item)

        by_cap_task = {}
        task_macro_by_cap = {}
        for cap in COUNTS:
            task_means = []
            for task in TASKS:
                items = cells[(cap, task)]
                mean = sum(item["score"] for item in items) / len(items)
                task_means.append(mean)
                by_cap_task[f"{cap}/{task}"] = {
                    "rows": len(items),
                    "official_mean": mean,
                    "eos_rate": sum(item["ended_eos"] for item in items) / len(items),
                    "cap_exhaustion_rate": sum(item["cap_hit"] for item in items) / len(items),
                }
            task_macro_by_cap[str(cap)] = sum(task_means) / len(task_means)
        arm_rows[arm] = by_id
        arm_summary[arm] = {
            "rows": len(records),
            "task_macro_by_cap": task_macro_by_cap,
            "by_cap_task": by_cap_task,
        }

    contrasts = {}
    for treatment, control in CONTRASTS:
        contrast = {}
        for cap in COUNTS:
            differences = {}
            by_task = {}
            for task in TASKS:
                row_ids = [
                    row_id
                    for row_id, row in prepared.items()
                    if row["length_cap"] == cap and row["task"] == task
                ]
                values = [
                    arm_rows[treatment][row_id]["score"] - arm_rows[control][row_id]["score"]
                    for row_id in row_ids
                ]
                differences[task] = values
                by_task[task] = sum(values) / len(values)
            draws = task_stratified_bootstrap(
                differences, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + cap
            )
            contrast[str(cap)] = {
                "estimate": sum(by_task.values()) / len(by_task),
                "paired_task_stratified_bootstrap_ci95": [
                    percentile(draws, 0.025),
                    percentile(draws, 0.975),
                ],
                "by_task": by_task,
            }
        contrasts[f"{treatment}_minus_{control}"] = contrast

    result = {
        "status": "COMPLETE",
        "rows_per_arm": 980,
        "tasks": list(TASKS),
        "arm_sources": {
            "original_run": str(args.original_run.resolve()),
            "strong_run": str(args.strong_run.resolve()),
        },
        "aggregation": "official score; task-equal macro within length",
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed_base": BOOTSTRAP_SEED,
            "method": "paired row bootstrap within each task, then equal-task macro",
        },
        "arms": arm_summary,
        "contrasts": contrasts,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
