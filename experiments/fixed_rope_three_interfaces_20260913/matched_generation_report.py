#!/usr/bin/env python3
"""Build a paired task-equal range report from completed generation JSONL files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .pipeline import atomic_json, bootstrap_range_contrast


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_sources(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError("--source must be ARM=PATH")
        arm, path = value.split("=", 1)
        if not arm or not Path(path).is_file():
            raise ValueError(f"invalid result source: {value}")
        result[arm].append(Path(path))
    return dict(result)


def normalize_arm(paths: list[Path]) -> list[dict]:
    result = []
    seen = set()
    for path in paths:
        for row in read_jsonl(path):
            prompt = str(row.get("prompt_sha256", ""))
            if not prompt or prompt in seen:
                raise ValueError(f"missing or repeated prompt identity in {path}")
            if "ruler_official_score" not in row:
                raise ValueError(f"row lacks official RULER score in {path}")
            seen.add(prompt)
            result.append({
                **row,
                "prompt_sha256": prompt,
                "official_score": float(row["ruler_official_score"]),
                # Current generated rows do not carry a registered semantic
                # cluster spanning lengths, so bootstrap pairs within each
                # task-length cell rather than inventing one.
                "mini_semantic_id": prompt,
            })
    return result


def log_auc(curve: dict[int, float], lengths: list[int]) -> float:
    return sum(
        0.5 * (curve[left] + curve[right]) * math.log(right / left)
        for left, right in zip(lengths, lengths[1:])
    ) / math.log(lengths[-1] / lengths[0])


def family(task: str) -> str:
    if task.startswith("niah_"):
        return "retrieval"
    if task == "vt":
        return "tracking"
    if task in {"cwe", "fwe"}:
        return "aggregation"
    return "qa"


def summarize(rows: list[dict], *, tasks: list[str], lengths: list[int]) -> dict:
    by_length = {}
    task_curves = {task: {} for task in tasks}
    for length in lengths:
        task_entries = {}
        for task in tasks:
            cell = [row for row in rows if row["task"] == task and int(row["length_cap"]) == length]
            if not cell:
                raise ValueError(f"empty cell {task}/{length}")
            task_entries[task] = {
                "rows": len(cell),
                "official": float(np.mean([row["official_score"] for row in cell])),
                "eos_rate": float(np.mean([row["ended_eos"] for row in cell])),
                "cap_rate": float(np.mean([row["hit_cap"] for row in cell])),
            }
            task_curves[task][length] = task_entries[task]["official"]
        by_length[str(length)] = {
            "task_macro_official": float(np.mean([value["official"] for value in task_entries.values()])),
            "task_macro_eos_rate": float(np.mean([value["eos_rate"] for value in task_entries.values()])),
            "task_macro_cap_rate": float(np.mean([value["cap_rate"] for value in task_entries.values()])),
            "tasks": task_entries,
        }
    curve = {length: by_length[str(length)]["task_macro_official"] for length in lengths}
    family_tasks: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        family_tasks[family(task)].append(task)
    family_auc = {
        name: log_auc({
            length: float(np.mean([task_curves[task][length] for task in members]))
            for length in lengths
        }, lengths)
        for name, members in family_tasks.items()
    }
    return {
        "rows": len(rows), "rows_per_task_length": sorted({
            by_length[str(length)]["tasks"][task]["rows"] for length in lengths for task in tasks
        }),
        "by_length": by_length,
        "log_length_auc": log_auc(curve, lengths),
        "worst_length_score": min(curve.values()),
        "family_log_length_auc": family_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], required=True, help="ARM=PATH; repeat as needed")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", default=[], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.out.exists() or args.candidate in args.baseline or len(set(args.baseline)) != len(args.baseline):
        raise ValueError("invalid comparison identity or pre-existing output")
    sources = parse_sources(args.source)
    wanted = {args.candidate, *args.baseline}
    if set(sources) != wanted:
        raise ValueError("sources must cover exactly candidate and baselines")
    arms = {name: normalize_arm(paths) for name, paths in sources.items()}
    prompts = {name: {row["prompt_sha256"] for row in rows} for name, rows in arms.items()}
    reference_prompts = prompts[args.candidate]
    if any(value != reference_prompts for value in prompts.values()):
        raise ValueError("comparison arms are not paired on exactly the same prompts")
    tasks = sorted({row["task"] for row in arms[args.candidate]})
    lengths = sorted({int(row["length_cap"]) for row in arms[args.candidate]})
    summaries = {name: summarize(rows, tasks=tasks, lengths=lengths) for name, rows in arms.items()}
    contrasts = {}
    families = {task: family(task) for task in tasks}
    for offset, baseline in enumerate(args.baseline):
        candidate_summary = summaries[args.candidate]
        baseline_summary = summaries[baseline]
        contrasts[baseline] = {
            "delta_log_length_auc": candidate_summary["log_length_auc"] - baseline_summary["log_length_auc"],
            "delta_worst_length_score": candidate_summary["worst_length_score"] - baseline_summary["worst_length_score"],
            "delta_by_length": {
                str(length): candidate_summary["by_length"][str(length)]["task_macro_official"]
                - baseline_summary["by_length"][str(length)]["task_macro_official"]
                for length in lengths
            },
            "bootstrap": bootstrap_range_contrast(
                arms[args.candidate], arms[baseline], tasks=tasks, lengths=lengths,
                task_families=families, draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + offset,
            ),
        }
    report = {
        "status": "MATCHED_GENERATION_RANGE_REPORT_V1",
        "candidate": args.candidate, "baselines": args.baseline,
        "tasks": tasks, "lengths": lengths, "paired_prompts": len(reference_prompts),
        "metric_contract": "RULER official contains -> task-equal length macro -> trapezoidal log-length AUC",
        "source_files": {name: [str(path) for path in paths] for name, paths in sources.items()},
        "summaries": summaries, "contrasts": contrasts,
        "uncertainty_note": "positive-resample fraction is not a posterior probability or automatic gate",
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"], "paired_prompts": report["paired_prompts"],
        "auc": {name: value["log_length_auc"] for name, value in summaries.items()},
        "worst": {name: value["worst_length_score"] for name, value in summaries.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
