#!/usr/bin/env python3
"""Build one-length paired task-equal reports for fixed-table comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .matched_generation_report import normalize_arm, parse_sources
from .pipeline import atomic_json, bootstrap_point_contrast, summarize_point


def build_point_report(
    arms: dict[str, list[dict]], *, candidate: str, baselines: list[str],
    draws: int, seed: int,
) -> dict:
    if candidate not in arms or any(name not in arms for name in baselines):
        raise ValueError("candidate or baseline arm missing")
    prompts = {name: {row["prompt_sha256"] for row in rows} for name, rows in arms.items()}
    reference = prompts[candidate]
    if any(value != reference for value in prompts.values()):
        raise ValueError("point-comparison arms are not paired on exactly the same prompts")
    tasks = sorted({row["task"] for row in arms[candidate]})
    lengths = {int(row["length_cap"]) for rows in arms.values() for row in rows}
    if len(lengths) != 1:
        raise ValueError("matched point report requires exactly one common length")
    length = lengths.pop()
    expected_cells = {(task, length) for task in tasks}
    for name, rows in arms.items():
        if {(row["task"], int(row["length_cap"])) for row in rows} != expected_cells:
            raise ValueError(f"arm {name} has different task coverage")
    summaries = {
        name: summarize_point(rows, tasks, length)
        for name, rows in arms.items()
    }
    contrasts = {}
    for offset, baseline in enumerate(baselines):
        candidate_summary = summaries[candidate]
        baseline_summary = summaries[baseline]
        contrasts[baseline] = {
            "delta_task_macro_official": (
                candidate_summary["task_macro_official"]
                - baseline_summary["task_macro_official"]
            ),
            "delta_by_task": {
                task: (
                    candidate_summary["by_length"][str(length)]["tasks"][task]["official"]
                    - baseline_summary["by_length"][str(length)]["tasks"][task]["official"]
                )
                for task in tasks
            },
            "bootstrap": bootstrap_point_contrast(
                arms[candidate], arms[baseline], tasks=tasks,
                draws=draws, seed=seed + offset,
            ),
        }
    return {
        "status": "MATCHED_GENERATION_POINT_REPORT_V1",
        "candidate": candidate,
        "baselines": baselines,
        "length": length,
        "tasks": tasks,
        "paired_prompts": len(reference),
        "metric_contract": "RULER official contains -> task-equal one-length macro",
        "summaries": summaries,
        "contrasts": contrasts,
        "uncertainty_note": "positive-resample fraction is not a posterior probability or automatic gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], required=True, help="ARM=PATH")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", default=[], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    parser.add_argument(
        "--allow-baseline-superset", action="store_true",
        help="filter baseline-only extra prompts to the candidate prompt set and report the dropped count",
    )
    args = parser.parse_args()
    if args.out.exists() or args.candidate in args.baseline or len(set(args.baseline)) != len(args.baseline):
        raise ValueError("invalid comparison identity or pre-existing output")
    sources = parse_sources(args.source)
    wanted = {args.candidate, *args.baseline}
    if set(sources) != wanted:
        raise ValueError("sources must cover exactly candidate and baselines")
    arms = {name: normalize_arm(paths) for name, paths in sources.items()}
    source_row_counts = {name: len(rows) for name, rows in arms.items()}
    if args.allow_baseline_superset:
        reference_prompts = {row["prompt_sha256"] for row in arms[args.candidate]}
        arms = {
            name: rows if name == args.candidate else [
                row for row in rows if row["prompt_sha256"] in reference_prompts
            ]
            for name, rows in arms.items()
        }
    result = build_point_report(
        arms, candidate=args.candidate, baselines=args.baseline,
        draws=args.bootstrap_draws, seed=args.bootstrap_seed,
    )
    result["source_files"] = {
        name: [str(path) for path in paths] for name, paths in sources.items()
    }
    result["source_row_counts"] = source_row_counts
    result["matched_row_counts"] = {name: len(rows) for name, rows in arms.items()}
    result["baseline_superset_filtering"] = bool(args.allow_baseline_superset)
    atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "paired_prompts": result["paired_prompts"],
        "macro": {
            name: summary["task_macro_official"]
            for name, summary in result["summaries"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
