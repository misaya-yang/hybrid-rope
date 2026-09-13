#!/usr/bin/env python3
"""Paired task-stratified bootstrap sensitivity analysis for frozen-table rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("baseline must be LABEL=JSONL")
    return label, Path(path)


def read_rows(paths: list[Path]) -> dict[str, dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate row ids")
    return by_id


def bootstrap_task_equal(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    *,
    candidate_metric: str,
    baseline_metric: str,
    samples: int,
    seed: int,
) -> dict:
    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline row ids differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row_id, candidate_row in candidate.items():
        baseline_row = baseline[row_id]
        if candidate_row["task"] != baseline_row["task"]:
            raise ValueError(f"task identity differs: {row_id}")
        grouped[candidate_row["task"]].append(
            float(candidate_row[candidate_metric]) - float(baseline_row[baseline_metric])
        )
    task_arrays = {task: np.asarray(values, dtype=np.float64) for task, values in grouped.items()}
    point_by_task = {task: float(values.mean()) for task, values in task_arrays.items()}
    point = float(np.mean(list(point_by_task.values())))
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        task_means = [float(values[rng.integers(0, len(values), len(values))].mean()) for values in task_arrays.values()]
        draws[index] = np.mean(task_means)
    return {
        "point_delta": point,
        "row_stratified_percentile_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_positive": float(np.mean(draws > 0.0)),
        "by_task_point_delta": point_by_task,
        "rows": len(candidate),
        "tasks": sorted(task_arrays),
        "samples": samples,
        "seed": seed,
        "interpretation": "row-stratified sensitivity interval; not a population-level confidence guarantee",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=parse_labeled_path, action="append", required=True)
    parser.add_argument("--candidate-metric", required=True)
    parser.add_argument("--baseline-metric", required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate = read_rows([args.candidate])
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for label, path in args.baseline:
        grouped_paths[label].append(path)
    results = {
        label: bootstrap_task_equal(
            candidate,
            read_rows(paths),
            candidate_metric=args.candidate_metric,
            baseline_metric=args.baseline_metric,
            samples=args.samples,
            seed=args.seed,
        )
        for label, paths in grouped_paths.items()
    }
    payload = {"status": "COMPLETE", "contrasts": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
