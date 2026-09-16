#!/usr/bin/env python3
"""Quantify how RULER task-macro deltas vary with rows sampled per task."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sample-sizes", default="5,10,20,50,100,200")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--external-gate-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sample_sizes.split(",") if value.strip())
    if not sizes or sizes != tuple(sorted(set(sizes))) or sizes[-1] <= 0 or args.draws <= 0:
        raise ValueError("invalid sample sizes or draws")

    panel = {row["row_id"]: row for row in read_jsonl(args.panel)}
    candidate = {row["row_id"]: row for row in read_jsonl(args.candidate)}
    baseline = {row["row_id"]: row for row in read_jsonl(args.baseline)}
    if set(panel) != set(candidate) or set(panel) != set(baseline):
        raise ValueError("panel and paired outputs do not contain identical row IDs")
    by_task: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row_id, source in panel.items():
        task = str(source["task"])
        left, right = candidate[row_id], baseline[row_id]
        if any(left.get(key) != right.get(key) or left.get(key) != source.get(key)
               for key in ("task", "prompt_sha256", "references")):
            raise ValueError(f"paired row identity drift: {row_id}")
        by_task[task].append((
            int(source["source_order_index"]),
            float(left["ruler_official_score"]) - float(right["ruler_official_score"]),
        ))
    if set(by_task) != set(TASKS):
        raise ValueError("RULER-13 task coverage drift")
    arrays = {}
    for task in TASKS:
        ordered = sorted(by_task[task])
        if [index for index, _ in ordered] != list(range(len(ordered))):
            raise ValueError(f"source-order indices drift: {task}")
        arrays[task] = np.asarray([value for _, value in ordered], dtype=np.float64)
    population = len(next(iter(arrays.values())))
    if any(len(values) != population for values in arrays.values()) or any(n > population for n in sizes):
        raise ValueError("sample size exceeds the common per-task population")
    full_delta = float(np.mean([values.mean() for values in arrays.values()]))
    external = None
    if args.external_gate_report:
        gate = json.loads(args.external_gate_report.read_text())
        external = {
            "model": gate.get("model"), "scale": gate.get("scale"),
            "evaluation_length": gate.get("evaluation_length"),
            "rows_per_task": gate.get("ruler", {}).get("rows_per_task"),
            "delta": gate.get("ruler", {}).get("delta_tailspline_minus_mrpro"),
            "ci95": gate.get("ruler", {}).get("paired_inference", {}).get("ci95"),
            "comparison_boundary": (
                "different S and absolute length; shown alongside sampling calibration but not "
                "interpreted as a sample-size-only difference"
            ),
        }
    rng = np.random.default_rng(args.seed)
    results = {}
    for n in sizes:
        nested = float(np.mean([values[:n].mean() for values in arrays.values()]))
        draws = np.empty(args.draws, dtype=np.float64)
        for draw in range(args.draws):
            draws[draw] = float(np.mean([
                rng.choice(values, size=n, replace=False).mean() for values in arrays.values()
            ]))
        deviation = draws - full_delta
        random_summary = {
            "mean": float(draws.mean()),
            "std": float(draws.std(ddof=1)),
            "interval95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "probability_positive": float(np.mean(draws > 0)),
            "absolute_error_median": float(np.median(np.abs(deviation))),
            "absolute_error_p95": float(np.quantile(np.abs(deviation), 0.95)),
        }
        if (external and n == external.get("rows_per_task")
                and external.get("delta") is not None):
            random_summary["fraction_at_least_external_gate_delta"] = float(
                np.mean(draws >= float(external["delta"]))
            )
            random_summary["external_comparison_is_cross_condition"] = True
        results[str(n)] = {
            "nested_source_order_delta": nested,
            "nested_minus_full": nested - full_delta,
            "nested_to_full_ratio": nested / full_delta if full_delta else None,
            "random_subsample": random_summary,
        }
    report = {
        "status": "RULER_SAMPLING_STABILITY_COMPLETE_V1",
        "identity": {
            "tasks": list(TASKS), "population_rows_per_task": population,
            "paired_population_rows": population * len(TASKS),
            "candidate": str(args.candidate), "baseline": str(args.baseline),
            "selection": "nested source-order prefixes plus uniform without-replacement subsamples within each task",
        },
        "full_population_delta": full_delta,
        "draws_per_sample_size": args.draws,
        "seed": args.seed,
        "sample_sizes": results,
        "external_s16_128k_gate": external,
        "conclusion_boundary": (
            "These intervals are finite-population sampling distributions within the completed "
            "S4/32K 200/task panel, not confidence intervals for an external population. "
            "It does not explain S16/128K versus S4/32K differences as sampling alone."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"status": report["status"], "full_delta": full_delta,
                      "nested10": results.get("10", {}).get("nested_source_order_delta")}))


if __name__ == "__main__":
    main()
