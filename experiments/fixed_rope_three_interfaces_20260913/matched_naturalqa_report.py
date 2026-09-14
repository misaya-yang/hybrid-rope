#!/usr/bin/env python3
"""Validate and compare two paired natural-QA generation files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.eval.longbench_metrics import qa_f1_score


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_macro(values: dict[str, dict], ids: list[str]) -> tuple[float, dict[str, float]]:
    cells = defaultdict(list)
    for row_id in ids:
        cells[values[row_id]["task"]].append(float(values[row_id]["whole_response_f1"]))
    means = {task: float(np.mean(cells[task])) for task in TASKS if cells[task]}
    if set(means) != set(TASKS):
        raise ValueError(f"task-equal macro lacks tasks: {set(TASKS) - set(means)}")
    return float(np.mean(list(means.values()))), means


def bootstrap(panel, candidate, baseline, ids, *, draws=20_000, seed=20260914):
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in TASKS:
        clusters = defaultdict(list)
        for row_id in ids:
            if panel[row_id]["task"] == task:
                clusters[panel[row_id]["document_cluster_id"]].append(row_id)
        cluster_sums = np.asarray([
            sum(candidate[row_id]["whole_response_f1"] - baseline[row_id]["whole_response_f1"] for row_id in group)
            for group in clusters.values()
        ])
        cluster_counts = np.asarray([len(group) for group in clusters.values()])
        sampled = rng.integers(len(cluster_sums), size=(draws, len(cluster_sums)))
        task_draws.append(
            cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1)
        )
    result = np.mean(task_draws, axis=0)
    return {
        "bootstrap_mean": float(np.mean(result)),
        "ci95": np.quantile(result, [0.025, 0.975]).tolist(),
        "probability_delta_gt_zero": float(np.mean(result > 0)),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-name", default="tailspline")
    parser.add_argument("--baseline-name", default="mrpro")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel_rows = rows(args.panel)
    panel = {str(row["row_id"]): row for row in panel_rows}
    if len(panel_rows) != 631 or len(panel) != 631:
        raise ValueError("natural-QA panel is not the frozen 631-row pool")
    outputs = {}
    for name, path in ((args.candidate_name, args.candidate), (args.baseline_name, args.baseline)):
        values = rows(path)
        mapping = {str(row["row_id"]): row for row in values}
        if len(values) != 631 or set(mapping) != set(panel):
            raise ValueError(f"unpaired natural-QA rows: {name}")
        for row_id, row in mapping.items():
            source = panel[row_id]
            for key in ("task", "prompt_sha256", "input_tokens", "references"):
                if row.get(key) != source.get(key):
                    raise ValueError(f"input identity drift: {name}/{row_id}/{key}")
            score = qa_f1_score(row["output_text"], row["references"])
            if abs(score - float(row["whole_response_f1"])) > 1e-12:
                raise ValueError(f"score drift: {name}/{row_id}")
        outputs[name] = mapping

    ids = list(panel)
    candidate_macro, candidate_tasks = task_macro(outputs[args.candidate_name], ids)
    baseline_macro, baseline_tasks = task_macro(outputs[args.baseline_name], ids)
    llama_extended = [row_id for row_id in ids if panel[row_id]["llama_native_stratum"] == "extended"]
    extended_counts = Counter(panel[row_id]["task"] for row_id in llama_extended)
    extended_result = {"rows": len(llama_extended), "rows_by_task": dict(extended_counts)}
    if set(extended_counts) == set(TASKS):
        extended_candidate, extended_candidate_tasks = task_macro(
            outputs[args.candidate_name], llama_extended,
        )
        extended_baseline, extended_baseline_tasks = task_macro(
            outputs[args.baseline_name], llama_extended,
        )
        extended_result.update({
            "candidate_macro_f1": extended_candidate,
            "baseline_macro_f1": extended_baseline,
            "candidate_minus_baseline": {
                "estimate": extended_candidate - extended_baseline,
                "by_task": {
                    task: extended_candidate_tasks[task] - extended_baseline_tasks[task]
                    for task in TASKS
                },
                **bootstrap(
                    panel, outputs[args.candidate_name], outputs[args.baseline_name],
                    llama_extended, seed=20260915,
                ),
            },
        })
    else:
        extended_result["task_equal_macro"] = None
        extended_result["note"] = "At least one task has no >8K Llama row; only per-task reporting is valid."
    result = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_LLAMA_NATURAL_QA_FROZEN631_PAIRED_V2",
        "rows_per_arm": 631,
        "tasks": list(TASKS),
        "primary_pool": "exact historical 631-row OLMo-tokenizer >4096 frozen source pool, retokenized from original LongBench rows for Llama",
        "metric": "five-task equal macro whole-response LongBench-normalized token F1",
        "arms": {
            args.candidate_name: {"macro_f1": candidate_macro, "by_task": candidate_tasks},
            args.baseline_name: {"macro_f1": baseline_macro, "by_task": baseline_tasks},
        },
        "candidate_minus_baseline": {
            "estimate": candidate_macro - baseline_macro,
            "by_task": {task: candidate_tasks[task] - baseline_tasks[task] for task in TASKS},
            **bootstrap(panel, outputs[args.candidate_name], outputs[args.baseline_name], ids),
        },
        "llama_length_audit": {
            "native_length": 8192,
            "within_native_rows": len(ids) - len(llama_extended),
            "extended_rows": len(llama_extended),
            "input_token_min": min(row["input_tokens"] for row in panel_rows),
            "input_token_max": max(row["input_tokens"] for row in panel_rows),
            "rows_by_task": dict(Counter(row["task"] for row in panel_rows)),
            "extended_effect": extended_result,
        },
        "output_health": {
            name: {
                "ended_eos": sum(bool(row["ended_eos"]) for row in values.values()),
                "hit_cap": sum(bool(row["hit_cap"]) for row in values.values()),
                "empty": sum(bool(row["empty"]) for row in values.values()),
            }
            for name, values in outputs.items()
        },
        "raw_sha256": {args.candidate_name: sha256(args.candidate), args.baseline_name: sha256(args.baseline)},
        "panel_sha256": sha256(args.panel),
        "bootstrap_estimand": "Within each task, resample source_context_sha256 clusters and divide sampled cluster score sums by sampled question counts; then average task means equally. This matches the row-weighted point estimand while preserving document clustering.",
        "scope": "Paired five-task natural-QA transfer evidence on a previously defined finite source pool; not full LongBench and not an independent sample.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"estimate": result["candidate_minus_baseline"]["estimate"], "ci95": result["candidate_minus_baseline"]["ci95"]}))


if __name__ == "__main__":
    main()
