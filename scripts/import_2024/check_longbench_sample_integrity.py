#!/usr/bin/env python3
"""
Audit per-task LongBench sample completeness for fair-protocol run directories.

This script checks whether each task JSON contains full per-sample scores or only
preview examples, and emits a machine-readable report for paper risk control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True, help="Run root containing downstream_eval_* profiles")
    parser.add_argument("--expected_num_scored", type=int, default=80)
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args()


def load_tasks(method_json: Path) -> dict:
    try:
        data = json.loads(method_json.read_text())
    except Exception:
        return {}
    models = data.get("models", {})
    if not isinstance(models, dict):
        return {}
    # Prefer hybrid_lora, fallback to first model key.
    if "hybrid_lora" in models and isinstance(models["hybrid_lora"], dict):
        return models["hybrid_lora"].get("tasks", {}) or {}
    for v in models.values():
        if isinstance(v, dict) and isinstance(v.get("tasks"), dict):
            return v["tasks"]
    return {}


def classify_task(task_blob: dict, expected_num_scored: int) -> dict:
    num_scored = task_blob.get("num_scored")
    per_sample = task_blob.get("per_sample_scores", [])
    examples = task_blob.get("examples", [])

    per_sample_len = len(per_sample) if isinstance(per_sample, list) else 0
    examples_len = len(examples) if isinstance(examples, list) else 0

    if isinstance(num_scored, int) and per_sample_len == num_scored and per_sample_len > 0:
        status = "full_per_sample"
    elif per_sample_len == 0 and isinstance(num_scored, int) and num_scored == expected_num_scored and examples_len > 0:
        status = "preview_only"
    elif per_sample_len > 0:
        status = "partial_per_sample"
    else:
        status = "missing"

    return {
        "status": status,
        "num_scored": num_scored,
        "per_sample_len": per_sample_len,
        "examples_len": examples_len,
    }


def recommendation_from_counts(status_counts: dict[str, int]) -> str:
    preview_only = status_counts["preview_only"]
    full_per_sample = status_counts["full_per_sample"]
    if preview_only > 0 and full_per_sample == 0:
        return (
            "Use task-level paired analysis (n=6) in main text; move per-sample bootstrap "
            "to appendix only after full traces are available."
        )
    return "Per-sample significance can be reported if all primary comparisons are from full_per_sample traces."


def audit(run_dir: Path, expected_num_scored: int) -> dict:
    profiles = sorted([p for p in run_dir.glob("downstream_eval_*") if p.is_dir()])
    results = {"run_dir": str(run_dir), "expected_num_scored": expected_num_scored, "profiles": {}}

    status_counts = {"full_per_sample": 0, "preview_only": 0, "partial_per_sample": 0, "missing": 0}

    for profile_dir in profiles:
        longbench_dir = profile_dir / "longbench"
        if not longbench_dir.is_dir():
            continue
        profile_report = {}
        for method_json in sorted(longbench_dir.glob("*.json")):
            method = method_json.stem
            tasks = load_tasks(method_json)
            task_report = {}
            for task_name, task_blob in tasks.items():
                if not isinstance(task_blob, dict):
                    continue
                entry = classify_task(task_blob, expected_num_scored)
                task_report[task_name] = entry
                status_counts[entry["status"]] += 1
            profile_report[method] = task_report
        results["profiles"][profile_dir.name] = profile_report

    total = sum(status_counts.values())
    results["summary"] = {
        "status_counts": status_counts,
        "total_task_entries": total,
        "has_full_per_sample": status_counts["full_per_sample"] > 0,
        "has_preview_only_risk": status_counts["preview_only"] > 0,
        "recommendation": recommendation_from_counts(status_counts),
    }
    return results


def main() -> None:
    args = parse_args()
    report = audit(args.run_dir, args.expected_num_scored)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    print("[ok] wrote integrity report:", args.out_json)
    print("[summary]", json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
