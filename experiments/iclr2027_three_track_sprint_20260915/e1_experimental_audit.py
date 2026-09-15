#!/usr/bin/env python3
"""Attach E1 table identity, runtime identity and complete result diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


LENGTHS = (8192, 16384, 32768)
WEIGHTS = (0.25, 0.5, 0.25)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def task_auc(summary: dict, task: str) -> float:
    return sum(
        weight * float(summary["by_length"][str(length)]["tasks"][task]["official"])
        for length, weight in zip(LENGTHS, WEIGHTS)
    )


def run_health(path: Path) -> dict:
    values = rows(path)
    if len(values) != 390:
        raise ValueError(f"E1 raw generation count drift: {path}")
    return {
        "rows": len(values),
        "ended_eos": sum(bool(row["ended_eos"]) for row in values),
        "hit_cap": sum(bool(row["hit_cap"]) for row in values),
        "empty": sum(bool(row.get("empty", not str(row.get("output_text", "")).strip())) for row in values),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--control-table", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = read(args.report)
    candidate = read(args.candidate_table)
    control = read(args.control_table)
    if report.get("status") != "TAILSPLINE_LLAMA_CLASSIC_REPORT_V1":
        raise ValueError("E1 report is incomplete")
    if candidate["band_envelope"] != control["band_envelope"] or candidate["gain"] != control["gain"]:
        raise ValueError("E1 table support/gain drift")
    delta_sum = float(candidate["sum_m"]) - float(control["sum_m"])
    if abs(delta_sum) > 1e-6:
        raise ValueError("E1 installed exponent sums are not matched")
    candidate_summary = report["summaries"]["tailspline"]["full13"]
    control_summary = report["summaries"]["dose_control_c"]["full13"]
    tasks = report["tasks"]
    task_delta = {
        task: task_auc(candidate_summary, task) - task_auc(control_summary, task) for task in tasks
    }
    tolerance = 1e-12
    outcome = {
        "wins": sum(value > tolerance for value in task_delta.values()),
        "losses": sum(value < -tolerance for value in task_delta.values()),
        "ties": sum(abs(value) <= tolerance for value in task_delta.values()),
    }
    low, high = candidate["band_envelope"]
    increments_candidate = candidate["increments"]
    increments_control = control["increments"]
    candidate_contract = read(args.candidate_run / "contract.json")
    control_contract = read(args.control_run / "contract.json")
    contrast = report["contrasts"]["dose_control_c"]
    result = {
        "status": "E1_MATCHED_DISPLACEMENT_EXPERIMENT_AUDIT_COMPLETE_V1",
        "comparison": "TailSpline minus dose-control C",
        "table_identity": {
            "band": candidate["band_envelope"], "scale": candidate["scale"], "gain": candidate["gain"],
            "candidate_sum_m": candidate["sum_m"], "control_sum_m": control["sum_m"],
            "sum_m_delta": delta_sum,
            "candidate_increment_centroid": candidate["increment_centroid"],
            "control_increment_centroid": control["increment_centroid"],
            "entry_increment_delta": increments_candidate[low] - increments_control[low],
            "tail_increment_delta": increments_candidate[high - 1] - increments_control[high - 1],
            "maximum_absolute_exponent_delta": max(
                abs(left - right) for left, right in zip(candidate["exponents"], control["exponents"])
            ),
            "table_sha256_float32": {
                "tailspline": candidate["table_sha256_float32"], "dose_control_c": control["table_sha256_float32"],
            },
        },
        "runtime_identity": {
            "tailspline_batch_size": candidate_contract["batch_size"],
            "dose_control_c_batch_size": control_contract["batch_size"],
            "prefill_chunk_size_equal": candidate_contract["prefill_chunk_size"] == control_contract["prefill_chunk_size"],
            "prompt_set_equal": set(candidate_contract["row_ids"]) == set(control_contract["row_ids"]),
            "remaining_issue": "Batch1 versus batch2 sensitivity is measured by the frozen 39-cell E0 probe.",
        },
        "results": {
            "tailspline_full13_auc": candidate_summary["log_length_auc"],
            "control_full13_auc": control_summary["log_length_auc"],
            "delta_full13_auc": contrast["observed"]["delta_full13_auc"],
            "delta_full13_auc_ci95": contrast["full13"]["delta_log_auc"]["interval95"],
            "delta_niah_auc": contrast["observed"]["delta_niah_auc"],
            "delta_ppl_auc": contrast["observed"]["delta_combined_ppl_auc"],
            "task_auc_delta": task_delta,
            "task_outcome": outcome,
        },
        "output_health": {
            "tailspline": run_health(args.candidate_run / "generations.jsonl"),
            "dose_control_c": run_health(args.control_run / "generations.jsonl"),
        },
        "source_sha256": {
            "report": sha256(args.report), "candidate_table": sha256(args.candidate_table),
            "control_table": sha256(args.control_table),
        },
        "claim_boundary": "E1 isolates a matched-total-displacement shape change, but its current generation comparison is cross-batch until E0 sensitivity is read.",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
