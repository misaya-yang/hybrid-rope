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


def qualify_runtime(candidate_contract, control_contract, candidate_rows, control_rows):
    """Fail known identity drift; missing historical metadata never becomes PASS."""
    keys = ("batch_size", "prefill_chunk_size", "row_ids", "generation_length_caps",
            "left_pad_batches", "row_split", "unadapted", "base_arm")
    differences = [key for key in keys if candidate_contract.get(key, False if key == "left_pad_batches" else None)
                   != control_contract.get(key, False if key == "left_pad_batches" else None)]
    if any(not row.get("prompt_sha256") for row in candidate_rows + control_rows):
        raise ValueError("E1 raw lacks prompt identity")
    candidate = {row["eval_id"]: row for row in candidate_rows}
    control = {row["eval_id"]: row for row in control_rows}
    if len(candidate) != 390 or len(control) != 390 or set(candidate) != set(control):
        raise ValueError("Incomplete or duplicated E1 raw identities")
    for key in candidate:
        for field in ("prompt_sha256", "input_tokens", "length_cap", "task", "references", "max_new_tokens"):
            if candidate[key].get(field) != control[key].get(field):
                raise ValueError(f"E1 raw identity drift: {key}/{field}")
    # These receipts are optional for old runs, but equality of missing values
    # cannot establish checkpoint, tokenizer or implementation identity.
    required = ("model_revision", "tokenizer_revision", "chat_template_sha256",
                "backend", "precision", "positions_mask_identity", "decoder_identity", "scorer_identity")
    missing = [key for key in required if candidate_contract.get(key) is None or control_contract.get(key) is None]
    differences += [key for key in required if key not in missing and candidate_contract[key] != control_contract[key]]
    hard = [key for key in differences if key not in {"batch_size", "row_ids"}]
    status = "FAIL" if hard else "QUALIFIED_ONLY" if differences or missing else "PASS"
    return {"status": status, "different_fields": differences, "unrecorded_fields": missing,
            "probe_scope": "E0 covers 39 TailSpline rows only; it cannot promote a full T/C comparison to PASS."}


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
    cv, uv = candidate["table"]["values_float32"], control["table"]["values_float32"]
    lo, hi = candidate["band_envelope"]
    if len(cv) != len(uv) or cv[:lo+1] != uv[:lo+1] or cv[hi:] != uv[hi:] or candidate["scale"] != control["scale"]:
        raise ValueError("E1 actual support or outer bands differ")
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
    candidate_rows = rows(args.candidate_run / "generations.jsonl")
    control_rows = rows(args.control_run / "generations.jsonl")
    runtime = qualify_runtime(candidate_contract, control_contract, candidate_rows, control_rows)
    # Recompute the point estimand from raw scores, rather than trusting report status.
    for values, summary in [(candidate_rows, candidate_summary), (control_rows, control_summary)]:
        for task in tasks:
            raw_auc = 0.0
            for length, weight in zip(LENGTHS, WEIGHTS):
                cell = [r for r in values if r["task"] == task and int(r["length_cap"]) == length]
                if len(cell) != 10:
                    raise ValueError("E1 must contain 10 rows in every task-length cell")
                raw_auc += weight * sum(float(r["ruler_official_score"]) for r in cell) / len(cell)
            if abs(raw_auc - task_auc(summary, task)) > 1e-12:
                raise ValueError("E1 report disagrees with raw scores")
    lm = {}
    for name, run in [("tailspline", args.candidate_run), ("dose_control_c", args.control_run)]:
        lm_path = run / "lm_rows.jsonl"
        values = rows(lm_path)
        identities = {(int(row["document"]), int(row["length"])) for row in values}
        if len(values) != 138 or identities != {(d, length) for d in range(46) for length in LENGTHS}:
            raise ValueError("E1 LM raw is not the complete matched 46 x 3 panel")
        lm[name] = {"rows": len(values), "sha256": sha256(lm_path)}
    contrast = report["contrasts"]["dose_control_c"]
    result = {
        "status": "E1_MATCHED_DISPLACEMENT_EXPERIMENT_AUDIT_COMPLETE_V2",
        "mathematical_constraints": "PASS",
        "runtime_match": runtime["status"],
        "raw_complete": True,
        "evidence_role": "confirmatory_shape" if runtime["status"] == "PASS" else "cross_runtime_diagnostic",
        "primary_estimand": "task_equal_full13_log_auc_T_minus_C",
        "supports_tail_only_mediation": False,
        "runtime_qualification": runtime,
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
            "remaining_issue": "A 39-row E0 probe does not establish complete T/C runtime equivalence.",
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
        "lm_raw_health": lm,
        "source_sha256": {
            "report": sha256(args.report), "candidate_table": sha256(args.candidate_table),
            "control_table": sha256(args.control_table),
        },
        "claim_boundary": "Matched-dose mathematics; task attribution requires full runtime identity. E0 alone cannot qualify E1. No tail-only mediation or equivalence claim follows.",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
