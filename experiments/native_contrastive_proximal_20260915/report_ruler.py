#!/usr/bin/env python3
"""Build the paired OLMo Native-versus-NCP RULER development report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS


ARMS = ("native", "ncp")
ROWS_PER_TASK = 60
EXPECTED_ROWS = len(TASKS) * ROWS_PER_TASK


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def task_macro(rows: dict[str, dict]) -> tuple[float, dict[str, float]]:
    by_task = {}
    for task in TASKS:
        values = [
            float(row["ruler_official_score"])
            for row in rows.values()
            if row["task"] == task
        ]
        if len(values) != ROWS_PER_TASK:
            raise ValueError(f"{task} does not contain {ROWS_PER_TASK} rows")
        by_task[task] = float(np.mean(values))
    return float(np.mean(list(by_task.values()))), by_task


def paired_task_bootstrap(
    candidate: dict[str, dict], native: dict[str, dict],
    *, draws: int = 20_000, seed: int = 20261109,
) -> dict:
    rng = np.random.default_rng(seed)
    task_draws = []
    task_deltas = []
    pair_deltas = []
    for task in TASKS:
        ids = sorted(row_id for row_id, row in native.items() if row["task"] == task)
        delta = np.asarray([
            float(candidate[row_id]["ruler_official_score"])
            - float(native[row_id]["ruler_official_score"])
            for row_id in ids
        ])
        pair_deltas.extend(delta.tolist())
        task_deltas.append(float(delta.mean()))
        sampled = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
        task_draws.append(sampled)
    values = np.mean(task_draws, axis=0)
    pair_deltas = np.asarray(pair_deltas)
    return {
        "delta": float(np.mean(task_deltas)),
        "paired_task_equal_bootstrap_ci95": interval(values),
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
        "candidate_only_wins": int(np.count_nonzero(pair_deltas > 0.0)),
        "native_only_wins": int(np.count_nonzero(pair_deltas < 0.0)),
        "ties": int(np.count_nonzero(pair_deltas == 0.0)),
    }


def load_run(directory: Path) -> dict[str, dict]:
    status = json.loads((directory / "status.json").read_text())
    rows = read_jsonl(directory / "generations.jsonl")
    if status != {"status": "COMPLETE", "rows": EXPECTED_ROWS, "lm_rows": 0}:
        raise ValueError(f"incomplete run status: {directory.name}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"incomplete generation rows: {directory.name}")
    mapping = {str(row["row_id"]): row for row in rows}
    if len(mapping) != EXPECTED_ROWS:
        raise ValueError(f"duplicate row ID: {directory.name}")
    if Counter(row["task"] for row in rows) != Counter({task: ROWS_PER_TASK for task in TASKS}):
        raise ValueError(f"task counts drift: {directory.name}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--ncp-run", type=Path, required=True)
    parser.add_argument("--table-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "native-halfturn-ruler4k-source-order-v1":
        raise ValueError("the NCP experiment must reuse the frozen half-turn panel")
    if manifest.get("rows") != EXPECTED_ROWS:
        raise ValueError("RULER panel row count drift")
    audit = json.loads(args.table_audit.read_text())
    if audit.get("status") != "NATIVE_CONTRASTIVE_PROXIMAL_CPU_AUDIT_V1":
        raise ValueError("NCP table audit is incomplete")
    if not all(audit.get("checks", {}).values()):
        raise ValueError("NCP table audit contains a failed check")

    runs = {
        "native": load_run(args.native_run),
        "ncp": load_run(args.ncp_run),
    }
    identities = set(runs["native"])
    if set(runs["ncp"]) != identities:
        raise ValueError("Native and NCP prompt IDs are not paired")
    for row_id in identities:
        reference = runs["native"][row_id]
        current = runs["ncp"][row_id]
        for key in ("task", "prompt_sha256", "references", "input_tokens", "max_new_tokens"):
            if current.get(key) != reference.get(key):
                raise ValueError(f"Native/NCP prompt drift: {row_id}/{key}")

    scores = {}
    task_scores = {}
    for arm in ARMS:
        scores[arm], task_scores[arm] = task_macro(runs[arm])
    comparison = paired_task_bootstrap(runs["ncp"], runs["native"])
    report = {
        "status": "OLMO_NATIVE_NCP_RULER4K_PAIRED_REPORT_V1",
        "model_id": manifest["model_id"],
        "panel": {
            "rows": EXPECTED_ROWS,
            "tasks": len(TASKS),
            "rows_per_task": ROWS_PER_TASK,
            "inputs_sha256": manifest["panel"]["inputs_sha256"],
            "selection_mode": "source-order",
            "content_padding": False,
            "reused_asset_contract": manifest["contract"],
        },
        "table": {
            "candidate_table_sha256_float32": audit["candidate_table_sha256_float32"],
            "gain": 1.0,
            "model_outputs_used_to_build_table": False,
        },
        "arm_macro_scores": scores,
        "arm_task_scores": task_scores,
        "comparison": {"ncp_minus_native": comparison},
        "decision": {
            "point_estimate_positive": comparison["delta"] > 0.0,
            "paired_ci_above_zero": comparison["paired_task_equal_bootstrap_ci95"][0] > 0.0,
        },
        "claim_boundary": (
            "Fresh model execution for one public-parameter NCP table on the previously frozen "
            "OLMo Native-window RULER development panel. It tests for a task signal but is not "
            "an independent confirmation or a cross-checkpoint universal claim."
        ),
    }
    atomic_json(args.out, report)
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
