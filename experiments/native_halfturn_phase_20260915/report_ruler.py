#!/usr/bin/env python3
"""Build the paired four-arm Native half-turn RULER report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS


ARMS = ("native", "contract", "reverse", "v1")


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
        values = [float(row["ruler_official_score"]) for row in rows.values() if row["task"] == task]
        if len(values) != 60:
            raise ValueError(f"{task} does not contain 60 rows")
        by_task[task] = float(np.mean(values))
    return float(np.mean(list(by_task.values()))), by_task


def paired_task_bootstrap(
    left: dict[str, dict],
    right: dict[str, dict],
    *,
    draws: int = 20_000,
    seed: int = 20261102,
) -> dict:
    rng = np.random.default_rng(seed)
    task_draws = []
    for task in TASKS:
        ids = sorted(row_id for row_id, row in left.items() if row["task"] == task)
        delta = np.asarray([
            float(left[row_id]["ruler_official_score"])
            - float(right[row_id]["ruler_official_score"])
            for row_id in ids
        ])
        sampled = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
        task_draws.append(sampled)
    values = np.mean(task_draws, axis=0)
    return {
        "delta": float(values.mean()),
        "paired_task_equal_bootstrap_ci95": interval(values),
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def load_run(directory: Path, *, expected_rows: int) -> dict[str, dict]:
    status = json.loads((directory / "status.json").read_text())
    rows = read_jsonl(directory / "generations.jsonl")
    if status != {"status": "COMPLETE", "rows": expected_rows, "lm_rows": 0}:
        raise ValueError(f"incomplete run status: {directory.name}")
    if len(rows) != expected_rows:
        raise ValueError(f"incomplete generation rows: {directory.name}")
    mapping = {str(row["row_id"]): row for row in rows}
    if len(mapping) != expected_rows:
        raise ValueError(f"duplicate row ID: {directory.name}")
    if Counter(row["task"] for row in rows) != Counter({task: 60 for task in TASKS}):
        raise ValueError(f"task counts drift: {directory.name}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="ARM=RUN_DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "native-halfturn-ruler4k-source-order-v1" or manifest.get("rows") != 780:
        raise ValueError("Native half-turn RULER asset contract drift")
    run_paths = {}
    for spec in args.run:
        arm, separator, path = spec.partition("=")
        if not separator or arm not in ARMS or arm in run_paths:
            raise ValueError("--run must provide each unique ARM=RUN_DIRECTORY")
        run_paths[arm] = Path(path)
    if set(run_paths) != set(ARMS):
        raise ValueError("the frozen report requires Native, contract, reverse and V1")
    runs = {arm: load_run(path, expected_rows=780) for arm, path in run_paths.items()}
    identities = set(runs["native"])
    if any(set(rows) != identities for rows in runs.values()):
        raise ValueError("four-arm prompt IDs are not paired")
    for row_id in identities:
        reference = runs["native"][row_id]
        for arm in ARMS[1:]:
            current = runs[arm][row_id]
            for key in ("task", "prompt_sha256", "references", "input_tokens", "max_new_tokens"):
                if current.get(key) != reference.get(key):
                    raise ValueError(f"four-arm prompt drift: {arm}/{row_id}/{key}")

    scores = {}
    task_scores = {}
    for arm in ARMS:
        scores[arm], task_scores[arm] = task_macro(runs[arm])
    comparisons = {
        "contract_minus_native": paired_task_bootstrap(runs["contract"], runs["native"]),
        "contract_minus_reverse": paired_task_bootstrap(
            runs["contract"], runs["reverse"], seed=20261103,
        ),
        "v1_minus_native": paired_task_bootstrap(runs["v1"], runs["native"], seed=20261104),
    }
    primary = comparisons["contract_minus_native"]
    direction = comparisons["contract_minus_reverse"]
    report = {
        "status": "OLMO_NATIVE_HALFTURN_RULER4K_FOUR_ARM_REPORT_V1",
        "model_id": manifest["model_id"],
        "panel": {
            "rows": 780,
            "tasks": 13,
            "rows_per_task": 60,
            "inputs_sha256": manifest["panel"]["inputs_sha256"],
            "selection_mode": "source-order",
            "content_padding": False,
        },
        "arm_macro_scores": scores,
        "arm_task_scores": task_scores,
        "comparisons": comparisons,
        "decision": {
            "primary_contract_vs_native_ci_above_zero": primary["paired_task_equal_bootstrap_ci95"][0] > 0.0,
            "primary_contract_vs_native_point_at_least_3pp": primary["delta"] >= 0.03,
            "direction_contract_vs_reverse_ci_above_zero": direction["paired_task_equal_bootstrap_ci95"][0] > 0.0,
            "ruler_directional_gate_pass": bool(
                primary["paired_task_equal_bootstrap_ci95"][0] > 0.0
                and primary["delta"] >= 0.03
                and direction["paired_task_equal_bootstrap_ci95"][0] > 0.0
            ),
        },
        "claim_boundary": (
            "Frozen single-checkpoint Native-window RULER result. The final Native-enhancement "
            "claim also requires the separately frozen language-model health endpoint; V1 is a "
            "historical checkpoint-calibrated reference, not an equal-dose direction control."
        ),
    }
    atomic_json(args.out, report)
    print(json.dumps(report["decision"], sort_keys=True))


if __name__ == "__main__":
    main()

