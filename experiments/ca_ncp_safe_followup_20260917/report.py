#!/usr/bin/env python3
"""Report the three safe-alignment arms against frozen CA-NCP baselines."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.ca_ncp_native_20260917.io_utils import atomic_json, file_sha256
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from experiments.native_enhancement_oral_20260915.reanalyze_existing import row_metrics

from . import ARMS


BASELINES = ("N0", "C0", "P0", "P1")
PAIR_FIELDS = ("task", "prompt_sha256", "references", "input_tokens", "max_new_tokens")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(path: Path) -> dict[str, dict]:
    status = json.loads((path / "status.json").read_text())
    rows = read_jsonl(path / "generations.jsonl")
    if status != {"status": "COMPLETE", "rows": 130, "lm_rows": 0} or len(rows) != 130:
        raise ValueError(f"incomplete run: {path}")
    mapping = {str(row["row_id"]): row for row in rows}
    if len(mapping) != 130 or Counter(row["task"] for row in rows) != Counter({task: 10 for task in TASKS}):
        raise ValueError(f"run identity differs: {path}")
    return mapping


def means(rows: list[dict]) -> dict:
    metrics = [row_metrics(row) for row in rows]
    result = {}
    for field in ("official", "reference_set_recall", "ended_eos", "hit_cap", "empty", "generated_tokens"):
        result[field] = float(np.mean([row[field] for row in metrics]))
    precision = [row["typed_set_precision"] for row in metrics if row["typed_set_precision"] is not None]
    result["typed_set_precision"] = float(np.mean(precision)) if precision else None
    return result


def bootstrap(runs: dict[str, dict[str, dict]], *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = list(BASELINES) + list(ARMS)
    macros = {name: np.zeros(draws) for name in names}
    for task in TASKS:
        ids = sorted(row_id for row_id, row in runs["N0"].items() if row["task"] == task)
        index = rng.integers(len(ids), size=(draws, len(ids)))
        for name in names:
            values = np.asarray([float(runs[name][row_id]["ruler_official_score"]) for row_id in ids])
            macros[name] += values[index].mean(axis=1) / len(TASKS)
    contrasts = {}
    for arm in ARMS:
        reference = "N0" if arm.startswith("N_") else "P0"
        values = macros[arm] - macros[reference]
        contrasts[f"{arm}_minus_{reference}"] = {
            "paired_task_equal_bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "fraction_gt_zero": float(np.mean(values > 0)),
        }
    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261218)
    args = parser.parse_args()
    runs = {name: load_run(args.source_root / "runs" / name) for name in BASELINES}
    runs.update({name: load_run(args.root / "runs" / name) for name in ARMS})
    identities = set(runs["N0"])
    if any(set(rows) != identities for rows in runs.values()):
        raise ValueError("safe follow-up and baseline row identities differ")
    for row_id in identities:
        baseline = runs["N0"][row_id]
        for name, rows in runs.items():
            if any(rows[row_id].get(field) != baseline.get(field) for field in PAIR_FIELDS):
                raise ValueError(f"paired input drift: {name}/{row_id}")
    names = list(BASELINES) + list(ARMS)
    by_task, health, scores = {}, {}, {}
    for name in names:
        by_task[name] = {
            task: means([row for row in runs[name].values() if row["task"] == task]) for task in TASKS
        }
        health[name] = means(list(runs[name].values()))
        scores[name] = float(np.mean([by_task[name][task]["official"] for task in TASKS]))
    contrasts = {}
    for arm in ARMS:
        reference = "N0" if arm.startswith("N_") else "P0"
        contrasts[f"{arm}_minus_{reference}"] = scores[arm] - scores[reference]
        contrasts[f"{arm}_minus_C0"] = scores[arm] - scores["C0"]
    intervals = bootstrap(runs, draws=args.draws, seed=args.seed)
    for name, value in intervals.items():
        value["estimate"] = contrasts[name]
    report = {
        "status": "CA_NCP_SAFE_FOLLOWUP_FULL13_X10_COMPLETE_V1",
        "evaluation_role": "post-failure development diagnostic on the same fixed Full-13x10 panel",
        "formal_fixed_panel_scores": scores,
        "formal_contrasts": contrasts,
        "paired_stability": intervals,
        "by_task": by_task,
        "output_health": health,
        "method_receipt_sha256": file_sha256(args.root / "METHOD_RECEIPT.json"),
        "raw_sha256": {
            name: file_sha256((args.root if name in ARMS else args.source_root) / "runs" / name / "generations.jsonl")
            for name in names
        },
        "decision_contract": {
            "new_candidate": "a P arm must exceed frozen C0 point score; freeze it before independent confirmation",
            "mechanism_only": "P0 < P arm <= C0 identifies a safer mechanism but is not a new best method",
            "close_route": "if all new arms are <= their N0/P0 control, stop activation-driven coordinate reassignment",
            "no_posthoc_combination": True,
        },
        "statistics": {
            "draws": args.draws,
            "seed": args.seed,
            "point_score_authority": "fixed-panel point scores; bootstrap is stability analysis",
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({"status": report["status"], "scores": scores, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
