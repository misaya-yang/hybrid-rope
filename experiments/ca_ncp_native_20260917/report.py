#!/usr/bin/env python3
"""Build the complete paired five-arm CA-NCP Full-13 report."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from experiments.native_enhancement_oral_20260915.reanalyze_existing import row_metrics
from . import ARMS
from .core import tensor_sha256
from .io_utils import atomic_json, file_sha256


PAIR_FIELDS = ("task", "prompt_sha256", "references", "input_tokens", "max_new_tokens")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(path: Path, expected_rows: int) -> tuple[dict[str, dict], dict, dict]:
    status = json.loads((path / "status.json").read_text())
    contract = json.loads((path / "contract.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    rows = read_jsonl(path / "generations.jsonl")
    if status != {"status": "COMPLETE", "rows": expected_rows, "lm_rows": 0} or len(rows) != expected_rows:
        raise ValueError(f"incomplete run: {path}")
    mapping = {str(row["row_id"]): row for row in rows}
    if len(mapping) != expected_rows or Counter(row["task"] for row in rows) != Counter({task: 10 for task in TASKS}):
        raise ValueError(f"run row/task identity differs: {path}")
    return mapping, contract, summary


def means(rows: list[dict]) -> dict:
    metrics = [row_metrics(row) for row in rows]
    result = {}
    for field in (
        "official", "reference_set_recall", "ended_eos", "hit_cap", "empty", "generated_tokens",
    ):
        result[field] = float(np.mean([row[field] for row in metrics]))
    precision = [row["typed_set_precision"] for row in metrics if row["typed_set_precision"] is not None]
    result["typed_set_precision"] = float(np.mean(precision)) if precision else None
    result["typed_set_precision_rows"] = len(precision)
    return result


def bootstrap(runs: dict[str, dict[str, dict]], *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    arm_draws = {arm: [] for arm in ARMS}
    for task in TASKS:
        ids = sorted(row_id for row_id, row in runs["N0"].items() if row["task"] == task)
        indices = rng.integers(len(ids), size=(draws, len(ids)))
        for arm in ARMS:
            values = np.asarray([float(runs[arm][row_id]["ruler_official_score"]) for row_id in ids])
            arm_draws[arm].append(values[indices].mean(axis=1))
    macros = {arm: np.mean(values, axis=0) for arm, values in arm_draws.items()}
    contrasts = {
        "P1_minus_N0": macros["P1"] - macros["N0"],
        "P1_minus_P0": macros["P1"] - macros["P0"],
        "P0_minus_C0": macros["P0"] - macros["C0"],
        "N1_minus_N0": macros["N1"] - macros["N0"],
        "interaction": (macros["P1"] - macros["P0"]) - (macros["N1"] - macros["N0"]),
        "relative_10pct_margin": macros["P1"] - 1.1 * macros["N0"],
    }
    return {
        name: {
            "paired_task_equal_bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "fraction_gt_zero": float(np.mean(values > 0)),
        }
        for name, values in contrasts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261217)
    args = parser.parse_args()
    if args.draws < 1:
        raise ValueError("draws must be positive")
    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "CA_NCP_NATIVE_FULL13_X10_REUSE_V1_1" or manifest.get("rows") != 130:
        raise ValueError("pilot asset contract differs")
    method = json.loads((args.construction / "METHOD_RECEIPT.json").read_text())
    alignment = json.loads((args.alignment / "ALIGNMENT_RECEIPT.json").read_text())
    if alignment["method_receipt_sha256"] != file_sha256(args.construction / "METHOD_RECEIPT.json"):
        raise ValueError("alignment belongs to another method receipt")
    runs, contracts, summaries = {}, {}, {}
    for arm in ARMS:
        runs[arm], contracts[arm], summaries[arm] = load_run(args.run_root / arm, 130)
    identities = set(runs["N0"])
    if any(set(runs[arm]) != identities for arm in ARMS):
        raise ValueError("five arms have different row identities")
    for row_id in identities:
        baseline = runs["N0"][row_id]
        for arm in ARMS:
            if any(runs[arm][row_id].get(field) != baseline.get(field) for field in PAIR_FIELDS):
                raise ValueError(f"paired input drift: {arm}/{row_id}")
    expected_alignment_hash = alignment["alignment_sha256"]
    for arm in ("N1", "P1"):
        value = contracts[arm].get("ca_ncp_alignment") or {}
        if value.get("sha256") != expected_alignment_hash:
            raise ValueError(f"{arm} did not install the frozen alignment")
    for arm in ("N0", "C0", "P0"):
        if contracts[arm].get("ca_ncp_alignment") is not None:
            raise ValueError(f"{arm} unexpectedly installed an alignment")
    table_hashes = {
        "N0": method["native_table_sha256_float32"],
        "C0": method["ncp_table_sha256_float32"],
        "P0": method["carrier_table_sha256_float32"],
        "N1": method["native_table_sha256_float32"],
        "P1": method["carrier_table_sha256_float32"],
    }
    for arm in ARMS:
        table = summaries[arm].get("table") or {}
        values = np.asarray(table.get("values_float32"), dtype=np.float32)
        if values.shape != (64,) or tensor_sha256(values) != table_hashes[arm]:
            raise ValueError(f"{arm} runtime frequency table differs from the frozen method")
        installed_alignment = summaries[arm].get("ca_ncp_alignment")
        if arm in ("N1", "P1"):
            if not installed_alignment or installed_alignment.get("alignment_sha256") != expected_alignment_hash:
                raise ValueError(f"{arm} summary lacks the frozen alignment")
        elif installed_alignment is not None:
            raise ValueError(f"{arm} summary unexpectedly records an alignment")
    arm_scores, by_task, health = {}, {}, {}
    for arm in ARMS:
        by_task[arm] = {}
        for task in TASKS:
            rows = [row for row in runs[arm].values() if row["task"] == task]
            by_task[arm][task] = means(rows)
        arm_scores[arm] = float(np.mean([by_task[arm][task]["official"] for task in TASKS]))
        health[arm] = means(list(runs[arm].values()))
    contrasts = {
        "P1_minus_N0": arm_scores["P1"] - arm_scores["N0"],
        "P1_minus_P0": arm_scores["P1"] - arm_scores["P0"],
        "P0_minus_C0": arm_scores["P0"] - arm_scores["C0"],
        "N1_minus_N0": arm_scores["N1"] - arm_scores["N0"],
        "interaction": (arm_scores["P1"] - arm_scores["P0"]) - (arm_scores["N1"] - arm_scores["N0"]),
        "relative_10pct_margin": arm_scores["P1"] - 1.1 * arm_scores["N0"],
    }
    intervals = bootstrap(runs, draws=args.draws, seed=args.seed)
    for name, value in contrasts.items():
        intervals[name]["estimate"] = float(value)
    report = {
        "status": "CA_NCP_FULL13_X10_FIVE_ARM_COMPLETE_V1_1",
        "rows_per_arm": 130,
        "metric": "official RULER score, task-equal across Full-13",
        "formal_fixed_panel_scores": arm_scores,
        "formal_contrasts": contrasts,
        "paired_stability": intervals,
        "by_task": by_task,
        "output_health": health,
        "table_sha256_float32": table_hashes,
        "alignment_sha256": expected_alignment_hash,
        "method_receipt_sha256": file_sha256(args.construction / "METHOD_RECEIPT.json"),
        "statistics_receipt_sha256": alignment["statistics_receipt_sha256"],
        "panel_sha256": manifest["panel"]["inputs_sha256"],
        "raw_sha256": {arm: file_sha256(args.run_root / arm / "generations.jsonl") for arm in ARMS},
        "contracts_sha256": {arm: file_sha256(args.run_root / arm / "contract.json") for arm in ARMS},
        "statistics": {
            "draws": args.draws,
            "seed": args.seed,
            "bootstrap": "common within-task paired resampling of all five arms; fixed tasks remain equally weighted",
            "point_score_authority": "formal fixed-panel scores and exact contrasts; bootstrap is stability analysis",
        },
        "scope": "First frozen 130-row CA-NCP development pilot; no automatic expansion or parameter search.",
    }
    atomic_json(args.out, report)
    print(json.dumps({"status": report["status"], "scores": arm_scores, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
