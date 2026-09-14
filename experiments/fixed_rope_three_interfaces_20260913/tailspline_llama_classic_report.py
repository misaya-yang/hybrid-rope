#!/usr/bin/env python3
"""Strict paired report for the TailSpline Llama Full-13 + PPL46 contract.

The small set of module-level identity constants is intentionally overrideable
by the OLMo wrapper.  The scoring, pairing and bootstrap implementation remains
one shared path across checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .matched_generation_report import log_auc, summarize
from .pipeline import atomic_json, bootstrap_range_contrast
from .tables import find_table, tensor_sha256, validate_table


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
NIAH_TASKS = TASKS[:8]
POINT_DEPTH_TASKS = TASKS[:6]
LENGTHS = (8192, 16384, 32768)
COUNTS = {8192: 10, 16384: 10, 32768: 10}
DEPTHS = (0.10, 0.30, 0.50, 0.70, 0.90)
EXPECTED_BAND = (18, 35)
PPL_CONTRACT = "TAILSPLINE_LLAMA_PPL46_V1"
REPORT_STATUS = "TAILSPLINE_LLAMA_CLASSIC_REPORT_V1"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_mapping(values: list[str], *, label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use ARM=PATH")
        arm, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not arm or arm in result or not path.exists():
            raise ValueError(f"invalid or repeated {label}: {value}")
        result[arm] = path
    return result


def validate_arm(run: Path, receipt_path: Path) -> tuple[list[dict], list[dict]]:
    status = json.loads((run / "status.json").read_text())
    expected_rows = sum(COUNTS.values()) * len(TASKS)
    expected_lm = 46 * len(LENGTHS)
    if status != {"status": "COMPLETE", "rows": expected_rows, "lm_rows": expected_lm}:
        raise ValueError(f"incomplete classic run {run}: {status}")
    generations = read_jsonl(run / "generations.jsonl")
    lm_rows = read_jsonl(run / "lm_rows.jsonl")
    contract = json.loads((run / "contract.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    receipt_values, receipt_gain = validate_table(find_table(receipt), pairs=64)
    static = contract.get("static_table") or {}
    static_values, static_gain = validate_table(static, pairs=64)
    if (
        not np.array_equal(receipt_values, static_values)
        or receipt_gain != static_gain
        or tensor_sha256(static_values) != receipt.get("table_sha256_float32")
        or receipt.get("band_envelope") != list(EXPECTED_BAND)
        or float(receipt_gain) != 1.0 + 0.1 * math.log(4.0)
    ):
        raise ValueError(f"table/gain/band receipt drift in {run}")
    if contract.get("generation_length_caps") != list(LENGTHS):
        raise ValueError(f"generation length grid drift in {run}")
    if contract.get("lm_lengths") != list(LENGTHS) or contract.get("lm_limit_documents") != 0:
        raise ValueError(f"PPL grid/document limit drift in {run}")
    expected_cells = Counter({(task, length): COUNTS[length] for task in TASKS for length in LENGTHS})
    cells = Counter((row.get("task"), int(row.get("length_cap", -1))) for row in generations)
    if cells != expected_cells or len(generations) != expected_rows:
        raise ValueError(f"Full-13 cell counts drift in {run}")
    if len({row.get("prompt_sha256") for row in generations}) != expected_rows:
        raise ValueError(f"missing or duplicate prompt hashes in {run}")
    for row in generations:
        if (
            not isinstance(row.get("generated_ids"), list)
            or "ruler_official_score" not in row
            or "ended_eos" not in row
            or "hit_cap" not in row
        ):
            raise ValueError(f"incomplete generation row in {run}")
    for task in POINT_DEPTH_TASKS:
        for length in LENGTHS:
            observed = Counter(
                tuple(row.get("depth_target") or [])
                for row in generations
                if row["task"] == task and int(row["length_cap"]) == length
            )
            if observed != Counter({(depth,): 2 for depth in DEPTHS}):
                raise ValueError(f"missing 10/30/50/70/90 depth coverage in {run}/{task}/{length}")
    expected_lm_ids = [(document, length) for document in range(46) for length in LENGTHS]
    if (
        len(lm_rows) != expected_lm
        or [(row.get("document"), row.get("length")) for row in lm_rows] != expected_lm_ids
    ):
        raise ValueError(f"PPL row order/count drift in {run}")
    normalized = [{
        **row,
        "official_score": float(row["ruler_official_score"]),
        "mini_semantic_id": row["prompt_sha256"],
    } for row in generations]
    return normalized, lm_rows


def mean_curve(rows: list[dict], tasks: tuple[str, ...]) -> dict[int, float]:
    return {
        length: float(np.mean([
            np.mean([
                row["official_score"] for row in rows
                if row["task"] == task and int(row["length_cap"]) == length
            ])
            for task in tasks
        ]))
        for length in LENGTHS
    }


def depth_summary(rows: list[dict]) -> dict:
    result = {}
    for depth in DEPTHS:
        curve = {}
        for length in LENGTHS:
            values = [
                row["official_score"] for row in rows
                if row["task"] in POINT_DEPTH_TASKS
                and int(row["length_cap"]) == length
                and tuple(row.get("depth_target") or []) == (depth,)
            ]
            if not values:
                raise ValueError(f"empty NIAH depth cell {depth}/{length}")
            curve[length] = float(np.mean(values))
        result[str(int(depth * 100))] = {
            "by_length": {str(length): value for length, value in curve.items()},
            "log_length_auc": log_auc(curve, list(LENGTHS)),
        }
    return result


def ppl_summary(rows: list[dict], datasets: list[str]) -> dict:
    if len(datasets) != 46:
        raise ValueError("PPL manifest must map exactly 46 documents")
    groups = {"combined": set(range(46))}
    for dataset in sorted(set(datasets)):
        groups[dataset] = {index for index, value in enumerate(datasets) if value == dataset}
    result = {}
    for name, documents in groups.items():
        curve = {}
        entries = {}
        for length in LENGTHS:
            selected = [
                row for row in rows
                if row["document"] in documents and int(row["length"]) == length
            ]
            loss_sum = sum(float(row["whole_loss_sum"]) for row in selected)
            count = sum(int(row["whole_target_count"]) for row in selected)
            nll = loss_sum / count
            curve[length] = math.exp(nll)
            entries[str(length)] = {
                "documents": len(selected), "target_tokens": count,
                "whole_nll": nll, "whole_ppl": curve[length],
            }
        result[name] = {
            "by_length": entries,
            "log_length_ppl_auc": log_auc(curve, list(LENGTHS)),
        }
    source_equal_curve = {
        length: float(np.mean([
            result[name]["by_length"][str(length)]["whole_ppl"]
            for name in result if name != "combined"
        ]))
        for length in LENGTHS
    }
    result["source_equal_diagnostic"] = {
        "by_length": {str(length): value for length, value in source_equal_curve.items()},
        "log_length_ppl_auc": log_auc(source_equal_curve, list(LENGTHS)),
    }
    return result


def ppl_bootstrap(
    candidate: list[dict], baseline: list[dict], *, draws: int, seed: int,
) -> dict:
    candidate_map = {(row["document"], row["length"]): row for row in candidate}
    baseline_map = {(row["document"], row["length"]): row for row in baseline}
    if set(candidate_map) != set(baseline_map):
        raise ValueError("PPL arms are not document/length paired")
    rng = np.random.default_rng(seed)
    delta = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(0, 46, size=46)
        curves = []
        for mapping in (candidate_map, baseline_map):
            curve = {}
            for length in LENGTHS:
                rows = [mapping[(int(document), length)] for document in sampled]
                nll = (
                    sum(float(row["whole_loss_sum"]) for row in rows)
                    / sum(int(row["whole_target_count"]) for row in rows)
                )
                curve[length] = math.exp(nll)
            curves.append(log_auc(curve, list(LENGTHS)))
        delta[draw] = curves[0] - curves[1]
    return {
        "draws": draws, "seed": seed,
        "resampling": "paired held-out documents; same document draw shared across lengths",
        "delta_log_length_ppl_auc": {
            "mean": float(delta.mean()),
            "interval95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
            "negative_is_better": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], required=True, help="ARM=RUN_DIR")
    parser.add_argument("--receipt", action="append", default=[], required=True, help="ARM=TABLE_JSON")
    parser.add_argument("--ppl-manifest", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", action="append", default=[], required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.candidate in args.baseline:
        raise ValueError("invalid comparison identity or pre-existing output")
    runs = parse_mapping(args.run, label="--run")
    receipts = parse_mapping(args.receipt, label="--receipt")
    wanted = {args.candidate, *args.baseline}
    if set(runs) != wanted or set(receipts) != wanted:
        raise ValueError("runs and receipts must cover exactly candidate and baselines")
    ppl_manifest = json.loads(args.ppl_manifest.read_text())
    if (
        ppl_manifest.get("contract") != PPL_CONTRACT
        or ppl_manifest.get("documents") != 46
        or ppl_manifest.get("lengths") != list(LENGTHS)
    ):
        raise ValueError("unexpected PPL46 manifest")
    datasets = [record["dataset"] for record in ppl_manifest["document_records"]]
    loaded = {
        arm: validate_arm(path, receipts[arm])
        for arm, path in runs.items()
    }
    prompts = {arm: {row["prompt_sha256"] for row in rows} for arm, (rows, _) in loaded.items()}
    if any(value != prompts[args.candidate] for value in prompts.values()):
        raise ValueError("Full-13 arms are not exactly prompt paired")
    summaries = {}
    for arm, (rows, lm_rows) in loaded.items():
        full13 = summarize(rows, tasks=list(TASKS), lengths=list(LENGTHS))
        passkey_curve = mean_curve(rows, ("niah_single_1",))
        niah_curve = mean_curve(rows, NIAH_TASKS)
        summaries[arm] = {
            "full13": full13,
            "passkey_reusing_ruler_niah_single_1": {
                "by_length": {str(length): value for length, value in passkey_curve.items()},
                "log_length_auc": log_auc(passkey_curve, list(LENGTHS)),
            },
            "niah_task_equal": {
                "by_length": {str(length): value for length, value in niah_curve.items()},
                "log_length_auc": log_auc(niah_curve, list(LENGTHS)),
            },
            "point_niah_depths": depth_summary(rows),
            "ppl": ppl_summary(lm_rows, datasets),
        }
    contrasts = {}
    family_map = {
        task: "niah" if task in NIAH_TASKS else
        "tracking" if task == "vt" else
        "aggregation" if task in {"cwe", "fwe"} else "qa"
        for task in TASKS
    }
    for offset, baseline in enumerate(args.baseline):
        candidate_rows, candidate_lm = loaded[args.candidate]
        baseline_rows, baseline_lm = loaded[baseline]
        contrasts[baseline] = {
            "full13": bootstrap_range_contrast(
                candidate_rows, baseline_rows, tasks=list(TASKS), lengths=list(LENGTHS),
                task_families=family_map, draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + offset,
            ),
            "observed": {
                "delta_full13_auc": (
                    summaries[args.candidate]["full13"]["log_length_auc"]
                    - summaries[baseline]["full13"]["log_length_auc"]
                ),
                "delta_passkey_auc": (
                    summaries[args.candidate]["passkey_reusing_ruler_niah_single_1"]["log_length_auc"]
                    - summaries[baseline]["passkey_reusing_ruler_niah_single_1"]["log_length_auc"]
                ),
                "delta_niah_auc": (
                    summaries[args.candidate]["niah_task_equal"]["log_length_auc"]
                    - summaries[baseline]["niah_task_equal"]["log_length_auc"]
                ),
                "delta_combined_ppl_auc": (
                    summaries[args.candidate]["ppl"]["combined"]["log_length_ppl_auc"]
                    - summaries[baseline]["ppl"]["combined"]["log_length_ppl_auc"]
                ),
            },
            "ppl": ppl_bootstrap(
                candidate_lm, baseline_lm, draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 100 + offset,
            ),
        }
        observed = contrasts[baseline]["observed"]
        directions = {
            "ppl": observed["delta_combined_ppl_auc"] < 0.0,
            "passkey_niah": observed["delta_niah_auc"] > 0.0,
            "full13": observed["delta_full13_auc"] > 0.0,
        }
        contrasts[baseline]["expansion_gate"] = {
            "endpoint_positive_directions": directions,
            "wins": sum(directions.values()),
            "eligible_for_50_per_cell_confirmation": sum(directions.values()) >= 2,
            "basis": "first complete panel point estimates; confirmation, not claim acceptance",
        }
    strong_baselines = [name for name in ("mrpro",) if name in contrasts]
    report = {
        "status": REPORT_STATUS,
        "candidate": args.candidate, "baselines": args.baseline,
        "tasks": list(TASKS), "lengths": list(LENGTHS),
        "rows_per_arm": sum(COUNTS.values()) * len(TASKS),
        "lm_rows_per_arm": 46 * len(LENGTHS),
        "paired_prompts": len(prompts[args.candidate]),
        "summaries": summaries, "contrasts": contrasts,
        "confirmation_gate": {
            "required_strong_baselines": strong_baselines,
            "eligible_for_50_per_cell_confirmation": bool(strong_baselines) and all(
                contrasts[name]["expansion_gate"]["eligible_for_50_per_cell_confirmation"]
                for name in strong_baselines
            ),
            "rule": "at least two of PPL, passkey/NIAH and Full-13 point directions vs MrPro",
        },
        "endpoint_contract": (
            "PPL AUC lower is better; passkey/NIAH and Full-13 task-equal AUC higher is better; "
            "local task reversals do not automatically veto aggregate family endpoints"
        ),
        "independence_note": (
            "Passkey reuses official niah_single_1 rows and is not counted as an independent dataset."
        ),
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"], "paired_prompts": report["paired_prompts"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
