#!/usr/bin/env python3
"""Summarize the preregistered staged Qwen K32 packed-natural NLL panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import summarize_qwen_k32_natural_nll as base

PRIMARY_ARMS = base.ARMS[:2]
BASELINE_ARMS = base.ARMS[2:]
STAGE_STATUS = {
    "primary": "QWEN_K32_PACKED_NATURAL_NLL_PRIMARY_COMPLETE",
    "baseline": "QWEN_K32_PACKED_NATURAL_NLL_BASELINE_COMPLETE",
    "control": "QWEN_K32_PACKED_NATURAL_NLL_CONTROL_COMPLETE",
}


def load_stage(root: Path, stage: str) -> dict:
    arms = PRIMARY_ARMS if stage == "primary" else BASELINE_ARMS
    paths = {name: root / filename for name, filename in (
        ("results", "results.json"), ("run_manifest", "run_manifest.json"),
        ("examples", "examples.jsonl"))}
    result = json.loads(paths["results"].read_text())
    manifest = json.loads(paths["run_manifest"].read_text())
    rows = [json.loads(line) for line in paths["examples"].read_text().splitlines() if line.strip()]
    hashes = {name + "_sha256": base.sha256(path) for name, path in paths.items()}
    if (
        result.get("status") != STAGE_STATUS[stage]
        or result.get("stage") != stage
        or manifest.get("status") != base.FROZEN_STATUS
        or manifest.get("stage") != stage
        or manifest.get("arm_order") != list(arms)
        or manifest.get("declared_arm_order") != list(base.ARMS)
        or result.get("examples_sha256") != hashes["examples_sha256"]
        or result.get("run_manifest_sha256") != hashes["run_manifest_sha256"]
        or manifest.get("checkpoint_weight_sha256") != base.WEIGHT_SHA256
        or manifest.get("config_sha256") != base.CONFIG_SHA256
        or manifest.get("lengths") != list(base.LENGTHS)
        or manifest.get("natural_streams") != base.STREAMS
        or manifest.get("executed_streams") != base.STREAMS
        or manifest.get("target_tokens") != base.TARGET_TOKENS
        or manifest.get("packing_contract") != base.PACKING_CONTRACT
        or manifest.get("use_cache") is not False
        or manifest.get("compile") is not False
        or manifest.get("model_updates") != 0
        or manifest.get("profile_selection") is not False
        or manifest.get("all_profiles_loaded_before_inference") is not True
    ):
        raise ValueError("staged packed-natural run contract drift")
    bound = {key: base.checked_hash(manifest.get(key), key) for key in (
        "checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256",
        "data_rows_sha256", "script_sha256", "model_source_sha256",
        "attention_source_sha256", "boundary_receipt_sha256")}
    evaluator = Path(__file__).resolve().parents[1] / "eval" / "eval_qwen_k32_natural_nll.py"
    if bound["script_sha256"] != base.sha256(evaluator):
        raise ValueError("evaluator source hash differs from the frozen staged run")
    if manifest.get("boundary_receipt_status") != "QWEN_K32_PACKED_NATURAL_TARGET_BOUNDARY_SAFE_V1":
        raise ValueError("staged run has no admitted boundary-safe receipt")
    profiles = base.validate_profiles(manifest)
    source = base.validate_source(manifest.get("source"))
    tokenizer_files = base.validate_tokenizer_files(manifest.get("tokenizer_files"))
    ids = [f"qwen-k32-natural-{index:03d}" for index in range(base.STREAMS)]
    expected = {(arm, sid, length) for arm in arms for sid in ids for length in base.LENGTHS}
    cells, metadata_by_cell, stream_identity = {}, {}, {}
    metadata = ("family", "variant", "split", "source_row_start", "source_row_end",
                "source_document_count", "source_set_sha256", "prompt_ids_sha256",
                "target_ids_sha256", "target_start", "target_tokens")
    for row in rows:
        arm, sid, length = row.get("arm"), row.get("sample_id"), row.get("length")
        key = arm, sid, length
        profile = profiles.get(arm)
        loss = row.get("nll")
        if (
            profile is None or key not in expected or key in cells
            or row.get("family") != "natural" or row.get("variant") != "packed_natural"
            or row.get("split") != "holdout"
            or row.get("target_start") != length - base.TARGET_TOKENS
            or row.get("target_tokens") != base.TARGET_TOKENS
            or row.get("table_sha256_float32") != profile["tensor_sha256"]
            or row.get("attention_scaling") != profile["attention_scaling"]
            or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
            or type(row.get("source_row_start")) is not int
            or type(row.get("source_row_end")) is not int
            or row["source_row_start"] > row["source_row_end"]
            or type(row.get("source_document_count")) is not int
            or row["source_document_count"] <= 0
        ):
            raise ValueError("invalid staged NLL row")
        for field in ("source_set_sha256", "prompt_ids_sha256", "target_ids_sha256"):
            base.checked_hash(row.get(field), field)
        row_metadata = tuple(row.get(field) for field in metadata)
        cell_id = sid, length
        if cell_id in metadata_by_cell and metadata_by_cell[cell_id] != row_metadata:
            raise ValueError("paired staged metadata differs across profiles")
        metadata_by_cell[cell_id] = row_metadata
        identity = (row.get("source_row_start"), row.get("source_row_end"),
                    row.get("source_document_count"), row.get("source_set_sha256"),
                    row.get("target_ids_sha256"))
        if sid in stream_identity and stream_identity[sid] != identity:
            raise ValueError("source stream or target differs across staged lengths")
        stream_identity[sid] = identity
        cells[key] = float(loss)
    if set(cells) != expected or result.get("rows") != len(expected):
        raise ValueError("staged NLL grid is incomplete")
    stored = result.get("curves")
    if not isinstance(stored, dict) or set(stored) != set(arms):
        raise ValueError("staged terminal curves are incomplete")
    for arm in arms:
        if set(stored[arm]) != {str(length) for length in base.LENGTHS}:
            raise ValueError("staged terminal lengths are incomplete")
        for length in base.LENGTHS:
            values = [cells[arm, sid, length] for sid in ids]
            mean = float(np.mean(values))
            native_delta = 0.0 if arm == "Native" else float(np.mean([
                cells[arm, sid, length] - cells["Native", sid, length]
                for sid in ids
            ])) if "Native" in arms else stored[arm][str(length)]["mean_paired_delta_vs_native"]
            item = stored[arm][str(length)]
            if (
                item.get("streams") != base.STREAMS
                or not math.isclose(item.get("mean_tail_nll"), mean, rel_tol=0, abs_tol=1e-12)
                or ("Native" in arms and not math.isclose(
                    item.get("mean_paired_delta_vs_native"), native_delta,
                    rel_tol=0, abs_tol=1e-12))
            ):
                raise ValueError("staged terminal curve differs from raw rows")
    return {"stage": stage, "arms": arms, "result": result, "manifest": manifest,
            "rows": rows, "cells": cells, "metadata": metadata_by_cell, "bound": bound,
            "profiles": profiles, "source": source, "tokenizer_files": tokenizer_files,
            "raw_hashes": hashes}


def validate_primary_receipt(bundle: dict, path: Path, *, authorized: bool) -> dict:
    receipt = json.loads(path.read_text())
    expected = summarize_primary(bundle)
    if receipt != expected or receipt.get("stage_b_authorized") is not authorized:
        raise ValueError("primary receipt does not match the frozen primary rows")
    return receipt


def summarize_primary(bundle: dict) -> dict:
    ids = [f"qwen-k32-natural-{index:03d}" for index in range(base.STREAMS)]
    values = np.array([[[bundle["cells"][arm, sid, length] for arm in PRIMARY_ARMS]
                        for length in base.LENGTHS] for sid in ids], dtype=np.float64)
    indices = np.random.default_rng(base.BOOTSTRAP_SEED).integers(
        0, base.STREAMS, size=(base.BOOTSTRAP_SAMPLES, base.STREAMS))
    draws, means = values[indices].mean(axis=1), values.mean(axis=0)
    retention = base.finite_exp(float(means[0, 0] - means[0, 1]))
    delta_draws = draws[:, 1, 1] - draws[:, 1, 0]
    ci = np.quantile(delta_draws, [.025, .975]).tolist()
    passes_retention = bool(retention is not None and retention >= .875)
    resolver = passes_retention and ci[1] < 0
    curves = {str(length): {
        arm: {"mean_nll": float(means[length_index, arm_index]),
              "ppl": base.finite_exp(float(means[length_index, arm_index]))}
        for arm_index, arm in enumerate(PRIMARY_ARMS)}
        for length_index, length in enumerate(base.LENGTHS)}
    return {
        "status": "QWEN_K32_PACKED_NATURAL_NLL_PRIMARY_SUMMARIZED",
        "stage_b_authorized": resolver,
        "classification": {"resolver": "PASS" if resolver else "NOT_PASS"},
        "curves": curves,
        "index_native_ppl_retention_32k": {
            "value": retention, "threshold": .875, "passes_point_gate": passes_retention},
        "index_minus_native_64k": {
            "mean_delta_nll": float(means[1, 1] - means[1, 0]),
            "joint_paired_stream_ci95": ci},
        "decision_rule": (
            "Stage B is authorized iff 32K index PPL retention >= 0.875 and the "
            "64K index-minus-Native paired NLL CI upper bound is below zero."
        ),
        "bootstrap": {"seed": base.BOOTSTRAP_SEED, "replicates": base.BOOTSTRAP_SAMPLES,
                      "unit": "same 32 stream indices resampled jointly across profiles and lengths"},
        "identity": {**bundle["bound"], "profiles": bundle["profiles"],
                     "source": bundle["source"], "tokenizer_files": bundle["tokenizer_files"]},
        "raw_hashes": bundle["raw_hashes"],
        "summary_code_sha256": base.sha256(Path(__file__)),
        "evidence_limit": "Paired final-256 natural NLL primary resolver; no YaRN, QA, or K-causal claim.",
    }


def summarize_final(primary: dict, baseline: dict, primary_receipt_path: Path) -> dict:
    validate_primary_receipt(primary, primary_receipt_path, authorized=True)
    if baseline["manifest"].get("primary_receipt_sha256") != base.sha256(primary_receipt_path):
        raise ValueError("baseline run is not bound to the supplied primary receipt")
    for key in ("checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256",
                "data_rows_sha256", "boundary_receipt_sha256"):
        if primary["bound"][key] != baseline["bound"][key]:
            raise ValueError("primary and baseline identities differ")
    for cell, metadata in primary["metadata"].items():
        if baseline["metadata"].get(cell) != metadata:
            raise ValueError("primary and baseline row metadata differ")

    rows = primary["rows"] + baseline["rows"]
    cells = {**primary["cells"], **baseline["cells"]}
    ids = [f"qwen-k32-natural-{index:03d}" for index in range(base.STREAMS)]
    curves = {}
    for arm in base.ARMS:
        curves[arm] = {}
        for length in base.LENGTHS:
            values = [cells[arm, sid, length] for sid in ids]
            native = [cells["Native", sid, length] for sid in ids]
            curves[arm][str(length)] = {
                "streams": base.STREAMS,
                "mean_tail_nll": float(np.mean(values)),
                "mean_paired_delta_vs_native": float(np.mean(
                    np.asarray(values) - np.asarray(native))),
            }
    result = {"status": base.TERMINAL_STATUS, "rows": len(rows), "curves": curves}
    manifest = dict(primary["manifest"])
    manifest.update(status=base.FROZEN_STATUS, arm_order=list(base.ARMS))
    report = base.summarize(result, manifest, rows, raw_hashes={
        "primary_receipt_sha256": base.sha256(primary_receipt_path),
        **{"primary_" + key: value for key, value in primary["raw_hashes"].items()},
        **{"baseline_" + key: value for key, value in baseline["raw_hashes"].items()},
    })
    degradation = {}
    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    indices = rng.integers(0, base.STREAMS, size=(base.BOOTSTRAP_SAMPLES, base.STREAMS))
    per_arm = {}
    for arm in base.ARMS:
        values = np.array([cells[arm, sid, base.LENGTHS[1]] - cells[arm, sid, base.LENGTHS[0]]
                           for sid in ids], dtype=np.float64)
        draws = values[indices].mean(axis=1)
        per_arm[arm] = values
        degradation[arm] = {"mean_delta_nll_64k_minus_32k": float(values.mean()),
                            "paired_stream_ci95": np.quantile(draws, [.025, .975]).tolist()}
    for left, right, name in (("normalized_raw_index", "Native", "index_minus_native"),
                              ("normalized_raw_index", "official_equation_yarn", "index_minus_yarn")):
        values = per_arm[left] - per_arm[right]
        draws = values[indices].mean(axis=1)
        degradation[name] = {"mean_difference_in_length_degradation": float(values.mean()),
                             "paired_stream_ci95": np.quantile(draws, [.025, .975]).tolist()}
    report["status"] = "QWEN_K32_PACKED_NATURAL_NLL_STAGED_SUMMARIZED"
    report["length_degradation_secondary"] = degradation
    report["staging"] = {"primary_receipt_sha256": base.sha256(primary_receipt_path),
                         "stage_b_was_conditionally_authorized": True}
    report["summary_code_sha256"] = base.sha256(Path(__file__))
    return report


def summarize_failure(primary: dict, control: dict, primary_receipt_path: Path) -> dict:
    validate_primary_receipt(primary, primary_receipt_path, authorized=False)
    if control["manifest"].get("primary_receipt_sha256") != base.sha256(primary_receipt_path):
        raise ValueError("control run is not bound to the supplied primary receipt")
    for key in ("checkpoint_weight_sha256", "config_sha256", "data_manifest_sha256",
                "data_rows_sha256", "boundary_receipt_sha256"):
        if primary["bound"][key] != control["bound"][key]:
            raise ValueError("primary and control identities differ")
    for cell, metadata in primary["metadata"].items():
        if control["metadata"].get(cell) != metadata:
            raise ValueError("primary and control row metadata differ")

    ids = [f"qwen-k32-natural-{index:03d}" for index in range(base.STREAMS)]
    cells = {**primary["cells"], **control["cells"]}
    values = np.array([[[cells[arm, sid, length] for arm in base.ARMS]
                        for length in base.LENGTHS] for sid in ids], dtype=np.float64)
    indices = np.random.default_rng(base.BOOTSTRAP_SEED).integers(
        0, base.STREAMS, size=(base.BOOTSTRAP_SAMPLES, base.STREAMS))
    draws, means = values[indices].mean(axis=1), values.mean(axis=0)
    yarn_retention = base.finite_exp(float(means[0, 0] - means[0, 2]))
    yarn_delta = draws[:, 1, 2] - draws[:, 1, 0]
    yarn_ci = np.quantile(yarn_delta, [.025, .975]).tolist()
    control_pass = bool(yarn_retention is not None and yarn_retention >= .875 and yarn_ci[1] < 0)
    primary_receipt = summarize_primary(primary)
    return {
        "status": "QWEN_K32_PACKED_NATURAL_NLL_FAILURE_DIAGNOSED",
        "primary_resolver": primary_receipt,
        "yarn_positive_control": {
            "ppl_retention_32k": yarn_retention,
            "retention_threshold": .875,
            "mean_delta_nll_64k_vs_native": float(means[1, 2] - means[1, 0]),
            "paired_stream_ci95": yarn_ci,
            "passes_both_gates": control_pass,
        },
        "diagnosis": (
            "INDEX_SPECIFIC_FAILURE" if control_pass else "SHARED_OR_UNRESOLVED_NATURAL_FAILURE"
        ),
        "next_gpu_action": "STOP",
        "raw_hashes": {
            "primary_receipt_sha256": base.sha256(primary_receipt_path),
            **{"primary_" + key: value for key, value in primary["raw_hashes"].items()},
            **{"control_" + key: value for key, value in control["raw_hashes"].items()},
        },
        "evidence_limit": (
            "Failure attribution using the preregistered YaRN positive control; no profile rescue, "
            "new split, or method selection."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("primary", "final", "failure"), required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--primary-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        primary = load_stage(args.primary_root, "primary")
        if args.stage == "primary":
            if args.baseline_root is not None or args.primary_receipt is not None:
                raise ValueError("primary summary accepts only --primary-root")
            report = summarize_primary(primary)
        elif args.stage == "final":
            if args.baseline_root is None or args.primary_receipt is None:
                raise ValueError("final summary requires baseline root and primary receipt")
            report = summarize_final(
                primary, load_stage(args.baseline_root, "baseline"), args.primary_receipt)
        else:
            if args.baseline_root is None or args.primary_receipt is None:
                raise ValueError("failure summary requires control root and primary receipt")
            report = summarize_failure(
                primary, load_stage(args.baseline_root, "control"), args.primary_receipt)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_OR_INCOMPLETE_STAGED_NLL_PANEL"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(report["status"])
    return 0 if report["status"] in {
        "QWEN_K32_PACKED_NATURAL_NLL_PRIMARY_SUMMARIZED",
        "QWEN_K32_PACKED_NATURAL_NLL_STAGED_SUMMARIZED",
        "QWEN_K32_PACKED_NATURAL_NLL_FAILURE_DIAGNOSED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
