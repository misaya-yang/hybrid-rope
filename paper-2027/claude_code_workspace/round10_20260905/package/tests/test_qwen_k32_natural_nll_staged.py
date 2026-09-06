"""Synthetic CPU-only checks for staged packed-natural NLL receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import summarize_qwen_k32_natural_nll as base
from scripts.analysis import summarize_qwen_k32_natural_nll_staged as staged
from scripts.eval import eval_qwen_k32_natural_nll as evaluator


def hashed(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_stage(root: Path, stage: str, primary_receipt_sha256=None,
                index32=3.05, index64=3.3, native64=3.5, yarn64=3.4):
    root.mkdir()
    arms = staged.PRIMARY_ARMS if stage == "primary" else staged.BASELINE_ARMS
    profiles = [{"name": arm, "tensor_sha256": values[0], "file_sha256": values[1],
                 "attention_scaling": values[2]} for arm, values in base.PROFILE.items()]
    manifest = {
        "status": base.FROZEN_STATUS, "stage": stage, "arm_order": list(arms),
        "declared_arm_order": list(base.ARMS), "checkpoint_weight_sha256": base.WEIGHT_SHA256,
        "config_sha256": base.CONFIG_SHA256, "data_manifest_sha256": hashed("data-manifest"),
        "data_rows_sha256": hashed("data-rows"),
        "source": {"name": "fresh.parquet", "sha256": hashed("source"), "start_row": 20000,
                   "consumed_row_range": [20000, 29999]},
        "tokenizer_files": [{"name": "tokenizer.json", "sha256": hashed("tokenizer"), "bytes": 123}],
        "packing_contract": base.PACKING_CONTRACT, "lengths": list(base.LENGTHS),
        "natural_streams": base.STREAMS, "executed_streams": base.STREAMS,
        "target_tokens": base.TARGET_TOKENS,
        "profiles": profiles,
        "script_sha256": base.sha256(Path("scripts/eval/eval_qwen_k32_natural_nll.py")),
        "model_source_sha256": hashed("model"), "attention_source_sha256": hashed("attention"),
        "boundary_receipt_sha256": hashed("boundary"),
        "boundary_receipt_status": "QWEN_K32_PACKED_NATURAL_TARGET_BOUNDARY_SAFE_V1",
        "use_cache": False, "compile": False, "model_updates": 0,
        "profile_selection": False, "all_profiles_loaded_before_inference": True,
        "primary_receipt_sha256": primary_receipt_sha256,
    }
    means = {
        ("Native", 32768): 3.0, ("normalized_raw_index", 32768): index32,
        ("Native", 65536): native64, ("normalized_raw_index", 65536): index64,
        ("official_equation_yarn", 32768): 3.1,
        ("official_equation_yarn", 65536): yarn64,
    }
    rows = []
    for index in range(base.STREAMS):
        sid = f"qwen-k32-natural-{index:03d}"
        for length in base.LENGTHS:
            for arm in arms:
                rows.append({"arm": arm, "sample_id": sid, "length": length,
                    "family": "natural", "variant": "packed_natural", "split": "holdout",
                    "source_row_start": 20000 + index * 10,
                    "source_row_end": 20009 + index * 10, "source_document_count": 10,
                    "source_set_sha256": hashed(f"source-set-{index}"),
                    "prompt_ids_sha256": hashed(f"prompt-{index}-{length}"),
                    "target_ids_sha256": hashed(f"target-{index}"),
                    "target_start": length - base.TARGET_TOKENS,
                    "target_tokens": base.TARGET_TOKENS,
                    "table_sha256_float32": base.PROFILE[arm][0],
                    "attention_scaling": base.PROFILE[arm][2],
                    "nll": means[arm, length] + index / 10000})
    examples = root / "examples.jsonl"
    examples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = root / "run_manifest.json"
    run.write_text(json.dumps(manifest))
    curves = {}
    native = {(row["sample_id"], row["length"]): row["nll"]
              for row in rows if row["arm"] == "Native"}
    for arm in arms:
        curves[arm] = {}
        for length in base.LENGTHS:
            cell = [row for row in rows if row["arm"] == arm and row["length"] == length]
            curves[arm][str(length)] = {
                "streams": len(cell),
                "mean_tail_nll": sum(row["nll"] for row in cell) / len(cell),
                "mean_paired_delta_vs_native": (
                    sum(row["nll"] - native[row["sample_id"], length] for row in cell) / len(cell)
                    if native else None),
            }
    result = {"status": staged.STAGE_STATUS[stage], "stage": stage,
              "curves": curves, "rows": len(rows),
              "examples_sha256": base.sha256(examples),
              "run_manifest_sha256": base.sha256(run)}
    (root / "results.json").write_text(json.dumps(result))


def test_primary_pass_authorizes_stage_b(tmp_path):
    root = tmp_path / "primary"
    write_stage(root, "primary")
    report = staged.summarize_primary(staged.load_stage(root, "primary"))
    assert report["classification"] == {"resolver": "PASS"}
    assert report["stage_b_authorized"] is True
    assert report["index_minus_native_64k"]["mean_delta_nll"] == pytest.approx(-.2)


def test_primary_failure_never_authorizes_stage_b(tmp_path):
    root = tmp_path / "primary"
    write_stage(root, "primary", index32=3.3)
    report = staged.summarize_primary(staged.load_stage(root, "primary"))
    assert report["stage_b_authorized"] is False
    assert report["classification"] == {"resolver": "NOT_PASS"}


def test_final_requires_exact_passing_primary_receipt(tmp_path):
    primary_root = tmp_path / "primary"
    write_stage(primary_root, "primary")
    primary = staged.load_stage(primary_root, "primary")
    receipt_path = tmp_path / "primary.json"
    receipt_path.write_text(json.dumps(staged.summarize_primary(primary), indent=2, sort_keys=True) + "\n")
    baseline_root = tmp_path / "baseline"
    write_stage(baseline_root, "baseline", primary_receipt_sha256=base.sha256(receipt_path))
    report = staged.summarize_final(
        primary, staged.load_stage(baseline_root, "baseline"), receipt_path)
    assert report["status"] == "QWEN_K32_PACKED_NATURAL_NLL_STAGED_SUMMARIZED"
    assert report["classification"] == {"resolver": "PASS", "index_vs_yarn": "INDEX_FAVORED"}
    assert "length_degradation_secondary" in report
    receipt_path.write_text(receipt_path.read_text() + "\n")
    with pytest.raises(ValueError):
        staged.summarize_final(
            primary, staged.load_stage(baseline_root, "baseline"), receipt_path)


def test_evaluator_accepts_only_matching_passing_primary_receipt(tmp_path):
    primary_root = tmp_path / "primary"
    write_stage(primary_root, "primary")
    report = staged.summarize_primary(staged.load_stage(primary_root, "primary"))
    path = tmp_path / "primary.json"
    path.write_text(json.dumps(report))
    profiles = [{"name": arm, "tensor_sha256": values[0], "file_sha256": values[1],
                 "attention_scaling": values[2]} for arm, values in base.PROFILE.items()]
    data = {"config_sha256": base.CONFIG_SHA256,
            "manifest_sha256": hashed("data-manifest"),
            "file": {"sha256": hashed("data-rows")}}
    assert evaluator.validate_primary_receipt(
        path, base.WEIGHT_SHA256, data, profiles, hashed("boundary"), authorized=True)[
        "stage_b_authorized"] is True
    report["stage_b_authorized"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not authorize"):
        evaluator.validate_primary_receipt(
            path, base.WEIGHT_SHA256, data, profiles, hashed("boundary"), authorized=True)


def test_failed_primary_runs_bound_yarn_positive_control(tmp_path):
    primary_root = tmp_path / "primary"
    write_stage(primary_root, "primary", index64=3.7)
    primary = staged.load_stage(primary_root, "primary")
    receipt_path = tmp_path / "primary.json"
    receipt_path.write_text(json.dumps(staged.summarize_primary(primary), indent=2, sort_keys=True) + "\n")
    control_root = tmp_path / "control"
    write_stage(control_root, "control", primary_receipt_sha256=base.sha256(receipt_path), yarn64=3.2)
    report = staged.summarize_failure(
        primary, staged.load_stage(control_root, "control"), receipt_path)
    assert report["status"] == "QWEN_K32_PACKED_NATURAL_NLL_FAILURE_DIAGNOSED"
    assert report["diagnosis"] == "INDEX_SPECIFIC_FAILURE"
    assert report["next_gpu_action"] == "STOP"


def test_evaluator_requires_matching_boundary_receipt(tmp_path):
    data = {"manifest_sha256": hashed("data-manifest"),
            "file": {"sha256": hashed("data-rows")}}
    receipt = {"status": evaluator.BOUNDARY_STATUS, "model_evaluation_status": "NOT_RUN",
               "streams": evaluator.DOCUMENTS, "paired_lengths": list(evaluator.GRID),
               "target_tokens": evaluator.TARGET_TOKENS, "safe_streams": evaluator.DOCUMENTS,
               "unsafe_streams": [], "data_manifest_sha256": data["manifest_sha256"],
               "data_rows_sha256": data["file"]["sha256"]}
    path = tmp_path / "boundary.json"
    path.write_text(json.dumps(receipt))
    assert evaluator.validate_boundary_receipt(path, data)["status"] == evaluator.BOUNDARY_STATUS
    receipt["unsafe_streams"] = ["one"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unsafe"):
        evaluator.validate_boundary_receipt(path, data)
