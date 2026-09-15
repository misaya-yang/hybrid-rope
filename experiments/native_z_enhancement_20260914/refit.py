#!/usr/bin/env python3
"""Refit the successful Native-Z5 path on all 50 frozen development documents.

This is deliberately not another hyperparameter search.  The original V1 run
selected step 35 on its selection split.  V2 freezes that optimizer budget and
uses the union of the original design, selection, and internal-confirm splits
to estimate one final five-degree table.  The already-used PPL46 panel is only a
development gate; a promoted table still requires fresh task evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.native_z_enhancement_20260914.consensus_report import (
    paired_nll_contrast,
)
from experiments.native_z_enhancement_20260914.optimize import (
    append_jsonl,
    atomic_json,
    evaluate_indices,
    heldout_prefix_rows,
    mean_dense,
    segment_losses,
)
from experiments.native_z_enhancement_20260914.report import read_jsonl
from scripts.lib.rope.knot_allocation import float32_sha256, install_z5_knot


REFIT_STEPS = 35
REFIT_LR = 3e-3
EXPECTED_SPLITS = {"design": 16, "selection": 16, "internal_confirm": 18}


def calibration_indices(manifest: dict) -> list[int]:
    """Return the exact 50-row union of the original frozen development splits."""

    splits: dict[str, list[int]] = defaultdict(list)
    for record in manifest["optimization"]["records"]:
        splits[str(record["split"])].append(int(record["row"]))
    if {name: len(rows) for name, rows in splits.items()} != EXPECTED_SPLITS:
        raise ValueError("Native-Z5 development split identity drift")
    rows = sorted(row for values in splits.values() for row in values)
    if rows != list(range(50)):
        raise ValueError("Native-Z5 refit requires the exact rows 0..49")
    return rows


def heldout_mapping(original_rows: list[dict], refit_rows: list[dict]):
    by_arm: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for row in original_rows + refit_rows:
        arm = str(row["arm"])
        key = (int(row["document"]), int(row["length"]))
        if key in by_arm[arm]:
            raise ValueError(f"duplicate held-out row: {arm}/{key}")
        by_arm[arm][key] = row
    expected = {"native", "native_z5", "native_z5_refit"}
    if set(by_arm) != expected:
        raise ValueError(f"held-out arms drift: {sorted(by_arm)}")
    reference = set(by_arm["native"])
    if any(set(by_arm[arm]) != reference for arm in expected):
        raise ValueError("held-out NLL rows are not exactly paired")
    return by_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-optimization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=REFIT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=REFIT_LR)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.steps != REFIT_STEPS or not math.isclose(
        args.learning_rate, REFIT_LR, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("Native-Z5 refit freezes V1's selected step 35 and lr=0.003")

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text()) == {"status": "COMPLETE"}:
        print(json.dumps({"status": "SKIP_COMPLETE"}))
        return
    progress = output / "progress.jsonl"
    if progress.exists():
        raise FileExistsError("refit is not resumable; use a new output after interruption")

    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "OLMO_NATIVE_Z5_ASSETS_V2":
        raise ValueError("Native-Z5 asset contract drift")
    indices = calibration_indices(manifest)
    token_rows = np.load(
        args.assets / "pg19_validation_50x4097.npy", mmap_mode="r", allow_pickle=False
    )
    if tuple(token_rows.shape) != (50, 4097):
        raise ValueError("Native-Z5 optimization tensor drift")

    original = json.loads(
        (args.original_optimization / "optimization_result.json").read_text()
    )
    if original.get("status") != "OLMO_NATIVE_Z5_OPTIMIZATION_COMPLETE_V1":
        raise ValueError("original Native-Z5 optimization is incomplete")
    if int(original.get("best_step", -1)) != REFIT_STEPS:
        raise ValueError("V1 selected-step identity drift")

    import torch
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model, seed_all
    from experiments.olmo_recovery_20260912.runtime import validate_cuda

    seed_all(args.seed)
    environment = validate_cuda()
    model, _, _ = load_model(args.model.resolve(), "Native", training=False)
    native_rotary = model.model.rotary_emb
    native_values = native_rotary.inv_freq.detach().cpu().float().numpy().copy()
    knot, install_receipt = install_z5_knot(model, support_factor=1.0)
    if knot.receipt()["active_sha256_float32"] != float32_sha256(native_values):
        raise RuntimeError("Z5 Native initialization table is not bit-exact")

    started = time.perf_counter()
    native = evaluate_indices(model, token_rows, indices)
    optimizer = torch.optim.AdamW(
        [knot.gap_logits], lr=args.learning_rate, weight_decay=0.0, fused=True
    )
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        segment_values = []
        for index in indices:
            ids = torch.as_tensor(
                token_rows[index], dtype=torch.long, device="cuda"
            ).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                segments = segment_losses(model, ids)
                objective = segments.mean()
            (objective / len(indices)).backward()
            segment_values.append(segments.detach().float().cpu().numpy())
        gradient = knot.gap_logits.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Native-Z5 refit gradient is missing or non-finite")
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        torch.nn.utils.clip_grad_norm_([knot.gap_logits], 1.0, error_if_nonfinite=True)
        optimizer.step()
        knot.project_()
        append_jsonl(
            progress,
            {
                "step": step,
                "calibration_segment_nll": np.mean(
                    np.stack(segment_values), axis=0
                ).tolist(),
                "calibration_dense_nll": float(np.mean(segment_values)),
                "gradient_norm": gradient_norm,
                "maximum_gap_logit_abs": float(knot.gap_logits.detach().abs().max()),
            },
        )

    candidate = evaluate_indices(model, token_rows, indices)
    active_values = knot.realized_inv_freq().detach().cpu().float().numpy()
    table = {
        "status": "FROZEN_NATIVE_Z5_REFIT_TABLE_V2",
        "candidate_id": "olmo2_1b_native_z5_all50_refit",
        "role": "checkpoint_calibrated_refit_candidate",
        "values_float32": active_values.tolist(),
        "gain": 1.0,
        "table_sha256_float32": float32_sha256(active_values),
        "native_table_sha256_float32": float32_sha256(native_values),
        "construction": {
            **knot.receipt(),
            "optimizer": {
                "name": "AdamW",
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            "calibration_rows": len(indices),
            "calibration_rule": (
                "union of V1 design16, selection16, and internal-confirm18; "
                "fixed V1-selected step 35; no V2 trajectory selection"
            ),
            "objective": "mean of four equal 1024-target segment NLLs",
        },
    }
    atomic_json(output / "table.json", table)

    holdout_values = np.load(
        Path(manifest["heldout_nll"]["array"]), mmap_mode="r", allow_pickle=False
    )
    heldout_refit = heldout_prefix_rows(model, holdout_values, arm="native_z5_refit")
    with (output / "heldout_nll_rows.jsonl").open("w") as stream:
        for row in heldout_refit:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    original_rows = read_jsonl(args.original_optimization / "heldout_nll_rows.jsonl")
    by_arm = heldout_mapping(original_rows, heldout_refit)
    refit_native = paired_nll_contrast(by_arm, "native_z5_refit", "native")
    refit_v1 = paired_nll_contrast(
        by_arm, "native_z5_refit", "native_z5", seed=20260919
    )
    advance = bool(
        refit_native["4096"]["paired_document_bootstrap_ci95"][1] < 0.0
        and refit_v1["4096"]["delta_candidate_minus_baseline"] < 0.0
    )
    result = {
        "status": "OLMO_NATIVE_Z5_ALL50_REFIT_COMPLETE_V2",
        "scientific_question": (
            "Does a no-selection all50 refit strengthen the successful nonlinear "
            "Native-Z5 path without adding frequency degrees of freedom?"
        ),
        "checkpoint": str(args.model.resolve()),
        "model_weight_updates": 0,
        "parameterization": install_receipt,
        "optimizer": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "calibration": {
            "rows": len(indices),
            "native_dense_nll": mean_dense(native),
            "candidate_dense_nll": mean_dense(candidate),
            "delta_candidate_minus_native": mean_dense(candidate) - mean_dense(native),
        },
        "heldout_development_gate": {
            "refit_minus_native": refit_native,
            "refit_minus_v1": refit_v1,
        },
        "decision": {
            "advance_to_fresh_task_confirmation": advance,
            "refit_4k_nll_ci_below_native": (
                refit_native["4096"]["paired_document_bootstrap_ci95"][1] < 0.0
            ),
            "refit_4k_point_below_v1": (
                refit_v1["4096"]["delta_candidate_minus_baseline"] < 0.0
            ),
        },
        "table": {
            key: table[key]
            for key in (
                "candidate_id",
                "table_sha256_float32",
                "native_table_sha256_float32",
                "gain",
            )
        },
        "environment": environment,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "claim_boundary": (
            "The reused PPL46 panel is a development gate, not fresh confirmation. "
            "Promotion requires a newly generated task panel and cannot reuse the "
            "observed RULER130/Natural-QA99 as confirmatory evidence."
        ),
    }
    atomic_json(output / "refit_result.json", result)
    atomic_json(status_path, {"status": "COMPLETE"})
    print(json.dumps({"status": result["status"], **result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
