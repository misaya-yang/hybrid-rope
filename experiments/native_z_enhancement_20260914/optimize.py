#!/usr/bin/env python3
"""Optimize only five Native-support z degrees on frozen OLMo-2-1B."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts.lib.rope.knot_allocation import float32_sha256, install_z5_knot


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def segment_losses(model, ids):
    """Return four equal 1024-target losses; their mean is dense 4K NLL."""
    import torch
    import torch.nn.functional as F

    if tuple(ids.shape) != (1, 4097):
        raise ValueError("Native optimization rows must contain 4097 tokens")
    position_ids = torch.arange(4097, device=ids.device).unsqueeze(0)
    hidden = model.model(
        input_ids=ids, position_ids=position_ids, use_cache=False, return_dict=True,
    ).last_hidden_state[0]
    losses = []
    for segment in range(4):
        target_start = 1 + 1024 * segment
        target_stop = target_start + 1024
        selected = hidden[target_start - 1:target_stop - 1]
        logits = model.lm_head(selected).float()
        targets = ids[0, target_start:target_stop]
        losses.append(F.cross_entropy(logits, targets, reduction="mean"))
    return torch.stack(losses)


def evaluate_indices(model, token_rows, indices):
    import torch

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in indices:
            ids = torch.as_tensor(token_rows[index], dtype=torch.long, device="cuda").unsqueeze(0)
            values = segment_losses(model, ids).float().cpu().numpy()
            records.append({
                "row": int(index),
                "segment_nll": values.tolist(),
                "dense_nll": float(values.mean()),
            })
    return records


def mean_dense(values):
    return float(np.mean([row["dense_nll"] for row in values]))


def heldout_prefix_rows(model, values, *, arm):
    import torch
    import torch.nn.functional as F

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for document in range(values.shape[0]):
            ids = torch.as_tensor(values[document, :4097].copy(), dtype=torch.long, device="cuda").unsqueeze(0)
            position_ids = torch.arange(4097, device="cuda").unsqueeze(0)
            hidden = model.model(
                input_ids=ids, position_ids=position_ids, use_cache=False, return_dict=True,
            ).last_hidden_state[0]
            for length in (1024, 2048, 4096):
                total_loss = 0.0
                total_count = 0
                for start in range(0, length, 256):
                    stop = min(start + 256, length)
                    logits = model.lm_head(hidden[start:stop]).float()
                    targets = ids[0, start + 1:stop + 1]
                    losses = F.cross_entropy(logits, targets, reduction="sum")
                    total_loss += float(losses)
                    total_count += stop - start
                records.append({
                    "arm": arm,
                    "document": document,
                    "length": length,
                    "loss_sum": total_loss,
                    "target_count": total_count,
                    "nll": total_loss / total_count,
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.steps != 40 or not math.isclose(args.learning_rate, 3e-3, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Native-Z5 confirmation freezes the prior F3 optimizer budget: 40 steps, lr=0.003")

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text()) == {"status": "COMPLETE"}:
        print(json.dumps({"status": "SKIP_COMPLETE"}))
        return
    progress = output / "progress.jsonl"
    if progress.exists():
        raise FileExistsError("optimization is not resumable; use a new output after an interrupted run")

    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "OLMO_NATIVE_Z5_ASSETS_V2":
        raise ValueError("Native-Z5 asset contract drift")
    token_rows = np.load(args.assets / "pg19_validation_50x4097.npy", mmap_mode="r", allow_pickle=False)
    if tuple(token_rows.shape) != (50, 4097):
        raise ValueError("Native-Z5 optimization tensor drift")
    splits = defaultdict(list)
    for row in manifest["optimization"]["records"]:
        splits[row["split"]].append(int(row["row"]))
    if {key: len(value) for key, value in splits.items()} != {
        "design": 16, "selection": 16, "internal_confirm": 18,
    }:
        raise ValueError("Native-Z5 split identity drift")

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
    native = {}
    for split, indices in splits.items():
        native[split] = evaluate_indices(model, token_rows, indices)

    optimizer = torch.optim.AdamW(
        [knot.gap_logits], lr=args.learning_rate, weight_decay=0.0, fused=True,
    )
    best_state = knot.gap_logits.detach().cpu().clone()
    best_step = 0
    best_selection = mean_dense(native["selection"])
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        design_segments = []
        for index in splits["design"]:
            ids = torch.as_tensor(token_rows[index], dtype=torch.long, device="cuda").unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                segments = segment_losses(model, ids)
                objective = segments.mean()
            (objective / len(splits["design"])).backward()
            design_segments.append(segments.detach().float().cpu().numpy())
        gradient = knot.gap_logits.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Native-Z5 gradient is missing or non-finite")
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        torch.nn.utils.clip_grad_norm_([knot.gap_logits], 1.0, error_if_nonfinite=True)
        optimizer.step()
        knot.project_()
        selection = evaluate_indices(model, token_rows, splits["selection"])
        selection_mean = mean_dense(selection)
        if selection_mean < best_selection:
            best_selection = selection_mean
            best_state = knot.gap_logits.detach().cpu().clone()
            best_step = step
        append_jsonl(progress, {
            "step": step,
            "design_segment_nll": np.mean(np.stack(design_segments), axis=0).tolist(),
            "design_dense_nll": float(np.mean(design_segments)),
            "selection_dense_nll": selection_mean,
            "selection_delta_vs_native": selection_mean - mean_dense(native["selection"]),
            "gradient_norm": gradient_norm,
            "maximum_gap_logit_abs": float(knot.gap_logits.detach().abs().max()),
        })

    knot.set_gap_logits_(best_state)
    candidate = {
        split: evaluate_indices(model, token_rows, indices)
        for split, indices in splits.items()
    }
    active_values = knot.realized_inv_freq().detach().cpu().float().numpy()
    table = {
        "status": "FROZEN_NATIVE_Z5_TABLE_V1",
        "candidate_id": "olmo2_1b_native_z5_pg19_validation",
        "role": "checkpoint_calibrated_candidate",
        "values_float32": active_values.tolist(),
        "gain": 1.0,
        "table_sha256_float32": float32_sha256(active_values),
        "native_table_sha256_float32": float32_sha256(native_values),
        "construction": {
            **knot.receipt(),
            "best_step": best_step,
            "optimizer": {"name": "AdamW", "steps": args.steps, "learning_rate": args.learning_rate, "seed": args.seed},
            "objective": "mean of four equal 1024-target segment NLLs, algebraically identical to dense 4K NLL",
            "selection": "lowest PG19-validation selection-split dense 4K NLL across steps 0..40",
        },
    }
    atomic_json(output / "table.json", table)

    holdout_values = np.load(Path(manifest["heldout_nll"]["array"]), mmap_mode="r", allow_pickle=False)
    knot.reset_native_()
    heldout_native = heldout_prefix_rows(model, holdout_values, arm="native")
    knot.set_gap_logits_(best_state)
    heldout_candidate = heldout_prefix_rows(model, holdout_values, arm="native_z5")
    nll_path = output / "heldout_nll_rows.jsonl"
    with nll_path.open("w") as stream:
        for row in heldout_native + heldout_candidate:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "status": "OLMO_NATIVE_Z5_OPTIMIZATION_COMPLETE_V1",
        "scientific_question": "Can z-only post-hoc calibration improve a frozen mature checkpoint inside its Native window?",
        "checkpoint": str(args.model.resolve()),
        "model_weight_updates": 0,
        "parameterization": install_receipt,
        "optimizer": {"steps": args.steps, "learning_rate": args.learning_rate, "seed": args.seed},
        "best_step": best_step,
        "native": {split: {"dense_nll": mean_dense(rows), "rows": len(rows)} for split, rows in native.items()},
        "candidate": {split: {"dense_nll": mean_dense(rows), "rows": len(rows)} for split, rows in candidate.items()},
        "delta_candidate_minus_native": {
            split: mean_dense(candidate[split]) - mean_dense(native[split]) for split in splits
        },
        "table": {key: table[key] for key in ("candidate_id", "table_sha256_float32", "native_table_sha256_float32", "gain")},
        "environment": environment,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "claim_boundary": "Checkpoint-calibrated five-degree z table; not zero-search, not a universal analytic construction, and not an extrapolation result.",
    }
    atomic_json(output / "optimization_result.json", result)
    atomic_json(status_path, {"status": "COMPLETE"})
    print(json.dumps({"status": result["status"], "best_step": best_step, "table": table["table_sha256_float32"]}, sort_keys=True))


if __name__ == "__main__":
    main()
