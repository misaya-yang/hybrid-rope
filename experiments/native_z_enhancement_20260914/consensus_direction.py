#!/usr/bin/env python3
"""Build a split-consistent signed Native direction in the fixed Z5 tangent space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.native_z_enhancement_20260914.optimize import (
    atomic_json,
    evaluate_indices,
    heldout_prefix_rows,
    mean_dense,
    segment_losses,
)
from scripts.lib.rope.knot_allocation import float32_sha256, install_z5_knot


ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
DESIGN_BLOCK_SIZES = (5, 5, 6)
MAX_ABS_GAP_LOGIT_AT_ALPHA_ONE = 1.0


def zero_sum_basis(size: int) -> np.ndarray:
    """Return an orthonormal basis for vectors orthogonal to all-ones."""

    if size < 2:
        raise ValueError("zero-sum basis needs at least two coordinates")
    raw = np.eye(size, dtype=np.float64)[:, :-1]
    raw[-1, :] = -1.0
    basis, _ = np.linalg.qr(raw, mode="reduced")
    if not np.allclose(basis.T @ basis, np.eye(size - 1), atol=1e-12):
        raise RuntimeError("zero-sum basis is not orthonormal")
    if not np.allclose(basis.sum(axis=0), 0.0, atol=1e-12):
        raise RuntimeError("basis does not remove the shared-logit null direction")
    return basis


def minimum_norm_convex_combination(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the minimum-norm point in the convex hull of at most a few vectors."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("vectors must have shape [blocks, dimensions]")
    best = None
    count = values.shape[0]
    gram = values @ values.T
    for mask in range(1, 1 << count):
        active = [index for index in range(count) if mask & (1 << index)]
        sub = gram[np.ix_(active, active)]
        ones = np.ones(len(active), dtype=np.float64)
        system = np.block([
            [sub, ones[:, None]],
            [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
        ])
        target = np.concatenate((np.zeros(len(active), dtype=np.float64), [1.0]))
        solution, _, _, _ = np.linalg.lstsq(system, target, rcond=1e-12)
        active_weights = solution[:-1]
        if not np.isfinite(active_weights).all() or abs(active_weights.sum() - 1.0) > 1e-8:
            continue
        if np.min(active_weights) < -1e-10:
            continue
        active_weights = np.maximum(active_weights, 0.0)
        active_weights /= active_weights.sum()
        weights = np.zeros(count, dtype=np.float64)
        weights[active] = active_weights
        point = weights @ values
        candidate = (float(point @ point), weights, point)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("failed to solve the three-block convex-hull problem")
    return best[1], best[2]


def maximin_direction(
    block_gradients: np.ndarray, log_frequency_jacobian: np.ndarray,
) -> dict:
    """Solve the common-descent max-min problem after removing the logit null mode."""

    gradients = np.asarray(block_gradients, dtype=np.float64)
    jacobian = np.asarray(log_frequency_jacobian, dtype=np.float64)
    if gradients.ndim != 2 or jacobian.ndim != 2 or gradients.shape[1] != jacobian.shape[1]:
        raise ValueError("gradient/Jacobian dimensions do not match")
    basis = zero_sum_basis(gradients.shape[1])
    reduced_gradients = gradients @ basis
    reduced_jacobian = jacobian @ basis
    metric = reduced_jacobian.T @ reduced_jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues[0] <= 1e-12 or not np.isfinite(eigenvalues).all():
        raise RuntimeError(f"Z5 tangent metric is singular: {eigenvalues.tolist()}")
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    whitened = reduced_gradients @ inverse_sqrt
    weights, hull_point = minimum_norm_convex_combination(whitened)
    margin = float(np.linalg.norm(hull_point))
    if margin <= 1e-10:
        return {
            "advance": False,
            "margin": margin,
            "convex_weights": weights.tolist(),
            "metric_eigenvalues": eigenvalues.tolist(),
            "pairwise_gradient_cosines": pairwise_cosines(gradients),
            "reason": "origin lies in or numerically touches the convex hull of block gradients",
        }
    whitened_direction = -hull_point / margin
    reduced_direction = inverse_sqrt @ whitened_direction
    full_direction = basis @ reduced_direction
    metric_norm = float(reduced_direction @ metric @ reduced_direction)
    slopes = gradients @ full_direction
    if not np.all(slopes < -margin + 1e-8) or not math.isclose(metric_norm, 1.0, abs_tol=1e-8):
        raise RuntimeError("max-min direction certificate failed")
    maximum = float(np.max(np.abs(full_direction)))
    applied = full_direction * (MAX_ABS_GAP_LOGIT_AT_ALPHA_ONE / maximum)
    return {
        "advance": True,
        "margin": margin,
        "convex_weights": weights.tolist(),
        "metric_eigenvalues": eigenvalues.tolist(),
        "unit_metric_direction_reduced": reduced_direction.tolist(),
        "unit_metric_direction_gap_logits": full_direction.tolist(),
        "unit_metric_block_slopes": slopes.tolist(),
        "applied_direction_gap_logits": applied.tolist(),
        "applied_block_slopes": (gradients @ applied).tolist(),
        "normalization": (
            f"alpha=1 has max absolute gap-logit change "
            f"{MAX_ABS_GAP_LOGIT_AT_ALPHA_ONE}; existing hard bound is separate"
        ),
        "pairwise_gradient_cosines": pairwise_cosines(gradients),
    }


def pairwise_cosines(gradients: np.ndarray) -> dict[str, float]:
    values = np.asarray(gradients, dtype=np.float64)
    result = {}
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            denominator = float(np.linalg.norm(values[left]) * np.linalg.norm(values[right]))
            result[f"{left}-{right}"] = float(values[left] @ values[right] / denominator)
    return result


def log_frequency_jacobian(knot) -> np.ndarray:
    """Autodiff d(log omega_k)/d(gap_logit_j) at exact Native initialization."""

    import torch

    knot.reset_native_()
    values = knot.realized_inv_freq().double().log()
    rows = []
    for index in range(values.numel()):
        gradient = torch.autograd.grad(
            values[index], knot.gap_logits, retain_graph=True, allow_unused=False,
        )[0]
        rows.append(gradient.detach().double().cpu().numpy())
    result = np.stack(rows)
    null_residual = float(np.max(np.abs(result.sum(axis=1))))
    if not np.isfinite(result).all() or null_residual > 1e-5:
        raise RuntimeError("log-frequency Jacobian does not respect the softmax null direction")
    # Autograd is sourced from FP32 gap logits, so the exact softmax-shift null
    # identity arrives with roughly 1e-7 round-off.  Project it explicitly
    # before constructing the five-dimensional metric.
    result -= result.mean(axis=1, keepdims=True)
    return result


def split_blocks(indices: list[int]) -> list[list[int]]:
    if len(indices) != sum(DESIGN_BLOCK_SIZES):
        raise ValueError("consensus design split must contain exactly 16 rows")
    result = []
    start = 0
    for size in DESIGN_BLOCK_SIZES:
        result.append(indices[start:start + size])
        start += size
    return result


def block_gradients(model, knot, token_rows, blocks) -> tuple[np.ndarray, list[dict]]:
    import torch

    gradients = []
    receipts = []
    for block_index, indices in enumerate(blocks):
        knot.reset_native_()
        knot.gap_logits.grad = None
        segment_values = []
        for index in indices:
            ids = torch.as_tensor(
                token_rows[index].copy(), dtype=torch.long, device="cuda",
            ).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                segments = segment_losses(model, ids)
                objective = segments.mean()
            (objective / len(indices)).backward()
            segment_values.append(segments.detach().float().cpu().numpy())
        gradient = knot.gap_logits.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("consensus block gradient is missing or non-finite")
        value = gradient.detach().double().cpu().numpy()
        value -= value.mean()
        gradients.append(value)
        receipts.append({
            "block": block_index,
            "rows": indices,
            "mean_segment_nll": np.mean(np.stack(segment_values), axis=0).tolist(),
            "dense_nll": float(np.mean(segment_values)),
            "gradient_gap_logits": value.tolist(),
            "gradient_norm": float(np.linalg.norm(value)),
        })
    return np.stack(gradients), receipts


def segment_mean(records: list[dict]) -> list[float]:
    return np.mean(np.asarray([row["segment_nll"] for row in records]), axis=0).tolist()


def make_table(knot, state, native_values, *, candidate_id: str, construction: dict) -> dict:
    knot.set_gap_logits_(state)
    active = knot.realized_inv_freq().detach().cpu().float().numpy()
    return {
        "status": "FROZEN_NATIVE_CONSENSUS_TABLE_V1",
        "candidate_id": candidate_id,
        "role": "checkpoint_calibrated_consensus_control",
        "values_float32": active.tolist(),
        "gain": 1.0,
        "table_sha256_float32": float32_sha256(active),
        "native_table_sha256_float32": float32_sha256(native_values),
        "construction": {**knot.receipt(), **construction},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-optimization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "COMPLETE":
        print(json.dumps({"status": "SKIP_COMPLETE"}))
        return
    if any(output.glob("*.incomplete")):
        raise FileExistsError("consensus output contains an incomplete artifact")

    manifest = json.loads((args.assets / "manifest.json").read_text())
    if manifest.get("contract") != "OLMO_NATIVE_Z5_ASSETS_V2":
        raise ValueError("Native consensus asset contract drift")
    native_status = json.loads((args.native_optimization / "status.json").read_text())
    if native_status != {"status": "COMPLETE"}:
        raise ValueError("original Native-Z5 optimization must complete first")
    token_rows = np.load(args.assets / "pg19_validation_50x4097.npy", mmap_mode="r", allow_pickle=False)
    splits = {name: [] for name in ("design", "selection", "internal_confirm")}
    for row in manifest["optimization"]["records"]:
        splits[row["split"]].append(int(row["row"]))
    if {key: len(value) for key, value in splits.items()} != {
        "design": 16, "selection": 16, "internal_confirm": 18,
    }:
        raise ValueError("Native consensus split identity drift")

    import torch
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model, seed_all
    from experiments.olmo_recovery_20260912.runtime import validate_cuda

    seed_all(args.seed)
    environment = validate_cuda()
    model, _, _ = load_model(args.model.resolve(), "Native", training=False)
    native_values = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy().copy()
    knot, install_receipt = install_z5_knot(model, support_factor=1.0)
    started = time.perf_counter()
    gradients, block_receipts = block_gradients(
        model, knot, token_rows, split_blocks(splits["design"]),
    )
    jacobian = log_frequency_jacobian(knot)
    certificate = maximin_direction(gradients, jacobian)
    result = {
        "status": "NATIVE_Z5_CONSENSUS_DIRECTION_V1",
        "scientific_question": (
            "Does the fixed five-dimensional Native tangent contain a signed LM-risk "
            "direction that descends on all frozen design blocks?"
        ),
        "checkpoint": str(args.model.resolve()),
        "model_weight_updates": 0,
        "parameterization": install_receipt,
        "design_blocks": block_receipts,
        "certificate": certificate,
        "alphas": list(ALPHAS),
        "environment": environment,
        "runtime_seconds_before_line_search": time.perf_counter() - started,
        "claim_boundary": (
            "A positive margin is a first-order certificate in the Native Z5 tangent only; "
            "held-out model/task results determine empirical value."
        ),
    }
    if not certificate["advance"]:
        result["decision"] = "STOP_NO_COMMON_FIRST_ORDER_DIRECTION"
        atomic_json(output / "consensus_result.json", result)
        atomic_json(status_path, {"status": "COMPLETE", "advance": False})
        print(json.dumps({"status": result["decision"], "margin": certificate["margin"]}))
        return

    direction = torch.as_tensor(
        certificate["applied_direction_gap_logits"],
        dtype=knot.gap_logits.dtype,
        device=knot.gap_logits.device,
    )
    native_selection = evaluate_indices(model, token_rows, splits["selection"])
    native_selection_mean = mean_dense(native_selection)
    line_search = []
    best_alpha = 0.0
    best_selection = native_selection_mean
    for alpha in ALPHAS:
        state = direction * alpha
        knot.set_gap_logits_(state)
        selection = evaluate_indices(model, token_rows, splits["selection"])
        value = mean_dense(selection)
        receipt = knot.receipt()
        line_search.append({
            "alpha": alpha,
            "selection_dense_nll": value,
            "delta_vs_native": value - native_selection_mean,
            "selection_segment_nll": segment_mean(selection),
            "table_sha256_float32": receipt["active_sha256_float32"],
            "minimum_normalized_gap": receipt["minimum_normalized_gap"],
            "maximum_abs_gap_logit": float(state.abs().max()),
        })
        if value < best_selection - 1e-12:
            best_selection = value
            best_alpha = alpha
    result["line_search"] = line_search
    result["best_alpha"] = best_alpha
    result["selection_native_nll"] = native_selection_mean
    result["selection_best_nll"] = best_selection
    if best_alpha == 0.0:
        result["decision"] = "STOP_NO_SELECTION_IMPROVEMENT"
        atomic_json(output / "consensus_result.json", result)
        atomic_json(status_path, {"status": "COMPLETE", "advance": False})
        print(json.dumps({"status": result["decision"], "margin": certificate["margin"]}))
        return

    plus_state = direction * best_alpha
    minus_state = -plus_state
    evaluations = {}
    for name, state in (("native", torch.zeros_like(direction)), ("consensus_plus", plus_state), ("consensus_minus", minus_state)):
        knot.set_gap_logits_(state)
        evaluations[name] = {
            split: evaluate_indices(model, token_rows, indices)
            for split, indices in splits.items()
        }
    result["split_evaluations"] = {
        name: {
            split: {
                "rows": len(rows),
                "dense_nll": mean_dense(rows),
                "segment_nll": segment_mean(rows),
            }
            for split, rows in split_values.items()
        }
        for name, split_values in evaluations.items()
    }
    common_construction = {
        "method": "split_consistent_signed_native_direction_v1",
        "design_block_sizes": list(DESIGN_BLOCK_SIZES),
        "selection_alphas": list(ALPHAS),
        "selected_alpha": best_alpha,
        "direction_certificate": certificate,
        "selection_uses_task_outputs": False,
    }
    plus_table = make_table(
        knot, plus_state, native_values,
        candidate_id="olmo2_1b_native_z5_consensus_plus",
        construction={**common_construction, "direction_sign": "+"},
    )
    minus_table = make_table(
        knot, minus_state, native_values,
        candidate_id="olmo2_1b_native_z5_consensus_minus",
        construction={**common_construction, "direction_sign": "-"},
    )
    atomic_json(output / "table_plus.json", plus_table)
    atomic_json(output / "table_minus.json", minus_table)

    heldout_values = np.load(Path(manifest["heldout_nll"]["array"]), mmap_mode="r", allow_pickle=False)
    heldout_rows = []
    for name, state in (("consensus_plus", plus_state), ("consensus_minus", minus_state)):
        knot.set_gap_logits_(state)
        heldout_rows.extend(heldout_prefix_rows(model, heldout_values, arm=name))
    heldout_path = output / "heldout_nll_rows.jsonl"
    with heldout_path.open("w") as stream:
        for row in heldout_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result.update({
        "decision": "ADVANCE_TO_HELDOUT_TASKS",
        "tables": {
            "consensus_plus": plus_table["table_sha256_float32"],
            "consensus_minus": minus_table["table_sha256_float32"],
        },
        "heldout_rows_sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    })
    atomic_json(output / "consensus_result.json", result)
    atomic_json(status_path, {"status": "COMPLETE", "advance": True})
    print(json.dumps({
        "status": result["decision"], "margin": certificate["margin"],
        "best_alpha": best_alpha, "tables": result["tables"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
