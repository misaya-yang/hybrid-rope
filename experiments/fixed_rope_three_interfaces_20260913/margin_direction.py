#!/usr/bin/env python3
"""Propose fixed-table transition directions from full-model answer margins.

The calibration objective is the smooth bottleneck margin of one declared
answer path.  EOS is excluded by default for RULER-contains alignment and can
be included explicitly for complete-output studies.  This is not a replacement
for free generation or the official scorer.  Candidate directions preserve the
transition endpoint; the ``moment`` family also preserves cumulative exponent
dose to first order.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .tables import (
    atomic_json, find_table, make_receipt, model_geometry, read_json,
    runtime_native_inv_freq, validate_table,
)


PROPOSAL_FORMAT = "ANSWER_MARGIN_TRANSITION_PROPOSAL_V1"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_calibration_rows(
    rows: list[dict], *, split: str, tasks: tuple[str, ...],
    lengths: tuple[int, ...], rows_per_cell: int,
) -> list[dict]:
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split and row.get("task") in tasks and int(row.get("length_cap", -1)) in lengths:
            cells[(str(row["task"]), int(row["length_cap"]))].append(row)
    selected = []
    for task in tasks:
        for length in lengths:
            values = sorted(cells[(task, length)], key=lambda row: str(row["row_id"]))
            if len(values) < rows_per_cell:
                raise ValueError(f"insufficient calibration rows for {task}/{length}")
            selected.extend(values[:rows_per_cell])
    return selected


def log_auc_weights(lengths: tuple[int, ...]) -> dict[int, float]:
    if len(lengths) == 1:
        return {lengths[0]: 1.0}
    if len(lengths) < 1 or any(left >= right for left, right in zip(lengths, lengths[1:])):
        raise ValueError("lengths must be strictly increasing")
    logs = np.log(np.asarray(lengths, dtype=np.float64))
    weights = np.zeros(len(lengths), dtype=np.float64)
    gaps = np.diff(logs)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    weights /= logs[-1] - logs[0]
    return {length: float(weight) for length, weight in zip(lengths, weights)}


def direction_catalog(size: int, family: str) -> list[tuple[str, np.ndarray]]:
    values = []
    if family == "moment":
        for start in range(size - 2):
            base = np.zeros(size, dtype=np.float64)
            base[start : start + 3] = (1.0, -2.0, 1.0)
            for sign in (1.0, -1.0):
                values.append((f"moment_{start + 1}_{'plus' if sign > 0 else 'minus'}", sign * base))
    elif family == "gap":
        for left in range(size):
            for right in range(left + 1, size):
                base = np.zeros(size, dtype=np.float64)
                base[left], base[right] = 1.0, -1.0
                for sign in (1.0, -1.0):
                    values.append((f"gap_{left + 1}_{right + 1}_{'plus' if sign > 0 else 'minus'}", sign * base))
    else:
        raise ValueError(f"unknown direction family: {family}")
    return values


def select_direction(
    *, family: str, increments: np.ndarray,
    gradients_by_length: dict[int, np.ndarray], lengths: tuple[int, ...],
) -> dict:
    weights = log_auc_weights(lengths)
    candidates = []
    for name, vector in direction_catalog(len(increments), family):
        derivatives = {
            length: float(np.dot(gradients_by_length[length], vector))
            for length in lengths
        }
        auc_derivative = sum(weights[length] * derivatives[length] for length in lengths)
        common_descent = all(value < 0.0 for value in derivatives.values())
        boundary_descent = derivatives[lengths[0]] <= 0.0 and derivatives[lengths[-1]] <= 0.0
        candidates.append({
            "name": name,
            "vector": vector.tolist(),
            "derivative_by_length": {str(key): value for key, value in derivatives.items()},
            "log_auc_derivative": float(auc_derivative),
            "max_length_derivative": float(max(derivatives.values())),
            "common_descent": common_descent,
            "native_and_endpoint_nondestructive_first_order": boundary_descent,
        })
    selected = min(
        candidates,
        key=lambda item: (
            not item["common_descent"],
            not item["native_and_endpoint_nondestructive_first_order"],
            item["max_length_derivative"],
            item["log_auc_derivative"],
            item["name"],
        ),
    )
    return {**selected, "family": family, "catalog_size": len(candidates)}


def feasible_step(increments: np.ndarray, direction: np.ndarray, *, fraction: float, max_exponent_shift: float) -> float:
    if not 0.0 < fraction <= 1.0 or max_exponent_shift <= 0.0:
        raise ValueError("invalid trust-region settings")
    # The CAL check evaluates both the predicted and opposite signs, so the
    # step must keep both symmetric interventions strictly feasible.
    limits = [increments[index] / abs(value) for index, value in enumerate(direction) if value != 0.0]
    if not limits:
        raise ValueError("direction has no negative component")
    positivity_limit = min(limits)
    cumulative = np.cumsum(direction)
    phase_limit = max_exponent_shift / max(abs(cumulative).max(), 1e-30)
    step = fraction * min(positivity_limit, phase_limit)
    if step <= 0.0 or np.any(increments + step * direction <= 0.0):
        raise ValueError("failed to construct a strictly feasible transition step")
    return float(step)


def exponents_from_increments(*, pairs: int, low: int, high: int, increments: np.ndarray) -> np.ndarray:
    if len(increments) != high - low or not np.isclose(increments.sum(), 1.0, atol=1e-10):
        raise ValueError("increments do not match the fixed band/endpoint")
    exponents = np.zeros(pairs, dtype=np.float64)
    exponents[low + 1 : high + 1] = np.cumsum(increments)
    exponents[high + 1 :] = 1.0
    return exponents


def smooth_margin_loss(
    model, row: dict, *, temperature: float, grad: bool, include_eos: bool,
    margin_position: str,
):
    import torch

    prompt = list(row["prompt_ids"])
    target = list(row["target_ids"])
    if not target or not row.get("target_includes_eos"):
        raise ValueError("calibration source must expose a nonempty answer path ending in EOS")
    if not include_eos:
        target = target[:-1]
    if not target:
        raise ValueError("answer-margin objective has no non-EOS target token")
    ids = torch.tensor([[*prompt, *target[:-1]]], dtype=torch.long, device="cuda")
    labels = torch.tensor(target, dtype=torch.long, device="cuda")
    context = torch.enable_grad() if grad else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target), return_dict=True).logits[0].float()
        gold = logits.gather(-1, labels[:, None]).squeeze(-1)
        competitors = logits.scatter(-1, labels[:, None], -torch.inf).amax(-1)
        margins = gold - competitors
        if margin_position == "first":
            loss = -margins[0]
        elif margin_position == "bottleneck":
            loss = temperature * torch.logsumexp(-margins / temperature, dim=0)
        else:
            raise ValueError(f"unknown margin position policy: {margin_position}")
    return loss, margins.detach().cpu().float().numpy()


def aggregate_gradient_records(records: list[dict], lengths: tuple[int, ...]) -> dict[int, np.ndarray]:
    result = {}
    for length in lengths:
        tasks = sorted({record["task"] for record in records if record["length_cap"] == length})
        task_gradients = []
        for task in tasks:
            values = [
                np.asarray(record["gradient_epsilon"], dtype=np.float64)
                for record in records
                if record["length_cap"] == length and record["task"] == task
            ]
            task_gradients.append(np.mean(values, axis=0))
        if not task_gradients:
            raise ValueError(f"empty gradient length {length}")
        result[length] = np.mean(task_gradients, axis=0)
    return result


def evaluate_state(
    model, allocation, rows: list[dict], increments: np.ndarray, *,
    temperature: float, include_eos: bool, margin_position: str,
) -> dict:
    import torch

    logits = torch.as_tensor(np.log(increments), dtype=allocation.increment_logits.dtype, device="cuda")
    allocation.set_state_(logits, allocation.initial_log_gain)
    cells: dict[tuple[int, str], list[tuple[float, float, bool]]] = defaultdict(list)
    row_records = []
    model.eval()
    for row in rows:
        loss, margins = smooth_margin_loss(
            model, row, temperature=temperature, grad=False, include_eos=include_eos,
            margin_position=margin_position,
        )
        entry = {
            "row_id": row["row_id"], "task": row["task"],
            "length_cap": int(row["length_cap"]), "smooth_loss": float(loss),
            "bottleneck_margin": float(margins.min()),
            "all_path_margins_positive": bool(np.all(margins > 0.0)),
            "margins": margins.tolist(),
        }
        row_records.append(entry)
        cells[(entry["length_cap"], entry["task"])].append(
            (entry["smooth_loss"], entry["bottleneck_margin"], entry["all_path_margins_positive"])
        )
    by_length = {}
    for length in sorted({key[0] for key in cells}):
        tasks = sorted(key[1] for key in cells if key[0] == length)
        by_length[str(length)] = {
            "task_equal_smooth_loss": float(np.mean([
                np.mean([value[0] for value in cells[(length, task)]]) for task in tasks
            ])),
            "task_equal_bottleneck_margin": float(np.mean([
                np.mean([value[1] for value in cells[(length, task)]]) for task in tasks
            ])),
            "complete_path_positive_rate": float(np.mean([
                np.mean([value[2] for value in cells[(length, task)]]) for task in tasks
            ])),
        }
    return {"by_length": by_length, "rows": row_records}


def execute(args) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments.olmo_recovery_20260912.native_relative_allocation import install_native_relative_allocation
    from experiments.olmo_recovery_20260912.runtime import validate_cuda

    environment = validate_cuda()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config_dict = config.to_dict()
    geometry = model_geometry(config_dict)
    parent = read_json(args.parent_table)
    if parent.get("model_geometry") != geometry or float(parent.get("scale", float("nan"))) != args.scale:
        raise ValueError("parent table geometry or scale differs")
    values, gain = validate_table(find_table(parent), pairs=int(geometry["pairs"]))
    exponents = np.asarray(parent["exponents"], dtype=np.float64)
    band = parent.get("band_envelope")
    if not isinstance(band, list) or len(band) != 2:
        raise ValueError("parent does not expose one transition band")
    low, high = map(int, band)
    increments = np.diff(exponents[low : high + 1])
    if np.any(increments <= 0.0) or not np.isclose(increments.sum(), 1.0, atol=1e-6):
        raise ValueError("parent transition must be strictly positive and full-depth")
    rows = select_calibration_rows(
        read_jsonl(args.data), split=args.split, tasks=tuple(args.task),
        lengths=tuple(args.length), rows_per_cell=args.rows_per_cell,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa",
    )
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("margin-gradient calibration requires zero-dropout inference")
    model.eval().requires_grad_(False)
    allocation = install_native_relative_allocation(
        model, low=low, high=high, scale=args.scale,
        initial_exponents=exponents, initial_gain=gain,
        initial_inv_freq=values,
    )
    allocation.log_gain.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    gradient_records = []
    for index, row in enumerate(rows, 1):
        loss, margins = smooth_margin_loss(
            model, row, temperature=args.temperature, grad=True,
            include_eos=args.include_eos, margin_position=args.margin_position,
        )
        gradient_logits = torch.autograd.grad(loss, allocation.increment_logits)[0].detach().cpu().double().numpy()
        gradient_epsilon = gradient_logits / increments
        gradient_records.append({
            "row_id": row["row_id"], "task": row["task"], "length_cap": int(row["length_cap"]),
            "smooth_loss": float(loss.detach()), "bottleneck_margin": float(margins.min()),
            "all_path_margins_positive": bool(np.all(margins > 0.0)),
            "gradient_logits": gradient_logits.tolist(),
            "gradient_epsilon": gradient_epsilon.tolist(),
        })
        atomic_json(args.out / "live.json", {"phase": "gradient", "completed": index, "total": len(rows)})
    gradients = aggregate_gradient_records(gradient_records, tuple(args.length))
    proposals = {}
    native = runtime_native_inv_freq(geometry).astype(np.float64)
    for family in ("gap", "moment"):
        selected = select_direction(
            family=family, increments=increments,
            gradients_by_length=gradients, lengths=tuple(args.length),
        )
        vector = np.asarray(selected["vector"], dtype=np.float64)
        step = feasible_step(
            increments, vector, fraction=args.step_fraction,
            max_exponent_shift=args.max_exponent_shift,
        )
        states = {"parent": increments, "predicted": increments + step * vector,
                  "opposite": increments - step * vector}
        finite = {
            name: evaluate_state(
                model, allocation, rows, state, temperature=args.temperature,
                include_eos=args.include_eos, margin_position=args.margin_position,
            )
            for name, state in states.items()
        }
        tables = {}
        for name in ("predicted", "opposite"):
            state = states[name]
            candidate_exponents = exponents_from_increments(
                pairs=int(geometry["pairs"]), low=low, high=high, increments=state,
            )
            installed = (native * np.power(args.scale, -candidate_exponents)).astype(np.float32)
            receipt = make_receipt(
                candidate_id=f"{parent['candidate_id']}_margin_{family}_{name}",
                model_id=parent["model_id"], role="candidate" if name == "predicted" else "control",
                scale=args.scale, geometry=geometry, values=installed, gain=gain,
                construction={
                    "method": "answer_margin_local_transition_direction",
                    "family": family, "direction": vector.tolist(), "step": step,
                    "calibration_split": args.split,
                    "selection_rule": "minimax length derivative; Native/endpoint nondestructive preferred; log-AUC tie break",
                },
                source=str(args.parent_table), parent_candidate_id=parent["candidate_id"],
                changed_variables=["transition_increment_allocation"],
            )
            table_path = args.out / "tables" / f"{family}_{name}.json"
            atomic_json(table_path, receipt)
            tables[name] = str(table_path)
        parent_loss = {key: value["task_equal_smooth_loss"] for key, value in finite["parent"]["by_length"].items()}
        predicted_loss = {key: value["task_equal_smooth_loss"] for key, value in finite["predicted"]["by_length"].items()}
        opposite_loss = {key: value["task_equal_smooth_loss"] for key, value in finite["opposite"]["by_length"].items()}
        selected.update(
            step=step, tables=tables, finite_response=finite,
            predicted_direction_matches_finite_response=all(
                predicted_loss[key] < parent_loss[key] for key in parent_loss
            ),
            opposite_direction_is_worse=all(
                opposite_loss[key] > parent_loss[key] for key in parent_loss
            ),
        )
        proposals[family] = selected
    result = {
        "status": PROPOSAL_FORMAT,
        "model": str(args.model), "data": str(args.data), "parent_table": str(args.parent_table),
        "parent_candidate_id": parent["candidate_id"], "parent_table_sha256_float32": parent["table_sha256_float32"],
        "scale": args.scale, "band": band, "gain_fixed": gain,
        "calibration": {
            "split": args.split, "tasks": args.task, "lengths": args.length,
            "rows_per_cell": args.rows_per_cell, "row_ids": [row["row_id"] for row in rows],
            "target": (
                "smooth bottleneck margin of one declared complete answer+EOS path"
                if args.include_eos
                else "smooth bottleneck margin of declared answer tokens; EOS excluded to match RULER contains"
            ),
            "include_eos": args.include_eos,
            "margin_position": args.margin_position,
            "temperature": args.temperature,
        },
        "gradient_records": gradient_records,
        "gradient_by_length_epsilon": {str(key): value.tolist() for key, value in gradients.items()},
        "proposals": proposals,
        "environment": environment,
        "scope": "checkpoint-conditioned CAL proposal; free generation on disjoint rows is required",
    }
    atomic_json(args.out / "proposal.json", result)
    atomic_json(args.out / "status.json", {"status": "COMPLETE", "rows": len(rows)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--split", default="fit")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--length", action="append", type=int, default=[])
    parser.add_argument("--rows-per-cell", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--max-exponent-shift", type=float, default=0.03)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--margin-position", choices=("bottleneck", "first"), default="bottleneck")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.task = args.task or ["niah_single_1", "niah_multikey_1", "qa_2"]
    args.length = sorted(args.length or [4096, 8192, 16384])
    if (
        args.out.exists() or args.rows_per_cell < 1 or args.temperature <= 0.0
        or len(set(args.task)) != len(args.task) or len(set(args.length)) != len(args.length)
    ):
        raise ValueError("invalid or pre-existing margin-direction run")
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "model": str(args.model), "data": str(args.data),
            "parent_table": str(args.parent_table), "out": str(args.out),
            "tasks": args.task, "lengths": args.length, "rows_per_cell": args.rows_per_cell,
        }, indent=2))
        return
    args.out.mkdir(parents=True)
    result = execute(args)
    print(json.dumps({
        "status": result["status"], "rows": len(result["gradient_records"]),
        "directions": {
            key: {
                "name": value["name"], "common_descent": value["common_descent"],
                "finite_match": value["predicted_direction_matches_finite_response"],
            }
            for key, value in result["proposals"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
