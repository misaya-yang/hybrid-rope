#!/usr/bin/env python3
"""Constrained full-z repair of Llama Native-window FWE first-token margins.

Unlike the earlier log-frequency-gap pilot, the QP below sees the exact
Native-relative feasibility family: fixed endpoints, exponent box constraints,
and monotone exponents.  Free generation remains the only acceptance test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .tables import (
    atomic_json, find_table, make_receipt, model_geometry, read_json,
    runtime_native_inv_freq, validate_table,
)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def exponent_feasibility(
    initial_exponents: np.ndarray, delta_interior: np.ndarray, *, tolerance: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    initial = np.asarray(initial_exponents, dtype=np.float64)
    delta = np.asarray(delta_interior, dtype=np.float64)
    if initial.ndim != 1 or delta.shape != (len(initial) - 2,):
        raise ValueError("exponent feasibility dimensions differ")
    active = initial.copy()
    active[1:-1] += delta
    valid = bool(
        active.min() >= -tolerance and active.max() <= 1.0 + tolerance
        and np.all(np.diff(active) >= -tolerance)
        and abs(float(active[0] - initial[0])) <= tolerance
        and abs(float(active[-1] - initial[-1])) <= tolerance
    )
    return valid, active


def minimum_norm_exponent_step(
    margins: np.ndarray,
    jacobian: np.ndarray,
    initial_exponents: np.ndarray,
    *,
    target_margin: float,
    trust_bound: float,
) -> np.ndarray:
    """Solve the linearized repair QP inside the declared exponent polytope."""
    from scipy.optimize import minimize

    margins = np.asarray(margins, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    initial = np.asarray(initial_exponents, dtype=np.float64)
    variables = len(initial) - 2
    if (
        initial.ndim != 1 or len(initial) < 3 or jacobian.shape != (len(margins), variables)
        or not np.isfinite(margins).all() or not np.isfinite(jacobian).all()
        or not np.isfinite(initial).all() or not math.isfinite(target_margin) or trust_bound <= 0.0
        or initial.min() < -1e-8 or initial.max() > 1.0 + 1e-8
        or np.any(np.diff(initial) < -1e-8)
    ):
        raise ValueError("invalid constrained exponent-margin system")
    lower = np.maximum(-trust_bound, -initial[1:-1])
    upper = np.minimum(trust_bound, 1.0 - initial[1:-1])
    # Each row represents m[i+1]+d[i+1] - m[i]-d[i] >= 0.
    monotone_matrix = np.zeros((len(initial) - 1, variables), dtype=np.float64)
    for edge in range(len(initial) - 1):
        if 1 <= edge <= len(initial) - 2:
            monotone_matrix[edge, edge - 1] -= 1.0
        if 1 <= edge + 1 <= len(initial) - 2:
            monotone_matrix[edge, edge] += 1.0
    initial_gaps = np.diff(initial)
    # Distinguish mathematical infeasibility of the linearized constraints from
    # a numerical failure of the minimum-norm SLSQP solve.
    from scipy.optimize import linprog

    linear_feasibility = linprog(
        c=np.zeros(variables, dtype=np.float64),
        A_ub=np.vstack((-jacobian, -monotone_matrix)),
        b_ub=np.concatenate((margins - target_margin, initial_gaps)),
        bounds=list(zip(lower, upper)), method="highs",
    )
    if linear_feasibility.status == 2:
        raise RuntimeError("linearized exponent polytope is infeasible")
    if not linear_feasibility.success:
        raise RuntimeError(
            f"linearized feasibility solver failed numerically: {linear_feasibility.message}"
        )
    result = minimize(
        fun=lambda value: 0.5 * float(np.dot(value, value)),
        x0=np.zeros(variables, dtype=np.float64),
        jac=lambda value: value,
        bounds=list(zip(lower, upper)),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda value: margins + jacobian @ value - target_margin,
                "jac": lambda value: jacobian,
            },
            {
                "type": "ineq",
                "fun": lambda value: initial_gaps + monotone_matrix @ value,
                "jac": lambda value: monotone_matrix,
            },
        ],
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"constrained full-z QP failed: {result.message}")
    step = np.asarray(result.x, dtype=np.float64)
    feasible, _ = exponent_feasibility(initial, step, tolerance=2e-7)
    if not feasible or np.max(np.abs(step)) > trust_bound + 2e-7:
        raise RuntimeError("constrained full-z QP returned an infeasible step")
    return step


def maximum_feasible_line_scale(
    initial_exponents: np.ndarray,
    step: np.ndarray,
    *,
    trust_bound: float,
) -> float:
    """Largest nonnegative alpha preserving box, monotone, and trust constraints."""
    initial = np.asarray(initial_exponents, dtype=np.float64)
    direction = np.asarray(step, dtype=np.float64)
    if direction.shape != (len(initial) - 2,) or trust_bound <= 0.0:
        raise ValueError("invalid exponent ray")
    limits = []
    for value, delta in zip(initial[1:-1], direction):
        if delta > 0.0:
            limits.extend(((1.0 - value) / delta, trust_bound / delta))
        elif delta < 0.0:
            limits.extend((value / -delta, trust_bound / -delta))
    full_direction = np.concatenate(([0.0], direction, [0.0]))
    for gap, slope in zip(np.diff(initial), np.diff(full_direction)):
        # SLSQP may leave ~1e-15 slopes on an exactly flat active boundary.
        if slope < -1e-10:
            limits.append(gap / -slope)
    finite = [float(value) for value in limits if math.isfinite(value)]
    return min(finite) if finite else math.inf


def first_margin(model, row: dict, *, grad: bool):
    import torch

    target = list(row["target_ids"])
    if not target or not row.get("target_includes_eos") or len(target) < 2:
        raise ValueError("repair target must include answer tokens and terminal EOS")
    answer = target[:-1]
    ids = torch.tensor([[*row["prompt_ids"], *answer[:-1]]], dtype=torch.long, device="cuda")
    labels = torch.tensor(answer, dtype=torch.long, device="cuda")
    context = torch.enable_grad() if grad else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=ids, use_cache=False, logits_to_keep=len(answer), return_dict=True,
        ).logits[0].float()
        gold = logits.gather(-1, labels[:, None]).squeeze(-1)
        competitor = logits.scatter(-1, labels[:, None], -torch.inf).amax(-1)
        margins = gold - competitor
    return margins[0], margins.detach().cpu().float().numpy()


def execute(args) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from .exponent_box_z import install_exponent_box_z

    validate_cuda()
    parent = read_json(args.parent_table)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    geometry = model_geometry(config.to_dict())
    if parent.get("model_geometry") != geometry or float(parent.get("scale", float("nan"))) != args.scale:
        raise ValueError("parent table geometry or scale mismatch")
    parent_values, gain = validate_table(find_table(parent), pairs=int(geometry["pairs"]))
    initial_exponents = np.asarray(parent["exponents"], dtype=np.float64)
    valid_initial, _ = exponent_feasibility(
        initial_exponents, np.zeros(len(initial_exponents) - 2),
    )
    if not valid_initial:
        raise ValueError("parent is outside the constrained Native-relative family")
    rows = sorted(
        [row for row in read_jsonl(args.data) if row.get("split") == "fit"],
        key=lambda row: str(row["row_id"]),
    )
    if len(rows) != args.fit_rows:
        raise ValueError("fit row count differs from the frozen repair contract")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa",
    )
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("constrained full-z repair requires zero dropout")
    model.eval().requires_grad_(False)
    allocation = install_exponent_box_z(
        model, scale=args.scale, initial_exponents=initial_exponents,
        initial_inv_freq=parent_values, gain=gain,
    )
    installed_native = allocation.native_inv_freq.detach().cpu().float().numpy()
    if not np.array_equal(installed_native, runtime_native_inv_freq(geometry)):
        raise ValueError("live model Native table differs from the project runtime reference")
    if not np.array_equal(allocation.realized_inv_freq().detach().cpu().float().numpy(), parent_values):
        raise ValueError("zero-delta allocation does not reproduce the parent table exactly")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    margins, jacobian, traces = [], [], []
    for index, row in enumerate(rows, 1):
        margin, all_margins = first_margin(model, row, grad=True)
        gradient = torch.autograd.grad(margin, allocation.delta_interior)[0].detach().cpu().double().numpy()
        margins.append(float(margin.detach()))
        jacobian.append(gradient)
        traces.append({
            "row_id": row["row_id"], "first_margin": float(margin.detach()),
            "all_answer_margins": all_margins.tolist(), "gradient": gradient.tolist(),
        })
        atomic_json(args.out / "live.json", {"phase": "gradient", "completed": index, "total": len(rows)})
    margins_array = np.asarray(margins, dtype=np.float64)
    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    try:
        step = minimum_norm_exponent_step(
            margins_array, jacobian_array, initial_exponents,
            target_margin=args.target_margin, trust_bound=args.exponent_bound,
        )
        qp_status = "feasible"
    except RuntimeError as error:
        step = None
        qp_status = str(error)
    candidates = []
    model.eval()
    if step is not None:
        alpha_max = maximum_feasible_line_scale(
            initial_exponents, step, trust_bound=args.exponent_bound,
        )
        requested_scales = sorted(set([
            *args.line_scale, min(alpha_max, 2.0),
        ]))
        for line_scale in requested_scales:
            delta = line_scale * step
            feasible, active_exponents = exponent_feasibility(
                initial_exponents, delta, tolerance=2e-7,
            )
            if not feasible or np.max(np.abs(delta)) > args.exponent_bound + 2e-7:
                continue
            allocation.set_delta_(delta)
            exact = []
            for row in rows:
                margin, all_margins = first_margin(model, row, grad=False)
                exact.append({
                    "row_id": row["row_id"], "first_margin": float(margin),
                    "all_answer_margins": all_margins.tolist(),
                })
            values = allocation.realized_inv_freq().detach().cpu().float().numpy()
            candidates.append({
                "line_scale": float(line_scale), "delta_interior": delta.tolist(),
                "max_abs_exponent_delta": float(np.max(np.abs(delta))),
                "first_margins": [record["first_margin"] for record in exact],
                "all_first_margins_positive": all(record["first_margin"] > 0.0 for record in exact),
                "valid_native_relative_exponent_box": True,
                "min_exponent": float(active_exponents.min()),
                "max_exponent": float(active_exponents.max()),
                "monotone_exponents": bool(np.all(np.diff(active_exponents) >= -2e-7)),
                "values_float32": values.tolist(), "traces": exact,
            })
    accepted = next((item for item in candidates if all(
        margin >= args.target_margin - 1e-7 for margin in item["first_margins"]
    )), None)
    receipt_path = None
    if accepted is not None:
        values = np.asarray(accepted.pop("values_float32"), dtype=np.float32)
        allocation.set_delta_(np.asarray(accepted["delta_interior"], dtype=np.float64))
        receipt = make_receipt(
            candidate_id=f"{parent['candidate_id']}_constrained_full_z_fwe_repair",
            model_id=parent["model_id"], role="candidate", scale=args.scale,
            geometry=geometry, values=values, gain=gain,
            construction={
                **allocation.receipt(),
                "method": "minimum_norm_exponent_box_first_token_margin_repair",
                "target_margin": args.target_margin,
                "trust_bound": args.exponent_bound,
                "linearized_step": step.tolist(),
                "accepted_line_scale": accepted["line_scale"],
                "fit_row_ids": [row["row_id"] for row in rows],
                "constraints": ["fixed m[0]=0", "fixed m[-1]=1", "0<=m<=1", "monotone m"],
            },
            source=str(args.parent_table), parent_candidate_id=parent["candidate_id"],
            changed_variables=["full_native_relative_exponent_allocation"],
        )
        receipt_path = args.out / "candidate_table.json"
        atomic_json(receipt_path, receipt)
    result = {
        "status": "CONSTRAINED_FULL_Z_FWE_REPAIR_V1",
        "parent_table": str(args.parent_table), "parent_candidate_id": parent["candidate_id"],
        "gain_fixed": gain, "fit_rows": [row["row_id"] for row in rows],
        "initial_first_margins": margins, "margin_jacobian": jacobian_array.tolist(),
        "qp_status": qp_status,
        "maximum_feasible_line_scale": (
            maximum_feasible_line_scale(initial_exponents, step, trust_bound=args.exponent_bound)
            if step is not None else None
        ),
        "minimum_norm_linearized_step": step.tolist() if step is not None else None,
        "target_margin": args.target_margin, "trust_bound": args.exponent_bound,
        "line_search": candidates, "accepted": accepted,
        "candidate_table": str(receipt_path) if receipt_path else None,
        "scope": "CAL fit first-token repair only; select FWE and 32K free generation required",
    }
    atomic_json(args.out / "proposal.json", result)
    atomic_json(args.out / "status.json", {"status": "COMPLETE", "candidate": receipt_path is not None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--fit-rows", type=int, default=3)
    parser.add_argument("--target-margin", type=float, default=0.125)
    parser.add_argument("--exponent-bound", type=float, default=0.03)
    parser.add_argument("--line-scale", type=float, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.line_scale = args.line_scale or [0.25, 0.5, 0.75, 1.0]
    if (
        args.out.exists() or args.fit_rows < 1 or not math.isfinite(args.target_margin)
        or args.exponent_bound <= 0.0
    ):
        raise ValueError("invalid or pre-existing constrained full-z repair run")
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "model": str(args.model), "data": str(args.data),
            "parent_table": str(args.parent_table), "target_margin": args.target_margin,
            "exponent_bound": args.exponent_bound, "line_scale": args.line_scale,
        }, indent=2))
        return
    args.out.mkdir(parents=True)
    result = execute(args)
    print(json.dumps({
        "status": result["status"], "initial_first_margins": result["initial_first_margins"],
        "qp_status": result["qp_status"], "accepted": result["accepted"],
        "candidate_table": result["candidate_table"],
    }, indent=2))


if __name__ == "__main__":
    main()
