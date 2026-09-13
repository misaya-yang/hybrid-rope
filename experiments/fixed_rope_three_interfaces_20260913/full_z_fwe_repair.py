#!/usr/bin/env python3
"""Minimum-displacement full-z repair for Llama Native-window FWE termination."""
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


def minimum_norm_margin_step(
    margins: np.ndarray, jacobian: np.ndarray, *, target_margin: float, bound: float,
) -> np.ndarray:
    """Solve min 0.5||d||² subject to margin+Jd>=target and box bounds."""
    from scipy.optimize import minimize

    margins = np.asarray(margins, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or margins.shape != (jacobian.shape[0],) or bound <= 0.0:
        raise ValueError("invalid full-z linearized margin system")
    result = minimize(
        fun=lambda value: 0.5 * float(np.dot(value, value)),
        x0=np.zeros(jacobian.shape[1], dtype=np.float64),
        jac=lambda value: value,
        bounds=[(-bound, bound)] * jacobian.shape[1],
        constraints=[{
            "type": "ineq",
            "fun": lambda value: margins + jacobian @ value - target_margin,
            "jac": lambda value: jacobian,
        }],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"full-z minimum-norm subproblem failed: {result.message}")
    step = np.asarray(result.x, dtype=np.float64)
    if np.max(np.abs(step)) > bound + 1e-7:
        raise RuntimeError("full-z solver exceeded its box bound")
    return step


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
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(answer), return_dict=True).logits[0].float()
        gold = logits.gather(-1, labels[:, None]).squeeze(-1)
        competitor = logits.scatter(-1, labels[:, None], -torch.inf).amax(-1)
        margins = gold - competitor
    return margins[0], margins.detach().cpu().float().numpy()


def execute(args) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from .target_support_z import install_target_support_z

    validate_cuda()
    parent = read_json(args.parent_table)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    geometry = model_geometry(config.to_dict())
    if parent.get("model_geometry") != geometry or float(parent.get("scale", float("nan"))) != args.scale:
        raise ValueError("parent table geometry or scale mismatch")
    parent_values, gain = validate_table(find_table(parent), pairs=int(geometry["pairs"]))
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
        raise ValueError("full-z repair requires zero dropout")
    model.eval().requires_grad_(False)
    allocation = install_target_support_z(
        model, initial_inv_freq=parent_values, gain=gain,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads();model.train()
    margins, jacobian, traces = [], [], []
    for index, row in enumerate(rows, 1):
        margin, all_margins = first_margin(model, row, grad=True)
        gradient = torch.autograd.grad(margin, allocation.gap_delta_logits)[0].detach().cpu().double().numpy()
        margins.append(float(margin.detach()));jacobian.append(gradient)
        traces.append({
            "row_id": row["row_id"], "first_margin": float(margin.detach()),
            "all_answer_margins": all_margins.tolist(), "gradient": gradient.tolist(),
        })
        atomic_json(args.out / "live.json", {"phase": "gradient", "completed": index, "total": len(rows)})
    margins_array = np.asarray(margins, dtype=np.float64)
    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    step = minimum_norm_margin_step(
        margins_array, jacobian_array, target_margin=args.target_margin,
        bound=args.gap_logit_bound,
    )
    candidates = []
    model.eval()
    for scale in args.line_scale:
        delta = scale * step
        if np.max(np.abs(delta)) > args.gap_logit_bound + 1e-7:
            continue
        allocation.set_gap_delta_(delta)
        exact = []
        for row in rows:
            margin, all_margins = first_margin(model, row, grad=False)
            exact.append({
                "row_id": row["row_id"], "first_margin": float(margin),
                "all_answer_margins": all_margins.tolist(),
            })
        values = allocation.realized_inv_freq().detach().cpu().float().numpy()
        native = runtime_native_inv_freq(geometry).astype(np.float64)
        exponents = -np.log(values.astype(np.float64) / native) / math.log(args.scale)
        valid_native_relative = bool(exponents.min() >= -2e-6 and exponents.max() <= 1.0 + 2e-6)
        candidates.append({
            "line_scale": float(scale), "gap_delta": delta.tolist(),
            "max_abs_gap_logit_delta": float(np.max(np.abs(delta))),
            "first_margins": [record["first_margin"] for record in exact],
            "all_first_margins_positive": all(record["first_margin"] > 0.0 for record in exact),
            "valid_native_relative_exponent_box": valid_native_relative,
            "min_exponent": float(exponents.min()), "max_exponent": float(exponents.max()),
            "values_float32": values.tolist(), "traces": exact,
        })
    accepted = next(
        (item for item in candidates if item["all_first_margins_positive"] and item["valid_native_relative_exponent_box"]),
        None,
    )
    receipt_path = None
    if accepted is not None:
        values = np.asarray(accepted.pop("values_float32"), dtype=np.float32)
        allocation.set_gap_delta_(np.asarray(accepted["gap_delta"], dtype=np.float64))
        receipt = make_receipt(
            candidate_id=f"{parent['candidate_id']}_full_z_fwe_repair",
            model_id=parent["model_id"], role="candidate", scale=args.scale,
            geometry=geometry, values=values, gain=gain,
            construction={
                **allocation.receipt(),
                "method": "minimum_norm_full_z_first_token_margin_repair",
                "target_margin": args.target_margin,
                "linearized_step": step.tolist(),
                "accepted_line_scale": accepted["line_scale"],
                "fit_row_ids": [row["row_id"] for row in rows],
            },
            source=str(args.parent_table), parent_candidate_id=parent["candidate_id"],
            changed_variables=["full_z_log_frequency_gap_distribution"],
            allow_nonmonotone_exponents=True,
        )
        receipt_path = args.out / "candidate_table.json"
        atomic_json(receipt_path, receipt)
    result = {
        "status": "FULL_Z_FWE_REPAIR_PROPOSAL_V1",
        "parent_table": str(args.parent_table), "parent_candidate_id": parent["candidate_id"],
        "gain_fixed": gain, "fast_endpoint_fixed": float(parent_values[0]),
        "slow_endpoint_fixed": float(parent_values[-1]),
        "fit_rows": [row["row_id"] for row in rows],
        "initial_first_margins": margins, "margin_jacobian": jacobian_array.tolist(),
        "minimum_norm_linearized_step": step.tolist(), "target_margin": args.target_margin,
        "line_search": candidates, "accepted": accepted,
        "candidate_table": str(receipt_path) if receipt_path else None,
        "scope": "CAL first-token repair; disjoint FWE and 32K free generation required",
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
    parser.add_argument("--target-margin", type=float, default=0.25)
    parser.add_argument("--gap-logit-bound", type=float, default=2.0)
    parser.add_argument("--line-scale", type=float, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.line_scale = args.line_scale or [0.5, 1.0, 1.5, 2.0]
    if args.out.exists() or args.fit_rows < 1 or args.target_margin <= 0.0 or args.gap_logit_bound <= 0.0:
        raise ValueError("invalid or pre-existing full-z repair run")
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "model": str(args.model), "data": str(args.data),
            "parent_table": str(args.parent_table), "target_margin": args.target_margin,
            "gap_logit_bound": args.gap_logit_bound, "line_scale": args.line_scale,
        }, indent=2))
        return
    args.out.mkdir(parents=True)
    result = execute(args)
    print(json.dumps({
        "status": result["status"], "initial_first_margins": result["initial_first_margins"],
        "accepted": result["accepted"], "candidate_table": result["candidate_table"],
    }, indent=2))


if __name__ == "__main__":
    main()
