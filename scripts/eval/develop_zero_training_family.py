#!/usr/bin/env python3
"""Development-stage construction for families F2/F3/F4 of the tournament.

GPU, frozen-checkpoint, zero-weight-update development on the ``D`` split only
(``ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830`` §4).  F1 needs no gradient
machinery: it is a frozen morph grid scored by the evaluator and reduced by the
selection rule.  This runner implements the three families that calibrate on
development outcomes:

*   ``F2 PC-RETENTION-PROJECT`` — measure the Native-prefix gradient and an
    empirical Gauss--Newton approximation with respect to the five hat
    coefficients, project the F1 phase-chord direction into the registered
    Native-loss trust region, and keep the largest projected step that passes
    the Native-prefix guard.  It reads only Native-prefix behaviour during
    construction.
*   ``F3 Z5-BEHAVIOUR`` — optimise the five knot degrees of freedom to minimise
    far-tail NLL subject to hard Native-prefix and long-dense guards, from the
    predeclared initialisations.
*   ``F4 SR-Z5`` — F3 repeated across the frozen support-factor grid.

Every family emits at most one frozen float32 representative table plus a
receipt.  The representative is chosen on ``D`` before any selection-split row is
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eval_zero_training_tournament import (  # noqa: E402
    LOGIT_CHUNK,
    _require_gpu_authorization,
    configure_flash_only,
    load_model,
)
from scripts.lib.rope.hat_projection import (  # noqa: E402
    install_hat_basis,
    table_from_coefficients,
)
from scripts.lib.rope.knot_allocation import (  # noqa: E402
    install_z5_knot,
)

BIN_WIDTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("F2", "F3", "F4"), required=True)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True, help="development split D rows.jsonl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--batch-documents", type=int, default=8)
    parser.add_argument("--initialize-from", type=Path, default=None,
                        help="F1-winner table npy to seed the F3/F4 'f1_winner' init")
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_losses(model: Any, input_ids: Any, native_length: int) -> dict[str, Any]:
    """Differentiable mean NLL for the three guard/objective endpoints.

    Runs one forward and reduces the cross-entropy over the required target
    ranges, keeping gradients with respect to the rotary parameters.
    """

    import torch
    import torch.nn.functional as F

    length = int(input_ids.shape[1])
    position_ids = torch.arange(length, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    hidden = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state[0]
    targets = input_ids[0, 1:]
    total = length - 1
    logits = model.lm_head(hidden)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets, reduction="none"
    )
    positions = torch.arange(1, total + 1, device=per_token.device)
    prefix_mask = positions <= native_length - 1
    far_mask = positions > total - BIN_WIDTH
    return {
        "native_prefix": per_token[prefix_mask].mean(),
        "long_dense": per_token.mean(),
        "far_tail": per_token[far_mask].mean(),
    }


def _dev_rows(path: Path, native_length: int, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["multiplier"]) == 4:
                rows.append(row)
    rows.sort(key=lambda r: int(r["source_row"]))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no development rows")
    return rows


def _native_reference(model: Any, rows: list[dict[str, Any]], native_length: int) -> dict[str, float]:
    import torch

    reference = {"native_prefix": [], "long_dense": [], "far_tail": []}
    with torch.inference_mode():
        for row in rows:
            ids = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
            losses = endpoint_losses(model, ids, native_length)
            for key in reference:
                reference[key].append(float(losses[key]))
            del ids
    return {key: float(np.mean(vals)) for key, vals in reference.items()}


def develop_f2(args, portfolio, model, rows, environment) -> dict[str, Any]:
    import torch

    f2 = portfolio["families"]["F2_PC_RETENTION_PROJECT"]
    coefficients = np.asarray(f2["coefficients"], dtype=np.float64)
    trust = float(f2["trust_region"]["native_prefix_mean_delta"])
    alpha_grid = [float(a) for a in f2["trust_region"]["line_search_alpha_grid"]]
    margin = float(portfolio["selection_rule"]["native_prefix_guard"])

    hat, install_receipt = install_hat_basis(model)
    native_reference = _native_reference(model, rows, int(args.native_length))
    native_prefix_ref = native_reference["native_prefix"]

    def mean_native_prefix(alpha: float) -> float:
        hat.set_coefficients_(torch.as_tensor(alpha * coefficients, dtype=torch.float32))
        values = []
        with torch.inference_mode():
            for row in rows:
                ids = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
                losses = endpoint_losses(model, ids, int(args.native_length))
                values.append(float(losses["native_prefix"]))
                del ids
        return float(np.mean(values))

    # Gradient + empirical Gauss--Newton at Native (alpha=0) on Native-prefix loss.
    hat.set_coefficients_(torch.zeros(coefficients.shape[0]))
    grads = []
    for row in rows:
        hat.zero_grad(set_to_none=True)
        ids = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
        losses = endpoint_losses(model, ids, int(args.native_length))
        losses["native_prefix"].backward()
        grads.append(hat.coefficients.grad.detach().cpu().numpy().astype(np.float64))
        del ids
    grads = np.stack(grads)
    mean_grad = grads.mean(axis=0)
    gn_hessian = (grads.T @ grads) / grads.shape[0]

    def trust_bound_alpha() -> float:
        c = coefficients
        gc = float(mean_grad @ c)
        chc = float(c @ gn_hessian @ c)
        best = 0.0
        for alpha in sorted(alpha_grid):
            predicted = alpha * gc + 0.5 * alpha * alpha * chc
            if predicted <= trust:
                best = alpha
        return best

    bound = trust_bound_alpha()
    chosen_alpha = 0.0
    chosen_mean = native_prefix_ref
    scan = []
    for alpha in sorted(alpha_grid, reverse=True):
        if alpha > bound:
            continue
        realized = mean_native_prefix(alpha)
        scan.append({"alpha": alpha, "mean_native_prefix_nll": realized,
                     "delta": realized - native_prefix_ref})
        if realized <= native_prefix_ref + margin:
            chosen_alpha = alpha
            chosen_mean = realized
            break

    table = table_from_coefficients(hat.original_inv_freq, chosen_alpha * coefficients)
    return {
        "family": "F2_PC_RETENTION_PROJECT",
        "construction_label": "GRADIENT_CALIBRATED_Z",
        "native_prefix_reference": native_prefix_ref,
        "trust_bound_alpha": bound,
        "alpha_scan": scan,
        "chosen_alpha": chosen_alpha,
        "chosen_mean_native_prefix_nll": chosen_mean,
        "mean_grad_norm": float(np.linalg.norm(mean_grad)),
        "install_receipt": install_receipt,
        "table": table,
        "family_stop": chosen_alpha == 0.0,
    }


def develop_z5(args, portfolio, model, rows, environment, support_factor: float, init_tables: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch

    f3 = portfolio["families"]["F3_Z5_BEHAVIOUR"]
    budget = f3["optimizer_budget"]
    margin = float(budget["guard_margin"])
    steps = int(budget["steps"])
    lr = float(budget["learning_rate"])
    batch_documents = int(args.batch_documents or budget["batch_documents"])
    native_length = int(args.native_length)

    native_reference = _native_reference(model, rows, native_length)
    best: dict[str, Any] = {"table": None, "far_tail": float("inf"), "init": None}
    init_reports = {}

    for init_name, init_table in init_tables.items():
        knot, install_receipt = install_z5_knot(model, support_factor=support_factor)
        from scripts.lib.rope.knot_allocation import init_gap_logits_from_table

        knot.set_gap_logits_(init_gap_logits_from_table(init_table, knot.original_inv_freq, support_factor=support_factor))
        optimizer = torch.optim.AdamW([knot.gap_logits], lr=lr, weight_decay=float(budget["weight_decay"]))
        batch = rows[:batch_documents]
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for row in batch:
                ids = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
                losses = endpoint_losses(model, ids, native_length)
                violation_prefix = torch.relu(losses["native_prefix"] - native_reference["native_prefix"] - margin)
                violation_dense = torch.relu(losses["long_dense"] - native_reference["long_dense"] - margin)
                objective = objective + losses["far_tail"] + 100.0 * (violation_prefix.square() + violation_dense.square())
                del ids
            (objective / float(len(batch))).backward()
            torch.nn.utils.clip_grad_norm_([knot.gap_logits], 1.0)
            optimizer.step()
            knot.project_()

        table = knot.realized_inv_freq().detach().cpu().numpy().astype("<f4")
        full = _evaluate_table_on_dev(model, table, rows, native_length)
        init_reports[init_name] = {
            "final_far_tail": full["far_tail"],
            "final_native_prefix_delta": full["native_prefix"] - native_reference["native_prefix"],
            "final_long_dense_delta": full["long_dense"] - native_reference["long_dense"],
        }
        feasible = (
            full["native_prefix"] - native_reference["native_prefix"] <= margin
            and full["long_dense"] - native_reference["long_dense"] <= margin
        )
        if feasible and full["far_tail"] < best["far_tail"]:
            best = {"table": table, "far_tail": full["far_tail"], "init": init_name}
        del optimizer

    return {
        "family": "F3_Z5_BEHAVIOUR" if support_factor == 1.0 else "F4_SR_Z5",
        "construction_label": "FORWARD_CALIBRATED",
        "support_factor": support_factor,
        "native_reference": native_reference,
        "initialisation_reports": init_reports,
        "best": {
            "init": best["init"],
            "far_tail": best["far_tail"] if best["table"] is not None else None,
        },
        "table": best["table"],
        "family_stop": best["table"] is None,
    }


def _evaluate_table_on_dev(model: Any, table: np.ndarray, rows: list[dict[str, Any]], native_length: int) -> dict[str, float]:
    import torch

    from scripts.lib.rope.inject import apply_inv_freq_inplace

    apply_inv_freq_inplace(model, torch.as_tensor(table, dtype=torch.float64))
    means = {"native_prefix": [], "long_dense": [], "far_tail": []}
    with torch.inference_mode():
        for row in rows:
            ids = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
            losses = endpoint_losses(model, ids, native_length)
            for key in means:
                means[key].append(float(losses[key]))
            del ids
    return {key: float(np.mean(vals)) for key, vals in means.items()}


def main() -> int:
    args = parse_args()
    _require_gpu_authorization(args)
    portfolio = json.loads(args.portfolio_manifest.resolve().read_text(encoding="utf-8"))
    native_length = int(args.native_length)
    rows = _dev_rows(args.rows.resolve(), native_length, int(args.limit_rows))
    environment = configure_flash_only()
    model = load_model(args.checkpoint.resolve(), native_length * 4)
    started = time.perf_counter()

    if args.family == "F2":
        report = develop_f2(args, portfolio, model, rows, environment)
    else:
        init_tables: dict[str, np.ndarray] = {}
        f3 = portfolio["families"]["F3_Z5_BEHAVIOUR"]["initialisations"]
        native_path = next(
            entry["path"]
            for entry in portfolio["families"]["F1_PC_MORPH"]["tables"]
            if entry["is_bitwise_native"]
        )
        init_tables["native"] = np.load(native_path, allow_pickle=False)
        init_tables["learned_teacher"] = np.load(
            f3["learned_teacher"]["source"], allow_pickle=False
        )
        init_tables["coarse_budgeted"] = np.load(
            f3["coarse_budgeted"]["source"], allow_pickle=False
        )
        if args.initialize_from is not None:
            init_tables["f1_winner"] = np.load(args.initialize_from.resolve(), allow_pickle=False)
        if args.family == "F3":
            report = develop_z5(args, portfolio, model, rows, environment, 1.0, init_tables)
        else:
            # F4 nominates the best (support, table) across the frozen grid.
            candidates = []
            for support in portfolio["families"]["F4_SR_Z5"]["support_grid"]:
                sub = develop_z5(args, portfolio, model, rows, environment, float(support), init_tables)
                if sub["table"] is not None:
                    candidates.append(sub)
            if not candidates:
                report = {"family": "F4_SR_Z5", "table": None, "family_stop": True,
                          "support_grid": portfolio["families"]["F4_SR_Z5"]["support_grid"]}
            else:
                report = min(candidates, key=lambda c: c["best"]["far_tail"])
                report["support_grid"] = portfolio["families"]["F4_SR_Z5"]["support_grid"]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": f"ZERO_TRAINING_{args.family}_DEVELOPMENT_COMPLETE",
        "family": report.get("family"),
        "construction_label": report.get("construction_label"),
        "family_stop": bool(report.get("family_stop")),
        "environment": environment,
        "runtime_seconds": time.perf_counter() - started,
        "report": {k: v for k, v in report.items() if k != "table"},
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if report.get("table") is not None:
        table = np.ascontiguousarray(report["table"], dtype="<f4")
        table_path = output / f"{report['family']}_representative.npy"
        np.save(table_path, table, allow_pickle=False)
        receipt["representative"] = {
            "path": str(table_path),
            "table_sha256_float32": float32_sha256(table),
        }
    (output / f"{args.family}_development_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": receipt["status"],
        "family_stop": receipt["family_stop"],
        "representative": receipt.get("representative", {}).get("table_sha256_float32"),
    }, sort_keys=True))
    return 0 if not receipt["family_stop"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
