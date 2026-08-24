#!/usr/bin/env python3
"""Calibrate one fixed-support RoPE table on 1x/2x endpoint-tail NLL."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_ruler_flash_attention,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    configure_cuda,
    seed_everything,
    sha256_file,
)
from scripts.lib.rope.fixed_support_z import (
    float32_sha256,
    install_fixed_support_z,
)


STATUS = "OLMO_FIXED_SUPPORT_DIRECT_Z_COMPLETE_V1"
SMOKE_STATUS = "OLMO_FIXED_SUPPORT_DIRECT_Z_SMOKE_COMPLETE_V1"
DESIGN_ROWS = (0, 1)
HELDOUT_ROWS = (2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--natural-2x-tensor", type=Path, required=True)
    parser.add_argument("--natural-2x-receipt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--id-nll-epsilon", type=float, default=0.05)
    parser.add_argument("--retention-penalty", type=float, default=100.0)
    parser.add_argument("--maximum-heldout-2x-regression", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_natural_tensor(
    tensor_path: Path,
    receipt_path: Path,
    *,
    checkpoint: Path,
    expected_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path = tensor_path.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    tensor_sha = sha256_file(tensor_path)
    if (
        receipt.get("status") != "PURE_TEXT_ROWS_READY"
        or receipt.get("training_or_parameter_updates") is not False
        or int(receipt.get("length", -1)) != int(expected_length)
        or Path(str(receipt.get("model", ""))).resolve() != checkpoint
        or Path(str(receipt.get("tensor", ""))).resolve() != tensor_path
        or str(receipt.get("tensor_sha256")) != tensor_sha
        or [int(value) for value in receipt.get("shape", [])]
        != [4, int(expected_length)]
    ):
        raise RuntimeError("2x natural-text receipt identity drift")
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.long
        or tuple(tensor.shape) != (4, int(expected_length))
    ):
        raise RuntimeError("2x natural-text tensor shape/dtype drift")
    return tensor.contiguous(), {
        "tensor_path": str(tensor_path),
        "tensor_sha256": tensor_sha,
        "tensor_bytes": int(tensor_path.stat().st_size),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "documents": list(receipt.get("documents", [])),
        "design_rows": list(DESIGN_ROWS),
        "heldout_rows": list(HELDOUT_ROWS),
    }


def _paired_tail_nll(
    model: Any,
    input_ids: torch.Tensor,
    *,
    native_length: int,
    tail_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model.model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    length = int(input_ids.shape[1])
    target_1x = torch.arange(
        native_length - tail_tokens,
        native_length,
        device=input_ids.device,
    )
    target_2x = torch.arange(
        length - tail_tokens,
        length,
        device=input_ids.device,
    )

    def loss_at(target: torch.Tensor) -> torch.Tensor:
        selected_hidden = hidden[:, target - 1, :]
        logits = model.lm_head(selected_hidden).float()
        labels = input_ids[:, target]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )

    return loss_at(target_1x), loss_at(target_2x)


def _evaluate_rows(
    model: Any,
    tokens: torch.Tensor,
    rows: tuple[int, ...],
    *,
    native_length: int,
    tail_tokens: int,
) -> list[dict[str, float]]:
    result = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            input_ids = tokens[int(row)][None, :].to("cuda")
            nll_1x, nll_2x = _paired_tail_nll(
                model,
                input_ids,
                native_length=native_length,
                tail_tokens=tail_tokens,
            )
            result.append({
                "row": int(row),
                "nll_1x": float(nll_1x),
                "nll_2x": float(nll_2x),
            })
            del input_ids
    return result


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("DIRECT_Z_GPU_AUTHORIZED") != "YES":
        raise PermissionError(
            "direct-z GPU run requires --authorize and DIRECT_Z_GPU_AUTHORIZED=YES"
        )
    if int(args.steps) < 1 or (args.smoke and int(args.steps) != 1):
        raise ValueError("steps must be positive and smoke requires exactly one step")
    if not 8 <= int(args.tail_tokens) <= 256:
        raise ValueError("tail-tokens must lie in [8,256]")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (
            args.learning_rate,
            args.id_nll_epsilon,
            args.retention_penalty,
            args.maximum_heldout_2x_regression,
        )
    ):
        raise ValueError("optimization constants must be finite and positive")

    output_table = args.output_table.expanduser().resolve()
    output_result = args.output_result.expanduser().resolve()
    progress_path = args.progress.expanduser().resolve()
    if output_table.exists() or output_result.exists() or progress_path.exists():
        raise FileExistsError("direct-z outputs must use new paths")

    checkpoint = args.checkpoint.expanduser().resolve()
    ready = args.checkpoint_ready_receipt.expanduser().resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, ready)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    native_length = int(config["max_position_embeddings"])
    target_length = 2 * native_length
    tokens, data_receipt = _load_natural_tensor(
        args.natural_2x_tensor,
        args.natural_2x_receipt,
        checkpoint=checkpoint,
        expected_length=target_length,
    )

    seed_everything(int(args.seed))
    environment = configure_cuda()
    model = load_model(checkpoint)
    if str(model.config.model_type) != "olmo2":
        raise RuntimeError("current direct-z pilot is pinned to OLMo-2")
    model.config.max_position_embeddings = target_length
    configure_ruler_flash_attention(model)
    model.eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    native_design = _evaluate_rows(
        model,
        tokens,
        DESIGN_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    native_heldout = _evaluate_rows(
        model,
        tokens,
        HELDOUT_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    native_by_row = {int(row["row"]): row for row in native_design}

    direct_z, install_receipt = install_fixed_support_z(model)
    initial_design = _evaluate_rows(
        model,
        tokens,
        DESIGN_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    initial_parity = max(
        abs(float(observed[key]) - float(native_by_row[int(observed["row"])][key]))
        for observed in initial_design
        for key in ("nll_1x", "nll_2x")
    )
    if initial_parity > 1e-4:
        raise RuntimeError(f"exact Native initialization parity drift: {initial_parity}")

    optimizer = torch.optim.AdamW(
        [direct_z.gap_delta_logits],
        lr=float(args.learning_rate),
        weight_decay=0.0,
        fused=True,
    )
    best_state = direct_z.gap_delta_logits.detach().cpu().clone()
    best_design_2x = _mean(native_design, "nll_2x")
    best_step = 0

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_rows: list[dict[str, float]] = []
        for row in DESIGN_ROWS:
            input_ids = tokens[int(row)][None, :].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll_1x, nll_2x = _paired_tail_nll(
                    model,
                    input_ids,
                    native_length=native_length,
                    tail_tokens=int(args.tail_tokens),
                )
                native_1x = float(native_by_row[int(row)]["nll_1x"])
                violation = torch.relu(
                    nll_1x - native_1x - float(args.id_nll_epsilon)
                )
                objective = (
                    nll_2x
                    + float(args.retention_penalty) * torch.square(violation)
                )
            (objective / float(len(DESIGN_ROWS))).backward()
            step_rows.append({
                "row": int(row),
                "nll_1x": float(nll_1x.detach()),
                "nll_2x": float(nll_2x.detach()),
                "violation": float(violation.detach()),
                "objective": float(objective.detach()),
            })
            del input_ids

        gradient = direct_z.gap_delta_logits.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("direct-z gradient is absent or non-finite")
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach()))
        torch.nn.utils.clip_grad_norm_([direct_z.gap_delta_logits], 1.0)
        current_1x = _mean(step_rows, "nll_1x")
        current_2x = _mean(step_rows, "nll_2x")
        design_1x_feasible = all(
            float(row["nll_1x"])
            <= float(native_by_row[int(row["row"])]["nll_1x"])
            + float(args.id_nll_epsilon)
            for row in step_rows
        )
        if (
            design_1x_feasible
            and current_2x < best_design_2x
        ):
            best_design_2x = current_2x
            best_state = direct_z.gap_delta_logits.detach().cpu().clone()
            best_step = step - 1
        optimizer.step()
        direct_z.project_()
        _append_jsonl(progress_path, {
            "step": step,
            "rows": step_rows,
            "mean_nll_1x": current_1x,
            "mean_nll_2x": current_2x,
            "gradient_norm": gradient_norm,
            "maximum_gap_logit_delta": float(
                direct_z.gap_delta_logits.detach().abs().max()
            ),
        })

    final_design = _evaluate_rows(
        model,
        tokens,
        DESIGN_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    final_1x_feasible = all(
        float(row["nll_1x"])
        <= float(native_by_row[int(row["row"])]["nll_1x"])
        + float(args.id_nll_epsilon)
        for row in final_design
    )
    if final_1x_feasible and _mean(final_design, "nll_2x") < best_design_2x:
        best_state = direct_z.gap_delta_logits.detach().cpu().clone()
        best_design_2x = _mean(final_design, "nll_2x")
        best_step = int(args.steps)
    direct_z.set_gap_delta_(best_state)

    candidate_design = _evaluate_rows(
        model,
        tokens,
        DESIGN_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    candidate_heldout = _evaluate_rows(
        model,
        tokens,
        HELDOUT_ROWS,
        native_length=native_length,
        tail_tokens=int(args.tail_tokens),
    )
    native_heldout_by_row = {int(row["row"]): row for row in native_heldout}
    heldout_rows = []
    for row in candidate_heldout:
        native_row = native_heldout_by_row[int(row["row"])]
        heldout_rows.append({
            **row,
            "delta_nll_1x": float(row["nll_1x"] - native_row["nll_1x"]),
            "delta_nll_2x": float(row["nll_2x"] - native_row["nll_2x"]),
        })

    table = np.ascontiguousarray(
        direct_z.realized_inv_freq().detach().cpu().numpy(),
        dtype="<f4",
    )
    method_receipt = direct_z.receipt()
    heldout_1x_mean = _mean(heldout_rows, "delta_nll_1x")
    heldout_1x_max = max(float(row["delta_nll_1x"]) for row in heldout_rows)
    heldout_2x_mean = _mean(heldout_rows, "delta_nll_2x")
    heldout_2x_max = max(float(row["delta_nll_2x"]) for row in heldout_rows)
    scientific_gates = {
        "candidate_differs_from_native": bool(method_receipt["candidate_differs_from_native"]),
        "heldout_1x_no_harm": heldout_1x_max <= float(args.id_nll_epsilon),
        "heldout_2x_mean_improves": heldout_2x_mean < 0.0,
        "heldout_2x_no_large_row_regression": (
            heldout_2x_max <= float(args.maximum_heldout_2x_regression)
        ),
    }
    scientific_passed = bool(all(scientific_gates.values()))
    status = SMOKE_STATUS if args.smoke else STATUS
    passed = True if args.smoke else scientific_passed
    result = {
        "status": status,
        "passed": passed,
        "scientific_passed": scientific_passed,
        "scientific_question": (
            "Can one endpoint-fixed table improve frozen-checkpoint 2x natural NLL "
            "while satisfying a 1x final-tail NLL no-harm constraint?"
        ),
        "claim_boundary": (
            "This is zero model-weight update but calibrated table optimization; "
            "it is not zero-search and collision is not the selection objective."
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "composite_sha256": checkpoint_digest,
            "ready_receipt_sha256": sha256_file(ready),
            "config_sha256": sha256_file(checkpoint / "config.json"),
        },
        "data": data_receipt,
        "protocol": {
            "native_length": native_length,
            "target_ratio": 2,
            "target_length": target_length,
            "same_table_at_1x_and_2x": True,
            "sampled_support": "exact Native endpoints and log span",
            "objective": "2x tail NLL plus per-row 1x tail-NLL no-harm penalty",
            "allocation_prior_used_for_selection": False,
            "collision_used_for_selection": False,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "tail_tokens": int(args.tail_tokens),
            "nll_scope": "teacher-forced final tail only at each 1x/2x endpoint",
            "id_nll_epsilon": float(args.id_nll_epsilon),
            "retention_penalty": float(args.retention_penalty),
            "maximum_heldout_2x_regression": float(
                args.maximum_heldout_2x_regression
            ),
            "seed": int(args.seed),
            "model_weight_updates": 0,
            "table_optimizer_steps": int(args.steps),
        },
        "initialization": {
            "receipt": install_receipt,
            "maximum_native_nll_parity_delta": initial_parity,
        },
        "selection": {
            "best_step": best_step,
            "native_design": native_design,
            "candidate_design": candidate_design,
            "native_heldout": native_heldout,
            "candidate_heldout": candidate_heldout,
            "heldout_deltas": heldout_rows,
            "heldout_delta_nll_1x_mean": heldout_1x_mean,
            "heldout_delta_nll_1x_max": heldout_1x_max,
            "heldout_delta_nll_2x_mean": heldout_2x_mean,
            "heldout_delta_nll_2x_max": heldout_2x_max,
            "gates": scientific_gates,
        },
        "table": {
            **method_receipt,
            "path": str(output_table),
            "active_sha256_float32": float32_sha256(table),
        },
        "runtime": {
            "environment": environment,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "source_sha256": {
            "optimizer": sha256_file(Path(__file__).resolve()),
            "parameterization": sha256_file(
                Path(__file__).resolve().parents[1] / "lib/rope/fixed_support_z.py"
            ),
        },
        "next_gate": (
            "Run held-out PG-19/RULER only if passed=true and this is not a smoke run."
        ),
    }
    _atomic_npy(output_table, table)
    result["table"]["file_sha256"] = sha256_file(output_table)
    _atomic_json(output_result, result)
    print(json.dumps({
        "status": status,
        "passed": passed,
        "scientific_passed": scientific_passed,
        "best_step": best_step,
        "heldout_delta_nll_1x_mean": heldout_1x_mean,
        "heldout_delta_nll_2x_mean": heldout_2x_mean,
        "gates": scientific_gates,
        "table_sha256_float32": method_receipt["active_sha256_float32"],
        "output": str(output_result),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
