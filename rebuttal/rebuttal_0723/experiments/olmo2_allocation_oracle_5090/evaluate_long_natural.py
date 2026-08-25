#!/usr/bin/env python3
"""Evaluate the frozen oracle four-grid on physical 8K/16K natural text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import TrainingBackbone
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    fused_loss_module,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
)

from .oracle import sha256_file
from .train import RESULT_STATUS, _base_model, _load_model


STATUS = "OLMO2_ALLOCATION_ORACLE_LONG_NATURAL_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--view", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def _load_view(
    path: Path, *, checkpoint: Path
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = path.resolve()
    receipt_path = path.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if (
        receipt.get("status") != "PURE_TEXT_ROWS_READY"
        or receipt.get("training_or_parameter_updates") is not False
        or Path(str(receipt.get("model", ""))).resolve() != checkpoint
        or Path(str(receipt.get("tensor", ""))).resolve() != path
        or sha256_file(path) != receipt.get("tensor_sha256")
        or list(rows.shape) != receipt.get("shape")
        or rows.dtype != torch.int64
    ):
        raise RuntimeError(f"natural-text view receipt drift: {path}")
    return rows, receipt


@torch.inference_mode()
def _evaluate_view(
    *, model: Any, backbone: Any, loss_module: Any, rows: torch.Tensor
) -> dict[str, Any]:
    per_row = []
    for index, host_ids in enumerate(rows):
        ids = host_ids[None, :].to("cuda", non_blocking=True)
        hidden = backbone(ids[:, :-1], None)
        labels = ids[:, 1:]
        metrics = {}
        for name, selected_hidden, selected_labels in (
            ("full", hidden, labels),
            ("tail512", hidden[:, -512:], labels[:, -512:]),
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = loss_module(
                    _base_model(model).lm_head.weight,
                    selected_hidden.reshape(-1, selected_hidden.shape[-1]),
                    selected_labels.reshape(-1),
                )
                value = value.loss if hasattr(value, "loss") else value
            metrics[f"{name}_nll"] = float(value)
        per_row.append({"row": index, **metrics})
        del ids, hidden, labels
    return {
        "rows": len(per_row),
        "mean_full_nll": float(np.mean([row["full_nll"] for row in per_row])),
        "mean_tail512_nll": float(
            np.mean([row["tail512_nll"] for row in per_row])
        ),
        "per_row": per_row,
    }


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED") != "YES":
        raise PermissionError("long-natural evaluation requires both authorization factors")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    oracle_path = args.oracle_result.resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("status") != RESULT_STATUS or oracle.get("smoke"):
        raise RuntimeError("completed full oracle result is required")
    run_dir = oracle_path.parent
    adapter_path = run_dir / "adapter" / "adapter_model.safetensors"
    state_path = run_dir / "allocation_state.pt"
    if (
        sha256_file(adapter_path) != oracle["artifacts"]["adapter_model_sha256"]
        or sha256_file(state_path) != oracle["artifacts"]["allocation_state_sha256"]
    ):
        raise RuntimeError("oracle artifact receipt drift")

    checkpoint = args.checkpoint.resolve()
    views = {
        path.resolve(): _load_view(path, checkpoint=checkpoint)
        for path in args.view
    }
    if sorted(rows.shape[1] for rows, _ in views.values()) != [8192, 16384]:
        raise RuntimeError("exactly one physical 8K and one physical 16K view are required")

    environment = configure_cuda()
    model, allocation = _load_model(checkpoint)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    load_result = set_peft_model_state_dict(model, load_file(adapter_path))
    if load_result.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {load_result.unexpected_keys}")
    lora_parameters = [
        value
        for name, value in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    final_lora = [value.detach().clone() for value in lora_parameters]
    saved = torch.load(state_path, map_location="cpu", weights_only=True)
    learned_logits = saved["gap_delta_logits"]
    if len(final_lora) != 64 or not torch.equal(
        saved["native_inv_freq"], allocation.native_inv_freq.cpu()
    ):
        raise RuntimeError("oracle parameter identity drift")

    model.to("cuda").eval()
    backbone = torch.compile(
        TrainingBackbone(_base_model(model).model),
        fullgraph=True,
        dynamic=False,
        mode=str(args.compile_mode),
    )
    loss_module = fused_loss_module()
    results: dict[str, Any] = {}
    for table_name, logits in (("native", torch.zeros_like(learned_logits)), ("learned", learned_logits)):
        with torch.no_grad():
            allocation.gap_delta_logits.copy_(logits.to(allocation.gap_delta_logits.device))
        for lora_name, enabled in (("off", False), ("on", True)):
            with torch.no_grad():
                for parameter, final_value in zip(lora_parameters, final_lora, strict=True):
                    parameter.copy_(final_value if enabled else torch.zeros_like(final_value))
            results[f"{table_name}_table__lora_{lora_name}"] = {
                str(rows.shape[1]): _evaluate_view(
                    model=model,
                    backbone=backbone,
                    loss_module=loss_module,
                    rows=rows,
                )
                for rows, _ in views.values()
            }

    atomic_json(
        output,
        {
            "status": STATUS,
            "evaluation_code_sha256": sha256_file(Path(__file__)),
            "environment": environment,
            "oracle_result": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "views": {
                str(rows.shape[1]): {
                    "path": str(path),
                    "tensor_sha256": receipt["tensor_sha256"],
                    "rows": int(rows.shape[0]),
                }
                for path, (rows, receipt) in views.items()
            },
            "metric": "teacher-forced causal NLL on physical contiguous FineWeb-Edu text",
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
