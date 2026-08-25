#!/usr/bin/env python3
"""Frozen table x Q/K-LoRA attribution for the allocation oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import TrainingBackbone
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.phase_adaptation import (
    PhaseAdaptationView,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    fused_loss_module,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope import (
    RetentionView,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import atomic_json

from .oracle import NATIVE_LENGTH, sha256_file
from .train import (
    RESULT_STATUS,
    _base_model,
    _evaluate_phase,
    _load_model,
)


ATTRIBUTION_STATUS = "OLMO2_ALLOCATION_ORACLE_ATTRIBUTION_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-view", type=Path, required=True)
    parser.add_argument("--retention-view", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def _dense_retention(
    *, model: Any, backbone: Any, loss_module: Any, view: RetentionView
) -> dict[str, Any]:
    total = 0.0
    tokens = 0
    for start in range(0, len(view.input_ids), 4):
        ids = torch.as_tensor(
            np.asarray(view.input_ids[start : start + 4]),
            device="cuda",
            dtype=torch.long,
        )
        hidden = backbone(ids[:, :-1], None)
        labels = ids[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = loss_module(
                _base_model(model).lm_head.weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )
            value = value.loss if hasattr(value, "loss") else value
        count = int(labels.numel())
        total += float(value) * count
        tokens += count
        del ids, hidden, labels, value
    return {"rows": int(len(view.input_ids)), "tokens": tokens, "mean_nll": total / tokens}


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED") != "YES":
        raise PermissionError("GPU attribution requires both authorization factors")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"attribution output already exists: {output}")

    oracle_path = args.oracle_result.resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("status") != RESULT_STATUS or oracle.get("smoke"):
        raise RuntimeError("completed full oracle result is required")
    run_dir = oracle_path.parent
    artifact_paths = {
        "adapter_model": run_dir / "adapter" / "adapter_model.safetensors",
        "allocation_state": run_dir / "allocation_state.pt",
        "allocation_table": run_dir / "allocation_table.npy",
    }
    expected = {
        "adapter_model": oracle["artifacts"]["adapter_model_sha256"],
        "allocation_state": oracle["artifacts"]["allocation_state_sha256"],
        "allocation_table": oracle["artifacts"]["allocation_table_sha256"],
    }
    if any(sha256_file(path) != expected[name] for name, path in artifact_paths.items()):
        raise RuntimeError("oracle artifact receipt drift")

    phase = PhaseAdaptationView(args.phase_view.resolve())
    retention = RetentionView.load(args.retention_view.resolve())
    rows = phase.validation_rows
    if len(rows) != 128:
        raise RuntimeError("the attribution contract requires all 128 held-out rows")

    model, allocation = _load_model(args.checkpoint.resolve())
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    load_result = set_peft_model_state_dict(
        model, load_file(artifact_paths["adapter_model"])
    )
    if load_result.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {load_result.unexpected_keys}")
    lora_parameters = [
        value
        for name, value in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    final_lora = [value.detach().clone() for value in lora_parameters]
    if len(final_lora) != 64 or not all(torch.isfinite(value).all() for value in final_lora):
        raise RuntimeError("final Q/K LoRA state drift")
    saved = torch.load(artifact_paths["allocation_state"], map_location="cpu", weights_only=True)
    learned_logits = saved["gap_delta_logits"]
    if not torch.equal(saved["native_inv_freq"], allocation.native_inv_freq.cpu()):
        raise RuntimeError("allocation Native table drift")

    model.to("cuda").eval()
    backbone = torch.compile(
        TrainingBackbone(_base_model(model).model),
        fullgraph=True,
        dynamic=False,
        mode=str(args.compile_mode),
    )
    loss_module = fused_loss_module()
    offsets = (0, NATIVE_LENGTH, 3 * NATIVE_LENGTH, 15 * NATIVE_LENGTH)
    results: dict[str, Any] = {}

    for table_name, logits in (("native", torch.zeros_like(learned_logits)), ("learned", learned_logits)):
        with torch.no_grad():
            allocation.gap_delta_logits.copy_(logits.to(allocation.gap_delta_logits.device))
        for lora_name, enabled in (("off", False), ("on", True)):
            with torch.no_grad():
                for parameter, final_value in zip(lora_parameters, final_lora, strict=True):
                    parameter.copy_(final_value if enabled else torch.zeros_like(final_value))
            key = f"{table_name}_table__lora_{lora_name}"
            results[key] = {
                "phase": {
                    str(offset): _evaluate_phase(
                        model=model,
                        backbone=backbone,
                        loss_module=loss_module,
                        view=phase,
                        rows=rows,
                        offset=offset,
                    )
                    for offset in offsets
                },
                "retention_4k_dense": _dense_retention(
                    model=model,
                    backbone=backbone,
                    loss_module=loss_module,
                    view=retention,
                ),
            }

    atomic_json(
        output,
        {
            "status": ATTRIBUTION_STATUS,
            "evaluation_code_sha256": sha256_file(Path(__file__)),
            "oracle_result": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "rows": 128,
            "phase_offsets": list(offsets),
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
