#!/usr/bin/env python3
"""Create a no-GPU READY receipt for the allocation oracle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.phase_adaptation import (
    PhaseAdaptationView,
)
from rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.train_phase_adarope import (
    RawReplayView,
    RetentionView,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import atomic_json
from scripts.lib.rope.fixed_support_z import FixedSupportZRotaryEmbedding

from .oracle import (
    NATIVE_LENGTH,
    PAIR_COUNT,
    code_hashes,
    deterministic_offsets,
    position_ids_for_offsets,
    protocol,
    sha256_file,
    tensor_sha256,
)
from .train import READY_STATUS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--phase-view", type=Path, required=True)
    parser.add_argument("--raw-replay", type=Path, required=True)
    parser.add_argument("--retention-view", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ready = args.ready_receipt.resolve()
    output = args.output.resolve()
    if ready.exists() or output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError("READY and output paths must be new")
    if int(args.steps) <= 0 or int(args.validation_rows) not in range(1, 129):
        raise ValueError("steps and validation rows are outside registered bounds")

    checkpoint = args.checkpoint.resolve()
    checkpoint_receipt_path = args.checkpoint_ready_receipt.resolve()
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text(encoding="utf-8"))
    checkpoint_record = checkpoint_receipt.get("checkpoint", {})
    weight = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    if (
        checkpoint_receipt.get("status") != "OLMO2_INSTRUCT_4K_CONVERSION_READY"
        or Path(checkpoint_record.get("checkpoint_path", "")).resolve() != checkpoint
        or checkpoint_record.get("status") != "verified"
        or not weight.is_file()
        or int(checkpoint_record.get("files", {}).get("model.safetensors", {}).get("bytes", -1)) != weight.stat().st_size
        or not config_path.is_file()
    ):
        raise RuntimeError("checkpoint identity receipt drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": NATIVE_LENGTH,
        "rope_theta": 500_000,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError("checkpoint configuration drift")

    phase = PhaseAdaptationView(args.phase_view.resolve())
    raw = RawReplayView.load(args.raw_replay.resolve())
    retention = RetentionView.load(args.retention_view.resolve())
    if len(phase.validation_rows) < int(args.validation_rows):
        raise RuntimeError("insufficient held-out phase validation rows")

    native = 1.0 / (
        float(config["rope_theta"])
        ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0)
    )
    allocation = FixedSupportZRotaryEmbedding(native)
    if allocation.pair_count != PAIR_COUNT or not torch.equal(allocation.realized_inv_freq(), native):
        raise RuntimeError("allocation does not initialize exactly at Native")
    with torch.no_grad():
        allocation.gap_delta_logits.copy_(torch.linspace(-1.0, 1.0, PAIR_COUNT - 1))
    allocation.project_()
    active = allocation.realized_inv_freq()
    if not torch.equal(active[[0, -1]], native[[0, -1]]) or not torch.all(active[:-1] > active[1:]):
        raise RuntimeError("allocation endpoint/ordering invariant failed")

    offsets = deterministic_offsets(seed=int(args.seed), step=1, accumulation=0)
    positions = position_ids_for_offsets(
        query_starts=np.asarray(phase.query_starts[phase.training_rows[:4]]),
        offsets=offsets,
    )
    if positions.shape != (4, NATIVE_LENGTH - 1) or np.any(np.diff(positions, axis=1) <= 0):
        raise RuntimeError("phase position preflight failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 2_000_000_000:
        raise RuntimeError("less than 2GB free for the allocation oracle")
    run_protocol = protocol(steps=int(args.steps), seed=int(args.seed), smoke=bool(args.smoke))
    command = [
        sys.executable,
        "-m",
        "rebuttal.rebuttal_0723.experiments.olmo2_allocation_oracle_5090.train",
        "--checkpoint", str(checkpoint),
        "--checkpoint-ready-receipt", str(checkpoint_receipt_path),
        "--phase-view", str(phase.root),
        "--raw-replay", str(args.raw_replay.resolve()),
        "--retention-view", str(args.retention_view.resolve()),
        "--ready-receipt", str(ready),
        "--output", str(output),
        "--steps", str(args.steps),
        "--validation-rows", str(args.validation_rows),
        "--compile-mode", str(args.compile_mode),
        "--seed", str(args.seed),
        "--authorize",
    ]
    if args.smoke:
        command.append("--smoke")
    payload = {
        "status": READY_STATUS,
        "authorization_boundary": "No CUDA action occurred; GPU execution still requires both explicit authorization factors.",
        "protocol": run_protocol,
        "code_sha256": code_hashes(),
        "output": str(output),
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "weight_bytes": int(weight.stat().st_size),
                "owner_sha256": checkpoint_record["files"]["model.safetensors"]["sha256"],
                "config_sha256": sha256_file(config_path),
            },
            "checkpoint_ready_receipt": {
                "path": str(checkpoint_receipt_path),
                "sha256": sha256_file(checkpoint_receipt_path),
            },
            "phase_view": {
                "path": str(phase.root),
                "manifest_sha256": sha256_file(phase.root / "manifest.json"),
                "training_rows": int(len(phase.training_rows)),
                "validation_rows": int(len(phase.validation_rows)),
            },
            "raw_replay": {
                "path": str(args.raw_replay.resolve()),
                "manifest_sha256": sha256_file(args.raw_replay.resolve() / "manifest.json"),
                "rows": int(len(raw.input_ids)),
            },
            "retention_view": {
                "path": str(args.retention_view.resolve()),
                "manifest_sha256": sha256_file(args.retention_view.resolve() / "manifest.json"),
                "rows": int(len(retention.input_ids)),
            },
        },
        "numerical_preflight": {
            "native_table_float32_sha256": tensor_sha256(native),
            "test_table_float32_sha256": tensor_sha256(active),
            "fixed_endpoints": True,
            "strictly_decreasing": True,
            "offsets": offsets.tolist(),
            "maximum_position_id": int(positions.max()),
        },
        "gpu_command": command,
    }
    atomic_json(ready, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

