#!/usr/bin/env python3
"""Create the no-GPU READY receipt for the fixed hybrid exact candidate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    MODEL_CONTRACT,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .evaluate_instruct_ruler_screen import validate_data
from .olmo2_exact_method import (
    HYBRID_FREQUENCY_SHA256,
    QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
    hybrid_evq_low12_inv_freq,
    phase_geometry_receipt,
    qk_evq_tail_output_mask,
)
from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
    ROOT_STATUS,
    SUPERVISION_CONTRACT,
)
from .train_4k_hybrid_exact import (
    ADAPTATION,
    ALPHA,
    RANK,
    READY_STATUS,
    _load_routing_views,
    protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import load_fixed_view


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
TOKENIZER_SHA256 = (
    "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
)
EXACT_SCREEN_LENGTHS = (4_096, 8_192, 16_384)
EXACT_SCREEN_ROWS_PER_LENGTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--exact-screen-data", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--screen-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=12_000_000_000,
    )
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def code_entry(path: Path) -> dict[str, Any]:
    return file_entry(path)


def validate_checkpoint(
    checkpoint: Path,
    checkpoint_ready: Path,
) -> dict[str, Any]:
    digest = ready_checkpoint_digest(checkpoint, checkpoint_ready)
    if digest != MODEL_SHA256:
        raise RuntimeError("checkpoint is not the fixed OLMo-2 1.485B model")
    weight = checkpoint / "model.safetensors"
    weight_stat = weight.stat()
    realized_weight_sha256 = sha256_file(weight)
    if realized_weight_sha256 != MODEL_SHA256:
        raise RuntimeError("OLMo checkpoint weight hash drift")
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    expected = {
        "architectures": MODEL_CONTRACT["architectures"],
        "hidden_size": MODEL_CONTRACT["hidden_size"],
        "intermediate_size": MODEL_CONTRACT["intermediate_size"],
        "num_hidden_layers": MODEL_CONTRACT["num_hidden_layers"],
        "num_attention_heads": MODEL_CONTRACT["num_attention_heads"],
        "num_key_value_heads": MODEL_CONTRACT["num_key_value_heads"],
        "head_dim": MODEL_CONTRACT["head_dim"],
        "max_position_embeddings": MODEL_CONTRACT[
            "max_position_embeddings"
        ],
        "rope_theta": MODEL_CONTRACT["rope_theta"],
        "vocab_size": MODEL_CONTRACT["vocab_size"],
        "tie_word_embeddings": MODEL_CONTRACT["tie_word_embeddings"],
        "model_type": "olmo2",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError(f"checkpoint config drift for {name}")
    tokenizer_digest = sha256_file(checkpoint / "tokenizer.json")
    if tokenizer_digest != TOKENIZER_SHA256:
        raise RuntimeError("OLMo tokenizer identity drift")
    return {
        "path": str(checkpoint),
        "digest": digest,
        "config": expected,
        "config_file": file_entry(checkpoint / "config.json"),
        "weights": {
            "path": str(weight),
            "bytes": int(weight_stat.st_size),
            "sha256": realized_weight_sha256,
            "mtime_ns": int(weight_stat.st_mtime_ns),
        },
        "tokenizer_sha256": tokenizer_digest,
        "checkpoint_ready_receipt_use": "checkpoint_identity_only",
    }


def validate_routing_data(
    checkpoint: Path,
    root: Path,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != ROOT_STATUS
        or int(manifest.get("format_version", -1)) != 2
        or manifest.get("supervision_contract") != SUPERVISION_CONTRACT
        or int(manifest.get("eos_token_id", -1))
        != OLMO2_EOS_TOKEN_ID
        or int(manifest.get("hard_maximum_training_length", -1))
        != LENGTH
        or int(manifest.get("hard_maximum_training_position_id", -1))
        != LENGTH - 1
        or manifest.get("final_eos_supervised") is not True
        or manifest.get("labels_only_cover_answer_and_final_eos")
        is not True
        or manifest.get("answer_string_tokenizer_roundtrip_exact")
        is not True
        or manifest.get("train_eval_values_disjoint") is not True
        or manifest.get("official_eval_template_used_for_training")
        is not False
    ):
        raise RuntimeError("routing collection contract drift")
    if manifest.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise RuntimeError("routing collection tokenizer drift")
    train_view, calibration_view = _load_routing_views(
        checkpoint=checkpoint,
        routing_root=root,
    )
    sets = {}
    for name, view in (
        ("train", train_view),
        ("calibration", calibration_view),
    ):
        manifest_sha = sha256_file(root / name / "manifest.json")
        if manifest_sha != manifest["sets"][name]["manifest_sha256"]:
            raise RuntimeError(f"routing {name} manifest drift")
        sets[name] = {
            "rows": int(len(view.input_ids)),
            "shape": [int(value) for value in view.input_ids.shape],
            "manifest_sha256": manifest_sha,
        }
    return {
        "path": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "status": manifest["status"],
        "format_version": int(manifest["format_version"]),
        "sets": sets,
        "final_eos_supervised": True,
        "causal_alignment": (
            "labels are stored at answer/EOS token positions and shifted "
            "right once by routing_batch before loss"
        ),
    }


def validate_natural_replay(prepared_data: Path) -> dict[str, Any]:
    root = prepared_data / "longalign_paired_L4096"
    manifest_path = root / "manifest.json"
    view = load_fixed_view(root)
    if tuple(view.input_ids.shape)[1] != LENGTH:
        raise RuntimeError("natural replay storage is not fixed 4K")
    if np.any(view.lengths[view.training_rows] != LENGTH):
        raise RuntimeError("natural replay training rows are not exactly 4K")
    for name, expected in view.manifest["files"].items():
        candidate = root / name
        if (
            candidate.stat().st_size != int(expected["bytes"])
            or sha256_file(candidate) != expected["sha256"]
        ):
            raise RuntimeError(f"natural replay hash drift: {candidate}")
    return {
        "path": str(prepared_data),
        "natural_view": str(root),
        "natural_manifest_sha256": sha256_file(manifest_path),
        "shape": [int(value) for value in view.input_ids.shape],
        "training_rows": int(len(view.training_rows)),
        "maximum_training_length": LENGTH,
    }


def validate_method_identity() -> dict[str, Any]:
    frequency = hybrid_evq_low12_inv_freq()
    if tensor_sha256(frequency) != HYBRID_FREQUENCY_SHA256:
        raise RuntimeError("hybrid frequency identity drift")
    config = type(
        "Config",
        (),
        {
            "hidden_size": MODEL_CONTRACT["hidden_size"],
            "num_attention_heads": MODEL_CONTRACT[
                "num_attention_heads"
            ],
            "num_key_value_heads": MODEL_CONTRACT[
                "num_key_value_heads"
            ],
            "head_dim": MODEL_CONTRACT["head_dim"],
        },
    )()
    mask = qk_evq_tail_output_mask(config)
    if tensor_sha256(mask) != QK_EVQ_TAIL_OUTPUT_MASK_SHA256:
        raise RuntimeError("Q/K tail mask identity drift")
    return {
        "frequency_sha256_float32": HYBRID_FREQUENCY_SHA256,
        "qk_output_mask_sha256": QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
        "qk_mask_active_outputs": int(mask.sum().item()),
        "qk_mask_total_outputs": int(mask.numel()),
        "geometry": phase_geometry_receipt(),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees CUDA")

    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    prepared_data = args.prepared_data.resolve()
    routing_data = args.routing_data.resolve()
    exact_screen_data = args.exact_screen_data.resolve()
    run_output = args.run_output.resolve()
    screen_output = args.screen_output.resolve()
    gate_output = args.gate_output.resolve()
    receipt_output = args.receipt_output.resolve()
    for path in (
        run_output,
        run_output.with_name(run_output.name + ".incomplete"),
        screen_output,
        gate_output,
        receipt_output,
    ):
        if path.exists():
            raise FileExistsError(path)
    output_parent = run_output.parent
    if not output_parent.is_dir():
        raise FileNotFoundError(output_parent)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient output storage")

    checkpoint_entry = validate_checkpoint(
        checkpoint,
        checkpoint_ready,
    )
    routing_entry = validate_routing_data(checkpoint, routing_data)
    natural_entry = validate_natural_replay(prepared_data)
    exact_data_receipt, exact_rows = validate_data(
        exact_screen_data,
        checkpoint,
        "niah_single_1",
        EXACT_SCREEN_LENGTHS,
        EXACT_SCREEN_ROWS_PER_LENGTH,
    )
    if len(exact_rows) != (
        len(EXACT_SCREEN_LENGTHS) * EXACT_SCREEN_ROWS_PER_LENGTH
    ):
        raise RuntimeError("minimal exact-screen row count drift")

    here = Path(__file__).resolve()
    trainer = here.with_name("train_4k_hybrid_exact.py")
    shared_trainer = here.with_name(
        "train_4k_counterfactual_routing.py"
    )
    method = here.with_name("olmo2_exact_method.py")
    evaluator = here.with_name("evaluate_instruct_ruler_screen.py")
    gate = here.with_name("gate_olmo2_hybrid_exact_screen.py")
    conversion = here.parents[1] / "olmo2_lora_conversion.py"
    receipt = {
        "status": READY_STATUS,
        "scope": (
            "OLMo-2 1.485B only; physical training <=4K; Hybrid-EVQ plus "
            "Q/K-tail LoRA; minimal full-string exact+EOS gate before any "
            "expanded evaluation"
        ),
        "scientific_question": (
            "Can a static low-frequency EVQ subspace and matching Q/K-only "
            "LoRA learn strict 8K/16K generation without the 4K collapse "
            "observed after full-EVQ QKVO adaptation?"
        ),
        "new_capability_hypothesis": (
            "Low12 is the largest tail whose maximum Native-to-EVQ phase "
            "shift remains below 0.5 rad at 4K while becoming nontrivial at "
            "8K/16K; restricting LoRA outputs to that same subspace removes "
            "V/O and high-frequency interference."
        ),
        "existing_evidence": (
            "The prior full-EVQ QKVO query-gap arm improved first-number "
            "retrieval but did not establish whole-string exactness or broad "
            "4K retention. Full-EVQ natural/KL variants did not solve broad "
            "transfer."
        ),
        "smallest_missing_evidence": (
            "Whole decoded continuation exactness with terminal EOS at "
            "4K/8K/16K from a non-forgotten base-start adapter."
        ),
        "stop_condition": (
            "Run only the 8-row-per-length exact screen after training. If "
            "the fail-closed gate stops, do not expand evaluation or claim "
            "success; preserve raw outputs and diagnose the failed component "
            "before versioning another method."
        ),
        "protocol": protocol(),
        "method_identity": validate_method_identity(),
        "inputs": {
            "checkpoint": checkpoint_entry,
            "checkpoint_ready_receipt": file_entry(checkpoint_ready),
            "prepared_data": natural_entry,
            "routing_data": routing_entry,
            "exact_screen_data": {
                "path": str(exact_screen_data),
                **exact_data_receipt,
                "rows": len(exact_rows),
            },
        },
        "code": {
            "preflight": code_entry(here),
            "trainer": code_entry(trainer),
            "shared_trainer": code_entry(shared_trainer),
            "method": code_entry(method),
            "conversion": code_entry(conversion),
            "exact_evaluator": code_entry(evaluator),
            "exact_gate": code_entry(gate),
        },
        "run_output": str(run_output),
        "screen_output": str(screen_output),
        "gate_output": str(gate_output),
        "commands": {
            "train": [
                "python3",
                "-m",
                (
                    "rebuttal.rebuttal_0723.experiments."
                    "olmo2_lora_maturity.train_4k_hybrid_exact"
                ),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-ready-receipt",
                str(checkpoint_ready),
                "--prepared-data",
                str(prepared_data),
                "--routing-data",
                str(routing_data),
                "--ready-receipt",
                str(receipt_output),
                "--output",
                str(run_output),
            ],
            "minimal_exact_screen": [
                "python3",
                "-m",
                (
                    "rebuttal.rebuttal_0723.experiments."
                    "olmo2_lora_maturity.evaluate_instruct_ruler_screen"
                ),
                "--checkpoint",
                str(checkpoint),
                "--ready-receipt",
                str(checkpoint_ready),
                "--data-root",
                str(exact_screen_data),
                "--output",
                str(screen_output),
                "--frequency",
                "hybrid_evq_low12",
                "--adapter",
                str(run_output / "adapter.pt"),
                "--adaptation",
                ADAPTATION,
                "--rank",
                str(RANK),
                "--alpha",
                str(ALPHA),
                "--task",
                "niah_single_1",
                "--lengths",
                *[str(value) for value in EXACT_SCREEN_LENGTHS],
                "--limit-per-length",
                str(EXACT_SCREEN_ROWS_PER_LENGTH),
            ],
            "gate": [
                "python3",
                "-m",
                (
                    "rebuttal.rebuttal_0723.experiments."
                    "olmo2_lora_maturity.gate_olmo2_hybrid_exact_screen"
                ),
                "--training-result",
                str(run_output / "results.json"),
                "--evaluation-result",
                str(screen_output / "results.json"),
                "--output",
                str(gate_output),
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "liger_kernel": importlib.metadata.version("liger-kernel"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "storage": {
            "path": str(output_parent),
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
        "claim_boundary": (
            "READY proves input/code/protocol consistency only. It does not "
            "prove training success, strict generation, or 4K retention."
        ),
    }
    atomic_json(receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "receipt": str(receipt_output),
                "receipt_sha256": sha256_file(receipt_output),
                "protocol": receipt["protocol"],
                "claim_boundary": receipt["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
