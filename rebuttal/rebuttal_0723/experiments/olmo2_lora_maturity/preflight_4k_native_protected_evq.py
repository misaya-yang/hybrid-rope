#!/usr/bin/env python3
"""Create the no-GPU READY receipt for Native-protected EVQ restoration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .evq_attention_restoration import (
    LINEARARD_COMMIT,
    LINEARARD_KERNEL_SHA256,
)
from .native_protected_evq import (
    DIAGNOSTIC_STATUS,
    PREPARED_STATUS,
    native_protected_evq_inv_freq,
    qk_unprotected_output_mask,
)
from .train_4k_native_protected_evq import (
    load_selection,
    protocol_from_args,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import load_fixed_view


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("protected", "full-evq-control"),
        default="protected",
    )
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--linearard-root", type=Path, required=True)
    parser.add_argument("--gpu-ready-receipt", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1024.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.9)
    parser.add_argument("--maximum-gradient-norm", type=float, default=5.0)
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_805)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=12_000_000_000
    )
    return parser.parse_args()


def file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_linearard(root: Path) -> dict[str, Any]:
    root = root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINEARARD_COMMIT:
        raise RuntimeError("LinearARD commit drift")
    files = {}
    for name, expected_sha256 in LINEARARD_KERNEL_SHA256.items():
        path = root / "kernels" / "attention_KL" / name
        entry = file_receipt(path)
        if entry["sha256"] != expected_sha256:
            raise RuntimeError(f"LinearARD kernel drift for {name}")
        files[name] = entry
    return {
        "path": str(root),
        "commit": commit,
        "files": files,
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("protected-EVQ preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("protected-EVQ preflight unexpectedly sees CUDA")
    receipt_output = args.receipt_output.resolve()
    gpu_ready = args.gpu_ready_receipt.resolve()
    run_output = args.run_output.resolve()
    for path in (
        receipt_output,
        gpu_ready,
        run_output,
        run_output.with_name(run_output.name + ".incomplete"),
    ):
        if path.exists():
            raise FileExistsError(path)
    if not run_output.parent.is_dir():
        raise FileNotFoundError(run_output.parent)
    free_bytes = shutil.disk_usage(run_output.parent).free
    if free_bytes < int(args.minimum_free_bytes):
        raise RuntimeError("insufficient protected-EVQ output storage")

    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, checkpoint_ready
    )
    if checkpoint_digest != MODEL_SHA256:
        raise RuntimeError("checkpoint is not the fixed OLMo-2 1.485B model")
    selection_path = args.selection_receipt.resolve()
    selection_receipt, selected_protected = load_selection(selection_path)
    protected = (
        selected_protected
        if str(args.arm) == "protected"
        else ()
    )
    if (
        selection_receipt.get("status") != DIAGNOSTIC_STATUS
        or selection_receipt["checkpoint"]["composite_sha256"]
        != checkpoint_digest
    ):
        raise RuntimeError("selection checkpoint identity drift")
    diagnostic_source = Path(
        selection_receipt["source"]["diagnostic"]["path"]
    )
    method_source = Path(selection_receipt["source"]["method"]["path"])
    if (
        sha256_file(diagnostic_source)
        != selection_receipt["source"]["diagnostic"]["sha256"]
        or sha256_file(method_source)
        != selection_receipt["source"]["method"]["sha256"]
    ):
        raise RuntimeError("selection source changed after diagnosis")

    training_view = args.training_view.resolve()
    view = load_fixed_view(training_view)
    if Path(selection_receipt["training_view"]["path"]).resolve() != (
        training_view
    ):
        raise RuntimeError("selection/training view path drift")
    calibration_rows = np.asarray(
        selection_receipt["calibration"]["rows"], dtype=np.int64
    )
    if (
        np.any(np.asarray(view.split[calibration_rows]) == 0)
        or np.any(np.asarray(view.lengths[calibration_rows]) != 4_096)
    ):
        raise RuntimeError("diagnostic rows leaked into restoration training")
    training_files = {
        name: file_receipt(training_view / name)
        for name in (
            "manifest.json",
            "input_ids.npy",
            "assistant_mask.npy",
            "lengths.npy",
            "split.npy",
        )
    }
    for name, entry in training_files.items():
        selected_entry = selection_receipt["training_view"]["files"].get(name)
        if selected_entry is None or (
            selected_entry["sha256"] != entry["sha256"]
            or int(selected_entry["bytes"]) != entry["bytes"]
        ):
            raise RuntimeError(
                f"training view changed after diagnostic for {name}"
            )

    hybrid = native_protected_evq_inv_freq(protected)
    config = type(
        "Config",
        (),
        {
            "hidden_size": 2_048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
        },
    )()
    output_mask = qk_unprotected_output_mask(config, protected)
    protocol = protocol_from_args(
        args,
        selection_receipt=selection_path,
        protected_pairs=selected_protected,
    )
    here = Path(__file__).resolve()
    trainer = here.with_name("train_4k_native_protected_evq.py")
    method = here.with_name("native_protected_evq.py")
    relation_loss = here.with_name("evq_attention_restoration.py")
    diagnostic = here.with_name("diagnose_4k_native_band_importance.py")
    nll_evaluator = here.with_name(
        "evaluate_native_protected_evq_nll.py"
    )
    ruler_evaluator = here.with_name(
        "evaluate_instruct_ruler_transfer.py"
    )
    two_wiki_evaluator = here.with_name(
        "evaluate_2wiki_phase_adaptation.py"
    )
    source = {
        "preflight": file_receipt(here),
        "trainer": file_receipt(trainer),
        "method": file_receipt(method),
        "relation_loss": file_receipt(relation_loss),
        "diagnostic": file_receipt(diagnostic),
        "nll_evaluator": file_receipt(nll_evaluator),
        "ruler_evaluator": file_receipt(ruler_evaluator),
        "two_wiki_evaluator": file_receipt(two_wiki_evaluator),
    }
    linearard = validate_linearard(args.linearard_root.resolve())
    receipt = {
        "status": PREPARED_STATUS,
        "classification": (
            "DESIGN_ONLY mature-model retrofit; not submitted EVQ-Cosh "
            "evidence and not a GPU launch authorization"
        ),
        "scientific_question": (
            "After a Native-only functional diagnostic identifies a small "
            "stable important rotary subset, can exact Native frequencies "
            "and zero direct Q/K-LoRA updates on that subset preserve broad "
            "4K attention while EVQ remains active elsewhere?"
        ),
        "existing_evidence": (
            "Full mature-model EVQ replacement plus selective Q/K adaptation "
            "recovers 4K 2Wiki but remains 29.75 RULER macro points below "
            "Native; LeRoPE independently shows that trained models can "
            "concentrate positional function in a small frequency subset."
        ),
        "smallest_missing_evidence": (
            "A passed Native band-concentration diagnostic followed by one "
            "bounded 4K attention-restoration run and capability gate."
        ),
        "protocol": protocol,
        "method_identity": {
            "arm": str(args.arm),
            "hybrid_frequency_sha256_float32": tensor_sha256(hybrid),
            "qk_output_mask_sha256": tensor_sha256(output_mask),
            "protected_native_pair_indices": list(protected),
            "diagnostic_selected_pair_indices": list(selected_protected),
            "unprotected_evq_pair_indices": [
                index
                for index in range(64)
                if index not in set(protected)
            ],
            "direct_lora_update_is_zero_on_protected_coordinates": True,
            "structural_identity_boundary": (
                "Protected frequencies and direct LoRA output coordinates "
                "are exact; deeper protected activations are not guaranteed "
                "identical because earlier hidden states can change."
            ),
        },
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                "composite_sha256": checkpoint_digest,
                "ready_receipt": file_receipt(checkpoint_ready),
            },
            "training_view": {
                "path": str(training_view),
                "files": training_files,
                "training_rows": int(len(view.training_rows)),
                "maximum_training_length": int(
                    np.asarray(view.lengths[view.training_rows]).max()
                ),
            },
            "selection_receipt": file_receipt(selection_path),
            "linearard_root": linearard,
        },
        "source": source,
        "outputs": {
            "gpu_ready_receipt": str(gpu_ready),
            "run_output": str(run_output),
        },
        "storage": {
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": int(args.minimum_free_bytes),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu_smoke_gate": (
            "Pinned kernel parity, BF16/Flash eligibility, full teacher+student "
            "memory fit, finite loss, and nonzero gradients on all 32 LoRA-B "
            "tensors. Any failure stops before training."
        ),
        "post_training_gate": {
            "arms": [
                "fresh untouched Native",
                "fresh immediate selected hybrid with zero adapter",
                "trained selected hybrid",
            ],
            "4k_first": [
                "2Wiki 200-row token-F1 and exact",
                "RULER 13-family 20-row-per-cell macro and per-family scores",
                "128-row held-out natural NLL",
                "independent QA/MCQA retention",
            ],
            "thresholds": {
                "2wiki_f1_max_drop_points_vs_native": 5.0,
                "ruler_macro_max_drop_points_vs_native": 10.0,
                "natural_nll_max_increase": 0.10,
                "native_positive_family_near_zero_collapse_allowed": False,
                "independent_capability_material_collapse_allowed": False,
            },
            "long_evaluation": (
                "8K then 16K only after every 4K gate passes, with the "
                "adapter frozen and strict autoregressive task metrics."
            ),
        },
        "stop_condition": (
            "Do not train if the diagnostic or smoke fails. Stop after one "
            "registered run if any 4K capability gate fails; do not infer "
            "success from attention loss, NLL, or PPL."
        ),
    }
    atomic_json(receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": PREPARED_STATUS,
                "receipt": str(receipt_output),
                "receipt_sha256": sha256_file(receipt_output),
                "protected_pair_indices": list(protected),
                "hybrid_frequency_sha256": tensor_sha256(hybrid),
                "qk_output_mask_sha256": tensor_sha256(output_mask),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
