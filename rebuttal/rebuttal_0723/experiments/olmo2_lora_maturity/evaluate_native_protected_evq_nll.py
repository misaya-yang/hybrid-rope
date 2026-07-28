#!/usr/bin/env python3
"""Evaluate matched natural-text NLL for Native-protected EVQ arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    sha256_file,
)

from .native_protected_evq import (
    ADAPTATION_NAME,
    FREQUENCY_NAME,
    apply_native_protected_evq,
    install_masked_qk_lora,
)
from .train_4k_native_protected_evq import load_selection
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import apply_frequency, evaluate_natural_nll


STATUS = "OLMO2_NATIVE_PROTECTED_EVQ_NLL_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "native",
            "immediate-protected",
            "trained-protected",
            "immediate-full-evq-control",
            "trained-full-evq-control",
        ),
        required=True,
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1024.0)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(4_096,)
    )
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--tail-tokens", type=int, default=1_024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    lengths = tuple(sorted(set(int(value) for value in args.lengths)))
    if (
        not lengths
        or any(value not in {4_096, 8_192, 16_384} for value in lengths)
    ):
        raise RuntimeError("NLL lengths must be an ordered 4K/8K/16K subset")
    trained = args.arm.startswith("trained-")
    adapter = None if args.adapter is None else args.adapter.resolve()
    if trained != (adapter is not None):
        raise RuntimeError(
            "trained arms require one adapter; immediate/native arms forbid it"
        )
    checkpoint = args.checkpoint.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    selection_path = args.selection_receipt.resolve()
    _, selected = load_selection(selection_path)
    model = load_model(checkpoint)
    if args.arm == "native":
        frequency = apply_frequency(model, "native")
        protected = None
        output_mask = None
    else:
        protected = (
            ()
            if "full-evq-control" in args.arm
            else selected
        )
        frequency = apply_native_protected_evq(model, protected)
        output_mask = None
        if trained:
            _, output_mask = install_masked_qk_lora(
                model,
                protected_pairs=protected,
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
            metadata = load_adapter(adapter, model, None)
            expected = {
                "base_checkpoint_sha256": checkpoint_digest,
                "frequency": FREQUENCY_NAME,
                "frequency_sha256_float32": frequency[
                    "active_sha256_float32"
                ],
                "adaptation": ADAPTATION_NAME,
                "rank": int(args.rank),
                "alpha": float(args.alpha),
                "training_sequence_length": 4_096,
                "qk_output_mask_sha256": tensor_sha256(output_mask),
                "protected_native_pair_indices": list(protected),
            }
            for name, value in expected.items():
                if metadata.get(name) != value:
                    raise RuntimeError(f"adapter metadata drift for {name}")
    runtime = configure_cuda()
    model.config.use_cache = False
    model.eval()
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    metrics = evaluate_natural_nll(
        model=model,
        background_dir=args.background_dir.resolve(),
        lengths=lengths,
        rows=int(args.rows),
        tail_tokens=int(args.tail_tokens),
    )
    receipt = {
        "status": STATUS,
        "arm": str(args.arm),
        "checkpoint_sha256": checkpoint_digest,
        "selection_receipt_sha256": sha256_file(selection_path),
        "adapter_sha256": None if adapter is None else sha256_file(adapter),
        "frequency": frequency,
        "protected_native_pair_indices": (
            None if protected is None else list(protected)
        ),
        "protocol": {
            "lengths": list(lengths),
            "rows": int(args.rows),
            "tail_tokens": int(args.tail_tokens),
            "teacher_forced_nll_only": True,
        },
        "metrics": metrics,
        "runtime": {
            **runtime,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "claim_boundary": (
            "Natural-text NLL is a retention/probability endpoint and does "
            "not establish retrieval, QA, or autoregressive capability."
        ),
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
