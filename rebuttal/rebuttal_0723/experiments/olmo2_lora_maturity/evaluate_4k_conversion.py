#!/usr/bin/env python3
"""Evaluate a 4K-only EVQ-LoRA adapter on frozen 16K causal sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .causal_data import load_probe_set
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    evaluate_causal,
    evaluate_natural_nll,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--causal-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["canary_full_heldout"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--natural-tail-tokens", type=int, default=1_024)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()

    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    model = load_model(checkpoint)
    frequency = apply_frequency(model, "evq")
    adapter_receipt: dict[str, Any] | None = None
    if args.adapter is not None:
        adapter_path = args.adapter.resolve()
        readout = install_adaptation(
            model,
            "qkvo_answer",
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        if readout is not None:
            raise RuntimeError("4K conversion adapter unexpectedly has readout")
        metadata = load_adapter(adapter_path, model, None)
        if metadata.get("base_checkpoint_sha256") != checkpoint_digest:
            raise RuntimeError("adapter base checkpoint digest mismatch")
        if metadata.get("frequency") != "evq":
            raise RuntimeError("adapter was not trained with EVQ")
        if int(metadata.get("training_sequence_length", -1)) != 4_096:
            raise RuntimeError("adapter violates the 4K training contract")
        if (
            metadata.get("frequency_sha256_float32")
            != frequency["active_sha256_float32"]
        ):
            raise RuntimeError("adapter frequency digest mismatch")
        adapter_receipt = {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "metadata": metadata,
        }
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.to("cuda")

    natural_nll = (
        {}
        if args.background_dir is None
        else evaluate_natural_nll(
            model=model,
            background_dir=args.background_dir.resolve(),
            lengths=(4_096, 8_192, 16_384),
            rows=int(args.natural_eval_rows),
            tail_tokens=int(args.natural_tail_tokens),
        )
    )

    causal_root = args.causal_data.resolve()
    collection_path = causal_root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in collection["sets"]}
    requested = list(dict.fromkeys(str(value) for value in args.sets))
    if not requested or not set(requested) <= set(entries):
        raise RuntimeError(
            f"unknown causal sets: {sorted(set(requested) - set(entries))}"
        )
    answer_ids = [
        int(value)
        for value in collection["registered_eval_value_token_ids"]
    ]
    results = {}
    for name in requested:
        entry = entries[name]
        data = load_probe_set(causal_root / entry["relative_path"])
        rows, summary = evaluate_causal(
            model=model,
            data=data,
            answer_token_ids=answer_ids,
            batch_size=int(args.batch_size),
        )
        results[name] = {
            "purpose": entry["purpose"],
            "dataset_sha256": entry["dataset_sha256"],
            "summary": summary,
            "rows": rows,
        }
        print(
            json.dumps(
                {
                    "set": name,
                    "overall": summary["overall"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    receipt = {
        "status": "OLMO2_4K_CONVERSION_16K_EVAL_COMPLETE",
        "metric_boundary": (
            "teacher-forced one-token source-causal evaluation at 16K; "
            "not an autoregressive downstream claim"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "frequency": frequency,
        "adapter": adapter_receipt,
        "causal_collection_sha256": sha256_file(collection_path),
        "sets": requested,
        "natural_nll": natural_nll,
        "results": results,
        "runtime": runtime,
        "protocol": {
            "maximum_adapter_training_length": (
                None if adapter_receipt is None else 4_096
            ),
            "evaluation_length": 16_384,
            "batch_size": int(args.batch_size),
            "full_vocabulary_rank": True,
            "source_deletion": True,
            "association_swap": True,
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "result_sha256": sha256_file(output / "results.json"),
                "natural_nll": natural_nll,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
