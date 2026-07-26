#!/usr/bin/env python3
"""Factor value-token shift from context/template shift for a saved LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    evaluate,
    install_adaptation,
    load_model,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    EVAL_DISTRACTOR_COUNTS,
    EVAL_SOURCE_FRACTIONS,
    TRAIN_DISTRACTOR_COUNTS,
    TRAIN_SOURCE_FRACTIONS,
    annotate_rows,
    apply_schedule,
    build_probe_set,
    configure_cuda,
    hash_paths,
    load_documents,
    one_token_values,
    seed_everything,
    select_document_rows,
    sha256_file,
)


def load_adapter(
    path: Path,
    model: Any,
    readout: torch.nn.Module | None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state"]
    expected = dict(trainable_named_parameters(model, readout))
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise RuntimeError(
            f"adapter parameter mismatch: missing={missing}, extra={extra}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(
                state[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    return dict(payload.get("metadata", {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--documents-16k", type=Path, required=True)
    parser.add_argument(
        "--documents-16k-metadata", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=("geo", "evq"), required=True)
    parser.add_argument(
        "--adaptation",
        choices=("qkvo_answer", "qkvo_causal_margin"),
        required=True,
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--examples-per-cell", type=int, default=72)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument(
        "--train-source-fractions",
        type=float,
        nargs="+",
        default=list(TRAIN_SOURCE_FRACTIONS),
        help=(
            "Actual source fractions used to train the supplied adapter. "
            "Do not rely on the sparse-protocol default for dense-position "
            "adapters."
        ),
    )
    args = parser.parse_args()

    train_source_fractions = tuple(
        float(value) for value in args.train_source_fractions
    )
    if not train_source_fractions or any(
        value <= 0.0 or value >= 1.0
        for value in train_source_fractions
    ):
        raise ValueError(
            "train source fractions must be strictly between 0 and 1"
        )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed_everything(int(args.seed))
    runtime = configure_cuda()

    checkpoint = args.checkpoint.resolve()
    adapter_path = args.adapter.resolve()
    documents_path = args.documents_16k.resolve()
    metadata_path = args.documents_16k_metadata.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    train_values, eval_values = one_token_values(tokenizer)
    documents, metadata = load_documents(
        documents_path,
        metadata_path,
        expected_length=16_384,
    )
    document_rows = {
        "train": select_document_rows(metadata, split="validation"),
        "eval": select_document_rows(metadata, split="test"),
    }

    model = load_model(checkpoint)
    frequency = apply_schedule(model, args.schedule)
    readout = install_adaptation(
        model,
        args.adaptation,
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    adapter_metadata = load_adapter(
        adapter_path,
        model,
        readout,
    )
    if adapter_metadata.get("schedule") != args.schedule:
        raise RuntimeError(
            "adapter schedule does not match requested schedule"
        )
    model.to("cuda")
    if readout is not None:
        readout.to("cuda")

    cell_specs = (
        ("train_context_train_values", "train", train_values),
        ("train_context_eval_values", "train", eval_values),
        ("eval_context_train_values", "eval", train_values),
        ("eval_context_eval_values", "eval", eval_values),
    )
    combined_answer_ids = [
        token_id for _, token_id in train_values + eval_values
    ]
    cells: dict[str, Any] = {}
    for cell_index, (name, context_split, values) in enumerate(cell_specs):
        phase = "train" if context_split == "train" else "eval"
        data = build_probe_set(
            tokenizer=tokenizer,
            documents=documents,
            document_metadata=metadata,
            document_rows=document_rows[context_split],
            values=values,
            length=16_384,
            count=int(args.examples_per_cell),
            seed=int(args.seed) + 200_000 + 10_000 * cell_index,
            source_fractions=(
                train_source_fractions
                if context_split == "train"
                else EVAL_SOURCE_FRACTIONS
            ),
            distractor_counts=(
                TRAIN_DISTRACTOR_COUNTS
                if context_split == "train"
                else EVAL_DISTRACTOR_COUNTS
            ),
            phase=phase,
        )
        rows, summary = evaluate(
            model=model,
            readout=readout,
            data_by_length={16_384: data},
            answer_token_ids=combined_answer_ids,
            batch_size=int(args.eval_batch_size),
        )
        annotate_rows(rows, {16_384: data})
        cells[name] = {
            "context_split": context_split,
            "value_split": (
                "train" if values is train_values else "eval"
            ),
            "dataset_sha256": data.digest(),
            "summary": summary["L16384"],
            "rows": rows,
        }

    receipt = {
        "status": "OLMO2_LORA_OOD_FACTORIAL_COMPLETE",
        "metric_boundary": (
            "teacher-forced one-token diagnostic; factors value-token "
            "shift from context/template/document/position shift"
        ),
        "checkpoint": str(checkpoint),
        "adapter": str(adapter_path),
        "schedule": args.schedule,
        "adaptation": args.adaptation,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "examples_per_cell": int(args.examples_per_cell),
        "context_geometry": {
            "train_source_fractions": list(train_source_fractions),
            "eval_source_fractions": list(EVAL_SOURCE_FRACTIONS),
            "train_distractor_counts": list(TRAIN_DISTRACTOR_COUNTS),
            "eval_distractor_counts": list(EVAL_DISTRACTOR_COUNTS),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_files_sha256": hash_paths(
            (
                adapter_path,
                documents_path,
                metadata_path,
            )
        ),
        "adapter_metadata": adapter_metadata,
        "frequency": frequency,
        "runtime": runtime,
        "cells": cells,
    }
    temporary = output / "results.json.incomplete"
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "results.json")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "summary": {
                    name: value["summary"]
                    for name, value in cells.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
