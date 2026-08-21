#!/usr/bin/env python3
"""Strict autoregressive gate on held-out continuous-8K natural spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_ruler_flash_attention,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    sha256_file,
)

from .continuous_8k_adaptation import Continuous8KView
from .far_only_evq_residual import (
    install_far_pass_chord_residual,
    load_far_only_adapter,
    peek_far_only_adapter,
    set_far_pass_chord_route,
)


STATUS = "OLMO2_CONTINUOUS_8K_NATURAL_SPAN_GENERATION_DIAGNOSTIC_COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    return parser.parse_args()


@torch.inference_mode()
def generate_row(
    *, model: Any, prompt: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, int]:
    generated: list[torch.Tensor] = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=prompt,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :].float()
        target = reference[0, 0]
        rank = int((logits[0] > logits[0, target]).sum()) + 1
        past = outputs.past_key_values
        next_token = logits.argmax(dim=-1)
        for _ in range(reference.shape[1]):
            generated.append(next_token)
            if bool(torch.all(next_token == int(model.config.eos_token_id))):
                break
            outputs = model(
                input_ids=next_token[:, None],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    return torch.stack(generated, dim=1), rank


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    view = Continuous8KView(args.training_view.resolve())
    rows = view.validation_rows[: int(args.rows)]
    if len(rows) != int(args.rows):
        raise RuntimeError("diagnostic row count drift")

    configure_cuda()
    checkpoint = args.checkpoint.resolve()
    adapter = args.adapter.resolve()
    config, adapter_metadata = peek_far_only_adapter(adapter)
    model = load_model(checkpoint)
    install_far_pass_chord_residual(model, config)
    load_far_only_adapter(
        adapter,
        model,
        expected_checkpoint_sha256=adapter_metadata["base_checkpoint_sha256"],
    )
    configure_ruler_flash_attention(model)
    set_far_pass_chord_route(model, True)
    model.config.use_cache = True
    model.eval()
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    exact: list[float] = []
    first_top1: list[float] = []
    ranks: list[int] = []
    examples = []
    for row_index in rows.tolist():
        label_row = np.asarray(view.labels[int(row_index)], dtype=np.int64)
        first = int(np.argmax(label_row != -100))
        reference = torch.from_numpy(label_row[first:].copy())[None, :].to(
            "cuda"
        )
        prompt = torch.from_numpy(
            np.asarray(
                view.input_ids[int(row_index), :first], dtype=np.int64
            ).copy()
        )[None, :].to("cuda")
        prediction, rank = generate_row(
            model=model, prompt=prompt, reference=reference
        )
        matched = bool(
            prediction.shape == reference.shape
            and torch.equal(prediction, reference)
        )
        first_matched = bool(prediction[0, 0] == reference[0, 0])
        exact.append(float(matched))
        first_top1.append(float(first_matched))
        ranks.append(rank)
        examples.append(
            {
                "row": int(row_index),
                "prompt_tokens": int(prompt.shape[1]),
                "reference": [int(value) for value in reference[0].tolist()],
                "prediction": [int(value) for value in prediction[0].tolist()],
                "first_token_rank": rank,
                "first_token_top1": first_matched,
                "full_answer_plus_eos_exact": matched,
            }
        )
        del prompt, reference, prediction

    result = {
        "status": STATUS,
        "metric_boundary": (
            "Held-out rows from the independent natural-span training family; "
            "this is an internal capability gate, not RULER transfer evidence."
        ),
        "checkpoint": str(checkpoint),
        "training_view_manifest_sha256": sha256_file(
            view.root / "manifest.json"
        ),
        "adapter_sha256": sha256_file(adapter),
        "adapter_metadata": adapter_metadata,
        "rows": len(rows),
        "first_token_top1": float(np.mean(first_top1)),
        "first_token_top1_count": int(sum(first_top1)),
        "median_first_token_rank": float(np.median(ranks)),
        "mean_first_token_rank": float(np.mean(ranks)),
        "full_answer_plus_eos_exact": float(np.mean(exact)),
        "full_answer_plus_eos_exact_count": int(sum(exact)),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "examples": examples,
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
