#!/usr/bin/env python3
"""Matched qualitative dialogue smoke test for OLMo-2 adapters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_ruler_flash_attention,
    greedy_generate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    configure_cuda,
)

from .evaluate_instruct_ruler_screen import validate_adapter_metadata
from .prepare_data import atomic_json, sha256_file
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import apply_frequency


PROMPTS = (
    {
        "id": "arithmetic_exact",
        "text": "What is 2 + 3? Reply with only the number.",
        "expected_exact": "5",
    },
    {
        "id": "instruction_exact",
        "text": "Reply with exactly the word BLUE and nothing else.",
        "expected_exact": "BLUE",
    },
    {
        "id": "sequence",
        "text": "Continue the sequence 2, 4, 6, 8. Reply with the next number.",
        "expected_exact": "10",
    },
    {
        "id": "short_explanation",
        "text": "In one short sentence, explain why leaves usually look green.",
        "expected_exact": None,
    },
    {
        "id": "dialogue_memory",
        "text": (
            "My name is Lin. Answer the question directly: what name did I "
            "just give you?"
        ),
        "expected_exact": None,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency",
        choices=("native", "evq"),
        default="evq",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, ready_receipt)
    configure_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = load_model(checkpoint)
    frequency = apply_frequency(model, str(args.frequency))
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("dialogue smoke does not admit a readout")
    adapter = args.adapter.resolve()
    metadata = load_adapter(adapter, model, None)
    validate_adapter_metadata(
        metadata,
        checkpoint_digest=checkpoint_digest,
        frequency=frequency,
        frequency_name=str(args.frequency),
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to("cuda")

    rows: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = greedy_generate(
            model,
            input_ids,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=int(tokenizer.eos_token_id),
        )[0].detach().cpu()
        torch.cuda.synchronize()
        token_ids = [int(value) for value in generated.tolist()]
        visible_ids = (
            token_ids[:-1]
            if token_ids and token_ids[-1] == tokenizer.eos_token_id
            else token_ids
        )
        prediction = str(
            tokenizer.decode(
                visible_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        expected = prompt["expected_exact"]
        rows.append(
            {
                **prompt,
                "prediction": prediction,
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "terminal_eos": bool(
                    token_ids
                    and token_ids[-1] == int(tokenizer.eos_token_id)
                ),
                "exact_if_registered": (
                    None if expected is None else prediction == expected
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    receipt = {
        "status": "OLMO2_SIMPLE_DIALOGUE_SMOKE_COMPLETE_V1",
        "scope": (
            "matched qualitative smoke only; not a benchmark or a general "
            "instruction-following claim"
        ),
        "checkpoint_sha256": checkpoint_digest,
        "adapter": {
            "path": str(adapter),
            "sha256": sha256_file(adapter),
            "metadata": metadata,
        },
        "frequency": frequency,
        "protocol": {
            "greedy": True,
            "max_new_tokens": int(args.max_new_tokens),
            "chat_template": "checkpoint tokenizer",
            "prompt_count": len(PROMPTS),
        },
        "rows": rows,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
