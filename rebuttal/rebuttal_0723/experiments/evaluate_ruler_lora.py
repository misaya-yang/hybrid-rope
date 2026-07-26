#!/usr/bin/env python3
"""Evaluate a matched Geo/EVQ OLMo-2 LoRA adapter on frozen RULER rows."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    TAU,
    assert_model_config,
    endpoint_geo_inv_freq,
    patch_endpoint_evq,
    sha256_file,
    tensor_sha256,
    trainable_parameter_count,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    GENERATION_TOKENS,
    TARGET_DELIMITER,
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
    load_completed,
    resolve_released_snapshot,
    row_sha256,
    score_prediction,
    selected_rows,
    validate_data,
    write_json,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)


TASK = "niah_single_1"


def apply_schedule(model: Any, schedule: str) -> dict[str, Any]:
    if schedule == "geo":
        active = (
            model.model.rotary_emb.inv_freq.detach().cpu().float()
        )
        expected = endpoint_geo_inv_freq()
        if not torch.equal(active, expected):
            raise RuntimeError("released checkpoint Geo frequency drift")
        return {
            "active_schedule": "geo",
            "tau": 0.0,
            "active_sha256_float32": tensor_sha256(active),
        }
    if schedule == "evq":
        receipt = patch_endpoint_evq(model, tau=TAU)
        receipt["active_schedule"] = "evq"
        receipt["active_sha256_float32"] = receipt[
            "evq_sha256_float32"
        ]
        return receipt
    raise ValueError(f"unknown schedule {schedule!r}")


def load_adapter(
    *,
    model: Any,
    adapter_path: Path,
    schedule: str,
    expected_base_sha256: str,
    expected_frequency_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(
        adapter_path, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict):
        raise RuntimeError("adapter payload is not a mapping")
    state = payload.get("state")
    metadata = payload.get("metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise RuntimeError("adapter payload lacks state or metadata")
    trained_schedule = metadata.get(
        "schedule", metadata.get("frequency")
    )
    if trained_schedule != schedule:
        raise RuntimeError(
            "adapter schedule does not match evaluation schedule"
        )
    if metadata.get("base_checkpoint_sha256") != expected_base_sha256:
        raise RuntimeError("adapter base checkpoint hash mismatch")
    if (
        metadata.get("frequency_sha256_float32")
        != expected_frequency_sha256
    ):
        raise RuntimeError("adapter frequency hash mismatch")
    adaptation = str(metadata.get("adaptation"))
    if adaptation != "qkvo_answer":
        raise RuntimeError(
            "formal RULER gate currently admits qkvo_answer only"
        )
    readout = install_adaptation(
        model,
        adaptation,
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
    )
    if readout is not None:
        raise RuntimeError("unexpected readout adapter")
    named = dict(trainable_named_parameters(model, readout))
    if set(named) != set(state):
        raise RuntimeError(
            "adapter parameter names do not match installed modules"
        )
    with torch.no_grad():
        for name, parameter in named.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"adapter shape drift at {name}")
            parameter.copy_(
                value.to(device=parameter.device, dtype=parameter.dtype)
            )
    return {
        "path": str(adapter_path),
        "sha256": sha256_file(adapter_path),
        "metadata": metadata,
    }


def load_model(
    *,
    checkpoint: Path,
    schedule: str,
    adapter_path: Path | None,
    ready_receipt: Path | None,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    if ready_receipt is None:
        snapshot = resolve_released_snapshot(checkpoint)
        base_checkpoint_sha256 = ":".join(
            sha256_file(path)
            for path in sorted(checkpoint.glob("model-*.safetensors"))
        )
    else:
        base_checkpoint_sha256 = ready_checkpoint_digest(
            checkpoint, ready_receipt
        )
        ready = json.loads(ready_receipt.read_text(encoding="utf-8"))
        snapshot = {
            "name": "step30000_63B",
            "revision": ready["checkpoint"]["revision"],
            "weights": ready["checkpoint"]["files"],
        }
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=(
            torch.float32
            if ready_receipt is None
            else torch.bfloat16
        ),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    assert_model_config(model.config)
    if trainable_parameter_count(model) != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError("RULER model parameter-count drift")
    frequency = apply_schedule(model, schedule)
    adapter = None
    if adapter_path is not None:
        adapter = load_adapter(
            model=model,
            adapter_path=adapter_path,
            schedule=schedule,
            expected_base_sha256=base_checkpoint_sha256,
            expected_frequency_sha256=frequency[
                "active_sha256_float32"
            ],
        )
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to(torch.device("cuda", 0))
    return model, {
        "name": snapshot["name"],
        "revision": snapshot["revision"],
        "weights": snapshot["weights"],
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "frequency": frequency,
    }, adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--ready-receipt", type=Path)
    parser.add_argument("--schedule", choices=("geo", "evq"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 16_384],
    )
    parser.add_argument("--limit-per-cell", type=int, default=100)
    args = parser.parse_args()

    lengths = tuple(int(value) for value in args.lengths)
    if lengths != (4_096, 8_192, 16_384):
        raise RuntimeError(
            "formal RULER gate requires fixed 4K/8K/16K cells"
        )
    if int(args.limit_per_cell) < 100:
        raise RuntimeError("formal RULER gate requires at least 100 rows/cell")
    output = args.output.resolve()
    if (output / "results.json").exists():
        raise FileExistsError(output / "results.json")

    configure_cuda()
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability[0] != 12:
        raise RuntimeError(
            f"RULER runner requires Blackwell, got {name}"
        )
    data_receipt = validate_data(
        args.data_root.resolve(),
        tasks=(TASK,),
        lengths=lengths,
    )
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    adapter_path = (
        None if args.adapter is None else args.adapter.resolve()
    )
    ready_receipt = (
        None
        if args.ready_receipt is None
        else args.ready_receipt.resolve()
    )
    model, snapshot, adapter = load_model(
        checkpoint=checkpoint,
        schedule=args.schedule,
        adapter_path=adapter_path,
        ready_receipt=ready_receipt,
    )
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    completed = load_completed(examples_path)
    expected = len(lengths) * int(args.limit_per_cell)
    rows = selected_rows(
        Path(data_receipt["files"][TASK]["path"]),
        lengths=lengths,
        limit_per_cell=int(args.limit_per_cell),
    )
    torch.cuda.reset_peak_memory_stats()

    with examples_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = (TASK, int(row["max_length"]), int(row["index"]))
            if key in completed:
                continue
            delimiter = TARGET_DELIMITER.get(TASK, " ")
            prompt = row["input"] + delimiter + row["gen_prefix"]
            input_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to("cuda")
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=GENERATION_TOKENS[TASK],
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prediction = tokenizer.decode(
                output_ids[0].detach().cpu(),
                skip_special_tokens=True,
            )
            result = {
                "task": TASK,
                "max_length": int(row["max_length"]),
                "index": int(row["index"]),
                "row_sha256": row_sha256(row),
                "input_tokens": int(input_ids.numel()),
                "generated_tokens": int(output_ids.numel()),
                "prediction": prediction,
                "references": row["outputs"],
                "score": score_prediction(
                    prediction, row["outputs"]
                ),
                "elapsed_seconds": elapsed,
            }
            handle.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            completed.add(key)
            print(
                f"{len(completed)}/{expected} {TASK} "
                f"L={row['max_length']} score={result['score']:.3f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )

    result_rows = [
        json.loads(line)
        for line in examples_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    relevant = [
        row
        for row in result_rows
        if row["task"] == TASK
        and int(row["max_length"]) in lengths
        and int(row["index"]) < int(args.limit_per_cell)
    ]
    if len(relevant) != expected:
        raise RuntimeError(
            f"RULER result count {len(relevant)} != {expected}"
        )
    cells = {}
    for length in lengths:
        scores = [
            float(row["score"])
            for row in relevant
            if int(row["max_length"]) == length
        ]
        cells[str(length)] = sum(scores) / len(scores)
    macro = sum(cells.values()) / len(cells)
    if not math.isfinite(macro):
        raise RuntimeError("RULER produced a non-finite score")
    receipt = {
        "status": "RULER_LORA_EVALUATION_COMPLETE",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": {
            "name": snapshot["name"],
            "revision": snapshot["revision"],
            "path": str(checkpoint),
            "weights": snapshot["weights"],
            "base_checkpoint_sha256": snapshot[
                "base_checkpoint_sha256"
            ],
        },
        "schedule": args.schedule,
        "frequency": snapshot["frequency"],
        "adapter": adapter,
        "data": data_receipt,
        "protocol": {
            "task": TASK,
            "lengths": list(lengths),
            "limit_per_cell": int(args.limit_per_cell),
            "greedy": True,
            "precision": "bf16_autocast_fp32_weights",
            "compile": False,
            "training_task_overlap": False,
        },
        "runtime": {
            "gpu": name,
            "compute_capability": list(capability),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "results": {
            "cells": cells,
            "macro_average": macro,
            "examples": len(relevant),
            "examples_sha256": sha256_file(examples_path),
        },
    }
    write_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
