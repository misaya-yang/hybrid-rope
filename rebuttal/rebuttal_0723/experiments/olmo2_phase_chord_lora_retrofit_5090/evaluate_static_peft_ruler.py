#!/usr/bin/env python3
"""Evaluate a static-table standard-PEFT bundle on prepared full RULER."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .frequency_assets import float32_sha256, load_frequency_assets
from .receipts import (
    ARMS,
    atomic_json,
    checkpoint_receipt,
    load_standard_peft_bundle,
    require_gpu_authorization,
    sha256_file,
    tree_hashes,
)


STATUS = "OLMO2_STATIC_PEFT_RULER_COMPLETE_V1"
ALLOWED_LENGTHS = (4_096, 8_192, 16_384)
EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def strict_eos_boundary(
    token_ids: Sequence[int], eos_token_id: int
) -> dict[str, Any]:
    values = [int(value) for value in token_ids]
    eos_indices = [
        index for index, value in enumerate(values) if value == int(eos_token_id)
    ]
    first = None if not eos_indices else eos_indices[0]
    return {
        "eos_token_id": int(eos_token_id),
        "eos_observed": first is not None,
        "first_eos_index": first,
        "eos_count": len(eos_indices),
        "tokens_after_first_eos": (
            None if first is None else len(values) - first - 1
        ),
        "strict_terminal_eos_boundary": bool(
            first is not None and first == len(values) - 1 and len(eos_indices) == 1
        ),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    lengths: Sequence[int],
) -> dict[str, Any]:
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
        TASK_CONFIGS,
    )

    cells: dict[str, dict[str, Any]] = {}
    family_scores: list[float] = []
    length_scores: dict[str, list[float]] = {str(length): [] for length in lengths}
    for task in tasks:
        cells[task] = {}
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["task"] == task
                and int(row["nominal_length"]) == int(length)
            ]
            if not selected:
                continue
            score = float(
                np.mean([float(row["official_task_score"]) for row in selected])
            )
            recall = float(
                np.mean([float(row["reference_recall"]) for row in selected])
            )
            eos = float(
                np.mean(
                    [
                        float(row["eos_boundary"]["strict_terminal_eos_boundary"])
                        for row in selected
                    ]
                )
            )
            cells[task][str(length)] = {
                "examples": len(selected),
                "role": str(TASK_CONFIGS[task]["role"]),
                "official_metric": str(TASK_CONFIGS[task]["official_metric"]),
                "official_task_score": score,
                "reference_recall": recall,
                "strict_terminal_eos_boundary_rate": eos,
                "mean_generated_tokens": float(
                    np.mean([int(row["generated_tokens"]) for row in selected])
                ),
                "mean_elapsed_seconds": float(
                    np.mean([float(row["elapsed_seconds"]) for row in selected])
                ),
            }
            family_scores.append(score)
            length_scores[str(length)].append(score)
    if not family_scores:
        raise RuntimeError("RULER aggregation has no completed cell")
    macro_by_length = {
        length: float(np.mean(scores))
        for length, scores in length_scores.items()
        if scores
    }
    return {
        "cells": cells,
        "macro_official_score": float(np.mean(family_scores)),
        "macro_official_score_by_length": macro_by_length,
        "macro_definition": "unweighted mean over prepared task-length cells",
    }


def _bundle_contract(
    *,
    bundle: Path,
    arm: str,
    checkpoint_sha256: str,
    frequency_manifest: Path,
) -> dict[str, Any]:
    metadata = _json(bundle / "metadata.json")
    tables = load_frequency_assets(frequency_manifest)
    key = "Native" if arm == "native" else arm
    target_sha = float32_sha256(tables[key])
    if (
        metadata.get("arm") != arm
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
        or metadata.get("frequency_manifest_sha256")
        != sha256_file(frequency_manifest)
        or metadata.get("target_float32_sha256") != target_sha
    ):
        raise RuntimeError("static PEFT bundle metadata drift")
    adapter_config = _json(bundle / "adapter" / "adapter_config.json")
    targets = adapter_config.get("target_modules")
    if isinstance(targets, str):
        targets = [targets]
    if (
        int(adapter_config.get("r", -1)) != 64
        or float(adapter_config.get("lora_alpha", -1)) != 128.0
        or float(adapter_config.get("lora_dropout", -1)) != 0.0
        or str(adapter_config.get("bias")) != "none"
        or set(str(value) for value in (targets or []))
        != EXPECTED_TARGET_MODULES
    ):
        raise RuntimeError("standard PEFT QKVO rank64/alpha128 config drift")
    return {
        "arm": arm,
        "metadata_sha256": sha256_file(bundle / "metadata.json"),
        "adapter_config_sha256": sha256_file(
            bundle / "adapter" / "adapter_config.json"
        ),
        "target_float32_sha256": target_sha,
        "bundle_files": tree_hashes(bundle),
    }


def _load_model(checkpoint: Path, bundle: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return load_standard_peft_bundle(base_model=base, bundle=bundle)


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    maturity = package.parent / "olmo2_lora_maturity"
    generation = package.parent / "olmo2_1b_evq" / "evaluate_ruler.py"
    paths = {
        "evaluator": Path(__file__).resolve(),
        "receipts": package / "receipts.py",
        "frequency_assets": package / "frequency_assets.py",
        "prepared_data": maturity / "prepare_instruct_ruler_transfer.py",
        "reference_evaluator": maturity / "evaluate_instruct_ruler_transfer.py",
        "generation": generation,
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--frequency-manifest", type=Path, required=True)
    parser.add_argument("--adapter-bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(ALLOWED_LENGTHS))
    parser.add_argument("--limit-per-cell", type=int, default=20)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda,
        configure_ruler_flash_attention,
        greedy_generate,
        row_sha256,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        _validate_data,
        official_string_match_all,
        official_task_score,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
        TASK_CONFIGS,
        chat_input_ids,
    )

    lengths = tuple(int(value) for value in args.lengths)
    if tuple(sorted(set(lengths))) != lengths or any(
        value not in ALLOWED_LENGTHS for value in lengths
    ):
        raise RuntimeError("RULER lengths must be an ordered subset of 4K/8K/16K")
    tasks = None if args.tasks is None else tuple(str(value) for value in args.tasks)
    checkpoint = args.checkpoint.resolve()
    checkpoint_info = checkpoint_receipt(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    bundle = args.adapter_bundle.resolve()
    bundle_info = _bundle_contract(
        bundle=bundle,
        arm=str(args.arm),
        checkpoint_sha256=checkpoint_info["weight_sha256"],
        frequency_manifest=args.frequency_manifest.resolve(),
    )
    data_receipt, prepared_rows = _validate_data(
        root=args.data_root.resolve(),
        checkpoint=checkpoint,
        requested_tasks=tasks,
        requested_lengths=lengths,
        limit_per_cell=int(args.limit_per_cell),
    )
    selected_tasks = tuple(
        dict.fromkeys(str(row["_task"]) for row in prepared_rows)
    )
    selected_lengths = tuple(
        sorted(set(int(row["_nominal_length"]) for row in prepared_rows))
    )
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    incomplete.mkdir(parents=True)

    configure_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    model, load_receipt = _load_model(checkpoint, bundle)
    if (
        load_receipt["active_inv_freq_float32_sha256"]
        != bundle_info["target_float32_sha256"]
    ):
        raise RuntimeError("loaded static table differs from bundle metadata")
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()
    raw_path = incomplete / "examples.jsonl"
    completed: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as handle:
        for ordinal, row in enumerate(prepared_rows, start=1):
            task = str(row["_task"])
            length = int(row["_nominal_length"])
            local_index = int(row["_local_index"])
            generation_tokens = int(row["_generation_tokens"])
            chat_ids = chat_input_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["input"]}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            )
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            input_ids = torch.cat((chat_ids, prefix_ids), dim=1).to("cuda")
            if input_ids.shape[1] + generation_tokens > length:
                raise RuntimeError(
                    f"{task} exceeds L{length}: {input_ids.shape[1]}+{generation_tokens}"
                )
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=generation_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated_ids = [
                int(value) for value in output_ids[0].detach().cpu().tolist()
            ]
            prediction = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            references = [str(value) for value in row["outputs"]]
            metric = str(TASK_CONFIGS[task]["official_metric"])
            score = official_task_score(prediction, references, metric)
            recall = official_string_match_all(prediction, references)
            result = {
                "ordinal": ordinal,
                "task": task,
                "task_role": str(TASK_CONFIGS[task]["role"]),
                "nominal_length": length,
                "local_index": local_index,
                "source_row_index": int(row["index"]),
                "row_sha256": row_sha256(
                    {
                        name: value
                        for name, value in row.items()
                        if not name.startswith("_")
                    }
                ),
                "input_tokens": int(input_ids.numel()),
                "input_token_ids": [
                    int(value) for value in input_ids[0].detach().cpu().tolist()
                ],
                "maximum_generation_tokens": generation_tokens,
                "generated_tokens": len(generated_ids),
                "generated_token_ids": generated_ids,
                "decoded_prediction": prediction,
                "references": references,
                "official_metric": metric,
                "official_task_score": float(score),
                "reference_recall": float(recall),
                "all_references_found": float(recall == 1.0),
                "eos_boundary": strict_eos_boundary(
                    generated_ids, int(tokenizer.eos_token_id)
                ),
                "elapsed_seconds": elapsed,
            }
            handle.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()
            completed.append(result)
            print(
                f"{ordinal}/{len(prepared_rows)} {task} L={length} "
                f"official={score:.3f} seconds={elapsed:.2f}",
                flush=True,
            )
    aggregates = aggregate_rows(completed, selected_tasks, selected_lengths)
    result = {
        "schema_version": 1,
        "status": STATUS,
        "arm": str(args.arm),
        "checkpoint": checkpoint_info,
        "frequency_manifest": {
            "path": str(args.frequency_manifest.resolve()),
            "sha256": sha256_file(args.frequency_manifest.resolve()),
            "target_float32_sha256": bundle_info["target_float32_sha256"],
        },
        "adapter_bundle": bundle_info,
        "adapter_load": load_receipt,
        "prepared_data": data_receipt,
        "protocol": {
            "tasks": list(selected_tasks),
            "lengths": list(selected_lengths),
            "limit_per_cell": int(args.limit_per_cell),
            "greedy": True,
            "prepared_rows": len(prepared_rows),
            "decode_skip_special_tokens": True,
            "decode_cleanup": False,
        },
        "aggregates": aggregates,
        "raw_examples": {
            "relative_path": "examples.jsonl",
            "sha256": sha256_file(raw_path),
            "rows": len(completed),
            "contains_input_token_ids": True,
            "contains_generated_token_ids": True,
            "contains_decoded_prediction": True,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "code_sha256": _code_hashes(),
        "metric_boundary": (
            "Official RULER scores and strict generated-token EOS boundaries "
            "are capability evidence for this frozen adapter only; they do not "
            "establish universality, SOTA, or cross-seed significance."
        ),
    }
    atomic_json(incomplete / "results.json", result)
    result["results_payload_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_gpu_authorization(
        cli_authorize=bool(args.authorize), environment=os.environ
    )
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "macro_official_score": result["aggregates"][
                    "macro_official_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
