#!/usr/bin/env python3
"""Flash-only, resumable RULER evaluation for the released 11B-token model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    GEO1000_REVISION,
    GEO1000_WEIGHT_FILES,
    GEO2000_REVISION,
    GEO2000_WEIGHT_FILES,
    GEO5000_REVISION,
    GEO5000_WEIGHT_FILES,
    assert_model_config,
    endpoint_geo_inv_freq,
    sha256_file,
    trainable_parameter_count,
    validate_weight_files,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_ruler_data import (
    LENGTHS,
    LM_EVAL_COMMIT,
    TASKS,
)


GENERATION_TOKENS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "ruler_vt": 30,
    "ruler_cwe": 120,
    "ruler_fwe": 50,
    "ruler_qa_squad": 32,
    "ruler_qa_hotpot": 32,
}
TARGET_DELIMITER = {"ruler_cwe": "\n\n"}
QUICK_TASKS = (
    "niah_single_1",
    "niah_multikey_2",
    "ruler_vt",
    "ruler_fwe",
)
RULER_FLASH_ATTENTION_IMPLEMENTATION = "evq_ruler_flash_kv_sdpa"
RELEASED_SNAPSHOTS = {
    "geo1000": {
        "revision": GEO1000_REVISION,
        "weights": GEO1000_WEIGHT_FILES,
    },
    "geo2000": {
        "revision": GEO2000_REVISION,
        "weights": GEO2000_WEIGHT_FILES,
    },
    "geo5000": {
        "revision": GEO5000_REVISION,
        "weights": GEO5000_WEIGHT_FILES,
    },
}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def score_prediction(prediction: str, references: list[str]) -> float:
    prediction = "".join(
        char if char.isprintable() else "\n" for char in prediction
    ).strip().lower()
    return sum(ref.lower() in prediction for ref in references) / len(references)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def ruler_causal_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    if attention_mask is not None:
        raise RuntimeError("RULER Flash-only attention received a padding mask")
    return None


def ruler_flash_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    del module
    if attention_mask is not None:
        raise RuntimeError("RULER Flash-only attention received a mask")
    query_length = query.shape[-2]
    key_length = key.shape[-2]
    if query_length != key_length and query_length != 1:
        raise RuntimeError(
            "RULER Flash-only attention only admits full prefill or "
            "single-token KV-cache decode"
        )
    query_heads = int(query.shape[-3])
    key_heads = int(key.shape[-3])
    if query_heads % key_heads != 0:
        raise RuntimeError("RULER Flash-only attention received incompatible GQA heads")
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(dropout),
        scale=scaling,
        is_causal=query_length > 1,
        enable_gqa=query_heads != key_heads,
    )
    return output.transpose(1, 2).contiguous(), None


def configure_ruler_flash_attention(model: Any) -> None:
    AttentionInterface.register(
        RULER_FLASH_ATTENTION_IMPLEMENTATION,
        ruler_flash_forward,
    )
    AttentionMaskInterface.register(
        RULER_FLASH_ATTENTION_IMPLEMENTATION,
        ruler_causal_mask,
    )
    model.config._attn_implementation = (
        RULER_FLASH_ATTENTION_IMPLEMENTATION
    )


def validate_data(
    root: Path,
    *,
    tasks: tuple[str, ...],
    lengths: tuple[int, ...],
) -> dict[str, Any]:
    manifest_path = root / "ruler_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RULER_DATA_VERIFIED":
        raise RuntimeError("RULER data is not verified")
    if manifest["implementation"]["commit"] != LM_EVAL_COMMIT:
        raise RuntimeError("RULER implementation commit drift")
    if tuple(manifest["protocol"]["lengths"]) != lengths:
        raise RuntimeError("RULER lengths drift")
    if not set(tasks).issubset(manifest["implementation"]["tasks"]):
        raise RuntimeError("RULER task set drift")
    files = {}
    for task in tasks:
        receipt = manifest["files"][task]
        path = root / receipt["path"]
        digest = sha256_file(path)
        if digest != receipt["sha256"]:
            raise RuntimeError(f"RULER data hash drift: {task}")
        files[task] = {
            "path": str(path),
            "sha256": digest,
            "rows": int(receipt["rows"]),
        }
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "files": files,
    }


def resolve_released_snapshot(checkpoint: Path) -> dict[str, Any]:
    """Match a local released directory to the pinned 1k/2k/5k weight table."""
    errors: list[str] = []
    for name, expected in RELEASED_SNAPSHOTS.items():
        try:
            weights = validate_weight_files(checkpoint, expected["weights"])
        except Exception as exc:  # noqa: BLE001 - collect mismatches then fail closed
            errors.append(f"{name}: {exc}")
            continue
        return {
            "name": name,
            "revision": expected["revision"],
            "weights": weights,
        }
    raise RuntimeError(
        "checkpoint does not match any pinned released snapshot "
        f"(geo1000/geo2000/geo5000): {'; '.join(errors)}"
    )


def load_model(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    snapshot = resolve_released_snapshot(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    assert_model_config(model.config)
    if trainable_parameter_count(model) != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError("RULER model parameter-count drift")
    native = model.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32)
    if not torch.equal(native, endpoint_geo_inv_freq()):
        raise RuntimeError("released Geo frequency drift")
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to(torch.device("cuda", 0))
    return model, snapshot


def greedy_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    generated: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if (
                eos_token_id is not None
                and bool(torch.all(next_token == eos_token_id))
            ):
                break
            outputs = model(
                input_ids=next_token[:, None],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def load_completed(path: Path) -> set[tuple[str, int, int]]:
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(
            (row["task"], int(row["max_length"]), int(row["index"]))
        )
    return completed


def selected_rows(
    path: Path,
    *,
    lengths: tuple[int, ...],
    limit_per_cell: int,
) -> list[dict[str, Any]]:
    counts = {length: 0 for length in lengths}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        length = int(row["max_length"])
        if length not in counts or counts[length] >= limit_per_cell:
            continue
        rows.append(row)
        counts[length] += 1
    expected = {length: limit_per_cell for length in lengths}
    if counts != expected:
        raise RuntimeError(f"RULER cell count drift: {counts} != {expected}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("quick", "full"), default="quick"
    )
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(LENGTHS)
    )
    parser.add_argument("--limit-per-cell", type=int, default=100)
    args = parser.parse_args()

    lengths = tuple(args.lengths)
    if (
        not lengths
        or tuple(sorted(set(lengths))) != lengths
        or any(length < 2_048 or length > 32_768 for length in lengths)
        or any(length % 1_024 for length in lengths)
    ):
        raise RuntimeError(
            "RULER lengths must be unique ascending 1K multiples in [2K, 32K]"
        )
    if args.limit_per_cell < 1 or args.limit_per_cell > 500:
        raise RuntimeError("limit-per-cell must be in [1, 500]")
    tasks = QUICK_TASKS if args.suite == "quick" else TASKS
    configure_cuda()
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability[0] != 12:
        raise RuntimeError(
            f"RULER runner requires Blackwell, got {name} sm_{capability[0]}{capability[1]}"
        )
    data_receipt = validate_data(
        args.data_root.resolve(),
        tasks=tasks,
        lengths=lengths,
    )
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model, snapshot = load_model(checkpoint)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    completed = load_completed(examples_path)
    expected = len(tasks) * len(lengths) * args.limit_per_cell
    torch.cuda.reset_peak_memory_stats()

    with examples_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            rows = selected_rows(
                Path(data_receipt["files"][task]["path"]),
                lengths=lengths,
                limit_per_cell=args.limit_per_cell,
            )
            for row in rows:
                key = (task, int(row["max_length"]), int(row["index"]))
                if key in completed:
                    continue
                delimiter = TARGET_DELIMITER.get(task, " ")
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
                    max_new_tokens=GENERATION_TOKENS[task],
                    eos_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                prediction = tokenizer.decode(
                    output_ids[0].detach().cpu(),
                    skip_special_tokens=True,
                )
                result = {
                    "task": task,
                    "max_length": int(row["max_length"]),
                    "index": int(row["index"]),
                    "row_sha256": row_sha256(row),
                    "input_tokens": int(input_ids.numel()),
                    "generated_tokens": int(output_ids.numel()),
                    "prediction": prediction,
                    "references": row["outputs"],
                    "score": score_prediction(prediction, row["outputs"]),
                    "elapsed_seconds": elapsed,
                }
                handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                completed.add(key)
                print(
                    f"{len(completed)}/{expected} {task} "
                    f"L={row['max_length']} score={result['score']:.3f} "
                    f"seconds={elapsed:.2f}",
                    flush=True,
                )

    rows = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relevant = [
        row
        for row in rows
        if row["task"] in tasks
        and int(row["max_length"]) in lengths
        and int(row["index"]) < args.limit_per_cell
    ]
    if len(relevant) != expected:
        raise RuntimeError(
            f"RULER result count {len(relevant)} != expected {expected}"
        )
    cells: dict[str, dict[str, float]] = {}
    for task in tasks:
        cells[task] = {}
        for length in lengths:
            scores = [
                float(row["score"])
                for row in relevant
                if row["task"] == task
                and int(row["max_length"]) == length
            ]
            cells[task][str(length)] = sum(scores) / len(scores)
    mean_score = sum(
        score for task in cells.values() for score in task.values()
    ) / (len(tasks) * len(lengths))
    receipt = {
        "status": "RULER_EVALUATION_COMPLETE",
        "checkpoint": {
            "name": snapshot["name"],
            "revision": snapshot["revision"],
            "path": str(checkpoint),
            "weights": snapshot["weights"],
        },
        "data": data_receipt,
        "protocol": {
            "suite": args.suite,
            "tasks": list(tasks),
            "lengths": list(lengths),
            "limit_per_cell": args.limit_per_cell,
            "greedy": True,
            "precision": "bf16_autocast_fp32_weights",
            "attention": RULER_FLASH_ATTENTION_IMPLEMENTATION,
            "compile": False,
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
            "macro_average": mean_score,
            "examples": len(relevant),
            "examples_sha256": sha256_file(examples_path),
        },
    }
    if not math.isfinite(mean_score):
        raise RuntimeError("RULER produced a non-finite score")
    write_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
