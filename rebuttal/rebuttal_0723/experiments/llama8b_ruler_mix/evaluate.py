#!/usr/bin/env python3
"""Resumable Flash-only 13-task RULER evaluation for Llama-3-8B."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from experiments.lora_evq_v2.train_evq_lora import (
    inject_inv_freq,
    load_frequency_artifact,
    verify_model_inv_freq,
)

from .common import (
    DEFAULT_EVAL_LENGTHS,
    EVAL_SOURCE_STATUS,
    EVAL_STATUS,
    GENERATION_TOKENS,
    OFFICIAL_METRIC,
    TASKS,
    TRAIN_LENGTH,
    TRAIN_SOURCE_STATUS,
    append_jsonl,
    atomic_json,
    configure_cuda,
    load_jsonl,
    official_score,
    row_sha256,
    sha256_file,
)


ATTENTION_BACKEND = "llama8b_ruler_flash_gqa_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("evq_cosh", "native_geo", "native_base"),
        required=True,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[8_192, 16_384, 32_768],
    )
    parser.add_argument("--limit-per-cell", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--required-gpu-substring",
        default="RTX PRO 6000",
    )
    return parser.parse_args()


def ruler_causal_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    if attention_mask is not None:
        raise RuntimeError("Flash-only RULER evaluation forbids padding masks")
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
    if attention_mask is not None:
        raise RuntimeError("Flash-only RULER evaluation received a mask")
    query_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    if query_length != key_length and query_length != 1:
        raise RuntimeError(
            "RULER attention admits only full prefill or one-token decode"
        )
    groups = int(getattr(module, "num_key_value_groups", 1))
    if groups < 1 or query.shape[1] != key.shape[1] * groups:
        raise RuntimeError("RULER attention received invalid GQA geometry")
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(dropout),
        scale=scaling,
        is_causal=query_length > 1,
        enable_gqa=groups > 1,
    )
    return output.transpose(1, 2).contiguous(), None


def configure_ruler_attention(model: Any) -> None:
    AttentionInterface.register(ATTENTION_BACKEND, ruler_flash_forward)
    AttentionMaskInterface.register(ATTENTION_BACKEND, ruler_causal_mask)
    model.config._attn_implementation = ATTENTION_BACKEND


def qa_query(row: dict[str, Any]) -> str:
    tail = str(row["input"]).rsplit("Question:", 1)[-1]
    return tail.split("<|eot_id|>", 1)[0].strip()


def validate_sources(
    root: Path,
    *,
    tasks: tuple[str, ...],
    lengths: tuple[int, ...],
) -> dict[str, Any]:
    train_path = root / "train_manifest.json"
    eval_path = root / "eval_manifest.json"
    train_manifest = json.loads(train_path.read_text(encoding="utf-8"))
    eval_manifest = json.loads(eval_path.read_text(encoding="utf-8"))
    if train_manifest.get("status") != TRAIN_SOURCE_STATUS:
        raise RuntimeError("RULER training-source status drift")
    if eval_manifest.get("status") != EVAL_SOURCE_STATUS:
        raise RuntimeError("RULER evaluation-source status drift")
    if tuple(eval_manifest["protocol"]["evaluation_lengths"]) != (
        DEFAULT_EVAL_LENGTHS
    ):
        raise RuntimeError("RULER evaluation-length manifest drift")
    for manifest in (train_manifest, eval_manifest):
        for record in manifest["files"].values():
            candidate = root / record["path"]
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(record["size_bytes"])
                or sha256_file(candidate) != record["sha256"]
            ):
                raise RuntimeError(f"RULER source drift: {candidate}")
    separation = {}
    for task in tasks:
        training = load_jsonl(
            root / "train" / f"L{TRAIN_LENGTH}" / task / "test.jsonl"
        )
        train_hashes = {row_sha256(row) for row in training}
        train_queries = (
            {qa_query(row) for row in training}
            if task in {"qa_1", "qa_2"}
            else set()
        )
        task_record = {}
        for length in lengths:
            evaluation = load_jsonl(
                root / "eval" / f"L{length}" / task / "test.jsonl"
            )
            exact = train_hashes & {
                row_sha256(row) for row in evaluation
            }
            if exact:
                raise RuntimeError(
                    f"training/evaluation row overlap: {task} L{length}"
                )
            cell = {"exact_row_overlap": 0}
            if task in {"qa_1", "qa_2"}:
                query_overlap = train_queries & {
                    qa_query(row) for row in evaluation
                }
                if query_overlap:
                    raise RuntimeError(
                        f"training/evaluation QA query overlap: "
                        f"{task} L{length}"
                    )
                cell["qa_query_overlap"] = 0
            task_record[f"L{length}"] = cell
        separation[task] = task_record
    return {
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "train_manifest_sha256": sha256_file(train_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "separation": separation,
    }


def adapter_receipt(root: Path) -> dict[str, Any]:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "custom_inv_freq.pt",
    )
    files = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    for optional in ("experiment_meta.json", "result.json"):
        path = root / optional
        if path.is_file():
            files[optional] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {"path": str(root), "files": files}


def terminator_ids(model: Any, tokenizer: Any) -> set[int]:
    values: set[int] = set()
    candidates = (
        tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (tuple, list)):
            values.update(int(value) for value in candidate)
        else:
            values.add(int(candidate))
    return values


def greedy_generate(
    *,
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    terminators: set[int],
) -> torch.Tensor:
    base = (
        model.get_base_model()
        if isinstance(model, PeftModel)
        else model
    )
    backbone = base.model
    lm_head = base.lm_head
    generated: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=True,
            return_dict=True,
        )
        logits = lm_head(outputs.last_hidden_state[:, -1, :])
        past = outputs.past_key_values
        del outputs
        for _ in range(max_new_tokens):
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            if int(next_token.item()) in terminators:
                break
            outputs = backbone(
                input_ids=next_token[:, None],
                attention_mask=None,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = lm_head(outputs.last_hidden_state[:, -1, :])
            del outputs
    if not generated:
        return torch.empty(0, dtype=torch.long)
    return torch.stack(generated, dim=1).squeeze(0).cpu()


def normalized_exact(prediction: str, references: list[Any]) -> float:
    normalized = " ".join(prediction.lower().strip().split())
    return max(
        float(normalized == " ".join(str(value).lower().strip().split()))
        for value in references
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(int(row["length"]), str(row["task"]))].append(row)
    cell_results = {}
    for (length, task), values in sorted(cells.items()):
        key = f"L{length}/{task}"
        cell_results[key] = {
            "length": length,
            "task": task,
            "n": len(values),
            "official_score": sum(
                float(value["official_score"]) for value in values
            )
            / len(values),
            "normalized_exact": sum(
                float(value["normalized_exact"]) for value in values
            )
            / len(values),
            "mean_prompt_tokens": sum(
                int(value["prompt_tokens"]) for value in values
            )
            / len(values),
            "elapsed_seconds": sum(
                float(value["elapsed_seconds"]) for value in values
            ),
        }
    by_length = {}
    for length in sorted({int(row["length"]) for row in rows}):
        length_cells = [
            value
            for value in cell_results.values()
            if int(value["length"]) == length
        ]
        by_length[f"L{length}"] = {
            "tasks": len(length_cells),
            "macro_official_score": sum(
                float(value["official_score"]) for value in length_cells
            )
            / len(length_cells),
            "macro_normalized_exact": sum(
                float(value["normalized_exact"]) for value in length_cells
            )
            / len(length_cells),
        }
    return {"cells": cell_results, "by_length": by_length}


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    adapter = args.adapter.resolve() if args.adapter is not None else None
    sources = args.sources.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    tasks = tuple(args.tasks)
    lengths = tuple(int(value) for value in args.lengths)
    if not set(tasks).issubset(TASKS):
        raise ValueError("unknown RULER task")
    if not set(lengths).issubset(DEFAULT_EVAL_LENGTHS):
        raise ValueError("unregistered evaluation length")
    if int(args.limit_per_cell) not in (20, 100):
        raise ValueError("registered cell sizes are 20-screen or 100-formal")
    if args.method == "native_base" and adapter is not None:
        raise ValueError("native_base must not receive an adapter")
    if args.method != "native_base" and adapter is None:
        raise ValueError(f"{args.method} requires an adapter")
    if output.exists():
        result = json.loads(
            (output / "result.json").read_text(encoding="utf-8")
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if incomplete.exists() and not args.resume:
        raise FileExistsError(
            f"{incomplete} exists; pass --resume to continue"
        )
    incomplete.mkdir(parents=True, exist_ok=True)

    source_receipt = validate_sources(
        sources,
        tasks=tasks,
        lengths=lengths,
    )
    adapter_files = (
        adapter_receipt(adapter)
        if adapter is not None
        else None
    )
    runtime = configure_cuda()
    if args.required_gpu_substring not in runtime["gpu_name"]:
        raise RuntimeError(
            f"wrong GPU for registered evaluation: {runtime['gpu_name']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    configure_ruler_attention(base)
    if args.method == "native_base":
        model = base
        frequency_receipt = {
            "method": "native_base",
            "source": "checkpoint_config",
            "rope_theta": float(base.config.rope_theta),
        }
        frequency_verification = {
            "model_type": str(base.config.model_type),
            "max_position_embeddings": int(
                base.config.max_position_embeddings
            ),
            "rope_theta": float(base.config.rope_theta),
        }
    else:
        assert adapter is not None
        inv_freq, _, frequency_receipt = load_frequency_artifact(
            adapter / "custom_inv_freq.pt",
            expected_method=args.method,
        )
        inject_inv_freq(base, inv_freq)
        verify_model_inv_freq(base, inv_freq)
        model = PeftModel.from_pretrained(
            base,
            adapter,
            is_trainable=False,
        )
        inject_inv_freq(model, inv_freq)
        frequency_verification = verify_model_inv_freq(model, inv_freq)
    model.eval()
    model.config.use_cache = True
    terminators = terminator_ids(model, tokenizer)

    predictions_path = incomplete / "predictions.jsonl"
    completed: set[tuple[int, str, int]] = set()
    predictions: list[dict[str, Any]] = []
    if predictions_path.is_file():
        predictions = load_jsonl(predictions_path)
        completed = {
            (int(row["length"]), str(row["task"]), int(row["row_index"]))
            for row in predictions
        }

    for length in lengths:
        for task in tasks:
            path = sources / "eval" / f"L{length}" / task / "test.jsonl"
            rows = load_jsonl(path)[: int(args.limit_per_cell)]
            if len(rows) != int(args.limit_per_cell):
                raise RuntimeError(f"insufficient evaluation rows: {path}")
            for row_index, row in enumerate(rows):
                key = (length, task, row_index)
                if key in completed:
                    continue
                prompt_text = str(row["input"]) + str(
                    row.get("answer_prefix", "")
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids
                if prompt_ids.shape[1] >= length:
                    raise RuntimeError(
                        f"prompt consumes full budget: {task} L{length}"
                    )
                prompt_ids = prompt_ids.to("cuda")
                torch.cuda.synchronize()
                started = time.perf_counter()
                generated = greedy_generate(
                    model=model,
                    input_ids=prompt_ids,
                    max_new_tokens=GENERATION_TOKENS[task],
                    terminators=terminators,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                prediction = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                )
                references = list(row["outputs"])
                record = {
                    "length": length,
                    "task": task,
                    "row_index": row_index,
                    "source_index": int(row["index"]),
                    "source_row_sha256": row_sha256(row),
                    "prompt_tokens": int(prompt_ids.shape[1]),
                    "generated_tokens": int(generated.numel()),
                    "prediction": prediction,
                    "references": [str(value) for value in references],
                    "official_metric": OFFICIAL_METRIC[task],
                    "official_score": official_score(
                        prediction,
                        references,
                        OFFICIAL_METRIC[task],
                    ),
                    "normalized_exact": normalized_exact(
                        prediction,
                        references,
                    ),
                    "elapsed_seconds": elapsed,
                }
                append_jsonl(predictions_path, record)
                predictions.append(record)
                completed.add(key)
                del prompt_ids, generated

    expected = len(lengths) * len(tasks) * int(args.limit_per_cell)
    if len(predictions) != expected:
        raise RuntimeError(
            f"evaluation completeness drift: {len(predictions)} != {expected}"
        )
    summary = summarize(predictions)
    result = {
        "status": EVAL_STATUS,
        "method": args.method,
        "evidence_boundary": (
            "autoregressive official RULER-family evaluation of the "
            "unmodified instruction checkpoint; no adapter or "
            "RULER-family adaptation"
            if args.method == "native_base"
            else (
                "autoregressive official RULER-family evaluation after "
                "RULER-family supervised adaptation; not zero-shot"
            )
        ),
        "protocol": {
            "tasks": list(tasks),
            "lengths": list(lengths),
            "rows_per_cell": int(args.limit_per_cell),
            "decoding": "greedy",
            "attention": "PyTorch Flash-only SDPA with GQA",
            "metric": (
                "official string_match_all for retrieval/counting and "
                "string_match_part for QA"
            ),
        },
        "summary": summary,
        "artifacts": {
            "adapter": adapter_files,
            "checkpoint": {
                "config_sha256": sha256_file(
                    checkpoint / "config.json"
                ),
                "index_sha256": sha256_file(
                    checkpoint / "model.safetensors.index.json"
                ),
            },
            "train_sources_manifest_sha256": source_receipt[
                "train_manifest_sha256"
            ],
            "eval_sources_manifest_sha256": source_receipt[
                "eval_manifest_sha256"
            ],
            "train_source_status": source_receipt["train_manifest"][
                "status"
            ],
            "eval_source_status": source_receipt["eval_manifest"][
                "status"
            ],
            "train_eval_separation": source_receipt["separation"],
            "predictions_sha256": sha256_file(predictions_path),
            "prediction_rows": len(predictions),
        },
        "frequency": {
            "artifact": frequency_receipt,
            "verification": frequency_verification,
        },
        "runtime": runtime,
    }
    atomic_json(incomplete / "result.json", result)
    incomplete.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
