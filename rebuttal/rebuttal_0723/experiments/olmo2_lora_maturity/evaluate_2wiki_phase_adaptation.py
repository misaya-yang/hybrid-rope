#!/usr/bin/env python3
"""Evaluate OLMo 2Wiki QA with real 4K/8K/16K autoregressive prompts."""

from __future__ import annotations

import argparse
import json
import re
import string
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
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evq_attention_restoration import (
    install_qkv_lora,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    sha256_file,
)

from .phase_adaptation import LENGTH
from .prepare_2wiki_phase_data import STATUS as DATA_STATUS
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import apply_frequency
from .evaluate_instruct_ruler_transfer import (
    apply_evq_official_yarn,
    apply_repo_fixed_ramp,
    official_yarn_config,
    validate_adapter_training_substrate,
    verify_official_yarn,
)


RESULT_STATUS = "OLMO2_2WIKI_PHASE_EVALUATION_COMPLETE_V1"
MAX_NEW_TOKENS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument(
        "--frequency",
        choices=(
            "native",
            "evq",
            "official_yarn",
            "evq_official_yarn",
            "repo_fixed_ramp",
            "evq_repo_fixed_ramp",
        ),
        required=True,
    )
    parser.add_argument(
        "--adaptation",
        choices=(
            "qkvo_answer",
            "qk_answer",
            "qkv_attention_restoration",
        ),
        default="qkvo_answer",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument(
        "--yarn-original-max-position-embeddings",
        type=int,
        default=4_096,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=(4_096, 8_192, 16_384),
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--fill-to-budget",
        action="store_true",
        help=(
            "append answer-filtered 2Wiki distractor passages before the "
            "query until the physical prompt reaches each token budget"
        ),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )
    return " ".join(text.split())


def token_f1(prediction: str, references: list[str]) -> float:
    pred = normalize_text(prediction).split()
    best = 0.0
    for reference in references:
        gold = normalize_text(reference).split()
        if not pred or not gold:
            continue
        pred_counts = {token: pred.count(token) for token in set(pred)}
        gold_counts = {token: gold.count(token) for token in set(gold)}
        overlap = sum(
            min(count, gold_counts.get(token, 0))
            for token, count in pred_counts.items()
        )
        if overlap:
            precision = overlap / len(pred)
            recall = overlap / len(gold)
            best = max(
                best,
                2 * precision * recall / (precision + recall),
            )
    return float(best)


def normalized_exact(prediction: str, references: list[str]) -> float:
    value = normalize_text(prediction)
    return float(
        any(value == normalize_text(reference) for reference in references)
    )


def truncate_user_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    budget: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    raw_ids = list(
        tokenizer(prompt, add_special_tokens=False).input_ids
    )
    candidate = raw_ids
    truncated = False
    head = len(raw_ids)
    tail = 0
    while True:
        content = tokenizer.decode(
            candidate,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if int(chat_ids.shape[1]) + MAX_NEW_TOKENS <= int(budget):
            break
        allowed = len(candidate) - max(
            1,
            int(chat_ids.shape[1]) + MAX_NEW_TOKENS - int(budget),
        )
        if allowed <= 64:
            raise RuntimeError(f"2Wiki prompt cannot fit L{budget}")
        head = allowed // 2
        tail = allowed - head
        candidate = raw_ids[:head] + raw_ids[-tail:]
        truncated = True
    if "Question:" not in content or "Answer:" not in content:
        raise RuntimeError("middle truncation removed the QA query")
    return chat_ids, {
        "budget": int(budget),
        "raw_user_tokens": len(raw_ids),
        "kept_user_tokens": len(candidate),
        "chat_input_tokens": int(chat_ids.shape[1]),
        "truncated": truncated,
        "kept_head_tokens": head if truncated else len(raw_ids),
        "kept_tail_tokens": tail if truncated else 0,
    }


def split_official_prompt(prompt: str) -> tuple[str, str, str]:
    prefix_marker = "The following are given passages.\n"
    suffix_marker = (
        "\n\nAnswer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question:"
    )
    prefix_end = prompt.find(prefix_marker)
    suffix_start = prompt.rfind(suffix_marker)
    if prefix_end < 0 or suffix_start < 0:
        raise RuntimeError("2Wiki official prompt boundary drift")
    context_start = prefix_end + len(prefix_marker)
    if not context_start < suffix_start:
        raise RuntimeError("2Wiki context boundary drift")
    return (
        prompt[:context_start],
        prompt[context_start:suffix_start],
        prompt[suffix_start:],
    )


def answer_filtered_filler(
    *,
    row_index: int,
    rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    references = [
        normalize_text(value)
        for value in rows[row_index]["answers"]
        if normalize_text(value)
    ]
    references_to_filter = [
        reference for reference in references if reference not in {"yes", "no"}
    ]
    pieces: list[str] = []
    identities: list[str] = []
    for distance in range(1, len(rows)):
        candidate = rows[(row_index + distance) % len(rows)]
        _, context, _ = split_official_prompt(str(candidate["prompt"]))
        normalized_context = normalize_text(context)
        padded_context = f" {normalized_context} "
        if any(
            f" {reference} " in padded_context
            for reference in references_to_filter
        ):
            continue
        pieces.append(
            f"\n\nDistractor passage set {len(pieces) + 1}:\n{context}"
        )
        identities.append(str(candidate["qa_identity_sha256"]))
        if len(pieces) == 4:
            break
    if not pieces:
        raise RuntimeError("no answer-filtered 2Wiki distractors available")
    return "".join(pieces), identities


def fill_user_prompt_to_budget(
    *,
    tokenizer: Any,
    prompt: str,
    budget: int,
    filler: str,
    filler_identities: list[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_ids, base_receipt = truncate_user_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        budget=budget,
    )
    if base_receipt["truncated"]:
        return base_ids, {
            **base_receipt,
            "filled": False,
            "filler_tokens": 0,
            "filler_candidate_qa_identities": [],
            "target_chat_input_tokens": int(budget) - MAX_NEW_TOKENS,
        }
    prefix, context, suffix = split_official_prompt(prompt)
    filler_ids = list(
        tokenizer(filler, add_special_tokens=False).input_ids
    )
    if not filler_ids:
        raise RuntimeError("empty 2Wiki distractor tokenization")
    target = int(budget) - MAX_NEW_TOKENS
    low = 0
    high = len(filler_ids)
    best_ids = base_ids
    best_count = 0
    best_content = prompt
    while low <= high:
        middle = (low + high) // 2
        filler_text = tokenizer.decode(
            filler_ids[:middle],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        content = prefix + context + filler_text + suffix
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        length = int(chat_ids.shape[1])
        if length <= target:
            best_ids = chat_ids
            best_count = middle
            best_content = content
            low = middle + 1
        else:
            high = middle - 1
    unused = target - int(best_ids.shape[1])
    if unused > 2:
        raise RuntimeError(
            f"2Wiki distractor fill left {unused} unused tokens "
            f"at L{budget}"
        )
    if "Question:" not in best_content or "Answer:" not in best_content:
        raise RuntimeError("2Wiki distractor fill removed the query")
    return best_ids, {
        "budget": int(budget),
        "raw_user_tokens": base_receipt["raw_user_tokens"],
        "kept_user_tokens": None,
        "chat_input_tokens": int(best_ids.shape[1]),
        "truncated": False,
        "filled": True,
        "filler_tokens": int(best_count),
        "filler_candidate_qa_identities": filler_identities,
        "target_chat_input_tokens": target,
    }


def load_rows(root: Path, limit: int) -> tuple[dict[str, Any], list[Any]]:
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != DATA_STATUS
        or manifest["evaluation"]["dataset"]
        != "THUDM/LongBench:2wikimqa"
    ):
        raise RuntimeError("2Wiki evaluation data contract drift")
    path = root / "evaluation_rows.jsonl"
    if sha256_file(path) != manifest["files"][
        "evaluation_rows.jsonl"
    ]["sha256"]:
        raise RuntimeError("2Wiki evaluation row hash drift")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 1 <= int(limit) <= len(rows):
        raise ValueError("invalid 2Wiki evaluation limit")
    return manifest, rows[: int(limit)]


def load_completed(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["budget"]), int(row["local_index"]))
            if key in completed:
                raise RuntimeError(f"duplicate completed 2Wiki row: {key}")
            completed[key] = row
    return completed


def main() -> None:
    args = parse_args()
    budgets = tuple(int(value) for value in args.budgets)
    if (
        budgets != tuple(sorted(set(budgets)))
        or any(value not in {LENGTH, 2 * LENGTH, 4 * LENGTH} for value in budgets)
    ):
        raise ValueError("budgets must be an ordered subset of 4K/8K/16K")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    run_manifest_path = output / "run_manifest.json"

    checkpoint = args.checkpoint.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, args.checkpoint_ready_receipt.resolve()
    )
    data_root = args.data_root.resolve()
    data_manifest, rows = load_rows(data_root, int(args.limit))
    adapter_argument = (
        None if args.adapter is None else args.adapter.resolve()
    )
    run_manifest = {
        "status": "OLMO2_2WIKI_PHASE_EVAL_RUN_V1",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "frequency_helper_sha256": sha256_file(
            Path(__file__).resolve().parent
            / "evaluate_instruct_ruler_transfer.py"
        ),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_ready_receipt_sha256": sha256_file(
            args.checkpoint_ready_receipt.resolve()
        ),
        "data_manifest_sha256": sha256_file(
            data_root / "manifest.json"
        ),
        "evaluation_rows_sha256": sha256_file(
            data_root / "evaluation_rows.jsonl"
        ),
        "adapter_sha256": (
            None
            if adapter_argument is None
            else sha256_file(adapter_argument)
        ),
        "role": str(args.role),
        "frequency": str(args.frequency),
        "adaptation": (
            str(args.adaptation)
            if adapter_argument is not None
            else None
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "budgets": list(budgets),
        "limit": int(args.limit),
        "fill_to_budget": bool(args.fill_to_budget),
        "yarn_factor": (
            float(args.yarn_factor)
            if "yarn" in str(args.frequency)
            or "fixed_ramp" in str(args.frequency)
            else None
        ),
        "yarn_original_max_position_embeddings": (
            int(args.yarn_original_max_position_embeddings)
            if "official_yarn" in str(args.frequency)
            else None
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy": True,
    }
    if examples_path.exists() != run_manifest_path.exists():
        raise RuntimeError(
            "2Wiki examples and run manifest must be resumed together"
        )
    if run_manifest_path.is_file():
        observed = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if observed != run_manifest:
            raise RuntimeError("2Wiki resume run-manifest drift")
    else:
        atomic_json(run_manifest_path, run_manifest)
    completed_rows = load_completed(examples_path)
    expected_keys = {
        (budget, local_index)
        for budget in budgets
        for local_index in range(len(rows))
    }
    if not set(completed_rows).issubset(expected_keys):
        raise RuntimeError("2Wiki completed rows escape registered matrix")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer EOS is unavailable")
    model_config = None
    if args.frequency in {"official_yarn", "evq_official_yarn"}:
        model_config = official_yarn_config(
            checkpoint,
            factor=float(args.yarn_factor),
            original_max_position_embeddings=int(
                args.yarn_original_max_position_embeddings
            ),
        )
    model = load_model(checkpoint, config=model_config)
    if args.frequency in {"official_yarn", "evq_official_yarn"}:
        official_yarn = verify_official_yarn(model, model_config)
        frequency = (
            official_yarn
            if args.frequency == "official_yarn"
            else apply_evq_official_yarn(
                model, model_config, official_yarn
            )
        )
    elif args.frequency in {
        "repo_fixed_ramp",
        "evq_repo_fixed_ramp",
    }:
        frequency = apply_repo_fixed_ramp(
            model,
            substrate=(
                "native"
                if args.frequency == "repo_fixed_ramp"
                else "evq"
            ),
            factor=float(args.yarn_factor),
        )
    else:
        frequency = apply_frequency(model, args.frequency)
    adapter_path = None
    adapter_metadata = None
    if adapter_argument is not None:
        if args.adaptation == "qkv_attention_restoration":
            install_qkv_lora(
                model,
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
            readout = None
        else:
            readout = install_adaptation(
                model,
                str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        if readout is not None:
            raise RuntimeError("2Wiki evaluation forbids a readout")
        adapter_path = adapter_argument
        adapter_metadata = load_adapter(adapter_path, model, None)
        if args.frequency in {"official_yarn", "repo_fixed_ramp"}:
            validate_adapter_training_substrate(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency_name="native",
                frequency_sha256=tensor_sha256(
                    endpoint_geo_inv_freq()
                ),
                adaptation=str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        elif args.frequency in {
            "evq_official_yarn",
            "evq_repo_fixed_ramp",
        }:
            validate_adapter_training_substrate(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency_name="evq",
                frequency_sha256=tensor_sha256(
                    endpoint_evq_inv_freq()
                ),
                adaptation=str(args.adaptation),
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        else:
            expected_adapter = {
                "base_checkpoint_sha256": checkpoint_digest,
                "frequency": args.frequency,
                "frequency_sha256_float32": frequency[
                    "active_sha256_float32"
                ],
                "adaptation": str(args.adaptation),
                "rank": int(args.rank),
                "alpha": float(args.alpha),
                "training_sequence_length": LENGTH,
            }
            for name, expected in expected_adapter.items():
                if adapter_metadata.get(name) != expected:
                    raise RuntimeError(
                        f"adapter metadata drift for {name}: "
                        f"{adapter_metadata.get(name)!r} != {expected!r}"
                    )
    runtime = configure_cuda()
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()

    cells: dict[str, dict[str, Any]] = {}
    with examples_path.open("a", encoding="utf-8") as handle:
        for budget in budgets:
            sums = {
                "token_f1": 0.0,
                "normalized_exact": 0.0,
                "terminal_eos": 0.0,
                "empty_prediction": 0.0,
                "elapsed_seconds": 0.0,
                "input_tokens": 0,
                "truncated": 0,
                "filled": 0,
            }
            for local_index, row in enumerate(rows):
                key = (budget, local_index)
                if key in completed_rows:
                    result = completed_rows[key]
                    if (
                        result.get("source_id") != row["source_id"]
                        or result.get("source_row_sha256")
                        != row["source_row_sha256"]
                        or result.get("qa_identity_sha256")
                        != row["qa_identity_sha256"]
                        or result.get("references")
                        != [str(value) for value in row["answers"]]
                        or result.get("role") != str(args.role)
                    ):
                        raise RuntimeError(
                            f"completed 2Wiki row identity drift: {key}"
                        )
                else:
                    if args.fill_to_budget:
                        filler, filler_identities = (
                            answer_filtered_filler(
                                row_index=local_index,
                                rows=rows,
                            )
                        )
                        input_ids, truncation = (
                            fill_user_prompt_to_budget(
                                tokenizer=tokenizer,
                                prompt=str(row["prompt"]),
                                budget=budget,
                                filler=filler,
                                filler_identities=filler_identities,
                            )
                        )
                    else:
                        input_ids, truncation = truncate_user_prompt(
                            tokenizer=tokenizer,
                            prompt=str(row["prompt"]),
                            budget=budget,
                        )
                    input_ids = input_ids.to("cuda")
                    started = time.perf_counter()
                    output_ids = greedy_generate(
                        model,
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        eos_token_id=int(tokenizer.eos_token_id),
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    generated = [
                        int(value)
                        for value in output_ids[0].detach().cpu().tolist()
                    ]
                    terminal_eos = float(
                        bool(generated)
                        and generated[-1]
                        == int(tokenizer.eos_token_id)
                    )
                    prediction = tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    references = [
                        str(value) for value in row["answers"]
                    ]
                    f1 = token_f1(prediction, references)
                    exact = normalized_exact(prediction, references)
                    result = {
                        "role": str(args.role),
                        "budget": budget,
                        "local_index": local_index,
                        "source_id": row["source_id"],
                        "source_row_sha256": row["source_row_sha256"],
                        "qa_identity_sha256": row[
                            "qa_identity_sha256"
                        ],
                        "references": references,
                        "prediction": prediction,
                        "generated_token_ids": generated,
                        "generated_tokens": len(generated),
                        "token_f1": f1,
                        "normalized_exact": exact,
                        "terminal_eos": terminal_eos,
                        "empty_prediction": float(not prediction),
                        "truncation": truncation,
                        "elapsed_seconds": elapsed,
                    }
                    handle.write(
                        json.dumps(
                            result, ensure_ascii=False, sort_keys=True
                        )
                        + "\n"
                    )
                    handle.flush()
                    print(
                        f"{args.role} L={budget} "
                        f"{local_index + 1}/{len(rows)} "
                        f"f1={f1:.3f} exact={exact:.0f} "
                        f"eos={terminal_eos:.0f}",
                        flush=True,
                    )
                truncation = result["truncation"]
                sums["token_f1"] += float(result["token_f1"])
                sums["normalized_exact"] += float(
                    result["normalized_exact"]
                )
                sums["terminal_eos"] += float(result["terminal_eos"])
                sums["empty_prediction"] += float(
                    result["empty_prediction"]
                )
                sums["elapsed_seconds"] += float(
                    result["elapsed_seconds"]
                )
                sums["input_tokens"] += int(
                    truncation["chat_input_tokens"]
                )
                sums["truncated"] += int(truncation["truncated"])
                sums["filled"] += int(truncation.get("filled", False))
            cells[str(budget)] = {
                "examples": len(rows),
                "mean_token_f1": sums["token_f1"] / len(rows),
                "normalized_exact": (
                    sums["normalized_exact"] / len(rows)
                ),
                "terminal_eos": sums["terminal_eos"] / len(rows),
                "empty_prediction": (
                    sums["empty_prediction"] / len(rows)
                ),
                "mean_elapsed_seconds": (
                    sums["elapsed_seconds"] / len(rows)
                ),
                "mean_input_tokens": sums["input_tokens"] / len(rows),
                "truncated_examples": int(sums["truncated"]),
                "filled_examples": int(sums["filled"]),
            }

    result = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "LongBench-v1 2wikimqa rows and official prompt with greedy "
            "generation, normalized QA token-F1 and exact. When "
            "fill_to_budget is enabled, answer-filtered 2Wiki distractor "
            "contexts make the physical prompts reach 4K/8K/16K; this "
            "is a deterministic task-family long-QA protocol rather than "
            "the unmodified LongBench leaderboard protocol."
        ),
        "role": str(args.role),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "frequency_helper_sha256": sha256_file(
            Path(__file__).resolve().parent
            / "evaluate_instruct_ruler_transfer.py"
        ),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "data": {
            "path": str(data_root),
            "manifest_sha256": sha256_file(
                data_root / "manifest.json"
            ),
            "evaluation_rows_sha256": sha256_file(
                data_root / "evaluation_rows.jsonl"
            ),
            "source": data_manifest["evaluation"],
        },
        "adapter": {
            "path": (
                None if adapter_path is None else str(adapter_path)
            ),
            "sha256": (
                None
                if adapter_path is None
                else sha256_file(adapter_path)
            ),
            "metadata": adapter_metadata,
        },
        "frequency": frequency,
        "protocol": {
            "budgets": list(budgets),
            "limit": int(args.limit),
            "max_new_tokens": MAX_NEW_TOKENS,
            "greedy": True,
            "chat_template": "checkpoint_native",
            "truncation": "middle",
            "fill_to_budget": bool(args.fill_to_budget),
            "adaptation": (
                str(args.adaptation)
                if adapter_path is not None
                else None
            ),
            "yarn_factor": (
                float(args.yarn_factor)
                if "yarn" in str(args.frequency)
                or "fixed_ramp" in str(args.frequency)
                else None
            ),
            "yarn_original_max_position_embeddings": (
                int(args.yarn_original_max_position_embeddings)
                if "official_yarn" in str(args.frequency)
                else None
            ),
            "distractor_answer_filter": (
                "normalized reference absent as a complete phrase from each "
                "appended context; yes/no references are exempt because "
                "their lexical occurrence is not answer leakage"
                if args.fill_to_budget
                else None
            ),
        },
        "results": {
            "cells": cells,
            "examples": len(rows) * len(budgets),
            "examples_sha256": sha256_file(examples_path),
            "macro_token_f1": sum(
                cell["mean_token_f1"] for cell in cells.values()
            )
            / len(cells),
            "macro_normalized_exact": sum(
                cell["normalized_exact"] for cell in cells.values()
            )
            / len(cells),
        },
        "runtime": {
            **runtime,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
    }
    atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
