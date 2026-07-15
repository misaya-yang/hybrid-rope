#!/usr/bin/env python3
"""Checkpoint-only EVQ attention-score and sparse-conversion experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.lora_evq_v2.eval_official_yarn_capability import (
    adapter_artifact_receipt,
    load_capability_suite,
    score_capability_prediction,
    score_generation_metrics,
    sha256_file,
    validate_adapter_identity,
)
from experiments.lora_evq_v2.prepare_legacy_model_manifest import (
    validate_model_manifest,
)
from experiments.lora_evq_v2.prepare_positional_distill_data import (
    tokenizer_source_fingerprint,
)
from experiments.lora_evq_v2.prepare_seed42_capability_data import (
    _repeat_to_length,
    build_passkey_examples,
    load_jsonl,
)
from experiments.lora_evq_v2.train_positional_distill import causal_backbone
from experiments.lora_evq_v2.train_evq_lora import (
    build_training_inv_freq,
    compute_evq_cosh_inv_freq,
    find_rotary_modules,
    inject_inv_freq,
    load_frequency_artifact,
    _load_strict_legacy_data,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)


LEGACY_EXAMPLE_SCHEMA = "evq_cosh.seed42_capability_example.v1"
LEGACY_MANIFEST_SCHEMA = "evq_cosh.seed42_capability_manifest.v1"
PASSKEY_SHA256 = "21f365daed1b77e06b0a870ccb20f4965cba454c802f48dd48825c8f7ff2990d"
PASSKEY_SIZE_BYTES = 26_083_876
PASSKEY_ROWS = 300
PASSKEY_RECEIPT_SCHEMA = "evq_cosh.sparse_conversion_passkey.v1"
PHASE0_SCHEMA = "evq_cosh.lora_sparse_conversion_phase0.v1"
PHASE0_SUMMARY_SCHEMA = "evq_cosh.lora_sparse_conversion_phase0_summary.v1"
PHASE1_SCHEMA = "evq_cosh.lora_sparse_conversion_phase1.v1"
PHASE1_SUMMARY_SCHEMA = "evq_cosh.lora_sparse_conversion_phase1_summary.v1"
RAW_CAPABILITY_SCHEMA = "evq_cosh.lora_raw_capability_eval.v1"
CANARY_SCHEMA = "evq_cosh.lora_source_dependence_canary.v1"
CANARY_SUMMARY_SCHEMA = "evq_cosh.lora_source_dependence_canary_summary.v1"
READOUT_TRACE_SCHEMA = "evq_cosh.readout_conversion_trace.v1"
READOUT_TRACE_MANIFEST_SCHEMA = "evq_cosh.readout_conversion_trace_manifest.v1"
ASSOCIATION_SWAP_TRACE_SCHEMA = "evq_cosh.readout_association_swap_trace.v1"
ASSOCIATION_SWAP_MANIFEST_SCHEMA = "evq_cosh.readout_association_swap_manifest.v1"
ASSOCIATION_SWAP_CASE_SCHEMA = "evq_cosh.readout_association_swap_case.v1"
ASSOCIATION_SWAP_CASE_MANIFEST_SCHEMA = (
    "evq_cosh.readout_association_swap_case_manifest.v1"
)
ATTENTION_IMPL = "evq_exact_block"
SPARSE_CONFIG = {
    "block_size": 128,
    "top_blocks": 16,
    "local_window": 1024,
    "sink_tokens": 4,
}
MODEL_CONTRACT = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
}
PASSKEY_QUERY = "\n\nWhat is the retrieval passkey? Return only the passkey."
CHAT_WRAPPER_MARKER = "EVQ_CHAT_WRAPPER_BOUNDARY_314159"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_sha256(prompt_ids: Sequence[int]) -> str:
    payload = json.dumps(list(prompt_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _script_sha256() -> str:
    return sha256_file(Path(__file__))


def _parse_ints(value: str | Iterable[int]) -> tuple[int, ...]:
    pieces = value.split(",") if isinstance(value, str) else value
    parsed = tuple(int(piece) for piece in pieces)
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        raise ValueError("needle tokenization is empty")
    return [
        index
        for index in range(len(values) - len(needle) + 1)
        if list(values[index : index + len(needle)]) == list(needle)
    ]


def _answer_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer(answer, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        raise ValueError("answer tokenization produced no tokens")
    return [int(token_id) for token_id in ids]


def _needle_span(tokenizer: Any, row: Mapping[str, Any]) -> tuple[int, int]:
    if row.get("task") != "passkey_retrieval" or len(row.get("answers", [])) != 1:
        raise ValueError("Phase 0 requires one deterministic passkey answer")
    needle_ids = _answer_ids(
        tokenizer,
        f"\nThe retrieval passkey is {row['answers'][0]}. Remember this exact passkey.\n",
    )
    matches = _find_subsequence(row["prompt_ids"], needle_ids)
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('example_id')} has {len(matches)} exact needle spans; expected one"
        )
    return matches[0], matches[0] + len(needle_ids)


def _answer_span(tokenizer: Any, row: Mapping[str, Any]) -> tuple[int, int]:
    answer_ids = _answer_ids(tokenizer, str(row["answers"][0]))
    matches = _find_subsequence(row["prompt_ids"], answer_ids)
    if len(matches) != 1:
        raise ValueError(
            f"{row.get('example_id')} has {len(matches)} exact answer spans; expected one"
        )
    return matches[0], matches[0] + len(answer_ids)


def build_association_swap_pair(
    tokenizer: Any,
    *,
    key_a_ids: Sequence[int],
    key_b_ids: Sequence[int],
    answer_a_ids: Sequence[int],
    answer_b_ids: Sequence[int],
    filler_ids: Sequence[int],
    first_token_frequency_buckets: Mapping[int, int],
    target_length: int = 16384,
    depth_percent: float = 50.0,
    block_size: int = 128,
    slot_gap_blocks: int = 2,
    mirror: bool = False,
) -> dict[str, Any]:
    """Build a token-multiset-matched two-key association-swap pair."""

    def tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
        normalized = tuple(int(value) for value in values)
        if not normalized or any(value < 0 for value in normalized):
            raise ValueError(f"{name} must contain non-negative token IDs")
        return normalized

    key_a = tokens(key_a_ids, "key_a_ids")
    key_b = tokens(key_b_ids, "key_b_ids")
    answer_a = tokens(answer_a_ids, "answer_a_ids")
    answer_b = tokens(answer_b_ids, "answer_b_ids")
    filler = tokens(filler_ids, "filler_ids")
    if len(key_a) != len(key_b) or key_a == key_b:
        raise ValueError("association keys must be distinct and token-length matched")
    if len(answer_a) != len(answer_b) or answer_a == answer_b:
        raise ValueError("association answers must be distinct and token-length matched")
    bucket_a = first_token_frequency_buckets.get(answer_a[0])
    bucket_b = first_token_frequency_buckets.get(answer_b[0])
    if bucket_a is None or bucket_b is None or int(bucket_a) != int(bucket_b):
        raise ValueError("answer first tokens must share a registered frequency bucket")
    if target_length <= 0 or block_size <= 0 or slot_gap_blocks <= 0:
        raise ValueError("target length, block size, and slot gap must be positive")
    if not 0.0 < float(depth_percent) < 100.0:
        raise ValueError("association depth must be strictly between 0 and 100")

    literal = lambda value: tuple(_answer_ids(tokenizer, value))
    prefix = literal("Inspect the archive and return only the value for the target entry.\n")
    record_prefix = literal("\nEntry ")
    record_infix = literal(" has exact value ")
    record_suffix = literal(".\n")
    query_prefix = literal("\nTarget entry ")
    query_infix = literal("; control entry ")
    query_suffix = literal(". Return only the target value.")

    def record(key: tuple[int, ...], answer: tuple[int, ...]) -> tuple[int, ...]:
        return record_prefix + key + record_infix + answer + record_suffix

    query_a = query_prefix + key_a + query_infix + key_b + query_suffix
    query_b = query_prefix + key_b + query_infix + key_a + query_suffix
    if len(query_a) != len(query_b) or Counter(query_a) != Counter(query_b):
        raise ValueError("association queries are not token-multiset matched")

    left_key, left_answer = (key_b, answer_b) if mirror else (key_a, answer_a)
    right_key, right_answer = (key_a, answer_a) if mirror else (key_b, answer_b)
    left_record = record(left_key, left_answer)
    right_record = record(right_key, right_answer)
    if len(left_record) != len(right_record):
        raise AssertionError("position-symmetric records have different token lengths")
    value_offset = len(record_prefix) + len(left_key) + len(record_infix)

    content_end = int(target_length) - len(query_a)
    maximum_block = (content_end - 1) // int(block_size)
    center_block = round(float(depth_percent) * maximum_block / 100.0)
    left_block = center_block - int(slot_gap_blocks)
    right_block = center_block + int(slot_gap_blocks)
    if left_block < 0 or right_block > maximum_block:
        raise ValueError("requested symmetric association slots fall outside the prompt")
    within_block = int(block_size) // 2
    left_answer_start = left_block * int(block_size) + within_block
    right_answer_start = right_block * int(block_size) + within_block
    if (
        left_answer_start + len(left_answer) > (left_block + 1) * int(block_size)
        or right_answer_start + len(right_answer) > (right_block + 1) * int(block_size)
    ):
        raise ValueError("an association answer crosses its registered block boundary")

    left_record_start = left_answer_start - value_offset
    right_record_start = right_answer_start - value_offset
    before_count = left_record_start - len(prefix)
    between_count = right_record_start - (left_record_start + len(left_record))
    after_count = content_end - (right_record_start + len(right_record))
    if min(before_count, between_count, after_count) < 0:
        raise ValueError("target length is too short for the symmetric association layout")
    context = (
        prefix
        + tuple(_repeat_to_length(filler, before_count))
        + left_record
        + tuple(_repeat_to_length(filler, between_count))
        + right_record
        + tuple(_repeat_to_length(filler, after_count))
    )
    prompt_a = context + query_a
    prompt_b = context + query_b
    if len(prompt_a) != target_length or len(prompt_b) != target_length:
        raise AssertionError("association builder changed the registered prompt length")
    if Counter(prompt_a) != Counter(prompt_b):
        raise AssertionError("association pair changed the prompt token multiset")

    left_span = (left_answer_start, left_answer_start + len(left_answer))
    right_span = (right_answer_start, right_answer_start + len(right_answer))
    answer_spans = (
        {"a": right_span, "b": left_span}
        if mirror
        else {"a": left_span, "b": right_span}
    )
    for name, answer in (("a", answer_a), ("b", answer_b)):
        for prompt in (prompt_a, prompt_b):
            matches = _find_subsequence(prompt, answer)
            if matches != [answer_spans[name][0]]:
                raise ValueError(f"answer {name} is not unique at its registered position")

    query_start = len(context)
    target_key_start = query_start + len(query_prefix)
    control_key_start = target_key_start + len(key_a) + len(query_infix)
    return {
        "prompts": {"query_a": list(prompt_a), "query_b": list(prompt_b)},
        "prompt_sha256": {
            "query_a": _prompt_sha256(prompt_a),
            "query_b": _prompt_sha256(prompt_b),
        },
        "answer_ids": {"a": list(answer_a), "b": list(answer_b)},
        "answer_spans": {name: list(span) for name, span in answer_spans.items()},
        "gold_spans": {
            "query_a": list(answer_spans["a"]),
            "query_b": list(answer_spans["b"]),
        },
        "query_key_spans": {
            "target": [target_key_start, target_key_start + len(key_a)],
            "control": [control_key_start, control_key_start + len(key_a)],
        },
        "slot_blocks": [left_block, right_block],
        "depth_percent": float(depth_percent),
        "actual_midpoint_depth_percent": 100.0
        * (left_answer_start + right_answer_start)
        / (2.0 * target_length),
        "first_token_frequency_bucket": int(bucket_a),
        "mirror": bool(mirror),
    }


def _association_pair_sha256(
    *,
    prompts: Sequence[Sequence[int]],
    candidate_first_token_ids: Sequence[int],
    gold_spans: Sequence[Sequence[int]],
    split: str,
    depth_percent: float,
    frequency_bucket: int,
    mirror: bool,
) -> str:
    return _json_sha256(
        {
            "prompt_sha256": [_prompt_sha256(prompt) for prompt in prompts],
            "candidate_first_token_ids": [
                int(value) for value in candidate_first_token_ids
            ],
            "gold_spans": [
                [int(value) for value in span] for span in gold_spans
            ],
            "split": str(split),
            "depth_percent": float(depth_percent),
            "frequency_bucket": int(frequency_bucket),
            "mirror": bool(mirror),
        }
    )


def _association_answer_pairs(
    counts: torch.Tensor,
    *,
    excluded_ids: set[int],
    seed: int,
) -> list[tuple[int, int, int]]:
    if counts.ndim != 1 or bool((counts < 0).any()):
        raise ValueError("training token counts must be a non-negative vector")
    grouped: defaultdict[int, list[int]] = defaultdict(list)
    for token_id in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist():
        token_id = int(token_id)
        if token_id in excluded_ids:
            continue
        count = int(counts[token_id])
        bucket = int(math.floor(math.log2(count + 1)))
        grouped[bucket].append(token_id)
    rng = random.Random(int(seed))
    pairs = []
    for bucket in sorted(grouped):
        token_ids = grouped[bucket]
        rng.shuffle(token_ids)
        pairs.extend(
            (token_ids[index], token_ids[index + 1], bucket)
            for index in range(0, len(token_ids) - 1, 2)
        )
    rng.shuffle(pairs)
    return pairs


def _training_token_counts(
    training_data_manifest: Path,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    data, manifest = _load_strict_legacy_data(training_data_manifest)
    tokens = data["tokens"].to(torch.int64)
    offsets = data["offsets"].to(torch.int64)
    counts = torch.bincount(tokens, minlength=MODEL_CONTRACT.get("vocab_size", 128256))
    for row_index in data["validation_indices"].to(torch.int64).tolist():
        start, end = (int(offsets[row_index]), int(offsets[row_index + 1]))
        counts -= torch.bincount(tokens[start:end], minlength=counts.numel())
    if bool((counts < 0).any()):
        raise RuntimeError("validation subtraction produced negative training counts")
    train_lengths = offsets[data["train_indices"] + 1] - offsets[data["train_indices"]]
    if int(counts.sum()) != int(train_lengths.sum()):
        raise RuntimeError("training token counts do not match the frozen train split")
    receipt = {
        "formula": "floor(log2(count + 1))",
        "training_token_count": int(counts.sum()),
        "counts_dtype": str(counts.dtype),
        "counts_sha256": hashlib.sha256(counts.numpy().tobytes()).hexdigest(),
        "tokens_sha256": manifest["files"]["tokens"]["sha256"],
        "offsets_sha256": manifest["files"]["offsets"]["sha256"],
        "train_indices_sha256": manifest["files"]["train_indices"]["sha256"],
    }
    return counts, manifest, receipt


def prepare_association_swap_cases(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    temporary_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else temporary_dir)

    counts, training_manifest, count_receipt = _training_token_counts(
        args.training_data_manifest
    )
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    if not _tokenizer_identity_matches(
        training_manifest.get("tokenizer", {}), expected_tokenizer
    ):
        raise ValueError("frozen training tokens use a different tokenizer")

    from transformers import AutoTokenizer

    local = Path(args.model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local,
    )
    if len(tokenizer) != counts.numel():
        raise ValueError("training token counts do not match tokenizer vocabulary size")
    filler_ids = _answer_ids(
        tokenizer,
        " Unrelated archival material is repeated here solely as fixed filler.",
    )
    excluded = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    excluded.update(filler_ids)
    key_candidates = sorted(
        (
            int(token_id)
            for token_id in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
            if int(token_id) not in excluded
        ),
        key=lambda token_id: (-int(counts[token_id]), token_id),
    )
    if len(key_candidates) < 2:
        raise ValueError("frozen training tokens contain fewer than two usable keys")
    key_a, key_b = key_candidates[:2]
    excluded.update((key_a, key_b))
    candidate_pairs = _association_answer_pairs(
        counts,
        excluded_ids=excluded,
        seed=42,
    )

    target_pairs = 256
    depths = (10.0, 25.0, 50.0, 75.0, 90.0)
    records = []
    split_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    temporary_dir.mkdir(parents=True)
    for answer_a, answer_b, bucket in candidate_pairs:
        index = len(records)
        if index == target_pairs:
            break
        split = "dev" if index < 128 else "test"
        split_index = index if split == "dev" else index - 128
        depth = depths[split_index % len(depths)]
        mirror = bool(split_index % 2)
        try:
            pair = build_association_swap_pair(
                tokenizer,
                key_a_ids=[key_a],
                key_b_ids=[key_b],
                answer_a_ids=[answer_a],
                answer_b_ids=[answer_b],
                filler_ids=filler_ids,
                first_token_frequency_buckets={answer_a: bucket, answer_b: bucket},
                target_length=16384,
                depth_percent=depth,
                block_size=SPARSE_CONFIG["block_size"],
                slot_gap_blocks=2,
                mirror=mirror,
            )
        except ValueError:
            continue
        prompts = [pair["prompts"]["query_a"], pair["prompts"]["query_b"]]
        gold_spans = [pair["gold_spans"]["query_a"], pair["gold_spans"]["query_b"]]
        candidates = [answer_a, answer_b]
        pair_sha256 = _association_pair_sha256(
            prompts=prompts,
            candidate_first_token_ids=candidates,
            gold_spans=gold_spans,
            split=split,
            depth_percent=depth,
            frequency_bucket=bucket,
            mirror=mirror,
        )
        record_name = f"records/{index:03d}_{pair_sha256}.pt"
        record_path = temporary_dir / record_name
        _atomic_torch(
            record_path,
            {
                "schema": ASSOCIATION_SWAP_CASE_SCHEMA,
                "pair_sha256": pair_sha256,
                "split": split,
                "depth_percent": depth,
                "frequency_bucket": bucket,
                "mirror": mirror,
                "prompt_ids": torch.tensor(prompts, dtype=torch.int32),
                "gold_spans": torch.tensor(gold_spans, dtype=torch.int32),
                "candidate_first_token_ids": torch.tensor(
                    candidates, dtype=torch.int32
                ),
            },
        )
        records.append(
            {
                "file": record_name,
                "sha256": sha256_file(record_path),
                "size_bytes": record_path.stat().st_size,
                "pair_sha256": pair_sha256,
                "split": split,
                "depth_percent": depth,
            }
        )
        split_counts[split] += 1
        depth_counts[f"{depth:g}"] += 1
        bucket_counts[str(bucket)] += 1
    if len(records) != target_pairs or split_counts != {"dev": 128, "test": 128}:
        raise RuntimeError("could not build the registered 128-dev/128-test swap set")
    manifest = {
        "schema": ASSOCIATION_SWAP_CASE_MANIFEST_SCHEMA,
        "status": "complete",
        "seed": 42,
        "target_length": 16384,
        "pair_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "depth_counts": dict(sorted(depth_counts.items())),
        "mirror_counts": {"false": 128, "true": 128},
        "frequency_bucket_counts": dict(sorted(bucket_counts.items())),
        "frequency": count_receipt,
        "training_data_manifest_sha256": sha256_file(args.training_data_manifest),
        "tokenizer": expected_tokenizer,
        "pair_set_sha256": _json_sha256(
            [record["pair_sha256"] for record in records]
        ),
        "records": records,
        "script_sha256": _script_sha256(),
    }
    _atomic_json(temporary_dir / "manifest.json", manifest)
    os.replace(temporary_dir, output_dir)
    return manifest


def load_association_swap_cases(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != ASSOCIATION_SWAP_CASE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("seed") != 42
        or manifest.get("pair_count") != 256
        or manifest.get("split_counts") != {"dev": 128, "test": 128}
    ):
        raise ValueError("association-swap case manifest contract mismatch")
    cases = []
    pair_hashes = []
    for item in manifest.get("records", []):
        relative = Path(str(item.get("file", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("association-swap case manifest has an unsafe path")
        path = root / relative
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"association-swap case receipt mismatch: {relative}")
        case = torch.load(path, map_location="cpu", weights_only=True)
        prompts = case.get("prompt_ids")
        spans = case.get("gold_spans")
        candidates = case.get("candidate_first_token_ids")
        if (
            case.get("schema") != ASSOCIATION_SWAP_CASE_SCHEMA
            or not all(torch.is_tensor(value) for value in (prompts, spans, candidates))
            or prompts.shape != (2, 16384)
            or spans.shape != (2, 2)
            or candidates.shape != (2,)
        ):
            raise ValueError("association-swap case tensor contract mismatch")
        if not torch.equal(torch.sort(prompts[0]).values, torch.sort(prompts[1]).values):
            raise ValueError("association-swap prompts changed the token multiset")
        for condition in range(2):
            start, end = (int(value) for value in spans[condition].tolist())
            if end != start + 1 or int(prompts[condition, start]) != int(
                candidates[condition]
            ):
                raise ValueError("association-swap gold span does not match its candidate")
        pair_sha256 = _association_pair_sha256(
            prompts=prompts.tolist(),
            candidate_first_token_ids=candidates.tolist(),
            gold_spans=spans.tolist(),
            split=str(case.get("split")),
            depth_percent=float(case.get("depth_percent")),
            frequency_bucket=int(case.get("frequency_bucket")),
            mirror=bool(case.get("mirror")),
        )
        if pair_sha256 != case.get("pair_sha256") or pair_sha256 != item.get(
            "pair_sha256"
        ):
            raise ValueError("association-swap pair hash mismatch")
        pair_hashes.append(pair_sha256)
        cases.append(case)
    if len(cases) != 256 or len(set(pair_hashes)) != 256:
        raise ValueError("association-swap cases are incomplete or duplicated")
    if _json_sha256(pair_hashes) != manifest.get("pair_set_sha256"):
        raise ValueError("association-swap pair-set hash mismatch")
    return manifest, cases


def _chat_wrapper_tokens(tokenizer: Any) -> tuple[list[int], list[int]]:
    marker_ids = _answer_ids(tokenizer, CHAT_WRAPPER_MARKER)
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_WRAPPER_MARKER}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(wrapped, Mapping):
        wrapped = wrapped.get("input_ids")
    wrapped_ids = [int(token_id) for token_id in wrapped]
    matches = _find_subsequence(wrapped_ids, marker_ids)
    if len(matches) != 1:
        raise ValueError("chat wrapper marker is not unique")
    start = matches[0]
    prefix, suffix = wrapped_ids[:start], wrapped_ids[start + len(marker_ids) :]
    probe = "EVQ chat wrapper parity probe."
    direct = tokenizer.apply_chat_template(
        [{"role": "user", "content": probe}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(direct, Mapping):
        direct = direct.get("input_ids")
    if prefix + _answer_ids(tokenizer, probe) + suffix != [int(value) for value in direct]:
        raise ValueError("segmented chat wrapper differs from tokenizer.apply_chat_template")
    return prefix, suffix


def _wrap_passkey_prompt_ids(
    prompt_ids: Sequence[int],
    *,
    query_suffix_ids: Sequence[int],
    chat_prefix_ids: Sequence[int],
    chat_suffix_ids: Sequence[int],
) -> list[int]:
    raw = [int(token_id) for token_id in prompt_ids]
    query = [int(token_id) for token_id in query_suffix_ids]
    if not query or raw[-len(query) :] != query:
        raise ValueError("passkey prompt does not end with the registered query")
    overhead = len(chat_prefix_ids) + len(chat_suffix_ids)
    cut_end = len(raw) - len(query)
    cut_start = cut_end - overhead
    if overhead <= 0 or cut_start <= 0:
        raise ValueError("passkey prompt has no safe filler budget for chat framing")
    wrapped = (
        [int(token_id) for token_id in chat_prefix_ids]
        + raw[:cut_start]
        + query
        + [int(token_id) for token_id in chat_suffix_ids]
    )
    if len(wrapped) != len(raw):
        raise AssertionError("chat wrapping changed the registered prompt length")
    return wrapped


def _chat_wrap_passkey_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chat_prefix, chat_suffix = _chat_wrapper_tokens(tokenizer)
    query_ids = _answer_ids(tokenizer, PASSKEY_QUERY)
    transformed = []
    for row in rows:
        raw_prompt = [int(token_id) for token_id in row["prompt_ids"]]
        answer_span = _answer_span(tokenizer, row)
        wrapped = _wrap_passkey_prompt_ids(
            raw_prompt,
            query_suffix_ids=query_ids,
            chat_prefix_ids=chat_prefix,
            chat_suffix_ids=chat_suffix,
        )
        cut_start = len(raw_prompt) - len(query_ids) - len(chat_prefix) - len(chat_suffix)
        if answer_span[1] > cut_start:
            raise ValueError("chat framing would remove the registered passkey span")
        updated = dict(row)
        updated["prompt_ids"] = wrapped
        updated["prompt_sha256"] = _prompt_sha256(wrapped)
        updated["source"] = {
            **dict(row["source"]),
            "raw_prompt_sha256": row["prompt_sha256"],
            "prompt_transform": "llama3_chat_wrap_preserve_length_v1",
        }
        if len(_find_subsequence(wrapped, _answer_ids(tokenizer, row["answers"][0]))) != 1:
            raise ValueError("chat-wrapped prompt lost the unique passkey answer")
        transformed.append(updated)
    return transformed, {
        "name": "llama3_chat_wrap_preserve_length_v1",
        "wrapper_prefix_tokens": len(chat_prefix),
        "wrapper_suffix_tokens": len(chat_suffix),
        "removed_filler_tokens": len(chat_prefix) + len(chat_suffix),
        "prompt_hashes_sha256": _json_sha256(
            sorted(row["prompt_sha256"] for row in transformed)
        ),
    }


def _legacy_passkey_rows(tokenizer: Any) -> list[dict[str, Any]]:
    rows = build_passkey_examples(tokenizer)
    legacy = []
    for row in rows:
        normalized = dict(row)
        normalized["schema"] = LEGACY_EXAMPLE_SCHEMA
        normalized.pop("generation_tokens", None)
        normalized.pop("scorer", None)
        legacy.append(normalized)
    return legacy


def prepare_passkey(args: argparse.Namespace) -> dict[str, Any]:
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    if reference.get("schema") != LEGACY_MANIFEST_SCHEMA:
        raise ValueError("reference capability manifest is not the frozen v1 manifest")
    expected = reference.get("files", {}).get("passkey.jsonl")
    if not isinstance(expected, Mapping):
        raise ValueError("reference manifest has no passkey.jsonl record")
    registered = {
        "sha256": PASSKEY_SHA256,
        "size_bytes": PASSKEY_SIZE_BYTES,
        "row_count": PASSKEY_ROWS,
    }
    for key, value in registered.items():
        if expected.get(key) != value:
            raise ValueError(f"reference passkey {key} differs from the preregistered value")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    rows = _legacy_passkey_rows(tokenizer)
    if len(rows) != PASSKEY_ROWS:
        raise ValueError(f"rebuilt {len(rows)} passkey rows; expected {PASSKEY_ROWS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "passkey.jsonl"
    if output.exists() or (args.output_dir / "manifest.json").exists():
        raise FileExistsError(args.output_dir)
    temporary = output.with_suffix(".jsonl.incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    actual = {
        "sha256": sha256_file(temporary),
        "size_bytes": temporary.stat().st_size,
        "row_count": len(rows),
    }
    if actual != registered:
        raise ValueError(f"rebuilt passkey artifact mismatch: {actual}")
    os.replace(temporary, output)

    receipt = {
        "schema": PASSKEY_RECEIPT_SCHEMA,
        "source_manifest_sha256": sha256_file(args.reference_manifest),
        "source_manifest_schema": reference["schema"],
        "example_schema": LEGACY_EXAMPLE_SCHEMA,
        "file": {"name": output.name, **actual},
        "tokenizer": reference.get("tokenizer"),
        "selection": {
            "lengths": [8192, 16384, 32768],
            "depths": [10, 25, 50, 75, 90],
            "trials_per_cell": 20,
            "seed": 42,
        },
    }
    _atomic_json(args.output_dir / "manifest.json", receipt)
    return receipt


def load_passkey_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PASSKEY_RECEIPT_SCHEMA:
        raise ValueError("passkey receipt schema mismatch")
    record = manifest.get("file")
    if not isinstance(record, Mapping) or record.get("name") != "passkey.jsonl":
        raise ValueError("passkey receipt has an unsafe file record")
    path = root / "passkey.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
    }
    expected = {key: record.get(key) for key in actual}
    if actual != expected or actual != {
        "sha256": PASSKEY_SHA256,
        "size_bytes": PASSKEY_SIZE_BYTES,
        "row_count": PASSKEY_ROWS,
    }:
        raise ValueError("passkey artifact does not match the frozen v1 file")
    rows = load_jsonl(path)
    for row in rows:
        if row.get("schema") != LEGACY_EXAMPLE_SCHEMA:
            raise ValueError("passkey example schema mismatch")
        if row.get("task") != "passkey_retrieval":
            raise ValueError("passkey file contains another task")
        if _prompt_sha256(row["prompt_ids"]) != row.get("prompt_sha256"):
            raise ValueError(f"passkey prompt hash mismatch: {row.get('example_id')}")
        if len(row["prompt_ids"]) != int(row["target_length"]):
            raise ValueError(f"passkey prompt length mismatch: {row.get('example_id')}")
    return manifest, rows


def _validate_config(model_name: str) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=Path(model_name).is_dir(),
    )
    rope = getattr(config, "rope_parameters", None)
    if rope is None:
        rope = getattr(config, "rope_scaling", None)
    rope = dict(rope) if isinstance(rope, Mapping) else {}
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope.get("rope_theta")
    observed = {
        key: (rope_theta if key == "rope_theta" else getattr(config, key, None))
        for key in MODEL_CONTRACT
    }
    if observed != MODEL_CONTRACT:
        raise ValueError(f"model architecture contract mismatch: {observed}")
    if rope and (rope.get("rope_type", "default") != "default" or set(rope) - {"rope_type", "rope_theta"}):
        raise ValueError(f"raw extrapolation requires default unscaled RoPE, found {rope}")
    return observed


def _validate_arm(
    *,
    adapter_dir: Path,
    substrate: str,
    training_manifest: Path,
    model_manifest: Path,
) -> dict[str, Any]:
    training_hash = sha256_file(training_manifest)
    model_manifest_hash = sha256_file(model_manifest)
    metadata = validate_adapter_identity(
        adapter_dir,
        substrate=substrate,
        training_manifest_sha256=training_hash,
    )
    if metadata.get("model_manifest_sha256") != model_manifest_hash:
        raise ValueError("adapter model-manifest hash mismatch")
    inv_freq, frequency_record, provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt",
        expected_method=substrate,
    )
    head_dim = int(frequency_record.get("head_dim", 2 * inv_freq.numel()))
    base = float(frequency_record.get("base", MODEL_CONTRACT["rope_theta"]))
    canonical, _ = build_training_inv_freq(
        rope_method=substrate,
        head_dim=head_dim,
        base=base,
        tau=1.414,
    )
    if not torch.allclose(
        inv_freq.to(torch.float64), canonical.to(torch.float64), rtol=0.0, atol=1e-12
    ):
        raise ValueError("frequency artifact differs from the canonical substrate")
    return {
        "substrate": substrate,
        "adapter_sha256": metadata["adapter_sha256"],
        "frequency": provenance,
        "model_manifest_sha256": model_manifest_hash,
        "training_manifest_sha256": training_hash,
        "protocol_sha256": _json_sha256(metadata.get("protocol")),
        "code_sha256": metadata.get("code_sha256"),
        "metadata": metadata,
    }


def _validate_stage2_arm(
    *,
    adapter_dir: Path,
    substrate: str,
    training_manifest: Path,
    model_manifest: Path,
) -> dict[str, Any]:
    metadata_path = adapter_dir / "stage2_meta.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    for path in (metadata_path, adapter_path, config_path, adapter_dir / "custom_inv_freq.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "evq_cosh.lora_stage2_retrieval.v1":
        raise ValueError("stage2 metadata schema mismatch")
    expected = {
        "status": "complete",
        "substrate": substrate,
        "max_steps": 50,
        "seed": 42,
        "objective": "answer_only_chat_causal_lm",
        "training_manifest_sha256": sha256_file(training_manifest),
        "model_manifest_sha256": sha256_file(model_manifest),
        "adapter_sha256": sha256_file(adapter_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"stage2 metadata mismatch at {key}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("r", -1)) != 64 or int(config.get("lora_alpha", -1)) != 128:
        raise ValueError("stage2 adapter rank/alpha mismatch")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("stage2 adapter target modules mismatch")
    inv_freq, _, provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt", expected_method=substrate
    )
    canonical, _ = build_training_inv_freq(
        rope_method=substrate,
        head_dim=2 * inv_freq.numel(),
        base=MODEL_CONTRACT["rope_theta"],
        tau=1.414,
    )
    if not torch.allclose(inv_freq.to(torch.float64), canonical, rtol=0.0, atol=1e-12):
        raise ValueError("stage2 frequency artifact differs from the canonical substrate")
    return {
        "substrate": substrate,
        "adapter_sha256": metadata["adapter_sha256"],
        "frequency": provenance,
        "model_manifest_sha256": metadata["model_manifest_sha256"],
        "training_manifest_sha256": metadata["training_manifest_sha256"],
        "stage2_metadata": metadata,
    }


def _tokenizer_identity_matches(
    recorded: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    identifier = recorded.get("identifier") or recorded.get("requested") or recorded.get("name_or_path")
    if identifier is not None:
        identifier = Path(str(identifier)).name
    return identifier == expected.get("identifier") and recorded.get("files") == expected.get("files")


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    config = _validate_config(args.model_name)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    passkey_manifest, rows = load_passkey_rows(args.passkey_root)

    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    recorded_tokenizer = passkey_manifest.get("tokenizer", {})
    if not _tokenizer_identity_matches(recorded_tokenizer, expected_tokenizer):
        raise ValueError("rebuilt passkey tokenizer differs from the model tokenizer")

    geo = _validate_arm(
        adapter_dir=args.geo_adapter,
        substrate="native_geo",
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    evq = _validate_arm(
        adapter_dir=args.evq_adapter,
        substrate="evq_cosh",
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    for key in ("model_manifest_sha256", "training_manifest_sha256", "code_sha256"):
        if geo[key] != evq[key]:
            raise ValueError(f"matched-arm identity differs at {key}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    spans = []
    answer_spans = []
    for row in rows:
        start, end = _needle_span(tokenizer, row)
        spans.append(end - start)
        answer_start, answer_end = _answer_span(tokenizer, row)
        if not start <= answer_start < answer_end <= end:
            raise ValueError(f"answer span is outside the needle: {row['example_id']}")
        answer_spans.append(answer_end - answer_start)

    suite_manifest, suite_rows = load_capability_suite(args.suite_root)
    receipt = {
        "schema": "evq_cosh.lora_sparse_conversion_dry_run.v1",
        "status": "pass",
        "model_contract": config,
        "passkey": {
            "manifest_sha256": sha256_file(args.passkey_root / "manifest.json"),
            "file_sha256": PASSKEY_SHA256,
            "rows": len(rows),
            "needle_span_tokens": {"min": min(spans), "max": max(spans)},
            "answer_span_tokens": {"min": min(answer_spans), "max": max(answer_spans)},
        },
        "capability_suite": {
            "manifest_sha256": sha256_file(args.suite_root / "manifest.json"),
            "rows": len(suite_rows),
            "task_counts": suite_manifest.get("task_counts"),
        },
        "arms": {
            "native_geo": {key: value for key, value in geo.items() if key != "metadata"},
            "evq_cosh": {key: value for key, value in evq.items() if key != "metadata"},
        },
        "sparse_config": SPARSE_CONFIG,
        "phase0_selection": {
            "trials": [0, 1],
            "lengths": [8192, 16384, 32768],
            "cases_per_arm": 30,
        },
        "script_sha256": _script_sha256(),
    }
    _atomic_json(args.output, receipt)
    return receipt


def _attention_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if all(
            hasattr(module, attr)
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "layer_idx")
        )
    ]
    expected = int(getattr(model.config, "num_hidden_layers", -1))
    if len(modules) != expected:
        raise RuntimeError(f"found {len(modules)} attention modules; expected {expected}")
    return modules


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    return hidden_states.repeat_interleave(int(repetitions), dim=1)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return value * cos + _rotate_half(value) * sin


def _static_keep(length: int, config: Mapping[str, int], device: torch.device) -> torch.Tensor:
    keep = torch.zeros(length, dtype=torch.bool, device=device)
    keep[: min(length, int(config["sink_tokens"]))] = True
    keep[max(0, length - int(config["local_window"])) :] = True
    return keep


def _block_layout(
    scores: torch.Tensor,
    config: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if scores.ndim != 4 or scores.shape[-2] != 1:
        raise ValueError("block selector requires [batch, heads, 1, keys] scores")
    length = scores.shape[-1]
    block_size = int(config["block_size"])
    padded_length = math.ceil(length / block_size) * block_size
    static = _static_keep(length, config, scores.device)
    remote = ~static
    padded_remote = F.pad(remote, (0, padded_length - length), value=False)
    valid_blocks = padded_remote.view(-1, block_size).any(dim=-1)
    masked = scores.masked_fill(~remote.view(1, 1, 1, -1), float("-inf"))
    masked = F.pad(masked, (0, padded_length - length), value=float("-inf"))
    block_scores = masked.view(*scores.shape[:-1], -1, block_size).amax(dim=-1)
    return static, valid_blocks, block_scores, padded_length


def _gold_block_bounds(
    scores: torch.Tensor,
    config: Mapping[str, int],
) -> tuple[int, int]:
    gold_span = config.get("gold_span")
    if not isinstance(gold_span, (list, tuple)) or len(gold_span) != 2:
        raise ValueError("gold-block mode requires a two-element gold_span")
    gold_start, gold_end = (int(value) for value in gold_span)
    if not 0 <= gold_start < gold_end <= scores.shape[-1]:
        raise ValueError("gold_span is outside the current KV index space")
    block_size = int(config["block_size"])
    return (
        (gold_start // block_size) * block_size,
        min(scores.shape[-1], math.ceil(gold_end / block_size) * block_size),
    )


def _sparse_keep_mask(
    scores: torch.Tensor,
    *,
    mode: str,
    config: Mapping[str, int],
) -> torch.Tensor:
    if mode in {"dense", "full"}:
        return torch.ones_like(scores, dtype=torch.bool)
    if mode == "gold_drop_all":
        block_start, block_end = _gold_block_bounds(scores, config)
        keep = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        keep[block_start:block_end] = False
        return keep.view(1, 1, 1, -1).expand_as(scores)
    if mode in {"head_score", "gold_drop", "oracle_gold"}:
        if "selected_query_heads" not in config:
            raise ValueError(f"{mode} requires selected_query_heads")
        selected_heads = tuple(int(head) for head in config["selected_query_heads"])
        if len(set(selected_heads)) != len(selected_heads) or any(
            head < 0 or head >= scores.shape[1] for head in selected_heads
        ):
            raise ValueError("selected_query_heads are invalid")
        dense = torch.ones_like(scores, dtype=torch.bool)
        if not selected_heads:
            return dense
        selected = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
        selected[list(selected_heads)] = True
        selected = selected.view(1, -1, 1, 1)
        if mode == "head_score":
            sparse = _sparse_keep_mask(scores, mode="score", config=config)
            return torch.where(selected, sparse, dense)

        block_start, block_end = _gold_block_bounds(scores, config)
        gold_block = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        gold_block[block_start:block_end] = True
        if mode == "gold_drop":
            causal = (~gold_block).view(1, 1, 1, -1).expand_as(scores)
        else:
            static = _static_keep(scores.shape[-1], config, scores.device)
            causal = (static | gold_block).view(1, 1, 1, -1).expand_as(scores)
        return torch.where(selected, causal, dense)
    if mode not in {"score", "fixed", "oracle_include_all"}:
        raise ValueError(f"unsupported attention mode: {mode}")
    static, valid_blocks, block_scores, padded_length = _block_layout(scores, config)
    block_size = int(config["block_size"])
    candidate_ids = torch.nonzero(valid_blocks, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        return static.view(1, 1, 1, -1).expand_as(scores)
    budget = min(int(config["top_blocks"]), int(candidate_ids.numel()))
    selected = torch.zeros_like(block_scores, dtype=torch.bool)
    if mode in {"score", "oracle_include_all"}:
        available = block_scores.masked_fill(
            ~valid_blocks.view(1, 1, 1, -1), float("-inf")
        )
        indices = available.topk(budget, dim=-1).indices
        selected.scatter_(-1, indices, True)
    else:
        count = int(candidate_ids.numel())
        offsets = [min(count - 1, int((index + 0.5) * count / budget)) for index in range(budget)]
        fixed_ids = candidate_ids[torch.tensor(offsets, device=scores.device)]
        selected[..., fixed_ids] = True
    if mode == "oracle_include_all":
        block_start, block_end = _gold_block_bounds(scores, config)
        forced_ids = range(block_start // block_size, math.ceil(block_end / block_size))
        for forced_id in forced_ids:
            if not bool(valid_blocks[forced_id]):
                continue
            missing = ~selected[..., forced_id]
            if not bool(missing.any()):
                continue
            droppable = block_scores.masked_fill(~selected, float("inf"))
            for protected_id in forced_ids:
                droppable[..., protected_id] = float("inf")
            drop_ids = droppable.argmin(dim=-1, keepdim=True)
            if not bool(torch.isfinite(droppable.amin(dim=-1)[missing]).all()):
                raise RuntimeError("oracle_include_all has no replaceable selected block")
            drop_mask = torch.zeros_like(selected)
            drop_mask.scatter_(-1, drop_ids, True)
            selected &= ~(drop_mask & missing.unsqueeze(-1))
            selected[..., forced_id] |= missing
    selected_tokens = selected.repeat_interleave(block_size, dim=-1)[..., : scores.shape[-1]]
    static_tokens = static.view(1, 1, 1, -1).expand_as(scores)
    if selected_tokens.shape[-1] != scores.shape[-1] or padded_length < scores.shape[-1]:
        raise AssertionError("block selection changed the token index space")
    return selected_tokens | static_tokens


def exact_block_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode attention; it masks original-position KV without gathering."""
    if query.shape[-2] != 1:
        raise RuntimeError("exact-score attention is decode-only; dense prefill must use SDPA")
    config = getattr(module, "_evq_sparse_config", None)
    if not isinstance(config, Mapping):
        raise RuntimeError("attention module has no sparse-conversion configuration")
    repetitions = query.shape[1] // key.shape[1]
    if repetitions <= 0 or query.shape[1] != key.shape[1] * repetitions:
        raise RuntimeError("query/KV head geometry is not an integer GQA mapping")
    key_states = _repeat_kv(key, repetitions)
    value_states = _repeat_kv(value, repetitions)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        weights = weights + attention_mask
    keep = _sparse_keep_mask(weights, mode=str(config["mode"]), config=config)
    weights = weights.masked_fill(~keep, torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights


def _register_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPL, exact_block_attention_forward)


def _set_attention_mode(
    model: torch.nn.Module,
    implementation: str,
    *,
    mode: str = "dense",
    config: Mapping[str, int] = SPARSE_CONFIG,
    retrieval_heads: Sequence[tuple[int, int]] | None = None,
    gold_span: Sequence[int] | None = None,
) -> None:
    heads_by_layer: defaultdict[int, list[int]] = defaultdict(list)
    if retrieval_heads is not None:
        for layer, head in retrieval_heads:
            heads_by_layer[int(layer)].append(int(head))
    for _, module in _attention_modules(model):
        module.config._attn_implementation = implementation
        module_config: dict[str, Any] = {**config, "mode": mode}
        if retrieval_heads is not None:
            module_config["selected_query_heads"] = heads_by_layer[int(module.layer_idx)]
        if gold_span is not None:
            module_config["gold_span"] = tuple(int(value) for value in gold_span)
        module._evq_sparse_config = module_config
    model.config._attn_implementation = implementation


def _load_arm_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU phase requires CUDA")
    validator = _validate_stage2_arm if getattr(args, "stage2", False) else _validate_arm
    identity = validator(
        adapter_dir=args.adapter_dir,
        substrate=args.substrate,
        training_manifest=args.training_data_manifest,
        model_manifest=args.model_manifest,
    )
    _validate_config(args.model_name)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = Path(args.model_name).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(torch.device("cuda"))
    model.eval()
    model.config.use_cache = True

    inv_freq, _, _ = load_frequency_artifact(
        args.adapter_dir / "custom_inv_freq.pt",
        expected_method=args.substrate,
    )
    geometry = resolve_model_rope_geometry(model.config)
    canonical, _ = build_training_inv_freq(
        rope_method=args.substrate,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    if not torch.allclose(inv_freq.to(torch.float64), canonical, rtol=0.0, atol=1e-12):
        raise ValueError("runtime frequency artifact differs from canonical")
    inject_inv_freq(model, inv_freq)
    for name, module in find_rotary_modules(model):
        if not hasattr(module, "attention_scaling"):
            raise RuntimeError(f"rotary module {name} has no attention_scaling")
        module.attention_scaling = 1.0
        original = getattr(module, "original_inv_freq", None)
        if torch.is_tensor(original):
            if original.shape != module.inv_freq.shape:
                raise RuntimeError(f"original_inv_freq shape mismatch at {name}")
            original.copy_(module.inv_freq)
    verify_model_inv_freq(model, inv_freq)
    if getattr(args, "stage2", False):
        identity["adapter_artifact_receipt"] = {
            "format_version": 1,
            "files": {
                name: {
                    "sha256": sha256_file(args.adapter_dir / name),
                    "size_bytes": (args.adapter_dir / name).stat().st_size,
                }
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "custom_inv_freq.pt",
                    "stage2_meta.json",
                )
            },
        }
    else:
        identity["adapter_artifact_receipt"] = adapter_artifact_receipt(args.adapter_dir)
    return model, tokenizer, identity


def _probe_metrics(
    scores: torch.Tensor,
    *,
    needle_start: int,
    needle_end: int,
    config: Mapping[str, int],
) -> dict[str, Any]:
    if scores.ndim != 2:
        raise ValueError("probe scores must have [heads, keys] shape")
    expanded = scores.unsqueeze(0).unsqueeze(2)
    static, valid_blocks, block_scores, _ = _block_layout(expanded, config)
    block_scores = block_scores[0, :, 0]
    block_size = int(config["block_size"])
    gold_blocks = sorted(set(range(needle_start // block_size, (needle_end - 1) // block_size + 1)))
    gold_remote = [block for block in gold_blocks if bool(valid_blocks[block])]
    dense_weights = F.softmax(scores.float(), dim=-1)
    dense_mass = dense_weights[:, needle_start:needle_end].sum(dim=-1)

    if gold_remote:
        gold_score = block_scores[:, gold_remote].amax(dim=-1)
        ranks = 1 + (block_scores > gold_score.unsqueeze(-1)).logical_and(
            valid_blocks.unsqueeze(0)
        ).sum(dim=-1)
        distractor_mask = valid_blocks.clone()
        distractor_mask[gold_remote] = False
        distractor = block_scores.masked_fill(~distractor_mask.unsqueeze(0), float("-inf")).amax(dim=-1)
        margin = gold_score - distractor
    else:
        ranks = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        margin = torch.full((scores.shape[0],), float("nan"), device=scores.device)

    keep16 = _sparse_keep_mask(expanded, mode="score", config=config)[0, :, 0]
    sparse_scores = scores.masked_fill(~keep16, torch.finfo(scores.dtype).min)
    sparse_weights = F.softmax(sparse_scores.float(), dim=-1)
    sparse_mass = sparse_weights[:, needle_start:needle_end].sum(dim=-1)
    mass_gain = sparse_mass / dense_mass.clamp_min(1e-30)
    full_lse = torch.logsumexp(scores.float(), dim=-1)
    kept_lse = torch.logsumexp(sparse_scores.float(), dim=-1)
    remote = ~static
    if bool(remote[needle_start:needle_end].any()):
        gold_token_score = scores[:, needle_start:needle_end].amax(dim=-1)
        token_rank = 1 + (scores > gold_token_score.unsqueeze(-1)).logical_and(
            remote.unsqueeze(0)
        ).sum(dim=-1)
        remote_tokens = int(remote.sum())
        token_percentile = 1.0 - (token_rank.float() - 1.0) / max(1, remote_tokens)
    else:
        token_rank = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        token_percentile = torch.full((scores.shape[0],), float("nan"), device=scores.device)

    hits: dict[str, list[bool]] = {}
    for budget in (8, 16, 32):
        selected_config = {**config, "top_blocks": budget}
        keep = _sparse_keep_mask(expanded, mode="score", config=selected_config)[0, :, 0]
        hits[str(budget)] = [
            bool(value)
            for value in keep[:, needle_start:needle_end].any(dim=-1).detach().cpu().tolist()
        ]
    return {
        "block_rank": [int(value) if gold_remote else None for value in ranks.detach().cpu().tolist()],
        "answer_token_rank": [
            int(value) if gold_remote else None for value in token_rank.detach().cpu().tolist()
        ],
        "answer_token_percentile": [
            float(value) if gold_remote else None for value in token_percentile.detach().cpu().tolist()
        ],
        "hit_at": hits,
        "margin": [float(value) if gold_remote else None for value in margin.detach().cpu().tolist()],
        "dense_answer_mass": [float(value) for value in dense_mass.detach().cpu().tolist()],
        "sparse_answer_mass_at_16": [float(value) for value in sparse_mass.detach().cpu().tolist()],
        "mass_gain_at_16": [float(value) for value in mass_gain.detach().cpu().tolist()],
        "removed_tail_log_normalizer": [
            float(value) for value in (full_lse - kept_lse).detach().cpu().tolist()
        ],
        "gold_is_static": not gold_remote,
        "gold_blocks": gold_blocks,
    }


@torch.inference_mode()
def _probe_one(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    *,
    needle_start: int,
    needle_end: int,
    config: Mapping[str, int],
) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    captured: dict[int, dict[str, Any]] = {}
    hooks = []

    def make_hook(layer_index: int):
        def hook(module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
            hidden = positional[0] if positional else keyword.get("hidden_states")
            embeddings = keyword.get("position_embeddings")
            if not torch.is_tensor(hidden) or not isinstance(embeddings, tuple):
                raise RuntimeError("attention hook did not receive hidden states and position embeddings")
            cos, sin = embeddings
            head_dim = int(module.head_dim)
            query = module.q_proj(hidden[:, -1:, :]).view(1, 1, -1, head_dim).transpose(1, 2)
            key = module.k_proj(hidden).view(1, hidden.shape[1], -1, head_dim).transpose(1, 2)
            query = _apply_rotary(query, cos[:, -1:, :], sin[:, -1:, :])
            key = _apply_rotary(key, cos, sin)
            if query.shape[1] % key.shape[1] != 0:
                raise RuntimeError("Phase 0 found a non-integer GQA head mapping")
            key = _repeat_kv(key, query.shape[1] // key.shape[1])
            scores = torch.matmul(query, key.transpose(2, 3))[0, :, 0] * float(module.scaling)
            captured[layer_index] = {
                "layer": layer_index,
                **_probe_metrics(
                    scores,
                    needle_start=needle_start,
                    needle_end=needle_end,
                    config=config,
                ),
            }

        return hook

    for _, module in _attention_modules(model):
        hooks.append(module.register_forward_pre_hook(make_hook(int(module.layer_idx)), with_kwargs=True))
    try:
        causal_backbone(model)(
            input_ids=tensor,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    if sorted(captured) != list(range(int(model.config.num_hidden_layers))):
        raise RuntimeError("Phase 0 did not capture every attention layer")
    del tensor, attention_mask
    return [captured[index] for index in sorted(captured)]


def _select_passkey_rows(
    rows: Sequence[dict[str, Any]],
    *,
    lengths: Sequence[int],
    trials: Sequence[int] | None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if int(row["target_length"]) in set(lengths)
        and (trials is None or int(row.get("source", {}).get("trial", -1)) in set(trials))
    ]
    expected = len(lengths) * 5 * (20 if trials is None else len(trials))
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} passkey rows; expected {expected}")
    return selected


def run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    _, all_rows = load_passkey_rows(args.passkey_root)
    lengths = _parse_ints(args.lengths)
    trials = _parse_ints(args.trials)
    rows = _select_passkey_rows(all_rows, lengths=lengths, trials=trials)
    model, tokenizer, identity = _load_arm_model(args)
    _set_attention_mode(model, "sdpa")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    for index, row in enumerate(rows, start=1):
        needle_start, needle_end = _needle_span(tokenizer, row)
        answer_start, answer_end = _answer_span(tokenizer, row)
        if not needle_start <= answer_start < answer_end <= needle_end:
            raise ValueError(f"answer span is outside the needle: {row['example_id']}")
        layers = _probe_one(
            model,
            row["prompt_ids"],
            needle_start=answer_start,
            needle_end=answer_end,
            config=SPARSE_CONFIG,
        )
        results.append(
            {
                "example_id": row["example_id"],
                "target_length": int(row["target_length"]),
                "depth_percent": float(row["depth_percent"]),
                "trial": int(row["source"]["trial"]),
                "prompt_sha256": row["prompt_sha256"],
                "needle_span": [needle_start, needle_end],
                "answer_span": [answer_start, answer_end],
                "layers": layers,
            }
        )
        print(
            json.dumps(
                {
                    "phase": 0,
                    "substrate": args.substrate,
                    "progress": f"{index}/{len(rows)}",
                    "length": row["target_length"],
                    "depth": row["depth_percent"],
                    "trial": row["source"]["trial"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output = {
        "schema": PHASE0_SCHEMA,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "adapter": identity,
        "passkey_sha256": PASSKEY_SHA256,
        "selection": {"lengths": list(lengths), "trials": list(trials)},
        "sparse_config": SPARSE_CONFIG,
        "query_contract": "last_prompt_token_predicting_first_answer_token",
        "script_sha256": _script_sha256(),
        "results": results,
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def _phase0_cases(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if document.get("schema") != PHASE0_SCHEMA:
        raise ValueError("Phase 0 raw schema mismatch")
    cases = {row["example_id"]: row for row in document["results"]}
    if len(cases) != len(document["results"]):
        raise ValueError("Phase 0 output has duplicate example IDs")
    return cases


def _head_values(case: Mapping[str, Any], key: str) -> dict[tuple[int, int], Any]:
    output = {}
    for layer in case["layers"]:
        values = layer[key]
        for head, value in enumerate(values):
            output[(int(layer["layer"]), head)] = value
    return output


def _phase0_gate_checks(lengths: Mapping[str, Any], gate_lengths: tuple[int, ...]) -> dict[str, bool]:
    if gate_lengths == (16384,):
        cell = lengths["16384"]
        return {
            "16k_evq_wins_at_least_8_of_10": cell["evq_wins"] >= 8,
            "16k_median_hit_delta_positive": cell["median_hit_delta_evq_minus_geo"] > 0,
            "16k_median_evq_hit_at_16_at_least_0p50": cell["median_evq_hit_at_16"] >= 0.50,
            "16k_median_geo_hit_at_16_at_most_0p25": cell["median_geo_hit_at_16"] <= 0.25,
        }
    if gate_lengths != (16384, 32768):
        raise ValueError("gate lengths must be 16384 or 16384,32768")
    return {
        "32k_evq_wins_at_least_8_of_10": lengths["32768"]["evq_wins"] >= 8,
        "32k_median_hit_delta_positive": lengths["32768"]["median_hit_delta_evq_minus_geo"] > 0,
        "32k_evq_mass_gain_at_least_1p5": lengths["32768"]["median_evq_mass_gain_at_16"] >= 1.5,
        "16k_no_hit_reversal": lengths["16384"]["median_hit_delta_evq_minus_geo"] >= 0,
    }


def summarize_phase0(args: argparse.Namespace) -> dict[str, Any]:
    geo_document = json.loads(args.geo.read_text(encoding="utf-8"))
    evq_document = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo_document.get("substrate") != "native_geo" or evq_document.get("substrate") != "evq_cosh":
        raise ValueError("Phase 0 summary received the wrong arms")
    if geo_document.get("selection") != evq_document.get("selection"):
        raise ValueError("Phase 0 arms used different selections")
    if geo_document.get("sparse_config") != evq_document.get("sparse_config"):
        raise ValueError("Phase 0 arms used different sparse configurations")
    geo = _phase0_cases(geo_document)
    evq = _phase0_cases(evq_document)
    if set(geo) != set(evq):
        raise ValueError("Phase 0 arms have different examples")

    reciprocal: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    for cases in (geo, evq):
        for case in cases.values():
            if int(case["target_length"]) != 8192:
                continue
            for head, rank in _head_values(case, "block_rank").items():
                if rank is not None:
                    reciprocal[head].append(1.0 / float(rank))
    if len(reciprocal) != 32 * 32:
        raise ValueError("8K calibration did not cover every layer/head")
    ranked_heads = sorted(
        reciprocal,
        key=lambda head: (-statistics.fmean(reciprocal[head]), head[0], head[1]),
    )
    retrieval_heads = ranked_heads[:32]

    gate_lengths = _parse_ints(args.gate_lengths)
    lengths: dict[str, Any] = {}
    for length in gate_lengths:
        paired = []
        for example_id in sorted(geo):
            if int(geo[example_id]["target_length"]) != length:
                continue
            arm_values = {}
            for label, cases in (("geo", geo), ("evq", evq)):
                ranks = _head_values(cases[example_id], "block_rank")
                gains = _head_values(cases[example_id], "mass_gain_at_16")
                selected_ranks = [ranks[head] for head in retrieval_heads if ranks[head] is not None]
                selected_gains = [gains[head] for head in retrieval_heads if ranks[head] is not None]
                if len(selected_ranks) != len(retrieval_heads):
                    raise ValueError(f"{example_id} has a static gold block at a test length")
                arm_values[label] = {
                    "hit_at_16": sum(rank <= 16 for rank in selected_ranks) / len(selected_ranks),
                    "mass_gain_at_16": statistics.median(selected_gains),
                }
            paired.append(
                {
                    "example_id": example_id,
                    **arm_values,
                    "hit_delta_evq_minus_geo": arm_values["evq"]["hit_at_16"]
                    - arm_values["geo"]["hit_at_16"],
                }
            )
        if len(paired) != 10:
            raise ValueError(f"Phase 0 gate requires 10 paired cases at {length}, found {len(paired)}")
        deltas = [row["hit_delta_evq_minus_geo"] for row in paired]
        lengths[str(length)] = {
            "paired_cases": len(paired),
            "evq_wins": sum(value > 0 for value in deltas),
            "median_evq_hit_at_16": statistics.median(
                row["evq"]["hit_at_16"] for row in paired
            ),
            "median_geo_hit_at_16": statistics.median(
                row["geo"]["hit_at_16"] for row in paired
            ),
            "median_hit_delta_evq_minus_geo": statistics.median(deltas),
            "median_evq_mass_gain_at_16": statistics.median(
                row["evq"]["mass_gain_at_16"] for row in paired
            ),
            "median_geo_mass_gain_at_16": statistics.median(
                row["geo"]["mass_gain_at_16"] for row in paired
            ),
            "cases": paired,
        }
    checks = _phase0_gate_checks(lengths, gate_lengths)
    output = {
        "schema": PHASE0_SUMMARY_SCHEMA,
        "status": "pass" if all(checks.values()) else "stop",
        "scope": "exploratory_16k" if gate_lengths == (16384,) else "preregistered_16k_32k",
        "checks": checks,
        "retrieval_head_contract": {
            "selection_length": 8192,
            "selection_metric": "pooled_geo_evq_mean_reciprocal_gold_block_rank",
            "count": len(retrieval_heads),
            "heads": [{"layer": layer, "head": head} for layer, head in retrieval_heads],
        },
        "lengths": lengths,
        "inputs": {"geo_sha256": sha256_file(args.geo), "evq_sha256": sha256_file(args.evq)},
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


def _configure_decode(
    model: torch.nn.Module,
    mode: str,
    *,
    retrieval_heads: Sequence[tuple[int, int]] | None = None,
    gold_span: Sequence[int] | None = None,
) -> None:
    _register_attention()
    _set_attention_mode(
        model,
        ATTENTION_IMPL,
        mode=mode,
        retrieval_heads=retrieval_heads,
        gold_span=gold_span,
    )


@torch.inference_mode()
def _prefill(model: torch.nn.Module, prompt_ids: Sequence[int]) -> tuple[Any, int]:
    if len(prompt_ids) < 2:
        raise ValueError("decode experiment requires at least two prompt tokens")
    _set_attention_mode(model, "sdpa")
    device = torch.device("cuda")
    prefix = torch.tensor([list(prompt_ids[:-1])], dtype=torch.long, device=device)
    output = causal_backbone(model)(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        use_cache=True,
        return_dict=True,
    )
    return output.past_key_values, int(prompt_ids[-1])


@torch.inference_mode()
def _decode_logits(
    model: torch.nn.Module,
    past: Any,
    token_id: int,
    *,
    seen_tokens: int,
) -> tuple[torch.Tensor, Any]:
    device = torch.device("cuda")
    token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, seen_tokens + 1), dtype=torch.long, device=device)
    output = model(
        input_ids=token,
        attention_mask=attention_mask,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    return output.logits[:, -1].float(), output.past_key_values


@torch.inference_mode()
def _target_rank(logits: torch.Tensor, label: int) -> int:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("target rank requires [1, vocab] logits")
    if not 0 <= int(label) < logits.shape[-1]:
        raise ValueError("target label is outside the vocabulary")
    target_logit = logits[0, int(label)]
    return int((logits[0] > target_logit).sum().item()) + 1


def causal_delta_rank(
    full_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    labels: torch.Tensor | Sequence[int] | int,
) -> torch.Tensor:
    """Strict rank of each label under ``full_logits - ablated_logits``."""
    if full_logits.shape != ablated_logits.shape or full_logits.ndim < 2:
        raise ValueError("full and ablated logits must share [..., vocab] shape")
    delta = full_logits.float() - ablated_logits.float()
    target_shape = delta.shape[:-1]
    target_ids = torch.as_tensor(labels, dtype=torch.long, device=delta.device)
    while target_ids.ndim < len(target_shape):
        target_ids = target_ids.unsqueeze(-1)
    try:
        target_ids = torch.broadcast_to(target_ids, target_shape)
    except RuntimeError as exc:
        raise ValueError("labels do not broadcast over the non-vocabulary axes") from exc
    if bool(((target_ids < 0) | (target_ids >= delta.shape[-1])).any()):
        raise ValueError("causal-delta label is outside the vocabulary")
    target = delta.gather(-1, target_ids.unsqueeze(-1))
    return (delta > target).sum(dim=-1) + 1


def per_layer_logit_lens(
    hidden_states: torch.Tensor,
    *,
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Apply the model's final norm and unembedding to per-layer states."""
    if hidden_states.ndim < 2:
        raise ValueError("per-layer hidden states require [..., layers, hidden] axes")
    return lm_head(norm(hidden_states)).float()


def per_layer_logit_lens_delta(
    full_hidden_states: torch.Tensor,
    ablated_hidden_states: torch.Tensor,
    *,
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Compute ``W_U Norm(h_full_l) - W_U Norm(h_ablate_l)`` per layer."""
    if full_hidden_states.shape != ablated_hidden_states.shape:
        raise ValueError("full and ablated per-layer hidden states must have equal shape")
    return per_layer_logit_lens(
        full_hidden_states, norm=norm, lm_head=lm_head
    ) - per_layer_logit_lens(ablated_hidden_states, norm=norm, lm_head=lm_head)


@torch.inference_mode()
def _answer_nll(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    mode: str,
    retrieval_heads: Sequence[tuple[int, int]] | None = None,
    gold_span: Sequence[int] | None = None,
) -> dict[str, float | int]:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(
        model,
        mode,
        retrieval_heads=retrieval_heads,
        gold_span=gold_span,
    )
    seen = len(prompt_ids) - 1
    total = 0.0
    ranks = []
    try:
        for label in answer_ids:
            logits, past = _decode_logits(model, past, current, seen_tokens=seen)
            target = torch.tensor([int(label)], dtype=torch.long, device=logits.device)
            total += float(F.cross_entropy(logits, target, reduction="sum").double().cpu())
            ranks.append(_target_rank(logits, int(label)))
            current = int(label)
            seen += 1
    finally:
        del past
    return {
        "nll_sum": total,
        "answer_tokens": len(answer_ids),
        "mean_logprob": -total / len(answer_ids),
        "first_token_rank": ranks[0],
        "mean_token_rank": statistics.fmean(ranks),
        "max_token_rank": max(ranks),
        "top_10_fraction": statistics.fmean(rank <= 10 for rank in ranks),
        "top_100_fraction": statistics.fmean(rank <= 100 for rank in ranks),
        "top_1000_fraction": statistics.fmean(rank <= 1000 for rank in ranks),
    }


def _counterfactual_prompts(
    prompt_ids: Sequence[int],
    *,
    needle_span: tuple[int, int],
    answer_span: tuple[int, int],
    original_answer_ids: Sequence[int],
    swapped_answer_ids: Sequence[int],
) -> dict[str, list[int]]:
    """Build equal-length source-swapped and source-removed token prompts."""
    prompt = [int(value) for value in prompt_ids]
    needle_start, needle_end = needle_span
    answer_start, answer_end = answer_span
    original = [int(value) for value in original_answer_ids]
    swapped_answer = [int(value) for value in swapped_answer_ids]
    if not (0 <= needle_start <= answer_start < answer_end <= needle_end <= len(prompt)):
        raise ValueError("counterfactual spans are invalid")
    if prompt[answer_start:answer_end] != original:
        raise ValueError("registered answer span does not match the original answer")
    if len(original) != len(swapped_answer) or original == swapped_answer:
        raise ValueError("counterfactual answers must be distinct and token-length matched")

    swapped = list(prompt)
    swapped[answer_start:answer_end] = swapped_answer
    if _find_subsequence(swapped, original):
        raise ValueError("source-swapped prompt still contains the original answer")

    width = needle_end - needle_start
    filler = None
    for start in range(0, len(prompt) - width + 1):
        end = start + width
        if start < needle_end and needle_start < end:
            continue
        candidate = prompt[start:end]
        if _find_subsequence(candidate, original) or _find_subsequence(
            candidate, swapped_answer
        ):
            continue
        filler = candidate
        break
    if filler is None:
        raise ValueError("no answer-free equal-length filler span exists")
    removed = list(prompt)
    removed[needle_start:needle_end] = filler
    if _find_subsequence(removed, original) or _find_subsequence(removed, swapped_answer):
        raise ValueError("source-removed prompt still contains a registered answer")
    if not (len(prompt) == len(swapped) == len(removed)):
        raise AssertionError("counterfactual prompt lengths differ")
    return {"original": prompt, "swapped": swapped, "source_removed": removed}


def _mean_nll(score: Mapping[str, float | int]) -> float:
    return float(score["nll_sum"]) / int(score["answer_tokens"])


def _counterfactual_pairs(
    tokenizer: Any, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["target_length"]), float(row["depth_percent"]))].append(row)
    pairs = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["source"]["trial"]))
        if len(group) != 2:
            raise ValueError(f"counterfactual canary requires two trials per cell: {key}")
        for row, swap_row in ((group[0], group[1]), (group[1], group[0])):
            original_ids = _answer_ids(tokenizer, str(row["answers"][0]))
            swapped_ids = _answer_ids(tokenizer, str(swap_row["answers"][0]))
            prompts = _counterfactual_prompts(
                row["prompt_ids"],
                needle_span=_needle_span(tokenizer, row),
                answer_span=_answer_span(tokenizer, row),
                original_answer_ids=original_ids,
                swapped_answer_ids=swapped_ids,
            )
            pairs.append(
                {
                    "example_id": row["example_id"],
                    "target_length": int(row["target_length"]),
                    "depth_percent": float(row["depth_percent"]),
                    "trial": int(row["source"]["trial"]),
                    "original_answer_ids": original_ids,
                    "swapped_answer_ids": swapped_ids,
                    "prompts": prompts,
                }
            )
    return pairs


def run_counterfactual_canary(args: argparse.Namespace) -> dict[str, Any]:
    _, all_rows = load_passkey_rows(args.passkey_root)
    rows = _select_passkey_rows(
        all_rows,
        lengths=_parse_ints(args.lengths),
        trials=_parse_ints(args.trials),
    )
    model, tokenizer, identity = _load_arm_model(args)
    if args.runtime_frequency != "trained":
        geometry = resolve_model_rope_geometry(model.config)
        if args.runtime_frequency == "native_geo":
            runtime_inv_freq, _ = build_training_inv_freq(
                rope_method="native_geo",
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=1.414,
            )
        elif args.runtime_frequency == "midpoint_geo":
            runtime_inv_freq = compute_evq_cosh_inv_freq(
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=0.0,
                midpoint=True,
            )
        else:
            runtime_inv_freq, _ = build_training_inv_freq(
                rope_method="evq_cosh",
                head_dim=geometry.head_dim,
                base=geometry.rope_base,
                tau=1.414,
            )
        inject_inv_freq(model, runtime_inv_freq)
        for name, module in find_rotary_modules(model):
            if not hasattr(module, "attention_scaling"):
                raise RuntimeError(f"rotary module {name} has no attention_scaling")
            module.attention_scaling = 1.0
            original = getattr(module, "original_inv_freq", None)
            if torch.is_tensor(original):
                if original.shape != module.inv_freq.shape:
                    raise RuntimeError(f"original_inv_freq shape mismatch at {name}")
                original.copy_(module.inv_freq)
        verify_model_inv_freq(model, runtime_inv_freq)
    pairs = _counterfactual_pairs(tokenizer, rows)
    if args.disable_adapter and not args.arm_name.startswith("base_"):
        raise ValueError("only base arms may disable the adapter")
    context = model.disable_adapter() if args.disable_adapter else nullcontext()
    results = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with context:
        for index, pair in enumerate(pairs, start=1):
            original_ids = pair["original_answer_ids"]
            swapped_ids = pair["swapped_answer_ids"]
            prompts = pair["prompts"]
            scores = {
                "original_own": _answer_nll(
                    model, prompts["original"], original_ids, mode="dense"
                ),
                "original_swapped": _answer_nll(
                    model, prompts["original"], swapped_ids, mode="dense"
                ),
                "swapped_own": _answer_nll(
                    model, prompts["swapped"], swapped_ids, mode="dense"
                ),
                "swapped_original": _answer_nll(
                    model, prompts["swapped"], original_ids, mode="dense"
                ),
                "removed_original": _answer_nll(
                    model, prompts["source_removed"], original_ids, mode="dense"
                ),
            }
            original_prefers_own = _mean_nll(scores["original_own"]) < _mean_nll(
                scores["original_swapped"]
            )
            swapped_prefers_own = _mean_nll(scores["swapped_own"]) < _mean_nll(
                scores["swapped_original"]
            )
            removal_delta = _mean_nll(scores["removed_original"]) - _mean_nll(
                scores["original_own"]
            )
            results.append(
                {
                    "example_id": pair["example_id"],
                    "target_length": pair["target_length"],
                    "depth_percent": pair["depth_percent"],
                    "trial": pair["trial"],
                    "prompt_sha256": {
                        name: _prompt_sha256(prompt) for name, prompt in prompts.items()
                    },
                    "scores": scores,
                    "original_prefers_own": original_prefers_own,
                    "swapped_prefers_own": swapped_prefers_own,
                    "pair_consistent": original_prefers_own and swapped_prefers_own,
                    "source_removal_delta_nll": removal_delta,
                    "source_removal_positive": removal_delta > 0.0,
                }
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm_name,
                        "progress": f"{index}/{len(pairs)}",
                        "depth": pair["depth_percent"],
                        "pair_consistent": results[-1]["pair_consistent"],
                        "source_removal_delta_nll": removal_delta,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    aggregate = {
        "cases": len(results),
        "pair_consistency": statistics.fmean(
            float(row["pair_consistent"]) for row in results
        ),
        "source_removal_positive_fraction": statistics.fmean(
            float(row["source_removal_positive"]) for row in results
        ),
        "source_removal_delta_nll_mean": statistics.fmean(
            float(row["source_removal_delta_nll"]) for row in results
        ),
        "source_removal_delta_nll_median": statistics.median(
            float(row["source_removal_delta_nll"]) for row in results
        ),
        "original_answer_nll": statistics.fmean(
            _mean_nll(row["scores"]["original_own"]) for row in results
        ),
    }
    output = {
        "schema": CANARY_SCHEMA,
        "status": "complete",
        "arm": args.arm_name,
        "raw_extrapolation": True,
        "runtime_frequency": args.runtime_frequency,
        "adapter_enabled": not args.disable_adapter,
        "identity": identity,
        "selection": {
            "lengths": list(_parse_ints(args.lengths)),
            "trials": list(_parse_ints(args.trials)),
        },
        "aggregate": aggregate,
        "results": results,
        "single_seed_supporting": True,
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def summarize_counterfactual_canary(args: argparse.Namespace) -> dict[str, Any]:
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (
            ("base_native", args.base_native),
            ("geo_lora_native", args.geo_lora_native),
            ("evq_lora_evq", args.evq_lora_evq),
        )
    }
    for name, document in documents.items():
        if document.get("schema") != CANARY_SCHEMA or document.get("arm") != name:
            raise ValueError(f"wrong canary arm: {name}")
    ids = [{row["example_id"] for row in document["results"]} for document in documents.values()]
    if ids[0] != ids[1] or ids[0] != ids[2]:
        raise ValueError("canary arms scored different examples")
    aggregate = {name: document["aggregate"] for name, document in documents.items()}
    checks = {
        "base_pair_consistency_at_least_0p60": aggregate["base_native"][
            "pair_consistency"
        ]
        >= 0.60,
        "geo_pair_consistency_at_least_0p60": aggregate["geo_lora_native"][
            "pair_consistency"
        ]
        >= 0.60,
        "base_source_removal_positive_at_least_0p60": aggregate["base_native"][
            "source_removal_positive_fraction"
        ]
        >= 0.60,
        "geo_source_removal_positive_at_least_0p60": aggregate["geo_lora_native"][
            "source_removal_positive_fraction"
        ]
        >= 0.60,
    }
    output = {
        "schema": CANARY_SUMMARY_SCHEMA,
        "status": "pass" if all(checks.values()) else "stop",
        "checks": checks,
        "aggregate": aggregate,
        "evq_minus_geo": {
            "pair_consistency": aggregate["evq_lora_evq"]["pair_consistency"]
            - aggregate["geo_lora_native"]["pair_consistency"],
            "source_removal_positive_fraction": aggregate["evq_lora_evq"][
                "source_removal_positive_fraction"
            ]
            - aggregate["geo_lora_native"]["source_removal_positive_fraction"],
            "source_removal_delta_nll_mean": aggregate["evq_lora_evq"][
                "source_removal_delta_nll_mean"
            ]
            - aggregate["geo_lora_native"]["source_removal_delta_nll_mean"],
            "original_answer_nll": aggregate["evq_lora_evq"]["original_answer_nll"]
            - aggregate["geo_lora_native"]["original_answer_nll"],
        },
        "inputs": {
            name: sha256_file(path)
            for name, path in (
                ("base_native", args.base_native),
                ("geo_lora_native", args.geo_lora_native),
                ("evq_lora_evq", args.evq_lora_evq),
            )
        },
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    mode: str,
    max_new_tokens: int,
    retrieval_heads: Sequence[tuple[int, int]] | None = None,
    gold_span: Sequence[int] | None = None,
) -> dict[str, Any]:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(
        model,
        mode,
        retrieval_heads=retrieval_heads,
        gold_span=gold_span,
    )
    seen = len(prompt_ids) - 1
    generated = []
    eos = tokenizer.eos_token_id
    try:
        for _ in range(max_new_tokens):
            logits, past = _decode_logits(model, past, current, seen_tokens=seen)
            current = int(logits.argmax(dim=-1).item())
            generated.append(current)
            seen += 1
            if eos is not None and current == int(eos):
                break
    finally:
        del past
    eos_terminated = eos is not None and generated and generated[-1] == int(eos)
    content = generated[:-1] if eos_terminated else generated
    return {
        "prediction": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "generated_ids": generated,
        "generated_token_count": len(content),
        "eos_terminated": eos_terminated,
    }


@torch.inference_mode()
def _first_step_logits(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    *,
    mode: str,
) -> torch.Tensor:
    past, current = _prefill(model, prompt_ids)
    _configure_decode(model, mode)
    try:
        logits, past = _decode_logits(
            model,
            past,
            current,
            seen_tokens=len(prompt_ids) - 1,
        )
        return logits.detach().cpu()
    finally:
        del past


@torch.inference_mode()
def _teacher_forced_logit_trace_from_prefill(
    model: torch.nn.Module,
    past: Any,
    current_token: int,
    answer_ids: Sequence[int],
    *,
    seen_tokens: int,
    mode: str,
    gold_span: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Trace every answer position and layer from an immutable prefill cache."""
    labels = tuple(int(label) for label in answer_ids)
    if not labels:
        raise ValueError("readout trace requires at least one answer token")
    backbone = causal_backbone(model)
    layers = getattr(backbone, "layers", None)
    norm = getattr(backbone, "norm", None)
    lm_head = model.get_output_embeddings()
    if not isinstance(layers, torch.nn.ModuleList) or norm is None or lm_head is None:
        raise RuntimeError("readout trace requires Llama-style layers, final norm, and lm_head")

    branch_past = copy.deepcopy(past)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise RuntimeError("decoder-layer hook did not receive [batch, tokens, hidden]")
            captured[layer_index] = hidden[:, -1].detach()

        return hook

    for index, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(index)))

    _configure_decode(model, mode, gold_span=gold_span)
    current = int(current_token)
    seen = int(seen_tokens)
    traces = []
    parity = []
    try:
        for label in labels:
            captured.clear()
            final_logits, branch_past = _decode_logits(
                model,
                branch_past,
                current,
                seen_tokens=seen,
            )
            if sorted(captured) != list(range(len(layers))):
                raise RuntimeError("readout trace did not capture every decoder layer")
            hidden = torch.stack([captured[index] for index in range(len(layers))])
            layer_logits = per_layer_logit_lens(hidden, norm=norm, lm_head=lm_head).squeeze(1)
            parity_logits = per_layer_logit_lens(
                hidden[-1:], norm=norm, lm_head=lm_head
            ).squeeze(1)
            max_abs = float((parity_logits[0] - final_logits[0]).abs().max().cpu())
            if max_abs > 1e-4:
                raise RuntimeError(f"final logit-lens parity failed: max_abs={max_abs}")
            layer_logits[-1] = final_logits[0]
            traces.append(layer_logits.detach().to(device="cpu", dtype=torch.bfloat16))
            parity.append(max_abs)
            current = int(label)
            seen += 1
    finally:
        for hook in hooks:
            hook.remove()
        del branch_past
    return {
        "logits": torch.stack(traces),
        "layer_indices": torch.arange(len(layers), dtype=torch.int16),
        "final_logit_parity_max_abs": max(parity),
    }


@torch.inference_mode()
def _paired_teacher_forced_logit_traces(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    *,
    gold_span: Sequence[int],
) -> dict[str, Any]:
    """Share one dense prefill across full and all-head gold-ablation branches."""
    past, current = _prefill(model, prompt_ids)
    try:
        full = _teacher_forced_logit_trace_from_prefill(
            model,
            past,
            current,
            answer_ids,
            seen_tokens=len(prompt_ids) - 1,
            mode="dense",
            gold_span=gold_span,
        )
        ablated = _teacher_forced_logit_trace_from_prefill(
            model,
            past,
            current,
            answer_ids,
            seen_tokens=len(prompt_ids) - 1,
            mode="gold_drop_all",
            gold_span=gold_span,
        )
    finally:
        del past
    if not torch.equal(full["layer_indices"], ablated["layer_indices"]):
        raise RuntimeError("full and ablated traces captured different layers")
    return {"full": full, "ablated": ablated}


def run_readout_trace(args: argparse.Namespace) -> dict[str, Any]:
    """Persist the minimal dense/gold-ablated tensors needed for readout analysis."""
    output_dir = args.output
    temporary_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else temporary_dir)

    _, all_rows = load_passkey_rows(args.data_root)
    lengths = _parse_ints(args.lengths)
    trials = _parse_ints(args.trials)
    rows = _select_passkey_rows(all_rows, lengths=lengths, trials=trials)
    model, tokenizer, identity = _load_arm_model(args)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    records = []
    temporary_dir.mkdir(parents=True)
    try:
        for index, row in enumerate(rows):
            if len(row["answers"]) != 1:
                raise ValueError("readout trace requires one registered gold answer per case")
            answer_ids = _answer_ids(tokenizer, row["answers"][0])
            gold_span = _answer_span(tokenizer, row)
            traces = _paired_teacher_forced_logit_traces(
                model,
                row["prompt_ids"],
                answer_ids,
                gold_span=gold_span,
            )
            record_name = f"records/{index:03d}_{row['prompt_sha256']}.pt"
            record_path = temporary_dir / record_name
            _atomic_torch(
                record_path,
                {
                    "schema": READOUT_TRACE_SCHEMA,
                    "substrate": args.substrate,
                    "prompt_sha256": row["prompt_sha256"],
                    "target_length": int(row["target_length"]),
                    "depth_percent": float(row["depth_percent"]),
                    "gold_token_ids": torch.tensor(answer_ids, dtype=torch.int32),
                    "layer_indices": traces["full"]["layer_indices"],
                    "full_logits": traces["full"]["logits"],
                    "ablated_logits": traces["ablated"]["logits"],
                    "final_logit_parity_max_abs": max(
                        traces["full"]["final_logit_parity_max_abs"],
                        traces["ablated"]["final_logit_parity_max_abs"],
                    ),
                },
            )
            shape = list(traces["full"]["logits"].shape)
            records.append(
                {
                    "file": record_name,
                    "sha256": sha256_file(record_path),
                    "size_bytes": record_path.stat().st_size,
                    "prompt_sha256": row["prompt_sha256"],
                    "target_length": int(row["target_length"]),
                    "depth_percent": float(row["depth_percent"]),
                    "shape": shape,
                }
            )
            print(
                json.dumps(
                    {
                        "command": "readout-trace",
                        "substrate": args.substrate,
                        "progress": f"{index + 1}/{len(rows)}",
                        "length": row["target_length"],
                        "depth": row["depth_percent"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        manifest = {
            "schema": READOUT_TRACE_MANIFEST_SCHEMA,
            "status": "complete",
            "measurement_label": "oracle-diagnostic",
            "single_seed_supporting": True,
            "substrate": args.substrate,
            "selection": {"lengths": list(lengths), "trials": list(trials)},
            "passkey_sha256": PASSKEY_SHA256,
            "adapter": {
                key: identity.get(key)
                for key in (
                    "substrate",
                    "adapter_sha256",
                    "model_manifest_sha256",
                    "training_manifest_sha256",
                    "protocol_sha256",
                    "code_sha256",
                )
            },
            "tensor_contract": {
                "logits": "bfloat16[answer_position,decoder_layer,vocabulary]",
                "full": "dense answer-side attention after one shared dense prefill",
                "ablated": "gold_drop_all answer-side attention after the same dense prefill",
                "rank": "1 + count(delta_logit > target_delta_logit)",
            },
            "records": records,
            "script_sha256": _script_sha256(),
            "runtime": {
                "seconds": time.time() - started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "cuda_device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
            },
        }
        _atomic_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, output_dir)
    except Exception:
        raise
    return manifest


def run_association_swap_trace(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output
    temporary_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else temporary_dir)

    case_manifest, cases = load_association_swap_cases(args.cases_root)
    if case_manifest.get("training_data_manifest_sha256") != sha256_file(
        args.training_data_manifest
    ):
        raise ValueError("association-swap cases use a different training manifest")
    model, _, identity = _load_arm_model(args)
    if case_manifest.get("tokenizer") != tokenizer_source_fingerprint(args.model_name):
        raise ValueError("association-swap cases use a different model tokenizer")

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    records = []
    actual_input_tokens = 0
    temporary_dir.mkdir(parents=True)
    for index, case in enumerate(cases):
        prompts = case["prompt_ids"]
        spans = case["gold_spans"]
        candidates = case["candidate_first_token_ids"]
        condition_traces = []
        for condition in range(2):
            prompt_ids = [int(value) for value in prompts[condition].tolist()]
            gold_span = [int(value) for value in spans[condition].tolist()]
            label = int(candidates[condition])
            condition_traces.append(
                _paired_teacher_forced_logit_traces(
                    model,
                    prompt_ids,
                    [label],
                    gold_span=gold_span,
                )
            )
            actual_input_tokens += len(prompt_ids) - 1 + 2
        layer_indices = condition_traces[0]["full"]["layer_indices"]
        if any(
            not torch.equal(layer_indices, traces["full"]["layer_indices"])
            or not torch.equal(layer_indices, traces["ablated"]["layer_indices"])
            for traces in condition_traces
        ):
            raise RuntimeError("association-swap conditions captured different layers")
        full_logits = torch.stack(
            [traces["full"]["logits"][0] for traces in condition_traces]
        )
        ablated_logits = torch.stack(
            [traces["ablated"]["logits"][0] for traces in condition_traces]
        )
        pair_sha256 = str(case["pair_sha256"])
        record_name = f"records/{index:03d}_{pair_sha256}.pt"
        record_path = temporary_dir / record_name
        parity = max(
            max(
                traces["full"]["final_logit_parity_max_abs"],
                traces["ablated"]["final_logit_parity_max_abs"],
            )
            for traces in condition_traces
        )
        _atomic_torch(
            record_path,
            {
                "schema": ASSOCIATION_SWAP_TRACE_SCHEMA,
                "substrate": args.substrate,
                "pair_sha256": pair_sha256,
                "split": str(case["split"]),
                "depth_percent": float(case["depth_percent"]),
                "candidate_first_token_ids": candidates.to(torch.int32),
                "layer_indices": layer_indices,
                "full_logits": full_logits,
                "ablated_logits": ablated_logits,
                "final_logit_parity_max_abs": float(parity),
            },
        )
        records.append(
            {
                "file": record_name,
                "sha256": sha256_file(record_path),
                "size_bytes": record_path.stat().st_size,
                "pair_sha256": pair_sha256,
                "split": str(case["split"]),
                "depth_percent": float(case["depth_percent"]),
                "shape": list(full_logits.shape),
            }
        )
        print(
            json.dumps(
                {
                    "command": "association-swap-trace",
                    "substrate": args.substrate,
                    "progress": f"{index + 1}/{len(cases)}",
                    "split": case["split"],
                    "depth": case["depth_percent"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    expected_input_tokens = 256 * 2 * (16384 - 1 + 2)
    if actual_input_tokens != expected_input_tokens:
        raise RuntimeError(
            f"association-swap input budget mismatch: {actual_input_tokens}"
        )
    manifest = {
        "schema": ASSOCIATION_SWAP_MANIFEST_SCHEMA,
        "status": "complete",
        "measurement_label": "oracle-diagnostic",
        "single_seed_supporting": True,
        "paper_claim": False,
        "substrate": args.substrate,
        "case_manifest_sha256": sha256_file(args.cases_root / "manifest.json"),
        "pair_set_sha256": case_manifest["pair_set_sha256"],
        "selection": {
            "seed": 42,
            "target_length": 16384,
            "dev_pairs": 128,
            "test_pairs": 128,
        },
        "adapter": {
            key: identity.get(key)
            for key in (
                "substrate",
                "adapter_sha256",
                "model_manifest_sha256",
                "training_manifest_sha256",
                "protocol_sha256",
                "code_sha256",
            )
        },
        "tensor_contract": {
            "logits": "bfloat16[query_condition,decoder_layer,vocabulary]",
            "query_condition": ["query_a", "query_b"],
            "full": "dense answer-side attention after one shared dense prefill",
            "ablated": "gold_drop_all answer-side attention after the same dense prefill",
        },
        "budget": {
            "shared_16k_prefills_this_arm": 512,
            "actual_input_tokens_this_arm": actual_input_tokens,
            "matched_two_arm_actual_input_tokens": 2 * actual_input_tokens,
        },
        "records": records,
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(temporary_dir / "manifest.json", manifest)
    os.replace(temporary_dir, output_dir)
    return manifest


def _phase1_rows(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    lengths = _parse_ints(args.lengths)
    if not set(lengths) <= {16384, 32768}:
        raise ValueError("Phase 1 lengths must be 16384 and/or 32768")
    if args.dataset == "passkey":
        _, rows = load_passkey_rows(args.data_root)
        trials = (0, 1) if args.selection == "pilot" else None
        selected = _select_passkey_rows(rows, lengths=lengths, trials=trials)
        normalized = []
        for row in selected:
            normalized.append(
                {
                    **row,
                    "metric": "exact_match",
                    "generation_tokens": 32,
                    "scorer": "normalized_exact_match",
                }
            )
        return PASSKEY_SHA256, normalized

    manifest, rows = load_capability_suite(args.data_root)
    selected = []
    for row in rows:
        if int(row["target_length"]) not in set(lengths):
            continue
        if row["suite"] == "ruler" and str(row["task"]).startswith("niah_"):
            selected.append(row)
        elif row["suite"] in {"nolima_hard_exact_context", "longbench"}:
            selected.append(row)
    if not selected:
        raise ValueError("retrieval suite selection is empty")
    return sha256_file(args.data_root / "manifest.json"), selected


def _score_phase1_row(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    mode: str,
    retrieval_heads: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    selected_head_modes = {"head_score", "gold_drop", "oracle_gold"}
    if mode in selected_head_modes and not retrieval_heads:
        raise ValueError(f"{mode} requires frozen retrieval heads")
    gold_span = (
        _answer_span(tokenizer, row)
        if mode in {"gold_drop", "gold_drop_all", "oracle_gold", "oracle_include_all"}
        else None
    )
    answer_scores = [
        _answer_nll(
            model,
            row["prompt_ids"],
            _answer_ids(tokenizer, answer),
            mode=mode,
            retrieval_heads=retrieval_heads,
            gold_span=gold_span,
        )
        for answer in row["answers"]
    ]
    selected_index = max(
        range(len(answer_scores)), key=lambda index: answer_scores[index]["mean_logprob"]
    )
    selected = answer_scores[selected_index]
    generation = _generate(
        model,
        tokenizer,
        row["prompt_ids"],
        mode=mode,
        max_new_tokens=int(row["generation_tokens"]),
        retrieval_heads=retrieval_heads,
        gold_span=gold_span,
    )
    metric_score = score_capability_prediction(
        row["metric"],
        generation["prediction"],
        row["answers"],
        source=row.get("source"),
    )
    generation.update(
        score_generation_metrics(
            generation["prediction"],
            row["answers"],
            eos_terminated=bool(generation["eos_terminated"]),
            generated_token_count=int(generation["generated_token_count"]),
        )
    )
    return {
        "example_id": row["example_id"],
        "suite": row["suite"],
        "task": row["task"],
        "target_length": int(row["target_length"]),
        "depth_percent": row.get("depth_percent"),
        "prompt_sha256": row["prompt_sha256"],
        "metric": row["metric"],
        "mode": mode,
        "nll_sum": selected["nll_sum"],
        "answer_tokens": selected["answer_tokens"],
        "mean_logprob": selected["mean_logprob"],
        "first_token_rank": selected["first_token_rank"],
        "mean_token_rank": selected["mean_token_rank"],
        "max_token_rank": selected["max_token_rank"],
        "top_10_fraction": selected["top_10_fraction"],
        "top_100_fraction": selected["top_100_fraction"],
        "top_1000_fraction": selected["top_1000_fraction"],
        "metric_score": metric_score,
        "selected_reference_index": selected_index,
        "reference_mean_logprobs": [score["mean_logprob"] for score in answer_scores],
        "references": list(row["answers"]),
        "generation": generation,
    }


def run_phase1(args: argparse.Namespace) -> dict[str, Any]:
    gate = json.loads(args.phase0_gate.read_text(encoding="utf-8"))
    if gate.get("schema") != PHASE0_SUMMARY_SCHEMA or gate.get("status") != "pass":
        raise ValueError("Phase 1 requires a passing Phase 0 gate")
    data_hash, rows = _phase1_rows(args)
    target_lengths = sorted({int(row["target_length"]) for row in rows})
    if any(str(length) not in gate.get("lengths", {}) for length in target_lengths):
        raise ValueError("Phase 1 requested a length not covered by the passing gate")
    modes = tuple(piece.strip() for piece in args.modes.split(",") if piece.strip())
    allowed_modes = {
        "dense",
        "score",
        "fixed",
        "head_score",
        "gold_drop",
        "gold_drop_all",
        "oracle_gold",
        "oracle_include_all",
    }
    if not modes or any(mode not in allowed_modes for mode in modes):
        raise ValueError(f"Phase 1 modes must be drawn from {sorted(allowed_modes)}")
    head_records = gate.get("retrieval_head_contract", {}).get("heads")
    if not isinstance(head_records, list) or len(head_records) != 32:
        raise ValueError("Phase 1 requires exactly 32 frozen retrieval heads")
    retrieval_heads = tuple((int(row["layer"]), int(row["head"])) for row in head_records)
    if len(set(retrieval_heads)) != len(retrieval_heads):
        raise ValueError("Phase 1 retrieval-head contract contains duplicates")
    if any(
        mode in {"gold_drop", "gold_drop_all", "oracle_gold", "oracle_include_all"}
        for mode in modes
    ) and args.dataset != "passkey":
        raise ValueError("gold-block causal modes require registered passkey needle spans")
    if args.chat_wrap_passkey and (args.dataset != "passkey" or set(modes) != {"dense"}):
        raise ValueError("chat-wrap diagnostic is restricted to dense passkey evaluation")
    model, tokenizer, identity = _load_arm_model(args)
    prompt_transform = {"name": "frozen_prompt_ids"}
    if args.chat_wrap_passkey:
        rows, prompt_transform = _chat_wrap_passkey_rows(tokenizer, rows)
    _register_attention()
    sanity_row = min(rows, key=lambda row: int(row["target_length"]))
    dense_logits = _first_step_logits(model, sanity_row["prompt_ids"], mode="dense")
    full_logits = _first_step_logits(model, sanity_row["prompt_ids"], mode="full")
    max_abs = float((dense_logits - full_logits).abs().max())
    sanity = {
        "example_id": sanity_row["example_id"],
        "max_abs_logit_difference": max_abs,
        "top1_equal": int(dense_logits.argmax()) == int(full_logits.argmax()),
        "tolerance": 1e-6,
    }
    if max_abs > sanity["tolerance"] or not sanity["top1_equal"]:
        raise RuntimeError(f"full-budget sparse attention differs from dense: {sanity}")
    del dense_logits, full_logits

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    total = len(rows) * len(modes)
    progress = 0
    for row in rows:
        for mode in modes:
            progress += 1
            result = _score_phase1_row(
                model,
                tokenizer,
                row,
                mode=mode,
                retrieval_heads=retrieval_heads,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": 1,
                        "substrate": args.substrate,
                        "progress": f"{progress}/{total}",
                        "mode": mode,
                        "task": row["task"],
                        "length": row["target_length"],
                        "metric_score": result["metric_score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    output = {
        "schema": PHASE1_SCHEMA,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "dataset": args.dataset,
        "selection": args.selection,
        "target_lengths": target_lengths,
        "data_sha256": data_hash,
        "prompt_transform": prompt_transform,
        "phase0_gate_sha256": sha256_file(args.phase0_gate),
        "retrieval_head_contract": gate["retrieval_head_contract"],
        "adapter": identity,
        "sparse_config": SPARSE_CONFIG,
        "operator_contract": {
            "scope": "answer_side_decode_after_dense_prompt_prefill",
            "score": "exact_per_query_head_max_qk_per_128_token_block",
            "head_score": "score_sparse_only_on_32_frozen_retrieval_heads_other_heads_dense",
            "gold_drop": "remove_gold_128_token_block_only_on_frozen_retrieval_heads",
            "gold_drop_all": "remove_gold_128_token_block_on_all_attention_heads",
            "oracle_gold": "retain_only_local_sink_and_gold_block_on_frozen_retrieval_heads",
            "oracle_include_all": "force_gold_block_into_all_head_score_selection_by_replacing_lowest_selected_block",
            "position_handling": "mask_only_original_rotary_kv_indices_no_reordering",
            "efficiency_claim": False,
            "fixed_control": "same_remote_block_budget_uniform_content_independent_blocks",
        },
        "full_budget_sanity": sanity,
        "script_sha256": _script_sha256(),
        "results": results,
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def run_raw_capability(args: argparse.Namespace) -> dict[str, Any]:
    """Score the frozen long-context suite without range scaling or a Phase-0 gate."""
    manifest, rows = load_capability_suite(args.data_root)
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    if manifest.get("tokenizer", {}).get("identifier") != expected_tokenizer.get(
        "identifier"
    ) or manifest.get("tokenizer", {}).get("files") != expected_tokenizer.get("files"):
        raise ValueError("capability suite tokenizer differs from the model tokenizer")
    lengths = set(_parse_ints(args.lengths))
    selected = [row for row in rows if int(row["target_length"]) in lengths]
    if not selected:
        raise ValueError("raw capability selection is empty")

    model, tokenizer, identity = _load_arm_model(args)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = []
    for index, row in enumerate(selected, start=1):
        result = _score_phase1_row(model, tokenizer, row, mode="dense")
        results.append(result)
        print(
            json.dumps(
                {
                    "arm": args.arm_name,
                    "progress": f"{index}/{len(selected)}",
                    "task": row["task"],
                    "length": row["target_length"],
                    "metric_score": result["metric_score"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output = {
        "schema": RAW_CAPABILITY_SCHEMA,
        "status": "complete",
        "arm": args.arm_name,
        "substrate": args.substrate,
        "single_seed_supporting": True,
        "raw_extrapolation": True,
        "selection": {"lengths": sorted(lengths), "rows": len(selected)},
        "data_sha256": sha256_file(args.data_root / "manifest.json"),
        "adapter": identity,
        "results": results,
        "aggregate": _phase1_aggregate(results),
        "script_sha256": _script_sha256(),
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_json(args.output, output)
    return output


def _phase1_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["target_length"]))].append(row)
    output = {}
    for (mode, length), cell in sorted(grouped.items()):
        tokens = sum(int(row["answer_tokens"]) for row in cell)
        output.setdefault(mode, {})[str(length)] = {
            "examples": len(cell),
            "metric_mean": statistics.fmean(float(row["metric_score"]) for row in cell),
            "nll": sum(float(row["nll_sum"]) for row in cell) / tokens,
            "answer_tokens": tokens,
        }
    return output


def summarize_phase1(args: argparse.Namespace) -> dict[str, Any]:
    geo = json.loads(args.geo.read_text(encoding="utf-8"))
    evq = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo.get("schema") != PHASE1_SCHEMA or evq.get("schema") != PHASE1_SCHEMA:
        raise ValueError("Phase 1 raw schema mismatch")
    if geo.get("substrate") != "native_geo" or evq.get("substrate") != "evq_cosh":
        raise ValueError("Phase 1 summary received the wrong arms")
    for key in (
        "dataset",
        "selection",
        "target_lengths",
        "data_sha256",
        "phase0_gate_sha256",
        "sparse_config",
        "prompt_transform",
    ):
        if geo.get(key) != evq.get(key):
            raise ValueError(f"Phase 1 arms differ at {key}")
    geo_keys = {(row["example_id"], row["mode"]) for row in geo["results"]}
    evq_keys = {(row["example_id"], row["mode"]) for row in evq["results"]}
    if geo_keys != evq_keys:
        raise ValueError("Phase 1 arms have different example/mode cells")
    aggregate = {
        "native_geo": _phase1_aggregate(geo["results"]),
        "evq_cosh": _phase1_aggregate(evq["results"]),
    }
    did = {}
    for mode in ("score", "fixed"):
        if mode not in aggregate["native_geo"] or mode not in aggregate["evq_cosh"]:
            continue
        did[mode] = {}
        for length in sorted(aggregate["native_geo"][mode]):
            geo_dense = aggregate["native_geo"]["dense"][length]
            geo_mode = aggregate["native_geo"][mode][length]
            evq_dense = aggregate["evq_cosh"]["dense"][length]
            evq_mode = aggregate["evq_cosh"][mode][length]
            did[mode][length] = {
                "metric": (evq_mode["metric_mean"] - evq_dense["metric_mean"])
                - (geo_mode["metric_mean"] - geo_dense["metric_mean"]),
                "nll_improvement": (evq_dense["nll"] - evq_mode["nll"])
                - (geo_dense["nll"] - geo_mode["nll"]),
            }
    score_cells = list(did.get("score", {}).values())
    pilot_checks = {
        "score_metric_did_at_least_0p20_each_length": bool(score_cells)
        and all(cell["metric"] >= 0.20 for cell in score_cells),
        "score_nll_did_positive_each_length": bool(score_cells)
        and all(cell["nll_improvement"] > 0 for cell in score_cells),
    }
    output = {
        "schema": PHASE1_SUMMARY_SCHEMA,
        "status": "positive" if all(pilot_checks.values()) else "stop",
        "pilot_checks": pilot_checks,
        "aggregate": aggregate,
        "difference_in_differences": did,
        "inputs": {"geo_sha256": sha256_file(args.geo), "evq_sha256": sha256_file(args.evq)},
        "single_seed_supporting": True,
    }
    _atomic_json(args.output, output)
    return output


def _add_arm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--training-data-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--substrate", choices=("native_geo", "evq_cosh"), required=True)
    parser.add_argument("--output", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-passkey")
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--reference-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    dry = commands.add_parser("dry-run")
    dry.add_argument("--model-name", required=True)
    dry.add_argument("--model-manifest", type=Path, required=True)
    dry.add_argument("--training-data-manifest", type=Path, required=True)
    dry.add_argument("--geo-adapter", type=Path, required=True)
    dry.add_argument("--evq-adapter", type=Path, required=True)
    dry.add_argument("--passkey-root", type=Path, required=True)
    dry.add_argument("--suite-root", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)

    phase0 = commands.add_parser("phase0")
    _add_arm_arguments(phase0)
    phase0.add_argument("--passkey-root", type=Path, required=True)
    phase0.add_argument("--lengths", default="8192,16384,32768")
    phase0.add_argument("--trials", default="0,1")

    phase0_summary = commands.add_parser("summarize-phase0")
    phase0_summary.add_argument("--geo", type=Path, required=True)
    phase0_summary.add_argument("--evq", type=Path, required=True)
    phase0_summary.add_argument(
        "--gate-lengths", choices=("16384", "16384,32768"), default="16384,32768"
    )
    phase0_summary.add_argument("--output", type=Path, required=True)

    phase1 = commands.add_parser("phase1")
    _add_arm_arguments(phase1)
    phase1.add_argument("--phase0-gate", type=Path, required=True)
    phase1.add_argument("--dataset", choices=("passkey", "retrieval-suite"), default="passkey")
    phase1.add_argument("--data-root", type=Path, required=True)
    phase1.add_argument("--selection", choices=("pilot", "full"), default="pilot")
    phase1.add_argument("--lengths", default="16384,32768")
    phase1.add_argument("--modes", default="dense,score,fixed")
    phase1.add_argument("--chat-wrap-passkey", action="store_true")

    phase1_summary = commands.add_parser("summarize-phase1")
    phase1_summary.add_argument("--geo", type=Path, required=True)
    phase1_summary.add_argument("--evq", type=Path, required=True)
    phase1_summary.add_argument("--output", type=Path, required=True)

    readout_trace = commands.add_parser("readout-trace")
    _add_arm_arguments(readout_trace)
    readout_trace.add_argument("--data-root", type=Path, required=True)
    readout_trace.add_argument("--lengths", default="16384")
    readout_trace.add_argument("--trials", default="0,1")

    prepare_swap = commands.add_parser("prepare-association-swap")
    prepare_swap.add_argument("--model-name", required=True)
    prepare_swap.add_argument("--training-data-manifest", type=Path, required=True)
    prepare_swap.add_argument("--output-dir", type=Path, required=True)

    swap_trace = commands.add_parser("association-swap-trace")
    _add_arm_arguments(swap_trace)
    swap_trace.add_argument("--cases-root", type=Path, required=True)

    raw_capability = commands.add_parser("raw-capability")
    _add_arm_arguments(raw_capability)
    raw_capability.add_argument(
        "--arm-name", choices=("geo_lora_native", "evq_lora_evq"), required=True
    )
    raw_capability.add_argument("--data-root", type=Path, required=True)
    raw_capability.add_argument("--lengths", default="16384")
    raw_capability.add_argument("--stage2", action="store_true")

    canary = commands.add_parser("counterfactual-canary")
    _add_arm_arguments(canary)
    canary.add_argument(
        "--arm-name",
        choices=(
            "base_native",
            "base_midpoint",
            "base_evq",
            "geo_lora_native",
            "geo_lora_evq_cross",
            "evq_lora_native_cross",
            "evq_lora_evq",
        ),
        required=True,
    )
    canary.add_argument("--passkey-root", type=Path, required=True)
    canary.add_argument("--lengths", default="8192")
    canary.add_argument("--trials", default="0,1")
    canary.add_argument("--disable-adapter", action="store_true")
    canary.add_argument(
        "--runtime-frequency",
        choices=("trained", "native_geo", "midpoint_geo", "evq_cosh"),
        default="trained",
    )

    canary_summary = commands.add_parser("summarize-counterfactual-canary")
    canary_summary.add_argument("--base-native", type=Path, required=True)
    canary_summary.add_argument("--geo-lora-native", type=Path, required=True)
    canary_summary.add_argument("--evq-lora-evq", type=Path, required=True)
    canary_summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare-passkey":
        result = prepare_passkey(args)
    elif args.command == "dry-run":
        result = dry_run(args)
    elif args.command == "phase0":
        result = run_phase0(args)
    elif args.command == "summarize-phase0":
        result = summarize_phase0(args)
    elif args.command == "phase1":
        result = run_phase1(args)
    elif args.command == "summarize-phase1":
        result = summarize_phase1(args)
    elif args.command == "readout-trace":
        result = run_readout_trace(args)
    elif args.command == "prepare-association-swap":
        result = prepare_association_swap_cases(args)
    elif args.command == "association-swap-trace":
        result = run_association_swap_trace(args)
    elif args.command == "raw-capability":
        result = run_raw_capability(args)
    elif args.command == "counterfactual-canary":
        result = run_counterfactual_canary(args)
    else:
        result = summarize_counterfactual_canary(args)
    print(json.dumps({key: result.get(key) for key in ("schema", "status", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
