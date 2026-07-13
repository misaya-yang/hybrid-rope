#!/usr/bin/env python3
"""Prepare token-parity-safe data for EVQ seed-42 retrieval repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from experiments.lora_evq_v2.prepare_positional_distill_data import (
    sha256_file,
    tokenizer_source_fingerprint,
)
from rebuttal.frequency_adaptation_8b.prepare_data import load_frozen_filler

from .protocol import StageSpec, get_stage


PURPOSE = "evq_seed42_retrieval_repair"
FORMAT_VERSION = 1
VALUE_TOKENS = 12
VALIDATION_GROUPS = 16
TEST_GROUPS = 32
PASSKEY_TRIALS_PER_DEPTH = 5
PASSKEY_DEPTHS = (10, 25, 50, 75, 90)
_TOKEN_PATTERN = re.compile(r"^ ?[A-Za-z]{3,14}$")

_WORDING = {
    "train": {
        "instruction": "Read the records and return only the current stored value.",
        "noun": "Record",
        "verb": "stores",
        "query": "Return the current value for Record",
    },
    "validation": {
        "instruction": "Inspect the entries and answer with only the requested contents.",
        "noun": "Entry",
        "verb": "contains",
        "query": "What are the current contents of Entry",
    },
    "test": {
        "instruction": "Use the document to recover exactly one requested value.",
        "noun": "Item",
        "verb": "currently holds",
        "query": "Recover the value assigned to Item",
    },
}


@dataclass(frozen=True)
class RenderedChat:
    """One full chat render plus measured source/answer token spans."""

    input_ids: torch.Tensor
    prompt_length: int
    answer_start: int
    answer_value_end: int
    answer_end: int
    source_start: int
    source_end: int
    source_value_start: int
    source_value_end: int

    @property
    def distance(self) -> int:
        return self.answer_start - 1 - self.source_value_start


@dataclass(frozen=True)
class RepairExample:
    """One measured training/evaluation row before tensor stacking."""

    rendered: RenderedChat
    messages: tuple[dict[str, str], ...]
    markers: dict[str, str]
    split: str
    task_type: str
    key_text: str
    value_text: str
    before_text: str
    after_text: str
    old_value_text: str | None = None
    variant: str = "train"
    group_id: str | None = None
    segment: int | None = None


def _token_ids(value: Any, name: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a one-dimensional token sequence")
    result = [int(token_id) for token_id in value]
    if not result or any(token_id < 0 for token_id in result):
        raise ValueError(f"{name} must contain non-negative token ids")
    return result


def _offsets(value: Any) -> list[tuple[int, int]]:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("offset mapping must be a sequence")
    output = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("offset mapping entries must contain start/end")
        output.append((int(item[0]), int(item[1])))
    return output


def _char_to_token_span(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
    *,
    name: str,
) -> tuple[int, int]:
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]
    if not indices:
        raise ValueError(f"{name} did not map to any token")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"{name} maps to a non-contiguous token span")
    return indices[0], indices[-1] + 1


def build_messages(
    split: str,
    *,
    before_text: str,
    after_text: str,
    key_text: str,
    value_text: str,
    old_value_text: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Build one split-specific complete user/assistant exchange."""
    if split not in _WORDING:
        raise ValueError("split must be train, validation, or test")
    if not key_text.strip() or not value_text.strip():
        raise ValueError("key and value text must be non-empty")
    words = _WORDING[split]
    current_line = f"{words['noun']} {key_text} {words['verb']}{value_text}."
    lines = [words["instruction"], ""]
    if old_value_text is not None:
        lines.append(
            f"{words['noun']} {key_text} previously held{old_value_text}."
        )
    if before_text:
        lines.append(before_text)
    lines.append(current_line)
    if after_text:
        lines.append(after_text)
    lines.append(f"{words['query']} {key_text}.")
    content = "\n".join(lines)
    if content.count(current_line) != 1:
        raise ValueError("current source line must occur exactly once")
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": value_text},
    ]
    return messages, {
        "user_content": content,
        "source_line": current_line,
        "source_value": value_text,
        "answer_value": value_text,
    }


def render_complete_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    markers: Mapping[str, str],
) -> RenderedChat:
    """Render the complete exchange and require direct token parity."""
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template string render did not return text")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer must return input_ids and offset_mapping")
    rendered_ids = _token_ids(encoded.get("input_ids"), "rendered chat input_ids")
    offsets = _offsets(encoded.get("offset_mapping"))
    if len(offsets) != len(rendered_ids):
        raise ValueError("rendered chat offsets do not match input ids")
    direct_ids = _token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
        ),
        "direct chat-template input_ids",
    )
    if rendered_ids != direct_ids:
        raise RuntimeError("full chat-template string/token parity failed")

    user_only = [dict(messages[0])]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            user_only,
            tokenize=True,
            add_generation_prompt=True,
        ),
        "chat generation prompt",
    )
    if direct_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("full chat render does not begin with the generation prompt")

    user_content = str(markers["user_content"])
    source_line = str(markers["source_line"])
    source_value = str(markers["source_value"])
    answer_value = str(markers["answer_value"])
    user_start = rendered.find(user_content)
    if user_start < 0 or rendered.find(user_content, user_start + 1) >= 0:
        raise ValueError("user content must occur exactly once in rendered chat")
    source_line_in_user = user_content.find(source_line)
    if source_line_in_user < 0 or user_content.find(source_line, source_line_in_user + 1) >= 0:
        raise ValueError("source line must occur exactly once in user content")
    source_value_in_line = source_line.find(source_value)
    if source_value_in_line < 0:
        raise ValueError("source value is absent from source line")
    source_char_start = user_start + source_line_in_user
    source_char_end = source_char_start + len(source_line)
    source_value_char_start = source_char_start + source_value_in_line
    source_value_char_end = source_value_char_start + len(source_value)

    answer_char_start = rendered.rfind(answer_value)
    if answer_char_start < user_start + len(user_content):
        raise ValueError("assistant answer occurrence could not be isolated")
    answer_char_end = answer_char_start + len(answer_value)
    source_start, source_end = _char_to_token_span(
        offsets, source_char_start, source_char_end, name="source line"
    )
    source_value_start, source_value_end = _char_to_token_span(
        offsets,
        source_value_char_start,
        source_value_char_end,
        name="source value",
    )
    answer_value_start, answer_value_end = _char_to_token_span(
        offsets, answer_char_start, answer_char_end, name="assistant answer"
    )
    answer_start = len(prompt_ids)
    if answer_value_start != answer_start:
        raise RuntimeError(
            f"assistant answer starts at token {answer_value_start}, expected generation boundary {answer_start}"
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None or direct_ids[-1] != int(eos_token_id):
        raise RuntimeError("complete chat render must terminate with tokenizer.eos_token_id")
    if answer_value_end >= len(direct_ids):
        raise RuntimeError("assistant answer is not followed by EOS")
    if answer_value_end != len(direct_ids) - 1:
        raise RuntimeError("assistant answer must be followed by exactly one EOS token")
    return RenderedChat(
        input_ids=torch.tensor(direct_ids, dtype=torch.int32),
        prompt_length=len(prompt_ids),
        answer_start=answer_start,
        answer_value_end=answer_value_end,
        answer_end=len(direct_ids),
        source_start=source_start,
        source_end=source_end,
        source_value_start=source_value_start,
        source_value_end=source_value_end,
    )


def partition_nonce_pools(
    tokenizer: Any,
    *,
    minimum: int = 512,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Build pairwise-disjoint split x key/value ordinary-token pools."""
    minimum = int(minimum)
    if minimum <= 0:
        raise ValueError("nonce minimum must be positive")
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("tokenizer must expose a positive vocab_size")
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    candidates: list[int] = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _TOKEN_PATTERN.fullmatch(str(decoded)):
            candidates.append(token_id)
    required = 6 * minimum
    if len(candidates) < required:
        raise ValueError(
            f"tokenizer exposes only {len(candidates)} suitable nonce tokens; need {required}"
        )
    selected = candidates[:required]
    output: dict[str, dict[str, tuple[int, ...]]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        output[split] = {}
        for role in ("key", "value"):
            output[split][role] = tuple(selected[offset : offset + minimum])
            offset += minimum
    return output


def partition_filler(validation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split frozen validation rows into non-overlapping validation/test views."""
    if not torch.is_tensor(validation) or validation.ndim != 2:
        raise ValueError("validation filler must be a two-dimensional tensor")
    if validation.shape[0] < 2:
        raise ValueError("validation filler must contain at least two rows")
    midpoint = int(validation.shape[0]) // 2
    if midpoint == 0 or midpoint == validation.shape[0]:
        raise ValueError("validation filler cannot be split into two row regions")
    return {
        "validation": validation[:midpoint].reshape(-1),
        "test": validation[midpoint:].reshape(-1),
    }


def task_schedule(count: int, *, seed: int) -> list[str]:
    """Return an exact, deterministic 75% KV / 25% update schedule."""
    count = int(count)
    if count <= 0 or count % 4:
        raise ValueError("task count must be positive and divisible by four")
    schedule = ["kv"] * (3 * count // 4) + ["update"] * (count // 4)
    random.Random(int(seed)).shuffle(schedule)
    return schedule


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _solve_exact_count(
    measure,
    *,
    target: int,
    maximum: int,
    name: str,
) -> int:
    """Find an exact count for a monotone rendered-token measurement."""
    target = int(target)
    maximum = int(maximum)
    if maximum < 0:
        raise ValueError(f"{name} maximum must be non-negative")
    low, high = 0, maximum
    while low < high:
        midpoint = (low + high) // 2
        if int(measure(midpoint)) < target:
            low = midpoint + 1
        else:
            high = midpoint
    start = max(0, low - 32)
    stop = min(maximum, low + 32)
    for count in range(start, stop + 1):
        if int(measure(count)) == target:
            return count
    raise ValueError(f"could not solve exact {name}={target} within {maximum} filler tokens")


def build_exact_retrieval_example(
    tokenizer: Any,
    *,
    split: str,
    before_filler_ids: Sequence[int],
    after_filler_ids: Sequence[int],
    key_text: str,
    value_text: str,
    seq_len: int,
    target_distance: int,
    task_type: str,
    old_value_text: str | None = None,
) -> RepairExample:
    """Solve exact length/distance on the final complete chat tokenization."""
    if task_type not in {"kv", "update"}:
        raise ValueError("task_type must be kv or update")
    if task_type == "update" and old_value_text is None:
        raise ValueError("update examples require old_value_text")
    if task_type == "kv" and old_value_text is not None:
        raise ValueError("kv examples must not include old_value_text")
    seq_len = int(seq_len)
    target_distance = int(target_distance)
    before_tokens = [int(value) for value in before_filler_ids]
    after_tokens = [int(value) for value in after_filler_ids]
    cache: dict[tuple[int, int], tuple[RenderedChat, list[dict[str, str]], dict[str, str], str, str]] = {}

    def render(before_count: int, after_count: int):
        key = (int(before_count), int(after_count))
        if key not in cache:
            before_text = _decode_tokens(tokenizer, before_tokens[: key[0]])
            after_text = _decode_tokens(tokenizer, after_tokens[: key[1]])
            messages, markers = build_messages(
                split,
                before_text=before_text,
                after_text=after_text,
                key_text=key_text,
                value_text=value_text,
                old_value_text=old_value_text,
            )
            rendered = render_complete_chat(tokenizer, messages, markers)
            cache[key] = (rendered, messages, markers, before_text, after_text)
        return cache[key]

    after_count = _solve_exact_count(
        lambda count: render(0, count)[0].distance,
        target=target_distance,
        maximum=min(len(after_tokens), seq_len),
        name="source-to-answer distance",
    )
    before_count = _solve_exact_count(
        lambda count: render(count, after_count)[0].input_ids.numel(),
        target=seq_len,
        maximum=min(len(before_tokens), seq_len),
        name="sequence length",
    )
    rendered, messages, markers, before_text, after_text = render(before_count, after_count)
    if rendered.input_ids.numel() != seq_len or rendered.distance != target_distance:
        raise RuntimeError("final full-chat render violates solved length/distance")
    if rendered.answer_end - rendered.answer_start != VALUE_TOKENS + 1:
        raise ValueError(
            "assistant answer must tokenize to 12 value tokens followed by exactly one EOS"
        )
    if rendered.source_value_end - rendered.source_value_start != VALUE_TOKENS:
        raise ValueError("source value must tokenize to exactly 12 tokens")
    return RepairExample(
        rendered=rendered,
        messages=tuple(dict(message) for message in messages),
        markers=dict(markers),
        split=split,
        task_type=task_type,
        key_text=key_text,
        value_text=value_text,
        before_text=before_text,
        after_text=after_text,
        old_value_text=old_value_text,
    )


def build_counterfactual_group(
    tokenizer: Any,
    example: RepairExample,
    *,
    swapped_value_text: str,
    removal_fill_id: int,
    group_id: str,
) -> tuple[RepairExample, RepairExample, RepairExample]:
    """Create position-matched original/swap/source-removal records."""
    original = replace(example, variant="original", group_id=str(group_id))
    swapped_messages, swapped_markers = build_messages(
        example.split,
        before_text=example.before_text,
        after_text=example.after_text,
        key_text=example.key_text,
        value_text=swapped_value_text,
        old_value_text=example.old_value_text,
    )
    swapped_rendered = render_complete_chat(tokenizer, swapped_messages, swapped_markers)
    for name in (
        "prompt_length",
        "answer_start",
        "answer_value_end",
        "answer_end",
        "source_start",
        "source_end",
        "source_value_start",
        "source_value_end",
    ):
        if getattr(swapped_rendered, name) != getattr(example.rendered, name):
            raise ValueError(f"swapped value changes registered token position {name}")
    if swapped_rendered.input_ids.numel() != example.rendered.input_ids.numel():
        raise ValueError("swapped value changes sequence length")
    swapped = RepairExample(
        rendered=swapped_rendered,
        messages=tuple(dict(message) for message in swapped_messages),
        markers=dict(swapped_markers),
        split=example.split,
        task_type=example.task_type,
        key_text=example.key_text,
        value_text=swapped_value_text,
        before_text=example.before_text,
        after_text=example.after_text,
        old_value_text=example.old_value_text,
        variant="swapped",
        group_id=str(group_id),
    )

    removed_ids = example.rendered.input_ids.clone()
    removed_ids[example.rendered.source_start : example.rendered.source_end] = int(
        removal_fill_id
    )
    removed_rendered = replace(example.rendered, input_ids=removed_ids)
    removed = replace(
        example,
        rendered=removed_rendered,
        variant="source_removed",
        group_id=str(group_id),
    )
    return original, swapped, removed


def _metadata(example: RepairExample) -> dict[str, Any]:
    rendered = example.rendered
    return {
        "split": example.split,
        "task_type": example.task_type,
        "variant": example.variant,
        "group_id": example.group_id,
        "segment": example.segment,
        "distance": rendered.distance,
        "source_start": rendered.source_start,
        "source_end": rendered.source_end,
        "source_value_start": rendered.source_value_start,
        "source_value_end": rendered.source_value_end,
        "answer_start": rendered.answer_start,
        "answer_value_end": rendered.answer_value_end,
        "answer_end": rendered.answer_end,
    }


def stack_bundle(
    examples: Sequence[RepairExample],
    *,
    stage: str,
    split: str,
    seed: int,
    segment: int | None = None,
    expected_seq_len: int | None = None,
) -> dict[str, Any]:
    """Stack a repair shard without materializing full label tensors."""
    if not examples:
        raise ValueError("cannot stack an empty repair bundle")
    seq_len = int(examples[0].rendered.input_ids.numel())
    if expected_seq_len is not None and seq_len != int(expected_seq_len):
        raise ValueError("repair bundle sequence length does not match expected value")
    prepared = [replace(example, segment=segment) for example in examples]
    if any(example.rendered.input_ids.numel() != seq_len for example in prepared):
        raise ValueError("repair bundle rows must have one sequence length")
    bundle = {
        "format_version": FORMAT_VERSION,
        "purpose": PURPOSE,
        "stage": str(stage),
        "split": str(split),
        "segment": segment,
        "seed": int(seed),
        "seq_len": seq_len,
        "input_ids": torch.stack(
            [example.rendered.input_ids.to(torch.int32) for example in prepared]
        ),
        "answer_start": torch.tensor(
            [example.rendered.answer_start for example in prepared], dtype=torch.int32
        ),
        "answer_end": torch.tensor(
            [example.rendered.answer_end for example in prepared], dtype=torch.int32
        ),
        "metadata": [_metadata(example) for example in prepared],
    }
    return validate_bundle(
        bundle,
        stage=stage,
        split=split,
        segment=segment,
        expected_seq_len=seq_len,
        expected_rows=len(prepared),
    )


def validate_bundle(
    bundle: Mapping[str, Any],
    *,
    stage: str,
    split: str,
    segment: int | None,
    expected_seq_len: int,
    expected_rows: int,
) -> Mapping[str, Any]:
    """Fail closed on one prepared repair tensor shard."""
    get_stage(stage)
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("purpose") != PURPOSE:
        raise ValueError("repair bundle identity mismatch")
    if bundle.get("stage") != stage or bundle.get("split") != split:
        raise ValueError("repair bundle stage/split mismatch")
    if bundle.get("segment") != segment:
        raise ValueError("repair bundle segment mismatch")
    input_ids = bundle.get("input_ids")
    starts = bundle.get("answer_start")
    ends = bundle.get("answer_end")
    metadata = bundle.get("metadata")
    if not torch.is_tensor(input_ids) or input_ids.dtype != torch.int32 or input_ids.ndim != 2:
        raise ValueError("repair input_ids must be a two-dimensional int32 tensor")
    rows, seq_len = input_ids.shape
    if rows != int(expected_rows) or seq_len != int(expected_seq_len):
        raise ValueError("repair bundle row/sequence shape mismatch")
    if bundle.get("seq_len") != seq_len:
        raise ValueError("repair bundle seq_len metadata mismatch")
    if not torch.is_tensor(starts) or not torch.is_tensor(ends):
        raise ValueError("repair bundle answer spans must be tensors")
    if starts.shape != (rows,) or ends.shape != (rows,):
        raise ValueError("repair bundle answer spans must have one entry per row")
    if not torch.all(ends == seq_len):
        raise ValueError("repair answer supervision must be a sequence tail")
    if not torch.all(ends - starts == VALUE_TOKENS + 1):
        raise ValueError("repair answer span must be 12 value tokens plus EOS")
    if not isinstance(metadata, list) or len(metadata) != rows:
        raise ValueError("repair bundle metadata must have one row per example")
    for index, row in enumerate(metadata):
        if not isinstance(row, Mapping):
            raise ValueError("repair metadata rows must be mappings")
        if row.get("split") != split:
            raise ValueError("repair split metadata mismatch")
        if row.get("segment") != segment:
            raise ValueError("repair segment metadata mismatch")
        measured = int(starts[index]) - 1 - int(row.get("source_value_start", -1))
        if measured != int(row.get("distance", -2)):
            raise ValueError("repair distance metadata mismatch")
    return bundle


def _render_passkey_prompt(
    tokenizer: Any,
    *,
    before_text: str,
    after_text: str,
    passkey: str,
) -> tuple[list[dict[str, str]], list[int]]:
    content = (
        "Read the document and remember its retrieval passkey.\n\n"
        f"{before_text}\nThe retrieval passkey is {passkey}.\n{after_text}\n"
        "What is the retrieval passkey? Return only the eight digits."
    )
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    direct = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    rendered_ids = _token_ids(encoded, "rendered passkey prompt")
    direct_ids = _token_ids(direct, "direct passkey prompt")
    if rendered_ids != direct_ids:
        raise RuntimeError("passkey full chat-template string/token parity failed")
    return messages, direct_ids


def build_passkey_prompt(
    tokenizer: Any,
    *,
    filler_ids: Sequence[int],
    target_length: int,
    depth_percent: float,
    passkey: str,
) -> dict[str, Any]:
    """Build an exact prompt-only chat-templated passkey row."""
    target_length = int(target_length)
    depth_percent = float(depth_percent)
    if not 0 <= depth_percent <= 100:
        raise ValueError("passkey depth must lie in [0, 100]")
    if not re.fullmatch(r"\d{8}", str(passkey)):
        raise ValueError("passkey must contain exactly eight digits")
    tokens = [int(value) for value in filler_ids]
    midpoint = len(tokens) // 2
    before_pool, after_pool = tokens[:midpoint], tokens[midpoint:]
    cache: dict[int, tuple[list[dict[str, str]], list[int]] | None] = {}

    def render(total: int):
        total = int(total)
        if total in cache:
            return cache[total]
        before_count = round(int(total) * depth_percent / 100.0)
        after_count = int(total) - before_count
        if before_count > len(before_pool) or after_count > len(after_pool):
            cache[total] = None
            return cache[total]
        before_text = _decode_tokens(tokenizer, before_pool[:before_count])
        after_text = _decode_tokens(tokenizer, after_pool[:after_count])
        cache[total] = _render_passkey_prompt(
            tokenizer,
            before_text=before_text,
            after_text=after_text,
            passkey=str(passkey),
        )
        return cache[total]

    def prompt_length(total: int) -> int:
        result = render(total)
        return target_length + 1 if result is None else len(result[1])

    maximum = min(len(before_pool) + len(after_pool), target_length * 2)
    total = _solve_exact_count(
        prompt_length,
        target=target_length,
        maximum=maximum,
        name="passkey prompt length",
    )
    result = render(total)
    if result is None:
        raise RuntimeError("solved passkey filler does not fit its pools")
    messages, prompt_ids = result
    return {
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
        "messages": messages,
        "answer": str(passkey),
        "target_length": target_length,
        "length_semantics": "prompt_tokens_before_generation",
        "depth_percent": depth_percent,
        "prompt_sha256": hashlib.sha256(
            torch.tensor(prompt_ids, dtype=torch.int64).numpy().tobytes()
        ).hexdigest(),
    }


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bundle_atomic(
    bundle: Mapping[str, Any],
    path: Path,
    *,
    stage: str,
    split: str,
    segment: int | None,
    expected_seq_len: int,
    expected_rows: int,
) -> dict[str, Any]:
    """Write, reload, validate, and atomically publish one tensor bundle."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        torch.save(dict(bundle), temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_bundle(
            loaded,
            stage=stage,
            split=split,
            segment=segment,
            expected_seq_len=expected_seq_len,
            expected_rows=expected_rows,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "kind": "retrieval_bundle",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": int(expected_rows),
        "seq_len": int(expected_seq_len),
        "stage": stage,
        "split": split,
        "segment": segment,
    }


def _validate_passkey_bundle(
    bundle: Mapping[str, Any],
    *,
    target_length: int,
    expected_rows: int = 25,
) -> Mapping[str, Any]:
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("purpose") != PURPOSE:
        raise ValueError("passkey bundle identity mismatch")
    if bundle.get("length_semantics") != "prompt_tokens_before_generation":
        raise ValueError("passkey bundle length semantics mismatch")
    prompt_ids = bundle.get("prompt_ids")
    metadata = bundle.get("metadata")
    if not torch.is_tensor(prompt_ids) or prompt_ids.dtype != torch.int32 or prompt_ids.ndim != 2:
        raise ValueError("passkey prompt_ids must be a two-dimensional int32 tensor")
    if prompt_ids.shape != (int(expected_rows), int(target_length)):
        raise ValueError("passkey bundle shape mismatch")
    if not isinstance(metadata, list) or len(metadata) != int(expected_rows):
        raise ValueError("passkey metadata row count mismatch")
    depths = [int(row.get("depth_percent", -1)) for row in metadata]
    if sorted(set(depths)) != list(PASSKEY_DEPTHS):
        raise ValueError("passkey bundle depth grid mismatch")
    if any(depths.count(depth) != PASSKEY_TRIALS_PER_DEPTH for depth in PASSKEY_DEPTHS):
        raise ValueError("passkey bundle must contain five trials per depth")
    if any(not re.fullmatch(r"\d{8}", str(row.get("answer", ""))) for row in metadata):
        raise ValueError("passkey bundle contains an invalid answer")
    return bundle


def write_passkey_bundle_atomic(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    target_length: int,
) -> dict[str, Any]:
    """Publish one 25-row prompt-only passkey tensor bundle."""
    if len(rows) != len(PASSKEY_DEPTHS) * PASSKEY_TRIALS_PER_DEPTH:
        raise ValueError("passkey rows do not cover the registered grid")
    bundle = {
        "format_version": FORMAT_VERSION,
        "purpose": PURPOSE,
        "target_length": int(target_length),
        "length_semantics": "prompt_tokens_before_generation",
        "prompt_ids": torch.stack(
            [row["prompt_ids"].to(torch.int32) for row in rows]
        ),
        "metadata": [
            {
                "answer": str(row["answer"]),
                "depth_percent": int(row["depth_percent"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "target_length": int(row["target_length"]),
                "length_semantics": str(row["length_semantics"]),
            }
            for row in rows
        ],
    }
    _validate_passkey_bundle(bundle, target_length=target_length)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        torch.save(bundle, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_passkey_bundle(loaded, target_length=target_length)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "kind": "passkey_bundle",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(rows),
        "seq_len": int(target_length),
        "target_length": int(target_length),
        "length_semantics": "prompt_tokens_before_generation",
    }


def write_manifest(
    root: Path,
    *,
    files: Mapping[str, Mapping[str, Any]],
    tokenizer_identity: Mapping[str, Any],
    filler_manifest_sha256: str,
    nonce_pool_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Write the repair data manifest last."""
    if not files:
        raise ValueError("repair manifest requires at least one file")
    for digest_name, digest in (
        ("filler_manifest_sha256", filler_manifest_sha256),
        ("nonce_pool_sha256", nonce_pool_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"{digest_name} must be a lowercase SHA-256 digest")
    manifest = {
        "format_version": FORMAT_VERSION,
        "purpose": PURPOSE,
        "status": "prepared_no_results",
        "seed": int(seed),
        "split_policy": "disjoint_split_role_nonce_filler_seed_and_wording",
        "training_shards": "independent_segment1_segment2",
        "task_mix": {"kv": 0.75, "update": 0.25, "policy": "exact_per_shard"},
        "answer_supervision": {"value_tokens": VALUE_TOKENS, "eos_tokens": 1},
        "passkey_length_semantics": "prompt_tokens_before_generation",
        "tokenizer": dict(tokenizer_identity),
        "filler_manifest_sha256": str(filler_manifest_sha256),
        "nonce_pool_sha256": str(nonce_pool_sha256),
        "wording_sha256": _json_sha256(_WORDING),
        "files": {name: dict(record) for name, record in sorted(files.items())},
    }
    root = Path(root)
    _atomic_json_dump(manifest, root / "manifest.json")
    return manifest


def validate_prepared_dir(root: Path) -> dict[str, Any]:
    """Hash-check and structurally validate every registered prepared file."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("purpose") != PURPOSE
        or manifest.get("status") != "prepared_no_results"
    ):
        raise ValueError("repair data manifest identity/status mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("repair data manifest has no files")
    for name, raw_record in files.items():
        if Path(name).name != name:
            raise ValueError(f"unsafe repair artifact filename: {name!r}")
        if not isinstance(raw_record, Mapping):
            raise ValueError("repair artifact record must be a mapping")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != raw_record.get("sha256"):
            raise RuntimeError(f"repair artifact SHA-256 mismatch: {name}")
        if path.stat().st_size != int(raw_record.get("size_bytes", -1)):
            raise RuntimeError(f"repair artifact size mismatch: {name}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if raw_record.get("kind") == "retrieval_bundle":
            validate_bundle(
                loaded,
                stage=str(raw_record["stage"]),
                split=str(raw_record["split"]),
                segment=raw_record.get("segment"),
                expected_seq_len=int(raw_record["seq_len"]),
                expected_rows=int(raw_record["rows"]),
            )
        elif raw_record.get("kind") == "passkey_bundle":
            _validate_passkey_bundle(
                loaded,
                target_length=int(raw_record["target_length"]),
                expected_rows=int(raw_record["rows"]),
            )
        else:
            raise ValueError(f"unknown repair artifact kind for {name}")
    return manifest


def artifact_plan() -> dict[str, dict[str, Any]]:
    """Return the immutable prepared-file matrix before materialization."""
    plan: dict[str, dict[str, Any]] = {}
    for stage_name in ("r8", "r16"):
        stage = get_stage(stage_name)
        for segment in (1, 2):
            plan[f"train_{stage_name}_segment{segment}.pt"] = {
                "kind": "retrieval_bundle",
                "stage": stage_name,
                "split": "train",
                "segment": segment,
                "rows": stage.examples_per_segment,
                "seq_len": stage.seq_len,
            }
        plan[f"validation_{stage_name}.pt"] = {
            "kind": "retrieval_bundle",
            "stage": stage_name,
            "split": "validation",
            "segment": None,
            "rows": VALIDATION_GROUPS * 3,
            "seq_len": stage.seq_len,
        }
        plan[f"test_{stage_name}.pt"] = {
            "kind": "retrieval_bundle",
            "stage": stage_name,
            "split": "test",
            "segment": None,
            "rows": TEST_GROUPS * 3,
            "seq_len": stage.seq_len,
        }
    for target_length in (8192, 16384, 32768):
        plan[f"passkey_{target_length}.pt"] = {
            "kind": "passkey_bundle",
            "rows": len(PASSKEY_DEPTHS) * PASSKEY_TRIALS_PER_DEPTH,
            "seq_len": target_length,
            "target_length": target_length,
            "length_semantics": "prompt_tokens_before_generation",
        }
    return plan


def _key_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    words = [_decode_tokens(tokenizer, [token_id]).strip() for token_id in token_ids]
    if any(not _TOKEN_PATTERN.fullmatch(word) for word in words):
        raise ValueError("sampled key token is not an ordinary word")
    return "-".join(words)


def _value_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    expected = [int(token_id) for token_id in token_ids]
    text = _decode_tokens(tokenizer, expected)
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if _token_ids(encoded, "nonce value") != expected:
        raise ValueError("sampled value tokens do not survive decode/encode round-trip")
    return text


def _sample_windows(
    filler: torch.Tensor,
    *,
    seq_len: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    flat = filler.reshape(-1)
    midpoint = flat.numel() // 2
    window = int(seq_len) + 512
    if midpoint < window or flat.numel() - midpoint < window:
        raise ValueError(
            f"filler region has {flat.numel()} tokens; need two windows of {window}"
        )
    left_start = rng.randrange(0, midpoint - window + 1)
    right_start = rng.randrange(midpoint, flat.numel() - window + 1)
    return (
        [int(value) for value in flat[left_start : left_start + window].tolist()],
        [int(value) for value in flat[right_start : right_start + window].tolist()],
    )


def _sample_repair_example(
    tokenizer: Any,
    *,
    stage: StageSpec,
    split: str,
    task_type: str,
    key_pool: Sequence[int],
    value_pool: Sequence[int],
    filler: torch.Tensor,
    rng: random.Random,
) -> tuple[RepairExample, str, int]:
    """Sample until nonce text survives the real full chat tokenizer."""
    last_error: Exception | None = None
    for _ in range(32):
        key_ids = rng.sample(list(key_pool), 3)
        value_ids = rng.sample(list(value_pool), VALUE_TOKENS * 3)
        value = value_ids[:VALUE_TOKENS]
        swapped = value_ids[VALUE_TOKENS : 2 * VALUE_TOKENS]
        old = value_ids[2 * VALUE_TOKENS :]
        try:
            key_text = _key_text(tokenizer, key_ids)
            value_text = _value_text(tokenizer, value)
            swapped_text = _value_text(tokenizer, swapped)
            old_text = _value_text(tokenizer, old) if task_type == "update" else None
            before, after = _sample_windows(filler, seq_len=stage.seq_len, rng=rng)
            target_distance = rng.randint(stage.min_distance, stage.max_distance)
            example = build_exact_retrieval_example(
                tokenizer,
                split=split,
                before_filler_ids=before,
                after_filler_ids=after,
                key_text=key_text,
                value_text=value_text,
                seq_len=stage.seq_len,
                target_distance=target_distance,
                task_type=task_type,
                old_value_text=old_text,
            )
            removal_id = before[0]
            return example, swapped_text, removal_id
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError("failed to sample a token-parity-safe retrieval example after 32 attempts") from last_error


def _build_train_shard(
    tokenizer: Any,
    *,
    stage: StageSpec,
    segment: int,
    pools: Mapping[str, Sequence[int]],
    filler: torch.Tensor,
    seed: int,
) -> list[RepairExample]:
    rng = random.Random(int(seed))
    schedule = task_schedule(stage.examples_per_segment, seed=seed)
    rows = []
    for task_type in schedule:
        example, _, _ = _sample_repair_example(
            tokenizer,
            stage=stage,
            split="train",
            task_type=task_type,
            key_pool=pools["key"],
            value_pool=pools["value"],
            filler=filler,
            rng=rng,
        )
        rows.append(replace(example, segment=int(segment)))
    return rows


def _build_eval_groups(
    tokenizer: Any,
    *,
    stage: StageSpec,
    split: str,
    groups: int,
    pools: Mapping[str, Sequence[int]],
    filler: torch.Tensor,
    seed: int,
) -> list[RepairExample]:
    rng = random.Random(int(seed))
    schedule = task_schedule(groups, seed=seed)
    rows: list[RepairExample] = []
    for index, task_type in enumerate(schedule):
        example, swapped, removal_id = _sample_repair_example(
            tokenizer,
            stage=stage,
            split=split,
            task_type=task_type,
            key_pool=pools["key"],
            value_pool=pools["value"],
            filler=filler,
            rng=rng,
        )
        rows.extend(
            build_counterfactual_group(
                tokenizer,
                example,
                swapped_value_text=swapped,
                removal_fill_id=removal_id,
                group_id=f"{stage.name}-{split}-{index:04d}",
            )
        )
    return rows


def _training_filler_regions(train_filler: torch.Tensor) -> dict[int, torch.Tensor]:
    if not torch.is_tensor(train_filler) or train_filler.ndim != 2 or train_filler.shape[0] < 2:
        raise ValueError("train filler must be a two-dimensional tensor with at least two rows")
    midpoint = int(train_filler.shape[0]) // 2
    return {1: train_filler[:midpoint].reshape(-1), 2: train_filler[midpoint:].reshape(-1)}


def _nonce_pool_digest(pools: Mapping[str, Mapping[str, Sequence[int]]]) -> str:
    normalized = {
        split: {role: [int(value) for value in values] for role, values in roles.items()}
        for split, roles in pools.items()
    }
    return _json_sha256(normalized)


def prepare_all(
    *,
    tokenizer: Any,
    tokenizer_identity: Mapping[str, Any],
    train_filler: torch.Tensor,
    validation_filler: torch.Tensor,
    filler_manifest_sha256: str,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Materialize every registered shard in a sibling incomplete directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    incomplete = output_dir.with_name(
        f".{output_dir.name}.{os.getpid()}.{time.time_ns()}.incomplete"
    )
    if incomplete.exists():
        raise FileExistsError(incomplete)
    incomplete.mkdir(parents=True, exist_ok=False)
    plan = artifact_plan()
    pools = partition_nonce_pools(tokenizer)
    train_regions = _training_filler_regions(train_filler)
    eval_regions = partition_filler(validation_filler)
    records: dict[str, dict[str, Any]] = {}
    try:
        for stage_index, stage_name in enumerate(("r8", "r16")):
            stage = get_stage(stage_name)
            for segment in (1, 2):
                filename = f"train_{stage_name}_segment{segment}.pt"
                row_seed = int(seed) + stage_index * 100_000 + segment * 10_000
                rows = _build_train_shard(
                    tokenizer,
                    stage=stage,
                    segment=segment,
                    pools=pools["train"],
                    filler=train_regions[segment],
                    seed=row_seed,
                )
                bundle = stack_bundle(
                    rows,
                    stage=stage_name,
                    split="train",
                    seed=row_seed,
                    segment=segment,
                    expected_seq_len=stage.seq_len,
                )
                records[filename] = write_bundle_atomic(
                    bundle,
                    incomplete / filename,
                    stage=stage_name,
                    split="train",
                    segment=segment,
                    expected_seq_len=stage.seq_len,
                    expected_rows=plan[filename]["rows"],
                )
            for split, groups, offset in (
                ("validation", VALIDATION_GROUPS, 30_000),
                ("test", TEST_GROUPS, 40_000),
            ):
                filename = f"{split}_{stage_name}.pt"
                row_seed = int(seed) + stage_index * 100_000 + offset
                rows = _build_eval_groups(
                    tokenizer,
                    stage=stage,
                    split=split,
                    groups=groups,
                    pools=pools[split],
                    filler=eval_regions[split],
                    seed=row_seed,
                )
                bundle = stack_bundle(
                    rows,
                    stage=stage_name,
                    split=split,
                    seed=row_seed,
                    segment=None,
                    expected_seq_len=stage.seq_len,
                )
                records[filename] = write_bundle_atomic(
                    bundle,
                    incomplete / filename,
                    stage=stage_name,
                    split=split,
                    segment=None,
                    expected_seq_len=stage.seq_len,
                    expected_rows=plan[filename]["rows"],
                )

        passkey_rng = random.Random(int(seed) + 900_000)
        passkey_filler = eval_regions["test"]
        used: set[str] = set()
        for target_length in (8192, 16384, 32768):
            filename = f"passkey_{target_length}.pt"
            rows = []
            window = target_length * 2 + 1024
            if passkey_filler.numel() < window:
                raise ValueError(f"test filler has too few tokens for {target_length}-token passkeys")
            for depth in PASSKEY_DEPTHS:
                for _ in range(PASSKEY_TRIALS_PER_DEPTH):
                    while True:
                        passkey = f"{passkey_rng.randrange(10_000_000, 100_000_000):08d}"
                        if passkey not in used:
                            used.add(passkey)
                            break
                    start = passkey_rng.randrange(0, passkey_filler.numel() - window + 1)
                    filler = [
                        int(value)
                        for value in passkey_filler[start : start + window].tolist()
                    ]
                    rows.append(
                        build_passkey_prompt(
                            tokenizer,
                            filler_ids=filler,
                            target_length=target_length,
                            depth_percent=depth,
                            passkey=passkey,
                        )
                    )
            records[filename] = write_passkey_bundle_atomic(
                rows,
                incomplete / filename,
                target_length=target_length,
            )

        write_manifest(
            incomplete,
            files=records,
            tokenizer_identity=tokenizer_identity,
            filler_manifest_sha256=filler_manifest_sha256,
            nonce_pool_sha256=_nonce_pool_digest(pools),
            seed=seed,
        )
        manifest = validate_prepared_dir(incomplete)
        os.replace(incomplete, output_dir)
        return manifest
    except Exception:
        # Keep the incomplete directory for diagnosis. It is never accepted by
        # the launcher because only the final manifest path is valid.
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer")
    parser.add_argument("--filler-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        manifest_path = args.output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = validate_prepared_dir(args.output_dir)
        print(json.dumps({"status": "valid", "manifest_sha256": sha256_file(manifest_path)}))
        return
    if not args.tokenizer:
        raise ValueError("--tokenizer is required for data preparation")
    if args.filler_dir is None:
        raise ValueError("--filler-dir is required for data preparation")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).expanduser().is_dir(),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("repair data preparation requires a fast tokenizer with offsets")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    train_filler, validation_filler, _ = load_frozen_filler(args.filler_dir)
    filler_manifest_path = args.filler_dir / "manifest.json"
    manifest = prepare_all(
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_source_fingerprint(args.tokenizer),
        train_filler=train_filler,
        validation_filler=validation_filler,
        filler_manifest_sha256=sha256_file(filler_manifest_path),
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": args.output_dir.name,
                "files": len(manifest["files"]),
                "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
