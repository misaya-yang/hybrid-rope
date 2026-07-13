#!/usr/bin/env python3
"""Build a frozen, CPU-prepared capability suite for the seed-42 comparison."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "evq_cosh.seed42_capability_example.v2"
MANIFEST_SCHEMA = "evq_cosh.seed42_capability_manifest.v2"
PROMPT_HASH_ENCODING = "sha256(compact-json-integer-array-v1)"

MCQA_REVISIONS = {
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
}

MCQA_SPECS = {
    "cais/mmlu": {"task": "mmlu", "config": "all", "split": "test"},
    "allenai/ai2_arc": {
        "task": "arc_challenge",
        "config": "ARC-Challenge",
        "split": "test",
    },
    "Rowan/hellaswag": {
        "task": "hellaswag",
        "config": None,
        "split": "validation",
    },
    "allenai/openbookqa": {
        "task": "openbookqa",
        "config": "main",
        "split": "validation",
    },
    "allenai/winogrande": {
        "task": "winogrande",
        "config": "winogrande_xl",
        "split": "validation",
    },
}

MCQA_ARROW_LAYOUT = {
    "cais/mmlu": ("cais___mmlu", "mmlu"),
    "allenai/ai2_arc": ("allenai___ai2_arc", "ai2_arc"),
    "Rowan/hellaswag": ("Rowan___hellaswag", "hellaswag"),
    "allenai/openbookqa": ("allenai___openbookqa", "openbookqa"),
    "allenai/winogrande": ("allenai___winogrande", "winogrande"),
}

DEFAULT_NOLIMA_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n"
    "{haystack}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts or "
    "strong, logical inferences.\n\nQuestion: {question}\n\n Return only the "
    "final answer with no additional explanation or reasoning."
)
NOLIMA_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"

LONGBENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\nStory: "
        "{context}\n\nNow, answer the question based on the story asconcisely as "
        "you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        "\n\nArticle: {context}\n\n Answer the question based on the above article "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        "\n\nQuestion: {input}\n\nAnswer:"
    ),
}

RECORD_FIELDS = {
    "schema",
    "example_id",
    "suite",
    "task",
    "target_length",
    "prompt_ids",
    "answers",
    "metric",
    "depth_percent",
    "choices",
    "answer_index",
    "source",
    "prompt_sha256",
    "generation_tokens",
    "scorer",
}

RULER_TASK_CONTRACTS = {
    "niah": {"match_type": "all", "generation_tokens": 128},
    "vt": {"match_type": "all", "generation_tokens": 30},
    "cwe": {"match_type": "all", "generation_tokens": 120},
    "fwe": {"match_type": "all", "generation_tokens": 50},
    "qa": {"match_type": "part", "generation_tokens": 32},
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a requested JSONL source, failing instead of returning partial data."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"requested JSONL source does not exist: {source_path}")

    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source_path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {source_path} at line {line_number}")
            rows.append(row)
    return rows


def truncate_document_only(
    prefix_ids: Sequence[int],
    document_ids: Sequence[int],
    suffix_ids: Sequence[int],
    max_prompt_tokens: int,
) -> list[int]:
    """Fit a prompt by truncating only its document portion."""
    if max_prompt_tokens < 0:
        raise ValueError("max_prompt_tokens must be non-negative")
    fixed_tokens = len(prefix_ids) + len(suffix_ids)
    if fixed_tokens > max_prompt_tokens:
        raise ValueError("prefix plus suffix exceed max_prompt_tokens; refusing to truncate suffix")
    document_budget = max_prompt_tokens - fixed_tokens
    return list(prefix_ids) + list(document_ids[:document_budget]) + list(suffix_ids)


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _chat_prompt_ids(
    tokenizer: Any,
    user_text: str,
    *,
    system_text: str | None = None,
) -> list[int]:
    """Freeze one complete Instruct chat prompt with string/token parity."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("capability preparation requires tokenizer.apply_chat_template")
    messages = []
    if system_text:
        messages.append({"role": "system", "content": str(system_text)})
    messages.append({"role": "user", "content": str(user_text)})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    direct = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(direct, Mapping):
        if "input_ids" not in direct:
            raise ValueError("chat-template token mapping has no input_ids")
        direct = direct["input_ids"]
    direct_ids = direct.tolist() if hasattr(direct, "tolist") else direct
    if direct_ids and isinstance(direct_ids[0], list):
        direct_ids = direct_ids[0]
    rendered_ids = _encode(tokenizer, str(rendered))
    normalized = [int(token_id) for token_id in direct_ids]
    if rendered_ids != normalized:
        raise ValueError("capability full chat-template string/token parity failed")
    return normalized


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not hasattr(tokenizer, "decode"):
        raise ValueError("capability preparation requires tokenizer.decode")
    return str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _chat_segmented_prompt_ids(
    tokenizer: Any,
    *,
    prefix_ids: Sequence[int],
    document_ids: Sequence[int],
    suffix_ids: Sequence[int],
    max_prompt_tokens: int,
    system_text: str | None = None,
) -> list[int]:
    """Maximize document tokens while preserving suffix and full chat framing."""
    fixed = list(prefix_ids) + list(suffix_ids)
    if len(_chat_prompt_ids(tokenizer, _decode_ids(tokenizer, fixed), system_text=system_text)) > max_prompt_tokens:
        raise ValueError("chat template plus fixed prompt exceeds max_prompt_tokens")
    low, high = 0, len(document_ids)
    best: list[int] | None = None
    while low <= high:
        count = (low + high) // 2
        raw = list(prefix_ids) + list(document_ids[:count]) + list(suffix_ids)
        candidate = _chat_prompt_ids(
            tokenizer, _decode_ids(tokenizer, raw), system_text=system_text
        )
        if len(candidate) <= int(max_prompt_tokens):
            best = candidate
            low = count + 1
        else:
            high = count - 1
    if best is None:
        raise ValueError("could not fit chat-templated prompt")
    return best


def _prompt_sha256(prompt_ids: Sequence[int]) -> str:
    payload = json.dumps(list(prompt_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return slug or "item"


def _normalize_answers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        answers = [str(answer) for answer in value]
    elif value is None:
        answers = []
    else:
        answers = [str(value)]
    if not answers:
        raise ValueError("source record has no answers")
    return answers


def _make_record(
    *,
    example_id: str,
    suite: str,
    task: str,
    target_length: int,
    prompt_ids: Sequence[int],
    answers: Sequence[str],
    metric: str,
    source: Mapping[str, Any],
    depth_percent: float | None = None,
    choices: Sequence[str] | None = None,
    answer_index: int | None = None,
    generation_tokens: int | None = None,
    scorer: str | None = None,
) -> dict[str, Any]:
    normalized_ids = [int(token_id) for token_id in prompt_ids]
    if generation_tokens is None:
        generation_tokens = 0 if metric == "mcqa" else (96 if metric == "qa_f1" else 32)
    if scorer is None:
        scorer = {
            "exact_match": "normalized_exact_match",
            "qa_f1": "longbench_qa_f1",
            "mcqa": "choice_mean_logprob_accuracy",
            "contains": "case_sensitive_contains",
            "ruler_string_match": "pinned_ruler_string_match",
        }[metric]
    return {
        "schema": SCHEMA,
        "example_id": str(example_id),
        "suite": str(suite),
        "task": str(task),
        "target_length": int(target_length),
        "prompt_ids": normalized_ids,
        "answers": [str(answer) for answer in answers],
        "metric": metric,
        "depth_percent": (None if depth_percent is None else float(depth_percent)),
        "choices": None if choices is None else [str(choice) for choice in choices],
        "answer_index": None if answer_index is None else int(answer_index),
        "source": dict(source),
        "prompt_sha256": _prompt_sha256(normalized_ids),
        "generation_tokens": int(generation_tokens),
        "scorer": str(scorer),
    }


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one frozen example and its token-ID digest."""
    missing = RECORD_FIELDS.difference(record)
    if missing:
        raise ValueError(f"record is missing required fields: {sorted(missing)}")
    if record["schema"] != SCHEMA:
        raise ValueError(f"unsupported record schema: {record['schema']!r}")
    for key in ("example_id", "suite", "task"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string")

    target_length = record["target_length"]
    if isinstance(target_length, bool) or not isinstance(target_length, int):
        raise ValueError("target_length must be an integer")
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    prompt_ids = record["prompt_ids"]
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("prompt_ids must be a non-empty list")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in prompt_ids):
        raise ValueError("prompt_ids must contain only integers")
    if any(token_id < 0 for token_id in prompt_ids):
        raise ValueError("prompt_ids must be non-negative")
    if len(prompt_ids) > target_length:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens but target_length is {target_length}")

    answers = record["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError("answers must be a non-empty list")
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError("answers must contain only strings")

    metric = record["metric"]
    if metric not in {"exact_match", "qa_f1", "mcqa", "contains", "ruler_string_match"}:
        raise ValueError(f"unsupported metric: {metric!r}")

    generation_tokens = record["generation_tokens"]
    if (
        isinstance(generation_tokens, bool)
        or not isinstance(generation_tokens, int)
        or generation_tokens < 0
    ):
        raise ValueError("generation_tokens must be a non-negative integer")
    if metric == "mcqa" and generation_tokens != 0:
        raise ValueError("MCQA records must not register generation tokens")
    if metric != "mcqa" and generation_tokens <= 0:
        raise ValueError("generative records need a positive generation budget")
    if not isinstance(record["scorer"], str) or not record["scorer"]:
        raise ValueError("scorer must be a non-empty string")

    depth = record["depth_percent"]
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, (int, float)):
            raise ValueError("depth_percent must be numeric or null")
        if not 0 <= float(depth) <= 100:
            raise ValueError("depth_percent must be between 0 and 100")

    choices = record["choices"]
    answer_index = record["answer_index"]
    if metric == "mcqa":
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError("MCQA records need at least two choices")
        if any(not isinstance(choice, str) for choice in choices):
            raise ValueError("MCQA choices must contain only strings")
        if isinstance(answer_index, bool) or not isinstance(answer_index, int):
            raise ValueError("MCQA answer_index must be an integer")
        if not 0 <= answer_index < len(choices):
            raise ValueError("MCQA answer_index is outside choices")
        if answers != [choices[answer_index]]:
            raise ValueError("MCQA answers must equal the indexed choice")
    elif choices is not None or answer_index is not None:
        raise ValueError("choices and answer_index are reserved for MCQA records")

    if not isinstance(record["source"], dict):
        raise ValueError("source must be an object")
    expected_hash = _prompt_sha256(prompt_ids)
    if record["prompt_sha256"] != expected_hash:
        raise ValueError("prompt_sha256 does not match prompt_ids")
    return dict(record)


def validate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate a suite and reject ambiguous duplicate evaluation cells."""
    validated: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    hashes_by_cell: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in records:
        item = validate_record(record)
        example_id = item["example_id"]
        if example_id in example_ids:
            raise ValueError(f"duplicate example ID: {example_id}")
        example_ids.add(example_id)

        cell = (item["task"], item["target_length"])
        prompt_hash = item["prompt_sha256"]
        if prompt_hash in hashes_by_cell[cell]:
            raise ValueError(
                "duplicate prompt hash within task/length cell: "
                f"task={cell[0]} target_length={cell[1]} hash={prompt_hash}"
            )
        hashes_by_cell[cell].add(prompt_hash)
        validated.append(item)
    return validated


def _repeat_to_length(values: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not values:
        raise ValueError("tokenizer produced no filler tokens")
    repeats, remainder = divmod(length, len(values))
    return list(values) * repeats + list(values[:remainder])


def build_passkey_examples(
    tokenizer: Any,
    lengths: Iterable[int] = (8192, 16384, 32768),
    depths: Iterable[float] = (10, 25, 50, 75, 90),
    trials: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build the deterministic passkey length/depth grid."""
    normalized_lengths = tuple(int(length) for length in lengths)
    normalized_depths = tuple(float(depth) for depth in depths)
    if not normalized_lengths or any(length <= 0 for length in normalized_lengths):
        raise ValueError("passkey lengths must be positive")
    if not normalized_depths or any(depth < 0 or depth > 100 for depth in normalized_depths):
        raise ValueError("passkey depths must be between 0 and 100")
    if trials <= 0:
        raise ValueError("passkey trials must be positive")

    prefix_ids = _encode(
        tokenizer,
        "Read the following document and remember the retrieval passkey.\n\n",
    )
    suffix_ids = _encode(
        tokenizer,
        "\n\nWhat is the retrieval passkey? Return only the passkey.",
    )
    filler_ids = _encode(
        tokenizer,
        "The archive contains ordinary records about weather, roads, gardens, markets, and public meetings. ",
    )
    rng = random.Random(seed)
    used_keys: set[str] = set()
    rows: list[dict[str, Any]] = []

    for target_length in normalized_lengths:
        for depth_percent in normalized_depths:
            for trial in range(trials):
                while True:
                    passkey = f"{rng.randrange(10000000, 100000000):08d}"
                    if passkey not in used_keys:
                        used_keys.add(passkey)
                        break
                needle_ids = _encode(
                    tokenizer,
                    f"\nThe retrieval passkey is {passkey}. Remember this exact passkey.\n",
                )
                fixed_count = len(prefix_ids) + len(suffix_ids) + len(needle_ids)
                if fixed_count > target_length:
                    raise ValueError(f"target length {target_length} is too short for the passkey prompt")
                filler_budget = target_length - fixed_count
                before_count = round(filler_budget * depth_percent / 100.0)
                after_count = filler_budget - before_count
                before = _repeat_to_length(filler_ids, before_count)
                after = _repeat_to_length(filler_ids, after_count)
                document_ids = before + needle_ids + after
                prompt_ids = truncate_document_only(
                    prefix_ids,
                    document_ids,
                    suffix_ids,
                    target_length,
                )
                depth_label = str(depth_percent).replace(".", "p")
                example_id = f"passkey-L{target_length}-d{depth_label}-t{trial:02d}-s{seed}"
                rows.append(
                    {
                        "schema": SCHEMA,
                        "example_id": example_id,
                        "suite": "passkey",
                        "task": "passkey_retrieval",
                        "target_length": target_length,
                        "prompt_ids": prompt_ids,
                        "answers": [passkey],
                        "metric": "exact_match",
                        "depth_percent": depth_percent,
                        "choices": None,
                        "answer_index": None,
                        "source": {
                            "kind": "deterministic_passkey",
                            "seed": seed,
                            "trial": trial,
                        },
                        "prompt_sha256": _prompt_sha256(prompt_ids),
                        "generation_tokens": 32,
                        "scorer": "normalized_exact_match",
                    }
                )
    return rows


def _expand_jsonl_sources(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        requested = [Path(paths)]
    else:
        requested = [Path(path) for path in paths]
    expanded: list[Path] = []
    for path in requested:
        if path.is_dir():
            matches = sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file())
            if not matches:
                raise FileNotFoundError(f"requested JSONL directory is empty: {path}")
            expanded.extend(matches)
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"requested JSONL source does not exist: {path}")
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(path)
    return deduplicated


def _infer_target_length(row: Mapping[str, Any], path: Path, prompt_length: int) -> int:
    for part in reversed(path.parts):
        match = re.search(r"(?<!\d)(\d{1,3})[kK](?!\w)", part)
        if match:
            return int(match.group(1)) * 1024
        match = re.search(r"(?<!\d)(4096|8192|16384|32768|65536|131072)(?!\d)", part)
        if match:
            return int(match.group(1))
    for key in ("target_length", "length", "context_length", "tokens"):
        value = row.get(key)
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return prompt_length


def _ruler_task(row: Mapping[str, Any], path: Path) -> str:
    for key in ("task", "task_name", "dataset"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    if path.stem.lower() in {"validation", "test", "train", "data"}:
        return path.parent.name
    return path.stem


def _ruler_contract(task: str) -> dict[str, Any]:
    family = str(task).split("_", 1)[0]
    try:
        return dict(RULER_TASK_CONTRACTS[family])
    except KeyError as exc:
        raise ValueError(f"RULER task {task!r} has no pinned scorer/generation contract") from exc


def import_ruler_examples(
    tokenizer: Any,
    paths: str | Path | Iterable[str | Path],
    *,
    max_examples_per_task_length: int | None = None,
) -> list[dict[str, Any]]:
    """Import official RULER JSONL rows without rewriting prompts or references."""
    if max_examples_per_task_length is not None and max_examples_per_task_length <= 0:
        raise ValueError("max_examples_per_task_length must be positive")
    records: list[dict[str, Any]] = []
    cell_counts: Counter[tuple[str, int]] = Counter()
    for path in _expand_jsonl_sources(paths):
        file_hash = _file_sha256(path)
        for line_index, row in enumerate(load_jsonl(path), start=1):
            prompt_input = row.get("input", row.get("prompt"))
            if not isinstance(prompt_input, str):
                raise ValueError(f"RULER row {path}:{line_index} has no string prompt")
            answer_prefix = row.get("answer_prefix", "")
            if not isinstance(answer_prefix, str):
                raise ValueError(f"RULER row {path}:{line_index} has invalid answer_prefix")
            prompt = prompt_input + answer_prefix
            reference_value = row.get("outputs", row.get("answers", row.get("answer", row.get("reference"))))
            answers = _normalize_answers(reference_value)
            prompt_ids = _encode(tokenizer, prompt)
            if not prompt_ids:
                raise ValueError(f"RULER row {path}:{line_index} has an empty prompt")
            task = _ruler_task(row, path)
            contract = _ruler_contract(task)
            target_length = _infer_target_length(row, path, len(prompt_ids))
            cell = (task, target_length)
            if (
                max_examples_per_task_length is not None
                and cell_counts[cell] >= int(max_examples_per_task_length)
            ):
                continue
            prompt_ids = _chat_prompt_ids(tokenizer, prompt)
            if len(prompt_ids) > target_length:
                raise ValueError(
                    f"RULER prompt {path}:{line_index} has {len(prompt_ids)} tokens "
                    f"and exceeds declared target length {target_length}"
                )
            source_id = row.get("id", row.get("index", line_index - 1))
            records.append(
                _make_record(
                    example_id=(f"ruler-{_slug(task)}-L{target_length}-{_slug(path.stem)}-{_slug(source_id)}"),
                    suite="ruler",
                    task=task,
                    target_length=target_length,
                    prompt_ids=prompt_ids,
                    answers=answers,
                    metric="ruler_string_match",
                    generation_tokens=int(contract["generation_tokens"]),
                    scorer=f"ruler_string_match_{contract['match_type']}",
                    source={
                        "kind": "official_ruler_jsonl",
                        "prompt_config": "generic/default",
                        "match_type": contract["match_type"],
                        "tokens_to_generate": int(contract["generation_tokens"]),
                        "length_semantics": "chat_prompt_tokens_before_generation",
                        "file": path.name,
                        "file_sha256": file_hash,
                        "line": line_index,
                        "source_id": str(source_id),
                    },
                )
            )
            cell_counts[cell] += 1
    return records


def _load_json(path: str | Path) -> Any:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"requested JSON source does not exist: {source_path}")
    try:
        return json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON source {source_path}: {exc}") from exc


def _replace_numbered_placeholders(text: str, values: Sequence[Any]) -> str:
    rendered = text
    for index, value in enumerate(values, start=1):
        rendered = rendered.replace("{" + str(index) + "}", str(value))
    return rendered


def build_nolima_hard_examples(
    tokenizer: Any,
    needle_set_path: str | Path,
    books_dir: str | Path,
    lengths: Iterable[int] = (16384, 32768),
    depths: Iterable[float] = (10, 25, 50, 75, 90),
    seed: int = 42,
    max_examples_per_length_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic NoLiMa-Hard prompts from official needles and books."""
    if max_examples_per_length_depth is not None and max_examples_per_length_depth <= 0:
        raise ValueError("max_examples_per_length_depth must be positive")
    needle_path = Path(needle_set_path)
    raw_needles = _load_json(needle_path)
    if isinstance(raw_needles, dict):
        raw_needles = raw_needles.get("needles", raw_needles.get("tests"))
    if not isinstance(raw_needles, list) or not raw_needles:
        raise ValueError("NoLiMa needle set must be a non-empty list")

    book_root = Path(books_dir)
    if not book_root.is_dir():
        raise FileNotFoundError(f"requested NoLiMa books directory does not exist: {book_root}")
    book_paths = sorted(path for path in book_root.rglob("*.txt") if path.is_file())
    if not book_paths:
        raise FileNotFoundError(f"requested NoLiMa books directory has no .txt files: {book_root}")

    normalized_lengths = tuple(int(length) for length in lengths)
    normalized_depths = tuple(float(depth) for depth in depths)
    if not normalized_lengths or any(length <= 0 for length in normalized_lengths):
        raise ValueError("NoLiMa lengths must be positive")
    if not normalized_depths or any(depth < 0 or depth > 100 for depth in normalized_depths):
        raise ValueError("NoLiMa depths must be between 0 and 100")

    needle_hash = _file_sha256(needle_path)
    records: list[dict[str, Any]] = []
    cell_counts: Counter[tuple[int, float]] = Counter()
    for book_index, book_path in enumerate(book_paths):
        book_text = book_path.read_text(encoding="utf-8")
        book_ids = _encode(tokenizer, book_text)
        if not book_ids:
            raise ValueError(f"NoLiMa book tokenized to an empty sequence: {book_path}")
        book_hash = _file_sha256(book_path)
        book_name = book_path.relative_to(book_root).as_posix()

        for experiment in raw_needles:
            if not isinstance(experiment, dict):
                raise ValueError("NoLiMa needle entries must be objects")
            experiment_id = str(experiment.get("id", "unknown"))
            needle_template = experiment.get("needle")
            questions = experiment.get("questions")
            tests = experiment.get("tests")
            if not isinstance(needle_template, str):
                raise ValueError(f"NoLiMa needle {experiment_id} has no needle string")
            if not isinstance(questions, dict) or not isinstance(tests, dict):
                raise ValueError(f"NoLiMa needle {experiment_id} has invalid tests")
            task_template = experiment.get("task_template", DEFAULT_NOLIMA_TEMPLATE)
            if not isinstance(task_template, str) or task_template.count("{haystack}") != 1:
                raise ValueError(f"NoLiMa needle {experiment_id} needs exactly one haystack placeholder")
            system_prompt = experiment.get("system_prompt", "")
            if not isinstance(system_prompt, str):
                raise ValueError(f"NoLiMa needle {experiment_id} has invalid system_prompt")
            character_set = experiment.get("character_set", [])
            if character_set and not isinstance(character_set, list):
                raise ValueError(f"NoLiMa needle {experiment_id} has invalid character_set")

            for question_type, question_template in questions.items():
                if not isinstance(question_template, str):
                    raise ValueError(f"NoLiMa needle {experiment_id} has invalid question")
                for test_id, test in tests.items():
                    if not isinstance(test, dict):
                        raise ValueError(f"NoLiMa test {test_id} must be an object")
                    input_args = test.get("input_args", [])
                    if not isinstance(input_args, list):
                        raise ValueError(f"NoLiMa test {test_id} has invalid input_args")
                    base_needle = _replace_numbered_placeholders(needle_template, input_args)
                    base_question = _replace_numbered_placeholders(question_template, input_args)

                    for target_length in normalized_lengths:
                        for depth_percent in normalized_depths:
                            cell = (target_length, depth_percent)
                            if (
                                max_examples_per_length_depth is not None
                                and cell_counts[cell]
                                >= int(max_examples_per_length_depth)
                            ):
                                continue
                            row_rng = random.Random(
                                f"{seed}|{experiment_id}|{question_type}|{test_id}|"
                                f"{book_index}|{target_length}|{depth_percent:g}"
                            )
                            selected_character: str | None = None
                            needle = base_needle
                            question = base_question
                            if "{CHAR}" in needle or "{CHAR}" in question:
                                if not character_set:
                                    raise ValueError(f"NoLiMa test {experiment_id}/{test_id} requires character_set")
                                selected_character = str(row_rng.choice(character_set))
                                needle = needle.replace("{CHAR}", selected_character)
                                question = question.replace("{CHAR}", selected_character)

                            if selected_character is not None:
                                answers = [selected_character]
                            else:
                                answers = _normalize_answers(test.get("gold_answers", experiment.get("gold_answers")))

                            before_haystack, after_haystack = task_template.split("{haystack}")
                            before_haystack = before_haystack.replace("{question}", question)
                            after_haystack = after_haystack.replace("{question}", question)
                            if "{question}" in before_haystack or "{question}" in after_haystack:
                                raise ValueError("NoLiMa task template has unresolved question")
                            prefix_text = before_haystack
                            prefix_ids = _encode(tokenizer, prefix_text)
                            suffix_ids = _encode(tokenizer, after_haystack)
                            needle_ids = _encode(tokenizer, f" {needle}\n")
                            max_filler = min(len(book_ids), target_length)
                            max_start = len(book_ids) - max_filler
                            start = row_rng.randrange(max_start + 1) if max_start else 0
                            filler = book_ids[start : start + max_filler]
                            low, high = 0, len(filler)
                            prompt_ids: list[int] | None = None
                            selected_filler_tokens = 0
                            while low <= high:
                                filler_count = (low + high) // 2
                                before_count = round(
                                    filler_count * depth_percent / 100.0
                                )
                                document_ids = (
                                    filler[:before_count]
                                    + needle_ids
                                    + filler[before_count:filler_count]
                                )
                                raw_ids = prefix_ids + document_ids + suffix_ids
                                candidate = _chat_prompt_ids(
                                    tokenizer,
                                    _decode_ids(tokenizer, raw_ids),
                                    system_text=system_prompt
                                    or NOLIMA_DEFAULT_SYSTEM_PROMPT,
                                )
                                if len(candidate) <= target_length:
                                    prompt_ids = candidate
                                    selected_filler_tokens = filler_count
                                    low = filler_count + 1
                                else:
                                    high = filler_count - 1
                            if prompt_ids is None:
                                raise ValueError(
                                    f"NoLiMa fixed chat prompt exceeds target length {target_length}"
                                )
                            test_name = f"{experiment_id}_{test_id}_{question_type}"
                            records.append(
                                _make_record(
                                    example_id=(
                                        f"nolima-hard-{_slug(test_name)}-{_slug(book_name)}-"
                                        f"L{target_length}-d{_slug(f'{depth_percent:g}')}"
                                    ),
                                    suite="nolima_hard_exact_context",
                                    task="nolima_hard_exact_context",
                                    target_length=target_length,
                                    prompt_ids=prompt_ids,
                                    answers=answers,
                                    metric="contains",
                                    generation_tokens=192,
                                    scorer="nolima_case_sensitive_contains",
                                    depth_percent=depth_percent,
                                    source={
                                        "kind": "nolima_hard_adapted_exact_context",
                                        "dataset": "amodaresi/NoLiMa",
                                        "needle_set": needle_path.name,
                                        "needle_set_sha256": needle_hash,
                                        "book": book_name,
                                        "book_sha256": book_hash,
                                        "test_name": test_name,
                                        "book_token_start": start,
                                        "book_filler_tokens": selected_filler_tokens,
                                        "selected_character": selected_character,
                                        "prompt_config": "llama3_full_chat",
                                        "official_metric": "contains",
                                        "official_model_max_tokens": 192,
                                        "length_semantics": "full_chat_prompt_tokens_before_generation",
                                        "protocol_note": "official scorer and generation budget; adapted exact full-prompt context length",
                                        "seed": seed,
                                    },
                                )
                            )
                            cell_counts[cell] += 1
    return records


def _longbench_task(row: Mapping[str, Any], path: Path) -> str | None:
    for key in ("dataset", "task"):
        candidate = row.get(key)
        if candidate is None:
            continue
        normalized = str(candidate).lower()
        for task in LONGBENCH_PROMPTS:
            if task in normalized:
                return task
        return None
    normalized_path = path.stem.lower()
    for task in LONGBENCH_PROMPTS:
        if task in normalized_path:
            return task
    return None


def build_longbench_examples(
    tokenizer: Any,
    paths: str | Path | Iterable[str | Path],
    max_prompt_tokens: int = 32768,
    min_complete_prompt_tokens: int = 8192,
    diagnostic_lengths: Iterable[int] = (),
    max_complete_examples_per_task: int | None = None,
) -> list[dict[str, Any]]:
    """Freeze complete LongBench v1 prompts plus explicitly requested diagnostics."""
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if not 0 < int(min_complete_prompt_tokens) <= int(max_prompt_tokens):
        raise ValueError("min_complete_prompt_tokens must fit inside max_prompt_tokens")
    if max_complete_examples_per_task is not None and max_complete_examples_per_task <= 0:
        raise ValueError("max_complete_examples_per_task must be positive")
    diagnostics = tuple(sorted({int(length) for length in diagnostic_lengths}))
    if any(length <= 0 or length > max_prompt_tokens for length in diagnostics):
        raise ValueError("diagnostic lengths must be within max_prompt_tokens")

    records: list[dict[str, Any]] = []
    complete_counts: Counter[str] = Counter()
    for path in _expand_jsonl_sources(paths):
        file_hash = _file_sha256(path)
        for line_index, row in enumerate(load_jsonl(path), start=1):
            task = _longbench_task(row, path)
            if task is None:
                continue
            context = row.get("context")
            question = row.get("input", row.get("question"))
            if not isinstance(context, str) or not isinstance(question, str):
                raise ValueError(f"LongBench row {path}:{line_index} needs context and input strings")
            answers = _normalize_answers(row.get("answers", row.get("answer")))
            template = LONGBENCH_PROMPTS[task]
            before_context, after_context = template.split("{context}")
            prefix_ids = _encode(tokenizer, before_context.replace("{input}", question))
            document_ids = _encode(tokenizer, context)
            suffix_ids = _encode(tokenizer, after_context.replace("{input}", question))
            complete_prompt = template.format(context=context, input=question)
            complete_ids = _chat_prompt_ids(tokenizer, complete_prompt)
            source_id = row.get("_id", row.get("id", line_index - 1))
            common_source = {
                "kind": "longbench_v1_jsonl",
                "file": path.name,
                "file_sha256": file_hash,
                "line": line_index,
                "source_id": str(source_id),
            }
            if (
                int(min_complete_prompt_tokens) <= len(complete_ids) <= max_prompt_tokens
                and (
                    max_complete_examples_per_task is None
                    or complete_counts[task] < int(max_complete_examples_per_task)
                )
            ):
                registered_length = next(
                    length for length in (8192, 16384, 32768) if len(complete_ids) <= length
                )
                records.append(
                    _make_record(
                        example_id=f"longbench-{task}-{_slug(source_id)}-complete",
                        suite="longbench",
                        task=task,
                        target_length=registered_length,
                        prompt_ids=complete_ids,
                        answers=answers,
                        metric="qa_f1",
                        generation_tokens=128,
                        scorer="longbench_v1_qa_f1",
                        source={
                            **common_source,
                            "selection": "complete",
                            "prompt_config": "llama3_full_chat",
                            "official_max_generation_tokens": 128,
                            "prompt_tokens": len(complete_ids),
                            "registered_length_bucket": registered_length,
                        },
                    )
                )
                complete_counts[task] += 1

            for target_length in diagnostics:
                if len(complete_ids) <= target_length:
                    continue
                prompt_ids = _chat_segmented_prompt_ids(
                    tokenizer,
                    prefix_ids=prefix_ids,
                    document_ids=document_ids,
                    suffix_ids=suffix_ids,
                    max_prompt_tokens=target_length,
                )
                records.append(
                    _make_record(
                        example_id=(f"longbench-{task}-{_slug(source_id)}-diag-L{target_length}"),
                        suite="longbench",
                        task=f"{task}_fixed_diagnostic",
                        target_length=target_length,
                        prompt_ids=prompt_ids,
                        answers=answers,
                        metric="qa_f1",
                        generation_tokens=128,
                        scorer="longbench_v1_qa_f1",
                        source={
                            **common_source,
                            "selection": "fixed_diagnostic",
                            "untruncated_prompt_tokens": len(complete_ids),
                            "prompt_config": "llama3_full_chat",
                            "official_max_generation_tokens": 128,
                        },
                    )
                )
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record["task"], record["target_length"], record["prompt_sha256"])
        source_id = str(record["source"]["source_id"])
        if key not in merged:
            item = dict(record)
            item["source"] = {
                **record["source"],
                "merged_source_ids": [source_id],
            }
            merged[key] = item
            continue
        item = merged[key]
        for answer in record["answers"]:
            if answer not in item["answers"]:
                item["answers"].append(answer)
        if source_id not in item["source"]["merged_source_ids"]:
            item["source"]["merged_source_ids"].append(source_id)
    return list(merged.values())


def _mcqa_values(dataset_name: str, row: Mapping[str, Any]) -> tuple[str, list[str], int]:
    if dataset_name == "cais/mmlu":
        question = row.get("question")
        choices = row.get("choices")
        answer_index = row.get("answer")
    elif dataset_name in {"allenai/ai2_arc", "allenai/openbookqa"}:
        question = row.get("question" if dataset_name == "allenai/ai2_arc" else "question_stem")
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, dict):
            raise ValueError("ARC row has invalid choices")
        choices = raw_choices.get("text")
        labels = raw_choices.get("label")
        answer_key = str(row.get("answerKey", ""))
        if not isinstance(labels, list) or answer_key not in [str(label) for label in labels]:
            raise ValueError("ARC answerKey is not present in choice labels")
        answer_index = [str(label) for label in labels].index(answer_key)
    elif dataset_name == "Rowan/hellaswag":
        question = row.get("ctx", row.get("context"))
        choices = row.get("endings")
        answer_index = row.get("label")
    elif dataset_name == "allenai/winogrande":
        question = row.get("sentence")
        choices = [row.get("option1"), row.get("option2")]
        answer_value = row.get("answer")
        answer_index = int(answer_value) - 1
    else:
        raise ValueError(f"unsupported MCQA source: {dataset_name}")

    if not isinstance(question, str):
        raise ValueError(f"{dataset_name} row has no question string")
    if not isinstance(choices, (list, tuple)) or len(choices) < 2:
        raise ValueError(f"{dataset_name} row has invalid choices")
    normalized_choices = [str(choice) for choice in choices]
    try:
        normalized_index = int(answer_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dataset_name} row has invalid answer index") from exc
    if not 0 <= normalized_index < len(normalized_choices):
        raise ValueError(f"{dataset_name} answer index is outside choices")
    return question, normalized_choices, normalized_index


def mcqa_arrow_path(cache_root: str | Path, dataset_name: str) -> Path:
    """Resolve one revision-pinned Datasets Arrow file without Hub metadata."""
    if dataset_name not in MCQA_REVISIONS:
        raise ValueError(f"MCQA source is not revision-pinned: {dataset_name}")
    dataset_dir, file_prefix = MCQA_ARROW_LAYOUT[dataset_name]
    spec = MCQA_SPECS[dataset_name]
    config = spec["config"] or "default"
    path = (
        Path(cache_root)
        / "datasets"
        / dataset_dir
        / config
        / "0.0.0"
        / MCQA_REVISIONS[dataset_name]
        / f"{file_prefix}-{spec['split']}.arrow"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_local_mcqa_arrow_datasets(
    cache_root: str | Path,
    sources: Iterable[str],
) -> dict[str, Any]:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("the datasets package is required to read MCQA Arrow files") from exc
    return {
        dataset_name: Dataset.from_file(str(mcqa_arrow_path(cache_root, dataset_name)))
        for dataset_name in sources
    }


def load_mcqa_examples(
    tokenizer: Any,
    *,
    sources: Iterable[str] = tuple(MCQA_REVISIONS),
    dataset_loader: Callable[..., Any] | None = None,
    max_prompt_tokens: int = 32768,
    max_examples_per_source: int | None = None,
    cache_dir: str | Path | None = None,
    local_datasets: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Load pinned Hugging Face MCQA sources with no unpinned fallback."""
    if dataset_loader is None:
        try:
            from datasets import load_dataset as dataset_loader
        except ImportError as exc:
            raise RuntimeError("the datasets package is required for MCQA preparation") from exc
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if max_examples_per_source is not None and max_examples_per_source <= 0:
        raise ValueError("max_examples_per_source must be positive")

    records: list[dict[str, Any]] = []
    for dataset_name in sources:
        records_before_source = len(records)
        if dataset_name not in MCQA_REVISIONS:
            raise ValueError(f"MCQA source is not revision-pinned: {dataset_name}")
        spec = MCQA_SPECS[dataset_name]
        if local_datasets is not None:
            if dataset_name not in local_datasets:
                raise FileNotFoundError(f"local MCQA dataset is missing: {dataset_name}")
            dataset = local_datasets[dataset_name]
        else:
            load_kwargs: dict[str, Any] = {
                "split": spec["split"],
                "revision": MCQA_REVISIONS[dataset_name],
            }
            if cache_dir is not None:
                load_kwargs["cache_dir"] = str(cache_dir)
            try:
                if spec["config"] is None:
                    dataset = dataset_loader(dataset_name, **load_kwargs)
                else:
                    dataset = dataset_loader(dataset_name, spec["config"], **load_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load requested MCQA source {dataset_name} at revision {MCQA_REVISIONS[dataset_name]}"
                ) from exc

        for row_index, row in enumerate(dataset):
            if max_examples_per_source is not None and row_index >= max_examples_per_source:
                break
            if not isinstance(row, Mapping):
                raise ValueError(f"{dataset_name} row {row_index} is not an object")
            question, choices, answer_index = _mcqa_values(dataset_name, row)
            labels = [chr(ord("A") + index) for index in range(len(choices))]
            choice_block = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
            prompt = f"Question: {question}\n\nChoices:\n{choice_block}\n\nAnswer:"
            prompt_ids = _chat_prompt_ids(tokenizer, prompt)
            if len(prompt_ids) > max_prompt_tokens:
                continue
            target_length = next(
                length for length in (8192, 16384, 32768) if len(prompt_ids) <= length
            )
            source_id = row.get("id", row.get("ind", row.get("qID", row_index)))
            task = str(spec["task"])
            records.append(
                _make_record(
                    example_id=f"mcqa-{task}-{_slug(source_id)}",
                    suite="mcqa",
                    task=task,
                    target_length=target_length,
                    prompt_ids=prompt_ids,
                    answers=[choices[answer_index]],
                    metric="mcqa",
                    choices=choices,
                    answer_index=answer_index,
                    source={
                        "kind": "huggingface_dataset",
                        "dataset": dataset_name,
                        "revision": MCQA_REVISIONS[dataset_name],
                        "config": spec["config"],
                        "split": spec["split"],
                        "source_id": str(source_id),
                        "row_index": row_index,
                        "prompt_config": "llama3_full_chat",
                        "prompt_tokens": len(prompt_ids),
                        "registered_length_bucket": target_length,
                    },
                )
            )
        if len(records) == records_before_source:
            raise ValueError(f"{dataset_name} produced no eligible records")
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_suite_atomic(
    *,
    output_dir: str | Path,
    records_by_file: Mapping[str, Iterable[Mapping[str, Any]]],
    tokenizer_identity: Mapping[str, Any],
    source_revisions: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and atomically publish suite files, with ``manifest.json`` last."""
    if not records_by_file:
        raise ValueError("records_by_file must not be empty")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for filename, records in records_by_file.items():
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError(f"invalid suite filename: {filename!r}")
        rows = [dict(record) for record in records]
        if not rows:
            raise ValueError(f"suite file {filename} has no records")
        normalized[filename] = rows

    all_records = validate_records(record for filename in sorted(normalized) for record in normalized[filename])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    manifest_incomplete = output_path / "manifest.json.incomplete"
    final_paths = {name: output_path / name for name in normalized}
    incomplete_paths = {name: output_path / f"{name}.incomplete" for name in normalized}
    existing_jsonl = set(output_path.glob("*.jsonl"))
    stale_jsonl = sorted(existing_jsonl.difference(final_paths.values()))
    conflict_candidates = {
        manifest_path,
        manifest_incomplete,
        *final_paths.values(),
        *incomplete_paths.values(),
        *existing_jsonl,
    }
    conflicts = sorted(path for path in conflict_candidates if path.exists())
    if conflicts and not overwrite:
        raise FileExistsError("capability-suite output already exists: " + ", ".join(path.name for path in conflicts))

    manifest_removed = False
    try:
        file_metadata: dict[str, dict[str, Any]] = {}
        for filename in sorted(normalized):
            rows = validate_records(normalized[filename])
            incomplete = incomplete_paths[filename]
            _write_jsonl(incomplete, rows)
            reloaded = load_jsonl(incomplete)
            validate_records(reloaded)
            if reloaded != rows:
                raise RuntimeError(f"round-trip validation failed for {filename}")
            prompt_lengths = [len(row["prompt_ids"]) for row in rows]
            file_metadata[filename] = {
                "sha256": _file_sha256(incomplete),
                "size_bytes": incomplete.stat().st_size,
                "row_count": len(rows),
                "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
                "prompt_length_tokens": {
                    "min": min(prompt_lengths),
                    "max": max(prompt_lengths),
                },
            }

        task_counts = dict(sorted(Counter(record["task"] for record in all_records).items()))
        all_prompt_lengths = [len(record["prompt_ids"]) for record in all_records]
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "example_schema": SCHEMA,
            "prompt_hash_encoding": PROMPT_HASH_ENCODING,
            "tokenizer": dict(tokenizer_identity),
            "source_revisions": dict(source_revisions),
            "files": file_metadata,
            "row_count": len(all_records),
            "task_counts": task_counts,
            "prompt_length_tokens": {
                "min": min(all_prompt_lengths),
                "max": max(all_prompt_lengths),
            },
        }

        if manifest_path.exists():
            manifest_path.unlink()
            manifest_removed = True
        for stale_path in stale_jsonl:
            stale_path.unlink()
        for filename in sorted(normalized):
            os.replace(incomplete_paths[filename], final_paths[filename])
        _write_json(manifest_incomplete, manifest)
        os.replace(manifest_incomplete, manifest_path)
        return manifest
    except Exception:
        if manifest_removed:
            manifest_path.unlink(missing_ok=True)
        raise
    finally:
        manifest_incomplete.unlink(missing_ok=True)
        for path in incomplete_paths.values():
            path.unlink(missing_ok=True)


def _public_tokenizer_identifier(value: Any) -> str:
    identifier = str(value)
    path = Path(identifier).expanduser()
    return path.name if path.is_absolute() else identifier


def _tokenizer_identity(tokenizer: Any, requested: str | Path | None) -> dict[str, Any]:
    public_requested = _public_tokenizer_identifier(
        requested if requested is not None else getattr(tokenizer, "name_or_path", "unknown")
    )
    identity: dict[str, Any] = {
        "identifier": public_requested,
        "requested": (None if requested is None else public_requested),
        "name_or_path": _public_tokenizer_identifier(getattr(tokenizer, "name_or_path", requested or "unknown")),
        "class": type(tokenizer).__name__,
    }
    if requested is not None:
        tokenizer_source = Path(requested).expanduser()
        if tokenizer_source.is_dir():
            identity["files"] = {
                name: _file_sha256(tokenizer_source / name)
                for name in (
                    "tokenizer.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
                if (tokenizer_source / name).is_file()
            }
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, key, None)
        if value is not None:
            identity[key] = int(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        revision = init_kwargs.get("revision", init_kwargs.get("_commit_hash"))
        if revision is not None:
            identity["revision"] = str(revision)
    return identity


def _load_tokenizer(tokenizer_path: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("the transformers package is required to load the tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        use_fast=True,
    )


def _namespace_value(args: argparse.Namespace, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _parse_sequence(value: Any, caster: Callable[[Any], Any]) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [piece.strip() for piece in value.split(",") if piece.strip()]
    else:
        values = list(value)
    return tuple(caster(item) for item in values)


def _local_source_revisions(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [{"file": path.name, "sha256": _file_sha256(path), "size_bytes": path.stat().st_size} for path in paths]


def prepare_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare every requested source, then publish a manifest-backed suite."""
    tokenizer_arg = _namespace_value(args, "tokenizer", "tokenizer_path")
    if tokenizer_arg is None:
        raise ValueError("tokenizer path is required")
    if isinstance(tokenizer_arg, (str, Path)):
        tokenizer = _load_tokenizer(tokenizer_arg)
        requested_tokenizer: str | Path | None = tokenizer_arg
    else:
        tokenizer = tokenizer_arg
        requested_tokenizer = None

    output_dir = _namespace_value(args, "output_dir")
    if output_dir is None:
        raise ValueError("output_dir is required")
    seed = int(_namespace_value(args, "seed", default=42))
    if seed != 42:
        raise ValueError("the frozen capability suite requires seed 42")

    records_by_file: dict[str, list[dict[str, Any]]] = {}
    source_revisions: dict[str, Any] = {}
    if not bool(_namespace_value(args, "skip_passkey", default=False)):
        passkey_rows = build_passkey_examples(
            tokenizer,
            lengths=(8192, 16384, 32768),
            depths=(10, 25, 50, 75, 90),
            trials=20,
            seed=42,
        )
        records_by_file["passkey.jsonl"] = passkey_rows
        source_revisions["passkey"] = {
            "generator": "deterministic_passkey",
            "seed": 42,
        }

    ruler_requested = _namespace_value(args, "ruler_jsonl", default=()) or ()
    if ruler_requested:
        ruler_paths = _expand_jsonl_sources(ruler_requested)
        ruler_rows = import_ruler_examples(
            tokenizer,
            ruler_paths,
            max_examples_per_task_length=_namespace_value(
                args, "max_ruler_per_task_length", default=None
            ),
        )
        if not ruler_rows:
            raise ValueError("requested RULER sources produced no records")
        records_by_file["ruler.jsonl"] = ruler_rows
        source_revisions["ruler"] = _local_source_revisions(ruler_paths)

    needle_set_path = _namespace_value(args, "nolima_needle_set", "nolima_needles", default=None)
    books_dir = _namespace_value(args, "nolima_books_dir", default=None)
    if (needle_set_path is None) != (books_dir is None):
        raise ValueError("NoLiMa preparation requires both nolima_needle_set and nolima_books_dir")
    if needle_set_path is not None:
        nolima_lengths = _parse_sequence(_namespace_value(args, "nolima_lengths", default=(16384, 32768)), int)
        nolima_depths = _parse_sequence(
            _namespace_value(args, "nolima_depths", default=(10, 25, 50, 75, 90)),
            float,
        )
        nolima_rows = build_nolima_hard_examples(
            tokenizer,
            needle_set_path,
            books_dir,
            lengths=nolima_lengths,
            depths=nolima_depths,
            seed=42,
            max_examples_per_length_depth=_namespace_value(
                args, "max_nolima_per_length_depth", default=None
            ),
        )
        if not nolima_rows:
            raise ValueError("requested NoLiMa sources produced no records")
        records_by_file["nolima_hard.jsonl"] = nolima_rows
        needle_path = Path(needle_set_path)
        book_paths = sorted(Path(books_dir).rglob("*.txt"))
        source_revisions["nolima_hard"] = {
            "needle_set": {
                "file": needle_path.name,
                "sha256": _file_sha256(needle_path),
            },
            "books": _local_source_revisions(book_paths),
        }

    longbench_requested = _namespace_value(args, "longbench_jsonl", default=()) or ()
    if longbench_requested:
        longbench_paths = _expand_jsonl_sources(longbench_requested)
        diagnostics = _parse_sequence(_namespace_value(args, "longbench_diagnostic_lengths", default=()), int)
        longbench_rows = build_longbench_examples(
            tokenizer,
            longbench_paths,
            max_prompt_tokens=32768,
            diagnostic_lengths=diagnostics,
            max_complete_examples_per_task=_namespace_value(
                args, "max_longbench_per_task", default=None
            ),
        )
        if not longbench_rows:
            raise ValueError("requested LongBench sources produced no selected NarrativeQA or Qasper records")
        records_by_file["longbench.jsonl"] = longbench_rows
        source_revisions["longbench"] = _local_source_revisions(longbench_paths)

    include_mcqa = bool(_namespace_value(args, "include_mcqa", default=True))
    skip_mcqa = bool(_namespace_value(args, "skip_mcqa", default=False))
    if include_mcqa and not skip_mcqa:
        requested_mcqa = _parse_sequence(_namespace_value(args, "mcqa_sources", default=tuple(MCQA_REVISIONS)), str)
        max_mcqa = _namespace_value(args, "max_mcqa_per_source", default=None)
        arrow_cache = _namespace_value(args, "mcqa_arrow_cache", default=None)
        local_datasets = (
            load_local_mcqa_arrow_datasets(arrow_cache, requested_mcqa)
            if arrow_cache is not None
            else None
        )
        mcqa_rows = load_mcqa_examples(
            tokenizer,
            sources=requested_mcqa,
            max_examples_per_source=max_mcqa,
            cache_dir=_namespace_value(args, "cache_dir", default=None),
            local_datasets=local_datasets,
        )
        if not mcqa_rows:
            raise ValueError("requested MCQA sources produced no records")
        records_by_file["mcqa.jsonl"] = mcqa_rows
        source_revisions["mcqa"] = {dataset: MCQA_REVISIONS[dataset] for dataset in requested_mcqa}

    source_revisions["preparation_selection"] = {
        "max_ruler_per_task_length": _namespace_value(
            args, "max_ruler_per_task_length", default=None
        ),
        "max_nolima_per_length_depth": _namespace_value(
            args, "max_nolima_per_length_depth", default=None
        ),
        "max_longbench_per_task": _namespace_value(
            args, "max_longbench_per_task", default=None
        ),
        "max_mcqa_per_source": _namespace_value(
            args, "max_mcqa_per_source", default=None
        ),
    }

    return write_suite_atomic(
        output_dir=output_dir,
        records_by_file=records_by_file,
        tokenizer_identity=_tokenizer_identity(tokenizer, requested_tokenizer),
        source_revisions=source_revisions,
        overwrite=bool(_namespace_value(args, "overwrite", default=False)),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the frozen seed-42 capability-evaluation suite")
    parser.add_argument("--tokenizer", required=True, help="LLaMA tokenizer path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ruler-jsonl", type=Path, nargs="+", default=())
    parser.add_argument("--max-ruler-per-task-length", type=int)
    parser.add_argument("--nolima-needle-set", type=Path)
    parser.add_argument("--nolima-books-dir", type=Path)
    parser.add_argument("--max-nolima-per-length-depth", type=int)
    parser.add_argument("--longbench-jsonl", type=Path, nargs="+", default=())
    parser.add_argument(
        "--longbench-diagnostic-lengths",
        default="",
        help="comma-separated fixed lengths; only the document may be truncated",
    )
    parser.add_argument("--max-longbench-per-task", type=int)
    parser.add_argument(
        "--mcqa-sources",
        default=",".join(MCQA_REVISIONS),
        help="comma-separated pinned Hugging Face dataset names",
    )
    parser.add_argument("--max-mcqa-per-source", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--mcqa-arrow-cache",
        type=Path,
        help="explicit Hugging Face cache root containing revision-pinned Arrow files",
    )
    parser.add_argument("--skip-mcqa", action="store_true")
    parser.add_argument("--skip-passkey", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(seed=42, include_mcqa=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = prepare_suite(args)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output_dir) / "manifest.json"),
                "row_count": manifest["row_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
