"""Build exact-distance token records for the frequency-adaptation curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from experiments.lora_evq_v2.prepare_positional_distill_data import (
    sha256_file,
    tokenizer_source_fingerprint,
)

from .curriculum import answer_only_labels, get_phase


def _tokens(values: Sequence[int], name: str) -> Tuple[int, ...]:
    tokens = tuple(int(value) for value in values)
    if not tokens:
        raise ValueError(f"{name} must contain at least one token")
    if any(value < 0 for value in tokens):
        raise ValueError(f"{name} contains a negative token id")
    return tokens


@dataclass(frozen=True)
class RetrievalTemplate:
    """Token-level template whose boundaries stay explicit and measurable."""

    instruction: Tuple[int, ...]
    source_prefix: Tuple[int, ...]
    source_infix: Tuple[int, ...]
    source_suffix: Tuple[int, ...]
    query_prefix: Tuple[int, ...]
    query_suffix: Tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "instruction",
            "source_prefix",
            "source_infix",
            "source_suffix",
            "query_prefix",
            "query_suffix",
        ):
            object.__setattr__(self, field_name, _tokens(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class RetrievalExample:
    input_ids: torch.Tensor
    answer_start: int
    answer_end: int
    source_start: int
    source_end: int
    source_value_start: int
    source_value_end: int
    query_key_start: int
    query_key_end: int
    distance: int
    query_key_distance: int
    task_type: str
    variant: str = "train"
    group_id: Optional[str] = None

    def labels(self) -> torch.Tensor:
        return answer_only_labels(self.input_ids, self.answer_start, self.answer_end)


def build_retrieval_example(
    *,
    template: RetrievalTemplate,
    key_ids: Sequence[int],
    value_ids: Sequence[int],
    filler_ids: Sequence[int],
    seq_len: int,
    target_distance: int,
    eos_token_id: int,
    task_type: str,
    pre_source_ids: Sequence[int] = (),
) -> RetrievalExample:
    """Construct one exact-length record with an exact source/answer-query distance."""
    if task_type not in {"kv", "update"}:
        raise ValueError("task_type must be 'kv' or 'update'")
    key = _tokens(key_ids, "key_ids")
    value = _tokens(value_ids, "value_ids")
    filler = tuple(int(token) for token in filler_ids)
    pre_source = tuple(int(token) for token in pre_source_ids)
    if task_type == "update" and not pre_source:
        raise ValueError("update examples require pre_source_ids with the old assignment")
    if task_type == "kv" and pre_source:
        raise ValueError("kv examples must not contain pre_source_ids")

    source = template.source_prefix + key + template.source_infix + value + template.source_suffix
    value_offset = len(template.source_prefix) + len(key) + len(template.source_infix)
    query = template.query_prefix + key + template.query_suffix
    answer = value + (int(eos_token_id),)

    # The RoPE-relevant query is the hidden position that predicts the first
    # answer token, i.e. the final assistant-header position at answer_start-1.
    fixed_distance = len(source) - value_offset + len(query) - 1
    after_count = int(target_distance) - fixed_distance
    if after_count < 0:
        raise ValueError(f"target_distance={target_distance} is shorter than template minimum {fixed_distance}")

    fixed_tokens = len(template.instruction) + len(pre_source) + len(source) + after_count + len(query) + len(answer)
    before_count = int(seq_len) - fixed_tokens
    if before_count < 0:
        raise ValueError("sequence length is too short for the requested distance and template")
    needed_filler = before_count + after_count
    if len(filler) < needed_filler:
        raise ValueError(f"filler_ids has {len(filler)} tokens but {needed_filler} are required")

    before = filler[:before_count]
    after = filler[before_count:needed_filler]
    prefix = template.instruction + pre_source + before
    source_start = len(prefix)
    source_end = source_start + len(source)
    source_value_start = source_start + value_offset
    source_value_end = source_value_start + len(value)
    query_start = source_end + len(after)
    query_key_start = query_start + len(template.query_prefix)
    query_key_end = query_key_start + len(key)
    answer_start = query_start + len(query)
    answer_end = answer_start + len(answer)

    input_ids = torch.tensor(
        prefix + source + after + query + answer,
        dtype=torch.int32,
    )
    if input_ids.numel() != seq_len:
        raise RuntimeError("constructed sequence does not match the requested length")
    measured_distance = answer_start - 1 - source_value_start
    if measured_distance != int(target_distance):
        raise RuntimeError("constructed source/query distance does not match its contract")
    query_key_distance = query_key_start - source_value_start

    return RetrievalExample(
        input_ids=input_ids,
        answer_start=answer_start,
        answer_end=answer_end,
        source_start=source_start,
        source_end=source_end,
        source_value_start=source_value_start,
        source_value_end=source_value_end,
        query_key_start=query_key_start,
        query_key_end=query_key_end,
        distance=measured_distance,
        query_key_distance=query_key_distance,
        task_type=task_type,
    )


def build_counterfactual_triplet(
    example: RetrievalExample,
    *,
    swapped_value_ids: Sequence[int],
    removal_fill_ids: Sequence[int],
    group_id: str,
) -> Tuple[RetrievalExample, RetrievalExample, RetrievalExample]:
    """Create position-matched original, source-swap, and source-removal rows."""
    swapped_value = _tokens(swapped_value_ids, "swapped_value_ids")
    value_length = example.source_value_end - example.source_value_start
    answer_value_length = example.answer_end - example.answer_start - 1
    if len(swapped_value) != value_length or len(swapped_value) != answer_value_length:
        raise ValueError("swapped value must preserve source and answer token lengths")
    removal_fill = tuple(int(token) for token in removal_fill_ids)
    if len(removal_fill) != example.source_end - example.source_start:
        raise ValueError("removal fill must exactly match the source span length")

    original = replace(example, variant="original", group_id=str(group_id))

    swapped_ids = example.input_ids.clone()
    swapped_ids[example.source_value_start : example.source_value_end] = torch.tensor(
        swapped_value, dtype=swapped_ids.dtype
    )
    swapped_ids[example.answer_start : example.answer_end - 1] = torch.tensor(swapped_value, dtype=swapped_ids.dtype)
    swapped = replace(
        example,
        input_ids=swapped_ids,
        variant="swapped",
        group_id=str(group_id),
    )

    removed_ids = example.input_ids.clone()
    removed_ids[example.source_start : example.source_end] = torch.tensor(removal_fill, dtype=removed_ids.dtype)
    removed = replace(
        example,
        input_ids=removed_ids,
        variant="source_removed",
        group_id=str(group_id),
    )
    return original, swapped, removed


def _example_metadata(example: RetrievalExample) -> Dict[str, Any]:
    return {
        "task_type": example.task_type,
        "variant": example.variant,
        "group_id": example.group_id,
        "distance": int(example.distance),
        "query_key_distance": int(example.query_key_distance),
        "source_start": int(example.source_start),
        "source_end": int(example.source_end),
        "source_value_start": int(example.source_value_start),
        "source_value_end": int(example.source_value_end),
        "query_key_start": int(example.query_key_start),
        "query_key_end": int(example.query_key_end),
    }


def stack_examples(
    examples: Sequence[RetrievalExample],
    *,
    phase: str,
    split: str,
    seed: int,
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    """Store fixed-length records compactly without materializing label tensors."""
    if not examples:
        raise ValueError("cannot stack an empty example sequence")
    seq_len = int(examples[0].input_ids.numel())
    if any(example.input_ids.numel() != seq_len for example in examples):
        raise ValueError("all examples in a bundle must have the same sequence length")
    bundle = {
        "format_version": 1,
        "purpose": "llama8b_rope_frequency_adaptation",
        "phase": str(phase),
        "split": str(split),
        "seed": int(seed),
        "seq_len": seq_len,
        "input_ids": torch.stack([example.input_ids.to(torch.int32) for example in examples]),
        "answer_start": torch.tensor([example.answer_start for example in examples], dtype=torch.int32),
        "answer_end": torch.tensor([example.answer_end for example in examples], dtype=torch.int32),
        "metadata": [_example_metadata(example) for example in examples],
        "protocol": dict(protocol),
    }
    return validate_bundle(
        bundle,
        expected_phase=str(phase),
        expected_split=str(split),
        expected_seq_len=seq_len,
    )


def validate_bundle(
    bundle: Dict[str, Any],
    *,
    expected_phase: Optional[str] = None,
    expected_split: Optional[str] = None,
    expected_seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Fail closed on malformed or position-inconsistent tensor bundles."""
    if not isinstance(bundle, dict) or int(bundle.get("format_version", -1)) != 1:
        raise ValueError("frequency-adaptation bundle must use format_version=1")
    if bundle.get("purpose", "llama8b_rope_frequency_adaptation") != "llama8b_rope_frequency_adaptation":
        raise ValueError("bundle purpose does not match frequency adaptation")
    if expected_phase is not None and bundle.get("phase") != expected_phase:
        raise ValueError(f"bundle phase mismatch: expected {expected_phase}, found {bundle.get('phase')}")
    if expected_split is not None and bundle.get("split") != expected_split:
        raise ValueError(f"bundle split mismatch: expected {expected_split}, found {bundle.get('split')}")
    input_ids = bundle.get("input_ids")
    answer_start = bundle.get("answer_start")
    answer_end = bundle.get("answer_end")
    metadata = bundle.get("metadata")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.dtype != torch.int32:
        raise ValueError("bundle input_ids must be a two-dimensional int32 tensor")
    rows, seq_len = input_ids.shape
    if rows <= 0:
        raise ValueError("bundle must contain at least one row")
    if expected_seq_len is not None and seq_len != int(expected_seq_len):
        raise ValueError(f"bundle sequence length mismatch: expected {expected_seq_len}, found {seq_len}")
    if int(bundle.get("seq_len", seq_len)) != seq_len:
        raise ValueError("bundle seq_len metadata does not match input_ids")
    for name, tensor in (("answer_start", answer_start), ("answer_end", answer_end)):
        if not torch.is_tensor(tensor) or tensor.ndim != 1 or tensor.numel() != rows:
            raise ValueError(f"bundle {name} must have one entry per row")
    if not isinstance(metadata, list) or len(metadata) != rows:
        raise ValueError("bundle metadata must have one dictionary per row")
    if not torch.all((0 <= answer_start) & (answer_start < answer_end) & (answer_end <= seq_len)):
        raise ValueError("bundle contains an invalid answer span")
    for index, row in enumerate(metadata):
        if not isinstance(row, dict):
            raise ValueError("bundle metadata rows must be dictionaries")
        distance = int(row.get("distance", -1))
        source_value_start = int(row.get("source_value_start", -1))
        query_key_start = int(row.get("query_key_start", -1))
        query_key_distance = int(row.get("query_key_distance", -1))
        first_answer_predictor = int(answer_start[index]) - 1
        if source_value_start >= 0:
            if first_answer_predictor - source_value_start != distance:
                raise ValueError(f"bundle row {index} has inconsistent distance metadata")
            if query_key_start - source_value_start != query_key_distance:
                raise ValueError(f"bundle row {index} has inconsistent query-key distance metadata")
    return bundle


def _sha256_json(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_torch_save(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_frozen_filler(filler_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load and hash-check the existing document-disjoint plain-text packs."""
    filler_dir = Path(filler_dir)
    manifest_path = filler_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("frozen filler directory is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "llama8b_positional_hidden_distillation":
        raise ValueError("filler manifest is not the frozen plain-text LLaMA-8B dataset")
    if manifest.get("split_policy") != "document_disjoint_validation_then_train":
        raise ValueError("filler manifest must use document-disjoint train/validation splits")

    tensors = {}
    for split in ("train", "validation"):
        record = manifest.get("files", {}).get(split, {})
        path = filler_dir / str(record.get("name", ""))
        if not path.is_file():
            raise FileNotFoundError(f"frozen {split} filler tensor is missing")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"frozen {split} filler tensor hash mismatch")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.dtype != torch.int32:
            raise ValueError(f"frozen {split} filler must be a two-dimensional int32 tensor")
        tensors[split] = tensor
    return tensors["train"], tensors["validation"], manifest


_TOKEN_PATTERN = re.compile(r"^ ?[A-Za-z]{3,14}$")


def build_nonce_pool(tokenizer, split: str, minimum: int = 2048) -> Tuple[int, ...]:
    """Derive disjoint train/eval pools of ordinary one-token words."""
    if split not in {"train", "eval"}:
        raise ValueError("nonce split must be 'train' or 'eval'")
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("tokenizer must expose a positive vocab_size")
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    candidates: List[int] = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if _TOKEN_PATTERN.fullmatch(decoded):
            candidates.append(token_id)
    required = 2 * int(minimum)
    if len(candidates) < required:
        raise ValueError(f"tokenizer exposes only {len(candidates)} suitable nonce tokens; need {required}")
    selected = candidates[:required]
    midpoint = len(selected) // 2
    return tuple(selected[:midpoint] if split == "train" else selected[midpoint:])


def _encode_literal(tokenizer, text: str) -> Tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return _tokens(encoded, f"template literal {text!r}")


def _chat_template_input_ids(value: Any) -> Tuple[int, ...]:
    """Normalize list-style and BatchEncoding-style tokenizer outputs."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("tokenizer chat template did not return a one-dimensional input_ids sequence")
    return _tokens(value, "chat template input_ids")


def _chat_boundaries(tokenizer) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    messages = [{"role": "user", "content": ""}]
    user_turn = _chat_template_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    generation_prompt = _chat_template_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if not user_turn or generation_prompt[: len(user_turn)] != user_turn:
        raise ValueError("tokenizer chat template does not expose a stable user-turn prefix")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None or user_turn[-1] != int(eos_token_id):
        raise ValueError("empty user turn must end with tokenizer.eos_token_id")
    assistant_header = generation_prompt[len(user_turn) :]
    if not assistant_header:
        raise ValueError("chat template add_generation_prompt produced no assistant header")
    return user_turn[:-1], (user_turn[-1],) + assistant_header


def build_template(tokenizer, split: str) -> RetrievalTemplate:
    """Use held-out wording for evaluation while retaining the same task semantics."""
    if split == "train":
        literals = {
            "instruction": "Read the document and return only the requested stored value.\n\n",
            "source_prefix": "\nRecord",
            "source_infix": " stores the exact value",
            "source_suffix": ".\n",
            "query_prefix": "\nQuestion: Return the value stored by Record",
            "query_suffix": ".\n",
        }
    elif split == "eval":
        literals = {
            "instruction": "Inspect the document. Respond only with the complete value for the requested entry.\n\n",
            "source_prefix": "\nEntry",
            "source_infix": " has current contents",
            "source_suffix": ".\n",
            "query_prefix": "\nQuery: What are the current contents of Entry",
            "query_suffix": "?\n",
        }
    else:
        raise ValueError("template split must be 'train' or 'eval'")
    encoded = {name: _encode_literal(tokenizer, value) for name, value in literals.items()}
    user_prefix, assistant_boundary = _chat_boundaries(tokenizer)
    encoded["instruction"] = user_prefix + encoded["instruction"]
    encoded["query_suffix"] = encoded["query_suffix"] + assistant_boundary
    return RetrievalTemplate(**encoded)


def _sample_filler(flat_filler: torch.Tensor, count: int, rng: random.Random) -> Tuple[int, ...]:
    if flat_filler.ndim != 1:
        raise ValueError("flat filler tensor must be one-dimensional")
    count = int(count)
    if count <= 0 or count > flat_filler.numel():
        raise ValueError("requested filler length does not fit the frozen token pool")
    start = rng.randrange(0, flat_filler.numel() - count + 1)
    return tuple(int(value) for value in flat_filler[start : start + count].tolist())


def generate_example(
    *,
    template: RetrievalTemplate,
    nonce_pool: Sequence[int],
    filler: torch.Tensor,
    seq_len: int,
    min_distance: int,
    max_distance: int,
    eos_token_id: int,
    value_tokens: int,
    rng: random.Random,
) -> Tuple[RetrievalExample, Tuple[int, ...]]:
    """Generate one KV/update row plus an independent same-length swap value."""
    task_type = "kv" if rng.random() < 0.75 else "update"
    required_nonce_tokens = 3 + 2 * value_tokens
    if task_type == "update":
        required_nonce_tokens += value_tokens
    sampled = tuple(rng.sample(tuple(nonce_pool), required_nonce_tokens))
    key = sampled[:3]
    value = sampled[3 : 3 + value_tokens]
    swapped = sampled[3 + value_tokens : 3 + 2 * value_tokens]
    target_distance = rng.randint(int(min_distance), int(max_distance))
    pre_source: Tuple[int, ...] = ()
    if task_type == "update":
        old_value = sampled[3 + 2 * value_tokens :]
        pre_source = template.source_prefix + key + template.source_infix + old_value + template.source_suffix
    filler_ids = _sample_filler(filler, int(seq_len), rng)
    return (
        build_retrieval_example(
            template=template,
            key_ids=key,
            value_ids=value,
            filler_ids=filler_ids,
            seq_len=seq_len,
            target_distance=target_distance,
            eos_token_id=eos_token_id,
            task_type=task_type,
            pre_source_ids=pre_source,
        ),
        swapped,
    )


def _build_phase_examples(
    *,
    phase_name: str,
    split: str,
    groups: int,
    template: RetrievalTemplate,
    nonce_pool: Sequence[int],
    filler: torch.Tensor,
    eos_token_id: int,
    value_tokens: int,
    seed: int,
) -> List[RetrievalExample]:
    phase = get_phase(phase_name)
    rng = random.Random(int(seed))
    flat_filler = filler.reshape(-1)
    output: List[RetrievalExample] = []
    count = phase.training_examples if split == "train" else int(groups)
    for index in range(count):
        example, swapped = generate_example(
            template=template,
            nonce_pool=nonce_pool,
            filler=flat_filler,
            seq_len=phase.seq_len,
            min_distance=phase.min_distance,
            max_distance=phase.max_distance,
            eos_token_id=eos_token_id,
            value_tokens=value_tokens,
            rng=rng,
        )
        if split == "train":
            output.append(example)
            continue
        removal_fill = _sample_filler(
            flat_filler,
            example.source_end - example.source_start,
            rng,
        )
        output.extend(
            build_counterfactual_triplet(
                example,
                swapped_value_ids=swapped,
                removal_fill_ids=removal_fill,
                group_id=f"{phase_name}-{index:04d}",
            )
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exact-distance answer-only data for 8B RoPE adaptation")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--filler-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-tokens", type=int, default=12)
    parser.add_argument("--eval-groups", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.value_tokens < 4:
        raise ValueError("value-tokens must be at least 4 to provide a useful answer signal")
    if args.eval_groups <= 0:
        raise ValueError("eval-groups must be positive")
    if args.output_dir.exists():
        raise FileExistsError("output directory must not already exist")
    outputs = [
        args.output_dir / f"{split}_{phase}.pt"
        for phase in ("warmup", "transition", "exact_8k", "exact_16k")
        for split in ("train", "eval")
    ] + [args.output_dir / "manifest.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("output directory already contains a curriculum artifact")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).expanduser().is_dir(),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    train_filler, validation_filler, filler_manifest = load_frozen_filler(args.filler_dir)
    tokenizer_identity = tokenizer_source_fingerprint(args.tokenizer)
    if filler_manifest.get("tokenizer") != tokenizer_identity:
        raise RuntimeError("runtime tokenizer does not match the frozen filler manifest")

    templates = {
        "train": build_template(tokenizer, "train"),
        "eval": build_template(tokenizer, "eval"),
    }
    nonce_pools = {
        "train": build_nonce_pool(tokenizer, "train"),
        "eval": build_nonce_pool(tokenizer, "eval"),
    }
    source_protocol = {
        "filler_manifest": "manifest.json",
        "filler_manifest_sha256": sha256_file(args.filler_dir / "manifest.json"),
        "tokenizer": tokenizer_identity,
        "nonce_policy": "disjoint_single_token_word_halves",
        "task_mix": {"kv": 0.75, "update": 0.25},
        "value_tokens": int(args.value_tokens),
        "answer_only": True,
        "query_position": "end",
    }

    records = {}
    phases = ("warmup", "transition", "exact_8k", "exact_16k")
    for phase_index, phase_name in enumerate(phases):
        for split, filler in (("train", train_filler), ("eval", validation_filler)):
            examples = _build_phase_examples(
                phase_name=phase_name,
                split=split,
                groups=args.eval_groups,
                template=templates[split],
                nonce_pool=nonce_pools[split],
                filler=filler,
                eos_token_id=int(tokenizer.eos_token_id),
                value_tokens=args.value_tokens,
                seed=args.seed + phase_index * 1000 + (0 if split == "train" else 500),
            )
            bundle = stack_examples(
                examples,
                phase=phase_name,
                split=split,
                seed=args.seed,
                protocol=source_protocol,
            )
            path = args.output_dir / f"{split}_{phase_name}.pt"
            _atomic_torch_save(bundle, path)
            records[path.name] = {
                "sha256": sha256_file(path),
                "rows": int(bundle["input_ids"].shape[0]),
                "seq_len": int(bundle["input_ids"].shape[1]),
                "tokens": int(bundle["input_ids"].numel()),
            }

    manifest = {
        "format_version": 1,
        "purpose": "llama8b_rope_frequency_adaptation",
        "status": "prepared_no_results",
        "seed": int(args.seed),
        "protocol": source_protocol,
        "protocol_sha256": _sha256_json(source_protocol),
        "files": records,
    }
    _atomic_json_dump(manifest, args.output_dir / "manifest.json")
    print(f"prepared {len(records)} matched curriculum bundles in {args.output_dir.name}; no model result was produced")


if __name__ == "__main__":
    main()
