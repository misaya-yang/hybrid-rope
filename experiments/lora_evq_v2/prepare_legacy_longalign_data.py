#!/usr/bin/env python3
"""Freeze the verified LongAlign-10k artifact for the legacy LoRA rerun.

There is intentionally no dataset download fallback here.  The caller must
provide a local JSONL and an explicit revision/hash receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch

try:
    from .legacy_lora_protocol import (
        OFFICIAL_LONGALIGN_SOURCE,
        OFFICIAL_LONGALIGN_REVISION,
        sha256_file,
        validate_source_receipt,
    )
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
except ImportError:  # direct script execution
    from legacy_lora_protocol import (
        OFFICIAL_LONGALIGN_SOURCE,
        OFFICIAL_LONGALIGN_REVISION,
        sha256_file,
        validate_source_receipt,
    )
    from prepare_positional_distill_data import tokenizer_source_fingerprint


LONGALIGN_REVISION = OFFICIAL_LONGALIGN_REVISION


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{Path(path).name}:{line_number} is not a JSON object")
            yield value


def normalize_legacy_messages(item: Mapping[str, Any]) -> Optional[List[Dict[str, str]]]:
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                return None
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if not isinstance(content, str):
                return None
            normalized.append({"role": role, "content": content})
        return normalized
    if "instruction" in item:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        if not all(isinstance(value, str) for value in (instruction, input_text, output)):
            return None
        user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ]
    if "question" in item:
        context = item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer", item.get("answers", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not all(isinstance(value, str) for value in (context, question, answer)):
            return None
        user_text = question if not context else f"{context}\n\n{question}"
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    return None


def tokenize_legacy_rows(
    rows: Iterable[Mapping[str, Any]],
    tokenizer,
    *,
    max_samples: int = 8000,
    max_seq_len: int = 8192,
    min_tokens: int = 64,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Preserve the historical select-then-tokenize behavior exactly."""
    if max_samples <= 0 or max_seq_len <= 0 or min_tokens <= 0:
        raise ValueError("sample, sequence, and minimum-token limits must be positive")
    unsupported = 0
    source_rows_seen = 0
    accepted = 0
    token_tensors: List[torch.Tensor] = []
    offsets = [0]
    too_short = 0
    fallback_templates = 0
    for item in rows:
        source_rows_seen += 1
        messages = normalize_legacy_messages(item)
        if messages is None:
            unsupported += 1
            continue
        accepted += 1
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            fallback_templates += 1
            parts = [
                f"<|start_header_id|>{message.get('role', 'user')}<|end_header_id|>\n\n"
                f"{message.get('content', '')}<|eot_id|>"
                for message in messages
            ]
            text = "<|begin_of_text|>" + "".join(parts)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        attention_mask = [int(value) for value in encoded["attention_mask"]]
        if len(input_ids) < min_tokens:
            too_short += 1
        else:
            if len(attention_mask) != len(input_ids) or any(value != 1 for value in attention_mask):
                raise ValueError("unpadded legacy tokenization must produce an all-ones attention mask")
            row_tensor = torch.tensor(input_ids, dtype=torch.int32)
            token_tensors.append(row_tensor)
            offsets.append(offsets[-1] + row_tensor.numel())
        if accepted >= max_samples:
            break
    if not token_tensors:
        raise ValueError("no legacy rows survived tokenization")
    compact = {
        "tokens": torch.cat(token_tensors),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
    }
    return compact, {
        "source_rows_seen": source_rows_seen,
        "accepted_source_rows": accepted,
        "unsupported_source_rows": unsupported,
        "tokenized_rows": len(token_tensors),
        "too_short_rows": too_short,
        "fallback_chat_templates": fallback_templates,
    }


def split_legacy_tokenized(
    tokenized: Dict[str, torch.Tensor],
    *,
    val_ratio: float = 0.02,
    split_seed: int = 42,
) -> Dict[str, torch.Tensor]:
    offsets = tokenized.get("offsets")
    if offsets is None or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("cannot split an empty tokenized dataset")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie strictly between zero and one")
    row_count = offsets.numel() - 1
    n_val = max(1, int(row_count * val_ratio))
    indices = list(range(row_count))
    random.Random(split_seed).shuffle(indices)
    validation_indices = set(indices[:n_val])
    return {
        "train_indices": torch.tensor(
            [index for index in indices if index not in validation_indices], dtype=torch.int32
        ),
        # Preserve the old set-iteration validation ordering for protocol fidelity.
        "validation_indices": torch.tensor(list(validation_indices), dtype=torch.int32),
    }


def compact_tokenized_row(tokenized: Dict[str, torch.Tensor], index: int) -> List[int]:
    offsets = tokenized["offsets"]
    start = int(offsets[index])
    end = int(offsets[index + 1])
    return tokenized["tokens"][start:end].tolist()


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_jsonl", type=Path, required=True)
    parser.add_argument("--expected_raw_sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_id", default=OFFICIAL_LONGALIGN_SOURCE)
    parser.add_argument("--revision", default=LONGALIGN_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=8000)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_jsonl.is_file():
        raise FileNotFoundError(args.raw_jsonl)
    actual_raw_sha256 = sha256_file(args.raw_jsonl)
    if actual_raw_sha256 != args.expected_raw_sha256:
        raise ValueError(
            "LongAlign raw SHA-256 mismatch: "
            f"expected {args.expected_raw_sha256}, found {actual_raw_sha256}"
        )
    receipt = validate_source_receipt({
        "source_id": args.source_id,
        "revision": args.revision,
        "split": args.split,
        "filename": "long.jsonl",
        "raw_sha256": actual_raw_sha256,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "tokens.pt"
    offsets_path = args.output_dir / "offsets.pt"
    train_path = args.output_dir / "train_indices.pt"
    validation_path = args.output_dir / "validation_indices.pt"
    manifest_path = args.output_dir / "manifest.json"
    if any(path.exists() for path in (token_path, offsets_path, train_path, validation_path, manifest_path)) and not args.overwrite:
        raise FileExistsError("frozen legacy data exists; pass --overwrite to replace it")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    tokenized, stats = tokenize_legacy_rows(
        iter_jsonl(args.raw_jsonl),
        tokenizer,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        min_tokens=args.min_tokens,
    )
    split = split_legacy_tokenized(
        tokenized,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    _atomic_torch_save(tokenized["tokens"], token_path)
    _atomic_torch_save(tokenized["offsets"], offsets_path)
    _atomic_torch_save(split["train_indices"], train_path)
    _atomic_torch_save(split["validation_indices"], validation_path)
    lengths = (tokenized["offsets"][1:] - tokenized["offsets"][:-1]).numpy()
    manifest = {
        "format_version": 1,
        "objective": "legacy_longalign_full_token_causal_lm_v2",
        "source": receipt,
        "tokenizer": tokenizer_source_fingerprint(args.tokenizer),
        "preparation": {
            "max_samples": args.max_samples,
            "max_seq_len": args.max_seq_len,
            "minimum_tokens": args.min_tokens,
            "validation_ratio": args.val_ratio,
            "split_seed": args.split_seed,
            "selection_order": "first_supported_rows_before_tokenization",
            "labels": "all_non_padding_input_tokens",
            "variable_length": True,
        },
        "statistics": {
            **stats,
            "train_rows": split["train_indices"].numel(),
            "validation_rows": split["validation_indices"].numel(),
            "minimum_length": int(np.min(lengths)),
            "maximum_length": int(np.max(lengths)),
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
        },
        "files": {
            "tokens": {"name": token_path.name, "sha256": sha256_file(token_path)},
            "offsets": {"name": offsets_path.name, "sha256": sha256_file(offsets_path)},
            "train_indices": {"name": train_path.name, "sha256": sha256_file(train_path)},
            "validation_indices": {
                "name": validation_path.name,
                "sha256": sha256_file(validation_path),
            },
        },
    }
    _atomic_json_dump(manifest, manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_rows": split["train_indices"].numel(),
        "validation_rows": split["validation_indices"].numel(),
    }, indent=2))


if __name__ == "__main__":
    main()
