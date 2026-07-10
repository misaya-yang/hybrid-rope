#!/usr/bin/env python3
"""Freeze plain-text data for the LLaMA-3-8B positional-distillation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_identifier(value: str) -> str:
    value = str(value).rstrip("/")
    if os.path.isabs(value):
        return Path(value).name
    return value


def iter_plain_text_jsonl(path: Path) -> Iterator[str]:
    """Yield plain text and fail closed on chat/instruction records."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"{path.name}:{line_number} is not a plain text record; "
                    "each row must contain a text field"
                )
            text = text.strip()
            if text:
                yield text


def pack_token_sequences(
    token_documents: Iterable[Sequence[int]],
    seq_len: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Pack tokenized documents into fixed sequences and return the remainder."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    pending: List[int] = []
    rows: List[List[int]] = []
    for document in token_documents:
        pending.extend(int(token) for token in document)
        while len(pending) >= seq_len:
            rows.append(pending[:seq_len])
            del pending[:seq_len]
    if rows:
        packed = torch.tensor(rows, dtype=torch.int32)
    else:
        packed = torch.empty((0, seq_len), dtype=torch.int32)
    return packed, pending


def iter_huggingface_text(
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if dataset_config:
        kwargs["name"] = dataset_config
    dataset = load_dataset(**kwargs)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in dataset:
        text = row.get(text_field) if isinstance(row, dict) else None
        if isinstance(text, str) and text.strip():
            yield text.strip()


def collect_fixed_sequences(
    tokenizer,
    texts: Iterable[str],
    seq_len: int,
    total_sequences: int,
) -> Tuple[torch.Tensor, int]:
    pending: List[int] = []
    rows: List[List[int]] = []
    documents_seen = 0
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    for text in texts:
        documents_seen += 1
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        pending.extend(int(token) for token in token_ids)
        pending.append(int(eos_token_id))
        while len(pending) >= seq_len and len(rows) < total_sequences:
            rows.append(pending[:seq_len])
            del pending[:seq_len]
        if len(rows) == total_sequences:
            break

    if len(rows) != total_sequences:
        raise RuntimeError(
            f"source ended after {len(rows)} packed sequences; "
            f"required {total_sequences}"
        )
    return torch.tensor(rows, dtype=torch.int32), documents_seen


def atomic_torch_save(value, path: Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value: dict, path: Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--local_jsonl", type=Path, default=None)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--train_sequences", type=int, default=2400)
    parser.add_argument("--validation_sequences", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.pt"
    validation_path = args.output_dir / "validation.pt"
    manifest_path = args.output_dir / "manifest.json"
    outputs = (train_path, validation_path, manifest_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(
            "frozen distillation data already exists; pass --overwrite to replace it"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
    )
    if args.local_jsonl is not None:
        texts = iter_plain_text_jsonl(args.local_jsonl)
        source = {
            "kind": "local_plain_text_jsonl",
            "name": args.local_jsonl.name,
            "sha256": sha256_file(args.local_jsonl),
        }
    else:
        texts = iter_huggingface_text(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            text_field=args.text_field,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        source = {
            "kind": "huggingface_streaming",
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
            "shuffle_buffer": args.shuffle_buffer,
        }

    total_sequences = args.validation_sequences + args.train_sequences
    packed, documents_seen = collect_fixed_sequences(
        tokenizer=tokenizer,
        texts=texts,
        seq_len=args.seq_len,
        total_sequences=total_sequences,
    )
    validation = packed[: args.validation_sequences].contiguous()
    train = packed[args.validation_sequences :].contiguous()
    atomic_torch_save(train, train_path)
    atomic_torch_save(validation, validation_path)

    manifest = {
        "format_version": 1,
        "purpose": "llama8b_positional_hidden_distillation",
        "source": source,
        "tokenizer": public_identifier(args.tokenizer),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "train_sequences": int(train.shape[0]),
        "validation_sequences": int(validation.shape[0]),
        "train_tokens": int(train.numel()),
        "validation_tokens": int(validation.numel()),
        "documents_seen": documents_seen,
        "files": {
            "train": {
                "name": train_path.name,
                "sha256": sha256_file(train_path),
            },
            "validation": {
                "name": validation_path.name,
                "sha256": sha256_file(validation_path),
            },
        },
    }
    atomic_json_dump(manifest, manifest_path)
    print(
        f"frozen {manifest['train_tokens']:,} train tokens and "
        f"{manifest['validation_tokens']:,} validation tokens in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
