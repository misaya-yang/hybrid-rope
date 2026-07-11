#!/usr/bin/env python3
"""Freeze the recovered LongAlpaca paper-lineage data without hiding provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

import numpy as np

try:
    from .legacy_lora_protocol import (
        LEGACY_OBJECTIVE,
        PAPER_LONGALPACA_PROVENANCE_STATUS,
        PAPER_LONGALPACA_REVISION,
        PAPER_LONGALPACA_SOURCE,
        sha256_file,
        validate_training_source_receipt,
    )
    from .prepare_legacy_longalign_data import (
        _atomic_json_dump,
        _atomic_torch_save,
        iter_jsonl,
        normalize_legacy_messages,
        split_legacy_tokenized,
        tokenize_legacy_rows,
    )
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
except ImportError:
    from legacy_lora_protocol import (
        LEGACY_OBJECTIVE,
        PAPER_LONGALPACA_PROVENANCE_STATUS,
        PAPER_LONGALPACA_REVISION,
        PAPER_LONGALPACA_SOURCE,
        sha256_file,
        validate_training_source_receipt,
    )
    from prepare_legacy_longalign_data import (
        _atomic_json_dump,
        _atomic_torch_save,
        iter_jsonl,
        normalize_legacy_messages,
        split_legacy_tokenized,
        tokenize_legacy_rows,
    )
    from prepare_positional_distill_data import tokenizer_source_fingerprint


def iter_longalpaca_json_array(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("ijson is required for bounded-memory LongAlpaca parsing") from exc
    with Path(path).open("rb") as handle:
        for item in ijson.items(handle, "item"):
            if not isinstance(item, dict):
                raise ValueError("LongAlpaca array entries must be JSON objects")
            yield item


def convert_longalpaca_records_to_legacy_jsonl(
    rows: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> Dict[str, int]:
    source_rows_seen = 0
    converted_rows = 0
    unsupported_rows = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in rows:
            source_rows_seen += 1
            messages = normalize_legacy_messages(item)
            if messages is None:
                unsupported_rows += 1
                continue
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            converted_rows += 1
    os.replace(temporary, output_path)
    return {
        "source_rows_seen": source_rows_seen,
        "converted_rows": converted_rows,
        "unsupported_rows": unsupported_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_json", type=Path, required=True)
    parser.add_argument("--expected_raw_sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=8000)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_json.is_file():
        raise FileNotFoundError(args.raw_json)
    actual_raw_sha256 = sha256_file(args.raw_json)
    if actual_raw_sha256 != args.expected_raw_sha256:
        raise ValueError(
            f"LongAlpaca raw SHA-256 mismatch: expected {args.expected_raw_sha256}, "
            f"found {actual_raw_sha256}"
        )
    source = validate_training_source_receipt({
        "source_id": PAPER_LONGALPACA_SOURCE,
        "revision": PAPER_LONGALPACA_REVISION,
        "split": "train",
        "filename": args.raw_json.name,
        "raw_sha256": actual_raw_sha256,
        "provenance_status": PAPER_LONGALPACA_PROVENANCE_STATUS,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "normalized_jsonl": args.output_dir / "longalpaca_12k_legacy.jsonl",
        "tokens": args.output_dir / "tokens.pt",
        "offsets": args.output_dir / "offsets.pt",
        "train_indices": args.output_dir / "train_indices.pt",
        "validation_indices": args.output_dir / "validation_indices.pt",
        "manifest": args.output_dir / "manifest.json",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("frozen LongAlpaca data exists; pass --overwrite to replace it")

    conversion = convert_longalpaca_records_to_legacy_jsonl(
        iter_longalpaca_json_array(args.raw_json),
        paths["normalized_jsonl"],
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    tokenized, stats = tokenize_legacy_rows(
        iter_jsonl(paths["normalized_jsonl"]),
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
    _atomic_torch_save(tokenized["tokens"], paths["tokens"])
    _atomic_torch_save(tokenized["offsets"], paths["offsets"])
    _atomic_torch_save(split["train_indices"], paths["train_indices"])
    _atomic_torch_save(split["validation_indices"], paths["validation_indices"])
    lengths = (tokenized["offsets"][1:] - tokenized["offsets"][:-1]).numpy()
    manifest = {
        "format_version": 1,
        "objective": LEGACY_OBJECTIVE,
        "source": source,
        "normalized_source": {
            "name": paths["normalized_jsonl"].name,
            "sha256": sha256_file(paths["normalized_jsonl"]),
            **conversion,
        },
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
            key: {"name": paths[key].name, "sha256": sha256_file(paths[key])}
            for key in ("tokens", "offsets", "train_indices", "validation_indices")
        },
    }
    _atomic_json_dump(manifest, paths["manifest"])
    print(json.dumps({
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "normalized_jsonl_sha256": manifest["normalized_source"]["sha256"],
        "train_rows": manifest["statistics"]["train_rows"],
        "validation_rows": manifest["statistics"]["validation_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
