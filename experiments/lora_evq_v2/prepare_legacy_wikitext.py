#!/usr/bin/env python3
"""Freeze revision-pinned WikiText-2 raw test tokens for matched PPL eval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    from .legacy_lora_protocol import sha256_file
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
except ImportError:
    from legacy_lora_protocol import sha256_file
    from prepare_positional_distill_data import tokenizer_source_fingerprint


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_SOURCE = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_RAW_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--expected_raw_sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_id", default=WIKITEXT_SOURCE)
    parser.add_argument("--revision", default=WIKITEXT_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_id != WIKITEXT_SOURCE or args.revision != WIKITEXT_REVISION:
        raise ValueError("WikiText evaluator requires the pinned official source/revision")
    if args.expected_raw_sha256 != WIKITEXT_RAW_SHA256:
        raise ValueError("WikiText expected SHA-256 is not the pinned official test parquet")
    if sha256_file(args.parquet) != args.expected_raw_sha256:
        raise ValueError("WikiText parquet SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "tokens.pt"
    manifest_path = args.output_dir / "manifest.json"
    if any(path.exists() for path in (token_path, manifest_path)) and not args.overwrite:
        raise FileExistsError("frozen WikiText evaluation data exists; pass --overwrite")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset("parquet", data_files=str(args.parquet), split="train")
    text = "\n".join(str(value) for value in dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    token_ids = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
    required = 5 * 32768
    if len(token_ids) < required:
        raise ValueError(f"WikiText token stream has {len(token_ids)} tokens; {required} required")
    tensor = torch.tensor(token_ids, dtype=torch.int32)
    temporary = token_path.with_suffix(".pt.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, token_path)
    manifest = {
        "format_version": 1,
        "objective": "legacy_wikitext2_matched_ppl_v1",
        "source": {
            "source_id": WIKITEXT_SOURCE,
            "revision": WIKITEXT_REVISION,
            "config": WIKITEXT_CONFIG,
            "split": "test",
            "filename": args.parquet.name,
            "raw_sha256": args.expected_raw_sha256,
        },
        "tokenizer": tokenizer_source_fingerprint(args.tokenizer),
        "token_count": len(token_ids),
        "lengths": [8192, 16384, 32768],
        "chunks_per_length": 5,
        "offset_rule": "chunk_index_times_context_length",
        "tokens": {"name": token_path.name, "sha256": sha256_file(token_path)},
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "token_count": len(token_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
