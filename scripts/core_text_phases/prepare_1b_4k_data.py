#!/usr/bin/env python3
"""Prepare 1B tokens of diverse data at seq_len=4096 for 350M MLA experiments.

Uses HF mirror (hf-mirror.com) for China-accessible downloads.

Sources (mixed):
  - FineWeb-Edu (education/wiki) ~400M tokens
  - Code (StarCoder via mirror) ~300M tokens
  - General web (SlimPajama/OpenWebText) ~300M tokens

Runs on CPU only. Does not require GPU.
"""
import os
# Set HF mirror BEFORE any HF imports
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import time, sys
import torch
import numpy as np
from pathlib import Path

SEQ_LEN = 4096
TARGET_TOKENS = 1_000_000_000  # 1B
VAL_TOKENS = 10_000_000  # 10M val
OUT_DIR = Path("/root/autodl-tmp/data/1b_diverse_4k")

SOURCES = [
    {
        "name": "fineweb-edu",
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "text_key": "text",
        "target_tokens": 400_000_000,
    },
    {
        "name": "starcoderdata",
        "hf_name": "bigcode/starcoderdata",
        "config": None,
        "text_key": "content",
        "target_tokens": 300_000_000,
        "fallback": {
            "name": "code_search_net",
            "hf_name": "code_search_net",
            "config": "all",
            "text_key": "whole_func_string",
        },
    },
    {
        "name": "slimpajama",
        "hf_name": "cerebras/SlimPajama-627B",
        "config": None,
        "text_key": "text",
        "target_tokens": 300_000_000,
        "fallback": {
            "name": "openwebtext",
            "hf_name": "Skylion007/openwebtext",
            "config": None,
            "text_key": "text",
        },
    },
]


def stream_tokenize(hf_name, config, text_key, tokenizer, max_tokens, name=""):
    """Stream-tokenize from HF dataset via mirror. Returns list of token IDs."""
    from datasets import load_dataset

    print(f"  [{name}] Loading {hf_name} (config={config}) via hf-mirror...")
    try:
        kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
        if config:
            ds = load_dataset(hf_name, name=config, **kwargs)
        else:
            ds = load_dataset(hf_name, **kwargs)
        ds = ds.shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"  [{name}] FAILED to load: {e}")
        return None

    all_ids = []
    total = 0
    t0 = time.time()
    n_docs = 0

    try:
        for i, example in enumerate(ds):
            text = example.get(text_key, "")
            if not text or len(text) < 50:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            total = len(all_ids)
            n_docs += 1

            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed / 1e6
                print(f"    [{name}] {total/1e6:.1f}M / {max_tokens/1e6:.0f}M tokens "
                      f"({n_docs} docs, {rate:.2f}M tok/s, {elapsed:.0f}s)")

            if total >= max_tokens:
                break
    except Exception as e:
        print(f"  [{name}] Stream error after {total/1e6:.1f}M tokens: {e}")
        if total < max_tokens * 0.1:
            return None

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {total/1e6:.1f}M tokens from {n_docs} docs in {elapsed:.0f}s")
    return all_ids[:max_tokens]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer from local cache
    from transformers import AutoTokenizer
    cache_dir = "/root/.cache/huggingface/hub/models--gpt2"
    print("Loading GPT-2 tokenizer (from cache)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Collect tokens from each source
    all_train_ids = []
    collected = 0
    remaining = TARGET_TOKENS
    source_stats = {}

    for source in SOURCES:
        if remaining <= 0:
            break
        target = min(source["target_tokens"], remaining)
        print(f"\n{'='*60}")
        print(f"  Source: {source['name']} (target: {target/1e6:.0f}M)")
        print(f"{'='*60}")

        ids = stream_tokenize(
            source["hf_name"], source["config"], source["text_key"],
            tokenizer, target, name=source["name"]
        )

        # Try fallback if primary fails
        if (ids is None or len(ids) < target * 0.1) and "fallback" in source:
            fb = source["fallback"]
            print(f"  Primary failed, trying fallback: {fb['name']}...")
            ids = stream_tokenize(
                fb["hf_name"], fb["config"], fb["text_key"],
                tokenizer, target, name=fb["name"]
            )

        if ids is None or len(ids) < 1000:
            print(f"  WARNING: Skipping {source['name']}")
            continue

        got = len(ids)
        all_train_ids.extend(ids)
        collected += got
        remaining -= got
        source_stats[source["name"]] = got
        print(f"  Running total: {collected/1e6:.0f}M / {TARGET_TOKENS/1e6:.0f}M tokens")

    print(f"\n{'='*60}")
    print(f"  Collection complete: {collected/1e6:.1f}M tokens")
    for name, count in source_stats.items():
        print(f"    {name}: {count/1e6:.0f}M ({100*count/collected:.0f}%)")
    print(f"{'='*60}")

    if collected < TARGET_TOKENS * 0.5:
        print(f"ERROR: Only got {collected/1e6:.0f}M tokens, need at least 500M. Aborting.")
        sys.exit(1)

    # Save train data (flat tensor)
    actual_tokens = (collected // SEQ_LEN) * SEQ_LEN  # trim to exact chunks
    flat = torch.tensor(all_train_ids[:actual_tokens], dtype=torch.int32)
    train_path = OUT_DIR / f"train_diverse_{actual_tokens}_{SEQ_LEN}.pt"
    torch.save(flat, train_path)
    print(f"\n  Train: {flat.numel()/1e6:.1f}M tokens ({flat.numel()//SEQ_LEN} chunks) -> {train_path}")

    # Create standard symlink for training script compatibility
    link_train = OUT_DIR / f"train_fineweb-edu_{TARGET_TOKENS}_{SEQ_LEN}.pt"
    if link_train.exists() or link_train.is_symlink():
        link_train.unlink()
    link_train.symlink_to(train_path.name)
    print(f"  Symlink: {link_train.name} -> {train_path.name}")

    # Validation data (from first available source)
    print(f"\n--- Validation data ({VAL_TOKENS/1e6:.0f}M tokens) ---")
    val_ids = stream_tokenize(
        SOURCES[0]["hf_name"], SOURCES[0]["config"], SOURCES[0]["text_key"],
        tokenizer, VAL_TOKENS, name="val"
    )
    if val_ids is None:
        # Use beginning of train data as val (different from training portion)
        print("  Using tail of collected data as validation...")
        val_ids = all_train_ids[-VAL_TOKENS:]

    val_flat = torch.tensor(val_ids[:VAL_TOKENS], dtype=torch.int32)
    val_path = OUT_DIR / f"val_fineweb-edu_{VAL_TOKENS}.pt"
    torch.save(val_flat, val_path)
    print(f"  Val: {val_flat.numel()/1e6:.1f}M tokens -> {val_path}")

    print(f"\n{'='*60}")
    print(f"  DATA PREPARATION COMPLETE")
    print(f"  Train: {flat.numel()/1e6:.0f}M tokens @ seq_len={SEQ_LEN}")
    print(f"  Val: {val_flat.numel()/1e6:.0f}M tokens")
    print(f"  Output: {OUT_DIR}")
    print(f"  Sources: {list(source_stats.keys())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
