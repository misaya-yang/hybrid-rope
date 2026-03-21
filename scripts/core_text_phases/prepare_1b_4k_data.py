#!/usr/bin/env python3
"""Prepare 1B tokens of diverse data at seq_len=4096.

Sources:
  - monology/pile-uncopyrighted ~600M (code, arxiv, books, web mixed)
  - Skylion007/openwebtext ~400M (general web)

NO fineweb-edu. Uses hf-mirror.com. CPU only.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time, sys
import torch
from pathlib import Path

SEQ_LEN = 4096
TARGET_TOKENS = 1_000_000_000
VAL_TOKENS = 5_000_000
OUT_DIR = Path("/root/autodl-tmp/data/1b_diverse_4k")

SOURCES = [
    {
        "name": "pile-uncopyrighted",
        "hf_name": "monology/pile-uncopyrighted",
        "config": None,
        "text_key": "text",
        "target_tokens": 600_000_000,
    },
    {
        "name": "openwebtext",
        "hf_name": "Skylion007/openwebtext",
        "config": None,
        "text_key": "text",
        "target_tokens": 400_000_000,
    },
]


def stream_tokenize(hf_name, config, text_key, tokenizer, max_tokens, name=""):
    from datasets import load_dataset
    print(f"  [{name}] Loading {hf_name} via hf-mirror...")
    try:
        kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
        if config:
            ds = load_dataset(hf_name, name=config, **kwargs)
        else:
            ds = load_dataset(hf_name, **kwargs)
        ds = ds.shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        return None

    all_ids = []
    total = 0
    t0 = time.time()
    n_docs = 0

    try:
        for example in ds:
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
        print(f"  [{name}] Error after {total/1e6:.1f}M tokens: {e}")
        if total < max_tokens * 0.1:
            return None

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {total/1e6:.1f}M tokens, {n_docs} docs, {elapsed:.0f}s")
    return all_ids[:max_tokens]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Vocab size: {tokenizer.vocab_size}")

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

        if ids is None or len(ids) < 1000:
            print(f"  WARNING: Skipping {source['name']}")
            continue

        got = len(ids)
        all_train_ids.extend(ids)
        collected += got
        remaining -= got
        source_stats[source["name"]] = got
        print(f"  Total: {collected/1e6:.0f}M / {TARGET_TOKENS/1e6:.0f}M")

    print(f"\n{'='*60}")
    print(f"  Collected {collected/1e6:.1f}M tokens")
    for n, c in source_stats.items():
        print(f"    {n}: {c/1e6:.0f}M ({100*c/collected:.0f}%)")
    print(f"{'='*60}")

    if collected < TARGET_TOKENS * 0.5:
        print(f"ERROR: Only {collected/1e6:.0f}M. Aborting.")
        sys.exit(1)

    # Save train
    actual = (collected // SEQ_LEN) * SEQ_LEN
    flat = torch.tensor(all_train_ids[:actual], dtype=torch.int32)
    train_path = OUT_DIR / f"train_diverse_{actual}_{SEQ_LEN}.pt"
    torch.save(flat, train_path)
    print(f"\n  Train: {flat.numel()/1e6:.1f}M tokens -> {train_path}")

    link = OUT_DIR / f"train_fineweb-edu_{TARGET_TOKENS}_{SEQ_LEN}.pt"
    if link.exists() or link.is_symlink(): link.unlink()
    link.symlink_to(train_path.name)

    # Val from tail (non-overlapping)
    val_ids = all_train_ids[actual:actual + VAL_TOKENS]
    if len(val_ids) < VAL_TOKENS:
        val_ids = all_train_ids[-VAL_TOKENS:]
    val_flat = torch.tensor(val_ids[:VAL_TOKENS], dtype=torch.int32)
    val_path = OUT_DIR / f"val_fineweb-edu_{VAL_TOKENS}.pt"
    torch.save(val_flat, val_path)
    print(f"  Val: {val_flat.numel()/1e6:.1f}M tokens -> {val_path}")

    print(f"\n  DONE: {flat.numel()/1e6:.0f}M train + {val_flat.numel()/1e6:.0f}M val")


if __name__ == "__main__":
    main()
