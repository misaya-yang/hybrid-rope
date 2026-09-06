#!/usr/bin/env python3
"""Prepare 1B tokens of diverse data at seq_len=4096 — v3, non-overlapping with v1 and v2.

Sources: same as v1/v2 (Pile-uncopyrighted + OpenWebText)
  v1: seed=42,   buffer=10000
  v2: seed=2024, buffer=100000
  v3: seed=3000, buffer=200000  ← maximum shuffle divergence

With 100B+ token corpus and only 1% sampled, different seeds + large buffers
give effectively zero content overlap across all three versions.

Purpose: Combined v1+v2+v3 = 3B tokens, enough to train 125M model to
Chinchilla-optimal (~2.5B) and beyond, to find the GEO vs EVQ inflection point.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time, sys
import torch
from pathlib import Path

SEQ_LEN = 4096
TARGET_TOKENS = 1_000_000_000
VAL_TOKENS = 5_000_000
OUT_DIR = Path("/root/autodl-tmp/data/1b_diverse_4k_v3")
SHUFFLE_SEED = 3000

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
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=200000)
        print(f"  [{name}] shuffle seed={SHUFFLE_SEED}, buffer=200000")
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
                eta = (max_tokens - total) / (total / elapsed) if total > 0 else 0
                print(f"    [{name}] {total/1e6:.1f}M / {max_tokens/1e6:.0f}M tokens "
                      f"({n_docs} docs, {rate:.2f}M tok/s, ETA {eta:.0f}s)")
            if total >= max_tokens:
                break
    except Exception as e:
        print(f"  [{name}] Error after {total/1e6:.1f}M tokens: {e}")
        if total < max_tokens * 0.5:
            return None

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {total/1e6:.1f}M tokens, {n_docs} docs in {elapsed:.0f}s")
    return all_ids[:max_tokens]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR}")
    print(f"Target: {TARGET_TOKENS/1e9:.1f}B tokens, seq_len={SEQ_LEN}")
    print(f"Sources: Pile-uncopyrighted(600M) + OpenWebText(400M), seed={SHUFFLE_SEED}, buffer=200000")
    print(f"vs v1: seed=42, buf=10000 | vs v2: seed=2024, buf=100000 — all non-overlapping")

    from transformers import AutoTokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    all_train_ids = []
    collected = 0
    source_stats = {}

    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"  Source: {source['name']} (target: {source['target_tokens']/1e6:.0f}M tokens)")
        print(f"{'='*60}")

        ids = stream_tokenize(
            source["hf_name"], source["config"], source["text_key"],
            tokenizer, source["target_tokens"], name=source["name"]
        )

        if ids is None or len(ids) < source["target_tokens"] * 0.5:
            print(f"WARNING: {source['name']} only got {len(ids) if ids else 0} tokens, skipping")
            continue

        got = len(ids)
        all_train_ids.extend(ids)
        collected += got
        source_stats[source["name"]] = got
        print(f"  Subtotal: {collected/1e6:.0f}M / {TARGET_TOKENS/1e6:.0f}M")

    ids = all_train_ids

    if collected < TARGET_TOKENS * 0.9:
        print(f"ERROR: Only got {collected/1e6:.0f}M tokens. Aborting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Collected {collected/1e6:.1f}M tokens")
    for n, c in source_stats.items():
        print(f"    {n}: {c/1e6:.0f}M ({100*c/collected:.0f}%)")

    actual = (collected // SEQ_LEN) * SEQ_LEN
    flat = torch.tensor(ids[:actual], dtype=torch.int32)
    n_chunks = flat.numel() // SEQ_LEN
    train_data = flat.view(n_chunks, SEQ_LEN)
    train_path = OUT_DIR / f"train_diverse_{actual}_{SEQ_LEN}.pt"
    torch.save(train_data, train_path)
    print(f"  Train: {train_data.shape} ({train_data.numel()/1e6:.1f}M tokens) -> {train_path}")

    link = OUT_DIR / f"train_fineweb-edu_{TARGET_TOKENS}_{SEQ_LEN}.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(train_path.name)
    print(f"  Symlink: {link.name} -> {train_path.name}")

    val_ids = ids[actual:actual + VAL_TOKENS]
    if len(val_ids) < VAL_TOKENS:
        val_ids = ids[-VAL_TOKENS:]
    val_flat = torch.tensor(val_ids[:VAL_TOKENS], dtype=torch.int32)
    val_path = OUT_DIR / f"val_fineweb-edu_{VAL_TOKENS}.pt"
    torch.save(val_flat, val_path)
    print(f"  Val: {val_flat.numel()/1e6:.1f}M tokens -> {val_path}")

    print(f"\n  DONE. {actual/1e9:.3f}B train tokens saved to {OUT_DIR}")
    print(f"  Total available: v1(1B) + v2(1B) + v3(1B) = 3B tokens")
    print(f"  Chinchilla-optimal for 125M: ~2.5B tokens — covered!")


if __name__ == "__main__":
    main()
