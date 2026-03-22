#!/usr/bin/env python3
"""Prepare 2nd batch of 1B tokens at seq_len=4096 for continued pretraining.

ALL sources are completely different from phase1 (pile + openwebtext).
Focus on long-form, high-quality text suitable for 4096+ context:
  - PG19 books (~200M) — full novels, naturally 10K-100K+ tokens each
  - arXiv papers (~300M) — scientific full-text, 5-10K tokens each
  - proof-pile-2 (~200M) — math, code, scientific literature
  - GitHub code clean (~300M) — long code files

Uses hf-mirror.com. CPU only. Does not affect GPU training.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time, sys
import torch
from pathlib import Path

SEQ_LEN = 4096
TARGET_TOKENS = 1_000_000_000
VAL_TOKENS = 5_000_000
OUT_DIR = Path("/root/autodl-tmp/data/1b_diverse_4k_phase2")

SOURCES = [
    {
        "name": "pg19-books",
        "hf_name": "emozilla/pg19",
        "config": None,
        "text_key": "text",
        "target_tokens": 200_000_000,
    },
    {
        "name": "arxiv-papers",
        "hf_name": "scientific_papers",
        "config": "arxiv",
        "text_key": "article",
        "target_tokens": 300_000_000,
    },
    {
        "name": "proof-pile-2",
        "hf_name": "EleutherAI/proof-pile-2",
        "config": "algebraic-stack",
        "text_key": "text",
        "target_tokens": 200_000_000,
    },
    {
        "name": "github-code-clean",
        "hf_name": "codeparrot/github-code-clean",
        "config": None,
        "text_key": "code",
        "target_tokens": 300_000_000,
    },
]

# If a source fails, try these
FALLBACKS = [
    {
        "name": "fineweb-base",
        "hf_name": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "text_key": "text",
    },
]


def stream_tokenize(hf_name, config, text_key, tokenizer, max_tokens, name=""):
    from datasets import load_dataset
    print(f"  [{name}] Loading {hf_name} (config={config}) via hf-mirror...")
    sys.stdout.flush()
    try:
        kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}
        if config:
            ds = load_dataset(hf_name, name=config, **kwargs)
        else:
            ds = load_dataset(hf_name, **kwargs)
        ds = ds.shuffle(seed=42, buffer_size=5000)
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        sys.stdout.flush()
        return None

    all_ids = []
    total = 0
    t0 = time.time()
    n_docs = 0

    try:
        for example in ds:
            text = example.get(text_key, "")
            if not text or len(text) < 100:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            total = len(all_ids)
            n_docs += 1
            if n_docs % 5000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed / 1e6 if elapsed > 0 else 0
                print(f"    [{name}] {total/1e6:.1f}M / {max_tokens/1e6:.0f}M tokens "
                      f"({n_docs} docs, {rate:.2f}M tok/s, {elapsed:.0f}s)")
                sys.stdout.flush()
            if total >= max_tokens:
                break
    except Exception as e:
        print(f"  [{name}] Error after {total/1e6:.1f}M tokens: {e}")
        sys.stdout.flush()
        if total < max_tokens * 0.1:
            return None

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {total/1e6:.1f}M tokens, {n_docs} docs, {elapsed:.0f}s")
    sys.stdout.flush()
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
        sys.stdout.flush()

        ids = stream_tokenize(
            source["hf_name"], source["config"], source["text_key"],
            tokenizer, target, name=source["name"]
        )

        if ids is None or len(ids) < target * 0.1:
            print(f"  Primary failed, trying fallbacks...")
            for fb in FALLBACKS:
                ids = stream_tokenize(
                    fb["hf_name"], fb["config"], fb["text_key"],
                    tokenizer, target, name=fb["name"]
                )
                if ids and len(ids) >= target * 0.3:
                    source["name"] = fb["name"]
                    break

        if ids is None or len(ids) < 1000:
            print(f"  WARNING: Skipping {source['name']}")
            continue

        got = len(ids)
        all_train_ids.extend(ids)
        collected += got
        remaining -= got
        source_stats[source["name"]] = got
        print(f"  Total: {collected/1e6:.0f}M / {TARGET_TOKENS/1e6:.0f}M")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  Collected {collected/1e6:.1f}M tokens (phase2)")
    for n, c in source_stats.items():
        pct = 100 * c / collected if collected > 0 else 0
        print(f"    {n}: {c/1e6:.0f}M ({pct:.0f}%)")
    print(f"{'='*60}")

    if collected < TARGET_TOKENS * 0.5:
        print(f"ERROR: Only {collected/1e6:.0f}M. Aborting.")
        sys.exit(1)

    # Save as 2D int64 (n_chunks, seq_len) — training script format
    actual = (collected // SEQ_LEN) * SEQ_LEN
    flat = torch.tensor(all_train_ids[:actual], dtype=torch.int64).view(-1, SEQ_LEN)
    train_path = OUT_DIR / f"train_phase2_{actual}_{SEQ_LEN}.pt"
    torch.save(flat, train_path)
    print(f"\n  Train: {flat.numel()/1e6:.1f}M tokens ({flat.shape[0]} chunks) -> {train_path}")

    # Symlink for training script compatibility
    link = OUT_DIR / f"train_fineweb-edu_{TARGET_TOKENS}_{SEQ_LEN}.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(train_path.name)

    # Val from tail (non-overlapping with train)
    val_ids = all_train_ids[actual:actual + VAL_TOKENS]
    if len(val_ids) < VAL_TOKENS:
        val_ids = all_train_ids[-VAL_TOKENS:]
    val_flat = torch.tensor(val_ids[:VAL_TOKENS], dtype=torch.int64)
    val_path = OUT_DIR / f"val_fineweb-edu_{VAL_TOKENS}.pt"
    torch.save(val_flat, val_path)
    print(f"  Val: {val_flat.numel()/1e6:.1f}M tokens -> {val_path}")

    print(f"\n  DONE: {flat.numel()/1e6:.0f}M train + {val_flat.numel()/1e6:.0f}M val (phase2)")
    print(f"  Sources: {list(source_stats.keys())}")


if __name__ == "__main__":
    main()
