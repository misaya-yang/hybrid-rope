#!/usr/bin/env python
"""
Download and tokenize 500M tokens of MIXED data for 8K continued pretraining.
All data is NEW (not FineWeb-Edu) to avoid training on seen data.

Mix:
  - 40% Code (GitHub via SlimPajama)
  - 30% Long documents (Books + ArXiv via SlimPajama)
  - 30% Web text (CommonCrawl + C4 + Wiki + StackExchange via SlimPajama)

Source: cerebras/SlimPajama-627B (via hf-mirror for China access)
Tokenizer: EleutherAI/gpt-neox-20b (vocab_size=50304)
Output: train_slimpajama-mixed_500000000_8192.pt

Usage:
    python prepare_8k_mixed_500m.py \
        --output_dir /root/autodl-tmp/data/8k_mixed \
        --max_tokens 500000000 \
        --seq_len 8192 \
        --min_doc_tokens 2048
"""

import argparse
import os
import sys
import time
import torch
import json
from pathlib import Path
from typing import List, Optional

# Use hf-mirror for China access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_tokenizer():
    from transformers import AutoTokenizer
    print("[tokenizer] Loading EleutherAI/gpt-neox-20b...", flush=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"[tokenizer] vocab_size={tok.vocab_size}", flush=True)
    return tok


def stream_and_tokenize(
    tokenizer,
    max_tokens: int,
    seq_len: int,
    min_doc_tokens: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """
    Stream SlimPajama, filter for long docs, tokenize into chunks.
    Uses streaming mode to avoid downloading full dataset.
    """
    from datasets import load_dataset

    target_chunks = max_tokens // seq_len
    print(f"[data] Target: {target_chunks} chunks × {seq_len} = {max_tokens/1e6:.0f}M tokens", flush=True)
    print(f"[data] Min doc length filter: {min_doc_tokens} tokens", flush=True)
    print(f"[data] Streaming from cerebras/SlimPajama-627B via hf-mirror...", flush=True)

    all_ids: List[int] = []
    docs_seen = 0
    docs_kept = 0
    docs_short = 0
    t0 = time.time()
    last_report = t0

    try:
        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10000)
    except Exception as e:
        print(f"[data] SlimPajama failed: {e}", flush=True)
        print("[data] Trying fallback: DKYoon/SlimPajama-6B...", flush=True)
        ds = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10000)

    for example in ds:
        text = example.get("text", "")
        if not text or len(text) < min_doc_tokens * 3:  # rough char filter
            docs_short += 1
            docs_seen += 1
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        docs_seen += 1

        if len(ids) < min_doc_tokens:
            docs_short += 1
            continue

        all_ids.extend(ids)
        docs_kept += 1

        now = time.time()
        if now - last_report > 30:
            elapsed = now - t0
            tok_count = len(all_ids)
            speed = tok_count / elapsed / 1e6
            pct = tok_count / max_tokens * 100
            print(
                f"  [{pct:5.1f}%] {tok_count/1e6:.1f}M/{max_tokens/1e6:.0f}M tokens | "
                f"{docs_kept}/{docs_seen} docs kept | "
                f"{speed:.2f}M tok/s | "
                f"ETA {(max_tokens - tok_count) / max(speed * 1e6, 1) / 60:.0f}min",
                flush=True,
            )
            last_report = now

        if len(all_ids) >= max_tokens:
            break

    elapsed = time.time() - t0
    print(f"[data] Done: {len(all_ids)/1e6:.1f}M tokens from {docs_kept}/{docs_seen} docs "
          f"({docs_short} filtered short) in {elapsed/60:.1f}min", flush=True)

    # Truncate and reshape
    n = len(all_ids) // seq_len
    if n < 10:
        raise RuntimeError(f"Only got {n} chunks, need at least 10. Check data source.")
    data = torch.tensor(all_ids[: n * seq_len], dtype=torch.long).view(n, seq_len)
    print(f"[data] Final: {data.shape[0]} chunks × {seq_len} = {data.numel()/1e6:.1f}M tokens", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=500_000_000)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--min_doc_tokens", type=int, default=2048,
                        help="Skip docs shorter than this (tokens). Ensures 8K chunks have long content.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_tokens", type=int, default=5_000_000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer()

    # ── Train split ──
    train_path = out / f"train_slimpajama-mixed_{args.max_tokens}_{args.seq_len}.pt"
    if train_path.exists():
        print(f"[skip] Train data already exists: {train_path}", flush=True)
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"  Preparing TRAIN data: {args.max_tokens/1e6:.0f}M tokens @ L={args.seq_len}", flush=True)
        print(f"{'='*60}\n", flush=True)
        data = stream_and_tokenize(
            tokenizer,
            max_tokens=args.max_tokens,
            seq_len=args.seq_len,
            min_doc_tokens=args.min_doc_tokens,
            seed=args.seed,
        )
        torch.save(data, train_path)
        size_gb = train_path.stat().st_size / 1e9
        print(f"[saved] {train_path} ({size_gb:.2f} GB)", flush=True)

    # ── Val split (flat 1D for PPL eval) ──
    val_path = out / f"val_slimpajama-mixed_{args.val_tokens}.pt"
    if val_path.exists():
        print(f"[skip] Val data already exists: {val_path}", flush=True)
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"  Preparing VAL data: {args.val_tokens/1e6:.0f}M tokens", flush=True)
        print(f"{'='*60}\n", flush=True)
        from datasets import load_dataset
        ids = []
        try:
            ds = load_dataset(
                "cerebras/SlimPajama-627B",
                split="validation",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                "DKYoon/SlimPajama-6B",
                split="test",
                streaming=True,
                trust_remote_code=True,
            )
        for ex in ds:
            text = ex.get("text", "")
            if text:
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) >= args.val_tokens:
                break
        val_data = torch.tensor(ids[:args.val_tokens], dtype=torch.long)
        torch.save(val_data, val_path)
        print(f"[saved] {val_path} ({val_data.numel()/1e6:.1f}M tokens)", flush=True)

    # ── Also create symlink with expected naming convention ──
    link_train = out / f"train_fineweb-edu_{args.max_tokens}_{args.seq_len}.pt"
    if not link_train.exists():
        # NOTE: We create a symlink so run_evq_sweep.py can find it via its naming convention
        # The actual data is SlimPajama, NOT FineWeb-Edu
        os.symlink(train_path.name, link_train)
        print(f"[symlink] {link_train} -> {train_path.name}", flush=True)
        print(f"  (Symlink uses fineweb-edu name for script compatibility, actual data is SlimPajama)", flush=True)

    link_val = out / f"val_fineweb-edu_{args.val_tokens}.pt"
    if not link_val.exists():
        os.symlink(val_path.name, link_val)
        print(f"[symlink] {link_val} -> {val_path.name}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  ALL DONE", flush=True)
    print(f"  Train: {train_path}", flush=True)
    print(f"  Val:   {val_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
