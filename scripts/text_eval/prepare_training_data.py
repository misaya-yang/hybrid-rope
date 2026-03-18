"""Tokenize and pack training data into a single .pt file.

Mix ratio:
  60% PG-19 books (long, matches PPL eval distribution)
  20% RedPajama long docs (diverse web/code/science)
  20% RedPajama short docs (maintain short-context ability)

Output: train_packed.pt containing tokenized chunks of seq_len tokens.
Also: test_packed.pt from PG-19 test split for PPL evaluation.

Usage:
    python prepare_training_data.py --model_dir /path/to/Llama-3-8B-Instruct \
        --data_dir /home/ubuntu/data --output_dir /home/ubuntu/data/packed \
        --seq_len 8192
"""
import argparse
import json
import os
import random
import torch
from pathlib import Path


def load_jsonl(path):
    docs = []
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return docs
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    print(f"  Loaded {len(docs)} docs from {path}")
    return docs


def tokenize_and_chunk(texts, tokenizer, seq_len, max_chunks=None):
    """Tokenize texts and split into fixed-length chunks."""
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    # Split into seq_len chunks
    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i:i + seq_len])
        if max_chunks and len(chunks) >= max_chunks:
            break

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/data")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/data/packed")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--max_train_chunks", type=int, default=4000,
                        help="Max training chunks (4000 x 8192 = 32M tokens)")
    parser.add_argument("--max_test_chunks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}")

    # Load all data sources
    print("\n=== Loading data ===")
    pg19_train = load_jsonl(os.path.join(args.data_dir, "pg19", "train.jsonl"))
    redpajama_long = load_jsonl(os.path.join(args.data_dir, "redpajama", "long.jsonl"))
    redpajama_short = load_jsonl(os.path.join(args.data_dir, "redpajama", "short.jsonl"))
    pg19_test = load_jsonl(os.path.join(args.data_dir, "pg19", "test.jsonl"))

    # Shuffle each source
    random.shuffle(pg19_train)
    random.shuffle(redpajama_long)
    random.shuffle(redpajama_short)

    # Tokenize each source
    print("\n=== Tokenizing ===")
    target_total = args.max_train_chunks
    n_pg19 = int(target_total * 0.6)
    n_rp_long = int(target_total * 0.2)
    n_rp_short = target_total - n_pg19 - n_rp_long

    print(f"  Target: {n_pg19} PG-19 + {n_rp_long} RP-long + {n_rp_short} RP-short = {target_total} chunks")

    chunks_pg19 = tokenize_and_chunk(pg19_train, tokenizer, args.seq_len, n_pg19)
    print(f"  PG-19: {len(chunks_pg19)} chunks")

    chunks_rp_long = tokenize_and_chunk(redpajama_long, tokenizer, args.seq_len, n_rp_long)
    print(f"  RP-long: {len(chunks_rp_long)} chunks")

    chunks_rp_short = tokenize_and_chunk(redpajama_short, tokenizer, args.seq_len, n_rp_short)
    print(f"  RP-short: {len(chunks_rp_short)} chunks")

    # Combine and shuffle
    all_train = chunks_pg19 + chunks_rp_long + chunks_rp_short
    random.shuffle(all_train)
    print(f"\n  Total train: {len(all_train)} chunks = {len(all_train) * args.seq_len / 1e6:.1f}M tokens")

    # Pack into tensor
    train_tensor = torch.tensor(all_train, dtype=torch.long)
    torch.save(train_tensor, os.path.join(args.output_dir, "train_packed.pt"))
    print(f"  Saved: train_packed.pt {train_tensor.shape}")

    # Test set
    if pg19_test:
        chunks_test = tokenize_and_chunk(pg19_test, tokenizer, args.seq_len, args.max_test_chunks)
        test_tensor = torch.tensor(chunks_test, dtype=torch.long)
        torch.save(test_tensor, os.path.join(args.output_dir, "test_packed.pt"))
        print(f"  Saved: test_packed.pt {test_tensor.shape}")

        # Also save longer test chunks for extrapolation eval
        for eval_len in [16384, 32768]:
            chunks_eval = tokenize_and_chunk(pg19_test, tokenizer, eval_len, 100)
            if chunks_eval:
                eval_tensor = torch.tensor(chunks_eval, dtype=torch.long)
                torch.save(eval_tensor, os.path.join(args.output_dir, f"test_{eval_len}.pt"))
                print(f"  Saved: test_{eval_len}.pt {eval_tensor.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
