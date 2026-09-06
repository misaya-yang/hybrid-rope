#!/usr/bin/env python3
"""
Phase 18: Base Generalization Sweep - Simple Version

Usage:
    python phase18_simple.py --pilot     # Run base=500K, seed=42 only
    python phase18_simple.py             # Run all bases

Paper Role:  Appendix — Simplified base sweep (single-seed)
Input:       FineWeb-Edu streaming data
Output:      results/core_text/phase18/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "supporting_eval"))

from run_evq_sweep import (
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    load_val,
    load_data,
    set_seed,
)
from transformers import AutoTokenizer

# ============ Config ============
BASES = [10_000, 100_000, 500_000, 1_000_000, 10_000_000]
SEEDS = [42, 137, 256]
D_HEAD = 64
SEQ_LEN = 512
TRAIN_TOKENS = 50_000_000
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

CFG_125M = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=64,
    intermediate_size=3072,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TRAIN_TOKENS,
    lr=6e-4,
    batch_size=16,
    micro_batch_size=8,
    grad_accum=2,
)


def geometric_inv_freq(dim, base):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def train_one_run(model, train_data, optimizer, scheduler, tokens_to_train, cfg, device):
    """Train for specified tokens."""
    model.train()
    tokens_trained = 0
    step = 0
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    grad_accum = cfg["grad_accum"]
    n_chunks = train_data.shape[0]
    
    while tokens_trained < tokens_to_train:
        optimizer.zero_grad()
        total_loss = 0
        
        for micro_step in range(grad_accum):
            # Sample batch
            indices = torch.randint(0, n_chunks, (batch_size,))
            batch = train_data[indices].to(device)
            
            # Forward
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )
            loss = loss / grad_accum
            
            # Backward
            loss.backward()
            total_loss += loss.item() * grad_accum
            tokens_trained += batch.numel()
            
            if tokens_trained >= tokens_to_train:
                break
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step % 100 == 0:
            print(f"    Step {step}: {tokens_trained/1e6:.1f}M tokens, loss={total_loss:.4f}")
    
    return tokens_trained, step


def run_experiment(work_dir, base, tau, seed, dry_run=False):
    """Run one experiment."""
    method = "evq" if tau > 0 else "geo"
    name = f"d64_base{base}_{method}_seed{seed}"
    result_file = work_dir / name / "result.json"
    
    if result_file.exists():
        print(f"  [SKIP] {name} already exists")
        return None
    
    print(f"\n[RUN] {name}: base={base}, tau={tau:.3f}, seed={seed}")
    
    if dry_run:
        print(f"  [DRY RUN]")
        return None
    
    # Setup
    set_seed(seed)
    device = DEVICE
    
    # Create model
    if tau > 0:
        inv_freq = evq_cosh_inv_freq(dim=D_HEAD, tau=tau, base=float(base))
    else:
        inv_freq = geometric_inv_freq(D_HEAD, base)
    
    model = GPT(CFG_125M, inv_freq=inv_freq).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {n_params:.1f}M params")
    
    # Load data
    print(f"  Loading data...")
    cache_dir = work_dir / "data_cache"
    
    # Try 50M cache first
    train_cache = cache_dir / "train_fineweb-edu_50100000_512.pt"
    if not train_cache.exists():
        train_cache = cache_dir / "train_fineweb-edu_100000000_512.pt"
    
    if train_cache.exists():
        print(f"  Loading from cache: {train_cache.name}")
        train_data = torch.load(train_cache, weights_only=True)
    else:
        print(f"  Downloading data...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        train_data = load_data(tokenizer, TRAIN_TOKENS, SEQ_LEN, "fineweb-edu", str(cache_dir))
    
    print(f"  Train data: {train_data.shape[0]} chunks ({train_data.numel()/1e6:.1f}M tokens)")
    
    # Load validation data
    val_cache = cache_dir / "val_fineweb-edu_5000000.pt"
    if val_cache.exists():
        val_data = torch.load(val_cache, weights_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        val_data = load_val(tokenizer, 5_000_000, "fineweb-edu", str(cache_dir))
    print(f"  Val data: {val_data.numel()/1e6:.1f}M tokens")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Scheduler
    total_steps = TRAIN_TOKENS // (CFG_125M["batch_size"] * SEQ_LEN)
    warmup = 100
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Train
    print(f"  Training {TRAIN_TOKENS/1e6:.0f}M tokens...")
    start = time.time()
    tokens_trained, steps = train_one_run(model, train_data, optimizer, scheduler, TRAIN_TOKENS, CFG_125M, device)
    train_time = time.time() - start
    print(f"  Training done: {train_time/60:.1f} min")
    
    # Eval
    print(f"  Evaluating...")
    model.eval()
    ppl_results = {}
    for L in EVAL_LENGTHS:
        ppl = eval_model(model, val_data, [L], eval_chunks=10)
        ppl_results[str(L)] = round(ppl.get(L, float('inf')), 2)
        print(f"    L={L}: PPL={ppl_results[str(L)]}")
    
    # Save
    result = {
        "name": name,
        "base": base,
        "tau": round(tau, 3),
        "seed": seed,
        "ppl": ppl_results,
        "train_tokens": tokens_trained,
        "train_time_sec": round(train_time, 1),
        "model_params_M": round(n_params, 2),
    }
    
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_file}")
    
    # Cleanup
    del model, train_data, val_data
    torch.mps.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    work_dir = Path("results/core_text/phase18_base_sweep")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Work dir: {work_dir}")
    
    tau_star = D_HEAD / math.sqrt(SEQ_LEN)
    
    if args.pilot:
        print("\n*** PILOT MODE ***")
        for tau in [0.0, tau_star]:
            run_experiment(work_dir, 500_000, tau, 42, args.dry_run)
    else:
        print("\n*** FULL SWEEP ***")
        for base in BASES:
            for seed in SEEDS:
                for tau in [0.0, tau_star]:
                    run_experiment(work_dir, base, tau, seed, args.dry_run)
    
    print("\n=== Phase 18 Complete ===")


if __name__ == "__main__":
    main()
