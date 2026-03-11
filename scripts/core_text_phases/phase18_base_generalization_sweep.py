#!/usr/bin/env python3
"""
Phase 18: Base Generalization Sweep + MLA-Realistic Config (125M, M4 Max Local)

Goal:
  (1) Prove EVQ gains generalize across base choices
  (2) Validate with MLA-era industrial config (d_head=64) matching DeepSeek V3 / GLM-5 / Kimi K2.5

Bases: 10K, 100K, 500K, 1M, 10M
Model: 125M (hidden=768, 12L, 12H, d_head=64)
Train: L=512, 50M tokens
Eval: 512, 1K, 2K, 4K, 8K, 16K (1x-32x extrapolation)

Environment:
  PHASE18_WORK=./results/phase18_base_sweep
  PHASE18_DATASET=fineweb-edu
  PHASE18_SEEDS=42,137,256
  PHASE18_MICRO_BATCH_SIZE=8
  PHASE18_GRAD_ACCUM=2
  PHASE18_RUN_ONLY=geo|evq|all
  PHASE18_BASES=10000,100000,500000,1000000,10000000
  PHASE18_MLA_COMPARE=0|1

Usage:
    python phase18_base_generalization_sweep.py              # Run all
    python phase18_base_generalization_sweep.py --pilot      # Run base=500K, seed=42 only
    python phase18_base_generalization_sweep.py --mla-compare # Run d_head=64 vs d_head=128 comparison
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
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
    evq_cosh_inv_freq,
    load_val,
    load_data,
    set_seed,
)

# ============ Config ============
BASES = [10_000, 100_000, 500_000, 1_000_000, 10_000_000]
SEEDS = [42, 137, 256]
D_HEAD_MLA = 64
D_HEAD_MHA = 128
SEQ_LEN = 512
TRAIN_TOKENS = 50_000_000
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

# 125M config: hidden=768, H=12, d_head=64
CFG_125M_MLA = dict(
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

# 125M config: hidden=768, H=6, d_head=128 (MHA comparison)
CFG_125M_MHA = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    intermediate_size=3072,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TRAIN_TOKENS,
    lr=6e-4,
    batch_size=16,
    micro_batch_size=8,
    grad_accum=2,
)

# ============ Helpers ============
def geometric_inv_freq(dim, base):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def collision_fraction(base, seq_len):
    """Compute collision block fraction c = ln(L) / ln(b)"""
    return math.log(seq_len) / math.log(base)


def result_exists(work_dir, name):
    result_file = work_dir / name / "result.json"
    return result_file.exists()


def save_result(work_dir, name, result):
    result_dir = work_dir / name
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_dir / 'result.json'}")


def train_model(model, train_loader, optimizer, scheduler, tokens_to_train, cfg, device):
    """Simple training loop."""
    model.train()
    tokens_trained = 0
    step = 0
    seq_len = cfg["seq_len"]
    micro_batch = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum"]
    
    while tokens_trained < tokens_to_train:
        accumulated = 0
        optimizer.zero_grad()
        total_loss = 0
        
        for micro_step in range(grad_accum):
            batch = next(train_loader)
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=DTYPE) if USE_AUTOCAST else nullcontext():
                logits = model(batch)
                # Compute loss: predict next token
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1)
                )
                loss = loss / grad_accum
            
            loss.backward()
            total_loss += loss.item()
            accumulated += batch.numel()
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


def make_train_loader(dataset_name, seq_len, micro_batch_size, seed, cache_dir=None):
    """Create training data loader from cached data.

    IMPORTANT: yields micro-batches (not full batches). The training loop
    calls next() once per grad-accum micro-step.
    """
    # Resolve cache directory
    if cache_dir is None:
        # Default: <project_root>/results/core_text/phase18_base_sweep/data_cache
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # scripts/core_text_phases → scripts → hybrid-rope
        cache_dir = str(project_root / "results" / "core_text" / "phase18_base_sweep" / "data_cache")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Try to find existing cache (any token count, same dataset+seq_len)
    data = None
    prefix = f"train_{dataset_name}_"
    suffix = f"_{seq_len}.pt"
    candidates = sorted(cache_path.glob(f"{prefix}*{suffix}"), reverse=True)

    for p in candidates:
        try:
            print(f"  [data] Loading from cache: {p}")
            data = torch.load(p, weights_only=True)
            print(f"  [data] Cached: {data.shape[0]} chunks ({data.numel()/1e6:.1f}M tokens)")
            break
        except Exception as e:
            print(f"  [data] Failed to load {p}: {e}")
            continue

    if data is None:
        # Fall back to loading fresh via run_evq_sweep
        from run_evq_sweep import load_data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        max_tokens = 100_000_000
        data = load_data(tokenizer, max_tokens=max_tokens, seq_len=seq_len,
                        dataset=dataset_name, cache_dir=cache_dir)

    # data shape: (n_chunks, seq_len)
    n_chunks = data.shape[0]
    print(f"  [data] Ready: {n_chunks} chunks, micro_batch={micro_batch_size}")

    # Simple random sampler
    torch.manual_seed(seed)

    def batch_generator():
        while True:
            indices = torch.randint(0, n_chunks, (micro_batch_size,))
            yield data[indices]

    return batch_generator()


def eval_model_simple(model, val_data, eval_lengths, eval_chunks=10, eval_seed=9999):
    """Simple eval function matching run_evq_sweep's eval_model."""
    import numpy as np
    model.eval()
    model.extend_rope(max(eval_lengths) + 100)

    ctx = torch.amp.autocast(DEVICE, dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(eval_seed)
    results = {}

    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            print(f"    L={L}: val_data too short, skipping")
            continue
        n_chunks_avail = max(max_start // L, 1)
        n_sample = min(eval_chunks, n_chunks_avail)
        offsets = sorted(rng.choice(max_start, size=n_sample, replace=False))
        
        for offset in offsets:
            chunk = val_data[offset : offset + L].unsqueeze(0).to(DEVICE)
            try:
                with ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                    )
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM at offset {offset}, skipping")
                    continue
                raise
        
        if losses:
            avg_loss = sum(losses) / len(losses)
            ppl = math.exp(min(avg_loss, 20))
            results[L] = ppl
        else:
            results[L] = float('inf')
    
    return results


def run_single_experiment(work_dir, cfg, base, tau, seed, d_head, device, dry_run=False):
    """Run one training + eval."""
    from transformers import AutoTokenizer
    
    method = "evq" if tau > 0 else "geo"
    name = f"d{d_head}_base{base}_{method}_tau{tau:.2f}_seed{seed}"
    
    if result_exists(work_dir, name):
        print(f"  [SKIP] {name} already exists")
        return None
    
    print(f"\n  [RUN] {name}")
    print(f"        base={base}, tau={tau:.3f}, d_head={d_head}, seed={seed}")
    
    if dry_run:
        print(f"  [DRY RUN] Would train {name}")
        return None
    
    # Setup
    set_seed(seed)
    
    # Create inv_freq
    if tau > 0:
        inv_freq = evq_cosh_inv_freq(head_dim=d_head, tau=tau, base=float(base))
    else:
        inv_freq = geometric_inv_freq(dim=d_head, base=float(base))
    
    # Create model
    model = GPT(cfg, inv_freq=inv_freq).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"        Model: {n_params:.1f}M params")
    
    # Load validation data
    print(f"  Loading validation data...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    val_data = load_val(tokenizer, max_tokens=5_000_000, dataset="fineweb-edu", 
                        cache_dir=str(work_dir / "data_cache"))
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Scheduler: warmup + cosine decay
    warmup_steps = 100
    tokens_per_step = cfg["micro_batch_size"] * cfg["grad_accum"] * cfg["seq_len"]
    total_steps = cfg["train_tokens"] // tokens_per_step

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train
    print(f"  Training ({total_steps} steps, {tokens_per_step} tok/step)...")
    train_loader = make_train_loader("fineweb-edu", cfg["seq_len"], cfg["micro_batch_size"], seed,
                                     cache_dir=str(work_dir / "data_cache"))
    start_time = time.time()
    
    tokens_trained, steps = train_model(
        model, train_loader, optimizer, scheduler,
        cfg["train_tokens"], cfg, device
    )
    
    train_time = time.time() - start_time
    print(f"  Training done: {train_time/60:.1f} min, {tokens_trained/1e6:.1f}M tokens")
    
    # Eval
    print(f"  Evaluating...")
    ppl_dict = eval_model_simple(model, val_data, EVAL_LENGTHS, eval_chunks=10)
    ppl_results = {str(k): round(v, 2) for k, v in ppl_dict.items()}
    for L, ppl in ppl_results.items():
        print(f"    L={L}: PPL={ppl}")
    
    # Save result
    result = {
        "name": name,
        "base": base,
        "tau": round(tau, 3),
        "seed": seed,
        "d_head": d_head,
        "method": method,
        "ppl": ppl_results,
        "train_tokens": tokens_trained,
        "train_time_sec": round(train_time, 1),
        "model_params_M": round(n_params, 2),
        "collision_fraction": round(collision_fraction(base, SEQ_LEN), 4),
    }
    
    save_result(work_dir, name, result)
    
    # Cleanup
    del model, val_data
    dev_str = device if isinstance(device, str) else device.type
    if dev_str == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif dev_str == "cuda":
        torch.cuda.empty_cache()
    
    return result


def run_mla_comparison(work_dir, device, dry_run=False):
    """Run d_head=64 vs d_head=128 comparison at base=500K."""
    print("\n" + "="*60)
    print("MLA vs MHA Comparison (base=500K only)")
    print("="*60)
    
    base = 500_000
    results = []
    
    for d_head, cfg in [(64, CFG_125M_MLA), (128, CFG_125M_MHA)]:
        tau_star = d_head / math.sqrt(SEQ_LEN)
        print(f"\n--- d_head={d_head}, tau*={tau_star:.3f} ---")
        
        for seed in SEEDS:
            for tau in [0.0, tau_star]:
                result = run_single_experiment(
                    work_dir, cfg, base, tau, seed, d_head, device, dry_run
                )
                if result:
                    results.append(result)
    
    return results


def run_main_sweep(work_dir, device, bases=None, seeds=None, dry_run=False):
    """Run main base sweep with d_head=64."""
    bases = bases or BASES
    seeds = seeds or SEEDS
    tau_star = D_HEAD_MLA / math.sqrt(SEQ_LEN)
    
    print("\n" + "="*60)
    print("Phase 18: Base Generalization Sweep (d_head=64)")
    print("="*60)
    print(f"Bases: {bases}")
    print(f"Seeds: {seeds}")
    print(f"tau* = {tau_star:.3f}")
    print(f"Device: {device}")
    print("="*60)
    
    results = []
    total_runs = len(bases) * len(seeds) * 2  # 2 methods per config
    run_count = 0
    
    for base in bases:
        c = collision_fraction(base, SEQ_LEN)
        print(f"\n{'='*60}")
        print(f"Base = {base:,} (collision fraction c = {c:.3f})")
        print(f"{'='*60}")
        
        for seed in seeds:
            for tau in [0.0, tau_star]:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}]")
                
                result = run_single_experiment(
                    work_dir, CFG_125M_MLA, base, tau, seed, D_HEAD_MLA, device, dry_run
                )
                if result:
                    results.append(result)
    
    return results


def generate_summary(work_dir):
    """Generate summary of all results."""
    print("\n" + "="*60)
    print("Generating Summary")
    print("="*60)
    
    all_results = []
    for result_file in work_dir.glob("*/result.json"):
        with open(result_file) as f:
            all_results.append(json.load(f))
    
    if not all_results:
        print("No results found")
        return
    
    # Group by config
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["d_head"], r["base"], r["method"])
        grouped[key].append(r)
    
    # Compute statistics
    summary = []
    for (d_head, base, method), runs in sorted(grouped.items()):
        ppls = {k: [r["ppl"][k] for r in runs] for k in runs[0]["ppl"].keys()}
        
        summary.append({
            "d_head": d_head,
            "base": base,
            "method": method,
            "n_runs": len(runs),
            "seeds": [r["seed"] for r in runs],
            "ppl_mean": {k: round(sum(v)/len(v), 2) for k, v in ppls.items()},
            "ppl_std": {k: round(np.std(v), 2) for k, v in ppls.items()},
            "collision_fraction": runs[0]["collision_fraction"],
        })
    
    # Save summary
    summary_file = work_dir / "phase18_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved: {summary_file}")
    
    # Print table
    print("\n--- Summary Table ---")
    print(f"{'d_head':<8} {'base':<12} {'method':<6} {'n':<4} {'PPL@512':<10} {'PPL@4K':<10} {'PPL@16K':<10}")
    print("-" * 70)
    for s in summary:
        ppl_512 = s["ppl_mean"].get("512", 0)
        ppl_4k = s["ppl_mean"].get("4096", 0)
        ppl_16k = s["ppl_mean"].get("16384", 0)
        print(f"{s['d_head']:<8} {s['base']:<12,} {s['method']:<6} {s['n_runs']:<4} {ppl_512:<10.2f} {ppl_4k:<10.2f} {ppl_16k:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 18: Base Generalization Sweep")
    parser.add_argument("--pilot", action="store_true", help="Run pilot: base=500K, seed=42 only")
    parser.add_argument("--mla-compare", action="store_true", help="Run MLA vs MHA comparison")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without training")
    parser.add_argument("--summary-only", action="store_true", help="Generate summary from existing results")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory")
    args = parser.parse_args()
    
    # Setup
    work_dir = Path(args.work_dir) if args.work_dir else Path("results/core_text/phase18_base_sweep")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    device = DEVICE
    print(f"Device: {device}")
    print(f"Work dir: {work_dir}")
    
    if args.summary_only:
        generate_summary(work_dir)
        return
    
    results = []
    
    if args.mla_compare:
        # Run MLA comparison only
        results = run_mla_comparison(work_dir, device, args.dry_run)
    elif args.pilot:
        # Pilot: base=500K, seed=42
        print("\n*** PILOT MODE ***")
        tau_star = D_HEAD_MLA / math.sqrt(SEQ_LEN)
        for tau in [0.0, tau_star]:
            result = run_single_experiment(
                work_dir, CFG_125M_MLA, 500_000, tau, 42, D_HEAD_MLA, device, args.dry_run
            )
            if result:
                results.append(result)
    else:
        # Main sweep
        results = run_main_sweep(work_dir, device, dry_run=args.dry_run)
    
    # Generate summary
    if results and not args.dry_run:
        generate_summary(work_dir)
    
    print("\n" + "="*60)
    print("Phase 18 Complete!")
    print(f"Results: {work_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
