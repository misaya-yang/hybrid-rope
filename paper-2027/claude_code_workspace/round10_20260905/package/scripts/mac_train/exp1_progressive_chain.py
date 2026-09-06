#!/usr/bin/env python3
"""
EXP-1: Progressive Chain 512→1024→2048 (125M, M4 Max)

验证 Phase 17b "EVQ越训越强 + YaRN相变" 在 125M 上复现并延伸到 2048。

Design:
  Stage 0: L=512,  50M tokens → checkpoint
  Stage 1: L=1024, 25M tokens (from Stage 0 ckpt) → checkpoint
  Stage 2: L=2048, 25M tokens (from Stage 1 ckpt) → checkpoint

  Each stage: eval raw + YaRN @ {512, 1K, 2K, 4K, 8K, 16K, 32K}
  Methods: Geo (τ=0) vs EVQ (τ*=d/√L_current)
  Seeds: 42, 137, 256

Usage:
  python exp1_progressive_chain.py              # Full run (3 seeds)
  python exp1_progressive_chain.py --pilot      # seed=42 only
  python exp1_progressive_chain.py --dry-run    # Print plan only
  python exp1_progressive_chain.py --summary    # Summarize existing results
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

# Setup paths
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent / "core_text_phases"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(CORE_DIR.parent / "supporting_eval"))

from run_evq_sweep import (
    GPT, DEVICE, DTYPE, USE_AUTOCAST,
    evq_cosh_inv_freq, load_val, load_data, set_seed,
)

# ============ Config ============
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # hybrid-rope/
WORK_DIR = PROJECT_ROOT / "results" / "mac_train" / "exp1_progressive_chain"
BASE = 500_000
D_HEAD = 64
SEEDS = [42, 137, 256]

STAGES = [
    {"name": "stage0_L512",  "seq_len": 512,  "tokens": 50_000_000},
    {"name": "stage1_L1024", "seq_len": 1024, "tokens": 25_000_000},
    {"name": "stage2_L2048", "seq_len": 2048, "tokens": 25_000_000},
]

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

CFG_BASE = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=D_HEAD,
    intermediate_size=3072,
    lr=6e-4,
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


def get_tau_star(d_head, seq_len):
    return d_head / math.sqrt(seq_len)


def make_cfg(stage):
    """Create config for a given stage."""
    cfg = dict(CFG_BASE)
    cfg["seq_len"] = stage["seq_len"]
    cfg["max_position_embeddings"] = stage["seq_len"]
    cfg["train_tokens"] = stage["tokens"]
    cfg["batch_size"] = cfg["micro_batch_size"] * cfg["grad_accum"]
    return cfg


def result_path(method, seed, stage_name):
    return WORK_DIR / f"{method}_seed{seed}" / stage_name / "result.json"


def ckpt_path(method, seed, stage_name):
    return WORK_DIR / f"{method}_seed{seed}" / stage_name / "model.pt"


def result_exists(method, seed, stage_name):
    return result_path(method, seed, stage_name).exists()


def save_result(method, seed, stage_name, result):
    p = result_path(method, seed, stage_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {p}")


def save_checkpoint(model, optimizer, method, seed, stage_name):
    p = ckpt_path(method, seed, stage_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, p)
    print(f"  Checkpoint: {p}")


def load_checkpoint(model, optimizer, method, seed, stage_name):
    p = ckpt_path(method, seed, stage_name)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Loaded checkpoint: {p}")


def make_train_loader(seq_len, micro_batch_size, seed):
    """Create training data loader."""
    cache_dir = str(WORK_DIR / "data_cache")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Try to find existing cache
    data = None
    prefix = f"train_fineweb-edu_"
    suffix = f"_{seq_len}.pt"
    candidates = sorted(cache_path.glob(f"{prefix}*{suffix}"), reverse=True)

    for p in candidates:
        try:
            print(f"  [data] Loading from cache: {p}")
            data = torch.load(p, weights_only=True)
            print(f"  [data] Cached: {data.shape[0]} chunks ({data.numel()/1e6:.1f}M tokens)")
            break
        except Exception as e:
            print(f"  [data] Failed: {e}")
            continue

    if data is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        max_tokens = 100_000_000
        data = load_data(tokenizer, max_tokens=max_tokens, seq_len=seq_len,
                        dataset="fineweb-edu", cache_dir=cache_dir)

    n_chunks = data.shape[0]
    torch.manual_seed(seed)

    def batch_gen():
        while True:
            idx = torch.randint(0, n_chunks, (micro_batch_size,))
            yield data[idx]

    return batch_gen()


def train_stage(model, cfg, device):
    """Train for one stage. Returns (model, optimizer, tokens_trained, steps)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1
    )

    warmup_steps = 100
    tokens_per_step = cfg["micro_batch_size"] * cfg["grad_accum"] * cfg["seq_len"]
    total_steps = cfg["train_tokens"] // tokens_per_step

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = make_train_loader(cfg["seq_len"], cfg["micro_batch_size"], seed=42)

    model.train()
    tokens_trained = 0
    step = 0
    t0 = time.time()

    while tokens_trained < cfg["train_tokens"]:
        optimizer.zero_grad()
        total_loss = 0

        for _ in range(cfg["grad_accum"]):
            batch = next(loader).to(device)
            with torch.autocast(device_type=device if isinstance(device, str) else device.type,
                              dtype=DTYPE) if USE_AUTOCAST else nullcontext():
                logits = model(batch)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1)
                ) / cfg["grad_accum"]
            loss.backward()
            total_loss += loss.item()
            tokens_trained += batch.numel()
            if tokens_trained >= cfg["train_tokens"]:
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Step {step}/{total_steps}: {tokens_trained/1e6:.1f}M tok, "
                  f"loss={total_loss:.4f}, {elapsed/60:.1f}min")

    elapsed = time.time() - t0
    print(f"  Training done: {step} steps, {tokens_trained/1e6:.1f}M tok, {elapsed/60:.1f}min")
    return optimizer, tokens_trained, step


def eval_model(model, val_data, eval_lengths, yarn_factor=None):
    """Evaluate model. If yarn_factor is set, apply YaRN-style NTK scaling."""
    model.eval()

    if yarn_factor is not None and yarn_factor > 1.0:
        # Simple NTK-aware scaling: base' = base * factor^(d/(d-2))
        orig_inv_freq = model.blocks[0].attn.rope.inv_freq.clone()
        scaled_base = BASE * (yarn_factor ** (D_HEAD / (D_HEAD - 2)))
        # Recompute inv_freq with scaled base for YaRN-like effect
        n = D_HEAD // 2
        new_inv_freq = torch.tensor(
            [1.0 / (scaled_base ** (2 * i / D_HEAD)) for i in range(n)],
            dtype=torch.float32, device=orig_inv_freq.device,
        )
        model.blocks[0].attn.rope.inv_freq.copy_(new_inv_freq)
        model.blocks[0].attn.rope._build(max(eval_lengths) + 100)
    else:
        model.extend_rope(max(eval_lengths) + 100)

    ctx = torch.amp.autocast(DEVICE, dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)
    results = {}

    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            continue
        n_sample = min(10, max(max_start // L, 1))
        offsets = sorted(rng.choice(max_start, size=n_sample, replace=False))

        for offset in offsets:
            chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                    )
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM, skipping")
                    break
                raise

        if losses:
            avg = sum(losses) / len(losses)
            results[L] = round(math.exp(min(avg, 20)), 2)

    # Restore original inv_freq if we changed it
    if yarn_factor is not None and yarn_factor > 1.0:
        model.blocks[0].attn.rope.inv_freq.copy_(orig_inv_freq)

    return results


def run_one_chain(method, seed, stages, device, dry_run=False):
    """Run a full progressive chain for one method+seed."""
    print(f"\n{'='*60}")
    print(f"Chain: {method}, seed={seed}")
    print(f"{'='*60}")

    # Check if all stages already done
    all_done = all(result_exists(method, seed, s["name"]) for s in stages)
    if all_done:
        print(f"  [SKIP] All stages complete for {method}/seed={seed}")
        return

    if dry_run:
        for s in stages:
            done = "DONE" if result_exists(method, seed, s["name"]) else "TODO"
            print(f"  [{done}] {s['name']}: L={s['seq_len']}, {s['tokens']/1e6:.0f}M tok")
        return

    set_seed(seed)

    # Load val data once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    val_data = load_val(tokenizer, max_tokens=5_000_000, dataset="fineweb-edu",
                        cache_dir=str(WORK_DIR / "data_cache"))

    model = None
    optimizer = None

    for i, stage in enumerate(stages):
        stage_name = stage["name"]
        cfg = make_cfg(stage)

        print(f"\n--- {stage_name} (L={stage['seq_len']}, {stage['tokens']/1e6:.0f}M tok) ---")

        if result_exists(method, seed, stage_name):
            print(f"  [SKIP] Already done, loading checkpoint for next stage")
            if i < len(stages) - 1:  # Need checkpoint for next stage
                tau = get_tau_star(D_HEAD, stage["seq_len"]) if method == "evq" else 0.0
                inv_freq = evq_cosh_inv_freq(D_HEAD, tau, BASE) if tau > 0 else geometric_inv_freq(D_HEAD, BASE)
                model = GPT(cfg, inv_freq=inv_freq).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                              betas=(0.9, 0.95), weight_decay=0.1)
                load_checkpoint(model, optimizer, method, seed, stage_name)
            continue

        # Create or update model
        tau = get_tau_star(D_HEAD, stage["seq_len"]) if method == "evq" else 0.0
        inv_freq = evq_cosh_inv_freq(D_HEAD, tau, BASE) if tau > 0 else geometric_inv_freq(D_HEAD, BASE)

        if model is None:
            # Stage 0: create fresh
            model = GPT(cfg, inv_freq=inv_freq).to(device)
        else:
            # Continue stages: update RoPE for new seq_len
            model.blocks[0].attn.rope.inv_freq.copy_(inv_freq)
            model.blocks[0].attn.rope._build(cfg["seq_len"])

        print(f"  tau*={tau:.3f}, method={method}")

        # Train
        optimizer, tokens_trained, steps = train_stage(model, cfg, device)

        # Save checkpoint
        save_checkpoint(model, optimizer, method, seed, stage_name)

        # Eval raw
        print(f"  Evaluating (raw)...")
        ppl_raw = eval_model(model, val_data, EVAL_LENGTHS)
        for L, ppl in sorted(ppl_raw.items()):
            print(f"    raw  L={L}: PPL={ppl}")

        # Eval with YaRN-style scaling (factor = max_eval / train_len)
        yarn_factor = max(EVAL_LENGTHS) / stage["seq_len"]
        print(f"  Evaluating (yarn, factor={yarn_factor:.1f})...")
        ppl_yarn = eval_model(model, val_data, EVAL_LENGTHS, yarn_factor=yarn_factor)
        for L, ppl in sorted(ppl_yarn.items()):
            print(f"    yarn L={L}: PPL={ppl}")

        # Save result
        result = {
            "method": method,
            "seed": seed,
            "stage": stage_name,
            "seq_len": stage["seq_len"],
            "tokens": tokens_trained,
            "tau": round(tau, 3),
            "base": BASE,
            "d_head": D_HEAD,
            "ppl_raw": {str(k): v for k, v in ppl_raw.items()},
            "ppl_yarn": {str(k): v for k, v in ppl_yarn.items()},
            "train_steps": steps,
        }
        save_result(method, seed, stage_name, result)

        # Memory cleanup between stages
        dev_str = device if isinstance(device, str) else device.type
        if dev_str == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    # Final cleanup
    del model, val_data
    dev_str = device if isinstance(device, str) else device.type
    if dev_str == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif dev_str == "cuda":
        torch.cuda.empty_cache()


def generate_summary():
    """Generate summary of all results."""
    print(f"\n{'='*60}")
    print("EXP-1 Summary: Progressive Chain")
    print(f"{'='*60}")

    all_results = []
    for p in WORK_DIR.glob("*/stage*/result.json"):
        with open(p) as f:
            all_results.append(json.load(f))

    if not all_results:
        print("No results found")
        return

    # Group by method × stage
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["method"], r["stage"])
        grouped[key].append(r)

    # Print comparison table
    print(f"\n{'method':<6} {'stage':<16} {'seeds':<4} {'raw@4K':<8} {'raw@8K':<8} {'raw@16K':<9} "
          f"{'yarn@4K':<9} {'yarn@8K':<9} {'yarn@16K':<9}")
    print("-" * 90)

    for (method, stage), runs in sorted(grouped.items()):
        n = len(runs)
        raw_4k = np.mean([r["ppl_raw"].get("4096", float('inf')) for r in runs])
        raw_8k = np.mean([r["ppl_raw"].get("8192", float('inf')) for r in runs])
        raw_16k = np.mean([r["ppl_raw"].get("16384", float('inf')) for r in runs])
        yarn_4k = np.mean([r["ppl_yarn"].get("4096", float('inf')) for r in runs])
        yarn_8k = np.mean([r["ppl_yarn"].get("8192", float('inf')) for r in runs])
        yarn_16k = np.mean([r["ppl_yarn"].get("16384", float('inf')) for r in runs])
        print(f"{method:<6} {stage:<16} {n:<4} {raw_4k:<8.1f} {raw_8k:<8.1f} {raw_16k:<9.1f} "
              f"{yarn_4k:<9.1f} {yarn_8k:<9.1f} {yarn_16k:<9.1f}")

    # Key comparisons
    print(f"\n--- Key Comparisons ---")
    for stage in ["stage0_L512", "stage1_L1024", "stage2_L2048"]:
        geo_runs = grouped.get(("geo", stage), [])
        evq_runs = grouped.get(("evq", stage), [])
        if geo_runs and evq_runs:
            geo_16k = np.mean([r["ppl_raw"].get("16384", float('inf')) for r in geo_runs])
            evq_16k = np.mean([r["ppl_raw"].get("16384", float('inf')) for r in evq_runs])
            evq_yarn_16k = np.mean([r["ppl_yarn"].get("16384", float('inf')) for r in evq_runs])
            adv = (1 - evq_16k / geo_16k) * 100 if geo_16k > 0 else 0
            print(f"  {stage}: EVQ raw={evq_16k:.1f} vs Geo raw={geo_16k:.1f} "
                  f"(EVQ advantage: {adv:.1f}%)")
            print(f"  {stage}: EVQ raw={evq_16k:.1f} vs EVQ yarn={evq_yarn_16k:.1f} "
                  f"({'raw wins' if evq_16k < evq_yarn_16k else 'yarn wins'})")

    # Save summary
    summary_file = WORK_DIR / "exp1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="EXP-1: Progressive Chain 512→1024→2048")
    parser.add_argument("--pilot", action="store_true", help="seed=42 only")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--summary", action="store_true", help="Summarize results")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary:
        generate_summary()
        return

    seeds = [42] if args.pilot else SEEDS
    device = DEVICE

    print(f"Device: {device}")
    print(f"Work dir: {WORK_DIR}")
    print(f"Seeds: {seeds}")
    print(f"Stages: {[s['name'] for s in STAGES]}")

    for method in ["geo", "evq"]:
        for seed in seeds:
            run_one_chain(method, seed, STAGES, device, args.dry_run)

    if not args.dry_run:
        generate_summary()

    print(f"\n{'='*60}")
    print("EXP-1 Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
