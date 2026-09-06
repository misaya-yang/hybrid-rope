#!/usr/bin/env python3
"""
Phase 11C: 454M Token-Scaling — Does more training make EVQ advantage stronger?

Reuses Phase 11 code but trains with 500M tokens (5× original 100M).
Only Geo and EVQ τ=4.0 to save compute.

Compare with existing Phase 11 results (100M tokens) for scaling trend.

Paper Role:  Table 4 — 454M PE-dominant scaling (supporting)
Input:       Phase 9 pre-tokenized data (FineWeb-Edu)
Output:      results/core_text/phase11c/
"""

import json, math, os, sys, time, gc, hashlib
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, DEVICE, DTYPE, USE_AUTOCAST,
    set_seed, get_batch_from_data,
)
from phase11_L256_extrap import (
    geometric_inv_freq, evq_cosh_inv_freq,
    CFG_350M, EVAL_LENGTHS,
    load_validation_data, eval_ppl,
)

# ── Config overrides ─────────────────────────────────────────────────
WORK = Path("/root/autodl-tmp/evq_phase11c_token_scaling")
PHASE9_TRAIN = Path("/root/autodl-tmp/evq_phase9/data/train_fineweb-edu_2000000000_2048.pt")
SEQ_LEN = 256


def load_train_data(max_tokens, seq_len=SEQ_LEN):
    cache_path = WORK / "data" / f"train_fineweb-edu_{max_tokens}_{seq_len}.pt"
    if cache_path.exists():
        print(f"  Loading cached data: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Loading Phase 9 data from {PHASE9_TRAIN} and re-chunking...")
    data = torch.load(PHASE9_TRAIN, weights_only=True)
    flat = data.reshape(-1)[:max_tokens]
    n = len(flat) // seq_len
    result = flat[:n * seq_len].reshape(n, seq_len)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    print(f"  Re-chunked: {result.shape} ({result.numel()/1e6:.0f}M tokens)")
    del data, flat
    gc.collect()
    return result


def train_model(model, train_data, cfg, seed=42):
    set_seed(seed)
    total_tokens = cfg["train_tokens"]
    bs = cfg["batch_size"]
    mbs = cfg["micro_batch_size"]
    ga = cfg["grad_accum"]
    seq_len = cfg["seq_len"]
    lr = cfg["lr"]

    tokens_per_step = bs * seq_len
    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(200, total_steps // 10)

    print(f"  Training: {total_tokens/1e6:.0f}M tok, bs={bs} (micro={mbs}×ga={ga}), "
          f"L={seq_len}, steps={total_steps}, lr={lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.1)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))

    model.train()
    n_samples = len(train_data)
    perm = torch.randperm(n_samples)
    ptr = 0
    t0 = time.time()
    log_interval = max(1, total_steps // 50)

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            for pg in opt.param_groups:
                pg["lr"] = lr * step / warmup_steps

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(ga):
            if ptr + mbs > n_samples:
                perm = torch.randperm(n_samples)
                ptr = 0
            indices = perm[ptr:ptr + mbs]
            ptr += mbs
            batch = get_batch_from_data(train_data, indices).to(DEVICE)

            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       batch[:, 1:].reshape(-1))
                loss_scaled = loss / ga

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / ga

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step > warmup_steps:
            sched.step()

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tps = (step * tokens_per_step) / elapsed
            print(f"    step {step:>6d}/{total_steps} ({step/total_steps*100:5.1f}%) | "
                  f"loss={accum_loss:.4f} | lr={opt.param_groups[0]['lr']:.2e} | "
                  f"{tps/1e6:.2f}M tok/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return elapsed


def run_one(run_id, inv_freq, seed, train_data, val_data, cfg):
    run_dir = WORK / run_id
    result_path = run_dir / "result.json"

    if result_path.exists():
        print(f"\n  SKIP {run_id} (result exists)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  RUN: {run_id}")
    freq_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:12]
    print(f"  inv_freq: min={inv_freq.min():.8f} max={inv_freq.max():.6f} hash={freq_hash}")
    print(f"{'='*70}")

    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)

    train_time = train_model(model, train_data, cfg, seed=seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"\n  Evaluating PPL...")
    ppl = eval_ppl(model, val_data)

    result = {
        "run_id": run_id,
        "seed": seed,
        "ppl": ppl,
        "train_time_sec": round(train_time, 1),
        "inv_freq_hash": freq_hash,
        "tokens": cfg["train_tokens"],
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137")
    parser.add_argument("--tokens", type=int, default=500_000_000,
                        help="Training tokens (default: 500M)")
    parser.add_argument("--micro_batch", type=int, default=128,
                        help="Micro batch size (128 for R6000 96GB)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    total_tokens = args.tokens

    cfg = CFG_350M.copy()
    cfg["train_tokens"] = total_tokens
    cfg["micro_batch_size"] = args.micro_batch
    cfg["grad_accum"] = cfg["batch_size"] // args.micro_batch

    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Phase 11C: 454M Token Scaling, L={SEQ_LEN}")
    print(f"  Tokens: {total_tokens/1e6:.0f}M, Seeds: {seeds}")
    print(f"  Batch: {cfg['batch_size']} (micro={cfg['micro_batch_size']}×ga={cfg['grad_accum']})")

    # Load data — need tokens + margin
    print("\n[1] Loading data...")
    train_data = load_train_data(max_tokens=total_tokens + 5_000_000)
    val_data = load_validation_data()
    print(f"  Train: {train_data.shape}, Val: {val_data.shape}")

    # Methods: only Geo and EVQ τ=4.0
    methods = {
        "geo": geometric_inv_freq(),
        "evq4.0": evq_cosh_inv_freq(tau=4.0),
    }

    all_results = {}
    for name, inv_freq in methods.items():
        for seed in seeds:
            r = run_one(f"454m_{name}_{total_tokens//1_000_000}M_seed{seed}",
                       inv_freq, seed, train_data, val_data, cfg)
            all_results[r["run_id"]] = r

    # Save aggregate
    agg_path = WORK / f"results_{total_tokens//1_000_000}M.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  PHASE 11C: 454M @ {total_tokens/1e6:.0f}M tokens, L_train={SEQ_LEN}")
    print(f"{sep}")

    def mean_ppl(method, L):
        vals = []
        for s in seeds:
            key = f"454m_{method}_{total_tokens//1_000_000}M_seed{s}"
            r = all_results.get(key, {})
            v = r.get("ppl", {}).get(str(L))
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    header = f"  {'Method':>12s}" + "".join(f" {'L='+str(L):>8s}" for L in EVAL_LENGTHS)
    print(header)
    print("  " + "-" * 75)
    for m in ["geo", "evq4.0"]:
        line = f"  {m:>12s}"
        for L in EVAL_LENGTHS:
            v = mean_ppl(m, L)
            line += f" {v:>8.1f}" if v else f" {'--':>8s}"
        print(line)

    print(f"\n  DELTA EVQ vs Geo:")
    line = f"  {'evq4.0':>12s}:"
    for L in EVAL_LENGTHS:
        geo = mean_ppl("geo", L)
        evq = mean_ppl("evq4.0", L)
        if geo and evq:
            line += f" {(evq/geo-1)*100:>+7.1f}%"
        else:
            line += f" {'--':>8s}"
    print(line)

    # Cross-token comparison with Phase 11 (100M)
    print(f"\n  CROSS-TOKEN COMPARISON (100M → {total_tokens//1_000_000}M):")
    phase11_path = Path("/root/autodl-tmp/evq_phase11_L256/all_results.json")
    if phase11_path.exists():
        with open(phase11_path) as f:
            p11 = json.load(f)

        def p11_mean(method, L):
            vals = []
            for s in [42, 137, 256]:
                r = p11.get(f"350m_{method}_seed{s}", {})
                v = r.get("ppl", {}).get(str(L))
                if v is not None:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None

        for m in ["geo", "evq4.0"]:
            line = f"  {m+' 100M':>16s}:"
            for L in EVAL_LENGTHS:
                v = p11_mean(m, L)
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)
            line = f"  {m+f' {total_tokens//1_000_000}M':>16s}:"
            for L in EVAL_LENGTHS:
                v = mean_ppl(m, L)
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)
            # Delta
            line = f"  {'Δ%':>16s}:"
            for L in EVAL_LENGTHS:
                old = p11_mean(m, L)
                new = mean_ppl(m, L)
                if old and new:
                    line += f" {(new/old-1)*100:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)
            print()

    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
