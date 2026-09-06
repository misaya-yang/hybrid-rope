#!/usr/bin/env python3
"""
Phase 11F: Token Scaling — 454M Geo vs EVQ τ=1.5, L=2048

Continue training existing 454M models (already trained with 100M tokens)
with 200M MORE tokens. Each method continues with its own frequencies.

Baseline: 50M tokens (Phase 9) and 100M tokens (Passkey 5%)
This adds: 300M total (100M base + 200M continued)

Shows how the Geo vs EVQ gap evolves with more training data.

Usage:
  python phase11f_token_scaling_454m.py --seeds 42
  python phase11f_token_scaling_454m.py --seeds 42,123,7 --extra_tokens 200000000
"""

import json, math, os, sys, time, gc, hashlib
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, DEVICE, DTYPE, USE_AUTOCAST,
    set_seed, get_batch_from_data,
    evq_cosh_inv_freq,
)

# ── Config ───────────────────────────────────────────────────────────

WORK = Path("/root/autodl-tmp/evq_phase11f_token_scaling")

# Base checkpoints — Passkey 5% (100M tokens, L=2048)
CKPT_DIR = Path("/root/autodl-tmp/evq_passkey_mix_5pct")

# Data
DATA_FILE = Path("/root/autodl-tmp/evq_passkey_mix_5pct/train_fineweb-edu_100000000_2048.pt")
VAL_FILE = Path("/root/autodl-tmp/evq_passkey_mix_5pct/val_fineweb-edu_5000000.pt")

SEQ_LEN = 2048
BASE = 500_000.0
DIM = 64
TAU = 1.5  # Match existing checkpoints

CFG = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
)

EVAL_LENGTHS = [2048, 4096, 8192, 16384]
EVAL_CHUNKS = 8


# ── Frequency allocation ────────────────────────────────────────────

def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


# ── Model loading ───────────────────────────────────────────────────

def load_model_from_ckpt(cfg, ckpt_path, inv_freq):
    """Load checkpoint, keeping its original inv_freq or overriding."""
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)

    # Remove rope buffers — use our inv_freq
    rope_keys = [k for k in state if ".rope." in k]
    for k in rope_keys:
        del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING: non-rope missing keys: {other_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    return model


# ── Training ─────────────────────────────────────────────────────────

def train_continued(model, train_data, extra_tokens, lr, micro_bs, grad_accum,
                    seed=42, save_dir=None):
    """Continue training with cosine LR schedule and epoch wrapping."""
    set_seed(seed)
    effective_bs = micro_bs * grad_accum
    min_lr = lr * 0.1

    tokens_per_step = effective_bs * SEQ_LEN
    total_steps = extra_tokens // tokens_per_step
    warmup_steps = min(200, max(1, total_steps // 20))

    print(f"  Training: {extra_tokens/1e6:.0f}M tokens, "
          f"bs={effective_bs} (micro={micro_bs}×ga={grad_accum}), "
          f"L={SEQ_LEN}, steps={total_steps}, lr={lr:.1e}, warmup={warmup_steps}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1,
        betas=(0.9, 0.95), fused=True,
    )
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))

    model.train()
    n_samples = len(train_data)
    perm = torch.randperm(n_samples)
    ptr = 0
    epoch = 1
    t0 = time.time()
    log_interval = max(1, total_steps // 40)

    for step in range(1, total_steps + 1):
        # LR schedule
        if step <= warmup_steps:
            cur_lr = lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            if ptr + micro_bs > n_samples:
                perm = torch.randperm(n_samples)
                ptr = 0
                epoch += 1
            indices = perm[ptr:ptr + micro_bs]
            ptr += micro_bs
            batch = get_batch_from_data(train_data, indices).to(DEVICE)

            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tps = (step * tokens_per_step) / elapsed
            print(f"    step {step:>5d}/{total_steps} ({step/total_steps*100:5.1f}%) | "
                  f"loss={accum_loss:.4f} | lr={cur_lr:.2e} | "
                  f"epoch={epoch} | {tps/1e6:.2f}M tok/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f}min), "
          f"{epoch} epochs over data")

    if save_dir:
        torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"  Model saved: {save_dir / 'model.pt'}")

    return elapsed


# ── Evaluation ───────────────────────────────────────────────────────

def eval_ppl(model, val_data, eval_lengths=EVAL_LENGTHS, n_chunks=EVAL_CHUNKS):
    model.eval()
    model.extend_rope(max(eval_lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)
    results = {}

    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            continue
        n = min(n_chunks, max(1, max_start // L))
        offsets = sorted(rng.choice(max_start, size=n, replace=False))

        for offset in offsets:
            chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        chunk[:, 1:].reshape(-1),
                    )
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM, skipping")
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del chunk
                torch.cuda.empty_cache()

        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L:>6d} ({L/SEQ_LEN:.0f}×): PPL={ppl:>8.2f}  ({len(losses)} chunks)")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 11F: Token Scaling — Geo vs EVQ continued training"
    )
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds (must match passkey_5pct checkpoints)")
    parser.add_argument("--extra_tokens", type=int, default=200_000_000,
                        help="Additional tokens beyond the 100M base (default: 200M)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for continued pretraining")
    parser.add_argument("--micro_batch", type=int, default=4,
                        help="Micro batch size (default: 4 for 5090 32GB)")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8, bs=32)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    micro_bs = args.micro_batch
    grad_accum = args.grad_accum
    effective_bs = micro_bs * grad_accum

    WORK.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(sep)
    print("  Phase 11F: Token Scaling — Geo vs EVQ τ=1.5 Continued Training")
    print(sep)
    print(f"  Base: 100M tokens (passkey 5%), adding {args.extra_tokens/1e6:.0f}M more")
    print(f"  Total effective: {(100_000_000 + args.extra_tokens)/1e6:.0f}M tokens")
    print(f"  Seeds: {seeds}")
    print(f"  LR: {args.lr:.1e}")
    print(f"  Batch: {effective_bs} (micro={micro_bs}×ga={grad_accum})")
    print()

    # ── Load data ──
    print("[1] Loading data...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Train data not found: {DATA_FILE}")
    train_data = torch.load(str(DATA_FILE), weights_only=True)
    print(f"  Train data: {train_data.shape} ({train_data.numel()/1e6:.0f}M tokens)")

    val_data = torch.load(str(VAL_FILE), weights_only=True)
    if val_data.dim() > 1:
        val_data = val_data.reshape(-1)
    print(f"  Val data: {val_data.shape} ({val_data.numel()/1e6:.1f}M tokens)")

    # ── Methods ──
    geo_freq = geometric_inv_freq()
    evq_freq = evq_cosh_inv_freq(head_dim=DIM, tau=TAU, base=BASE)

    methods = {
        "geo": ("tau0.00", geo_freq),
        "evq1.5": ("tau1.50", evq_freq),
    }

    all_results = {}

    for seed in seeds:
        for method_name, (ckpt_tau, inv_freq) in methods.items():
            run_id = f"cont_{method_name}_seed{seed}"
            run_dir = WORK / run_id
            result_path = run_dir / "result.json"

            if result_path.exists():
                print(f"\n  SKIP {run_id} (result exists)")
                with open(result_path) as f:
                    all_results[run_id] = json.load(f)
                continue

            # Find checkpoint
            ckpt_path = CKPT_DIR / f"350m_{ckpt_tau}_seed{seed}" / "model.pt"
            if not ckpt_path.exists():
                print(f"\n  SKIP {run_id} (no checkpoint: {ckpt_path})")
                continue

            print(f"\n{sep}")
            print(f"  RUN: {run_id}")
            freq_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:12]
            print(f"  Base: {ckpt_path}")
            print(f"  inv_freq hash={freq_hash}")
            print(sep)

            # Load model with its own frequency
            print(f"\n  [1] Loading {method_name} model...")
            model = load_model_from_ckpt(CFG, ckpt_path, inv_freq)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Model: {n_params:.1f}M params")

            # Pre-training eval
            print(f"\n  [2] Pre-training PPL...")
            ppl_before = eval_ppl(model, val_data, eval_lengths=[2048, 4096])

            # Continue training
            print(f"\n  [3] Continued training ({args.extra_tokens/1e6:.0f}M tokens)...")
            run_dir.mkdir(parents=True, exist_ok=True)
            train_time = train_continued(
                model, train_data, args.extra_tokens,
                lr=args.lr, micro_bs=micro_bs, grad_accum=grad_accum,
                seed=seed, save_dir=run_dir,
            )

            # Post-training eval
            print(f"\n  [4] Post-training PPL...")
            ppl_after = eval_ppl(model, val_data)

            result = {
                "run_id": run_id,
                "method": method_name,
                "seed": seed,
                "base_checkpoint": str(ckpt_path),
                "base_tokens": 100_000_000,
                "extra_tokens": args.extra_tokens,
                "total_tokens": 100_000_000 + args.extra_tokens,
                "lr": args.lr,
                "ppl_before": ppl_before,
                "ppl_after": ppl_after,
                "train_time_sec": round(train_time, 1),
                "inv_freq_hash": freq_hash,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results[run_id] = result

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  PHASE 11F SUMMARY: Token Scaling (100M → {(100_000_000+args.extra_tokens)/1e6:.0f}M)")
    print(f"{'=' * 80}")

    method_names = list(methods.keys())

    # Per-seed results
    print(f"\n  POST-TRAINING PPL:")
    header = f"  {'Method':>12s} {'Seed':>6s}"
    for L in EVAL_LENGTHS:
        header += f" {'L='+str(L):>8s}"
    print(header)
    print("  " + "-" * (20 + 9 * len(EVAL_LENGTHS)))

    for seed in seeds:
        for mname in method_names:
            run_id = f"cont_{mname}_seed{seed}"
            r = all_results.get(run_id, {})
            ppl = r.get("ppl_after", {})
            line = f"  {mname:>12s} {seed:>6d}"
            for L in EVAL_LENGTHS:
                v = ppl.get(str(L))
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

    # Mean PPL
    mean_ppls = {}
    for mname in method_names:
        mean_ppls[mname] = {}
        for L in EVAL_LENGTHS:
            vals = []
            for seed in seeds:
                r = all_results.get(f"cont_{mname}_seed{seed}", {})
                v = r.get("ppl_after", {}).get(str(L))
                if v is not None:
                    vals.append(v)
            if vals:
                mean_ppls[mname][str(L)] = sum(vals) / len(vals)

    if len(seeds) > 1:
        print(f"\n  MEAN PPL (across {len(seeds)} seeds):")
        for mname in method_names:
            line = f"  {mname:>12s}"
            for L in EVAL_LENGTHS:
                v = mean_ppls[mname].get(str(L))
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

    # Delta
    if "geo" in mean_ppls and "evq1.5" in mean_ppls:
        print(f"\n  DELTA EVQ vs Geo:")
        line = f"  {'evq1.5':>12s}:"
        for L in EVAL_LENGTHS:
            geo_v = mean_ppls["geo"].get(str(L))
            evq_v = mean_ppls["evq1.5"].get(str(L))
            if geo_v and evq_v:
                delta = (evq_v / geo_v - 1) * 100
                line += f" {delta:>+7.1f}%"
            else:
                line += f" {'--':>8s}"
        print(line)

    # Token scaling comparison
    print(f"\n  TOKEN SCALING COMPARISON (from existing baselines):")
    print(f"  {'Tokens':>10s}  {'Method':>8s}", end="")
    for L in EVAL_LENGTHS:
        print(f" {'L='+str(L):>8s}", end="")
    print()
    print("  " + "-" * (20 + 9 * len(EVAL_LENGTHS)))

    # Hardcoded baselines from Phase 9 (50M) and Passkey 5% (100M)
    baselines = {
        "50M": {
            "geo": {"2048": 86.6, "4096": 117.5, "8192": 174.8, "16384": 284.8},
            "evq1.5": {"2048": 86.9, "4096": 113.5, "8192": 157.3, "16384": 246.9},
        },
        "100M": {
            "geo": {"2048": 64.7, "4096": 98.9, "8192": 165.0, "16384": 266.8},
            "evq1.5": {"2048": 65.2, "4096": 90.2, "8192": 147.9, "16384": 235.9},
        },
    }

    for tok_label, methods_ppl in baselines.items():
        for mname in method_names:
            ppl = methods_ppl.get(mname, {})
            line = f"  {tok_label:>10s}  {mname:>8s}"
            for L in EVAL_LENGTHS:
                v = ppl.get(str(L))
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

    total_tok = (100_000_000 + args.extra_tokens) // 1_000_000
    for mname in method_names:
        line = f"  {str(total_tok)+'M':>10s}  {mname:>8s}"
        for L in EVAL_LENGTHS:
            v = mean_ppls.get(mname, {}).get(str(L))
            line += f" {v:>8.1f}" if v else f" {'--':>8s}"
        print(line)

    # Gap evolution
    print(f"\n  GAP EVOLUTION (EVQ vs Geo %):")
    print(f"  {'Tokens':>10s}", end="")
    for L in EVAL_LENGTHS:
        print(f" {'L='+str(L):>8s}", end="")
    print()
    for tok_label, methods_ppl in baselines.items():
        line = f"  {tok_label:>10s}"
        for L in EVAL_LENGTHS:
            g = methods_ppl.get("geo", {}).get(str(L))
            e = methods_ppl.get("evq1.5", {}).get(str(L))
            if g and e:
                line += f" {(e/g-1)*100:>+7.1f}%"
            else:
                line += f" {'--':>8s}"
        print(line)

    line = f"  {str(total_tok)+'M':>10s}"
    for L in EVAL_LENGTHS:
        g = mean_ppls.get("geo", {}).get(str(L))
        e = mean_ppls.get("evq1.5", {}).get(str(L))
        if g and e:
            line += f" {(e/g-1)*100:>+7.1f}%"
        else:
            line += f" {'--':>8s}"
    print(line)

    # Save
    agg_path = WORK / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
