#!/usr/bin/env python3
"""
Phase 11E: Continued Pretraining — Geo→EVQ Frequency Retrofit

Takes existing 454M Geo checkpoint (L_train=256, 100M tokens) and continues
pretraining at L=2048 with more tokens:
  Fork A (control):    Continue training with Geo inv_freq
  Fork B (experiment): Swap inv_freq to EVQ τ*, continue training

Both forks train for the same number of additional tokens on identical data.
Tests whether EVQ frequency allocation helps during context-length extension.

Usage:
  python phase11e_continued_pretrain.py --seeds 42
  python phase11e_continued_pretrain.py --seeds 42,123 --tokens 200000000 --tau 1.414
  python phase11e_continued_pretrain.py --seeds 42 --tau 1.414,2.0,4.0  # multi-tau sweep
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
    evq_cosh_inv_freq,
)

# ── Config ───────────────────────────────────────────────────────────

WORK = Path("/root/autodl-tmp/evq_phase11e_continued")

# Checkpoint sources — Phase 11 454M Geo (L_train=256, 100M tokens)
CKPT_DIR = Path("/root/autodl-tmp/evq_phase11_L256")

# Data sources — Phase 9 2B token dataset at L=2048
DATA_CANDIDATES = [
    Path("/root/autodl-tmp/evq_phase9/data/train_fineweb-edu_2000000000_2048.pt"),
]
VAL_CANDIDATES = [
    Path("/root/autodl-tmp/evq_phase9/data/val_fineweb-edu_5000000.pt"),
    Path("/root/autodl-tmp/evq_phase11_L256/val_fineweb-edu_5000000.pt"),
]

SEQ_LEN = 2048
BASE = 500_000.0
DIM = 64

CFG_454M = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    # Continued pretraining defaults
    train_tokens=200_000_000,
    lr=1e-4,            # Lower than original 3e-4
    batch_size=36,       # Effective batch size
    micro_batch_size=18, # For R6000 96GB (85GB peak)
    grad_accum=2,        # effective bs = 36
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


# ── Data loading ────────────────────────────────────────────────────

def find_first_existing(candidates, label="file"):
    for p in candidates:
        if p.exists():
            print(f"  Found {label}: {p}")
            return p
    raise FileNotFoundError(
        f"No {label} found. Tried:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def load_train_data(max_tokens=None):
    """Load training data. If max_tokens specified, truncate."""
    path = find_first_existing(DATA_CANDIDATES, "train data")
    data = torch.load(path, weights_only=True)
    print(f"  Raw train data: {data.shape} ({data.numel()/1e6:.0f}M tokens)")

    # If we have a large flat file, re-chunk to SEQ_LEN
    if data.dim() == 1:
        n = len(data) // SEQ_LEN
        data = data[:n * SEQ_LEN].reshape(n, SEQ_LEN)

    if max_tokens and data.numel() > max_tokens:
        n_seq = max_tokens // SEQ_LEN
        data = data[:n_seq]
        print(f"  Truncated to: {data.shape} ({data.numel()/1e6:.0f}M tokens)")

    return data


def load_val_data():
    """Load flat validation data for PPL eval."""
    path = find_first_existing(VAL_CANDIDATES, "val data")
    data = torch.load(path, weights_only=True)
    if data.dim() > 1:
        data = data.reshape(-1)
    print(f"  Val data: {data.shape} ({data.numel()/1e6:.1f}M tokens)")
    return data


def find_checkpoint(seed):
    """Find a Geo checkpoint for the given seed."""
    ckpt = CKPT_DIR / f"350m_geo_seed{seed}" / "model.pt"
    if ckpt.exists():
        print(f"  Found checkpoint: {ckpt}")
        return ckpt
    raise FileNotFoundError(f"No Geo checkpoint for seed={seed} at {ckpt}")


# ── Model loading with frequency swap ───────────────────────────────

def load_model_with_freq(cfg, ckpt_path, inv_freq):
    """Load checkpoint weights with a specified inv_freq.

    Strips rope buffers from the checkpoint state_dict so the model
    retains the provided inv_freq (geo or evq).
    """
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)

    # Remove rope buffers — we want to keep our desired inv_freq
    rope_keys = [k for k in state if ".rope." in k]
    for k in rope_keys:
        del state[k]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    # missing will include rope buffers, which is expected
    rope_missing = [k for k in missing if ".rope." in k]
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING: non-rope missing keys: {other_missing}")
    if rope_missing:
        print(f"  OK: {len(rope_missing)} rope buffer keys use new inv_freq")

    return model


# ── Training ─────────────────────────────────────────────────────────

def train_continued(model, train_data, cfg, seed=42, save_dir=None):
    """Continue training with cosine LR schedule.

    Handles epoch wrapping when train_tokens > available data.
    """
    set_seed(seed)
    total_tokens = cfg["train_tokens"]
    micro_bs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum"]
    effective_bs = micro_bs * grad_accum
    lr = cfg["lr"]
    min_lr = lr * 0.1

    tokens_per_step = effective_bs * SEQ_LEN
    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(200, max(1, total_steps // 20))

    print(f"  Training: {total_tokens/1e6:.0f}M tokens, "
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

    ckpt_fractions = [0.5, 1.0]  # Save at 50% and 100%
    next_ckpt_idx = 0

    for step in range(1, total_steps + 1):
        # LR schedule: linear warmup + cosine decay
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
            # Epoch wrapping
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

        # Logging
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tps = (step * tokens_per_step) / elapsed
            print(f"    step {step:>5d}/{total_steps} ({step/total_steps*100:5.1f}%) | "
                  f"loss={accum_loss:.4f} | lr={cur_lr:.2e} | "
                  f"epoch={epoch} | {tps/1e6:.2f}M tok/s | {elapsed:.0f}s")

        # Checkpointing
        if save_dir and next_ckpt_idx < len(ckpt_fractions):
            frac = ckpt_fractions[next_ckpt_idx]
            if step >= int(total_steps * frac):
                ckpt_path = save_dir / f"model_{int(frac*100)}pct.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"    SAVED checkpoint: {ckpt_path}")
                next_ckpt_idx += 1

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f}min), "
          f"{epoch} epochs over data")
    return elapsed


# ── Evaluation ───────────────────────────────────────────────────────

def eval_ppl(model, val_data, eval_lengths=EVAL_LENGTHS, n_chunks=EVAL_CHUNKS):
    """Evaluate perplexity at multiple sequence lengths."""
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
        description="Phase 11E: Continued Pretraining — Geo vs EVQ Retrofit"
    )
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds (must match existing Geo checkpoints)")
    parser.add_argument("--tokens", type=int, default=200_000_000,
                        help="Additional training tokens per fork (default: 200M)")
    parser.add_argument("--tau", default="1.414",
                        help="Comma-separated tau values for EVQ (default: 1.414 = d/√L)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for continued pretraining")
    parser.add_argument("--micro_batch", type=int, default=18,
                        help="Micro batch size (default: 18 for R6000 96GB)")
    parser.add_argument("--skip_geo", action="store_true",
                        help="Skip Geo fork (only train EVQ forks)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    taus = [float(t) for t in args.tau.split(",")]

    cfg = CFG_454M.copy()
    cfg["train_tokens"] = args.tokens
    cfg["lr"] = args.lr
    cfg["micro_batch_size"] = args.micro_batch
    cfg["grad_accum"] = cfg["batch_size"] // args.micro_batch

    WORK.mkdir(parents=True, exist_ok=True)

    # ── Header ──
    sep = "=" * 70
    print(sep)
    print("  Phase 11E: Continued Pretraining — Geo→EVQ Frequency Retrofit")
    print(sep)
    print(f"  Seeds: {seeds}")
    print(f"  Tokens: {args.tokens/1e6:.0f}M per fork")
    print(f"  EVQ tau: {taus}")
    print(f"  LR: {args.lr:.1e}")
    print(f"  Batch: {cfg['batch_size']} (micro={cfg['micro_batch_size']}×ga={cfg['grad_accum']})")
    print(f"  Eval lengths: {EVAL_LENGTHS}")
    print()

    # ── Load data ──
    print("[1] Loading data...")
    train_data = load_train_data()
    val_data = load_val_data()
    print()

    # ── Pre-compute inv_freqs ──
    geo_freq = geometric_inv_freq()
    evq_freqs = {}
    for tau in taus:
        evq_freqs[tau] = evq_cosh_inv_freq(head_dim=DIM, tau=tau, base=BASE)

    # ── Build method list ──
    methods = []
    if not args.skip_geo:
        methods.append(("geo", geo_freq))
    for tau in taus:
        methods.append((f"evq{tau:.3f}", evq_freqs[tau]))

    all_results = {}

    for seed in seeds:
        ckpt_path = find_checkpoint(seed)
        print()

        for method_name, inv_freq in methods:
            run_id = f"cont_{method_name}_seed{seed}"
            run_dir = WORK / run_id
            result_path = run_dir / "result.json"

            # Skip if already done
            if result_path.exists():
                print(f"\n  SKIP {run_id} (result exists)")
                with open(result_path) as f:
                    all_results[run_id] = json.load(f)
                continue

            print(f"\n{sep}")
            print(f"  RUN: {run_id}")
            freq_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:12]
            print(f"  Base checkpoint: {ckpt_path}")
            print(f"  inv_freq: min={inv_freq.min():.8f} max={inv_freq.max():.6f} hash={freq_hash}")
            print(sep)

            # Load model with desired frequency
            print(f"\n  [1] Loading model with {method_name} frequencies...")
            model = load_model_with_freq(cfg, ckpt_path, inv_freq)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Model: {n_params:.1f}M params")

            # Quick PPL before training (sanity check)
            print(f"\n  [2] Pre-training PPL (sanity check)...")
            ppl_before = eval_ppl(model, val_data,
                                  eval_lengths=[2048, 4096])

            # Train
            print(f"\n  [3] Continued pretraining ({args.tokens/1e6:.0f}M tokens)...")
            run_dir.mkdir(parents=True, exist_ok=True)
            train_time = train_continued(model, train_data, cfg,
                                         seed=seed, save_dir=run_dir)

            # Save final model
            torch.save(model.state_dict(), run_dir / "model.pt")

            # Evaluate
            print(f"\n  [4] Post-training PPL evaluation...")
            ppl_after = eval_ppl(model, val_data)

            result = {
                "run_id": run_id,
                "method": method_name,
                "seed": seed,
                "base_checkpoint": str(ckpt_path),
                "train_tokens": args.tokens,
                "lr": args.lr,
                "ppl_before": ppl_before,
                "ppl_after": ppl_after,
                "train_time_sec": round(train_time, 1),
                "inv_freq_hash": freq_hash,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results[run_id] = result
            print(f"  Result saved: {result_path}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  PHASE 11E SUMMARY: Continued Pretraining Comparison")
    print(f"{'=' * 80}")

    # Per-method PPL table
    method_names = [m[0] for m in methods]
    print(f"\n  POST-TRAINING PPL (per seed):")
    header = f"  {'Method':>16s} {'Seed':>6s}"
    for L in EVAL_LENGTHS:
        header += f" {'L='+str(L):>8s}"
    print(header)
    print("  " + "-" * (24 + 9 * len(EVAL_LENGTHS)))

    for seed in seeds:
        for mname in method_names:
            run_id = f"cont_{mname}_seed{seed}"
            r = all_results.get(run_id, {})
            ppl = r.get("ppl_after", {})
            line = f"  {mname:>16s} {seed:>6d}"
            for L in EVAL_LENGTHS:
                v = ppl.get(str(L))
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

    # Mean across seeds
    if len(seeds) > 1:
        print(f"\n  MEAN PPL (across {len(seeds)} seeds):")
        header = f"  {'Method':>16s}"
        for L in EVAL_LENGTHS:
            header += f" {'L='+str(L):>8s}"
        print(header)
        print("  " + "-" * (16 + 9 * len(EVAL_LENGTHS)))

        mean_ppls = {}
        for mname in method_names:
            line = f"  {mname:>16s}"
            mean_ppls[mname] = {}
            for L in EVAL_LENGTHS:
                vals = []
                for seed in seeds:
                    r = all_results.get(f"cont_{mname}_seed{seed}", {})
                    v = r.get("ppl_after", {}).get(str(L))
                    if v is not None:
                        vals.append(v)
                if vals:
                    m = sum(vals) / len(vals)
                    mean_ppls[mname][str(L)] = m
                    line += f" {m:>8.1f}"
                else:
                    line += f" {'--':>8s}"
            print(line)
    else:
        mean_ppls = {}
        for mname in method_names:
            mean_ppls[mname] = {}
            r = all_results.get(f"cont_{mname}_seed{seeds[0]}", {})
            ppl = r.get("ppl_after", {})
            for L in EVAL_LENGTHS:
                v = ppl.get(str(L))
                if v is not None:
                    mean_ppls[mname][str(L)] = v

    # Delta table
    if "geo" in mean_ppls:
        print(f"\n  DELTA vs Geo (continued pretraining):")
        for mname in method_names:
            if mname == "geo":
                continue
            line = f"  {mname:>16s}:"
            for L in EVAL_LENGTHS:
                geo_v = mean_ppls.get("geo", {}).get(str(L))
                evq_v = mean_ppls.get(mname, {}).get(str(L))
                if geo_v and evq_v:
                    delta = (evq_v / geo_v - 1) * 100
                    line += f" {delta:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)

    # Before vs After comparison
    print(f"\n  PRE vs POST training PPL (L=2048, in-distribution):")
    for seed in seeds:
        for mname in method_names:
            run_id = f"cont_{mname}_seed{seed}"
            r = all_results.get(run_id, {})
            before = r.get("ppl_before", {}).get("2048")
            after = r.get("ppl_after", {}).get("2048")
            if before and after:
                delta = (after / before - 1) * 100
                print(f"  {run_id:>30s}: {before:.1f} → {after:.1f} ({delta:+.1f}%)")

    # Save aggregate
    agg_path = WORK / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results saved to {agg_path}")


if __name__ == "__main__":
    main()
