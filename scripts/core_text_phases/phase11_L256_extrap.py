#!/usr/bin/env python3
"""
Phase 11: 350M L=256 Extrapolation Sweep
Train at L=256, eval extrapolation to 512/1K/2K/4K/8K (2×-32×).
Compare Geometric vs EVQ(τ=2.0) vs EVQ(τ=4.0).
τ*=d_head/√L_train=64/√256=4.0
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

# ── Config ──────────────────────────────────────────────────────────────
BASE = 500_000.0
DIM = 64  # d_head
SEQ_LEN = 256
TOKENS = 200_000_000  # 200M tokens — enough for 350M to converge at L=256

EVAL_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
EVAL_CHUNKS = 8

WORK = Path("/root/autodl-tmp/evq_phase11_L256")
DATA_CACHE_DIR = WORK / "data"

CFG_350M = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=3e-4,
    batch_size=256,       # huge batch, L=256 is very short
    micro_batch_size=64,  # fits easily in 102GB
    grad_accum=4,         # 64*4=256 effective batch
)


# ── Frequency builders ──────────────────────────────────────────────────
def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def evq_cosh_inv_freq(dim=DIM, tau=4.0, base=BASE):
    """Full EVQ-Cosh (no hybrid split, all dims EVQ)."""
    if abs(tau) < 1e-8:
        return geometric_inv_freq(dim, base)
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    freqs = 1.0 / (base ** phi)
    return freqs.float()


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=4.0, r=16):
    """Hybrid: top-r dims geometric, rest EVQ-Cosh."""
    n = dim // 2
    geo = torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float64)
    n_evq = n - r
    if n_evq <= 0:
        return geo.float()
    theta_max = geo[r].item()
    theta_min = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()


# ── Existing data paths (from Phase 9) ──────────────────────────────────
PHASE9_DATA = Path("/root/autodl-tmp/evq_phase9/data")
PHASE9_TRAIN = PHASE9_DATA / "train_fineweb-edu_2000000000_2048.pt"
PHASE9_VAL = PHASE9_DATA / "val_fineweb-edu_5000000.pt"


# ── Data ────────────────────────────────────────────────────────────────
def load_train_data(seq_len=SEQ_LEN, max_tokens=TOKENS + 5_000_000):
    """Load training data, re-chunking from Phase 9's 2048-len cache."""
    cache_path = DATA_CACHE_DIR / f"train_fineweb-edu_{max_tokens}_{seq_len}.pt"
    if cache_path.exists():
        print(f"  Loading cached data: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    # Re-chunk from Phase 9 data (976562 × 2048 = 2B tokens)
    print(f"  Loading Phase 9 data from {PHASE9_TRAIN}...")
    data = torch.load(PHASE9_TRAIN, weights_only=True)
    flat = data.reshape(-1)
    # Take only what we need
    need_tokens = max_tokens
    flat = flat[:need_tokens]
    n_samples = len(flat) // seq_len
    result = flat[:n_samples * seq_len].reshape(n_samples, seq_len)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    print(f"  Re-chunked: {data.shape} -> {result.shape} ({result.numel()/1e6:.0f}M tokens)")
    del data, flat
    gc.collect()
    return result


def load_validation_data():
    """Load flat validation data for PPL eval (from Phase 9)."""
    print(f"  Loading Phase 9 val data from {PHASE9_VAL}...")
    return torch.load(PHASE9_VAL, weights_only=True)


# ── Eval ────────────────────────────────────────────────────────────────
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
        offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
        for offset in offsets:
            chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           chunk[:, 1:].reshape(-1))
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM, skipping")
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del chunk
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L:>5d}: PPL={ppl:.2f}  ({len(losses)} chunks)")
    return results


# ── Training ────────────────────────────────────────────────────────────
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

    print(f"  Training: {total_tokens/1e6:.0f}M tokens, bs={bs} (micro={mbs}×ga={ga}), "
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
        # Warmup LR
        if step <= warmup_steps:
            warmup_lr = lr * step / warmup_steps
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(ga):
            # Get micro-batch
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
            cur_lr = opt.param_groups[0]["lr"]
            pct = step / total_steps * 100
            print(f"    step {step:>6d}/{total_steps} ({pct:5.1f}%) | "
                  f"loss={accum_loss:.4f} | lr={cur_lr:.2e} | "
                  f"{tps/1e6:.2f}M tok/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return elapsed


# ── Main ────────────────────────────────────────────────────────────────
def run_one(method_name, inv_freq, seed, train_data, val_data, cfg):
    """Train and evaluate one run."""
    run_id = f"350m_{method_name}_seed{seed}"
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

    # Train
    train_time = train_model(model, train_data, cfg, seed=seed)

    # Save model
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Eval PPL
    print(f"\n  Evaluating PPL...")
    ppl = eval_ppl(model, val_data)

    result = {
        "run_id": run_id,
        "method": method_name,
        "seed": seed,
        "ppl": ppl,
        "train_time_sec": round(train_time, 1),
        "inv_freq_hash": freq_hash,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137,256", help="Comma-separated seeds")
    parser.add_argument("--methods", default="geo,evq2.0,evq4.0",
                        help="Methods: geo, evq2.0, evq4.0, hybrid2.0, hybrid4.0")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_batch", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--r", type=int, default=16, help="Hybrid r (geo dims to keep)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    cfg = CFG_350M.copy()
    cfg["train_tokens"] = args.tokens
    cfg["batch_size"] = args.batch_size
    cfg["micro_batch_size"] = args.micro_batch
    cfg["grad_accum"] = args.batch_size // args.micro_batch
    cfg["seq_len"] = args.seq_len
    cfg["max_position_embeddings"] = args.seq_len

    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 11: L={args.seq_len} Extrapolation Sweep")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Tokens: {args.tokens/1e6:.0f}M")
    print(f"  Batch: {args.batch_size} (micro={args.micro_batch} × ga={cfg['grad_accum']})")
    print(f"  Eval lengths: {EVAL_LENGTHS}")

    # Load data
    print("\n[1] Loading data...")
    train_data = load_train_data(seq_len=args.seq_len,
                                  max_tokens=args.tokens + 5_000_000)
    val_data = load_validation_data()
    print(f"  Train: {train_data.shape}, Val: {val_data.shape}")

    # Build inv_freq for each method
    inv_freq_map = {}
    for m in methods:
        if m == "geo":
            inv_freq_map[m] = geometric_inv_freq()
        elif m.startswith("evq"):
            tau = float(m.replace("evq", ""))
            inv_freq_map[m] = evq_cosh_inv_freq(tau=tau)
        elif m.startswith("hybrid"):
            tau = float(m.replace("hybrid", ""))
            inv_freq_map[m] = hybrid_evq_inv_freq(tau=tau, r=args.r)
        else:
            raise ValueError(f"Unknown method: {m}")

    # Run all
    all_results = {}
    for method in methods:
        for seed in seeds:
            result = run_one(method, inv_freq_map[method], seed,
                           train_data, val_data, cfg)
            all_results[result["run_id"]] = result

    # Save aggregate
    agg_path = WORK / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  PHASE 11 RESULTS: L_train={args.seq_len}, 350M")
    print(f"{'='*90}")

    for method in methods:
        method_results = {s: all_results.get(f"350m_{method}_seed{s}", {})
                         for s in seeds}
        valid = [r for r in method_results.values() if r.get("ppl")]
        if not valid:
            continue

        print(f"\n  {method.upper():>10s}:")
        header = "    " + " ".join(f"{'L='+str(L):>10s}" for L in EVAL_LENGTHS)
        print(header)

        for seed in seeds:
            r = method_results.get(seed, {})
            ppl = r.get("ppl", {})
            vals = " ".join(f"{ppl.get(str(L), 0):>10.1f}" for L in EVAL_LENGTHS)
            print(f"    seed={seed:>3d}: {vals}")

        # Mean
        means = []
        for L in EVAL_LENGTHS:
            ppls = [method_results[s]["ppl"].get(str(L), 0)
                    for s in seeds if method_results[s].get("ppl", {}).get(str(L))]
            means.append(sum(ppls) / len(ppls) if ppls else 0)
        mean_str = " ".join(f"{m:>10.1f}" for m in means)
        print(f"    {'MEAN':>8s}: {mean_str}")

    # Delta table
    geo_results = {s: all_results.get(f"350m_geo_seed{s}", {}) for s in seeds}
    geo_valid = any(r.get("ppl") for r in geo_results.values())

    if geo_valid:
        print(f"\n  DELTA vs Geometric:")
        for method in methods:
            if method == "geo":
                continue
            print(f"    {method.upper():>10s}: ", end="")
            for L in EVAL_LENGTHS:
                geo_ppls = [geo_results[s]["ppl"].get(str(L), 0)
                           for s in seeds if geo_results[s].get("ppl", {}).get(str(L))]
                evq_ppls = [all_results.get(f"350m_{method}_seed{s}", {}).get("ppl", {}).get(str(L), 0)
                           for s in seeds
                           if all_results.get(f"350m_{method}_seed{s}", {}).get("ppl", {}).get(str(L))]
                if geo_ppls and evq_ppls:
                    geo_m = sum(geo_ppls) / len(geo_ppls)
                    evq_m = sum(evq_ppls) / len(evq_ppls)
                    delta = (evq_m / geo_m - 1) * 100
                    print(f"{delta:>+9.1f}%", end=" ")
                else:
                    print(f"{'N/A':>10s}", end=" ")
            print()

    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
