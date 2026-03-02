#!/usr/bin/env python3
"""
Phase 9B Pilot: ~750M model, L_train=2048, base=500K.
Runs: Geometric, Hybrid τ=1.5 r=16 (2 runs only for fast decision).
Goal: overnight A/B sanity check with lower cost.
"""

import json, math, os, sys, time
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
    eval_model, set_seed, load_data, load_val,
)
from eval_passkey_scratch import eval_passkey_nll_gap
from eval_multi_needle import eval_multi_needle_passkey

# ── Configuration ─────────────────────────────────────────
BASE      = 500_000.0
DIM       = 64
SEED      = 42
TOKENS    = 500_000_000      # fast pilot minimum
SEQ_LEN   = 2048
TAU       = 1.5

EVAL_LENGTHS = [1024, 2048, 4096, 8192, 16384]
EVAL_CHUNKS  = 6
PK_LENGTHS   = [1024, 2048, 4096, 8192]
PK_TRIALS    = 40

MN_LENGTHS = [2048, 4096, 8192]
MN_NEEDLES = 5
MN_TRIALS  = 8

WORK = Path("/root/autodl-tmp/evq_phase9")
DATA_CACHE_DIR = WORK / "data"

# ~750M: 1536 × 18L, head_dim=64, SwiGLU 4x
CFG_750M = dict(
    vocab_size=50304,
    hidden_size=1536,
    num_layers=18,
    num_heads=24,
    head_dim=64,
    intermediate_size=6144,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=3e-4,
    batch_size=32,       # effective
    micro_batch_size=16,
    grad_accum=2,
)


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2*i / dim)) for i in range(n)], dtype=torch.float32)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=TAU, r=16):
    n = dim // 2
    geo = torch.tensor([1.0 / (base ** (2*i / dim)) for i in range(n)], dtype=torch.float64)
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


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def train_model_ga(model, data, cfg, seed=42):
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg.get("micro_batch_size", cfg["batch_size"])
    grad_accum = cfg.get("grad_accum", 1)
    effective_bs = micro_bs * grad_accum

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    total_chunks = len(data)
    steps = total_chunks // effective_bs
    warmup = int(steps * 0.02)

    print(f"  Training: micro_bs={micro_bs}, grad_accum={grad_accum}, effective_bs={effective_bs}, steps={steps}, warmup={warmup}")

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for a in range(grad_accum):
            chunk_idx = s * effective_bs + a * micro_bs
            batch = data[perm[chunk_idx : chunk_idx + micro_bs]].to(DEVICE)
            ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(f"    step {s}/{steps}  loss={accum_loss:.4f}  lr={cur_lr:.2e}  ETA={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model


def run_single(tag, inv_freq, cfg, train_data, val_data, filler, tok):
    run_dir = WORK / f"seed{SEED}" / tag
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*70}\n  {tag}  (base={BASE:.0f}, 750M, L={SEQ_LEN}, {TOKENS/1e6:.0f}M tokens, seed={SEED})\n{'='*70}")

    set_seed(SEED)
    model = GPT(cfg, inv_freq).to(DEVICE)

    t0 = time.time()
    model = train_model_ga(model, train_data, cfg, seed=SEED)
    train_sec = time.time() - t0

    ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))

    print(f"  Passkey eval ({PK_TRIALS} trials/length)")
    pk = eval_passkey_nll_gap(model, tok, filler, lengths=PK_LENGTHS, depths=[0.5], num_trials=PK_TRIALS)
    g = pk.get("global", {})

    print(f"  Multi-needle eval ({MN_NEEDLES} needles, {MN_TRIALS} trials)")
    mn = eval_multi_needle_passkey(model, tok, filler, lengths=MN_LENGTHS,
                                   n_needles=MN_NEEDLES, num_trials=MN_TRIALS, seed=SEED)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.cpu().numpy())

    result = dict(
        method=tag, base=BASE, seed=SEED, tokens=TOKENS,
        model="750M", seq_len=SEQ_LEN,
        retrieval=g.get("retrieval_rate", 0), mean_nll_gap=g.get("mean_nll_gap", 0),
        ppl=ppl, ppl_16k=ppl_16k,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        multi_needle_global=mn.get("global", {}),
        multi_needle_by_length=mn.get("by_length", {}),
        train_sec=round(train_sec, 1),
        config=dict(
            hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"], lr=cfg["lr"],
            effective_bs=cfg["batch_size"], micro_bs=cfg.get("micro_batch_size"),
            grad_accum=cfg.get("grad_accum"),
        ),
    )
    save_json(result_file, result)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    print("#" * 70)
    print("  Phase 9B Pilot: 750M, L=2048, Geo vs Hybrid(τ=1.5,r=16), 500M tokens")
    print("#" * 70)

    cfg = CFG_750M.copy()
    n_chunks = TOKENS // SEQ_LEN
    n_steps = n_chunks // cfg["batch_size"]
    print(f"\n  Config: micro_bs={cfg['micro_batch_size']}, accum={cfg['grad_accum']}, effective_bs={cfg['batch_size']}, chunks={n_chunks}, steps={n_steps}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nLoading training data...")
    train_data = load_data(tok, TOKENS, SEQ_LEN, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    print("Loading validation data...")
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    runs = [
        ("geo_750m_2k_pilot", geometric_inv_freq()),
        ("hybrid1.5_r16_750m_2k_pilot", hybrid_evq_inv_freq(DIM, BASE, TAU, r=16)),
    ]

    results = {}
    for tag, inv_freq in runs:
        results[tag] = run_single(tag, inv_freq, cfg, train_data, val_data, filler, tok)

    geo = results["geo_750m_2k_pilot"]
    hyb = results["hybrid1.5_r16_750m_2k_pilot"]

    geo_ret = geo.get("passkey_global", {}).get("retrieval_rate", 0)
    hyb_ret = hyb.get("passkey_global", {}).get("retrieval_rate", 0)
    geo_ppl = geo.get("ppl_16k") or geo.get("ppl", {}).get("16384", 0)
    hyb_ppl = hyb.get("ppl_16k") or hyb.get("ppl", {}).get("16384", 0)

    print(f"\n{'='*70}")
    print("  PILOT SUMMARY (750M, 2 runs)")
    print(f"{'='*70}")
    print(f"  Geo    : ret={geo_ret:.4f}, ppl16k={geo_ppl}")
    print(f"  Hybrid : ret={hyb_ret:.4f}, ppl16k={hyb_ppl}")
    if geo_ret:
        print(f"  Hybrid vs Geo retrieval: {(hyb_ret/geo_ret - 1)*100:+.2f}%")
    if geo_ppl:
        print(f"  Hybrid vs Geo PPL@16K: {(hyb_ppl/geo_ppl - 1)*100:+.2f}%")

    summary = dict(
        phase="9B-pilot", model="750M", seq_len=SEQ_LEN, tokens=TOKENS,
        runs=["geo", "hybrid1.5_r16"], seed=SEED,
        results=results,
    )
    save_json(WORK / "phase9b_750m_2k_pilot_summary.json", summary)
    print(f"\nSaved: {WORK}/phase9b_750m_2k_pilot_summary.json")


if __name__ == "__main__":
    main()
