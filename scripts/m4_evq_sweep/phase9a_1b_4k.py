#!/usr/bin/env python3
"""
Phase 9A Group 1: 1B model, L_train=4096, base=500K, 500M tokens, seed=42.
Runs: Geometric, Hybrid τ=1.0 r=16, EVQ τ=1.0
τ*=64/√4096=1.0, matching 350M experiments exactly.

Includes: PPL eval, single-needle passkey, multi-needle passkey (5 needles).
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
    GPT, evq_cosh_inv_freq, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, set_seed, load_data, load_val,
    get_batch_from_data, maybe_wrap_with_passkey_mix, resolve_passkey_mix_ratio,
)
from eval_passkey_scratch import eval_passkey_nll_gap
from eval_multi_needle import eval_multi_needle_passkey

# ── Configuration ─────────────────────────────────────────
BASE      = 500_000.0
DIM       = 64
SEED      = 42
TOKENS    = 2_000_000_000
SEQ_LEN   = 4096        # L_train=4096 for Group 1
TAU       = 1.0          # τ*=64/√4096=1.0
PASSKEY_MIX_RATIO = resolve_passkey_mix_ratio(default=0.03)

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
EVAL_CHUNKS  = 10
PK_LENGTHS   = [1024, 2048, 4096, 8192]
PK_TRIALS    = 100

# Multi-needle config
MN_LENGTHS = [2048, 4096, 8192, 16384]
MN_NEEDLES = 5
MN_TRIALS  = 20

WORK = Path("/root/autodl-tmp/evq_phase9")
DATA_CACHE_DIR = WORK / "data"

# 1B model config (1713.7M params with SwiGLU)
CFG_1B = dict(
    vocab_size=50304,
    hidden_size=2048,
    num_layers=24,
    num_heads=32,       # 32 × 64 = 2048
    head_dim=64,
    intermediate_size=8192,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=3e-4,
    batch_size=4,       # 4K safety: halve batch to reduce VRAM
    micro_batch_size=4,
    grad_accum=1,
)


# ── Frequency generators ─────────────────────────────────
def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2*i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=TAU, r=16):
    n = dim // 2
    geo = torch.tensor(
        [1.0 / (base ** (2*i / dim)) for i in range(n)],
        dtype=torch.float64,
    )
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


# ── Training with gradient accumulation ───────────────────
def train_model_ga(model, data, cfg, seed=42):
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg.get("micro_batch_size", cfg["batch_size"])
    grad_accum = cfg.get("grad_accum", 1)
    effective_bs = micro_bs * grad_accum

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    total_chunks = len(data)
    steps = total_chunks // effective_bs
    warmup = int(steps * 0.02)

    print(f"  Training: micro_bs={micro_bs}, grad_accum={grad_accum}, "
          f"effective_bs={effective_bs}, steps={steps}, warmup={warmup}")

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
            )
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for a in range(grad_accum):
            chunk_idx = s * effective_bs + a * micro_bs
            batch = get_batch_from_data(data, perm[chunk_idx : chunk_idx + micro_bs]).to(DEVICE)
            ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
                loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(f"    step {s}/{steps}  loss={accum_loss:.4f}  "
                  f"lr={cur_lr:.2e}  ETA={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model


# ── Single run ────────────────────────────────────────────
def run_single(tag, inv_freq, cfg, train_data, val_data, filler, tok):
    run_dir = WORK / f"seed{SEED}" / tag
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  {tag}  (base={BASE:.0f}, 1B, L={SEQ_LEN}, 500M tokens, seed={SEED})")
    print(f"{'='*70}")

    import hashlib
    inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:12]
    print(f"  inv_freq: shape={inv_freq.shape}, max={inv_freq.max():.6f}, "
          f"min={inv_freq.min():.8f}, hash={inv_hash}")

    set_seed(SEED)
    model = GPT(cfg, inv_freq).to(DEVICE)

    t0 = time.time()
    model = train_model_ga(model, train_data, cfg, seed=SEED)
    train_sec = time.time() - t0
    print(f"  Training: {train_sec/60:.1f} min")

    # Save model weights
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.cpu().numpy())
    print(f"  Model saved to {run_dir / 'model.pt'}")

    # PPL eval
    ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))
    print(f"  PPL@16K = {ppl_16k}")

    # Single-needle passkey eval
    print(f"  Single-needle passkey eval ({PK_TRIALS} trials/length)...")
    pk = eval_passkey_nll_gap(model, tok, filler, lengths=PK_LENGTHS, depths=[0.5], num_trials=PK_TRIALS)
    g = pk.get("global", {})
    ret = g.get("retrieval_rate", 0)
    gap = g.get("mean_nll_gap", 0)
    print(f"  Single-needle: Retrieval = {ret:.4f},  NLL gap = {gap:.4f}")

    # Multi-needle passkey eval
    print(f"  Multi-needle passkey eval ({MN_NEEDLES} needles, {MN_TRIALS} trials)...")
    mn = eval_multi_needle_passkey(model, tok, filler, lengths=MN_LENGTHS,
                                    n_needles=MN_NEEDLES, num_trials=MN_TRIALS, seed=SEED)
    mn_g = mn.get("global", {})
    mn_per = mn_g.get("per_needle_retrieval", 0)
    mn_all = mn_g.get("all_needle_retrieval", 0)
    print(f"  Multi-needle: per_needle={mn_per:.4f}, all_needle={mn_all:.4f}")

    result = dict(
        method=tag, base=BASE, seed=SEED, tokens=TOKENS,
        model="1B", seq_len=SEQ_LEN,
        retrieval=ret, mean_nll_gap=gap,
        ppl=ppl, ppl_16k=ppl_16k,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        multi_needle_global=mn.get("global", {}),
        multi_needle_by_length=mn.get("by_length", {}),
        multi_needle_by_position=mn.get("by_needle_position", {}),
        train_sec=round(train_sec, 1),
        config=dict(
            passkey_mix_ratio=PASSKEY_MIX_RATIO,
            hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            lr=cfg["lr"], effective_bs=cfg["batch_size"],
            micro_bs=cfg.get("micro_batch_size"), grad_accum=cfg.get("grad_accum"),
        ),
    )
    save_json(result_file, result)

    del model
    torch.cuda.empty_cache()
    return result


# ── Main ──────────────────────────────────────────────────
def main():
    print("#" * 70)
    print(f"  Phase 9A Group 1: 1B, L=4096, tau=1.0 (base=500K, 500M tokens)")
    print("#" * 70)

    cfg = CFG_1B.copy()

    n_chunks = TOKENS // SEQ_LEN
    n_steps = n_chunks // cfg["batch_size"]
    print(f"\n  Config: micro_bs={cfg['micro_batch_size']}, accum={cfg['grad_accum']}, "
          f"effective_bs={cfg['batch_size']}, chunks={n_chunks}, steps={n_steps}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nLoading training data (500M tokens, seq=4096)...")
    train_data = load_data(tok, TOKENS, SEQ_LEN, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    print("Loading validation data...")
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    train_data = maybe_wrap_with_passkey_mix(
        train_data=train_data,
        filler_tokens=filler,
        tokenizer=tok,
        seq_len=SEQ_LEN,
        passkey_ratio=PASSKEY_MIX_RATIO,
    )

    # ── Three runs: Geo first, then Hybrid (fastest comparison), then EVQ ──
    runs = [
        ("geo_1b_4k",              geometric_inv_freq()),
        ("hybrid1.0_r16_1b_4k",   hybrid_evq_inv_freq(DIM, BASE, TAU, r=16)),
        ("evq1.0_1b_4k",          evq_cosh_inv_freq(DIM, TAU, BASE)),
    ]

    results = {}
    for tag, inv_freq in runs:
        r = run_single(tag, inv_freq, cfg, train_data, val_data, filler, tok)
        results[tag] = r

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 9A GROUP 1: 1B, L=4096, base=500K, 500M tokens, seed=42")
    print(f"{'='*70}")

    geo_r = results["geo_1b_4k"]
    geo_g = geo_r.get("passkey_global", {})
    geo_ret = geo_g.get("retrieval_rate", geo_r.get("retrieval", 0))
    geo_ppl = geo_r.get("ppl_16k") or geo_r.get("ppl", {}).get("16384", 0)
    geo_mn = geo_r.get("multi_needle_global", {})
    geo_mn_all = geo_mn.get("all_needle_retrieval", 0)

    print(f"  {'Method':<28s} {'Ret':>8s} {'PPL@16K':>9s} {'MN-all':>8s} {'NLL gap':>9s} {'vs Geo ret':>12s} {'vs Geo PPL':>12s}")
    print(f"  {'-'*92}")

    for tag, r in results.items():
        g = r.get("passkey_global", {})
        ret = g.get("retrieval_rate", r.get("retrieval", 0))
        ppl = r.get("ppl_16k") or r.get("ppl", {}).get("16384", 0)
        gap = g.get("mean_nll_gap", r.get("mean_nll_gap", 0))
        mn_g = r.get("multi_needle_global", {})
        mn_all = mn_g.get("all_needle_retrieval", 0)

        if tag == "geo_1b_4k":
            ret_d, ppl_d = "baseline", "baseline"
        else:
            ret_d = f"{(ret/geo_ret - 1)*100:+.1f}%" if geo_ret else "--"
            ppl_d = f"{(ppl/geo_ppl - 1)*100:+.1f}%" if geo_ppl else "--"

        print(f"  {tag:<28s} {ret:8.4f} {ppl:9.1f} {mn_all:8.4f} {gap:9.4f} {ret_d:>12s} {ppl_d:>12s}")

    # 350M reference
    print(f"\n  --- 350M reference (Phase 8F, base=500K, 4-seed mean) ---")
    print(f"  Geometric 350M:  retrieval=0.735, PPL@16K=175.7")
    print(f"  EVQ 1.0 350M:    retrieval=0.706, PPL@16K=193.9")
    print(f"  Hybrid 1.0 350M: retrieval=0.709, PPL@16K=177.0")

    # Verdict
    evq_r = results.get("evq1.0_1b_4k", {})
    hyb_r = results.get("hybrid1.0_r16_1b_4k", {})
    evq_g = evq_r.get("passkey_global", {})
    hyb_g = hyb_r.get("passkey_global", {})
    evq_ret = evq_g.get("retrieval_rate", evq_r.get("retrieval", 0))
    hyb_ret = hyb_g.get("retrieval_rate", hyb_r.get("retrieval", 0))

    hyb_beats = hyb_ret > geo_ret
    evq_beats = evq_ret > geo_ret

    print(f"\n  Verdict:")
    if geo_ret:
        print(f"    EVQ τ=1.0 vs Geo:    {(evq_ret/geo_ret-1)*100:+.1f}%  {'WIN' if evq_beats else 'LOSS'}")
        print(f"    Hybrid τ=1.0 vs Geo: {(hyb_ret/geo_ret-1)*100:+.1f}%  {'WIN' if hyb_beats else 'LOSS'}")

    # Multi-needle comparison
    hyb_mn = hyb_r.get("multi_needle_global", {})
    evq_mn = evq_r.get("multi_needle_global", {})
    print(f"\n  Multi-needle (5 needles):")
    print(f"    Geo:    all_needle={geo_mn.get('all_needle_retrieval', 0):.4f}")
    print(f"    Hybrid: all_needle={hyb_mn.get('all_needle_retrieval', 0):.4f}")
    print(f"    EVQ:    all_needle={evq_mn.get('all_needle_retrieval', 0):.4f}")

    summary = dict(
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        phase="9A-group1", model="1B", base=BASE, seed=SEED, tokens=TOKENS,
        seq_len=SEQ_LEN, tau=TAU,
        config=cfg,
        results={k: {
            "retrieval": v.get("passkey_global", {}).get("retrieval_rate", v.get("retrieval", 0)),
            "ppl_16k": v.get("ppl_16k") or v.get("ppl", {}).get("16384"),
            "mean_nll_gap": v.get("passkey_global", {}).get("mean_nll_gap", v.get("mean_nll_gap", 0)),
            "multi_needle": v.get("multi_needle_global", {}),
            "ppl": v.get("ppl", {}),
            "train_sec": v.get("train_sec"),
        } for k, v in results.items()},
        evq_beats_geo=evq_beats,
        hybrid_beats_geo=hyb_beats,
    )
    save_json(WORK / "phase9a_group1_summary.json", summary)
    print(f"\n  Saved: {WORK}/phase9a_group1_summary.json")


if __name__ == "__main__":
    main()
