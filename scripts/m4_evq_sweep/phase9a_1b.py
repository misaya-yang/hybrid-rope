#!/usr/bin/env python3
"""
Phase 9A: 1B model scale-up verification (base=500K, 200M tokens, seed=42).
Runs: Geometric, EVQ τ=1.0, Hybrid τ=1.0 r=16
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
)
from eval_passkey_scratch import eval_passkey_nll_gap

# ── Configuration ─────────────────────────────────────────
BASE      = 500_000.0
DIM       = 64
SEED      = 42
TOKENS    = 200_000_000

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
EVAL_CHUNKS  = 10
PK_LENGTHS   = [1024, 2048, 4096, 8192]
PK_TRIALS    = 100

WORK = Path("/root/autodl-tmp/evq_phase9")
DATA_CACHE_DIR = WORK / "data"

# 1B model config
CFG_1B = dict(
    vocab_size=50304,
    hidden_size=2048,
    num_layers=24,
    num_heads=32,       # 32 × 64 = 2048
    head_dim=64,
    intermediate_size=8192,
    max_position_embeddings=4096,
    seq_len=4096,
    train_tokens=TOKENS,
    lr=3e-4,
    batch_size=8,           # 200M/4096/8 = 6103 steps, same as 350M!
    micro_batch_size=8,
    grad_accum=1,
)


# ── Frequency generators ─────────────────────────────────
def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2*i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=1.0, r=16):
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
    """Train with gradient accumulation support."""
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg.get("micro_batch_size", cfg["batch_size"])
    grad_accum = cfg.get("grad_accum", 1)
    effective_bs = micro_bs * grad_accum

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Total optimizer steps (each step processes effective_bs chunks)
    total_chunks = len(data)
    steps = total_chunks // effective_bs
    warmup = int(steps * 0.02)

    print(f"  Training config: micro_bs={micro_bs}, grad_accum={grad_accum}, "
          f"effective_bs={effective_bs}, steps={steps}, warmup={warmup}")

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        # Cosine LR with warmup
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
            batch = data[perm[chunk_idx : chunk_idx + micro_bs]].to(DEVICE)

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
    print(f"  {tag}  (base={BASE:.0f}, 1B, 200M tokens, seed={SEED})")
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

    # PPL eval
    ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))
    print(f"  PPL@16K = {ppl_16k}")

    # Passkey eval
    print(f"  Passkey eval ({PK_TRIALS} trials/length)...")
    pk = eval_passkey_nll_gap(model, tok, filler, lengths=PK_LENGTHS, depths=[0.5], num_trials=PK_TRIALS)
    g = pk.get("global", {})
    ret = g.get("retrieval_rate", 0)
    gap = g.get("mean_nll_gap", 0)
    print(f"  Retrieval = {ret:.4f},  NLL gap = {gap:.4f}")

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "inv_freq.npy", inv_freq.cpu().numpy())

    result = dict(
        method=tag, base=BASE, seed=SEED, tokens=TOKENS,
        model="1B",
        retrieval=ret, mean_nll_gap=gap,
        ppl=ppl, ppl_16k=ppl_16k,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        train_sec=round(train_sec, 1),
        config=dict(
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
    print("  Phase 9A: 1B Scale-Up (base=500K, 200M tokens, seed=42)")
    print("#" * 70)

    cfg = CFG_1B.copy()

    # batch=8 with 200M tokens → 6103 steps (same as 350M experiments)
    # ~63GB VRAM usage with SDPA — fits well in 96GB
    print(f"\n  Config: batch_size={cfg['batch_size']}, 200M tokens → {48828//cfg['batch_size']} steps")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nLoading training data...")
    train_data = load_data(tok, TOKENS, cfg["seq_len"], "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    print("Loading validation data...")
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    # ── Three runs ────────────────────────────────────────
    runs = [
        ("geo_1b_4k",            geometric_inv_freq()),
        ("evq1.0_1b_4k",        evq_cosh_inv_freq(DIM, 1.0, BASE)),
        ("hybrid1.0_r16_1b_4k", hybrid_evq_inv_freq(DIM, BASE, 1.0, r=16)),
    ]

    results = {}
    for tag, inv_freq in runs:
        r = run_single(tag, inv_freq, cfg, train_data, val_data, filler, tok)
        results[tag] = r

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 9A SUMMARY  (1B, base=500K, 200M tokens, seed=42)")
    print(f"{'='*70}")

    geo_r = results["geo_1b_4k"]
    geo_g = geo_r.get("passkey_global", {})
    geo_ret = geo_g.get("retrieval_rate", geo_r.get("retrieval", 0))
    geo_ppl = geo_r.get("ppl_16k") or geo_r.get("ppl", {}).get("16384", 0)

    print(f"  {'Method':<28s} {'Ret':>8s} {'PPL@16K':>9s} {'NLL gap':>9s} {'vs Geo ret':>12s} {'vs Geo PPL':>12s} {'Time':>8s}")
    print(f"  {'-'*86}")

    for tag, r in results.items():
        g = r.get("passkey_global", {})
        ret = g.get("retrieval_rate", r.get("retrieval", 0))
        ppl = r.get("ppl_16k") or r.get("ppl", {}).get("16384", 0)
        gap = g.get("mean_nll_gap", r.get("mean_nll_gap", 0))
        t_min = r.get("train_sec", 0) / 60

        if tag == "geo_1b_4k":
            ret_d, ppl_d = "baseline", "baseline"
        else:
            ret_d = f"{(ret/geo_ret - 1)*100:+.1f}%" if geo_ret else "--"
            ppl_d = f"{(ppl/geo_ppl - 1)*100:+.1f}%" if geo_ppl else "--"

        print(f"  {tag:<28s} {ret:8.4f} {ppl:9.1f} {gap:9.4f} {ret_d:>12s} {ppl_d:>12s} {t_min:7.1f}m")

    # 350M reference
    print(f"\n  --- 350M reference (Phase 8F, base=500K, 4-seed mean) ---")
    print(f"  {'Geometric 350M':<28s} {'0.735':>8s} {'175.7':>9s}")
    print(f"  {'EVQ 1.0 350M':<28s} {'0.706':>8s} {'193.9':>9s}")
    print(f"  {'Hybrid 1.0 r16 350M':<28s} {'0.709':>8s} {'177.0':>9s}")

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
        print(f"    Hybrid vs Geo: {(hyb_ret/geo_ret-1)*100:+.1f}%  {'WIN' if hyb_beats else 'LOSS'}")
        print(f"    EVQ vs Geo:    {(evq_ret/geo_ret-1)*100:+.1f}%  {'WIN' if evq_beats else 'LOSS'}")
    if hyb_beats or evq_beats:
        print(f"    -> PROCEED to Phase 9B (multi-seed verification)")
    else:
        print(f"    -> Consider 120M tokens or larger scale")

    summary = dict(
        phase="9A", model="1B", base=BASE, seed=SEED, tokens=TOKENS,
        config=CFG_1B,
        results={k: {
            "retrieval": v.get("passkey_global", {}).get("retrieval_rate", v.get("retrieval", 0)),
            "ppl_16k": v.get("ppl_16k") or v.get("ppl", {}).get("16384"),
            "mean_nll_gap": v.get("passkey_global", {}).get("mean_nll_gap", v.get("mean_nll_gap", 0)),
            "ppl": v.get("ppl", {}),
            "train_sec": v.get("train_sec"),
        } for k, v in results.items()},
        hybrid_beats_geo=hyb_beats,
        evq_beats_geo=evq_beats,
    )
    save_json(WORK / "phase9a_summary.json", summary)
    print(f"\n  Saved: {WORK}/phase9a_summary.json")


if __name__ == "__main__":
    main()
