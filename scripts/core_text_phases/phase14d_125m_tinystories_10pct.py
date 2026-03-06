#!/usr/bin/env python3
"""
Phase 14D: 125M on TinyStories with 10% passkey mix.

Tests whether EVQ+YaRN synergy emerges at 125M scale when:
  1. Dataset is simpler (TinyStories vs FineWeb-Edu)
  2. Passkey signal is stronger (10% vs 5%)

Config: 3 seeds (42/123/7) × 2 methods (Geo, EVQ tau=1.5 r=0) = 6 runs.
Everything else matches the 350M 5% experiment (base=500K, head_dim=64, etc.)

Usage:
    python -u phase14d_125m_tinystories_10pct.py > /root/autodl-tmp/evq_phase14d/run.log 2>&1
"""

from __future__ import annotations

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

from run_evq_sweep import (
    GPT, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, evq_cosh_inv_freq, load_data, load_val,
    set_seed, get_batch_from_data, maybe_wrap_with_passkey_mix,
)
from eval_passkey_scratch import eval_passkey_nll_gap

# ===================================================================
# Config
# ===================================================================
BASE = 500_000.0
HEAD_DIM = 64
TRAIN_TOKENS = 100_000_000
SEQ_LEN = 2048
PASSKEY_MIX_RATIO = 0.10  # 10%
DATASET = "tinystories"
SEEDS = [42, 123, 7]
TAUS = [0.0, 1.5]

EVAL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
EVAL_CHUNKS = 8
PASSKEY_LENGTHS = [2048, 4096, 8192, 12288, 16384]
PASSKEY_DEPTHS = [0.1, 0.2, 0.5, 0.8, 0.9]
PASSKEY_TRIALS = 10
YARN_SCALE = 8.0

CFG_125M = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=HEAD_DIM,
    intermediate_size=3072,
    max_position_embeddings=SEQ_LEN,
    lr=3e-4,
    batch_size=16,
    micro_batch_size=4,
    grad_accum=4,
)

WORK = Path("/root/autodl-tmp/evq_phase14d")
DATA_CACHE_DIR = WORK / "data"


# ===================================================================
# YaRN — exact copy from eval_pe_baselines.py
# ===================================================================
def build_yarn_inv_freq(geo_inv, head_dim, scale):
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (geo_inv.double() / yarn_scale).float()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def method_label(tau):
    return "EVQ" if tau > 0 else "Geo"


def run_tag(tau, seed):
    return f"125m_tau{tau:.2f}_seed{seed}"


# ===================================================================
# Training
# ===================================================================
def train_model(model, data, cfg, seed=42):
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

    print(f"  Training: micro_bs={micro_bs}, grad_accum={grad_accum}, "
          f"effective_bs={effective_bs}, steps={steps}, warmup={warmup}, lr={lr}")

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            frac = (s - warmup) / max(steps - warmup, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * frac))
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

        if s % 100 == 0 or s == steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(f"    step {s}/{steps}  loss={accum_loss:.4f}  lr={cur_lr:.2e}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model, elapsed


# ===================================================================
# Evaluation
# ===================================================================
def per_length_pk(pk_summary, lengths):
    result = {}
    for L in lengths:
        trials = [v for k, v in pk_summary.items() if k.startswith(f"L={L}_")]
        if trials:
            result[str(L)] = round(np.mean([t.get("retrieval_rate", 0) for t in trials]) * 100)
    return result


def eval_run(model, tokenizer, filler_tokens, val_data, inv_freq_orig, tag):
    results = {}

    # Baseline
    method = method_label(1.5 if "tau1" in tag else 0)
    print(f"\n  [baseline] {method} raw")
    model.eval()
    with torch.no_grad():
        ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
        print(f"    PPL: {ppl}")
        pk = eval_passkey_nll_gap(model, tokenizer, filler_tokens,
                                  lengths=PASSKEY_LENGTHS, depths=PASSKEY_DEPTHS,
                                  num_trials=PASSKEY_TRIALS, seed=42)
    pk_pl = per_length_pk(pk.get("summary", {}), PASSKEY_LENGTHS)
    print(f"    PK per-length: {pk_pl}")
    results["baseline"] = {
        "ppl": ppl,
        "passkey_global": pk.get("global", {}),
        "passkey_summary": pk.get("summary", {}),
        "passkey_per_length": pk_pl,
    }

    # YaRN
    print(f"\n  [+YaRN] {method} + YaRN (scale={YARN_SCALE})")
    orig_inv = inv_freq_orig.clone()
    yarn_inv = build_yarn_inv_freq(orig_inv, HEAD_DIM, YARN_SCALE)
    model.blocks[0].attn.rope.inv_freq.copy_(yarn_inv)
    model.blocks[0].attn.rope._build(max(EVAL_LENGTHS) + 100)

    with torch.no_grad():
        ppl_yarn = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
        print(f"    PPL: {ppl_yarn}")
        pk_yarn = eval_passkey_nll_gap(model, tokenizer, filler_tokens,
                                       lengths=PASSKEY_LENGTHS, depths=PASSKEY_DEPTHS,
                                       num_trials=PASSKEY_TRIALS, seed=42)
    pk_yarn_pl = per_length_pk(pk_yarn.get("summary", {}), PASSKEY_LENGTHS)
    print(f"    PK per-length: {pk_yarn_pl}")
    results["yarn"] = {
        "ppl": ppl_yarn,
        "passkey_global": pk_yarn.get("global", {}),
        "passkey_summary": pk_yarn.get("summary", {}),
        "passkey_per_length": pk_yarn_pl,
    }

    # Restore
    model.blocks[0].attn.rope.inv_freq.copy_(orig_inv)
    model.blocks[0].attn.rope._build(SEQ_LEN + 100)
    return results


# ===================================================================
# Single run
# ===================================================================
def run_single(tau, seed, train_data, val_data, filler, tokenizer):
    tag = run_tag(tau, seed)
    method = method_label(tau)
    run_dir = WORK / tag

    result_file = run_dir / "result.json"
    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = CFG_125M.copy()

    print(f"\n{'='*60}")
    print(f"  {tag}  ({method}, 125M, seed={seed}, {DATASET}, {PASSKEY_MIX_RATIO:.0%} mix)")
    print(f"{'='*60}")

    inv_freq = evq_cosh_inv_freq(HEAD_DIM, tau=tau, base=BASE)
    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    model, train_time = train_model(model, train_data, cfg, seed=seed)

    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.cpu().numpy())

    inv_freq_orig = model.blocks[0].attn.rope.inv_freq.clone().cpu()
    eval_results = eval_run(model, tokenizer, filler, val_data, inv_freq_orig, tag)
    save_json(run_dir / "passkey_nll.json", eval_results)

    result = {"tag": tag, "method": method, "tau": tau, "seed": seed,
              "dataset": DATASET, "passkey_mix": PASSKEY_MIX_RATIO,
              "train_time_sec": round(train_time, 1), **eval_results}
    save_json(result_file, result)
    print(f"  Saved: {result_file}")

    del model
    torch.cuda.empty_cache()
    return result


# ===================================================================
# Report
# ===================================================================
def generate_report(all_results):
    lines = []
    lines.append("=" * 70)
    lines.append(f"  PHASE 14D: 125M on TinyStories, 10% passkey mix, YaRN scale=8")
    lines.append(f"  3 seeds (42/123/7), Geo vs EVQ (tau=1.5, r=0)")
    lines.append("=" * 70)

    for mode in ["baseline", "yarn"]:
        mode_label = "Baseline" if mode == "baseline" else "+YaRN (scale=8)"
        lines.append(f"\n  --- {mode_label} ---")
        hdr = f"  {'Config':<20} {'PK@2K':>6} {'PK@4K':>6} {'PK@8K':>6} {'PK@12K':>7} {'PK@16K':>7}  {'PPL@2K':>7} {'PPL@8K':>7} {'PPL@16K':>8}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for tau in TAUS:
            method = method_label(tau)
            seed_pks = {str(L): [] for L in PASSKEY_LENGTHS}
            seed_ppls = {str(L): [] for L in EVAL_LENGTHS}

            for seed in SEEDS:
                tag = run_tag(tau, seed)
                r = all_results.get(tag)
                if r is None:
                    continue
                mode_data = r.get(mode, {})
                pk_pl = mode_data.get("passkey_per_length", {})
                ppl = mode_data.get("ppl", {})

                for L in PASSKEY_LENGTHS:
                    v = pk_pl.get(str(L))
                    if v is not None:
                        seed_pks[str(L)].append(v)
                for L in EVAL_LENGTHS:
                    v = ppl.get(str(L))
                    if v is not None:
                        seed_ppls[str(L)].append(v)

                pk_vals = " ".join(f"{pk_pl.get(str(L), '?'):>5}%" for L in PASSKEY_LENGTHS)
                ppl_2k = ppl.get("2048", ppl.get(2048, "?"))
                ppl_8k = ppl.get("8192", ppl.get(8192, "?"))
                ppl_16k = ppl.get("16384", ppl.get(16384, "?"))
                if isinstance(ppl_2k, (int, float)):
                    ppl_str = f"{ppl_2k:>7.1f} {ppl_8k:>7.1f} {ppl_16k:>8.1f}"
                else:
                    ppl_str = f"{'?':>7} {'?':>7} {'?':>8}"
                lines.append(f"  {method+' s'+str(seed):<20} {pk_vals}  {ppl_str}")

            if all(len(seed_pks[str(L)]) >= 2 for L in PASSKEY_LENGTHS):
                pk_summary = ""
                for L in PASSKEY_LENGTHS:
                    vals = seed_pks[str(L)]
                    m, s = np.mean(vals), np.std(vals)
                    pk_summary += f" {m:>3.0f}±{s:<2.0f}%"
                ppl_summary = ""
                for L_key in ["2048", "8192", "16384"]:
                    vals = seed_ppls.get(L_key, [])
                    if vals:
                        ppl_summary += f" {np.mean(vals):>7.1f}"
                    else:
                        ppl_summary += f" {'?':>7}"
                lines.append(f"  {method+' MEAN±STD':<20}{pk_summary} {ppl_summary}")
            lines.append("")

    # Delta
    lines.append(f"\n  --- Delta (EVQ+YaRN - Geo+YaRN) ---")
    for L in PASSKEY_LENGTHS:
        evq_vals, geo_vals = [], []
        for seed in SEEDS:
            er = all_results.get(run_tag(1.5, seed), {}).get("yarn", {}).get("passkey_per_length", {})
            gr = all_results.get(run_tag(0.0, seed), {}).get("yarn", {}).get("passkey_per_length", {})
            if str(L) in er: evq_vals.append(er[str(L)])
            if str(L) in gr: geo_vals.append(gr[str(L)])
        if evq_vals and geo_vals:
            delta = np.mean(evq_vals) - np.mean(geo_vals)
            lines.append(f"    L={L:>5}: EVQ+YaRN={np.mean(evq_vals):.0f}%  Geo+YaRN={np.mean(geo_vals):.0f}%  Delta={delta:+.0f}pp")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================
def main():
    t_start = time.time()

    print("#" * 70)
    print(f"  Phase 14D: 125M on TinyStories, 10% passkey mix")
    print(f"  Seeds: 42, 123, 7  |  Methods: Geo, EVQ (tau=1.5, r=0)")
    print(f"  YaRN scale=8  |  100M tokens")
    print("#" * 70)

    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print(f"\nLoading training data ({DATASET})...")
    train_data_raw = load_data(tok, TRAIN_TOKENS, SEQ_LEN, DATASET, cache_dir=str(DATA_CACHE_DIR))

    print(f"Loading validation data ({DATASET})...")
    val_data = load_val(tok, 5_000_000, DATASET, cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    train_data = maybe_wrap_with_passkey_mix(
        train_data=train_data_raw,
        filler_tokens=filler,
        tokenizer=tok,
        seq_len=SEQ_LEN,
        passkey_ratio=PASSKEY_MIX_RATIO,
    )

    all_results = {}
    for tau in TAUS:
        for seed in SEEDS:
            tag = run_tag(tau, seed)
            result = run_single(tau, seed, train_data, val_data, filler, tok)
            all_results[tag] = result

    report = generate_report(all_results)
    print("\n" + report)

    save_json(WORK / "all_results.json", all_results)
    with open(WORK / "REPORT.txt", "w") as f:
        f.write(report)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f} hours")
    print(f"  Results: {WORK}/all_results.json")
    print(f"  Report:  {WORK}/REPORT.txt")


if __name__ == "__main__":
    main()
