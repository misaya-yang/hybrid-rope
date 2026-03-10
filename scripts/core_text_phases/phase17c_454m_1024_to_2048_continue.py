#!/usr/bin/env python3
"""
Phase 17C: 454M staged continuation 1024 → 2048, 500M tokens.

Stage 3 of the three-stage length extension protocol:
  Stage 1  (phase17):  L=512,  1B tokens, τ_geo=0 / τ_evq=2.8
  Stage 2  (phase17b): L=1024, 1B tokens, τ_geo=0 / τ_evq=2.0
  Stage 3  (phase17c): L=2048, 500M tokens, τ_geo=0 / τ_evq=√2 ≈ 1.4142

Paper-required hyperparameters:
  ┌─────────────────────────────┬──────────────────────────────────────────┐
  │ Parameter                   │ Value / Formula                          │
  ├─────────────────────────────┼──────────────────────────────────────────┤
  │ EVQ τ*                      │ d_head / √L_train = 64/√2048 ≈ 1.4142   │
  │ Geo (baseline)              │ τ = 0 (geometric RoPE)                   │
  │ θ_base                      │ 500,000                                  │
  │ L_train                     │ 2048                                     │
  │ Tokens                      │ 500M                                     │
  │ LR                          │ 2e-4 (cosine → 10%)                      │
  │ Warmup                      │ 2% of total steps                        │
  │ Optimizer                   │ AdamW, β=(0.9,0.95), wd=0.1              │
  │ Effective batch size         │ 20 (micro_bs=5 × grad_accum=4)           │
  │ Passkey mix                 │ 5%                                       │
  │ Eval lengths                │ 2K,4K,8K,16K,24K,32K,40K,48K            │
  └─────────────────────────────┴──────────────────────────────────────────┘

Init checkpoints (from Stage 2):
  geo: /root/autodl-tmp/evq_phase17b/454m_geo_seed42_continue1024/model.pt
  evq: /root/autodl-tmp/evq_phase17b/454m_evq_seed42_continue1024/model.pt

Data strategy:
  Rechunk existing stage-1 data (tokens 0–1B, only seen at L=512) to L=2048.
  Source: /root/autodl-tmp/evq_phase17b/train_rechunk_1000000000_1024.pt
  First half → 1B tokens → rechunk to L=2048 → 488K chunks (~1000M tokens)
  Take first 244K chunks = 500M tokens for stage 3.
  Fallback: stream fresh fineweb-edu if rechunk file missing.

Usage:
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/hybrid-rope
  python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py

Environment overrides:
  PHASE17C_WORK          — output directory
  PHASE17C_TOKENS        — training tokens (default 500M)
  PHASE17C_SEQ_LEN       — training length (default 2048)
  PHASE17C_SEED          — random seed (default 42)
  PHASE17C_TAU           — EVQ tau (default d_head/sqrt(L))
  PHASE17C_GEO_INIT_CKPT — geo init checkpoint
  PHASE17C_EVQ_INIT_CKPT — evq init checkpoint
  PHASE17C_RUN_ONLY      — geo|evq (run single fork)
  PHASE17C_MICRO_BATCH_SIZE — micro batch size (default 5)
  PHASE17C_GRAD_ACCUM    — gradient accumulation (default 4)
  PHASE17C_PASSKEY_MIX_RATIO — passkey mix ratio (default 0.05)
  PHASE17C_DATASET       — training dataset (default fineweb-edu)
"""

import gc
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

from eval_passkey_scratch import eval_passkey_nll_gap, make_passkey_training_sample  # noqa: E402
from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    load_data,
    load_val,
    resolve_passkey_mix_ratio,
    set_seed,
)

# ── Constants ─────────────────────────────────────────────────────────────

BASE = 500_000.0
DIM = 64  # head_dim
SEQ_LEN = int(os.environ.get("PHASE17C_SEQ_LEN", "2048"))
TAU = float(os.environ.get("PHASE17C_TAU", f"{DIM / math.sqrt(SEQ_LEN):.6f}"))
SEED = int(os.environ.get("PHASE17C_SEED", "42"))
TOKENS = int(os.environ.get("PHASE17C_TOKENS", "500000000"))
RUN_ONLY = os.environ.get("PHASE17C_RUN_ONLY", "").strip().lower()
DATASET = os.environ.get("PHASE17C_DATASET", "fineweb-edu").strip() or "fineweb-edu"

PASSKEY_MIX_RATIO = float(
    os.environ.get(
        "PHASE17C_PASSKEY_MIX_RATIO",
        str(resolve_passkey_mix_ratio(default=0.05)),
    )
)

EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17C_EVAL_LENGTHS",
        "2048,4096,8192,16384,24576,32768,40960,49152",
    ).split(",")
    if x.strip()
]
CKPT_EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17C_CKPT_EVAL_LENGTHS",
        "2048,4096,8192,16384",
    ).split(",")
    if x.strip()
]

CKPT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
CKPT_EVAL_CHUNKS = 10
EVAL_CHUNKS = 10

PK_LENGTHS = [2048, 4096, 8192]  # 16K passkey OOMs with torch.compile CUDA Graphs
CKPT_PK_TRIALS = 10
PK_TRIALS = 10
PK_DEPTHS = [0.5]

GPU_CHUNKS = int(os.environ.get("PHASE17C_GPU_CHUNKS", "8"))

WORK = Path(
    os.environ.get(
        "PHASE17C_WORK",
        "/root/autodl-tmp/evq_phase17c_2048_continue",
    )
)
DATA_CACHE_DIR = Path(os.environ.get("PHASE17C_DATA_CACHE_DIR", str(WORK / "data")))

GEO_INIT_CKPT = os.environ.get(
    "PHASE17C_GEO_INIT_CKPT",
    "/root/autodl-tmp/evq_phase17b/454m_geo_seed42_continue1024/model.pt",
).strip()
EVQ_INIT_CKPT = os.environ.get(
    "PHASE17C_EVQ_INIT_CKPT",
    "/root/autodl-tmp/evq_phase17b/454m_evq_seed42_continue1024/model.pt",
).strip()

# Stage-2 rechunk source: 2B tokens at L=1024.
# First half (chunks 0–976561) = tokens 0–1B, only seen at L=512 in stage 1.
RECHUNK_SOURCE = Path(os.environ.get(
    "PHASE17C_RECHUNK_SOURCE",
    "/root/autodl-tmp/evq_phase17b/train_rechunk_1000000000_1024.pt",
))

# ── Model config ──────────────────────────────────────────────────────────
# Same 454M arch as stages 1-2, only seq_len/batch adjusted for L=2048.

CFG_454M_2048 = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=2e-4,
    batch_size=20,          # effective
    micro_batch_size=int(os.environ.get("PHASE17C_MICRO_BATCH_SIZE", "5")),
    grad_accum=int(os.environ.get("PHASE17C_GRAD_ACCUM", "4")),
)


# ── Helpers ───────────────────────────────────────────────────────────────


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_state(path: str):
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint format at {path}")
    # Strip _orig_mod. prefix from torch.compile
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    # Remove RoPE buffers (will be rebuilt)
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


def load_model_with_target_inv(cfg, ckpt_path: str, target_inv: torch.Tensor):
    model = GPT(cfg, target_inv).to(DEVICE)
    state = _load_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    # Force-set inv_freq
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(
            target_inv.to(block.attn.rope.inv_freq.device)
        )
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    return model


def load_rechunked_train_data():
    """Load training data by rechunking stage-1 tokens from L=1024 to L=2048.

    The source file has 2B tokens at L=1024 (1953125 chunks).
    First half (chunks 0–976561) = tokens 0–1B, only seen at L=512 in stage 1.
    Rechunk to L=2048: 1B tokens → 488281 chunks.
    Slice to TOKENS/SEQ_LEN chunks for stage 3.
    """
    cache_name = f"train_rechunk_stage3_{TOKENS}_{SEQ_LEN}.pt"
    cache_path = DATA_CACHE_DIR / cache_name

    if cache_path.exists():
        print(f"  [data] loading stage-3 cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [data] cached: {data.shape[0]} chunks ({data.numel()/1e6:.1f}M tokens)")
        return data

    if not RECHUNK_SOURCE.exists():
        raise FileNotFoundError(
            f"Rechunk source not found: {RECHUNK_SOURCE}\n"
            "Set PHASE17C_RECHUNK_SOURCE or provide data manually."
        )

    print(f"  [data] loading rechunk source: {RECHUNK_SOURCE}")
    source = torch.load(RECHUNK_SOURCE, map_location="cpu", weights_only=True)
    print(f"  [data] source shape={source.shape}, tokens={source.numel()/1e9:.2f}B")

    # Take first half (tokens 0–1B, unseen at L≥1024)
    half = source.shape[0] // 2
    first_half = source[:half]  # 976562 chunks × 1024
    del source

    # Flatten and rechunk to L=2048
    flat = first_half.reshape(-1)
    del first_half
    need_tokens = TOKENS
    usable = (need_tokens // SEQ_LEN) * SEQ_LEN
    if flat.numel() < usable:
        raise ValueError(
            f"Need {usable} tokens but first-half only has {flat.numel()}"
        )
    data = flat[:usable].clone().view(-1, SEQ_LEN)
    del flat

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(
        f"  [data] built stage-3 cache: {cache_path} "
        f"({data.shape[0]} chunks, {data.numel()/1e6:.1f}M tokens)"
    )
    return data


def build_mixed_train_tensor(train_data: torch.Tensor, filler: torch.Tensor, tokenizer):
    """Mix passkey supervision into LM training data."""
    if PASSKEY_MIX_RATIO <= 0:
        print("  [passkey-train] mix disabled (pure LM)")
        return train_data

    mix_pct = int(round(PASSKEY_MIX_RATIO * 100))
    cache_path = DATA_CACHE_DIR / (
        f"mixed_{DATASET}_seed{SEED}_pk{mix_pct}_{TOKENS}_{SEQ_LEN}.pt"
    )
    if cache_path.exists():
        print(f"  [passkey-train] loading cached mixed tensor: {cache_path}")
        mixed = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [passkey-train] cached shape={tuple(mixed.shape)}")
        return mixed

    mixed = train_data.clone()
    n_total = mixed.shape[0]
    n_pk = int(n_total * PASSKEY_MIX_RATIO)
    if n_pk <= 0:
        return mixed

    print(f"  [passkey-train] pre-generating {n_pk} passkey samples...")
    pk_indices = torch.randperm(n_total)[:n_pk]
    t0 = time.time()
    for i, idx in enumerate(pk_indices.tolist(), start=1):
        mixed[idx] = make_passkey_training_sample(
            filler_tokens=filler,
            tokenizer=tokenizer,
            seq_len=SEQ_LEN,
            seed=idx,
        )
        if i % 10_000 == 0 or i == n_pk:
            print(f"    generated {i}/{n_pk} passkey samples")

    torch.save(mixed, cache_path)
    print(f"  [passkey-train] built in {time.time() - t0:.1f}s, cached to {cache_path}")
    return mixed


# ── Training loop ─────────────────────────────────────────────────────────


def train_model_ga(model, data, cfg, seed=42, on_step_end=None):
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum"]
    effective_bs = micro_bs * grad_accum

    adamw_kwargs = dict(lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if DEVICE == "cuda":
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    total_seqs = data.shape[0]
    steps = total_seqs // effective_bs
    warmup = max(1, int(steps * 0.02))
    if steps <= 0:
        raise ValueError(
            f"not enough chunks for one step: chunks={total_seqs}, effective_bs={effective_bs}"
        )

    print(
        f"  Train cfg: lr={lr:.1e}, min_lr={min_lr:.1e}, "
        f"micro_bs={micro_bs}, grad_accum={grad_accum}, "
        f"effective_bs={effective_bs}, chunks={total_seqs}, steps={steps}, "
        f"warmup={warmup}, gpu_chunks={GPU_CHUNKS}"
    )

    set_seed(seed)
    perm = torch.randperm(total_seqs)
    data = data[perm].contiguous()
    del perm
    t0 = time.time()

    chunk_size = total_seqs // GPU_CHUNKS
    global_step = 0

    for ci in range(GPU_CHUNKS):
        c_start = ci * chunk_size
        c_end = c_start + chunk_size if ci < GPU_CHUNKS - 1 else total_seqs
        chunk_len = c_end - c_start
        chunk_steps = chunk_len // effective_bs
        if chunk_steps <= 0:
            continue

        gpu_chunk = data[c_start:c_end].to(torch.int64).to(DEVICE).contiguous()
        print(
            f"  [gpu-chunk {ci+1}/{GPU_CHUNKS}] loaded {tuple(gpu_chunk.shape)} "
            f"({gpu_chunk.element_size() * gpu_chunk.numel() / 1e9:.1f}GB)"
        )

        for cs in range(chunk_steps):
            s = global_step
            if s >= steps:
                break

            # LR schedule: linear warmup → cosine decay to min_lr
            if s < warmup:
                cur_lr = lr * (s + 1) / warmup
            else:
                cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
                )
            for g in opt.param_groups:
                g["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            if DEVICE == "cuda" and hasattr(torch, "compiler"):
                torch.compiler.cudagraph_mark_step_begin()

            for a in range(grad_accum):
                idx_start = cs * effective_bs + a * micro_bs
                idx_end = idx_start + micro_bs
                batch = gpu_chunk[idx_start:idx_end]
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()

                ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
                with ctx:
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                    loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item()

            if s < warmup:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1

            step_num = s + 1
            if s % 50 == 0 or step_num == steps:
                elapsed = time.time() - t0
                eta = elapsed / step_num * (steps - step_num) if step_num > 0 else 0
                gpu_mem = (
                    torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
                )
                print(
                    f"    step {step_num}/{steps}  loss={accum_loss:.4f}  "
                    f"lr={cur_lr:.2e}  GPU={gpu_mem:.1f}GB  ETA={eta/60:.0f}min"
                )

            if on_step_end is not None:
                on_step_end(step_num, steps)

        del gpu_chunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model, elapsed


# ── Single run ────────────────────────────────────────────────────────────


def run_single_train(
    tag,
    init_ckpt_path,
    target_inv,
    cfg,
    train_data,
):
    """Train only — save checkpoints at 25/50/75/100%, no eval (avoids OOM)."""
    run_dir = WORK / f"seed{SEED}" / tag
    ckpt_dir = run_dir / "checkpoints"
    model_path = run_dir / "model.pt"

    if model_path.exists():
        print(f"\n[SKIP] {tag}: model.pt already exists")
        return str(model_path)

    if not Path(init_ckpt_path).exists():
        raise FileNotFoundError(f"missing init checkpoint: {init_ckpt_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  TRAIN: {tag}  |  init={init_ckpt_path}")
    print(
        f"  Continue @L={SEQ_LEN}, tokens={TOKENS/1e6:.0f}M, seed={SEED}, "
        f"tau={TAU:.6f}"
    )
    print(f"{'=' * 72}")

    set_seed(SEED)
    model = load_model_with_target_inv(cfg, init_ckpt_path, target_inv)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled (reduce-overhead)")
    print(
        f"  inv_freq set: max={target_inv.max().item():.8f} "
        f"min={target_inv.min().item():.8f}"
    )

    total_steps = len(train_data) // (cfg["micro_batch_size"] * cfg["grad_accum"])
    checkpoint_steps = sorted(
        set(max(1, min(total_steps, int(total_steps * frac))) for frac in CKPT_FRACTIONS)
    )
    print(f"  checkpoint_steps={checkpoint_steps}")

    ckpt_done = set()

    def on_step_end(step_num, steps_total):
        if step_num not in checkpoint_steps or step_num in ckpt_done:
            return
        ckpt_done.add(step_num)
        frac = step_num / steps_total
        ckpt_name = f"step_{step_num:05d}"
        ckpt_path = ckpt_dir / f"{ckpt_name}.pt"
        print(f"\n  [CKPT-SAVE] {tag} {ckpt_name} ({frac:.1%})")
        torch.save(model.state_dict(), ckpt_path)

    model, train_elapsed = train_model_ga(
        model=model,
        data=train_data,
        cfg=cfg,
        seed=SEED,
        on_step_end=on_step_end,
    )

    # Save final model
    torch.save(model.state_dict(), model_path)
    np.save(run_dir / "inv_freq.npy", target_inv.cpu().numpy())
    save_json(run_dir / "train_meta.json", dict(
        method=tag, base=BASE, seed=SEED, init_ckpt=init_ckpt_path,
        continue_tokens=TOKENS, seq_len=SEQ_LEN,
        tau=TAU if "evq" in tag else 0.0,
        train_sec=round(train_elapsed, 1),
        checkpoint_fractions=CKPT_FRACTIONS,
    ))

    print(f"\n  Training done: {model_path}  ({train_elapsed/60:.1f} min)")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return str(model_path)


def run_eval_for_tag(
    tag,
    target_inv,
    cfg,
    val_data,
    filler,
    tok,
    eval_lengths,
):
    """Eval a trained model — runs in a clean process without torch.compile."""
    run_dir = WORK / f"seed{SEED}" / tag
    model_path = run_dir / "model.pt"
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n[SKIP-EVAL] {tag}: result.json already exists")
        with open(result_file) as f:
            return json.load(f)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    print(f"\n{'=' * 72}")
    print(f"  EVAL: {tag}")
    print(f"{'=' * 72}")

    # Load model WITHOUT torch.compile — gives full VRAM for eval
    model = load_model_with_target_inv(cfg, str(model_path), target_inv)
    model.eval()
    print(f"  Model loaded (no torch.compile), eval_lengths={eval_lengths}")

    # PPL
    with torch.no_grad():
        ppl = eval_model(model, val_data, eval_lengths, EVAL_CHUNKS)

    # Passkey
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        pk = eval_passkey_nll_gap(
            model, tok, filler,
            lengths=PK_LENGTHS,
            depths=PK_DEPTHS,
            num_trials=PK_TRIALS,
        )

    result = dict(
        method=tag,
        base=BASE,
        seed=SEED,
        model="454M",
        seq_len=SEQ_LEN,
        tau=TAU if "evq" in tag else 0.0,
        ppl=ppl,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        config=dict(
            passkey_mix_ratio=PASSKEY_MIX_RATIO,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            lr=cfg["lr"],
            effective_bs=cfg["batch_size"],
            micro_bs=cfg["micro_batch_size"],
            grad_accum=cfg["grad_accum"],
            eval_lengths=eval_lengths,
            checkpoint_eval_lengths=CKPT_EVAL_LENGTHS,
        ),
    )
    save_json(result_file, result)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print("#" * 72)
    print("  Phase 17C: 454M staged continuation 1024 → 2048")
    print("  Strategy: TRAIN ALL FIRST, then EVAL ALL (avoids compile OOM)")
    print("#" * 72)

    cfg = CFG_454M_2048.copy()
    cfg["batch_size"] = cfg["micro_batch_size"] * cfg["grad_accum"]

    eval_lengths = sorted(set(int(v) for v in EVAL_LENGTHS if int(v) > 0))

    print(f"  device={DEVICE} dtype={DTYPE} autocast={USE_AUTOCAST}")
    print(f"  L_train={SEQ_LEN} tau*={TAU:.6f} tokens={TOKENS/1e6:.0f}M seed={SEED}")
    print(f"  lr={cfg['lr']:.1e} micro_bs={cfg['micro_batch_size']} "
          f"grad_accum={cfg['grad_accum']} effective_bs={cfg['batch_size']}")
    print(f"  passkey_mix={PASSKEY_MIX_RATIO:.2%}")
    print(f"  dataset={DATASET}")
    print(f"  eval_lengths={eval_lengths}")
    print(f"  geo_init: {GEO_INIT_CKPT}")
    print(f"  evq_init: {EVQ_INIT_CKPT}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build runs ──
    runs = [
        dict(
            tag="geo_454m_1024_to_2048_continue",
            init_ckpt=GEO_INIT_CKPT,
            target_inv=geometric_inv_freq(),
        ),
        dict(
            tag=f"evq{TAU:g}_454m_1024_to_2048_continue",
            init_ckpt=EVQ_INIT_CKPT,
            target_inv=evq_cosh_inv_freq(head_dim=DIM, tau=TAU, base=BASE),
        ),
    ]
    if RUN_ONLY in {"geo", "evq"}:
        if RUN_ONLY == "geo":
            runs = [r for r in runs if r["tag"].startswith("geo_")]
        else:
            runs = [r for r in runs if r["tag"].startswith("evq")]
        print(f"  run filter={RUN_ONLY} -> {[r['tag'] for r in runs]}")

    # ────────────────────────────────────────────────────────────────────
    # PHASE 1: TRAIN ALL MODELS (with torch.compile, no eval)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 72}")
    print("  PHASE 1: TRAINING (all models)")
    print(f"{'#' * 72}")

    # Load training data
    print("\n  Loading training data...")
    if RECHUNK_SOURCE.exists():
        train_data = load_rechunked_train_data()
    else:
        print(f"  [data] rechunk source not found, falling back to HF: {DATASET}")
        train_data = load_data(tok, TOKENS, SEQ_LEN, DATASET, cache_dir=str(DATA_CACHE_DIR))
    print(f"  train_data: {train_data.shape} ({train_data.numel()/1e6:.0f}M tokens)")

    # Load val for passkey mixing only
    print("  Loading validation data...")
    val_cache = Path("/root/autodl-tmp/evq_phase17/val_proof-pile-2_5000000.pt")
    if val_cache.exists():
        print(f"  Using cached val: {val_cache}")
        val_data = torch.load(val_cache, map_location="cpu", weights_only=True)
    else:
        val_data = load_val(tok, 5_000_000, DATASET, cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    # Mix passkey samples
    train_data = build_mixed_train_tensor(train_data, filler, tok)

    # ── Train all models ──
    model_paths = {}
    for run in runs:
        mp = run_single_train(
            tag=run["tag"],
            init_ckpt_path=run["init_ckpt"],
            target_inv=run["target_inv"],
            cfg=cfg,
            train_data=train_data,
        )
        model_paths[run["tag"]] = mp

    # Free training data & CUDA graphs before eval
    del train_data
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("\n  Training data freed. CUDA cache cleared.")

    # ────────────────────────────────────────────────────────────────────
    # PHASE 2: EVAL ALL MODELS (no torch.compile, full VRAM for long ctx)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 72}")
    print("  PHASE 2: EVALUATION (all models, no torch.compile)")
    print(f"{'#' * 72}")

    results = {}
    for run in runs:
        res = run_eval_for_tag(
            tag=run["tag"],
            target_inv=run["target_inv"],
            cfg=cfg,
            val_data=val_data,
            filler=filler,
            tok=tok,
            eval_lengths=eval_lengths,
        )
        results[run["tag"]] = res

    # ── Summary ──
    summary = dict(
        phase="17c_454m_1024_to_2048_continue",
        model="454M",
        seq_len=SEQ_LEN,
        tokens=TOKENS,
        seed=SEED,
        tau=TAU,
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        dataset=DATASET,
        runs=list(results.keys()),
        results=results,
    )
    summary_path = WORK / "phase17c_summary.json"
    save_json(summary_path, summary)
    print(f"\n  Summary saved: {summary_path}")

    # ── Print comparison table ──
    print(f"\n{'=' * 80}")
    print(f"  FINAL PPL COMPARISON")
    print(f"{'=' * 80}")
    for tag, res in results.items():
        ppl = res.get("ppl", {})
        line = f"  {tag:>45s}:"
        for L in eval_lengths:
            v = ppl.get(str(L))
            if v is not None:
                line += f"  {L//1024}K={v:.1f}"
            else:
                line += f"  {L//1024}K=--"
        print(line)


if __name__ == "__main__":
    main()
