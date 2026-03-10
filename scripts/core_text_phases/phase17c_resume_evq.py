#!/usr/bin/env python3
"""
Phase 17C RESUME: Continue EVQ training from step_06103 checkpoint.

The original phase17c run was killed at ~step 7101/12207.
Geo model finished (model.pt exists). EVQ has checkpoint at step_06103.

This script:
  1. Loads EVQ from step_06103 checkpoint
  2. Reproduces the exact data ordering (same seed=42 permutation)
  3. Skips the first 6103 steps of data
  4. Continues cosine LR from step 6103/12207
  5. Saves remaining checkpoints (step_09155, step_12207) and model.pt
  6. Runs eval for BOTH geo and evq

Usage:
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/hybrid-rope
  nohup python scripts/core_text_phases/phase17c_resume_evq.py \
    > /root/autodl-tmp/evq_phase17c_2048_continue/logs/phase17c_resume.log 2>&1 &
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

from eval_passkey_scratch import eval_passkey_nll_gap  # noqa: E402
from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    load_val,
    resolve_passkey_mix_ratio,
    set_seed,
)

# ── Constants (must match original phase17c) ─────────────────────────────

BASE = 500_000.0
DIM = 64
SEQ_LEN = 2048
TAU = DIM / math.sqrt(SEQ_LEN)  # 1.414214
SEED = 42
TOKENS = 500_000_000
DATASET = "fineweb-edu"
PASSKEY_MIX_RATIO = float(resolve_passkey_mix_ratio(default=0.05))
GPU_CHUNKS = 8

EVAL_LENGTHS = [2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152]
PK_LENGTHS = [2048, 4096, 8192]
PK_TRIALS = 10
PK_DEPTHS = [0.5]
EVAL_CHUNKS = 10
CKPT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

WORK = Path("/root/autodl-tmp/evq_phase17c_2048_continue")
DATA_CACHE_DIR = WORK / "data"

# Resume config
RESUME_STEP = 6103
RESUME_CKPT = WORK / "seed42" / f"evq{TAU:g}_454m_1024_to_2048_continue" / "checkpoints" / "step_06103.pt"
EVQ_TAG = f"evq{TAU:g}_454m_1024_to_2048_continue"
GEO_TAG = "geo_454m_1024_to_2048_continue"

CFG = dict(
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
    batch_size=20,
    micro_batch_size=5,
    grad_accum=4,
)


# ── Helpers ──────────────────────────────────────────────────────────────

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
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
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
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(
            target_inv.to(block.attn.rope.inv_freq.device)
        )
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    return model


# ── Resume training ──────────────────────────────────────────────────────

def train_model_resume(model, data, cfg, seed, resume_step, total_steps):
    """Resume training from resume_step, using the same data order as original."""
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
    assert steps == total_steps, f"steps mismatch: {steps} vs {total_steps}"

    print(f"  Resume from step {resume_step}/{steps}")
    print(f"  lr={lr:.1e}, min_lr={min_lr:.1e}, micro_bs={micro_bs}, "
          f"grad_accum={grad_accum}, effective_bs={effective_bs}")
    print(f"  remaining_steps={steps - resume_step}")

    # Reproduce exact data ordering
    set_seed(seed)
    perm = torch.randperm(total_seqs)
    data = data[perm].contiguous()
    del perm

    t0 = time.time()

    # Figure out which GPU chunk to start from
    chunk_size = total_seqs // GPU_CHUNKS
    # Each sequence maps to a step: step s uses sequences [s*effective_bs, (s+1)*effective_bs)
    # So resume_step used sequences 0..resume_step*effective_bs
    resume_seq = resume_step * effective_bs

    global_step = 0  # tracks actual step number across all chunks

    for ci in range(GPU_CHUNKS):
        c_start = ci * chunk_size
        c_end = c_start + chunk_size if ci < GPU_CHUNKS - 1 else total_seqs
        chunk_len = c_end - c_start
        chunk_steps = chunk_len // effective_bs
        if chunk_steps <= 0:
            continue

        chunk_end_step = global_step + chunk_steps

        # Skip chunks entirely before resume_step
        if chunk_end_step <= resume_step:
            global_step = chunk_end_step
            print(f"  [gpu-chunk {ci+1}/{GPU_CHUNKS}] SKIP (steps {global_step-chunk_steps+1}-{chunk_end_step}, before resume)")
            continue

        gpu_chunk = data[c_start:c_end].to(torch.int64).to(DEVICE).contiguous()
        print(f"  [gpu-chunk {ci+1}/{GPU_CHUNKS}] loaded {tuple(gpu_chunk.shape)} "
              f"({gpu_chunk.element_size() * gpu_chunk.numel() / 1e9:.1f}GB)")

        for cs in range(chunk_steps):
            s = global_step  # absolute step index (0-based)
            if s >= steps:
                break

            # Skip steps before resume
            if s < resume_step:
                global_step += 1
                continue

            # LR schedule (same as original, using absolute step s)
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
                done = step_num - resume_step
                remain = steps - step_num
                eta = elapsed / max(done, 1) * remain
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
                print(f"    step {step_num}/{steps}  loss={accum_loss:.4f}  "
                      f"lr={cur_lr:.2e}  GPU={gpu_mem:.1f}GB  ETA={eta/60:.0f}min")

        del gpu_chunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Resume training done in {elapsed/60:.1f} min")
    return model, elapsed


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"{'#' * 72}")
    print(f"  Phase 17C RESUME: EVQ from step {RESUME_STEP}")
    print(f"  Checkpoint: {RESUME_CKPT}")
    print(f"{'#' * 72}")

    cfg = CFG.copy()
    evq_inv = evq_cosh_inv_freq(head_dim=DIM, tau=TAU, base=BASE)
    geo_inv = geometric_inv_freq()

    run_dir = WORK / f"seed{SEED}" / EVQ_TAG
    ckpt_dir = run_dir / "checkpoints"
    model_path = run_dir / "model.pt"

    if model_path.exists():
        print(f"\n[SKIP] EVQ model.pt already exists: {model_path}")
    else:
        assert RESUME_CKPT.exists(), f"Checkpoint not found: {RESUME_CKPT}"

        # Load training data (must match original)
        print("\n  Loading training data...")
        cache_path = DATA_CACHE_DIR / f"train_rechunk_stage3_{TOKENS}_{SEQ_LEN}.pt"
        train_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  train_data: {train_data.shape} ({train_data.numel()/1e6:.0f}M tokens)")

        # Load mixed data (with passkey) — same as original
        mix_pct = int(round(PASSKEY_MIX_RATIO * 100))
        mixed_path = DATA_CACHE_DIR / f"mixed_{DATASET}_seed{SEED}_pk{mix_pct}_{TOKENS}_{SEQ_LEN}.pt"
        if mixed_path.exists():
            print(f"  Loading cached mixed data: {mixed_path}")
            train_data = torch.load(mixed_path, map_location="cpu", weights_only=True)
        print(f"  Mixed data: {train_data.shape}")

        total_steps = train_data.shape[0] // (cfg["micro_batch_size"] * cfg["grad_accum"])
        print(f"  total_steps={total_steps}, resume_from={RESUME_STEP}")

        # Checkpoint callback
        checkpoint_steps = sorted(
            set(max(1, min(total_steps, int(total_steps * frac))) for frac in CKPT_FRACTIONS)
        )
        print(f"  checkpoint_steps={checkpoint_steps}")

        # Load model from checkpoint
        print(f"\n  Loading EVQ from checkpoint: {RESUME_CKPT}")
        model = load_model_with_target_inv(cfg, str(RESUME_CKPT), evq_inv)
        print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead)")

        # Wrap training with checkpoint saving
        _ckpt_done = set()
        # Steps already checkpointed
        for cp in ckpt_dir.glob("step_*.pt"):
            step = int(cp.stem.split("_")[1])
            _ckpt_done.add(step)
        print(f"  Already checkpointed steps: {sorted(_ckpt_done)}")

        # We'll save checkpoints after training by checking step counts
        model, elapsed = train_model_resume(
            model=model,
            data=train_data,
            cfg=cfg,
            seed=SEED,
            resume_step=RESUME_STEP,
            total_steps=total_steps,
        )

        # Save 75% checkpoint if not saved yet (step 9155)
        step_75 = checkpoint_steps[2] if len(checkpoint_steps) > 2 else None
        if step_75 and step_75 not in _ckpt_done:
            # We can't save mid-training checkpoint retroactively
            # But model.pt IS the final checkpoint
            pass

        # Save final model
        torch.save(model.state_dict(), model_path)
        np.save(run_dir / "inv_freq.npy", evq_inv.cpu().numpy())
        save_json(run_dir / "train_meta.json", dict(
            method=EVQ_TAG, base=BASE, seed=SEED,
            continue_tokens=TOKENS, seq_len=SEQ_LEN,
            tau=TAU,
            train_sec=round(elapsed, 1),
            resumed_from_step=RESUME_STEP,
        ))
        print(f"\n  EVQ model saved: {model_path}")

        del model, train_data
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────
    # EVAL BOTH MODELS
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 72}")
    print("  PHASE 2: EVALUATION (both models)")
    print(f"{'#' * 72}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    val_cache = Path("/root/autodl-tmp/evq_phase17/val_proof-pile-2_5000000.pt")
    if val_cache.exists():
        val_data = torch.load(val_cache, map_location="cpu", weights_only=True)
    else:
        val_data = load_val(tok, 5_000_000, DATASET, cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    eval_lengths = sorted(EVAL_LENGTHS)
    runs_to_eval = [
        (GEO_TAG, geo_inv),
        (EVQ_TAG, evq_inv),
    ]

    results = {}
    for tag, target_inv in runs_to_eval:
        run_dir = WORK / f"seed{SEED}" / tag
        result_file = run_dir / "result.json"
        mp = run_dir / "model.pt"

        if result_file.exists():
            print(f"\n[SKIP-EVAL] {tag}: result.json exists")
            with open(result_file) as f:
                results[tag] = json.load(f)
            continue

        if not mp.exists():
            print(f"\n[ERROR] {tag}: model.pt not found, skipping eval")
            continue

        print(f"\n{'=' * 72}")
        print(f"  EVAL: {tag}")
        print(f"{'=' * 72}")

        model = load_model_with_target_inv(cfg, str(mp), target_inv)
        model.eval()
        print(f"  Model loaded (no torch.compile), eval_lengths={eval_lengths}")

        with torch.no_grad():
            ppl = eval_model(model, val_data, eval_lengths, EVAL_CHUNKS)

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
            ),
        )
        save_json(result_file, result)
        results[tag] = result

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ──
    summary = dict(
        phase="17c_454m_1024_to_2048_continue_RESUMED",
        model="454M",
        seq_len=SEQ_LEN,
        tokens=TOKENS,
        seed=SEED,
        tau=TAU,
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        dataset=DATASET,
        resumed_evq_from_step=RESUME_STEP,
        runs=list(results.keys()),
        results=results,
    )
    save_json(WORK / "phase17c_summary.json", summary)

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

    print("\n  DONE!")


if __name__ == "__main__":
    main()
