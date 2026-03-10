#!/usr/bin/env python3
"""
Phase 20: 1.5B staged length extension protocol for EVQ-Cosh NeurIPS experiments.

Three-stage progressive training pipeline:
  Stage 0 (baseline):    L=2048, 1B tokens, τ*=64/√2048≈1.414, init from pretrained
  Stage 1 (extension):   L=4096, 1.5B tokens, τ*=64/√4096=1.0, init from Stage 0
  Stage 2 (extreme):     L=8192, 1B tokens, τ*=64/√8192≈0.707, init from Stage 1

Each stage trains both geometric (τ=0) and EVQ (τ=τ*) rope variants, with:
  - Passkey retrieval supervision (5% mix)
  - Comprehensive PPL evaluation at multiple lengths
  - Full long-context analysis with YaRN overlay comparison
  - Checkpoint-based eval at 25/50/75/100% training progress
  - Dry-run mode for memory estimation

Paper-required hyperparameters follow scaled sequence length:
  LR decay: cosine with linear warmup (2%), min_lr=10% of lr
  AdamW: betas=(0.9, 0.95), weight_decay=0.1, fused=True on CUDA
  θ_base: 500,000 (geometric invariant across stages)

Environment variables:
  PHASE20_STAGE              — 0/1/2 (default 0)
  PHASE20_METHOD             — geo|evq|both (default both)
  PHASE20_SEED               — random seed (default 42)
  PHASE20_WORK               — output directory (default /root/autodl-tmp/evq_phase20)
  PHASE20_INIT_CKPT          — for stage>0, init checkpoint path (auto if unset)
  PHASE20_DRY_RUN            — if set, build model, estimate memory, run 1 step, exit
  PHASE20_TOKENS             — override token budget
  PHASE20_MICRO_BATCH_SIZE   — override micro batch size
  PHASE20_GRAD_ACCUM         — override gradient accumulation steps
  PHASE20_PASSKEY_MIX_RATIO  — passkey supervision ratio (default 0.05)
  PHASE20_DATASET            — training dataset (default fineweb-edu)
  PHASE20_EVAL_LENGTHS       — comma-separated eval lengths
  PHASE20_GPU_CHUNKS         — GPU memory chunks (default 8)

Usage:
  # Stage 0, both methods, seed 42
  python team/scripts/phase20_1_5b_progressive.py

  # Stage 1, EVQ only, seed 137
  PHASE20_STAGE=1 PHASE20_METHOD=evq PHASE20_SEED=137 \\
    python team/scripts/phase20_1_5b_progressive.py

  # Memory estimation
  PHASE20_DRY_RUN=1 python team/scripts/phase20_1_5b_progressive.py

  # Custom data & eval
  PHASE20_STAGE=2 PHASE20_DATASET=wikitext-103 \\
    PHASE20_EVAL_LENGTHS=8192,16384,32768,65536 \\
    python team/scripts/phase20_1_5b_progressive.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# HuggingFace mirror for mainland China
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "core_text_phases"))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "scripts" / "supporting_eval"))

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
HEAD_DIM = 64

STAGE = int(os.environ.get("PHASE20_STAGE", "0"))
METHOD = os.environ.get("PHASE20_METHOD", "both").strip().lower()
SEED = int(os.environ.get("PHASE20_SEED", "42"))
DATASET = os.environ.get("PHASE20_DATASET", "fineweb-edu").strip() or "fineweb-edu"

DRY_RUN = os.environ.get("PHASE20_DRY_RUN", "").strip()
INIT_CKPT = os.environ.get("PHASE20_INIT_CKPT", "").strip()

PASSKEY_MIX_RATIO = float(
    os.environ.get(
        "PHASE20_PASSKEY_MIX_RATIO",
        str(resolve_passkey_mix_ratio(default=0.05)),
    )
)

WORK = Path(
    os.environ.get(
        "PHASE20_WORK",
        "/root/autodl-tmp/evq_phase20",
    )
)
DATA_CACHE_DIR = Path(os.environ.get("PHASE20_DATA_CACHE_DIR", str(WORK / "data")))

GPU_CHUNKS = int(os.environ.get("PHASE20_GPU_CHUNKS", "8"))

CKPT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
CKPT_EVAL_CHUNKS = 10
EVAL_CHUNKS = 10

PK_TRIALS = 10
CKPT_PK_TRIALS = 5
PK_DEPTHS = [0.5]

# ── Stage configurations ───────────────────────────────────────────────────
# Each stage has unique τ* based on τ* = head_dim / sqrt(L_train)

STAGE_CONFIGS = {
    0: {
        "seq_len": 2048,
        "tokens": 1_000_000_000,
        "micro_bs": 4,
        "grad_accum": 2,
        "lr": 1e-4,
        "tau_star": HEAD_DIM / math.sqrt(2048),
    },
    1: {
        "seq_len": 4096,
        "tokens": 1_500_000_000,
        "micro_bs": 2,
        "grad_accum": 4,
        "lr": 1e-4,
        "tau_star": HEAD_DIM / math.sqrt(4096),
    },
    2: {
        "seq_len": 8192,
        "tokens": 1_000_000_000,
        "micro_bs": 1,
        "grad_accum": 8,
        "lr": 1e-4,
        "tau_star": HEAD_DIM / math.sqrt(8192),
    },
}

# ── Model config: 1.5B ─────────────────────────────────────────────────────

CFG_1_5B = dict(
    vocab_size=50304,
    hidden_size=1536,
    num_layers=32,
    num_heads=24,
    head_dim=64,
    intermediate_size=6144,
)

# Default eval lengths per stage
DEFAULT_EVAL_LENGTHS = {
    0: [2048, 4096, 8192, 16384],
    1: [4096, 8192, 16384, 32768],
    2: [8192, 16384, 32768, 65536],
}

# ── Helpers ────────────────────────────────────────────────────────────────


def geometric_inv_freq(dim=HEAD_DIM, base=BASE) -> torch.Tensor:
    """Standard geometric RoPE frequencies: inv_freq_i = 1/(base^(2i/dim))."""
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def save_json(path: Path, data) -> None:
    """Save JSON with automatic parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_state(path: str) -> Dict:
    """Load checkpoint state, handling torch.compile and RoPE buffers."""
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
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }

    # Remove RoPE buffers (will be rebuilt)
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]

    return state


def load_model_with_target_inv(
    cfg: Dict, ckpt_path: str, target_inv: torch.Tensor
) -> GPT:
    """Load model from checkpoint and force-set inv_freq to target."""
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


def estimate_memory(cfg: Dict, stage_cfg: Dict) -> None:
    """Estimate and print peak GPU memory usage."""
    seq_len = stage_cfg["seq_len"]
    micro_bs = stage_cfg["micro_bs"]
    grad_accum = stage_cfg["grad_accum"]

    # Model parameters
    vocab_size = cfg["vocab_size"]
    hidden = cfg["hidden_size"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    intermediate = cfg["intermediate_size"]

    # Parameter count estimate
    # Embedding: vocab_size * hidden
    # Per layer: attn (hidden^2 + 2*hidden) + mlp (hidden*intermediate + intermediate*hidden)
    # Total: vocab_size*hidden + num_layers * (hidden^2 + 2*hidden + 2*hidden*intermediate)
    param_bytes = (
        vocab_size * hidden * 4
        + num_layers
        * (
            hidden * hidden * 4 * 3  # qkv + out
            + 2 * hidden * 4  # biases
            + hidden * intermediate * 4 * 2  # up + down
            + intermediate * 4  # gate bias
        )
    )

    # Activation memory per token
    # Forward: hidden + intermediate per layer
    # Backward: hidden + intermediate per layer (stored)
    # KV cache: 2 * hidden * (num_heads/num_heads) * seq_len per layer
    act_per_token = (
        num_layers * (hidden + intermediate) * 2 * 4  # forward + backward
        + num_layers * 2 * hidden * seq_len * 4  # KV cache
    )

    # Total batch token count (effective)
    effective_bs = micro_bs * grad_accum
    total_tokens = seq_len * effective_bs

    # Memory estimate
    param_mem = param_bytes / 1e9
    act_mem = act_per_token * total_tokens / 1e9
    optim_mem = param_bytes * 2 / 1e9  # momentum + variance

    print(f"\n  {'='*60}")
    print(f"  Memory Estimation (stage {STAGE})")
    print(f"  {'='*60}")
    print(f"  Model params: {param_bytes/1e9:.2f} GB ({param_bytes/1e12*8:.2f} GiB in float32)")
    print(f"  Activations: {act_mem:.2f} GB (micro_bs={micro_bs}, grad_accum={grad_accum})")
    print(f"  Optimizer: {optim_mem:.2f} GB (AdamW)")
    print(f"  Total est: {param_mem + act_mem + optim_mem:.2f} GB")
    print(f"  {'='*60}\n")


def build_mixed_train_tensor(
    train_data: torch.Tensor, filler: torch.Tensor, tokenizer
) -> torch.Tensor:
    """Mix passkey supervision into LM training data."""
    if PASSKEY_MIX_RATIO <= 0:
        print("  [passkey-train] mix disabled (pure LM)")
        return train_data

    mix_pct = int(round(PASSKEY_MIX_RATIO * 100))
    seq_len = train_data.shape[1]
    cache_path = DATA_CACHE_DIR / (
        f"mixed_{DATASET}_seed{SEED}_pk{mix_pct}_{train_data.shape[0]}_{seq_len}.pt"
    )

    if cache_path.exists():
        print(f"  [passkey-train] loading cached: {cache_path}")
        mixed = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [passkey-train] cached shape={tuple(mixed.shape)}")
        return mixed

    mixed = train_data.clone()
    n_total = mixed.shape[0]
    n_pk = int(n_total * PASSKEY_MIX_RATIO)

    if n_pk <= 0:
        return mixed

    print(f"  [passkey-train] generating {n_pk} passkey samples...")
    pk_indices = torch.randperm(n_total)[:n_pk]
    t0 = time.time()

    for i, idx in enumerate(pk_indices.tolist(), start=1):
        mixed[idx] = make_passkey_training_sample(
            filler_tokens=filler,
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=idx,
        )
        if i % 10_000 == 0 or i == n_pk:
            print(f"    generated {i}/{n_pk} samples")

    torch.save(mixed, cache_path)
    print(
        f"  [passkey-train] complete in {time.time() - t0:.1f}s, "
        f"saved to {cache_path}"
    )
    return mixed


# ── Training loop ──────────────────────────────────────────────────────────


def train_model_ga(model, data: torch.Tensor, cfg: Dict, seed=42, on_step_end=None) -> Tuple[GPT, float]:
    """Train with gradient accumulation and cosine LR schedule."""
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
            f"not enough chunks: chunks={total_seqs}, effective_bs={effective_bs}"
        )

    print(
        f"  Train config: lr={lr:.1e}, min_lr={min_lr:.1e}, "
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

            # LR schedule
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

                ctx = (
                    torch.amp.autocast("cuda", dtype=DTYPE)
                    if USE_AUTOCAST
                    else nullcontext()
                )
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
                eta = (
                    elapsed / step_num * (steps - step_num) if step_num > 0 else 0
                )
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


# ── Single stage run ───────────────────────────────────────────────────────


def run_single_stage(
    tag: str,
    init_ckpt: str,
    target_inv: torch.Tensor,
    cfg: Dict,
    train_data: torch.Tensor,
) -> str:
    """Train one method for one stage."""
    run_dir = WORK / f"seed{SEED}" / tag
    ckpt_dir = run_dir / "checkpoints"
    model_path = run_dir / "model.pt"

    if model_path.exists():
        print(f"\n[SKIP] {tag}: model.pt exists")
        return str(model_path)

    if not Path(init_ckpt).exists():
        raise FileNotFoundError(f"missing checkpoint: {init_ckpt}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  TRAIN: {tag}")
    print(f"  Init: {init_ckpt}")
    print(f"  L={cfg['seq_len']}, tokens={cfg['train_tokens']/1e6:.0f}M, seed={SEED}")
    print(f"{'=' * 72}")

    set_seed(SEED)
    model = load_model_with_target_inv(cfg, init_ckpt, target_inv)

    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled (reduce-overhead)")

    print(
        f"  inv_freq set: max={target_inv.max().item():.8f} "
        f"min={target_inv.min().item():.8f}"
    )

    total_steps = len(train_data) // (cfg["micro_batch_size"] * cfg["grad_accum"])
    checkpoint_steps = sorted(
        set(
            max(1, min(total_steps, int(total_steps * frac)))
            for frac in CKPT_FRACTIONS
        )
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
        print(f"\n  [CKPT] {tag} {ckpt_name} ({frac:.1%})")
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
    save_json(
        run_dir / "train_meta.json",
        dict(
            method=tag,
            base=BASE,
            seed=SEED,
            init_ckpt=init_ckpt,
            stage=STAGE,
            train_tokens=cfg["train_tokens"],
            seq_len=cfg["seq_len"],
            tau=cfg.get("tau", 0.0),
            train_sec=round(train_elapsed, 1),
            checkpoint_fractions=CKPT_FRACTIONS,
        ),
    )

    print(f"\n  Training done: {model_path} ({train_elapsed/60:.1f} min)")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return str(model_path)


def run_eval_for_tag(
    tag: str,
    target_inv: torch.Tensor,
    cfg: Dict,
    val_data: torch.Tensor,
    filler: torch.Tensor,
    tok,
    eval_lengths: list,
) -> Dict:
    """Comprehensive evaluation of trained model."""
    run_dir = WORK / f"seed{SEED}" / tag
    model_path = run_dir / "model.pt"
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n[SKIP-EVAL] {tag}: result.json exists")
        with open(result_file) as f:
            return json.load(f)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    print(f"\n{'=' * 72}")
    print(f"  EVAL: {tag}")
    print(f"{'=' * 72}")

    # Load model WITHOUT torch.compile for full VRAM
    model = load_model_with_target_inv(cfg, str(model_path), target_inv)
    model.eval()
    print(f"  Model loaded, eval_lengths={eval_lengths}")

    # PPL evaluation
    with torch.no_grad():
        ppl = eval_model(model, val_data, eval_lengths, EVAL_CHUNKS)

    # Passkey evaluation
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    pk_lengths = [L for L in [2048, 4096, 8192, 16384] if L <= cfg["seq_len"]]

    with torch.no_grad():
        pk = eval_passkey_nll_gap(
            model,
            tok,
            filler,
            lengths=pk_lengths,
            depths=PK_DEPTHS,
            num_trials=PK_TRIALS,
        )

    result = dict(
        method=tag,
        base=BASE,
        seed=SEED,
        model="1.5B",
        stage=STAGE,
        seq_len=cfg["seq_len"],
        tau=cfg.get("tau", 0.0),
        ppl=ppl,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        config=dict(
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            lr=cfg["lr"],
            micro_bs=cfg["micro_batch_size"],
            grad_accum=cfg["grad_accum"],
            eval_lengths=eval_lengths,
        ),
    )

    save_json(result_file, result)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


# ── Dry run mode ───────────────────────────────────────────────────────────


def run_dry_run():
    """Build model, estimate memory, run 1 step, exit."""
    print(f"\n{'#' * 72}")
    print("  PHASE 20: DRY RUN (Memory Estimation)")
    print(f"{'#' * 72}")

    if STAGE not in STAGE_CONFIGS:
        print(f"  ERROR: invalid stage {STAGE}")
        sys.exit(1)

    stage_cfg = STAGE_CONFIGS[STAGE]
    cfg = CFG_1_5B.copy()
    cfg.update(
        {
            "max_position_embeddings": stage_cfg["seq_len"],
            "seq_len": stage_cfg["seq_len"],
            "train_tokens": stage_cfg["tokens"],
            "micro_batch_size": stage_cfg["micro_bs"],
            "grad_accum": stage_cfg["grad_accum"],
            "lr": stage_cfg["lr"],
            "batch_size": stage_cfg["micro_bs"] * stage_cfg["grad_accum"],
        }
    )

    print(f"  Stage: {STAGE}")
    print(f"  Config: L={stage_cfg['seq_len']}, vocab={cfg['vocab_size']}, "
          f"hidden={cfg['hidden_size']}, layers={cfg['num_layers']}")

    estimate_memory(cfg, stage_cfg)

    # Build model
    print("  Building model...")
    inv_freq = geometric_inv_freq()
    model = GPT(cfg, inv_freq).to(DEVICE)
    print(f"  Model built: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Test forward pass
    print("  Running test forward pass...")
    model.train()
    dummy_input = torch.randint(0, cfg["vocab_size"], (2, stage_cfg["seq_len"]))
    dummy_input = dummy_input.to(DEVICE)

    t0 = time.time()
    with torch.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext():
        logits = model(dummy_input)
    elapsed = time.time() - t0

    if DEVICE == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Forward pass: {elapsed:.2f}s, peak GPU: {peak_mem:.1f}GB")
        torch.cuda.reset_peak_memory_stats()
    else:
        print(f"  Forward pass: {elapsed:.2f}s")

    del model, dummy_input, logits
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n  Dry run complete. Use PHASE20_DRY_RUN=0 to proceed with training.")
    sys.exit(0)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    if DRY_RUN:
        run_dry_run()

    if STAGE not in STAGE_CONFIGS:
        print(f"ERROR: invalid stage {STAGE} (must be 0/1/2)")
        sys.exit(1)

    print("#" * 72)
    print("  Phase 20: 1.5B Progressive Length Extension (EVQ-Cosh NeurIPS)")
    print("#" * 72)

    stage_cfg = STAGE_CONFIGS[STAGE]
    tau_star = stage_cfg["tau_star"]

    # Override stage config from env
    if os.environ.get("PHASE20_TOKENS"):
        stage_cfg["tokens"] = int(os.environ["PHASE20_TOKENS"])
    if os.environ.get("PHASE20_MICRO_BATCH_SIZE"):
        stage_cfg["micro_bs"] = int(os.environ["PHASE20_MICRO_BATCH_SIZE"])
    if os.environ.get("PHASE20_GRAD_ACCUM"):
        stage_cfg["grad_accum"] = int(os.environ["PHASE20_GRAD_ACCUM"])

    cfg = CFG_1_5B.copy()
    cfg.update(
        {
            "max_position_embeddings": stage_cfg["seq_len"],
            "seq_len": stage_cfg["seq_len"],
            "train_tokens": stage_cfg["tokens"],
            "micro_batch_size": stage_cfg["micro_bs"],
            "grad_accum": stage_cfg["grad_accum"],
            "lr": stage_cfg["lr"],
            "batch_size": stage_cfg["micro_bs"] * stage_cfg["grad_accum"],
        }
    )

    eval_lengths = DEFAULT_EVAL_LENGTHS[STAGE]
    if os.environ.get("PHASE20_EVAL_LENGTHS"):
        eval_lengths = [
            int(x)
            for x in os.environ["PHASE20_EVAL_LENGTHS"].split(",")
            if x.strip()
        ]

    print(f"  device={DEVICE} dtype={DTYPE} autocast={USE_AUTOCAST}")
    print(f"  stage={STAGE} (L={stage_cfg['seq_len']}, τ*={tau_star:.6f})")
    print(f"  tokens={stage_cfg['tokens']/1e6:.0f}M seed={SEED}")
    print(f"  method={METHOD} passkey_mix={PASSKEY_MIX_RATIO:.2%}")
    print(f"  dataset={DATASET} eval_lengths={eval_lengths}")
    print(f"  work={WORK}")

    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Load data
    # ────────────────────────────────────────────────────────────────────

    print("\n  Loading data...")
    train_data = load_data(
        tok=None,
        num_tokens=stage_cfg["tokens"],
        seq_len=stage_cfg["seq_len"],
        dataset=DATASET,
        cache_dir=str(DATA_CACHE_DIR),
    )
    print(f"  train_data: {train_data.shape} ({train_data.numel()/1e6:.0f}M tokens)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    print("  Loading validation data...")
    val_data = load_val(
        tok=tok,
        num_tokens=5_000_000,
        dataset=DATASET,
        cache_dir=str(DATA_CACHE_DIR),
    )
    filler = val_data[:50000]

    # Mix passkey samples
    train_data = build_mixed_train_tensor(train_data, filler, tok)

    # ────────────────────────────────────────────────────────────────────
    # Determine init checkpoints
    # ────────────────────────────────────────────────────────────────────

    if STAGE == 0:
        # Stage 0: use pretrained or fail
        if not INIT_CKPT:
            print(
                "ERROR: stage 0 requires PHASE20_INIT_CKPT "
                "(path to pretrained 1.5B model)"
            )
            sys.exit(1)
        geo_init = INIT_CKPT
        evq_init = INIT_CKPT
    else:
        # Stage > 0: auto-discover from previous stage
        prev_stage = STAGE - 1
        if INIT_CKPT:
            # Manual override
            geo_init = INIT_CKPT
            evq_init = INIT_CKPT
        else:
            # Auto-discover from previous stage results
            prev_work = WORK / f"seed{SEED}"
            geo_tag = f"geo_1_5b_stage{prev_stage}"
            evq_tag = f"evq_1_5b_stage{prev_stage}"

            geo_init = str(prev_work / geo_tag / "model.pt")
            evq_init = str(prev_work / evq_tag / "model.pt")

            if not Path(geo_init).exists():
                raise FileNotFoundError(f"missing geo checkpoint: {geo_init}")
            if not Path(evq_init).exists():
                raise FileNotFoundError(f"missing evq checkpoint: {evq_init}")

    print(f"  geo_init: {geo_init}")
    print(f"  evq_init: {evq_init}")

    # ────────────────────────────────────────────────────────────────────
    # Build run list
    # ────────────────────────────────────────────────────────────────────

    runs = []
    if METHOD in {"geo", "both"}:
        runs.append(
            dict(
                tag=f"geo_1_5b_stage{STAGE}",
                init_ckpt=geo_init,
                target_inv=geometric_inv_freq(),
                tau=0.0,
            )
        )
    if METHOD in {"evq", "both"}:
        runs.append(
            dict(
                tag=f"evq_1_5b_stage{STAGE}",
                init_ckpt=evq_init,
                target_inv=evq_cosh_inv_freq(
                    head_dim=HEAD_DIM, tau=tau_star, base=BASE
                ),
                tau=tau_star,
            )
        )

    print(f"\n  Runs: {[r['tag'] for r in runs]}")

    # ────────────────────────────────────────────────────────────────────
    # PHASE 1: TRAINING
    # ────────────────────────────────────────────────────────────────────

    print(f"\n{'#' * 72}")
    print("  PHASE 1: TRAINING")
    print(f"{'#' * 72}")

    model_paths = {}
    for run in runs:
        cfg_with_tau = cfg.copy()
        cfg_with_tau["tau"] = run["tau"]

        mp = run_single_stage(
            tag=run["tag"],
            init_ckpt=run["init_ckpt"],
            target_inv=run["target_inv"],
            cfg=cfg_with_tau,
            train_data=train_data,
        )
        model_paths[run["tag"]] = mp

    # Free training data before eval
    del train_data
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n  Training data freed, cache cleared.")

    # ────────────────────────────────────────────────────────────────────
    # PHASE 2: EVALUATION
    # ────────────────────────────────────────────────────────────────────

    print(f"\n{'#' * 72}")
    print("  PHASE 2: EVALUATION")
    print(f"{'#' * 72}")

    results = {}
    for run in runs:
        cfg_with_tau = cfg.copy()
        cfg_with_tau["tau"] = run["tau"]

        res = run_eval_for_tag(
            tag=run["tag"],
            target_inv=run["target_inv"],
            cfg=cfg_with_tau,
            val_data=val_data,
            filler=filler,
            tok=tok,
            eval_lengths=eval_lengths,
        )
        results[run["tag"]] = res

    # ────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────

    summary = dict(
        phase="phase20_1_5b_progressive",
        model="1.5B",
        stage=STAGE,
        seq_len=stage_cfg["seq_len"],
        tokens=stage_cfg["tokens"],
        seed=SEED,
        tau_star=tau_star,
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        dataset=DATASET,
        runs=list(results.keys()),
        results=results,
    )
    summary_path = WORK / f"phase20_stage{STAGE}_summary.json"
    save_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    # ────────────────────────────────────────────────────────────────────
    # Results table
    # ────────────────────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"  FINAL PPL COMPARISON (Stage {STAGE})")
    print(f"{'=' * 80}")
    for tag, res in results.items():
        ppl = res.get("ppl", {})
        line = f"  {tag:>40s}:"
        for L in eval_lengths:
            v = ppl.get(str(L))
            if v is not None:
                line += f"  {L:>6d}={v:>6.1f}"
            else:
                line += f"  {L:>6d}={'--':>6s}"
        print(line)

    print(f"\n  Passkey depths={PK_DEPTHS}")
    for tag, res in results.items():
        pk_summ = res.get("passkey_summary", {})
        if pk_summ:
            line = f"  {tag:>40s} passkey:"
            for depth_str in sorted(pk_summ.keys()):
                success_rate = pk_summ[depth_str].get("success_rate", 0.0)
                line += f"  d={float(depth_str):.1f}:{success_rate*100:.0f}%"
            print(line)

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
