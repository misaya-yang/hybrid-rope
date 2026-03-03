#!/usr/bin/env python3
"""
Phase 14B: Fine-tune 750M Geo & Hybrid checkpoints on passkey data for ~50 steps,
then evaluate passkey retrieval at 2K/4K/8K with and without YaRN@4x.

Idea: the base models have weak passkey retrieval at long contexts because they
only saw 3% passkey mix during training at seq_len=2048.  A short fine-tune on
pure passkey data at longer contexts (4096/8192) should teach them to retrieve,
and then we compare Geo vs Hybrid to see which benefits more.

Usage:
    python phase14b_passkey_finetune.py [--steps 50] [--lr 1e-4] [--ft_seq_len 4096]
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST, set_seed
from eval_passkey_scratch import make_passkey_training_sample, eval_passkey_nll_gap

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = 500_000.0
DIM = 64
SEED = 42
TAU = 1.5

CFG_750M = dict(
    vocab_size=50304,
    hidden_size=1536,
    num_layers=18,
    num_heads=24,
    head_dim=64,
    intermediate_size=6144,
    max_position_embeddings=2048,
)

WORK = Path("/root/autodl-tmp/evq_phase14b")
DATA_CACHE_DIR = Path("/root/autodl-tmp/evq_phase9/data")

CKPT_GEO = Path("/root/autodl-tmp/evq_phase9/seed42/geo_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt")
CKPT_HYBRID = Path("/root/autodl-tmp/evq_phase9/seed42/hybrid1.5_r16_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt")


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=TAU, r=16):
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


def apply_yarn_scaling(inv_freq, scale_factor, original_max_len=2048):
    """YaRN progressive frequency scaling."""
    wavelength = 2 * math.pi / inv_freq
    low_freq_wavelen = original_max_len / 1.0
    high_freq_wavelen = original_max_len / 4.0
    smooth = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth = smooth.clamp(0, 1)
    scaled_freq = inv_freq / scale_factor
    new_freq = (1 - smooth) * scaled_freq + smooth * inv_freq
    return new_freq


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_model(inv_freq, ckpt_path):
    """Load GPT model from checkpoint."""
    cfg = CFG_750M.copy()
    model = GPT(cfg, inv_freq)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(DEVICE)
    return model


def generate_passkey_batch(filler_tokens, tokenizer, batch_size, seq_len, seed_offset=0):
    """Generate a batch of passkey training samples."""
    samples = []
    for i in range(batch_size):
        sample = make_passkey_training_sample(
            filler_tokens=filler_tokens,
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=seed_offset + i,
        )
        samples.append(sample)
    return torch.stack(samples, dim=0)


def finetune_passkey(model, filler_tokens, tokenizer, args):
    """Fine-tune model on passkey data for a small number of steps."""
    model.train()
    lr = args.lr
    steps = args.steps
    batch_size = args.batch_size
    ft_seq_len = args.ft_seq_len

    # Extend RoPE cache for the fine-tune sequence length
    model.extend_rope(ft_seq_len + 64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Linear warmup for 10% of steps, then constant
    warmup_steps = max(1, steps // 10)

    print(f"  Fine-tuning: steps={steps}, lr={lr}, bs={batch_size}, seq_len={ft_seq_len}, warmup={warmup_steps}")

    t0 = time.time()
    losses = []

    for s in range(steps):
        # LR schedule: linear warmup then constant
        if s < warmup_steps:
            cur_lr = lr * (s + 1) / warmup_steps
        else:
            cur_lr = lr
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Generate fresh passkey batch each step
        batch = generate_passkey_batch(
            filler_tokens, tokenizer, batch_size, ft_seq_len + 1,
            seed_offset=s * batch_size + 100000,
        ).to(DEVICE)

        opt.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
        with ctx:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step_loss = loss.item()
        losses.append(step_loss)

        if s % 10 == 0 or s == steps - 1:
            elapsed = time.time() - t0
            print(f"    step {s}/{steps}  loss={step_loss:.4f}  lr={cur_lr:.2e}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Fine-tune done in {elapsed:.1f}s ({steps} steps)")
    return losses


def eval_with_yarn(model, inv_freq_orig, tokenizer, filler_tokens, lengths, yarn_factor, tag):
    """Evaluate passkey with YaRN applied to a model."""
    if yarn_factor is not None and yarn_factor > 1.0:
        # Apply YaRN scaling to the model's RoPE inv_freq
        yarn_inv_freq = apply_yarn_scaling(inv_freq_orig, yarn_factor)
        # Replace the inv_freq in the model's RoPE
        rope = model.blocks[0].attn.rope
        rope.inv_freq.copy_(yarn_inv_freq)
        rope._build(max(lengths) + 64)
        label = f"{tag}+YaRN@{yarn_factor}x"
    else:
        # Restore original inv_freq
        rope = model.blocks[0].attn.rope
        rope.inv_freq.copy_(inv_freq_orig)
        rope._build(max(lengths) + 64)
        label = tag

    print(f"\n  Evaluating passkey: {label}")
    model.eval()
    with torch.no_grad():
        result = eval_passkey_nll_gap(
            model, tokenizer, filler_tokens,
            lengths=lengths,
            depths=[0.5],
            num_trials=20,
            seed=42,
        )

    g = result.get("global", {})
    print(f"    {label}: retrieval={g.get('retrieval_rate', 0):.4f}, gap={g.get('mean_nll_gap', 0):.4f}")

    return {
        "label": label,
        "global": g,
        "summary": result.get("summary", {}),
    }


def run_one_model(tag, inv_freq, ckpt_path, filler_tokens, tokenizer, args):
    """Run full pipeline for one model: eval_before -> finetune -> eval_after -> eval_yarn."""
    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")

    run_dir = WORK / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"  Loading checkpoint: {ckpt_path}")
    model = load_model(inv_freq, ckpt_path)
    inv_freq_orig = inv_freq.clone()

    eval_lengths = [2048, 4096, 8192]

    # 1. Eval BEFORE fine-tune
    print("\n  === Pre-finetune passkey eval ===")
    pre_result = eval_with_yarn(model, inv_freq_orig, tokenizer, filler_tokens, eval_lengths, None, f"{tag}_pre")

    pre_yarn = eval_with_yarn(model, inv_freq_orig, tokenizer, filler_tokens, eval_lengths, 4.0, f"{tag}_pre")

    # Restore original inv_freq before fine-tuning
    rope = model.blocks[0].attn.rope
    rope.inv_freq.copy_(inv_freq_orig)
    rope._build(args.ft_seq_len + 64)

    # 2. Fine-tune
    print("\n  === Fine-tuning on passkey data ===")
    losses = finetune_passkey(model, filler_tokens, tokenizer, args)

    # 3. Eval AFTER fine-tune (no YaRN)
    print("\n  === Post-finetune passkey eval ===")
    post_result = eval_with_yarn(model, inv_freq_orig, tokenizer, filler_tokens, eval_lengths, None, f"{tag}_post")

    # 4. Eval AFTER fine-tune (with YaRN@4x)
    post_yarn = eval_with_yarn(model, inv_freq_orig, tokenizer, filler_tokens, eval_lengths, 4.0, f"{tag}_post")

    # Restore original before saving
    rope.inv_freq.copy_(inv_freq_orig)

    # Save fine-tuned checkpoint
    ft_ckpt_path = run_dir / "finetuned.pt"
    torch.save(model.state_dict(), ft_ckpt_path)
    print(f"  Saved fine-tuned model: {ft_ckpt_path}")

    result = {
        "tag": tag,
        "ckpt_path": str(ckpt_path),
        "ft_steps": args.steps,
        "ft_lr": args.lr,
        "ft_seq_len": args.ft_seq_len,
        "ft_batch_size": args.batch_size,
        "ft_losses": losses,
        "pre_finetune": pre_result,
        "pre_finetune_yarn4x": pre_yarn,
        "post_finetune": post_result,
        "post_finetune_yarn4x": post_yarn,
    }

    save_json(run_dir / "result.json", result)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ft_seq_len", type=int, default=4096,
                        help="Sequence length for passkey fine-tuning (default: 4096)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("#" * 70)
    print("  Phase 14B: Passkey Fine-tune (50 steps) + Eval")
    print(f"  steps={args.steps}, lr={args.lr}, ft_seq_len={args.ft_seq_len}, bs={args.batch_size}")
    print("#" * 70)

    WORK.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load filler tokens from validation data
    val_path = DATA_CACHE_DIR / "val_fineweb-edu_5000000.pt"
    print(f"\n  Loading filler tokens from: {val_path}")
    val_data = torch.load(val_path, map_location="cpu", weights_only=True)
    filler = val_data.reshape(-1)[:100000]
    print(f"  Filler tokens: {filler.shape}")

    # Run both models
    results = {}

    results["geo"] = run_one_model(
        tag="geo_750m",
        inv_freq=geometric_inv_freq(),
        ckpt_path=CKPT_GEO,
        filler_tokens=filler,
        tokenizer=tok,
        args=args,
    )

    results["hybrid"] = run_one_model(
        tag="hybrid_750m",
        inv_freq=hybrid_evq_inv_freq(),
        ckpt_path=CKPT_HYBRID,
        filler_tokens=filler,
        tokenizer=tok,
        args=args,
    )

    # Summary
    print(f"\n{'='*70}")
    print("  PHASE 14B SUMMARY")
    print(f"{'='*70}")

    for key in ["geo", "hybrid"]:
        r = results[key]
        tag = r["tag"]
        pre_g = r["pre_finetune"]["global"]
        post_g = r["post_finetune"]["global"]
        post_y = r["post_finetune_yarn4x"]["global"]

        print(f"\n  {tag}:")
        print(f"    Pre-FT:          retrieval={pre_g.get('retrieval_rate', 0):.4f}, gap={pre_g.get('mean_nll_gap', 0):.4f}")
        print(f"    Post-FT:         retrieval={post_g.get('retrieval_rate', 0):.4f}, gap={post_g.get('mean_nll_gap', 0):.4f}")
        print(f"    Post-FT+YaRN@4x: retrieval={post_y.get('retrieval_rate', 0):.4f}, gap={post_y.get('mean_nll_gap', 0):.4f}")

    # Save overall summary
    save_json(WORK / "phase14b_summary.json", results)
    print(f"\n  Saved: {WORK}/phase14b_summary.json")


if __name__ == "__main__":
    main()
