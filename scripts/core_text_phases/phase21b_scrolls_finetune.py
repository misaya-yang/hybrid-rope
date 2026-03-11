#!/usr/bin/env python3
"""
Phase 21B: SCROLLS task-specific finetuning for 454M models.

Finetune Phase 17c pretrained checkpoints (L_train=2048) on SCROLLS tasks
at L=8192 (4× extrapolation). Compare EVQ vs Geo on downstream performance.

Supported tasks:
  - gov_report: Government report summarization (ROUGE)
  - qmsum: Meeting summarization (ROUGE)
  - quality: Long-document multiple choice QA (Accuracy)

Data format: causal LM with loss masking on answer tokens only.
  input:  "Summarize the following document:\n{context}\n\nSummary:"
  target: " {answer}"
  Loss computed only on target tokens.

Usage:
  python phase21b_scrolls_finetune.py \
      --init_ckpt /path/to/model.pt \
      --rope geo --base 500000 \
      --task gov_report --seq_len 8192 --yarn 0 \
      --lr 1e-5 --steps 25000 --dropout 0.1 \
      --seed 42 \
      --output_dir results/phase21b/gov_report_raw/geo_seed42/
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    evq_cosh_inv_freq,
    set_seed,
)

# ── Model configs ──────────────────────────────────────────────────────

TIER_CONFIGS = {
    "454m": dict(
        vocab_size=50304, hidden_size=1024, num_layers=24,
        num_heads=16, head_dim=64, intermediate_size=4096,
    ),
    "750m": dict(
        vocab_size=50304, hidden_size=1536, num_layers=18,
        num_heads=24, head_dim=64, intermediate_size=6144,
    ),
}

# ── Task-specific prompts ──────────────────────────────────────────────

TASK_PROMPTS = {
    "gov_report": {
        "input_template": "Summarize the following government report.\n\n{context}\n\nSummary:",
        "eval_metric": "rouge",
    },
    "qmsum": {
        "input_template": "Summarize the following meeting transcript with respect to the query.\n\nTranscript:\n{context}\n\nQuery: {query}\n\nSummary:",
        "eval_metric": "rouge",
    },
    "quality": {
        "input_template": "Read the following passage and answer the question.\n\n{context}\n\nQuestion: {question}\nOptions:\n{options}\n\nAnswer:",
        "eval_metric": "accuracy",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 21B: SCROLLS finetuning")
    p.add_argument("--init_ckpt", required=True, help="Path to pretrained model.pt")
    p.add_argument("--tier", default="750m", choices=["454m", "750m"],
                   help="Model tier (default: 750m)")
    p.add_argument("--rope", required=True, choices=["geo", "evq"])
    p.add_argument("--tau", type=float, default=1.4142, help="EVQ tau (only for --rope evq)")
    p.add_argument("--base", type=float, default=500000.0, help="RoPE base frequency")
    p.add_argument("--task", required=True, choices=["gov_report", "qmsum", "quality"])
    p.add_argument("--seq_len", type=int, default=8192, help="Finetuning sequence length")
    p.add_argument("--yarn", type=int, default=0, choices=[0, 1],
                   help="0=raw finetune, 1=apply YaRN scaling")
    p.add_argument("--yarn_scale", type=float, default=4.0, help="YaRN scale factor")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--micro_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", type=str, default="",
                   help="Local SCROLLS data directory (skip HF download)")
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="Limit training samples (0=all)")
    p.add_argument("--eval_every", type=int, default=5000,
                   help="Evaluate every N steps (0=only at end)")
    p.add_argument("--max_gen_tokens", type=int, default=512,
                   help="Max tokens to generate during eval")
    p.add_argument("--eval_samples", type=int, default=200,
                   help="Number of validation samples for ROUGE eval")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────

def geometric_inv_freq(dim=64, base=500000.0):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def apply_yarn_scaling(inv_freq, scale, original_max_pos=2048):
    """Apply YaRN NTK-aware scaling to inv_freq."""
    # NTK-aware interpolation: scale the base frequency
    # Equivalent to: base_new = base * scale^(dim/(dim-2))
    dim = len(inv_freq) * 2
    factor = scale ** (dim / (dim - 2))
    return inv_freq / factor


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
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


def add_dropout_to_model(model, p=0.1):
    """Monkey-patch dropout into the model for finetuning.

    Adds dropout after attention output and after MLP output in each block.
    """
    if p <= 0:
        return model

    for block in model.blocks:
        # Wrap attention forward with dropout
        orig_attn_forward = block.attn.forward
        attn_drop = nn.Dropout(p).to(next(block.parameters()).device)

        def make_attn_wrapper(orig_fn, drop):
            def wrapped(x):
                return drop(orig_fn(x))
            return wrapped
        block.attn.forward = make_attn_wrapper(orig_attn_forward, attn_drop)

        # Wrap MLP forward with dropout
        orig_mlp_forward = block.mlp.forward
        mlp_drop = nn.Dropout(p).to(next(block.parameters()).device)

        def make_mlp_wrapper(orig_fn, drop):
            def wrapped(x):
                return drop(orig_fn(x))
            return wrapped
        block.mlp.forward = make_mlp_wrapper(orig_mlp_forward, mlp_drop)

    print(f"  Dropout {p} added to all {len(model.blocks)} blocks (attn + mlp)")
    return model


# ── SCROLLS Data Loading ──────────────────────────────────────────────

def load_scrolls_data(task, split, data_dir=""):
    """Load SCROLLS task data from HuggingFace or local cache."""
    if data_dir:
        local_path = os.path.join(data_dir, task, f"{split}.jsonl")
        if os.path.exists(local_path):
            print(f"  Loading local: {local_path}")
            data = []
            with open(local_path) as f:
                for line in f:
                    data.append(json.loads(line))
            return data

    print(f"  Downloading tau/scrolls/{task} split={split}...")
    from datasets import load_dataset
    ds = load_dataset("tau/scrolls", task, split=split, trust_remote_code=True)
    return list(ds)


def format_sample(sample, task):
    """Format a SCROLLS sample into (prompt_str, answer_str)."""
    info = TASK_PROMPTS[task]

    if task == "gov_report":
        context = sample.get("input", "")
        answer = sample.get("output", "")
        prompt = info["input_template"].format(context=context)

    elif task == "qmsum":
        # QMSum: input field contains both transcript and query
        raw_input = sample.get("input", "")
        answer = sample.get("output", "")
        # SCROLLS format: input is the concatenation, parse query if present
        if "\nQuery:" in raw_input:
            parts = raw_input.rsplit("\nQuery:", 1)
            context = parts[0]
            query = parts[1].strip()
        else:
            context = raw_input
            query = "Summarize the key points."
        prompt = info["input_template"].format(context=context, query=query)

    elif task == "quality":
        raw_input = sample.get("input", "")
        answer = sample.get("output", "")
        # QuALITY SCROLLS format: input contains passage, question, options
        prompt = f"Read the following and answer the question.\n\n{raw_input}\n\nAnswer:"

    else:
        raise ValueError(f"Unknown task: {task}")

    return prompt, f" {answer}"


def prepare_finetune_data(tokenizer, samples, task, seq_len, seed):
    """Convert SCROLLS samples to token sequences with loss masks.

    Returns:
        input_ids: (N, seq_len) tensor of token IDs
        loss_masks: (N, seq_len) tensor of 0/1 loss masks
    """
    rng = random.Random(seed)
    rng.shuffle(samples)

    all_ids = []
    all_masks = []
    skipped = 0

    for sample in samples:
        prompt_str, answer_str = format_sample(sample, task)
        if not prompt_str or not answer_str.strip():
            skipped += 1
            continue

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)

        if len(answer_ids) == 0:
            skipped += 1
            continue

        total_len = len(prompt_ids) + len(answer_ids)

        if total_len > seq_len:
            # Truncate context (prompt), keep full answer
            max_prompt = seq_len - len(answer_ids)
            if max_prompt < 64:
                # Answer too long, truncate answer too
                max_prompt = seq_len // 2
                answer_ids = answer_ids[:seq_len - max_prompt]
            # Middle truncation for prompt
            if len(prompt_ids) > max_prompt:
                half = max_prompt // 2
                prompt_ids = prompt_ids[:half] + prompt_ids[-(max_prompt - half):]

        # Build sequence: prompt + answer, pad to seq_len
        seq = prompt_ids + answer_ids
        prompt_len = len(prompt_ids)
        answer_len = len(answer_ids)

        # Pad to seq_len (pad with EOS)
        pad_id = tokenizer.eos_token_id or 0
        pad_len = seq_len - len(seq)
        if pad_len > 0:
            seq = seq + [pad_id] * pad_len
        else:
            seq = seq[:seq_len]

        # Loss mask: 1 for answer tokens, 0 for prompt + padding
        mask = [0.0] * prompt_len + [1.0] * answer_len + [0.0] * max(0, pad_len)
        mask = mask[:seq_len]

        all_ids.append(seq)
        all_masks.append(mask)

    if skipped > 0:
        print(f"  Skipped {skipped} samples (empty prompt/answer)")

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    loss_masks = torch.tensor(all_masks, dtype=torch.float32)
    print(f"  Prepared {len(all_ids)} samples, seq_len={seq_len}")
    print(f"  Avg answer tokens: {loss_masks.sum(1).mean():.0f}")
    return input_ids, loss_masks


# ── Training Loop ─────────────────────────────────────────────────────

def train_finetune(model, input_ids, loss_masks, args, on_step_end=None):
    """Finetuning training loop with loss masking."""
    model.train()

    lr = args.lr
    min_lr = lr * 0.1
    micro_bs = args.micro_batch_size
    grad_accum = args.grad_accum
    effective_bs = micro_bs * grad_accum
    total_steps = args.steps
    warmup = args.warmup

    n_samples = input_ids.shape[0]
    steps_per_epoch = max(1, n_samples // effective_bs)
    n_epochs = math.ceil(total_steps / steps_per_epoch)

    adamw_kwargs = dict(lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if DEVICE == "cuda":
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    print(f"\n  Finetune config:")
    print(f"    lr={lr:.1e}, min_lr={min_lr:.1e}, warmup={warmup}")
    print(f"    micro_bs={micro_bs}, grad_accum={grad_accum}, effective_bs={effective_bs}")
    print(f"    total_steps={total_steps}, samples={n_samples}")
    print(f"    steps_per_epoch={steps_per_epoch}, epochs~={n_epochs}")

    set_seed(args.seed)
    t0 = time.time()
    global_step = 0
    running_loss = 0.0
    n_loss_samples = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_ids = input_ids[perm]
        epoch_masks = loss_masks[perm]

        # Move epoch data to GPU in chunks
        chunk_size = min(n_samples, 2000)
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            gpu_ids = epoch_ids[chunk_start:chunk_end].to(DEVICE)
            gpu_masks = epoch_masks[chunk_start:chunk_end].to(DEVICE)

            chunk_len = chunk_end - chunk_start
            chunk_steps = chunk_len // effective_bs

            for cs in range(chunk_steps):
                if global_step >= total_steps:
                    break

                # LR schedule: linear warmup → cosine decay
                if global_step < warmup:
                    cur_lr = lr * (global_step + 1) / warmup
                else:
                    progress = (global_step - warmup) / max(total_steps - warmup, 1)
                    cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for g in opt.param_groups:
                    g["lr"] = cur_lr

                opt.zero_grad(set_to_none=True)
                accum_loss = 0.0

                for a in range(grad_accum):
                    idx_start = cs * effective_bs + a * micro_bs
                    idx_end = idx_start + micro_bs
                    if idx_end > chunk_len:
                        break

                    batch_ids = gpu_ids[idx_start:idx_end]
                    batch_mask = gpu_masks[idx_start:idx_end]

                    # inputs = tokens[:-1], targets = tokens[1:]
                    inputs = batch_ids[:, :-1].contiguous()
                    targets = batch_ids[:, 1:].contiguous()
                    # Shift mask: mask[i] corresponds to predicting token[i+1]
                    # So loss_mask for position t means we compute loss for predicting target[t]
                    target_mask = batch_mask[:, 1:].contiguous()

                    ctx = (torch.amp.autocast("cuda", dtype=DTYPE)
                           if USE_AUTOCAST and DEVICE == "cuda" else nullcontext())
                    with ctx:
                        logits = model(inputs)
                        # Flatten for cross_entropy
                        flat_logits = logits.reshape(-1, logits.size(-1))
                        flat_targets = targets.reshape(-1)
                        flat_mask = target_mask.reshape(-1)

                        # Per-token CE loss
                        per_token_loss = F.cross_entropy(
                            flat_logits, flat_targets, reduction='none'
                        )
                        # Masked mean
                        mask_sum = flat_mask.sum()
                        if mask_sum > 0:
                            loss = (per_token_loss * flat_mask).sum() / mask_sum
                        else:
                            loss = per_token_loss.mean()
                        loss = loss / grad_accum

                    loss.backward()
                    accum_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                global_step += 1
                running_loss += accum_loss
                n_loss_samples += 1

                if global_step % 100 == 0 or global_step == total_steps:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / max(n_loss_samples, 1)
                    eta = elapsed / global_step * (total_steps - global_step)
                    gpu_mem = (
                        torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
                    )
                    print(
                        f"    step {global_step}/{total_steps}  loss={accum_loss:.4f}  "
                        f"avg={avg_loss:.4f}  lr={cur_lr:.2e}  "
                        f"GPU={gpu_mem:.1f}GB  ETA={eta/60:.0f}min"
                    )
                    running_loss = 0.0
                    n_loss_samples = 0

                if on_step_end is not None:
                    on_step_end(global_step, total_steps)

                if global_step >= total_steps:
                    break

            del gpu_ids, gpu_masks
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    elapsed = time.time() - t0
    print(f"  Finetuning done in {elapsed/60:.1f} min ({global_step} steps)")
    return model


# ── Generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=512, temperature=0.0):
    """Simple greedy/sampling generation for evaluation."""
    model.eval()
    device = next(model.parameters()).device

    ids = input_ids.clone().to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        # Only use last seq_len tokens if sequence gets too long
        ctx = ids if ids.size(1) <= 8192 else ids[:, -8192:]

        amp_ctx = (torch.amp.autocast("cuda", dtype=DTYPE)
                   if USE_AUTOCAST and DEVICE == "cuda" else nullcontext())
        with amp_ctx:
            logits = model(ctx)

        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)

        ids = torch.cat([ids, next_id], dim=1)

        # Stop on EOS
        if next_id.item() == 0:  # gpt-neox EOS = 0
            break

    return ids[0]


# ── ROUGE Evaluation ──────────────────────────────────────────────────

def evaluate_rouge(model, tokenizer, val_samples, task, args):
    """Generate summaries and compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("  [WARN] rouge-score not installed, skipping ROUGE eval")
        print("  Install with: pip install rouge-score")
        return {}

    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    n_eval = min(len(val_samples), args.eval_samples)
    samples = val_samples[:n_eval]

    print(f"\n  Evaluating ROUGE on {n_eval} samples...")
    t0 = time.time()

    for i, sample in enumerate(samples):
        prompt_str, answer_str = format_sample(sample, task)
        if not prompt_str or not answer_str.strip():
            continue

        gold = answer_str.strip()

        # Tokenize prompt (truncate to leave room for generation)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        max_prompt = args.seq_len - args.max_gen_tokens
        if len(prompt_ids) > max_prompt:
            half = max_prompt // 2
            prompt_ids = prompt_ids[:half] + prompt_ids[-(max_prompt - half):]

        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)

        # Extend rope if needed
        needed_len = len(prompt_ids) + args.max_gen_tokens
        model.extend_rope(needed_len)

        # Generate
        output_ids = generate(model, prompt_tensor, max_new_tokens=args.max_gen_tokens)
        generated_ids = output_ids[len(prompt_ids):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Score
        scores = scorer.score(gold, generated_text)
        for key in results:
            results[key].append(scores[key].fmeasure)

        if (i + 1) % 50 == 0:
            r1_avg = np.mean(results["rouge1"]) * 100
            print(f"    [{i+1}/{n_eval}] ROUGE-1={r1_avg:.1f}")

    elapsed = time.time() - t0
    summary = {}
    for key in results:
        if results[key]:
            summary[key] = {
                "mean": float(np.mean(results[key]) * 100),
                "std": float(np.std(results[key]) * 100),
                "n": len(results[key]),
            }
    summary["eval_time_sec"] = round(elapsed, 1)

    print(f"\n  ROUGE Results ({n_eval} samples, {elapsed:.0f}s):")
    for key in ["rouge1", "rouge2", "rougeL"]:
        if key in summary:
            m = summary[key]["mean"]
            s = summary[key]["std"]
            print(f"    {key}: {m:.2f} (+/- {s:.2f})")

    return summary


def evaluate_accuracy(model, tokenizer, val_samples, task, args):
    """Evaluate multiple-choice accuracy for QuALITY."""
    model.eval()
    correct = 0
    total = 0
    n_eval = min(len(val_samples), args.eval_samples)

    print(f"\n  Evaluating accuracy on {n_eval} samples...")

    for i, sample in enumerate(val_samples[:n_eval]):
        prompt_str, answer_str = format_sample(sample, task)
        if not prompt_str or not answer_str.strip():
            continue

        gold = answer_str.strip()

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        max_prompt = args.seq_len - 16
        if len(prompt_ids) > max_prompt:
            half = max_prompt // 2
            prompt_ids = prompt_ids[:half] + prompt_ids[-(max_prompt - half):]

        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        model.extend_rope(len(prompt_ids) + 16)

        output_ids = generate(model, prompt_tensor, max_new_tokens=16)
        gen_ids = output_ids[len(prompt_ids):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Check if first char of generation matches gold answer
        if gen_text and gold and gen_text[0].upper() == gold[0].upper():
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n_eval}] acc={correct/total*100:.1f}%")

    acc = correct / max(total, 1) * 100
    print(f"\n  Accuracy: {acc:.2f}% ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("#" * 72)
    print(f"  Phase 21B: SCROLLS Finetuning")
    print(f"  Task: {args.task}  |  RoPE: {args.rope}  |  YaRN: {args.yarn}")
    print(f"  Init: {args.init_ckpt}")
    print(f"  Seq_len: {args.seq_len}  |  Steps: {args.steps}  |  LR: {args.lr}")
    print("#" * 72)

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    result_file = output_dir / "result.json"
    if result_file.exists():
        print(f"\n[SKIP] result.json already exists: {result_file}")
        with open(result_file) as f:
            result = json.load(f)
        print(json.dumps(result.get("eval", {}), indent=2))
        return

    # ── Build inv_freq ──
    tier_cfg = TIER_CONFIGS[args.tier]
    dim = tier_cfg["head_dim"]
    if args.rope == "evq":
        inv_freq = evq_cosh_inv_freq(head_dim=dim, tau=args.tau, base=args.base)
        print(f"  RoPE: EVQ (tau={args.tau}, base={args.base})")
    else:
        inv_freq = geometric_inv_freq(dim, args.base)
        print(f"  RoPE: Geometric (base={args.base})")

    if args.yarn:
        inv_freq = apply_yarn_scaling(inv_freq, args.yarn_scale)
        print(f"  YaRN scaling applied: scale={args.yarn_scale}")

    print(f"  inv_freq: max={inv_freq.max():.8f} min={inv_freq.min():.10f}")

    # ── Load model ──
    cfg = tier_cfg.copy()
    cfg["max_position_embeddings"] = args.seq_len
    cfg["seq_len"] = args.seq_len

    print(f"\n  Loading model from {args.init_ckpt}...")
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = _load_state(args.init_ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")

    # Force-set inv_freq
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    model.extend_rope(args.seq_len)
    del state

    # Add dropout
    model = add_dropout_to_model(model, args.dropout)

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load SCROLLS data ──
    print(f"\n  Loading SCROLLS/{args.task} data...")
    train_samples = load_scrolls_data(args.task, "train", args.data_dir)
    val_samples = load_scrolls_data(args.task, "validation", args.data_dir)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")

    if args.max_train_samples > 0:
        train_samples = train_samples[:args.max_train_samples]
        print(f"  Limited to {len(train_samples)} train samples")

    # ── Prepare training data ──
    print(f"\n  Tokenizing training data (seq_len={args.seq_len})...")
    input_ids, loss_masks = prepare_finetune_data(
        tokenizer, train_samples, args.task, args.seq_len, args.seed
    )
    del train_samples

    # ── Train ──
    ckpt_steps = set()
    if args.eval_every > 0:
        for s in range(args.eval_every, args.steps + 1, args.eval_every):
            ckpt_steps.add(s)
    ckpt_steps.add(args.steps)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    saved_ckpts = set()

    def on_step_end(step, total):
        if step in ckpt_steps and step not in saved_ckpts:
            saved_ckpts.add(step)
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            print(f"\n  [CKPT] Saving step {step} → {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)

    model = train_finetune(model, input_ids, loss_masks, args, on_step_end=on_step_end)
    del input_ids, loss_masks

    # Save final model
    final_path = output_dir / "model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n  Final model saved: {final_path}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Evaluate ──
    task_info = TASK_PROMPTS[args.task]
    if task_info["eval_metric"] == "rouge":
        eval_result = evaluate_rouge(model, tokenizer, val_samples, args.task, args)
    else:
        eval_result = evaluate_accuracy(model, tokenizer, val_samples, args.task, args)

    # ── Save results ──
    result = {
        "phase": "21b",
        "task": args.task,
        "rope": args.rope,
        "tau": args.tau if args.rope == "evq" else None,
        "base": args.base,
        "yarn": args.yarn,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "lr": args.lr,
        "dropout": args.dropout,
        "seed": args.seed,
        "init_ckpt": args.init_ckpt,
        "eval": eval_result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(result_file, result)
    print(f"\n  Results saved: {result_file}")
    print(json.dumps(eval_result, indent=2))


if __name__ == "__main__":
    main()
