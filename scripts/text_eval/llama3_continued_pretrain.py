#!/usr/bin/env python3
"""
LLaMA-3-8B-Instruct Continued Pretraining: GEO vs EVQ-Cosh Head-to-Head

Continued pretraining of Meta-Llama-3-8B-Instruct with LoRA on Q/K projections,
comparing geometric (control) vs EVQ-Cosh (treatment) frequency allocation.

Model:
  - Meta-Llama-3-8B-Instruct, head_dim=128, rope_theta=500000, native ctx=8192
  - 64 inv_freq values (head_dim // 2)

EVQ-Cosh tau: head_dim / sqrt(L_pretrain) = 128 / sqrt(8192) = 1.414

Experiment design:
  Config 1 (GEO): Original geometric inv_freq — control
  Config 2 (EVQ): EVQ-Cosh tau=1.414 inv_freq — treatment
  Both: LoRA rank=64 on Q/K only, 2000 steps, seq_len=8192, cosine LR

Evaluation:
  - PPL at 8192 (in-dist), 16384 (2x), 32768 (4x) with YaRN for extrapolation
  - Passkey retrieval at 8192, 16384, 32768 (20 trials each)

Usage:
  # Smoke test on T4 16GB
  python llama3_continued_pretrain.py \\
      --model_dir /path/to/Meta-Llama-3-8B-Instruct \\
      --data_dir /home/ubuntu/data/packed \\
      --output_dir /home/ubuntu/results/llama3_h2h \\
      --steps 5 --pilot

  # Full run on RTX PRO 6000 96GB
  python llama3_continued_pretrain.py \\
      --model_dir /path/to/Meta-Llama-3-8B-Instruct \\
      --data_dir /home/ubuntu/data/packed \\
      --output_dir /home/ubuntu/results/llama3_h2h

  # EVQ only (skip GEO control)
  python llama3_continued_pretrain.py \\
      --model_dir /path/to/Meta-Llama-3-8B-Instruct \\
      --data_dir /home/ubuntu/data/packed \\
      --output_dir /home/ubuntu/results/llama3_h2h \\
      --configs evq
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_DIM = 128
ROPE_THETA = 500_000.0
NATIVE_CTX = 8192
N_FREQ = HEAD_DIM // 2  # 64

TRAIN_SEQ_LEN = 8192
DEFAULT_STEPS = 2000
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
WARMUP_FRACTION = 0.05
LORA_RANK = 64
LORA_ALPHA = 128  # alpha = 2 * rank is standard
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj"]

EVAL_LENGTHS = [8192, 16384, 32768]
PASSKEY_TRIALS = 20
PASSKEY_LENGTHS = [8192, 16384, 32768]


# ---------------------------------------------------------------------------
# EVQ-Cosh frequency allocation
# ---------------------------------------------------------------------------

def evq_cosh_inv_freq(dim: int, tau: float, base: float) -> torch.Tensor:
    """Compute EVQ-Cosh inverse frequency allocation.

    Args:
        dim: head_dim (must be even).
        tau: Concentration parameter. tau = head_dim / sqrt(L_pretrain).
        base: RoPE theta base.

    Returns:
        Tensor of shape (dim // 2,) with dtype float32.
    """
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def geometric_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Standard geometric RoPE inverse frequencies."""
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    return (1.0 / (float(base) ** (2.0 * idx / float(dim)))).float()


# ---------------------------------------------------------------------------
# YaRN-style NTK-aware interpolation for extrapolation eval
# ---------------------------------------------------------------------------

def yarn_inv_freq(
    original_inv_freq: torch.Tensor,
    scale: float,
    head_dim: int,
) -> torch.Tensor:
    """Apply YaRN-style NTK-aware interpolation to inv_freq.

    Low-frequency components (large period) are interpolated more,
    high-frequency components (small period) are kept near original.

    Args:
        original_inv_freq: Shape (K,), the base inv_freq.
        scale: Extrapolation ratio (target_len / native_ctx).
        head_dim: Model head dimension.

    Returns:
        Scaled inv_freq tensor of shape (K,).
    """
    if scale <= 1.0:
        return original_inv_freq.clone()

    K = len(original_inv_freq)
    idx = torch.arange(K, dtype=torch.float64)

    # YaRN ramp: smoothstep from channel start to end
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)

    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    # Smoothstep
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)

    # Temperature correction
    temperature = 1.0 + 0.07 * math.log2(scale)

    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (original_inv_freq.double() / yarn_scale).float()


# ---------------------------------------------------------------------------
# Utility: GPU memory reporting
# ---------------------------------------------------------------------------

def gpu_mem_str() -> str:
    """Return a human-readable GPU memory usage string."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    return f"VRAM: {allocated:.1f}G alloc / {reserved:.1f}G reserved / {total:.1f}G total"


def cleanup_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# inv_freq patching for LLaMA-3
# ---------------------------------------------------------------------------

def patch_inv_freq(model: nn.Module, new_inv_freq: torch.Tensor) -> int:
    """Patch all rotary embedding inv_freq buffers in-place.

    Works with transformers LlamaForCausalLM architecture.

    Args:
        model: The HuggingFace LLaMA model.
        new_inv_freq: 1-D tensor of new inverse frequencies.

    Returns:
        Number of layers patched.
    """
    patched = 0
    for layer in model.model.layers:
        rope = layer.self_attn.rotary_emb
        if hasattr(rope, "inv_freq") and rope.inv_freq is not None:
            device = rope.inv_freq.device
            dtype = rope.inv_freq.dtype
            rope.inv_freq = nn.Parameter(
                new_inv_freq.to(device=device, dtype=dtype),
                requires_grad=False,
            )
            # Clear any cached cos/sin
            for attr in ("_cos_cached", "_sin_cached", "cos_cached",
                         "sin_cached", "_cos_cache", "_sin_cache"):
                if hasattr(rope, attr):
                    try:
                        setattr(rope, attr, None)
                    except Exception:
                        pass
            if hasattr(rope, "max_seq_len_cached"):
                try:
                    rope.max_seq_len_cached = 0
                except Exception:
                    pass
            patched += 1
    return patched


def patch_inv_freq_for_eval(model: nn.Module, new_inv_freq: torch.Tensor) -> int:
    """Patch inv_freq for evaluation (e.g., with YaRN-scaled frequencies).

    Same as patch_inv_freq but searches more broadly for rotary modules.
    """
    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None:
            if torch.is_tensor(module.inv_freq) and module.inv_freq.numel() == len(new_inv_freq):
                device = module.inv_freq.device
                dtype = module.inv_freq.dtype
                module.inv_freq = nn.Parameter(
                    new_inv_freq.to(device=device, dtype=dtype),
                    requires_grad=False,
                )
                for attr in ("_cos_cached", "_sin_cached", "cos_cached",
                              "sin_cached", "_cos_cache", "_sin_cache"):
                    if hasattr(module, attr):
                        try:
                            setattr(module, attr, None)
                        except Exception:
                            pass
                if hasattr(module, "max_seq_len_cached"):
                    try:
                        module.max_seq_len_cached = 0
                    except Exception:
                        pass
                patched += 1
    return patched


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_packed_data(path: str, label: str = "data") -> torch.Tensor:
    """Load a pre-packed .pt tensor file.

    Expected shape: (N, seq_len) where each row is a packed token sequence.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    data = torch.load(str(p), map_location="cpu", weights_only=True)
    print(f"  Loaded {label}: {p.name} — shape {tuple(data.shape)}, "
          f"{data.numel() / 1e6:.1f}M tokens")
    return data


# ---------------------------------------------------------------------------
# Passkey retrieval evaluation
# ---------------------------------------------------------------------------

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a distant land, there existed a kingdom of great wisdom and prosperity. "
    "The scholars spent their days studying ancient texts and debating the nature of reality. "
    "Markets bustled with merchants selling exotic spices, fine silks, and precious gems. "
    "Children played in the cobblestone streets while elders told tales of adventures past. "
    "The rivers flowed gently through the valleys, carrying with them stories of a thousand years. "
    "Each season brought new challenges and opportunities for the people of this realm. "
    "Artists painted magnificent murals on the walls of grand temples and palaces. "
    "Musicians filled the air with melodies that echoed through the mountain passes. "
    "Scientists observed the stars and charted the movements of celestial bodies. "
    "Philosophers pondered the meaning of existence and the boundaries of knowledge. "
    "Engineers designed intricate machines and aqueducts that served the growing population. "
    "Farmers cultivated the fertile lands, producing abundant harvests year after year. "
    "The cycle of life continued, each generation building upon the achievements of the last. "
)


def build_passkey_prompt(
    tokenizer,
    target_length: int,
    passkey: str,
    depth_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, str]:
    """Build a passkey retrieval prompt for LLaMA-3-Instruct.

    Embeds a 5-digit passkey at a specified depth within filler text,
    then asks the model to retrieve it.

    Args:
        tokenizer: HuggingFace tokenizer.
        target_length: Desired total sequence length in tokens.
        passkey: The 5-digit passkey string (e.g., "73829").
        depth_ratio: Where to place the passkey (0.0=start, 1.0=end).
        seed: Random seed for filler variation.

    Returns:
        (input_ids, passkey) — input_ids is a 1-D tensor.
    """
    rng = random.Random(seed)

    # Tokenize filler text
    filler_ids = tokenizer.encode(FILLER_TEXT, add_special_tokens=False)

    # Build enough filler by repeating and shuffling sentences
    filler_sentences = FILLER_TEXT.split(". ")
    extended_filler = []
    while len(extended_filler) < target_length * 2:
        rng.shuffle(filler_sentences)
        for s in filler_sentences:
            extended_filler.extend(tokenizer.encode(s + ". ", add_special_tokens=False))

    # The passkey statement
    passkey_statement = f"The secret passkey is {passkey}. Remember this number."
    passkey_ids = tokenizer.encode(passkey_statement, add_special_tokens=False)

    # The query
    query_text = "\n\nWhat is the secret passkey mentioned in the text above? The passkey is"
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)

    # Calculate filler budget
    filler_budget = target_length - len(passkey_ids) - len(query_ids) - 4  # BOS + margin
    if filler_budget < 10:
        raise ValueError(f"target_length={target_length} too short for passkey prompt")

    # Split filler
    insert_pos = max(1, int(filler_budget * depth_ratio))
    before_ids = extended_filler[:insert_pos]
    after_ids = extended_filler[insert_pos:insert_pos + (filler_budget - insert_pos)]

    # Pad after_ids if needed
    while len(before_ids) + len(after_ids) < filler_budget:
        after_ids.extend(extended_filler[:filler_budget - len(before_ids) - len(after_ids)])

    # Assemble
    full_ids = before_ids + passkey_ids + after_ids[:filler_budget - len(before_ids)] + query_ids

    # Trim to target length (minus BOS which tokenizer adds)
    full_ids = full_ids[:target_length - 1]

    # Add BOS
    if tokenizer.bos_token_id is not None:
        full_ids = [tokenizer.bos_token_id] + full_ids

    # Final trim
    full_ids = full_ids[:target_length]

    return torch.tensor(full_ids, dtype=torch.long), passkey


@torch.no_grad()
def eval_passkey(
    model: nn.Module,
    tokenizer,
    context_lengths: List[int],
    n_trials: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run passkey retrieval evaluation.

    For each context length, generates n_trials prompts with random 5-digit
    passkeys at random depths, then checks if the model can retrieve them.

    Args:
        model: The (potentially LoRA-adapted) model.
        tokenizer: HuggingFace tokenizer.
        context_lengths: List of context lengths to test.
        n_trials: Number of trials per length.
        seed: Base random seed.

    Returns:
        Dict mapping "L={length}" to accuracy and details.
    """
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    results = {}

    for L in context_lengths:
        correct = 0
        total = 0
        trial_details = []

        print(f"    Passkey L={L}: ", end="", flush=True)

        for trial in range(n_trials):
            passkey = f"{rng.randint(10000, 99999)}"
            depth = rng.uniform(0.1, 0.9)
            trial_seed = seed + L * 1000 + trial

            try:
                input_ids, _ = build_passkey_prompt(
                    tokenizer, L, passkey, depth, trial_seed,
                )
                input_ids = input_ids.unsqueeze(0).to(device)

                # Generate a few tokens
                gen_ids = []
                cur_ids = input_ids
                for _ in range(8):  # Generate up to 8 tokens to capture the 5-digit number
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(cur_ids)
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                    gen_ids.append(next_token.item())
                    cur_ids = torch.cat([cur_ids, next_token], dim=1)

                # Decode generated tokens
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                # Check if passkey appears in generated text
                found = passkey in generated_text
                if found:
                    correct += 1
                total += 1

                trial_details.append({
                    "passkey": passkey,
                    "depth": round(depth, 3),
                    "generated": generated_text,
                    "correct": found,
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at trial {trial}, ", end="", flush=True)
                    cleanup_gpu()
                    # Skip remaining trials at this length
                    break
                raise
            finally:
                if "input_ids" in dir():
                    del input_ids
                if "cur_ids" in dir():
                    del cur_ids
                cleanup_gpu()

        acc = correct / max(total, 1)
        print(f"{correct}/{total} = {acc:.0%}")
        results[f"L={L}"] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total,
            "trials": trial_details,
        }

    return results


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    test_data: torch.Tensor,
    base_inv_freq: torch.Tensor,
    label: str = "",
    max_chunks: int = 50,
) -> Dict[str, float]:
    """Evaluate perplexity on pre-packed test data.

    For sequences longer than NATIVE_CTX (8192), applies YaRN scaling.

    Args:
        model: The model (with or without LoRA).
        test_data: Tensor of shape (N, seq_len).
        base_inv_freq: The base inv_freq used during training (geo or evq).
        label: Description for logging.
        max_chunks: Maximum number of chunks to evaluate.

    Returns:
        Dict with "ppl", "loss", "n_chunks", "seq_len".
    """
    model.eval()
    device = next(model.parameters()).device
    seq_len = test_data.shape[1]

    # Apply YaRN if extrapolating beyond native context
    if seq_len > NATIVE_CTX:
        scale = seq_len / NATIVE_CTX
        scaled_freq = yarn_inv_freq(base_inv_freq, scale, HEAD_DIM)
        n_patched = patch_inv_freq_for_eval(model, scaled_freq)
        print(f"    YaRN applied: scale={scale:.1f}x, patched {n_patched} layers")

    n_chunks = min(len(test_data), max_chunks)
    total_loss = 0.0
    total_tokens = 0

    print(f"    PPL eval{' (' + label + ')' if label else ''}: "
          f"seq_len={seq_len}, chunks={n_chunks} ... ", end="", flush=True)

    for i in range(n_chunks):
        chunk = test_data[i].unsqueeze(0).to(device)
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(chunk[:, :-1], labels=chunk[:, 1:])
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    # Manual cross-entropy if model doesn't return loss
                    logits = outputs if not hasattr(outputs, "logits") else outputs.logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        chunk[:, 1:].reshape(-1),
                    )
            total_loss += loss.item() * (seq_len - 1)
            total_tokens += (seq_len - 1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at chunk {i}, ", end="", flush=True)
                cleanup_gpu()
                break
            raise
        finally:
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Restore original inv_freq if we applied YaRN
    if seq_len > NATIVE_CTX:
        patch_inv_freq_for_eval(model, base_inv_freq)

    if total_tokens == 0:
        print("FAILED (no chunks evaluated)")
        return {"ppl": float("inf"), "loss": float("inf"), "n_chunks": 0, "seq_len": seq_len}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))  # Cap to avoid overflow
    print(f"loss={avg_loss:.4f}, PPL={ppl:.2f}")

    return {
        "ppl": round(ppl, 3),
        "loss": round(avg_loss, 6),
        "n_chunks": n_chunks,
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    model: nn.Module,
    train_data: torch.Tensor,
    steps: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    warmup_fraction: float,
    output_dir: Path,
    config_name: str,
) -> List[float]:
    """Run the training loop with cosine LR schedule.

    Args:
        model: The LoRA-adapted model.
        train_data: Tensor of shape (N, seq_len).
        steps: Total training steps.
        lr: Peak learning rate.
        batch_size: Micro-batch size.
        grad_accum: Gradient accumulation steps.
        warmup_fraction: Fraction of steps for linear warmup.
        output_dir: Directory to save checkpoints.
        config_name: Name for logging (e.g., "geo", "evq").

    Returns:
        List of per-step losses.
    """
    device = next(model.parameters()).device
    effective_bs = batch_size * grad_accum
    warmup_steps = max(1, int(steps * warmup_fraction))
    min_lr = lr * 0.1
    seq_len = train_data.shape[1]

    print(f"  Training config: steps={steps}, lr={lr:.1e}, "
          f"bs={effective_bs} (micro={batch_size}x{grad_accum}), "
          f"warmup={warmup_steps}, seq_len={seq_len}")

    # Only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M "
          f"({100.0 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    model.train()
    n_samples = len(train_data)
    perm = torch.randperm(n_samples)
    ptr = 0
    epoch = 1
    all_losses = []
    t0 = time.time()
    log_interval = max(1, steps // 20)

    for step in range(1, steps + 1):
        # LR schedule: linear warmup + cosine decay
        if step <= warmup_steps:
            cur_lr = lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(grad_accum):
            # Epoch wrapping
            if ptr + batch_size > n_samples:
                perm = torch.randperm(n_samples)
                ptr = 0
                epoch += 1

            indices = perm[ptr:ptr + batch_size]
            ptr += batch_size
            batch = train_data[indices].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch[:, :-1], labels=batch[:, 1:])
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    logits = outputs if not hasattr(outputs, "logits") else outputs.logits
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        batch[:, 1:].reshape(-1),
                    )
                loss_scaled = loss / grad_accum

            loss_scaled.backward()
            step_loss += loss.item() / grad_accum

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        all_losses.append(step_loss)

        # Logging
        if step % log_interval == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            tokens_processed = step * effective_bs * seq_len
            tps = tokens_processed / max(elapsed, 1e-6)
            print(f"    [{config_name}] step {step:>5d}/{steps} "
                  f"({step / steps * 100:5.1f}%) | "
                  f"loss={step_loss:.4f} | lr={cur_lr:.2e} | "
                  f"epoch={epoch} | {tps / 1e6:.2f}M tok/s | "
                  f"{elapsed:.0f}s | {gpu_mem_str()}")

    elapsed = time.time() - t0
    print(f"  Training done: {elapsed:.0f}s ({elapsed / 60:.1f}min), "
          f"{epoch} epoch(s)")

    return all_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLaMA-3-8B-Instruct continued pretraining: GEO vs EVQ-Cosh"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with train_packed.pt, test_packed.pt, etc.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results and checkpoints")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Peak learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Micro-batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM,
                        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM})")
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--tau", type=float, default=None,
                        help="EVQ tau (default: head_dim / sqrt(native_ctx) = 1.414)")
    parser.add_argument("--configs", type=str, default="geo,evq",
                        help="Comma-separated configs to run (default: geo,evq)")
    parser.add_argument("--passkey_trials", type=int, default=PASSKEY_TRIALS,
                        help=f"Passkey trials per length (default: {PASSKEY_TRIALS})")
    parser.add_argument("--max_ppl_chunks", type=int, default=50,
                        help="Max chunks for PPL evaluation (default: 50)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: skip long-context eval, reduce passkey trials")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_passkey", action="store_true",
                        help="Skip passkey evaluation")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: True)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    args = parser.parse_args()

    use_grad_ckpt = args.gradient_checkpointing and not args.no_gradient_checkpointing

    # Compute tau
    tau = args.tau if args.tau is not None else HEAD_DIM / math.sqrt(NATIVE_CTX)

    # Configs to run
    config_names = [c.strip().lower() for c in args.configs.split(",")]
    for c in config_names:
        if c not in ("geo", "evq"):
            print(f"ERROR: Unknown config '{c}'. Must be 'geo' or 'evq'.")
            sys.exit(1)

    # Pilot mode adjustments
    if args.pilot:
        eval_lengths = [8192]
        passkey_lengths = [8192]
        passkey_trials = min(5, args.passkey_trials)
        max_ppl_chunks = min(10, args.max_ppl_chunks)
        print("  PILOT MODE: reduced eval scope")
    else:
        eval_lengths = EVAL_LENGTHS
        passkey_lengths = PASSKEY_LENGTHS
        passkey_trials = args.passkey_trials
        max_ppl_chunks = args.max_ppl_chunks

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # ── Header ──────────────────────────────────────────────────────
    sep = "=" * 72
    print(sep)
    print("  LLaMA-3-8B-Instruct Continued Pretraining: GEO vs EVQ-Cosh")
    print(sep)
    print(f"  Model:          {args.model_dir}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Configs:        {config_names}")
    print(f"  Steps:          {args.steps}")
    print(f"  LR:             {args.lr:.1e}")
    print(f"  Effective BS:   {args.batch_size * args.grad_accum} "
          f"(micro={args.batch_size} x ga={args.grad_accum})")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  LoRA targets:   {TARGET_MODULES}")
    print(f"  EVQ tau:        {tau:.6f}")
    print(f"  Grad ckpt:      {use_grad_ckpt}")
    print(f"  Pilot mode:     {args.pilot}")
    print(f"  Seed:           {args.seed}")
    print(f"  Eval lengths:   {eval_lengths}")
    print(f"  Passkey trials: {passkey_trials}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  {gpu_mem_str()}")
    print(sep)
    print()

    # ── Set seed ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Pre-compute inv_freqs ───────────────────────────────────────
    geo_freq = geometric_inv_freq(HEAD_DIM, ROPE_THETA)
    evq_freq = evq_cosh_inv_freq(HEAD_DIM, tau, ROPE_THETA)

    print("[0] Frequency comparison (first 8 / last 8 of 64):")
    print(f"    GEO: {geo_freq[:8].tolist()}")
    print(f"         ... {geo_freq[-8:].tolist()}")
    print(f"    EVQ: {evq_freq[:8].tolist()}")
    print(f"         ... {evq_freq[-8:].tolist()}")
    print(f"    Max abs diff: {(geo_freq - evq_freq).abs().max().item():.8f}")
    print()

    # ── Load training data ──────────────────────────────────────────
    print("[1] Loading data...")
    train_path = data_dir / "train_packed.pt"
    train_data = load_packed_data(str(train_path), "train")

    # Load test data for each eval length
    test_datasets = {}
    for L in eval_lengths:
        if L == TRAIN_SEQ_LEN:
            test_path = data_dir / "test_packed.pt"
        else:
            test_path = data_dir / f"test_{L}.pt"

        if test_path.exists():
            test_datasets[L] = load_packed_data(str(test_path), f"test L={L}")
        else:
            print(f"  WARNING: {test_path} not found, skipping PPL eval at L={L}")
    print()

    # ── Load tokenizer ──────────────────────────────────────────────
    print("[2] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")
    print()

    # ── Run configs ─────────────────────────────────────────────────
    all_results = {}

    inv_freq_map = {
        "geo": ("Geometric (control)", geo_freq),
        "evq": (f"EVQ-Cosh tau={tau:.4f} (treatment)", evq_freq),
    }

    for config_name in config_names:
        config_label, inv_freq = inv_freq_map[config_name]
        run_dir = output_dir / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"

        # Skip if already completed
        if result_path.exists():
            print(f"\n  SKIP {config_name} (result.json exists)")
            with open(result_path) as f:
                all_results[config_name] = json.load(f)
            continue

        print(f"\n{sep}")
        print(f"  CONFIG: {config_name} — {config_label}")
        print(f"{sep}")

        # ── Load fresh model ────────────────────────────────────────
        print(f"\n  [A] Loading model...")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        print(f"  Model loaded. {gpu_mem_str()}")

        # ── Patch inv_freq ──────────────────────────────────────────
        print(f"\n  [B] Patching inv_freq ({config_name})...")
        n_patched = patch_inv_freq(model, inv_freq)
        print(f"  Patched {n_patched} rotary layers")

        # Verify patch
        sample_freq = model.model.layers[0].self_attn.rotary_emb.inv_freq
        print(f"  Verify layer 0 inv_freq[0]: {sample_freq[0].item():.8f} "
              f"(expected: {inv_freq[0].item():.8f})")

        # ── Setup LoRA ──────────────────────────────────────────────
        print(f"\n  [C] Setting up LoRA (rank={args.lora_rank}, targets={TARGET_MODULES})...")
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"  {gpu_mem_str()}")

        # ── Gradient checkpointing ──────────────────────────────────
        if use_grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("  Gradient checkpointing: enabled")

        # ── Train ───────────────────────────────────────────────────
        print(f"\n  [D] Training ({config_name})...")
        t_start = time.time()

        losses = train_loop(
            model=model,
            train_data=train_data,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            warmup_fraction=WARMUP_FRACTION,
            output_dir=run_dir,
            config_name=config_name,
        )

        train_time = time.time() - t_start

        # Save LoRA adapter
        adapter_dir = run_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"  LoRA adapter saved to: {adapter_dir}")

        # Save loss curve
        loss_path = run_dir / "losses.json"
        with open(loss_path, "w") as f:
            json.dump(losses, f)
        print(f"  Loss curve saved to: {loss_path}")

        # ── Evaluate PPL ────────────────────────────────────────────
        print(f"\n  [E] Evaluating PPL ({config_name})...")
        model.eval()

        # Disable gradient checkpointing for eval
        if use_grad_ckpt:
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass

        ppl_results = {}
        for L in eval_lengths:
            if L in test_datasets:
                ppl_results[f"L={L}"] = eval_ppl(
                    model, test_datasets[L], inv_freq,
                    label=f"{config_name} L={L}",
                    max_chunks=max_ppl_chunks,
                )

        # ── Evaluate Passkey ────────────────────────────────────────
        passkey_results = {}
        if not args.no_passkey:
            print(f"\n  [F] Evaluating passkey retrieval ({config_name})...")

            # For passkey at extrapolated lengths, apply YaRN
            for L in passkey_lengths:
                if L > NATIVE_CTX:
                    scale = L / NATIVE_CTX
                    scaled_freq = yarn_inv_freq(inv_freq, scale, HEAD_DIM)
                    patch_inv_freq_for_eval(model, scaled_freq)

                try:
                    passkey_at_L = eval_passkey(
                        model, tokenizer, [L],
                        n_trials=passkey_trials,
                        seed=args.seed,
                    )
                    passkey_results.update(passkey_at_L)
                except Exception as e:
                    print(f"    ERROR at L={L}: {e}")
                    passkey_results[f"L={L}"] = {"accuracy": -1, "error": str(e)}
                finally:
                    # Restore original freq
                    if L > NATIVE_CTX:
                        patch_inv_freq_for_eval(model, inv_freq)
                    cleanup_gpu()

        # ── Compile results ─────────────────────────────────────────
        result = {
            "config": config_name,
            "label": config_label,
            "model": args.model_dir,
            "tau": tau if config_name == "evq" else 0.0,
            "lora_rank": args.lora_rank,
            "lora_targets": TARGET_MODULES,
            "steps": args.steps,
            "lr": args.lr,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "train_time_sec": round(train_time, 1),
            "final_loss": round(losses[-1], 6) if losses else None,
            "ppl": ppl_results,
            "passkey": passkey_results,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Result saved: {result_path}")

        all_results[config_name] = result

        # ── Cleanup GPU ─────────────────────────────────────────────
        print(f"\n  [G] Cleaning up ({config_name})...")
        del model
        cleanup_gpu()
        print(f"  {gpu_mem_str()}")

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  COMPARISON TABLE: GEO vs EVQ-Cosh")
    print(f"{'=' * 80}")

    # PPL table
    print(f"\n  PERPLEXITY:")
    header = f"  {'Config':<12}"
    for L in eval_lengths:
        header += f"  {'L=' + str(L):>10}"
    print(header)
    print("  " + "-" * (12 + 12 * len(eval_lengths)))

    for cname in config_names:
        r = all_results.get(cname, {})
        ppl = r.get("ppl", {})
        line = f"  {cname:<12}"
        for L in eval_lengths:
            v = ppl.get(f"L={L}", {}).get("ppl")
            if v is not None and v != float("inf"):
                line += f"  {v:>10.2f}"
            else:
                line += f"  {'--':>10}"
        print(line)

    # Delta row
    if "geo" in all_results and "evq" in all_results:
        line = f"  {'delta %':<12}"
        for L in eval_lengths:
            geo_ppl = all_results["geo"].get("ppl", {}).get(f"L={L}", {}).get("ppl")
            evq_ppl = all_results["evq"].get("ppl", {}).get(f"L={L}", {}).get("ppl")
            if geo_ppl and evq_ppl and geo_ppl != float("inf") and evq_ppl != float("inf"):
                delta = (evq_ppl / geo_ppl - 1.0) * 100
                line += f"  {delta:>+9.1f}%"
            else:
                line += f"  {'--':>10}"
        print(line)

    # Passkey table
    if not args.no_passkey:
        print(f"\n  PASSKEY RETRIEVAL ACCURACY:")
        header = f"  {'Config':<12}"
        for L in passkey_lengths:
            header += f"  {'L=' + str(L):>10}"
        print(header)
        print("  " + "-" * (12 + 12 * len(passkey_lengths)))

        for cname in config_names:
            r = all_results.get(cname, {})
            pk = r.get("passkey", {})
            line = f"  {cname:<12}"
            for L in passkey_lengths:
                v = pk.get(f"L={L}", {}).get("accuracy")
                if v is not None and v >= 0:
                    line += f"  {v:>9.0%} "
                else:
                    line += f"  {'--':>10}"
            print(line)

    # Training loss
    print(f"\n  TRAINING:")
    print(f"  {'Config':<12}  {'Final Loss':>12}  {'Time (min)':>12}")
    print("  " + "-" * 40)
    for cname in config_names:
        r = all_results.get(cname, {})
        fl = r.get("final_loss")
        tt = r.get("train_time_sec")
        fl_str = f"{fl:.4f}" if fl is not None else "--"
        tt_str = f"{tt / 60:.1f}" if tt is not None else "--"
        print(f"  {cname:<12}  {fl_str:>12}  {tt_str:>12}")

    # Save aggregate
    agg_path = output_dir / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to: {agg_path}")

    print(f"\n{'=' * 80}")
    print("  DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
