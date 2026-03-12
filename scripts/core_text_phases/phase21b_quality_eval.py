#!/usr/bin/env python3
"""
Phase 21B: QuALITY multiple-choice QA with distractor padding.

Evaluates long-range retrieval ability by padding target documents with
distractor text to reach target context lengths. The model must locate
the relevant article buried in a long context to answer the question.

Format:
  [distractor_front] [target_article] [distractor_back]

  Question: {question}
  (A) ... (B) ... (C) ... (D) ...

  Answer:

The target article position is randomized (20%-80% of context) to
prevent positional shortcuts.

Usage:
  python phase21b_quality_eval.py \
      --model_pt /path/to/model.pt \
      --rope geo --base 500000 --tier 750m \
      --target_len 16384 \
      --eval_samples 200 \
      --data_dir /root/autodl-tmp/datasets/scrolls_quality \
      --output_dir /root/autodl-tmp/results/phase21b/quality_matrix/16k/geo_raw/
"""

import argparse
import json
import math
import os
import random
import re
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

from run_evq_sweep import (
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    evq_cosh_inv_freq,
    set_seed,
    apply_rope,
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


def parse_args():
    p = argparse.ArgumentParser(description="Phase 21B: QuALITY eval with distractor padding")
    p.add_argument("--model_pt", required=True)
    p.add_argument("--tier", default="750m", choices=["454m", "750m"])
    p.add_argument("--rope", required=True, choices=["geo", "evq"])
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--base", type=float, default=500000.0)
    p.add_argument("--yarn", type=int, default=0, choices=[0, 1])
    p.add_argument("--yarn_scale", type=float, default=2.0)
    p.add_argument("--target_len", type=int, default=8192,
                   help="Target total sequence length in tokens")
    p.add_argument("--eval_samples", type=int, default=200)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def geometric_inv_freq(dim=64, base=500000.0):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def apply_yarn_scaling(inv_freq, scale):
    dim = len(inv_freq) * 2
    factor = scale ** (dim / (dim - 2))
    return inv_freq / factor


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


# ── QuALITY data parsing ──────────────────────────────────────────────

def parse_quality_sample(sample):
    """Parse a SCROLLS QuALITY sample into (question_with_options, article, answer).

    SCROLLS format: input = "question\\n\\n (A)...\\n (B)...\\n (C)...\\n (D)...\\n\\n ARTICLE_TEXT"
    """
    inp = sample["input"]
    answer = sample["output"].strip()

    # Find where the article starts (after the options block)
    # Options end with (D) line, then there's usually a double newline before the article
    # Strategy: find the last option line "(D) ...", then split after it
    lines = inp.split("\n")

    # Find the line index of option (D)
    d_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("(D)") or stripped.startswith("(D "):
            d_idx = i
            break

    if d_idx == -1:
        # Fallback: try to find double newline after a short prefix
        # The question + options are usually in the first ~500 chars
        parts = inp.split("\n\n", 2)
        if len(parts) >= 3:
            question_opts = parts[0] + "\n\n" + parts[1]
            article = parts[2]
        else:
            question_opts = inp[:200]
            article = inp[200:]
    else:
        # Everything up to and including (D) line is question+options
        question_opts = "\n".join(lines[:d_idx + 1]).strip()
        # Everything after is the article
        article = "\n".join(lines[d_idx + 1:]).strip()

    return question_opts, article, answer


def build_distractor_pool(data_dir, task="quality", split="train"):
    """Load training articles as distractor text pool."""
    path = os.path.join(data_dir, task, f"{split}.jsonl")
    print(f"  Loading distractor pool: {path}")
    articles = []
    seen_articles = set()
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            _, article, _ = parse_quality_sample(sample)
            # Deduplicate by first 200 chars
            key = article[:200]
            if key not in seen_articles:
                seen_articles.add(key)
                articles.append(article)
    print(f"  Distractor pool: {len(articles)} unique articles")
    return articles


def construct_padded_sample(question_opts, article, target_len, tokenizer,
                            distractor_pool, rng):
    """Construct a distractor-padded sample matching finetune format.

    Finetune format:
      Read the following and answer the question.

      {question_opts}

      {context_with_distractors}

      Answer:

    The article is embedded among distractors at a random position (20%-80%).
    Returns: token IDs (list of ints)
    """
    # Match finetune format exactly:
    # prefix = "Read the following and answer the question.\n\n{question_opts}\n\n"
    # suffix = "\n\nAnswer:"
    prefix = f"Read the following and answer the question.\n\n{question_opts}\n\n"
    suffix = "\n\nAnswer:"

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    article_ids = tokenizer.encode(article, add_special_tokens=False)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    # Budget for context (article + distractors)
    overhead = 1 + len(prefix_ids) + len(suffix_ids)  # BOS + prefix + suffix
    context_budget = target_len - overhead
    if context_budget <= 0:
        # Fallback — just return prefix + suffix
        full_ids = [bos] + prefix_ids + suffix_ids
        return full_ids[:target_len]

    # If article fits within budget with no room for distractors
    if len(article_ids) >= context_budget:
        article_ids = article_ids[:context_budget]
        full_ids = [bos] + prefix_ids + article_ids + suffix_ids
        return full_ids[:target_len]

    # How many distractor tokens needed
    pad_needed = context_budget - len(article_ids)

    # Random position for target article (20%-80%)
    front_ratio = rng.uniform(0.2, 0.8)
    front_pad = int(pad_needed * front_ratio)
    back_pad = pad_needed - front_pad

    # Sample distractor tokens
    def sample_distractor_tokens(n_tokens):
        tokens = []
        indices = list(range(len(distractor_pool)))
        rng.shuffle(indices)
        for idx in indices:
            if len(tokens) >= n_tokens:
                break
            dist_text = "\n\n" + distractor_pool[idx] + "\n\n"
            dist_ids = tokenizer.encode(dist_text, add_special_tokens=False)
            tokens.extend(dist_ids)
        return tokens[:n_tokens]

    front_ids = sample_distractor_tokens(front_pad)
    back_ids = sample_distractor_tokens(back_pad)

    # Assemble: BOS + prefix + front_distractors + article + back_distractors + suffix
    full_ids = [bos] + prefix_ids + front_ids + article_ids + back_ids + suffix_ids

    # Ensure exact length
    if len(full_ids) > target_len:
        full_ids = full_ids[:target_len]

    return full_ids


# ── KV-cached generation (same as phase21b_eval_only.py) ──────────────

@torch.no_grad()
def generate_kv(model, input_ids, max_new_tokens=16, temperature=0.0):
    """Fast generation with KV cache."""
    model.eval()
    device = next(model.parameters()).device

    ids = input_ids.clone().to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    n_layers = len(model.blocks)
    seq_len = ids.size(1)

    amp_ctx = (torch.amp.autocast("cuda", dtype=DTYPE)
               if USE_AUTOCAST and device.type == "cuda" else nullcontext())

    # Prefill
    with amp_ctx:
        x = model.emb(ids)
        kv_caches = []

        for block in model.blocks:
            attn = block.attn
            ln_x = block.ln1(x)
            B, L, H = ln_x.shape
            qkv = attn.qkv(ln_x).view(B, L, 3, attn.nh, attn.hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            cos, sin = attn.rope(L)
            cos_b, sin_b = cos[None, None], sin[None, None]
            q = apply_rope(q, cos_b, sin_b)
            k = apply_rope(k, cos_b, sin_b)
            kv_caches.append((k, v))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn.o(out.transpose(1, 2).reshape(B, L, -1))
            x = x + attn_out
            x = x + block.mlp(block.ln2(x))

        logits = model.head(model.ln(x))
        next_logits = logits[:, -1, :]

    if temperature <= 0:
        next_id = next_logits.argmax(dim=-1, keepdim=True)
    else:
        probs = F.softmax(next_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)

    generated = [next_id]
    pos = seq_len

    if next_id.item() == 0:
        return torch.cat([ids, next_id], dim=1)[0]

    # Decode
    for step in range(1, max_new_tokens):
        with amp_ctx:
            x = model.emb(next_id)
            for i, block in enumerate(model.blocks):
                attn = block.attn
                ln_x = block.ln1(x)
                B, L_new, H = ln_x.shape
                qkv = attn.qkv(ln_x).view(B, L_new, 3, attn.nh, attn.hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                cos_full, sin_full = attn.rope(pos + 1)
                cos_pos = cos_full[pos:pos+1][None, None]
                sin_pos = sin_full[pos:pos+1][None, None]
                q = apply_rope(q, cos_pos, sin_pos)
                k = apply_rope(k, cos_pos, sin_pos)
                cached_k, cached_v = kv_caches[i]
                k_full = torch.cat([cached_k, k], dim=2)
                v_full = torch.cat([cached_v, v], dim=2)
                kv_caches[i] = (k_full, v_full)
                out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
                attn_out = attn.o(out.transpose(1, 2).reshape(B, L_new, -1))
                x = x + attn_out
                x = x + block.mlp(block.ln2(x))
            logits = model.head(model.ln(x))
            next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        generated.append(next_id)
        pos += 1
        if next_id.item() == 0:
            break

    all_ids = torch.cat([ids] + generated, dim=1)
    return all_ids[0]


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_quality(model, tokenizer, val_samples, distractor_pool, args):
    """Evaluate QuALITY accuracy with distractor-padded contexts."""
    model.eval()
    rng = random.Random(args.seed)

    n_eval = min(len(val_samples), args.eval_samples)
    samples = val_samples[:n_eval]

    correct = 0
    total = 0
    option_match = 0  # Matches by option letter (A/B/C/D)
    text_match = 0    # Matches by answer text

    print(f"\n  Evaluating QuALITY accuracy on {n_eval} samples...")
    print(f"  Target context length: {args.target_len} tokens")
    t0 = time.time()

    for i, sample in enumerate(samples):
        question_opts, article, gold_answer = parse_quality_sample(sample)
        if not article or not gold_answer:
            continue

        # Construct padded input
        input_ids = construct_padded_sample(
            question_opts, article, args.target_len,
            tokenizer, distractor_pool, rng
        )
        input_tensor = torch.tensor(input_ids, dtype=torch.long)

        # Extend rope if needed
        model.extend_rope(len(input_ids) + 32)

        # Generate answer (short — just need option letter or short text)
        output_ids = generate_kv(model, input_tensor, max_new_tokens=32)
        gen_ids = output_ids[len(input_ids):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Check correctness
        # Method 1: Option letter match (A/B/C/D)
        gold_letter = None
        # Find which option matches the gold answer
        for letter in ["A", "B", "C", "D"]:
            pattern = f"({letter})"
            if pattern in question_opts:
                # Extract option text
                next_l = chr(ord(letter) + 1)
                pat = r'\(' + letter + r'\)\s*(.*?)(?:\n|\(' + next_l + r'\)|$)'
                opt_match = re.search(pat, question_opts, re.DOTALL)
                if opt_match:
                    opt_text = opt_match.group(1).strip()
                    if gold_answer.lower() in opt_text.lower() or opt_text.lower() in gold_answer.lower():
                        gold_letter = letter
                        break

        is_correct = False

        # Check by letter
        gen_first = gen_text[:5].strip().upper() if gen_text else ""
        if gold_letter and gold_letter in gen_first[:2]:
            is_correct = True
            option_match += 1
        # Check by text similarity
        elif gold_answer.lower()[:20] in gen_text.lower()[:50]:
            is_correct = True
            text_match += 1
        # Check if generated text starts with correct option text
        elif gold_letter:
            for letter in ["A", "B", "C", "D"]:
                next_letter = chr(min(ord(letter) + 1, ord("D")))
                pattern = r'\(' + letter + r'\)\s*(.*?)(?:\n|\(' + next_letter + r'\)|$)'
                opt_match = re.search(pattern, question_opts, re.DOTALL)
                if opt_match:
                    opt_text = opt_match.group(1).strip()[:30]
                    if opt_text.lower() in gen_text.lower()[:50]:
                        if letter == gold_letter:
                            is_correct = True
                            text_match += 1
                        break

        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            acc = correct / max(total, 1) * 100
            elapsed_so_far = time.time() - t0
            rate = (i + 1) / elapsed_so_far
            eta = (n_eval - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{n_eval}] acc={acc:.1f}% ({correct}/{total})  "
                  f"({rate:.1f} samples/s, ETA={eta:.0f}s)")

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    random_baseline = 25.0

    print(f"\n  QuALITY Results ({n_eval} samples, {elapsed:.0f}s):")
    print(f"    Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"    Random baseline: {random_baseline}%")
    print(f"    Above random: {acc - random_baseline:+.2f}%")
    print(f"    (option_match={option_match}, text_match={text_match})")

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "random_baseline": random_baseline,
        "above_random": round(acc - random_baseline, 2),
        "option_match": option_match,
        "text_match": text_match,
        "eval_time_sec": round(elapsed, 1),
        "target_len": args.target_len,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print(f"  Phase 21B: QuALITY QA Eval (distractor-padded)")
    print(f"  Model: {args.model_pt}")
    print(f"  RoPE: {args.rope}  |  Target len: {args.target_len}")
    print("=" * 60)

    # Build inv_freq
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
        print(f"  YaRN scaling: scale={args.yarn_scale}")

    # Load model
    cfg = tier_cfg.copy()
    cfg["max_position_embeddings"] = args.target_len + 64
    cfg["seq_len"] = args.target_len + 64

    print(f"  Loading model from {args.model_pt}...")
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = _load_state(args.model_pt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")

    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    model.extend_rope(args.target_len + 64)
    del state

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    val_path = os.path.join(args.data_dir, "quality", "validation.jsonl")
    print(f"  Loading validation: {val_path}")
    val_samples = []
    with open(val_path) as f:
        for line in f:
            val_samples.append(json.loads(line))
    print(f"  Val: {len(val_samples)} samples")

    # Build distractor pool from training set
    distractor_pool = build_distractor_pool(args.data_dir)

    # Evaluate
    eval_result = evaluate_quality(model, tokenizer, val_samples, distractor_pool, args)

    # Save result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "21b",
        "task": "quality",
        "rope": args.rope,
        "tau": args.tau if args.rope == "evq" else None,
        "base": args.base,
        "yarn": args.yarn,
        "target_len": args.target_len,
        "eval": eval_result,
        "model_pt": args.model_pt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_file = output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_file}")
    print(json.dumps(eval_result, indent=2))


if __name__ == "__main__":
    main()
