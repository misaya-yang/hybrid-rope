#!/usr/bin/env python3
"""
RULER Benchmark Evaluation (Self-contained)
============================================
Implements the 6 most PE-sensitive tasks from RULER (COLM 2024, NVIDIA).
No dependency on NVIDIA's codebase — synthetic data generated in-script.

Tasks (following RULER protocol):
  1. S-NIAH   — Single Needle in a Haystack (vanilla passkey)
  2. MK-NIAH  — Multi-Key NIAH (1 target + 3 distractors)
  3. MV-NIAH  — Multi-Value NIAH (4 values sharing 1 key)
  4. MQ-NIAH  — Multi-Query NIAH (4 independent queries)
  5. KV-Retr  — Key-Value Retrieval (full-context UUID pairs)
  6. VT       — Variable Tracking (multi-hop chain resolution)

All tasks are evaluated at configurable context lengths [4K, 8K, 16K, 32K].
Outputs a JSON + prints a summary table for paper inclusion.

Usage:
    # EVQ-LoRA model
    python eval_ruler.py \
        --model_name /path/to/llama3-8b-instruct \
        --adapter_dir ./checkpoints/evq_r64_tau1414 \
        --output_dir ./results

    # Base model comparison
    python eval_ruler.py \
        --model_name /path/to/llama3-8b-instruct \
        --base_only \
        --output_dir ./results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────
# Haystack / filler generation
# ──────────────────────────────────────────────────────────

NOISE_SENTENCES = [
    "The development of efficient algorithms remains a central focus in computational research.",
    "Statistical methods provide a foundation for understanding complex systems.",
    "Infrastructure planning requires careful consideration of population growth trends.",
    "Material science innovations continue to drive advances in manufacturing processes.",
    "Environmental monitoring systems produce large volumes of time-series data.",
    "Regulatory frameworks must adapt to accommodate technological progress.",
    "Collaborative research initiatives often yield higher-impact results.",
    "Historical analysis reveals recurring patterns in economic cycles.",
    "The integration of automated systems has improved production consistency.",
    "Advances in communication technology have transformed global information exchange.",
    "Resource allocation strategies must balance short-term needs with long-term objectives.",
    "Quantitative assessment tools provide measurable indicators of system performance.",
    "Cross-disciplinary approaches often lead to breakthrough discoveries.",
    "Policy evaluation requires both qualitative and quantitative evidence.",
    "Standard operating procedures ensure reproducibility across experimental trials.",
    "Demographic shifts influence demand patterns across various market segments.",
    "Simulation models help predict outcomes under different scenario assumptions.",
    "Quality control protocols minimize variation in output specifications.",
    "Longitudinal studies provide insight into temporal trends and causal relationships.",
    "Data governance practices are essential for maintaining research integrity.",
]


def make_haystack(n_tokens: int, tokenizer, seed: int = 0) -> str:
    """Generate filler text of approximately *n_tokens* length."""
    rng = random.Random(seed)
    parts, est = [], 0
    while est < n_tokens:
        s = rng.choice(NOISE_SENTENCES)
        parts.append(s)
        est += len(s.split()) + 2  # rough
    text = " ".join(parts)
    ids = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
    if len(ids) > n_tokens:
        text = tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)
    return text


def random_uuid_short(rng: random.Random) -> str:
    return uuid.UUID(int=rng.getrandbits(128)).hex[:12]


def build_prompt(context: str, question: str, tokenizer, max_len: int):
    """Tokenise a user-turn prompt, truncating context if needed."""
    messages = [{"role": "user", "content": f"{context}\n\n{question}"}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                f"\n\n{context}\n\n{question}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    return enc["input_ids"]


def gen_text(model, tokenizer, input_ids, max_new: int = 64) -> str:
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────
# Task 1 — S-NIAH (Single Needle)
# ──────────────────────────────────────────────────────────

def task_s_niah(model, tokenizer, ctx_len: int, n_trials: int = 20,
                seed: int = 42) -> float:
    correct = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        key = str(rng.randint(10000, 99999))
        needle = f"The special identification code is {key}."
        depth = rng.uniform(0.1, 0.9)

        hay_before = make_haystack(int((ctx_len - 200) * depth), tokenizer, seed=t)
        hay_after = make_haystack(int((ctx_len - 200) * (1 - depth)), tokenizer, seed=t + 1000)
        context = f"{hay_before}\n{needle}\n{hay_after}"
        question = "What is the special identification code? Answer with only the number."

        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=20)
        if key in resp:
            correct += 1
    return correct / n_trials


# ──────────────────────────────────────────────────────────
# Task 2 — MK-NIAH (Multi-Key: 1 target + 3 distractors)
# ──────────────────────────────────────────────────────────

def task_mk_niah(model, tokenizer, ctx_len: int, n_trials: int = 20,
                 n_distractors: int = 3, seed: int = 42) -> float:
    correct = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        target_key = f"ITEM-{rng.randint(100,999)}"
        target_val = str(rng.randint(10000, 99999))
        distractors = [(f"ITEM-{rng.randint(100,999)}", str(rng.randint(10000, 99999)))
                       for _ in range(n_distractors)]

        all_needles = [(target_key, target_val)] + distractors
        rng.shuffle(all_needles)

        # Distribute needles across context
        usable = ctx_len - 400
        spacing = usable // (len(all_needles) + 1)
        segments = []
        for i, (k, v) in enumerate(all_needles):
            segments.append(make_haystack(spacing, tokenizer, seed=t * 100 + i))
            segments.append(f"\nRecord: {k} has code {v}.\n")
        segments.append(make_haystack(spacing, tokenizer, seed=t * 100 + 99))
        context = "".join(segments)

        question = f"What is the code for {target_key}? Answer with only the number."
        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=20)
        if target_val in resp:
            correct += 1
    return correct / n_trials


# ──────────────────────────────────────────────────────────
# Task 3 — MV-NIAH (Multi-Value: 4 values for 1 key)
# ──────────────────────────────────────────────────────────

def task_mv_niah(model, tokenizer, ctx_len: int, n_trials: int = 20,
                 n_values: int = 4, seed: int = 42) -> float:
    total_recall = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        key = f"PROJECT-{rng.randint(100,999)}"
        values = [str(rng.randint(1000, 9999)) for _ in range(n_values)]

        usable = ctx_len - 400
        spacing = usable // (n_values + 1)
        segments = []
        for i, v in enumerate(values):
            segments.append(make_haystack(spacing, tokenizer, seed=t * 200 + i))
            segments.append(f"\n{key} member-{i+1}: {v}.\n")
        segments.append(make_haystack(spacing, tokenizer, seed=t * 200 + 99))
        context = "".join(segments)

        question = (f"{key} has {n_values} members. List ALL member codes "
                    f"separated by commas. Answer with only the numbers.")
        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=60)

        found = sum(1 for v in values if v in resp)
        total_recall += found / n_values
    return total_recall / n_trials


# ──────────────────────────────────────────────────────────
# Task 4 — MQ-NIAH (Multi-Query: 4 independent queries)
# ──────────────────────────────────────────────────────────

def task_mq_niah(model, tokenizer, ctx_len: int, n_trials: int = 20,
                 n_queries: int = 4, seed: int = 42) -> float:
    total_recall = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        pairs = [(f"AGENT-{rng.randint(100,999)}", str(rng.randint(10000, 99999)))
                 for _ in range(n_queries)]

        usable = ctx_len - 400
        spacing = usable // (n_queries + 1)
        segments = []
        for i, (k, v) in enumerate(pairs):
            segments.append(make_haystack(spacing, tokenizer, seed=t * 300 + i))
            segments.append(f"\n{k} clearance: {v}.\n")
        segments.append(make_haystack(spacing, tokenizer, seed=t * 300 + 99))
        context = "".join(segments)

        keys_str = ", ".join(k for k, _ in pairs)
        question = (f"List the clearance codes for each: {keys_str}. "
                    f"Format: AGENT-XXX: YYYYY, one per line.")
        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=120)

        found = sum(1 for _, v in pairs if v in resp)
        total_recall += found / n_queries
    return total_recall / n_trials


# ──────────────────────────────────────────────────────────
# Task 5 — KV Retrieval (UUID saturated context)
# ──────────────────────────────────────────────────────────

def task_kv_retrieval(model, tokenizer, ctx_len: int, n_trials: int = 20,
                      seed: int = 42) -> float:
    correct = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        # Fill almost entire context with KV pairs
        n_pairs = max(10, (ctx_len - 300) // 40)
        pairs = [(random_uuid_short(rng), random_uuid_short(rng))
                 for _ in range(n_pairs)]

        kv_text = "\n".join(f"{k} => {v}" for k, v in pairs)
        query_idx = rng.randint(0, len(pairs) - 1)
        qk, qv = pairs[query_idx]

        context = f"KEY-VALUE STORE:\n{kv_text}\nEND OF STORE."
        question = f"What is the value mapped to key {qk}? Answer with only the value."
        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=30)
        if qv in resp:
            correct += 1
    return correct / n_trials


# ──────────────────────────────────────────────────────────
# Task 6 — Variable Tracking (multi-hop chain)
# ──────────────────────────────────────────────────────────

def task_variable_tracking(model, tokenizer, ctx_len: int, n_trials: int = 20,
                           n_hops: int = 4, seed: int = 42) -> float:
    correct = 0
    for t in range(n_trials):
        rng = random.Random(seed + t)
        # Create chain: X0 = value, X1 = X0, X2 = X1, ..., Xn = X(n-1)
        var_names = [f"VAR_{chr(65+i)}" for i in range(n_hops + 1)]
        init_value = str(rng.randint(1000, 9999))

        statements = [f"{var_names[0]} = {init_value}"]
        for i in range(1, n_hops + 1):
            statements.append(f"{var_names[i]} = {var_names[i-1]}")

        # Scatter statements across context with filler
        usable = ctx_len - 400
        spacing = usable // (len(statements) + 1)
        segments = []
        for i, stmt in enumerate(statements):
            segments.append(make_haystack(spacing, tokenizer, seed=t * 400 + i))
            segments.append(f"\nASSIGNMENT: {stmt}\n")
        segments.append(make_haystack(spacing, tokenizer, seed=t * 400 + 99))
        context = "".join(segments)

        last_var = var_names[-1]
        question = (f"Given the variable assignments above, what is the final "
                    f"value of {last_var}? Answer with only the number.")
        ids = build_prompt(context, question, tokenizer, ctx_len)
        resp = gen_text(model, tokenizer, ids, max_new=20)
        if init_value in resp:
            correct += 1
    return correct / n_trials


# ──────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────

def load_model(model_name, adapter_dir=None, inv_freq_path=None,
               load_in_4bit=False, bf16=True):
    """Load model for evaluation.

    Default: bf16 full precision (no quantization).
    When adapter_dir is given, loads bf16 base + LoRA adapter (no merge),
    so base weights stay full precision and adapter adds on top.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = {"trust_remote_code": True,
          "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
          "attn_implementation": "sdpa",
          "device_map": "auto"}
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)

    if inv_freq_path and os.path.exists(inv_freq_path):
        data = torch.load(inv_freq_path, map_location="cpu", weights_only=True)
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        from train_evq_lora import inject_inv_freq
        inject_inv_freq(model, inv_freq)
        print(f"[ROPE] Injected EVQ-cosh (τ={data.get('tau','?')})")

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        # 不做 merge_and_unload: bf16 base + bf16 adapter, 全精度推理
        print("[LORA] Adapter loaded (no merge, full-precision inference)")

    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "S-NIAH":  task_s_niah,
    "MK-NIAH": task_mk_niah,
    "MV-NIAH": task_mv_niah,
    "MQ-NIAH": task_mq_niah,
    "KV-Retr": task_kv_retrieval,
    "VT":      task_variable_tracking,
}


def parse_args():
    p = argparse.ArgumentParser(description="RULER Evaluation")
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_only", action="store_true")

    p.add_argument("--context_lengths", default="4096,8192,16384,32768")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--tasks", default="S-NIAH,MK-NIAH,MV-NIAH,MQ-NIAH,KV-Retr,VT")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 5 trials per task")

    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit quant (default: OFF for eval, use bf16 full precision)")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    n_trials = 5 if args.quick else args.n_trials

    inv_freq_path = None
    if args.adapter_dir and not args.base_only:
        c = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(c):
            inv_freq_path = c

    model, tokenizer = load_model(
        args.model_name,
        adapter_dir=None if args.base_only else args.adapter_dir,
        inv_freq_path=inv_freq_path,
        load_in_4bit=args.load_in_4bit, bf16=args.bf16)

    variant = "base" if args.base_only else "evq"
    results: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"Unknown task: {task_name}"); continue
        fn = TASK_REGISTRY[task_name]
        results[task_name] = {}
        print(f"\n{'─'*50}")
        print(f"  {task_name}")
        print(f"{'─'*50}")
        for cl in ctx_lengths:
            score = fn(model, tokenizer, cl, n_trials=n_trials)
            results[task_name][f"{cl//1024}K"] = round(score, 4)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {cl//1024:>3d}K: {score:5.1%} {bar}")

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RULER SUMMARY  ({variant})  [{elapsed/60:.1f} min]")
    print(f"{'='*70}")
    header = f"{'Task':<12s}" + "".join(f"  {cl//1024:>3d}K" for cl in ctx_lengths) + "   AVG"
    print(header)
    print("-" * len(header))

    task_avgs = []
    for tn in tasks:
        if tn not in results: continue
        row = f"{tn:<12s}"
        vals = []
        for cl in ctx_lengths:
            k = f"{cl//1024}K"
            v = results[tn].get(k, 0)
            vals.append(v)
            row += f"  {v:5.1%}"
        avg = np.mean(vals)
        task_avgs.append(avg)
        row += f"  {avg:5.1%}"
        print(row)

    overall = np.mean(task_avgs) if task_avgs else 0
    print("-" * len(header))
    print(f"{'OVERALL':<12s}" + " " * (6 * len(ctx_lengths)) + f"  {overall:5.1%}")

    # ── Save ─────────────────────────────────────────────
    out = {
        "variant": variant,
        "model": args.model_name,
        "context_lengths": ctx_lengths,
        "n_trials": n_trials,
        "results": results,
        "overall": round(overall, 4),
        "eval_time_min": round(elapsed / 60, 2),
    }
    path = os.path.join(args.output_dir, f"ruler_{variant}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
