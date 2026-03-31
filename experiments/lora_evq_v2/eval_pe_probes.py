#!/usr/bin/env python3
"""
PE Frequency-Allocation Probing Suite
======================================
4 synthetic tasks that directly probe position-encoding quality.
All tasks are fully controlled (synthetic), fast to run, and
produce continuous metrics (not just pass/fail).

Tasks:
  1. Multi-Depth Passkey (MDP)
     - 10 depths × 3 context lengths → accuracy heatmap
     - Tests: position retrieval at varying distances

  2. Multi-Needle Retrieval (MNR)
     - K=5 needles scattered in context, retrieve ALL of them
     - Tests: simultaneous multi-position tracking

  3. Positional Ordering Test (POT)
     - N=5 facts at random positions, reproduce in original order
     - Tests: relative position resolution (can model tell what came first?)

  4. Long-Range Key-Value Association (KVA)
     - Key-value pairs at start, query after long filler
     - Tests: information retention over extreme distance

Why these tasks matter for EVQ:
  EVQ-cosh concentrates frequency resolution at low frequencies,
  which preserves positional information at long range. Geometric
  RoPE and YaRN spread resolution more uniformly, wasting channels
  on short-range distinctions that attention already handles.
  These probes specifically test long-range position precision,
  where EVQ's allocation should shine.

Usage:
    python eval_pe_probes.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter_dir ./checkpoints/evq_r64_tau1414 \
        --output_dir ./results/pe_probes_evq

    # Base model comparison
    python eval_pe_probes.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --base_only \
        --output_dir ./results/pe_probes_base

    # Quick mode (fewer trials)
    python eval_pe_probes.py --adapter_dir ./checkpoints/evq_r64_tau1414 \
        --quick --output_dir ./results/pe_probes_quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Filler text generation
# ---------------------------------------------------------------------------

# Mix of plausible filler sentences to avoid repetition detection
FILLER_POOL = [
    "The temperature in the region has been relatively stable over recent decades.",
    "Researchers continue to investigate the underlying mechanisms of this phenomenon.",
    "Economic indicators suggest a moderate recovery in the coming quarters.",
    "The committee reviewed several proposals before reaching a consensus.",
    "Infrastructure development remains a priority for the local government.",
    "Statistical analysis reveals a complex interplay of multiple factors.",
    "The project timeline was adjusted to accommodate new requirements.",
    "Environmental monitoring data shows seasonal variation patterns.",
    "Collaboration between departments has improved overall efficiency.",
    "The survey results indicate diverse perspectives on the policy change.",
    "Historical records provide important context for current debates.",
    "Technological advancement continues to reshape industry practices.",
    "The budget allocation reflects shifting organizational priorities.",
    "Community engagement efforts have yielded positive feedback.",
    "Preliminary findings suggest further investigation is warranted.",
    "The regulatory framework has evolved to address emerging challenges.",
    "Quality assurance processes ensure consistent output standards.",
    "Market dynamics influence strategic planning decisions.",
    "Educational initiatives aim to bridge existing knowledge gaps.",
    "The implementation phase is expected to span several months.",
]


def generate_filler(n_tokens: int, tokenizer, seed: int = 0) -> str:
    """Generate plausible filler text of approximately n_tokens length."""
    rng = random.Random(seed)
    sentences = []
    est_tokens = 0
    while est_tokens < n_tokens:
        s = rng.choice(FILLER_POOL)
        sentences.append(s)
        est_tokens += len(s.split()) * 1.3  # rough token estimate
    text = " ".join(sentences)
    # Trim to approximate token count
    enc = tokenizer(text, truncation=False, return_tensors=None)
    if len(enc["input_ids"]) > n_tokens:
        # Truncate and decode back
        trimmed_ids = enc["input_ids"][:n_tokens]
        text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Task 1: Multi-Depth Passkey (MDP)
# ---------------------------------------------------------------------------

def eval_multi_depth_passkey(
    model, tokenizer,
    context_lengths: List[int] = [4096, 8192, 16384, 32768],
    n_depths: int = 10,
    n_trials: int = 5,
) -> Dict[str, Any]:
    """
    Place a passkey at different depth ratios within context.
    Returns: accuracy matrix [context_len × depth].
    """
    print("\n" + "=" * 60)
    print("TASK 1: Multi-Depth Passkey (MDP)")
    print("=" * 60)

    results = {}
    depth_ratios = [i / n_depths for i in range(1, n_depths + 1)]  # 0.1, 0.2, ..., 1.0

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens")
        depth_accs = {}

        for depth in depth_ratios:
            correct = 0
            for trial in range(n_trials):
                rng = random.Random(42 + trial + int(depth * 1000) + ctx_len)
                passkey = str(rng.randint(10000, 99999))

                # Generate filler
                needle = f"<<IMPORTANT>> The secret verification code is: {passkey}. Remember this code. <<END>>"
                needle_pos = int((ctx_len - 300) * depth)

                filler_before = generate_filler(needle_pos, tokenizer, seed=trial * 100)
                filler_after = generate_filler(ctx_len - needle_pos - 100, tokenizer, seed=trial * 100 + 50)

                context = f"{filler_before}\n\n{needle}\n\n{filler_after}"

                query = "What was the secret verification code mentioned in the text? Reply with ONLY the 5-digit number."
                messages = [{"role": "user", "content": f"{context}\n\n{query}"}]

                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
                input_ids = enc["input_ids"].to(model.device)

                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
                response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

                if passkey in response:
                    correct += 1

            acc = correct / n_trials
            depth_accs[f"depth_{depth:.1f}"] = acc
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            print(f"    depth={depth:.1f}: {acc:.0%} {bar}")

        results[f"ctx_{ctx_len}"] = depth_accs

    # Compute summary: average accuracy per context length
    summary = {}
    for ctx_key, depth_accs in results.items():
        avg = np.mean(list(depth_accs.values()))
        summary[ctx_key] = round(avg, 4)
    results["_avg_by_context"] = summary

    return results


# ---------------------------------------------------------------------------
# Task 2: Multi-Needle Retrieval (MNR)
# ---------------------------------------------------------------------------

def eval_multi_needle(
    model, tokenizer,
    context_lengths: List[int] = [8192, 16384, 32768],
    n_needles: int = 5,
    n_trials: int = 5,
) -> Dict[str, Any]:
    """
    Place K needles at random positions, retrieve all of them.
    Metric: fraction of needles correctly retrieved (0 to 1).
    """
    print("\n" + "=" * 60)
    print("TASK 2: Multi-Needle Retrieval (MNR)")
    print(f"  Needles per trial: {n_needles}")
    print("=" * 60)

    # Use distinct color-animal pairs as needles (easy to parse)
    COLORS = ["red", "blue", "green", "purple", "orange", "silver", "golden", "crimson"]
    ANIMALS = ["falcon", "dolphin", "panther", "phoenix", "serpent", "tiger", "eagle", "wolf"]

    results = {}

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens")
        trial_scores = []

        for trial in range(n_trials):
            rng = random.Random(42 + trial + ctx_len)

            # Generate K unique key-value pairs
            colors = rng.sample(COLORS, n_needles)
            animals = rng.sample(ANIMALS, n_needles)
            codes = [str(rng.randint(100, 999)) for _ in range(n_needles)]
            needles = []
            for c, a, code in zip(colors, animals, codes):
                needles.append((f"{c} {a}", code, f"The agent codenamed {c} {a} has ID number {code}."))

            # Place needles at evenly spaced positions with some jitter
            usable = ctx_len - 500
            spacing = usable // (n_needles + 1)
            positions = [spacing * (i + 1) + rng.randint(-spacing // 4, spacing // 4)
                        for i in range(n_needles)]
            positions.sort()

            # Build context
            segments = []
            prev_end = 0
            for pos, (name, code, needle_text) in zip(positions, needles):
                filler = generate_filler(pos - prev_end, tokenizer, seed=trial * 1000 + pos)
                segments.append(filler)
                segments.append(f"\n{needle_text}\n")
                prev_end = pos + 50  # approximate needle length

            segments.append(generate_filler(max(0, usable - prev_end), tokenizer, seed=trial * 2000))
            context = "".join(segments)

            # Query
            query_parts = [f"the {name}" for name, _, _ in needles]
            query = f"List the ID numbers for each of the following agents mentioned in the text: {', '.join(query_parts)}. Format: <agent>: <ID>"
            messages = [{"role": "user", "content": f"{context}\n\n{query}"}]

            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
            input_ids = enc["input_ids"].to(model.device)

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=200, do_sample=False)
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Score: count correctly retrieved codes
            found = 0
            for name, code, _ in needles:
                if code in response:
                    found += 1

            trial_scores.append(found / n_needles)

        avg_score = np.mean(trial_scores)
        results[f"ctx_{ctx_len}"] = {
            "recall": round(avg_score, 4),
            "trial_scores": [round(s, 4) for s in trial_scores],
        }
        bar = "█" * int(avg_score * 10) + "░" * (10 - int(avg_score * 10))
        print(f"    Recall: {avg_score:.1%} {bar}")

    return results


# ---------------------------------------------------------------------------
# Task 3: Positional Ordering Test (POT)
# ---------------------------------------------------------------------------

def eval_positional_ordering(
    model, tokenizer,
    context_lengths: List[int] = [8192, 16384, 32768],
    n_items: int = 5,
    n_trials: int = 5,
) -> Dict[str, Any]:
    """
    Place N distinct facts at random positions.
    Ask model to list them in the order they appeared.
    Metric: Kendall's tau correlation between predicted and true order.
    """
    print("\n" + "=" * 60)
    print("TASK 3: Positional Ordering Test (POT)")
    print(f"  Items per trial: {n_items}")
    print("=" * 60)

    EVENTS = [
        ("Alpha project", "launched on Monday"),
        ("Beta initiative", "started on Tuesday"),
        ("Gamma protocol", "activated on Wednesday"),
        ("Delta operation", "commenced on Thursday"),
        ("Epsilon program", "began on Friday"),
        ("Zeta deployment", "initiated on Saturday"),
        ("Eta campaign", "kicked off on Sunday"),
        ("Theta venture", "opened at dawn"),
    ]

    results = {}

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens")
        trial_taus = []

        for trial in range(n_trials):
            rng = random.Random(42 + trial + ctx_len)

            # Select N events
            events = rng.sample(EVENTS, n_items)

            # Place at random positions
            usable = ctx_len - 500
            positions = sorted(rng.sample(range(100, usable, 50), n_items))

            # Build context
            segments = []
            prev_end = 0
            for pos, (name, action) in zip(positions, events):
                filler = generate_filler(max(10, pos - prev_end), tokenizer, seed=trial * 3000 + pos)
                segments.append(filler)
                segments.append(f"\n<<EVENT>> {name} {action}. <<END>>\n")
                prev_end = pos + 30

            segments.append(generate_filler(max(10, usable - prev_end), tokenizer, seed=trial * 4000))
            context = "".join(segments)

            # Query
            shuffled_names = [name for name, _ in events]
            rng.shuffle(shuffled_names)
            query = (f"The following events were mentioned in the text: {', '.join(shuffled_names)}. "
                     f"List them in the exact order they appeared in the text, from first to last. "
                     f"Format: 1. <name>, 2. <name>, ...")

            messages = [{"role": "user", "content": f"{context}\n\n{query}"}]

            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
            input_ids = enc["input_ids"].to(model.device)

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=200, do_sample=False)
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).lower()

            # Parse predicted order
            true_order = [name.lower() for name, _ in events]
            pred_order = []
            for name in true_order:
                # Find position of this name in response
                pos_in_response = response.find(name.lower())
                pred_order.append(pos_in_response if pos_in_response >= 0 else 99999)

            # Compute Kendall's tau
            # Count concordant and discordant pairs
            concordant = 0
            discordant = 0
            for i in range(len(pred_order)):
                for j in range(i + 1, len(pred_order)):
                    if pred_order[i] < pred_order[j]:
                        concordant += 1
                    elif pred_order[i] > pred_order[j]:
                        discordant += 1

            n_pairs = len(pred_order) * (len(pred_order) - 1) / 2
            tau = (concordant - discordant) / n_pairs if n_pairs > 0 else 0
            trial_taus.append(tau)

        avg_tau = np.mean(trial_taus)
        results[f"ctx_{ctx_len}"] = {
            "kendall_tau": round(avg_tau, 4),
            "trial_taus": [round(t, 4) for t in trial_taus],
        }
        quality = "excellent" if avg_tau > 0.8 else "good" if avg_tau > 0.5 else "poor"
        print(f"    Kendall's τ: {avg_tau:.3f} ({quality})")

    return results


# ---------------------------------------------------------------------------
# Task 4: Long-Range Key-Value Association (KVA)
# ---------------------------------------------------------------------------

def eval_kv_association(
    model, tokenizer,
    context_lengths: List[int] = [8192, 16384, 32768],
    n_pairs: int = 10,
    n_trials: int = 5,
) -> Dict[str, Any]:
    """
    Place K key-value pairs at the START of context.
    Fill with long filler. Query ONE random key at the END.
    Tests: retention of structured info over extreme distance.
    """
    print("\n" + "=" * 60)
    print("TASK 4: Long-Range Key-Value Association (KVA)")
    print(f"  Key-value pairs: {n_pairs}")
    print("=" * 60)

    CITIES = ["Tokyo", "Paris", "London", "Berlin", "Sydney",
              "Moscow", "Cairo", "Mumbai", "Toronto", "Seoul",
              "Rome", "Madrid", "Oslo", "Lima", "Nairobi"]

    results = {}

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len} tokens")
        trial_scores = []

        for trial in range(n_trials):
            rng = random.Random(42 + trial + ctx_len)

            # Generate KV pairs
            cities = rng.sample(CITIES, n_pairs)
            codes = [f"{rng.randint(100,999)}-{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}" for _ in range(n_pairs)]

            kv_section = "REGISTRY OF CITY CODES:\n"
            for city, code in zip(cities, codes):
                kv_section += f"  - {city}: {code}\n"
            kv_section += "END OF REGISTRY.\n"

            # Long filler (most of the context)
            filler = generate_filler(ctx_len - 500, tokenizer, seed=trial * 5000)

            # Query a random key
            query_idx = rng.randint(0, n_pairs - 1)
            query_city = cities[query_idx]
            expected_code = codes[query_idx]

            context = f"{kv_section}\n{filler}"
            query = f"According to the registry of city codes at the beginning of this document, what is the code for {query_city}? Reply with ONLY the code."

            messages = [{"role": "user", "content": f"{context}\n\n{query}"}]

            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
            input_ids = enc["input_ids"].to(model.device)

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            hit = 1.0 if expected_code in response else 0.0
            trial_scores.append(hit)

        avg_score = np.mean(trial_scores)
        results[f"ctx_{ctx_len}"] = {
            "accuracy": round(avg_score, 4),
            "trial_scores": [round(s, 4) for s in trial_scores],
        }
        bar = "█" * int(avg_score * 10) + "░" * (10 - int(avg_score * 10))
        print(f"    Accuracy: {avg_score:.0%} {bar}")

    return results


# ---------------------------------------------------------------------------
# Model loading (reuse from eval_evq_lora.py)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: Optional[str] = None,
    inv_freq_path: Optional[str] = None,
    load_in_4bit: bool = False,
    bf16: bool = True,
):
    """Load model for evaluation. Default bf16 full precision.
    LoRA adapter loaded without merge — bf16 base + bf16 adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[MODEL] Loading: {model_name} ({'4-bit' if load_in_4bit else 'bf16'})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Inject custom inv_freq
    if inv_freq_path and os.path.exists(inv_freq_path):
        print(f"[ROPE] Loading custom inv_freq from {inv_freq_path}")
        data = torch.load(inv_freq_path, map_location="cpu", weights_only=True)
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        from train_evq_lora import inject_inv_freq
        result = inject_inv_freq(model, inv_freq)
        print(f"[ROPE] Injected (τ={data.get('tau', '?')})")

    if adapter_dir:
        print(f"[LORA] Loading adapter from {adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("[LORA] Adapter loaded (no merge, full-precision inference)")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PE Probing Suite")
    p.add_argument("--model_name", type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./results/pe_probes")
    p.add_argument("--base_only", action="store_true")

    p.add_argument("--context_lengths", type=str, default="4096,8192,16384,32768")
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer trials, fewer depths")

    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit quant (default: OFF for eval, use bf16 full precision)")
    p.add_argument("--bf16", action="store_true", default=True)

    # Task selection
    p.add_argument("--tasks", type=str, default="mdp,mnr,pot,kva",
                   help="Comma-separated task list: mdp,mnr,pot,kva")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
    tasks = [t.strip().lower() for t in args.tasks.split(",")]
    n_trials = 2 if args.quick else args.n_trials

    # Load model
    inv_freq_path = None
    if args.adapter_dir and not args.base_only:
        candidate = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(candidate):
            inv_freq_path = candidate

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_dir=None if args.base_only else args.adapter_dir,
        inv_freq_path=inv_freq_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
    )

    variant = "base" if args.base_only else "evq"
    all_results = {"variant": variant, "model": args.model_name, "context_lengths": ctx_lengths}

    t0 = time.time()

    if "mdp" in tasks:
        all_results["multi_depth_passkey"] = eval_multi_depth_passkey(
            model, tokenizer,
            context_lengths=ctx_lengths,
            n_depths=5 if args.quick else 10,
            n_trials=n_trials,
        )

    if "mnr" in tasks:
        all_results["multi_needle"] = eval_multi_needle(
            model, tokenizer,
            context_lengths=ctx_lengths,
            n_needles=5,
            n_trials=n_trials,
        )

    if "pot" in tasks:
        all_results["positional_ordering"] = eval_positional_ordering(
            model, tokenizer,
            context_lengths=ctx_lengths,
            n_items=5,
            n_trials=n_trials,
        )

    if "kva" in tasks:
        all_results["kv_association"] = eval_kv_association(
            model, tokenizer,
            context_lengths=ctx_lengths,
            n_pairs=10,
            n_trials=n_trials,
        )

    elapsed = time.time() - t0
    all_results["eval_time_minutes"] = round(elapsed / 60, 2)

    # Save
    result_path = os.path.join(args.output_dir, f"pe_probes_{variant}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"PE PROBING SUMMARY ({variant})")
    print("=" * 70)
    print(f"{'Task':<25s}", end="")
    for cl in ctx_lengths:
        print(f"  {cl//1024}K", end="")
    print()
    print("-" * (25 + 6 * len(ctx_lengths)))

    if "multi_depth_passkey" in all_results:
        mdp = all_results["multi_depth_passkey"]
        print(f"{'MDP (avg accuracy)':<25s}", end="")
        for cl in ctx_lengths:
            key = f"ctx_{cl}"
            if key in mdp:
                avg = np.mean(list(mdp[key].values()))
                print(f" {avg:4.0%}", end="")
            else:
                print(f"  N/A", end="")
        print()

    if "multi_needle" in all_results:
        mn = all_results["multi_needle"]
        print(f"{'MNR (5-needle recall)':<25s}", end="")
        for cl in ctx_lengths:
            key = f"ctx_{cl}"
            if key in mn:
                print(f" {mn[key]['recall']:4.0%}", end="")
            else:
                print(f"  N/A", end="")
        print()

    if "positional_ordering" in all_results:
        pot = all_results["positional_ordering"]
        print(f"{'POT (Kendall τ)':<25s}", end="")
        for cl in ctx_lengths:
            key = f"ctx_{cl}"
            if key in pot:
                print(f" {pot[key]['kendall_tau']:4.2f}", end="")
            else:
                print(f"  N/A", end="")
        print()

    if "kv_association" in all_results:
        kva = all_results["kv_association"]
        print(f"{'KVA (10-pair accuracy)':<25s}", end="")
        for cl in ctx_lengths:
            key = f"ctx_{cl}"
            if key in kva:
                print(f" {kva[key]['accuracy']:4.0%}", end="")
            else:
                print(f"  N/A", end="")
        print()

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
