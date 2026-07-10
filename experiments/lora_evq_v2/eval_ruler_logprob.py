#!/usr/bin/env python3
"""
RULER Evaluation via Log-Probability (不依赖 generation)
========================================================
给模型喂 context + 正确答案，看答案 token 位置的 log-prob。
完全绕过 generation style / instruction following 的影响。

指标：answer_logprob = 模型给正确答案 token 的平均 log-probability
     answer_rank = 正确答案 token 在 vocab 中的平均排名（越低越好）
     answer_acc = 正确答案 token 是否是 top-1 预测

这比 generation + string matching 更直接测量 PE 质量。
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, uuid
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ruler import make_haystack, NOISE_SENTENCES


def build_context_with_answer(context, question, answer, tokenizer, max_len):
    """Build full sequence: context + question + answer, return answer token positions."""
    messages = [
        {"role": "user", "content": f"{context}\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        full_text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                     f"\n\n{context}\n\n{question}<|eot_id|>"
                     f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>")

    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)

    # Find where answer tokens start
    # Build prompt-only (without answer)
    messages_prompt = [{"role": "user", "content": f"{context}\n\n{question}"}]
    try:
        prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                       f"\n\n{context}\n\n{question}<|eot_id|>"
                       f"<|start_header_id|>assistant<|end_header_id|>\n\n")

    prompt_enc = tokenizer(prompt_text, return_tensors=None, truncation=True, max_length=max_len)
    answer_start = len(prompt_enc["input_ids"])

    return full_enc, answer_start


def eval_logprob(model, tokenizer, input_ids, answer_start):
    """Compute log-prob metrics for answer tokens."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    if answer_start >= seq_len - 1:
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab]

    # For each answer position, check the log-prob of the actual next token
    answer_logprobs = []
    answer_ranks = []
    answer_top1 = []

    for pos in range(answer_start - 1, seq_len - 1):
        next_token = input_ids[0, pos + 1].item()
        token_logits = logits[pos]  # [vocab]
        log_probs = F.log_softmax(token_logits, dim=-1)

        lp = log_probs[next_token].item()
        answer_logprobs.append(lp)

        # Rank
        rank = (token_logits > token_logits[next_token]).sum().item() + 1
        answer_ranks.append(rank)
        answer_top1.append(1 if rank == 1 else 0)

    return {
        "mean_logprob": np.mean(answer_logprobs),
        "mean_rank": np.mean(answer_ranks),
        "top1_acc": np.mean(answer_top1),
        "n_tokens": len(answer_logprobs),
    }


# ──── Task generators (same as eval_ruler but return answer string) ────

def gen_s_niah(ctx_len, tokenizer, seed):
    rng = random.Random(seed)
    key = str(rng.randint(10000, 99999))
    needle = f"The special identification code is {key}."
    depth = rng.uniform(0.1, 0.9)
    hay_b = make_haystack(int((ctx_len - 200) * depth), tokenizer, seed=seed)
    hay_a = make_haystack(int((ctx_len - 200) * (1 - depth)), tokenizer, seed=seed + 1000)
    context = f"{hay_b}\n{needle}\n{hay_a}"
    question = "What is the special identification code? Answer with only the number."
    return context, question, key


def gen_mk_niah(ctx_len, tokenizer, seed, n_dist=3):
    rng = random.Random(seed)
    target_key = f"ITEM-{rng.randint(100,999)}"
    target_val = str(rng.randint(10000, 99999))
    distractors = [(f"ITEM-{rng.randint(100,999)}", str(rng.randint(10000, 99999)))
                   for _ in range(n_dist)]
    all_needles = [(target_key, target_val)] + distractors
    rng.shuffle(all_needles)
    usable = ctx_len - 400
    spacing = usable // (len(all_needles) + 1)
    segments = []
    for i, (k, v) in enumerate(all_needles):
        segments.append(make_haystack(spacing, tokenizer, seed=seed * 100 + i))
        segments.append(f"\nRecord: {k} has code {v}.\n")
    segments.append(make_haystack(spacing, tokenizer, seed=seed * 100 + 99))
    context = "".join(segments)
    question = f"What is the code for {target_key}? Answer with only the number."
    return context, question, target_val


def gen_kv_retrieval(ctx_len, tokenizer, seed):
    rng = random.Random(seed)
    def uuid_short():
        return uuid.UUID(int=rng.getrandbits(128)).hex[:12]
    n_pairs = max(10, (ctx_len - 300) // 40)
    pairs = [(uuid_short(), uuid_short()) for _ in range(n_pairs)]
    kv_text = "\n".join(f"{k} => {v}" for k, v in pairs)
    qi = rng.randint(0, len(pairs) - 1)
    qk, qv = pairs[qi]
    context = f"KEY-VALUE STORE:\n{kv_text}\nEND OF STORE."
    question = f"What is the value mapped to key {qk}? Answer with only the value."
    return context, question, qv


def gen_vt(ctx_len, tokenizer, seed, n_hops=4):
    rng = random.Random(seed)
    var_names = [f"VAR_{chr(65+i)}" for i in range(n_hops + 1)]
    init_value = str(rng.randint(1000, 9999))
    statements = [f"{var_names[0]} = {init_value}"]
    for i in range(1, n_hops + 1):
        statements.append(f"{var_names[i]} = {var_names[i-1]}")
    usable = ctx_len - 400
    spacing = usable // (len(statements) + 1)
    segments = []
    for i, stmt in enumerate(statements):
        segments.append(make_haystack(spacing, tokenizer, seed=seed * 400 + i))
        segments.append(f"\nASSIGNMENT: {stmt}\n")
    segments.append(make_haystack(spacing, tokenizer, seed=seed * 400 + 99))
    context = "".join(segments)
    question = f"What is the final value of {var_names[-1]}? Answer with only the number."
    return context, question, init_value


TASK_GENERATORS = {
    "S-NIAH": gen_s_niah,
    "MK-NIAH": gen_mk_niah,
    "KV-Retr": gen_kv_retrieval,
    "VT": gen_vt,
}


# ──── Model loading ────

def load_model(model_name, adapter_dir=None, inv_freq_path=None, bf16=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="sdpa", device_map="auto")

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("[LORA] Adapter loaded")

    if inv_freq_path and os.path.exists(inv_freq_path):
        data = torch.load(inv_freq_path, map_location="cpu", weights_only=True)
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
        inject_inv_freq(model, inv_freq)
        mods = find_rotary_modules(model)
        if mods:
            actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
            geo = compute_geometric_inv_freq(128, 500000.0)
            err_evq = (actual - inv_freq.to(torch.float64)).abs().max().item()
            err_geo = (actual - geo).abs().max().item()
            print(f"[ROPE] EVQ (τ={data.get('tau','?')}): "
                  f"{'✅ EVQ' if err_evq < err_geo else '❌ GEO!'}")

    model.eval()
    return model, tokenizer


# ──── Main ────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--context_lengths", default="4096,8192,16384,32768")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--tasks", default="S-NIAH,MK-NIAH,KV-Retr,VT")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    inv_freq_path = None
    if args.adapter_dir and not args.base_only:
        c = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(c):
            inv_freq_path = c

    model, tokenizer = load_model(
        args.model_name,
        adapter_dir=None if args.base_only else args.adapter_dir,
        inv_freq_path=inv_freq_path)

    variant = "base" if args.base_only else "evq"
    results = {}
    t0 = time.time()

    for task_name in tasks:
        if task_name not in TASK_GENERATORS:
            continue
        gen_fn = TASK_GENERATORS[task_name]
        results[task_name] = {}

        print(f"\n{'─'*50}")
        print(f"  {task_name} (log-prob)")
        print(f"{'─'*50}")

        for ctx_len in ctx_lengths:
            trial_results = []
            for t in range(args.n_trials):
                context, question, answer = gen_fn(ctx_len, tokenizer, seed=42 + t)
                enc, ans_start = build_context_with_answer(
                    context, question, answer, tokenizer, ctx_len + 200)

                metrics = eval_logprob(model, tokenizer, enc["input_ids"], ans_start)
                if metrics:
                    trial_results.append(metrics)

            if trial_results:
                avg_lp = np.mean([r["mean_logprob"] for r in trial_results])
                avg_rank = np.mean([r["mean_rank"] for r in trial_results])
                avg_top1 = np.mean([r["top1_acc"] for r in trial_results])
                results[task_name][f"{ctx_len//1024}K"] = {
                    "logprob": round(avg_lp, 4),
                    "rank": round(avg_rank, 2),
                    "top1_acc": round(avg_top1, 4),
                }
                bar = "█" * int(avg_top1 * 10) + "░" * (10 - int(avg_top1 * 10))
                print(f"    {ctx_len//1024:>3d}K: top1={avg_top1:5.1%} logp={avg_lp:.3f} rank={avg_rank:.1f} {bar}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"RULER LOG-PROB SUMMARY ({variant}) [{elapsed/60:.1f} min]")
    print(f"{'='*70}")
    header = f"{'Task':<12s}" + "".join(f"  {cl//1024:>3d}K" for cl in ctx_lengths)
    print(header)
    print("-" * len(header))
    for tn in tasks:
        if tn not in results:
            continue
        row = f"{tn:<12s}"
        for cl in ctx_lengths:
            k = f"{cl//1024}K"
            if k in results[tn]:
                row += f"  {results[tn][k]['top1_acc']:5.1%}"
            else:
                row += f"    N/A"
        print(row)

    out = {"variant": variant, "results": results, "eval_time_min": round(elapsed/60, 2)}
    path = os.path.join(args.output_dir, f"ruler_logprob_{variant}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
