#!/usr/bin/env python3
"""
EVQ Attention Quality Evaluation
=================================
不依赖 generation 文本匹配，直接测量 PE 质量：

1. PPL @ 多长度 (8K/16K/32K) — 纯 loss，不生成
2. Attention Distance Distribution — 长距离注意力权重对比
3. Passkey Attention Probe — 看 needle 位置的注意力是否集中

用法:
    python eval_attention_quality.py \
        --model_name /path/to/llama3-8b-instruct \
        --adapter_dir ./checkpoints/evq_r64_tau1414 \
        --output_dir ./results

    # Base only
    python eval_attention_quality.py \
        --model_name /path/to/llama3-8b-instruct \
        --base_only --output_dir ./results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. PPL at multiple context lengths (no generation)
# ---------------------------------------------------------------------------

def eval_ppl(model, tokenizer, text_path: Optional[str] = None,
             eval_lengths=[8192, 16384, 32768], n_chunks=5) -> Dict:
    """Sliding-window PPL at multiple context lengths. Pure loss, no generation."""
    print("\n" + "=" * 60)
    print("EVAL 1: Perplexity @ Multiple Lengths")
    print("=" * 60)

    if text_path and os.path.exists(text_path):
        text = Path(text_path).read_text()
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                         trust_remote_code=True)
        text = "\n".join(ds["text"])

    enc = tokenizer(text, return_tensors="pt", truncation=False)
    full_ids = enc["input_ids"][0]
    print(f"  Total tokens: {len(full_ids)}")

    results = {}
    for ctx_len in eval_lengths:
        if len(full_ids) < ctx_len * 2:
            print(f"  Skip {ctx_len}: not enough data")
            continue

        nlls = []
        for i in range(n_chunks):
            start = i * ctx_len
            if start + ctx_len > len(full_ids):
                break
            chunk = full_ids[start:start + ctx_len].unsqueeze(0)
            chunk = chunk.to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(chunk, labels=chunk)
                nlls.append(out.loss.item())

        if nlls:
            ppl = math.exp(np.mean(nlls))
            results[f"ppl_{ctx_len // 1024}K"] = round(ppl, 3)
            print(f"  PPL@{ctx_len // 1024}K = {ppl:.3f} (n={len(nlls)} chunks)")

    return results


# ---------------------------------------------------------------------------
# 2. Attention Distance Distribution
# ---------------------------------------------------------------------------

def eval_attention_distance(model, tokenizer, ctx_len=2048, n_samples=3,
                            text_path: Optional[str] = None) -> Dict:
    """
    用 hook 逐层提取 QK 点积算注意力距离分布。
    每次只处理一层的 attention，避免 OOM。
    """
    print("\n" + "=" * 60)
    print("EVAL 2: Attention Distance Distribution")
    print("=" * 60)

    if text_path and os.path.exists(text_path):
        text = Path(text_path).read_text()
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                         trust_remote_code=True)
        text = "\n".join(ds["text"])

    enc = tokenizer(text, return_tensors="pt", truncation=False)
    full_ids = enc["input_ids"][0]
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    # 用 hook 捕获每层的 Q, K
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            # LlamaAttention forward 内部：q, k 在 args 或通过 module 获取
            # 直接从 output 拿 hidden_states 没法拿到 QK
            # 改用在 rotary_emb 后捕获 — 但更简单的方法是：
            # 重新计算一次 QK attention score
            pass
        return hook_fn

    # 更简单的方法：用小 context，手动算 attention
    # 找到所有 attention 层
    attn_modules = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'q_proj') and hasattr(mod, 'k_proj'):
            attn_modules.append((name, mod))

    print(f"  Found {len(attn_modules)} attention layers, ctx_len={ctx_len}")

    avg_distances_per_layer = [[] for _ in range(len(attn_modules))]
    long_ratios_per_layer = [[] for _ in range(len(attn_modules))]

    for s in range(n_samples):
        start = s * ctx_len
        if start + ctx_len > len(full_ids):
            break
        chunk = full_ids[start:start + ctx_len].unsqueeze(0).to(device)

        # Get hidden states at each layer via hook
        hidden_states_per_layer = []

        hooks = []
        for idx, (name, mod) in enumerate(attn_modules):
            def make_pre_hook(layer_idx):
                def hook(module, args):
                    # args[0] is hidden_states input to attention
                    hidden_states_per_layer.append(args[0].detach())
                return hook
            h = mod.register_forward_pre_hook(make_pre_hook(idx))
            hooks.append(h)

        with torch.no_grad():
            model(chunk)

        for h in hooks:
            h.remove()

        # Now compute attention manually for each layer
        for idx, (name, mod) in enumerate(attn_modules):
            if idx >= len(hidden_states_per_layer):
                break
            hs = hidden_states_per_layer[idx]  # [1, seq, hidden]
            seq_len = hs.shape[1]

            # Project Q, K
            with torch.no_grad():
                q = mod.q_proj(hs)  # [1, seq, n_heads*head_dim]
                k = mod.k_proj(hs)

            # Reshape to heads
            q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)  # [1, heads, seq, dim]
            k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)

            # Apply rotary (get cos/sin from model)
            # Simpler: just compute raw QK similarity (pre-softmax)
            # This tells us the attention pattern structure
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, heads, seq, seq]

            # Causal mask
            causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)[0]  # [heads, seq, seq]

            # Distance matrix
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()

            # Average attention distance
            weighted = (attn_weights * dist.unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # [heads]
            avg_dist = weighted.mean().item()
            avg_distances_per_layer[idx].append(avg_dist)

            # Long-range ratio
            threshold = ctx_len // 4
            long_mask = (dist > threshold).unsqueeze(0).float()
            lr = (attn_weights * long_mask).sum().item() / (attn_weights.sum().item() + 1e-10)
            long_ratios_per_layer[idx].append(lr)

            del q, k, scores, attn_weights
            torch.cuda.empty_cache()

        hidden_states_per_layer.clear()
        print(f"  Sample {s}: done")

    # Summarize
    results = {}
    all_avg = []
    all_lr = []
    for l in range(len(attn_modules)):
        if avg_distances_per_layer[l]:
            ad = np.mean(avg_distances_per_layer[l])
            lr = np.mean(long_ratios_per_layer[l])
            results[f"avg_dist_L{l}"] = round(ad, 2)
            results[f"long_ratio_L{l}"] = round(lr, 6)
            all_avg.append(ad)
            all_lr.append(lr)

    results["avg_dist_mean"] = round(np.mean(all_avg), 2) if all_avg else 0
    results["long_ratio_mean"] = round(np.mean(all_lr), 6) if all_lr else 0

    print(f"\n  Overall avg attention distance: {results['avg_dist_mean']:.2f}")
    print(f"  Overall long-range ratio (>{ctx_len//4}): {results['long_ratio_mean']:.6f}")

    show = [0, 1, 2, 15, 16, 17, 29, 30, 31]
    print(f"\n  {'Layer':>8s}  {'AvgDist':>10s}  {'LongRatio':>12s}")
    for l in show:
        if f"avg_dist_L{l}" in results:
            print(f"  {'L'+str(l):>8s}  {results[f'avg_dist_L{l}']:10.2f}  {results[f'long_ratio_L{l}']:12.6f}")

    return results


# ---------------------------------------------------------------------------
# 3. Passkey Attention Probe (does attention focus on the needle?)
# ---------------------------------------------------------------------------

def eval_passkey_attention(model, tokenizer, ctx_len=8192, n_trials=3) -> Dict:
    """
    Insert a passkey at known position. Check if attention concentrates there.
    No generation needed — just check attention weights at the needle position.
    """
    import random
    print("\n" + "=" * 60)
    print("EVAL 3: Passkey Attention Probe")
    print("=" * 60)

    device = next(model.parameters()).device
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    results = {"trials": []}

    for t in range(n_trials):
        rng = random.Random(42 + t)
        passkey = str(rng.randint(10000, 99999))
        needle = f"The secret passkey is {passkey}. Remember this number."

        # Build context
        n_filler = ctx_len - 200
        filler_text = (filler * (n_filler // 15))[:n_filler * 4]
        depth = rng.uniform(0.2, 0.8)
        insert_char = int(len(filler_text) * depth)
        context = filler_text[:insert_char] + f" {needle} " + filler_text[insert_char:]

        query = f"What is the secret passkey? Answer with only the number."
        messages = [{"role": "user", "content": f"{context}\n\n{query}"}]

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len)
        input_ids = enc["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # Find needle token positions
        needle_enc = tokenizer(passkey, add_special_tokens=False)["input_ids"]
        needle_len = len(needle_enc)

        # Search for needle in input_ids
        ids_list = input_ids[0].tolist()
        needle_start = -1
        for i in range(len(ids_list) - needle_len + 1):
            if ids_list[i:i + needle_len] == needle_enc:
                needle_start = i
                break

        if needle_start < 0:
            print(f"  Trial {t}: needle not found in tokens, skip")
            continue

        needle_end = needle_start + needle_len
        print(f"  Trial {t}: passkey={passkey}, depth={depth:.2f}, "
              f"needle_pos=[{needle_start}-{needle_end}]/{seq_len}")

        # Forward pass with attention
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)

        # Check attention at the LAST token (where generation would start)
        # to the needle region
        n_layers = len(out.attentions)
        needle_attn_per_layer = []

        for l, attn in enumerate(out.attentions):
            # attn: [1, n_heads, seq, seq]
            # Look at last token attending to needle region
            last_to_all = attn[0, :, -1, :]  # [n_heads, seq]
            needle_attn = last_to_all[:, needle_start:needle_end].sum(dim=-1)  # [n_heads]
            avg_needle_attn = needle_attn.mean().item()
            needle_attn_per_layer.append(avg_needle_attn)

        # Summary for this trial
        avg_attn = np.mean(needle_attn_per_layer)
        max_layer_attn = max(needle_attn_per_layer)
        max_layer_idx = needle_attn_per_layer.index(max_layer_attn)

        trial_result = {
            "depth": round(depth, 3),
            "needle_pos": [needle_start, needle_end],
            "seq_len": seq_len,
            "avg_needle_attn": round(avg_attn, 6),
            "max_layer_attn": round(max_layer_attn, 6),
            "max_layer_idx": max_layer_idx,
        }
        results["trials"].append(trial_result)
        print(f"    Avg needle attention: {avg_attn:.6f}")
        print(f"    Max layer attention: L{max_layer_idx} = {max_layer_attn:.6f}")

    # Overall
    if results["trials"]:
        results["mean_needle_attn"] = round(
            np.mean([t["avg_needle_attn"] for t in results["trials"]]), 6)
        print(f"\n  Overall mean needle attention: {results['mean_needle_attn']:.6f}")

    return results


# ---------------------------------------------------------------------------
# Model loading (same as fixed eval scripts)
# ---------------------------------------------------------------------------

def load_model(model_name, adapter_dir=None, inv_freq_path=None,
               load_in_4bit=False, bf16=True):
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

    # Load adapter FIRST
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("[LORA] Adapter loaded")

    # Inject inv_freq AFTER adapter
    if inv_freq_path and os.path.exists(inv_freq_path):
        data = torch.load(inv_freq_path, map_location="cpu", weights_only=True)
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
        inject_inv_freq(model, inv_freq)
        # Verify
        mods = find_rotary_modules(model)
        if mods:
            actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
            geo = compute_geometric_inv_freq(128, 500000.0)
            err_evq = (actual - inv_freq.to(torch.float64)).abs().max().item()
            err_geo = (actual - geo).abs().max().item()
            print(f"[ROPE] EVQ injected (τ={data.get('tau','?')}), "
                  f"verify: {'✅ EVQ' if err_evq < err_geo else '❌ GEO!'}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--wikitext_path", default="/root/autodl-tmp/data/wikitext2/wikitext2_test.txt")

    # What to run
    p.add_argument("--no_ppl", action="store_true")
    p.add_argument("--no_attn_dist", action="store_true")
    p.add_argument("--no_passkey_probe", action="store_true")

    # Config
    p.add_argument("--ppl_lengths", default="8192,16384,32768")
    p.add_argument("--attn_ctx", type=int, default=2048,
                   help="Context length for attention distance analysis")
    p.add_argument("--attn_samples", type=int, default=3)
    p.add_argument("--passkey_trials", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inv_freq_path = None
    if args.adapter_dir and not args.base_only:
        c = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(c):
            inv_freq_path = c

    model, tokenizer = load_model(
        args.model_name,
        adapter_dir=None if args.base_only else args.adapter_dir,
        inv_freq_path=inv_freq_path,
        bf16=args.bf16)

    variant = "base" if args.base_only else "evq"
    all_results = {"variant": variant, "model": args.model_name}
    t0 = time.time()

    # 1. PPL
    if not args.no_ppl:
        ppl_lengths = [int(x) for x in args.ppl_lengths.split(",")]
        all_results["ppl"] = eval_ppl(model, tokenizer, args.wikitext_path, ppl_lengths)

    # 2. Attention distance
    if not args.no_attn_dist:
        all_results["attention_distance"] = eval_attention_distance(
            model, tokenizer, ctx_len=args.attn_ctx,
            n_samples=args.attn_samples, text_path=args.wikitext_path)

    # 3. Passkey attention probe
    if not args.no_passkey_probe:
        all_results["passkey_attention"] = eval_passkey_attention(
            model, tokenizer, ctx_len=8192, n_trials=args.passkey_trials)

    all_results["time_min"] = round((time.time() - t0) / 60, 2)

    path = os.path.join(args.output_dir, f"attn_quality_{variant}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"DONE ({variant}) — {all_results['time_min']:.1f} min")
    print(f"Results: {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
