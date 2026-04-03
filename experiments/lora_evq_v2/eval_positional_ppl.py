#!/usr/bin/env python3
"""
Positional PPL Evaluation — 统一评测脚本。
对任意 checkpoint 计算 16K WikiText 的分区间 PPL。
支持 Geometric / EVQ / YaRN 三种方法。

Usage:
    # Geometric LoRA
    python eval_positional_ppl.py --adapter_dir ./checkpoints/geo_s42 --method geo

    # EVQ LoRA
    python eval_positional_ppl.py --adapter_dir ./checkpoints/evq_s42 --method evq

    # YaRN LoRA
    python eval_positional_ppl.py --adapter_dir ./checkpoints/yarn_s42 --method yarn --yarn_factor 2.0

    # Base (no adapter)
    python eval_positional_ppl.py --base_only

    # Batch eval all checkpoints
    python eval_positional_ppl.py --batch_dir ./checkpoints/ --output results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model(model_name, adapter_dir=None, method="geo", yarn_factor=2.0, bf16=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }

    # YaRN needs rope_scaling at load time
    if method == "yarn":
        load_kwargs["rope_scaling"] = {
            "type": "yarn",
            "factor": yarn_factor,
            "original_max_position_embeddings": 8192,
        }

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Load adapter
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[LORA] Loaded adapter from {adapter_dir}")

    # EVQ: inject inv_freq AFTER adapter
    if method == "evq" and adapter_dir:
        freq_path = os.path.join(adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(freq_path):
            data = torch.load(freq_path, map_location="cpu", weights_only=True)
            from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
            inject_inv_freq(model, data["inv_freq"])
            # Verify
            mods = find_rotary_modules(model)
            if mods:
                actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
                geo = compute_geometric_inv_freq(128, 500000.0)
                err_evq = (actual - data["inv_freq"].to(torch.float64)).abs().max().item()
                err_geo = (actual - geo).abs().max().item()
                print(f"[ROPE] EVQ injected, τ={data.get('tau','?')}: "
                      f"{'✅' if err_evq < err_geo else '❌'}")

    model.eval()
    return model, tokenizer


def eval_positional_ppl(model, tokenizer, text_path, ctx_len=16384):
    """Compute PPL in position windows."""
    device = next(model.parameters()).device

    if os.path.exists(text_path):
        text = Path(text_path).read_text()
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test",
                         trust_remote_code=True)
        text = "\n".join(ds["text"])

    full_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Use multiple chunks for robustness
    n_chunks = 5
    windows = [
        (0, 4096, "0-4K"),
        (4096, 8192, "4K-8K"),
        (8192, 12288, "8K-12K"),
        (12288, min(ctx_len, 16384), "12K-16K"),
    ]

    # Accumulate losses per window across chunks
    window_losses = {name: [] for _, _, name in windows}

    for c in range(n_chunks):
        start = c * ctx_len
        if start + ctx_len > len(full_ids):
            break

        chunk = full_ids[start:start + ctx_len].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(chunk).logits[0]

        shift_logits = logits[:-1]
        shift_labels = chunk[0, 1:]
        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none")

        for w_start, w_end, name in windows:
            w_end = min(w_end, len(per_token_loss))
            if w_start < w_end:
                window_losses[name].append(per_token_loss[w_start:w_end].mean().item())

    results = {}
    for _, _, name in windows:
        if window_losses[name]:
            mean_loss = np.mean(window_losses[name])
            results[name] = {
                "loss": round(mean_loss, 4),
                "ppl": round(math.exp(mean_loss), 2),
                "n_chunks": len(window_losses[name]),
            }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",
                   default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--method", choices=["geo", "evq", "yarn", "base"], default="base")
    p.add_argument("--yarn_factor", type=float, default=2.0)
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--wikitext_path",
                   default="/root/autodl-tmp/data/wikitext2/wikitext2_test.txt")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: results/<method>_<seed>_ppl.json)")
    p.add_argument("--batch_dir", default=None,
                   help="Eval all checkpoints in this directory")
    args = p.parse_args()

    if args.base_only:
        args.method = "base"
        args.adapter_dir = None

    # Single eval
    if not args.batch_dir:
        label = f"{args.method}"
        if args.adapter_dir:
            label = os.path.basename(args.adapter_dir)

        print(f"\n{'='*60}")
        print(f"  Positional PPL: {label}")
        print(f"{'='*60}")

        model, tokenizer = load_model(
            args.model_name, args.adapter_dir,
            method=args.method, yarn_factor=args.yarn_factor)

        results = eval_positional_ppl(model, tokenizer, args.wikitext_path)

        for name, r in results.items():
            print(f"  {name}: PPL={r['ppl']:.2f} (loss={r['loss']:.4f})")

        output = {"label": label, "method": args.method, "results": results}
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved → {args.output}")
        return

    # Batch eval
    print(f"Batch eval: {args.batch_dir}")
    all_results = {}

    for ckpt_dir in sorted(glob.glob(os.path.join(args.batch_dir, "*"))):
        if not os.path.isdir(ckpt_dir):
            continue
        # Detect method from metadata
        meta_path = os.path.join(ckpt_dir, "experiment_meta.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        method = meta.get("method", meta.get("rope_method", "geo"))
        if method == "evq_cosh":
            method = "evq"
        yarn_factor = meta.get("yarn_factor", 2.0)
        label = os.path.basename(ckpt_dir)

        print(f"\n--- {label} ({method}) ---")
        model, tokenizer = load_model(
            args.model_name, ckpt_dir,
            method=method, yarn_factor=yarn_factor)

        results = eval_positional_ppl(model, tokenizer, args.wikitext_path)
        all_results[label] = {"method": method, "results": results}

        for name, r in results.items():
            print(f"  {name}: PPL={r['ppl']:.2f}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"POSITIONAL PPL SUMMARY")
    print(f"{'='*70}")
    header = f"{'Checkpoint':<30s}  {'0-4K':>8s}  {'4K-8K':>8s}  {'8K-12K':>8s}  {'12K-16K':>8s}"
    print(header)
    print("-" * len(header))
    for label, data in all_results.items():
        row = f"{label:<30s}"
        for window in ["0-4K", "4K-8K", "8K-12K", "12K-16K"]:
            if window in data["results"]:
                row += f"  {data['results'][window]['ppl']:8.2f}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
