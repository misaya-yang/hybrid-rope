#!/usr/bin/env python3
"""
Gold-Answer NLL on LongBench QA tasks.
======================================
给长文档 + 问题 + 正确答案，算答案部分的 NLL。
不需要 generation。context 在 8K-16K+，直接测 EVQ 的长距离理解能力。

用 LongBench 的 qasper, hotpotqa, narrativeqa (有 context + answers)。
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DEFAULT = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
CKPT_DEFAULT = "/root/autodl-tmp/lora_evq_v2/checkpoints/evq_r64_tau1414"


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
        from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
        inject_inv_freq(model, data["inv_freq"])
        mods = find_rotary_modules(model)
        if mods:
            actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
            geo = compute_geometric_inv_freq(128, 500000.0)
            err_evq = (actual - data["inv_freq"].to(torch.float64)).abs().max().item()
            err_geo = (actual - geo).abs().max().item()
            print(f"[ROPE] tau={data.get('tau','?')}, {'EVQ' if err_evq < err_geo else 'GEO!'}")
    model.eval()
    return model, tokenizer


def compute_answer_nll(model, tokenizer, context, question, answer, max_len=16384):
    """Compute NLL of answer tokens given context + question."""
    device = next(model.parameters()).device

    prompt = f"Read the following text and answer the question.\n\nText: {context}\n\nQuestion: {question}\n\nAnswer: "
    prompt_ids = tokenizer(prompt, return_tensors=None, truncation=True,
                           max_length=max_len - 200)["input_ids"]
    answer_ids = tokenizer(answer, return_tensors=None, add_special_tokens=False)["input_ids"]

    full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_len:
        excess = len(full_ids) - max_len
        prompt_ids = prompt_ids[excess:]
        full_ids = prompt_ids + answer_ids

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    ans_start = len(full_ids) - len(answer_ids)

    with torch.no_grad():
        logits = model(input_tensor).logits[0]

    nlls = []
    for i in range(ans_start, len(full_ids)):
        if i == 0:
            continue
        log_probs = F.log_softmax(logits[i - 1], dim=-1)
        nlls.append(-log_probs[full_ids[i]].item())

    return {
        "nll": np.mean(nlls) if nlls else float("inf"),
        "n_answer_tokens": len(answer_ids),
        "n_prompt_tokens": len(prompt_ids),
        "total_tokens": len(full_ids),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=MODEL_DEFAULT)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--max_len", type=int, default=16384)
    p.add_argument("--tasks", default="qasper,hotpotqa,narrativeqa")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    tasks = [t.strip() for t in args.tasks.split(",")]
    all_results = {"variant": variant}
    t0 = time.time()

    from datasets import load_dataset

    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"  {task_name} — Gold-Answer NLL")
        print(f"{'='*60}")

        try:
            ds = load_dataset("THUDM/LongBench", task_name, split="test",
                             trust_remote_code=True)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        nlls_short = []  # context <= 8K tokens
        nlls_long = []   # context > 8K tokens
        all_nlls = []
        n = min(len(ds), args.max_samples)

        for i in range(n):
            item = ds[i]
            context = item.get("context", "")
            question = item.get("input", "")
            answers = item.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            if not answers or not answers[0]:
                continue

            answer = answers[0]
            result = compute_answer_nll(model, tokenizer, context, question, answer,
                                       max_len=args.max_len)

            ctx_tokens = result["n_prompt_tokens"]
            is_long = ctx_tokens > 8192
            bucket = nlls_long if is_long else nlls_short
            bucket.append(result["nll"])
            all_nlls.append(result["nll"])

            if i < 3 or (i + 1) % 10 == 0:
                tag = "LONG" if is_long else "short"
                print(f"  [{i+1}/{n}] NLL={result['nll']:.4f} "
                      f"ctx={ctx_tokens} ans={result['n_answer_tokens']} [{tag}]")

        task_result = {
            "mean_nll": round(np.mean(all_nlls), 4) if all_nlls else None,
            "mean_nll_short": round(np.mean(nlls_short), 4) if nlls_short else None,
            "mean_nll_long": round(np.mean(nlls_long), 4) if nlls_long else None,
            "n_short": len(nlls_short),
            "n_long": len(nlls_long),
            "n_total": len(all_nlls),
        }
        all_results[task_name] = task_result

        print(f"\n  Overall NLL: {task_result['mean_nll']}")
        if nlls_short:
            print(f"  Short (<=8K): {task_result['mean_nll_short']} (n={len(nlls_short)})")
        if nlls_long:
            print(f"  Long (>8K):   {task_result['mean_nll_long']} (n={len(nlls_long)})")

    all_results["time_min"] = round((time.time() - t0) / 60, 2)
    path = os.path.join(args.output_dir, f"gold_nll_{variant}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
