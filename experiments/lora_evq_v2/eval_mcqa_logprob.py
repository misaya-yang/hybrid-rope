#!/usr/bin/env python3
"""
Multi-Choice QA via Log-Probability (不做 generation)
=====================================================
用 QuALITY 数据集的多选题，算每个选项的条件 NLL，选最低的。
直接出 accuracy%，reviewer 最买账的指标。

同时算 Gold-Answer NLL（方案3）。

Usage:
    # Base
    python eval_mcqa_logprob.py --base_only --output_dir ./results
    # EVQ-LoRA
    python eval_mcqa_logprob.py --adapter_dir ./checkpoints/evq_r64_tau1414 --output_dir ./results
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from typing import Dict, List, Optional
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
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
        inject_inv_freq(model, inv_freq)
        mods = find_rotary_modules(model)
        if mods:
            actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
            geo = compute_geometric_inv_freq(128, 500000.0)
            err_evq = (actual - inv_freq.to(torch.float64)).abs().max().item()
            err_geo = (actual - geo).abs().max().item()
            tag = "EVQ" if err_evq < err_geo else "GEO!"
            print(f"[ROPE] tau={data.get('tau','?')}, verify: {tag}")

    model.eval()
    return model, tokenizer


def compute_option_nll(model, tokenizer, prompt_text, option_text, max_len=16384):
    """Compute NLL of option_text conditioned on prompt_text."""
    device = next(model.parameters()).device

    prompt_ids = tokenizer(prompt_text, return_tensors=None, truncation=True,
                           max_length=max_len - 100)["input_ids"]
    option_ids = tokenizer(option_text, return_tensors=None, add_special_tokens=False)["input_ids"]

    full_ids = prompt_ids + option_ids
    if len(full_ids) > max_len:
        # Truncate prompt from left to fit
        excess = len(full_ids) - max_len
        full_ids = prompt_ids[excess:] + option_ids

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    answer_start = len(full_ids) - len(option_ids)

    with torch.no_grad():
        logits = model(input_tensor).logits[0]  # [seq, vocab]

    # NLL of answer tokens
    nlls = []
    for i in range(answer_start, len(full_ids)):
        if i == 0:
            continue
        log_probs = F.log_softmax(logits[i - 1], dim=-1)
        token_nll = -log_probs[full_ids[i]].item()
        nlls.append(token_nll)

    return np.mean(nlls) if nlls else float("inf")


def load_quality_data(max_samples=100, min_context_len=2000):
    """Load QuALITY dataset (multi-choice reading comprehension)."""
    print("[DATA] Loading QuALITY dataset...")

    try:
        from datasets import load_dataset
        # Try multiple dataset paths
        for name in ["emozilla/quality", "nyu-mll/quality", "QuALITY"]:
            try:
                ds = load_dataset(name, split="validation", trust_remote_code=True)
                print(f"[DATA] Loaded {name}: {len(ds)} samples")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("QuALITY not available")
    except Exception as e:
        print(f"[DATA] QuALITY load failed: {e}")
        print("[DATA] Generating synthetic MCQA instead")
        return generate_synthetic_mcqa(max_samples)

    samples = []
    for item in ds:
        context = item.get("article", item.get("context", ""))
        questions = item.get("questions", [])
        options_list = item.get("options", [])
        gold_labels = item.get("gold_label", item.get("answer", []))

        if isinstance(questions, str):
            questions = [questions]
        if isinstance(gold_labels, (int, str)):
            gold_labels = [gold_labels]

        for i, q in enumerate(questions):
            if i >= len(options_list) or i >= len(gold_labels):
                break
            opts = options_list[i] if isinstance(options_list[i], list) else options_list
            gold = gold_labels[i] if isinstance(gold_labels, list) else gold_labels

            if isinstance(gold, str) and gold.isdigit():
                gold = int(gold)
            if isinstance(gold, int):
                gold_idx = gold - 1 if gold >= 1 else gold  # 1-indexed to 0-indexed
            else:
                gold_idx = 0

            samples.append({
                "context": context,
                "question": q,
                "options": opts if isinstance(opts, list) else [opts],
                "gold_idx": gold_idx,
            })

        if len(samples) >= max_samples:
            break

    print(f"[DATA] Prepared {len(samples)} MCQA samples")
    return samples[:max_samples]


def generate_synthetic_mcqa(n=50):
    """Generate synthetic long-context MCQA if QuALITY unavailable."""
    import random
    rng = random.Random(42)

    filler_sentences = [
        "The temperature in the region has been relatively stable over recent decades.",
        "Researchers continue to investigate the underlying mechanisms of this phenomenon.",
        "Economic indicators suggest a moderate recovery in the coming quarters.",
        "The committee reviewed several proposals before reaching a consensus.",
        "Infrastructure development remains a priority for the local government.",
        "Statistical analysis reveals a complex interplay of multiple factors.",
    ]

    facts = [
        ("Project Alpha", "started in January", ["January", "March", "June", "September"]),
        ("The budget", "was set at 50 million", ["50 million", "30 million", "75 million", "100 million"]),
        ("Dr. Chen", "led the research team", ["Dr. Chen", "Dr. Smith", "Dr. Park", "Dr. Jones"]),
        ("The deadline", "is December 15th", ["December 15th", "November 1st", "January 30th", "March 1st"]),
        ("The headquarters", "is located in Tokyo", ["Tokyo", "London", "New York", "Berlin"]),
        ("The success rate", "reached 94 percent", ["94 percent", "78 percent", "86 percent", "67 percent"]),
    ]

    samples = []
    for i in range(n):
        fact = facts[i % len(facts)]
        subject, detail, options = fact

        # Build long context with fact buried in it
        n_filler = rng.randint(200, 400)
        paragraphs = []
        insert_pos = rng.randint(n_filler // 3, 2 * n_filler // 3)
        for j in range(n_filler):
            if j == insert_pos:
                paragraphs.append(f"{subject} {detail}. This was confirmed by multiple sources.")
            paragraphs.append(rng.choice(filler_sentences))

        context = " ".join(paragraphs)
        question = f"According to the text, what is true about {subject.lower()}?"

        # Shuffle options, track gold
        gold_idx = 0  # correct answer is always first before shuffle
        indexed = list(enumerate(options))
        rng.shuffle(indexed)
        new_gold = [j for j, (orig_i, _) in enumerate(indexed) if orig_i == 0][0]

        samples.append({
            "context": context,
            "question": question,
            "options": [opt for _, opt in indexed],
            "gold_idx": new_gold,
        })

    print(f"[DATA] Generated {len(samples)} synthetic MCQA samples")
    return samples


def eval_mcqa(model, tokenizer, samples, max_len=16384):
    """Run MCQA evaluation via logprob scoring."""
    print(f"\n{'='*60}")
    print(f"MCQA LogProb Evaluation ({len(samples)} samples)")
    print(f"{'='*60}")

    correct = 0
    total = 0
    gold_nlls = []
    labels = "ABCD"

    for i, s in enumerate(samples):
        context = s["context"]
        question = s["question"]
        options = s["options"]
        gold_idx = s["gold_idx"]

        # Build prompt
        opts_text = "\n".join(f"{labels[j]}. {opt}" for j, opt in enumerate(options))
        prompt = (f"Read the following text and answer the question.\n\n"
                  f"Text: {context}\n\n"
                  f"Question: {question}\n\n"
                  f"Options:\n{opts_text}\n\n"
                  f"Answer:")

        # Score each option
        option_nlls = []
        for j, opt in enumerate(options):
            answer_text = f" {labels[j]}. {opt}"
            nll = compute_option_nll(model, tokenizer, prompt, answer_text, max_len=max_len)
            option_nlls.append(nll)

        pred_idx = np.argmin(option_nlls)
        is_correct = (pred_idx == gold_idx)
        if is_correct:
            correct += 1
        total += 1

        # Gold answer NLL
        if gold_idx < len(option_nlls):
            gold_nlls.append(option_nlls[gold_idx])

        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] pred={labels[pred_idx]} gold={labels[gold_idx]} "
                  f"{'OK' if is_correct else 'FAIL'} "
                  f"nlls=[{', '.join(f'{n:.2f}' for n in option_nlls)}]")

    acc = correct / total if total > 0 else 0
    avg_gold_nll = np.mean(gold_nlls) if gold_nlls else 0

    print(f"\n  Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"  Gold-Answer NLL: {avg_gold_nll:.4f}")

    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "gold_answer_nll": round(avg_gold_nll, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=MODEL_DEFAULT)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--max_len", type=int, default=16384)
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

    # Load data
    samples = load_quality_data(max_samples=args.max_samples)

    t0 = time.time()
    results = eval_mcqa(model, tokenizer, samples, max_len=args.max_len)
    results["variant"] = variant
    results["time_min"] = round((time.time() - t0) / 60, 2)

    path = os.path.join(args.output_dir, f"mcqa_{variant}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
