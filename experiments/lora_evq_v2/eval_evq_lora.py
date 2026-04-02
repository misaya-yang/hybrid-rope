#!/usr/bin/env python3
"""
EVQ-Cosh LoRA Evaluation
========================
Runs three evaluation suites:
  1. PPL at multiple context lengths (WikiText-2)
  2. LongBench 6-task (long-context QA/summarization)
  3. Passkey retrieval (needle-in-haystack)

Compares EVQ-LoRA adapter vs base Instruct model.

Usage:
    # Full evaluation
    python eval_evq_lora.py \
        --adapter_dir ./checkpoints/evq_r64 \
        --output_dir ./results/evq_r64_eval

    # PPL only (fast)
    python eval_evq_lora.py \
        --adapter_dir ./checkpoints/evq_r64 \
        --eval_ppl --no_longbench --no_passkey

    # Base model only (no adapter)
    python eval_evq_lora.py \
        --base_only --output_dir ./results/base_instruct_eval
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
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: Optional[str] = None,
    inv_freq_path: Optional[str] = None,
    load_in_4bit: bool = False,
    bf16: bool = True,
):
    """Load model for evaluation.

    Default: bf16 full precision (no quantization).
    When adapter_dir is given, loads bf16 base + LoRA adapter (no merge),
    so base weights stay full precision and adapter adds on top.
    If GPU memory is insufficient, pass --load_in_4bit to fall back.
    """
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

    # Load adapter FIRST, then inject inv_freq (PEFT may reset rotary buffers)
    if adapter_dir:
        print(f"[LORA] Loading adapter from {adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print("[LORA] Adapter loaded (no merge, full-precision inference)")

    if inv_freq_path and os.path.exists(inv_freq_path):
        print(f"[ROPE] Loading custom inv_freq from {inv_freq_path}")
        data = torch.load(inv_freq_path, map_location="cpu", weights_only=True)
        inv_freq = data["inv_freq"] if isinstance(data, dict) else data
        from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
        result = inject_inv_freq(model, inv_freq)
        # Verify injection
        mods = find_rotary_modules(model)
        if mods:
            actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
            expected = inv_freq.detach().cpu().to(torch.float64)
            geo = compute_geometric_inv_freq(128, 500000.0)
            err_evq = (actual - expected).abs().max().item()
            err_geo = (actual - geo).abs().max().item()
            print(f"[ROPE] Injected into {result['patched_count']} modules (τ={data.get('tau', '?')})")
            print(f"[ROPE] Verify: vs_EVQ={err_evq:.2e}, vs_GEO={err_geo:.2e} "
                  f"{'✅ EVQ active' if err_evq < err_geo else '❌ STILL GEOMETRIC!'}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# PPL Evaluation
# ---------------------------------------------------------------------------

def eval_ppl(
    model,
    tokenizer,
    eval_lengths: List[int] = [8192, 16384, 32768],
    n_chunks: int = 5,
    data_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate perplexity at multiple context lengths on WikiText-2."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("PPL EVALUATION")
    print("=" * 60)

    # Load text
    if data_path and os.path.exists(data_path):
        text = Path(data_path).read_text()
    else:
        print("[PPL] Loading WikiText-2 from HuggingFace...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])

    # Tokenize full text
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    full_ids = enc["input_ids"][0]
    print(f"[PPL] Total tokens: {len(full_ids)}")

    results = {}
    for ctx_len in eval_lengths:
        if len(full_ids) < ctx_len * 2:
            print(f"[PPL] Skipping {ctx_len}: not enough data")
            continue

        nlls = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * ctx_len
            if start + ctx_len > len(full_ids):
                break
            chunk = full_ids[start : start + ctx_len].unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(chunk, labels=chunk)
                nlls.append(outputs.loss.item())

        if nlls:
            mean_nll = np.mean(nlls)
            ppl = math.exp(mean_nll)
            results[f"ppl@{ctx_len//1024}K"] = round(ppl, 3)
            results[f"nll@{ctx_len//1024}K"] = round(mean_nll, 4)
            status = "✅" if ppl < 15 else "⚠️" if ppl < 30 else "❌"
            print(f"  PPL@{ctx_len//1024}K = {ppl:.3f} (NLL={mean_nll:.4f}) {status}")

    return results


# ---------------------------------------------------------------------------
# Passkey Retrieval
# ---------------------------------------------------------------------------

def eval_passkey(
    model,
    tokenizer,
    test_lengths: List[int] = [8192, 16384, 32768],
    n_trials: int = 10,
) -> Dict[str, float]:
    """Needle-in-haystack passkey retrieval."""
    print("\n" + "=" * 60)
    print("PASSKEY RETRIEVAL")
    print("=" * 60)

    import random
    random.seed(42)

    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    results = {}

    for ctx_len in test_lengths:
        correct = 0
        for trial in range(n_trials):
            passkey = str(random.randint(10000, 99999))

            # Build context
            needle = f"The secret passkey is {passkey}. Remember this number."
            n_filler_tokens = ctx_len - 200  # reserve space for needle + query
            filler_text = (filler * (n_filler_tokens // 15))[:n_filler_tokens * 4]

            # Insert needle at random position
            insert_pos = random.randint(len(filler_text) // 4, 3 * len(filler_text) // 4)
            context = filler_text[:insert_pos] + f" {needle} " + filler_text[insert_pos:]

            query = "What is the secret passkey mentioned in the text above? Just give the number."

            messages = [
                {"role": "user", "content": f"{context}\n\n{query}"},
            ]

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{context}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=ctx_len)
            input_ids = enc["input_ids"].to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=1.0,
                )
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            if passkey in response:
                correct += 1

        accuracy = correct / n_trials
        results[f"passkey@{ctx_len//1024}K"] = round(accuracy, 3)
        status = "✅" if accuracy >= 0.95 else "⚠️" if accuracy >= 0.7 else "❌"
        print(f"  Passkey@{ctx_len//1024}K = {accuracy:.1%} ({correct}/{n_trials}) {status}")

    return results


# ---------------------------------------------------------------------------
# LongBench Evaluation
# ---------------------------------------------------------------------------

LONGBENCH_6_TASKS = [
    "qasper", "hotpotqa", "2wikimqa",
    "multi_news", "gov_report", "narrativeqa",
]

TASK_METRICS = {
    "qasper": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "multi_news": "rouge_l_f1",
    "gov_report": "rouge_l_f1",
    "narrativeqa": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "qmsum": "rouge_l_f1",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l_f1",
    "lsht": "classification",
    "passage_count": "classification",
    "passage_retrieval_en": "classification",
    "passage_retrieval_zh": "classification",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}


def compute_qa_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute token-level F1 between prediction and ground truths."""
    def _f1(pred_tokens, truth_tokens):
        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens) if pred_tokens else 0
        rec = len(common) / len(truth_tokens) if truth_tokens else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    pred_tokens = prediction.lower().split()
    best = 0.0
    for gt in ground_truths:
        gt_tokens = gt.lower().split()
        best = max(best, _f1(pred_tokens, gt_tokens))
    return best


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Simple ROUGE-L F1 implementation."""
    def _lcs_length(x, y):
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gt_tokens)
    prec = lcs / len(pred_tokens) if pred_tokens else 0
    rec = lcs / len(gt_tokens) if gt_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def eval_longbench(
    model,
    tokenizer,
    tasks: List[str] = None,
    max_gen_tokens: int = 256,
    max_context: int = 16384,
    max_samples_per_task: int = 50,
) -> Dict[str, Any]:
    """Evaluate on LongBench tasks."""
    from datasets import load_dataset

    if tasks is None:
        tasks = LONGBENCH_6_TASKS

    print("\n" + "=" * 60)
    print("LONGBENCH EVALUATION")
    print("=" * 60)

    all_results = {}

    for task_name in tasks:
        print(f"\n  [{task_name}] Loading...")
        try:
            ds = load_dataset("THUDM/LongBench", task_name, split="test",
                            trust_remote_code=True)
        except Exception as e:
            print(f"  [{task_name}] Failed to load: {e}")
            continue

        metric_type = TASK_METRICS.get(task_name, "qa_f1")
        scores = []
        n_eval = min(len(ds), max_samples_per_task)

        for i in range(n_eval):
            item = ds[i]
            context = item.get("context", "")
            question = item.get("input", "")
            answers = item.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            # Build prompt
            prompt_text = f"Read the following text and answer the question.\n\nText: {context}\n\nQuestion: {question}\n\nAnswer:"
            messages = [{"role": "user", "content": prompt_text}]

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            enc = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_context)
            input_ids = enc["input_ids"].to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
            pred = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Score
            if metric_type == "qa_f1":
                score = compute_qa_f1(pred, answers)
            elif metric_type in ("rouge_l_f1", "rouge_l_zh"):
                score = max(compute_rouge_l(pred, a) for a in answers) if answers else 0
            elif metric_type == "classification":
                score = 1.0 if any(a.lower() in pred.lower() for a in answers) else 0.0
            else:
                score = compute_qa_f1(pred, answers)

            scores.append(score)

        mean_score = np.mean(scores) if scores else 0
        all_results[task_name] = {
            "score": round(mean_score, 4),
            "n_samples": len(scores),
            "metric": metric_type,
        }
        print(f"  [{task_name}] Score: {mean_score:.4f} ({metric_type}, n={len(scores)})")

    # Overall
    task_scores = [v["score"] for v in all_results.values()]
    overall = np.mean(task_scores) if task_scores else 0
    all_results["_overall"] = round(overall, 4)
    print(f"\n  OVERALL: {overall:.4f}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EVQ-Cosh LoRA Evaluation")

    p.add_argument("--model_name", type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--adapter_dir", type=str, default=None,
                   help="Path to LoRA adapter (None = base model only)")
    p.add_argument("--output_dir", type=str, default="./results/evq_eval")
    p.add_argument("--base_only", action="store_true",
                   help="Evaluate base model without any adapter")

    # Eval selection
    p.add_argument("--eval_ppl", action="store_true", default=True)
    p.add_argument("--no_ppl", action="store_true")
    p.add_argument("--eval_passkey", action="store_true", default=True)
    p.add_argument("--no_passkey", action="store_true")
    p.add_argument("--eval_longbench", action="store_true", default=True)
    p.add_argument("--no_longbench", action="store_true")

    # PPL config
    p.add_argument("--ppl_lengths", type=str, default="8192,16384,32768")
    p.add_argument("--ppl_chunks", type=int, default=5)
    p.add_argument("--ppl_data_path", type=str, default=None)

    # Passkey config
    p.add_argument("--passkey_lengths", type=str, default="8192,16384,32768")
    p.add_argument("--passkey_trials", type=int, default=10)

    # LongBench config
    p.add_argument("--longbench_tasks", type=str, default="qasper,hotpotqa,2wikimqa,multi_news,gov_report,narrativeqa")
    p.add_argument("--longbench_max_samples", type=int, default=50)
    p.add_argument("--longbench_max_context", type=int, default=16384)

    # Hardware
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit quant (default: OFF for eval, use bf16 full precision)")
    p.add_argument("--bf16", action="store_true", default=True)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine inv_freq path
    inv_freq_path = None
    if args.adapter_dir and not args.base_only:
        candidate = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
        if os.path.exists(candidate):
            inv_freq_path = candidate

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_dir=None if args.base_only else args.adapter_dir,
        inv_freq_path=inv_freq_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
    )

    variant_name = "base_instruct" if args.base_only else "evq_lora"
    all_results = {"variant": variant_name, "model": args.model_name}

    # PPL
    if args.eval_ppl and not args.no_ppl:
        ppl_lengths = [int(x) for x in args.ppl_lengths.split(",")]
        ppl_results = eval_ppl(
            model, tokenizer,
            eval_lengths=ppl_lengths,
            n_chunks=args.ppl_chunks,
            data_path=args.ppl_data_path,
        )
        all_results["ppl"] = ppl_results

    # Passkey
    if args.eval_passkey and not args.no_passkey:
        passkey_lengths = [int(x) for x in args.passkey_lengths.split(",")]
        passkey_results = eval_passkey(
            model, tokenizer,
            test_lengths=passkey_lengths,
            n_trials=args.passkey_trials,
        )
        all_results["passkey"] = passkey_results

    # LongBench
    if args.eval_longbench and not args.no_longbench:
        lb_tasks = [t.strip() for t in args.longbench_tasks.split(",")]
        lb_results = eval_longbench(
            model, tokenizer,
            tasks=lb_tasks,
            max_samples_per_task=args.longbench_max_samples,
            max_context=args.longbench_max_context,
        )
        all_results["longbench"] = lb_results

    # Save results
    result_path = os.path.join(args.output_dir, f"eval_{variant_name}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {result_path}")
    print(f"{'=' * 60}")

    # Print comparison-ready summary
    print(f"\n📊 SUMMARY ({variant_name}):")
    if "ppl" in all_results:
        for k, v in all_results["ppl"].items():
            if k.startswith("ppl@"):
                print(f"  {k}: {v}")
    if "passkey" in all_results:
        for k, v in all_results["passkey"].items():
            print(f"  {k}: {v:.1%}")
    if "longbench" in all_results:
        lb = all_results["longbench"]
        for task in LONGBENCH_6_TASKS:
            if task in lb:
                print(f"  LB/{task}: {lb[task]['score']:.4f}")
        if "_overall" in lb:
            print(f"  LB/overall: {lb['_overall']:.4f}")


if __name__ == "__main__":
    main()
