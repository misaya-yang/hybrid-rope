#!/usr/bin/env python3
"""
LongBench NLL-based evaluation for from-scratch models.

Core idea: Instead of generating answers (requires instruction-tuned model),
we compute the conditional NLL of the gold answer given (context + question).
Lower NLL = model assigns higher probability to correct answer = better understanding.

This lets us evaluate downstream long-context tasks on base models (350M/750M)
without instruction fine-tuning.

Usage (custom GPT checkpoint):
    python eval_longbench_nll.py \
        --model_path /path/to/step_15258.pt \
        --tier 750m \
        --rope_type geo \
        --tasks qa4 \
        --max_context_len 4096 \
        --method_name geo_750m \
        --output_dir results/longbench_nll/

Usage (HuggingFace model):
    python eval_longbench_nll.py \
        --model_path /path/to/hf_model \
        --tasks qa4 \
        --max_context_len 4096 \
        --output_dir results/longbench_nll/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Setup path for importing from run_evq_sweep ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# HF mirror for Chinese mainland servers
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
try:
    import huggingface_hub.utils._pagination as _hf_pag
    _orig_get_next_page = _hf_pag._get_next_page
    def _patched_get_next_page(response):
        url = _orig_get_next_page(response)
        if url and "huggingface.co" in url:
            mirror = os.environ.get("HF_ENDPOINT", "").rstrip("/")
            if mirror and mirror != "https://huggingface.co":
                url = url.replace("https://huggingface.co", mirror)
        return url
    _hf_pag._get_next_page = _patched_get_next_page
except Exception:
    pass

# ── LongBench task definitions ──────────────────────────────────────────────

NLL_FRIENDLY_TASKS = {
    "qasper": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "hotpotqa": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "2wikimqa": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "narrativeqa": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "multifieldqa_en": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "musique": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "triviaqa": {"type": "qa", "dataset": "THUDM/LongBench", "split": "test"},
    "gov_report": {"type": "summary", "dataset": "THUDM/LongBench", "split": "test"},
    "multi_news": {"type": "summary", "dataset": "THUDM/LongBench", "split": "test"},
    "qmsum": {"type": "summary", "dataset": "THUDM/LongBench", "split": "test"},
    "samsum": {"type": "summary", "dataset": "THUDM/LongBench", "split": "test"},
    "trec": {"type": "classification", "dataset": "THUDM/LongBench", "split": "test"},
    "passage_retrieval_en": {"type": "retrieval", "dataset": "THUDM/LongBench", "split": "test"},
}

TASK_SETS = {
    "qa6": ["qasper", "hotpotqa", "2wikimqa", "narrativeqa", "multifieldqa_en", "musique"],
    "qa4": ["qasper", "hotpotqa", "2wikimqa", "narrativeqa"],
    "sum4": ["gov_report", "multi_news", "qmsum", "samsum"],
    "all": list(NLL_FRIENDLY_TASKS.keys()),
}

# ── Tier configs (matching run_evq_sweep / phase9f) ─────────────────────────

TIER_CONFIGS = {
    "350m": {
        "vocab_size": 50304, "hidden_size": 1024, "num_layers": 24,
        "num_heads": 16, "head_dim": 64, "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
    "750m": {
        "vocab_size": 50304, "hidden_size": 1536, "num_layers": 18,
        "num_heads": 24, "head_dim": 64, "intermediate_size": 6144,
        "max_position_embeddings": 2048,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="LongBench NLL-based eval for base models")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to .pt checkpoint or HF model directory")
    p.add_argument("--tokenizer_path", type=str, default="",
                   help="Tokenizer path (default: gpt-neox-20b for custom GPT)")
    p.add_argument("--tasks", type=str, default="qa4",
                   help="Comma-separated task names or preset: qa4, qa6, sum4, all")
    p.add_argument("--max_context_len", type=int, default=4096,
                   help="Maximum total sequence length (context+question+answer)")
    p.add_argument("--max_answer_tokens", type=int, default=256,
                   help="Maximum answer tokens to score")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Max samples per task (LongBench has ~200-500 per task)")
    p.add_argument("--truncation", type=str, default="middle", choices=["left", "middle"],
                   help="How to truncate long contexts")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--method_name", type=str, default="",
                   help="Label for this run (e.g., 'geo_750m', 'hybrid_750m')")

    # Custom GPT model args
    p.add_argument("--tier", type=str, default="", choices=["", "350m", "750m"],
                   help="Model tier for custom GPT (if set, uses custom GPT loading)")
    p.add_argument("--rope_type", type=str, default="geo", choices=["geo", "hybrid"],
                   help="RoPE type: geo (geometric) or hybrid (EVQ-Cosh)")
    p.add_argument("--tau", type=float, default=1.5, help="EVQ tau (for hybrid)")
    p.add_argument("--hybrid_r", type=int, default=16,
                   help="Number of geometric dims to keep in hybrid")
    p.add_argument("--base", type=float, default=500000.0, help="RoPE base frequency")
    return p.parse_args()


# ── inv_freq computation ─────────────────────────────────────────────────────

def geometric_inv_freq(dim=64, base=500000.0):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32
    )


def hybrid_evq_inv_freq(dim=64, base=500000.0, tau=1.5, r=16):
    n = dim // 2
    geo = torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float64
    )
    n_evq = n - r
    if n_evq <= 0:
        return geo.float()
    theta_max = geo[r].item()
    theta_min = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()


# ── Core NLL computation ────────────────────────────────────────────────────

@torch.no_grad()
def compute_answer_nll(
    model: torch.nn.Module,
    prompt_ids: List[int],
    answer_ids: List[int],
    device: torch.device,
    is_custom_gpt: bool = False,
) -> Tuple[float, int]:
    """
    Compute NLL of answer tokens conditioned on prompt.
    Supports both HF models (with labels) and custom GPT (raw logits).

    Returns: (mean_nll_per_token, num_answer_tokens)
    """
    if len(answer_ids) == 0:
        return float('nan'), 0

    input_ids = prompt_ids + answer_ids
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        if is_custom_gpt:
            # Custom GPT: forward returns logits (B, L, V)
            logits = model(x)
        else:
            # HF model: forward with labels
            labels = torch.full_like(x, -100)
            labels[:, len(prompt_ids):] = x[:, len(prompt_ids):]
            out = model(input_ids=x, labels=labels)
            return float(out.loss.item()), len(answer_ids)

    # Manual CE loss on answer tokens only
    # logits[t] predicts token[t+1], so for answer tokens at positions
    # [prompt_len, prompt_len+1, ..., end-1], we need:
    #   logits at [prompt_len-1, prompt_len, ..., end-2]
    #   targets at [prompt_len, prompt_len+1, ..., end-1]
    prompt_len = len(prompt_ids)
    answer_logits = logits[0, prompt_len - 1 : prompt_len + len(answer_ids) - 1]  # (A, V)
    answer_targets = x[0, prompt_len : prompt_len + len(answer_ids)]  # (A,)

    loss = F.cross_entropy(answer_logits.float(), answer_targets, reduction='mean')
    return float(loss.item()), len(answer_ids)


# ── LongBench data formatting ──────────────────────────────────────────────

def format_prompt(sample: dict, task_name: str) -> Tuple[str, str]:
    """Format a LongBench sample into (prompt, answer) strings."""
    context = sample.get("context", "")
    question = sample.get("input", "")
    answers = sample.get("answers", [])

    if not answers:
        return "", ""

    gold_answer = answers[0] if isinstance(answers, list) else str(answers)
    task_type = NLL_FRIENDLY_TASKS.get(task_name, {}).get("type", "qa")

    if task_type == "qa":
        prompt = f"Document:\n{context}\n\nQuestion: {question}\nAnswer:"
    elif task_type == "summary":
        prompt = f"Document:\n{context}\n\nSummary:"
    elif task_type == "classification":
        prompt = f"{context}\n\n{question}\nLabel:"
    else:
        prompt = f"{context}\n\n{question}\nAnswer:"

    return prompt, f" {gold_answer}"


def truncate_prompt_ids(
    tokenizer,
    prompt_ids: List[int],
    answer_ids: List[int],
    max_total_len: int,
    strategy: str = "middle",
) -> List[int]:
    """Truncate prompt to fit within max_total_len including answer."""
    budget = max_total_len - len(answer_ids)
    if budget <= 0:
        budget = max_total_len // 2

    if len(prompt_ids) <= budget:
        return prompt_ids

    if strategy == "left":
        return prompt_ids[-budget:]
    elif strategy == "middle":
        half = budget // 2
        return prompt_ids[:half] + prompt_ids[-(budget - half):]
    else:
        return prompt_ids[:budget]


# ── Data loading ────────────────────────────────────────────────────────────

def load_longbench_task(task_name: str, max_samples: int, seed: int) -> List[dict]:
    """Load a LongBench task from HuggingFace."""
    from datasets import load_dataset
    try:
        ds = load_dataset("THUDM/LongBench", task_name, split="test",
                         trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] Failed to load {task_name} from HF: {e}")
        print(f"  Trying local path...")
        local_path = f"data/longbench/{task_name}.jsonl"
        if os.path.exists(local_path):
            data = []
            with open(local_path) as f:
                for line in f:
                    data.append(json.loads(line))
            if len(data) > max_samples:
                import random
                rng = random.Random(seed)
                data = rng.sample(data, max_samples)
            return data
        raise

    data = list(ds)
    if len(data) > max_samples:
        import random
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)
    return data


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(args):
    """Load model and tokenizer. Supports custom GPT (.pt) and HF models."""
    from transformers import AutoTokenizer

    is_custom_gpt = bool(args.tier)

    if is_custom_gpt:
        # Custom GPT model from run_evq_sweep
        from run_evq_sweep import GPT

        cfg = TIER_CONFIGS[args.tier].copy()
        dim = cfg["head_dim"]

        if args.rope_type == "hybrid":
            inv_freq = hybrid_evq_inv_freq(dim, args.base, args.tau, args.hybrid_r)
            print(f"  RoPE: hybrid (tau={args.tau}, r={args.hybrid_r}, base={args.base})")
        else:
            inv_freq = geometric_inv_freq(dim, args.base)
            print(f"  RoPE: geometric (base={args.base})")

        model = GPT(cfg, inv_freq)
        state = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        # Pre-build RoPE cache for max context length
        model.extend_rope(args.max_context_len)
        model = model.to(args.device)
        model.eval()

        # Tokenizer: always gpt-neox-20b for our custom models
        tok_path = args.tokenizer_path or "EleutherAI/gpt-neox-20b"
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # HuggingFace model
        from transformers import AutoModelForCausalLM
        tok_path = args.tokenizer_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype_map = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype_map[args.dtype],
            device_map=args.device if args.device != "cuda" else "auto",
            trust_remote_code=True,
        )
        model.eval()

    return model, tokenizer, is_custom_gpt


# ── Main evaluation loop ────────────────────────────────────────────────────

def evaluate_task(
    model, tokenizer, task_name: str, args, is_custom_gpt: bool
) -> Dict[str, float]:
    """Evaluate a single LongBench task using NLL scoring."""
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    samples = load_longbench_task(task_name, args.max_samples, args.seed)
    print(f"  Loaded {len(samples)} samples")

    nlls = []
    lengths = []
    skipped = 0

    for i, sample in enumerate(samples):
        prompt_str, answer_str = format_prompt(sample, task_name)
        if not prompt_str or not answer_str.strip():
            skipped += 1
            continue

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)

        if len(answer_ids) > args.max_answer_tokens:
            answer_ids = answer_ids[:args.max_answer_tokens]

        prompt_ids = truncate_prompt_ids(
            tokenizer, prompt_ids, answer_ids,
            args.max_context_len, args.truncation
        )

        total_len = len(prompt_ids) + len(answer_ids)
        if total_len < 10:
            skipped += 1
            continue

        nll, n_tokens = compute_answer_nll(
            model, prompt_ids, answer_ids, device, is_custom_gpt
        )
        if not math.isnan(nll) and not math.isinf(nll):
            nlls.append(nll)
            lengths.append(total_len)

        if (i + 1) % 20 == 0:
            running_mean = np.mean(nlls) if nlls else float('nan')
            print(f"  [{i+1}/{len(samples)}] running NLL: {running_mean:.4f}")

    if not nlls:
        print(f"  [WARN] No valid samples for {task_name}")
        return {"mean_nll": float('nan'), "std_nll": float('nan'),
                "n_samples": 0, "skipped": skipped}

    result = {
        "mean_nll": float(np.mean(nlls)),
        "std_nll": float(np.std(nlls)),
        "median_nll": float(np.median(nlls)),
        "n_samples": len(nlls),
        "skipped": skipped,
        "mean_seq_len": float(np.mean(lengths)),
        "ppl_from_nll": float(np.exp(np.mean(nlls))),
    }

    print(f"  Result: NLL={result['mean_nll']:.4f} (+/- {result['std_nll']:.4f}), "
          f"PPL={result['ppl_from_nll']:.2f}, n={result['n_samples']}, "
          f"avg_len={result['mean_seq_len']:.0f}")

    return result


def main():
    args = parse_args()

    if args.tasks in TASK_SETS:
        task_list = TASK_SETS[args.tasks]
    else:
        task_list = [t.strip() for t in args.tasks.split(",")]

    print(f"LongBench NLL Evaluation")
    print(f"  Model: {args.model_path}")
    print(f"  Tier: {args.tier or 'HF'}")
    print(f"  RoPE: {args.rope_type}" + (f" (tau={args.tau}, r={args.hybrid_r})" if args.rope_type == "hybrid" else ""))
    print(f"  Tasks: {task_list}")
    print(f"  Max context: {args.max_context_len}")
    print(f"  Method: {args.method_name or 'unnamed'}")
    print()

    model, tokenizer, is_custom_gpt = load_model(args)

    all_results = {}
    for task_name in task_list:
        if task_name not in NLL_FRIENDLY_TASKS:
            print(f"  [SKIP] Unknown task: {task_name}")
            continue
        result = evaluate_task(model, tokenizer, task_name, args, is_custom_gpt)
        all_results[task_name] = result

    valid_nlls = [r["mean_nll"] for r in all_results.values()
                  if not math.isnan(r["mean_nll"])]
    if valid_nlls:
        all_results["_aggregate"] = {
            "mean_nll": float(np.mean(valid_nlls)),
            "n_tasks": len(valid_nlls),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = args.method_name or "model"
    ctx = args.max_context_len
    out_file = out_dir / f"longbench_nll_{method}_ctx{ctx}.json"

    save_data = {
        "model_path": args.model_path,
        "method": args.method_name,
        "tier": args.tier,
        "rope_type": args.rope_type,
        "tau": args.tau if args.rope_type == "hybrid" else None,
        "max_context_len": args.max_context_len,
        "tasks": task_list,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Method: {args.method_name}")
    print(f"  Context: {args.max_context_len}")
    for task, res in all_results.items():
        if task.startswith("_"):
            continue
        print(f"  {task:25s}  NLL={res['mean_nll']:.4f}  PPL={res['ppl_from_nll']:.2f}  (n={res['n_samples']})")
    if "_aggregate" in all_results:
        agg = all_results["_aggregate"]
        print(f"  {'AGGREGATE':25s}  NLL={agg['mean_nll']:.4f}  ({agg['n_tasks']} tasks)")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
