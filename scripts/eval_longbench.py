#!/usr/bin/env python3
"""
Evaluate LongBench tasks for base vs hybrid-lora.

Tasks (default):
- qasper (single-document QA, F1)
- hotpotqa (multi-document QA, F1)
- gov_report (long summarization, Rouge-L)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


LOG = logging.getLogger("eval_longbench")


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def attn_candidates(mode: str) -> List[Optional[str]]:
    if mode == "auto":
        return ["flash_attention_2", "sdpa", None]
    return [mode]


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: Optional[str],
    merge_lora: bool,
    attn_mode: str,
    trust_remote_code: bool,
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.model_max_length = 10_000_000

    model = None
    used_attn = "default"
    errs: List[str] = []
    for attn in attn_candidates(attn_mode):
        try:
            kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            if attn is not None:
                kwargs["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
            used_attn = attn or "default"
            break
        except Exception as e:
            errs.append(f"attn={attn}: {type(e).__name__}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
    if model is None:
        raise RuntimeError("Failed to load model:\n" + "\n".join(errs))

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed.")
        p = Path(adapter_path)
        if not p.exists():
            raise FileNotFoundError(f"adapter path not found: {p}")
        has_weight = (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
        if not has_weight:
            raise FileNotFoundError(f"no adapter weight file in {p}")
        LOG.info("Loading LoRA adapter: %s", p)
        model = PeftModel.from_pretrained(model, str(p), is_trainable=False)
        if merge_lora:
            LOG.info("Merging LoRA adapter into base model.")
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, used_attn


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1(pred: str, refs: List[str]) -> float:
    pred_tokens = normalize_text(pred).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    pred_counter = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    for r in refs:
        ref_tokens = normalize_text(r).split()
        if not ref_tokens:
            continue
        ref_counter = {}
        for t in ref_tokens:
            ref_counter[t] = ref_counter.get(t, 0) + 1
        overlap = 0
        for t, c in pred_counter.items():
            overlap += min(c, ref_counter.get(t, 0))
        if overlap == 0:
            continue
        p = overlap / max(1, len(pred_tokens))
        rr = overlap / max(1, len(ref_tokens))
        f1 = 2 * p * rr / max(1e-12, p + rr)
        best = max(best, f1)
    return best


def lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l_f1(pred: str, refs: List[str]) -> float:
    pred_tokens = normalize_text(pred).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for r in refs:
        ref_tokens = normalize_text(r).split()
        if not ref_tokens:
            continue
        lcs = lcs_len(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        p = lcs / len(pred_tokens)
        rr = lcs / len(ref_tokens)
        f1 = 2 * p * rr / max(1e-12, p + rr)
        best = max(best, f1)
    return best


def as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(as_text(i) for i in x)
    if isinstance(x, dict):
        return "\n".join([f"{k}: {as_text(v)}" for k, v in x.items()])
    return str(x)


def extract_context_question_refs(task: str, sample: Dict) -> Tuple[str, str, List[str], str]:
    context_keys = ["context", "article", "document", "documents", "passage", "input", "text"]
    question_keys = ["question", "query", "instruction"]
    answer_keys = ["answers", "answer", "output", "summary", "target", "label"]

    context = ""
    for k in context_keys:
        if k in sample and sample[k] is not None:
            context = as_text(sample[k])
            if context:
                break

    question = ""
    for k in question_keys:
        if k in sample and sample[k] is not None:
            question = as_text(sample[k])
            if question:
                break

    refs: List[str] = []
    for k in answer_keys:
        if k in sample and sample[k] is not None:
            v = sample[k]
            if isinstance(v, list):
                refs.extend([as_text(x) for x in v if as_text(x)])
            else:
                s = as_text(v)
                if s:
                    refs.append(s)
            if refs:
                break

    if task == "gov_report":
        prompt = (
            "You are given a long report.\n"
            "Write a concise, faithful summary.\n\n"
            f"Report:\n{context}\n\nSummary:"
        )
    else:
        prompt = (
            "Read the context and answer the question.\n"
            "Answer as briefly and accurately as possible.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    return prompt, question, refs, context


def truncate_prompt(tokenizer: AutoTokenizer, prompt: str, max_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    ids = ids[-max_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=False)


def load_task_dataset(task: str):
    tries = [
        ("THUDM/LongBench", task),
        ("THUDM/LongBench", task.lower()),
        ("THUDM/LongBench", None),
    ]
    last_err = None
    for name, cfg in tries:
        try:
            if cfg is None:
                ds = load_dataset(name, split="test", trust_remote_code=True)
            else:
                ds = load_dataset(name, cfg, split="test", trust_remote_code=True)
            return ds
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Cannot load LongBench task={task}: {last_err}")


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, ids["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    task: str,
    max_samples: int,
    max_input_tokens: int,
    max_new_tokens_qa: int,
    max_new_tokens_sum: int,
    seed: int,
) -> Dict:
    ds = load_task_dataset(task)
    n = min(max_samples, len(ds))
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]

    scores: List[float] = []
    records: List[Dict] = []
    metric_name = "f1" if task in {"qasper", "hotpotqa"} else "rouge_l_f1"
    max_new = max_new_tokens_qa if metric_name == "f1" else max_new_tokens_sum

    for k, i in enumerate(idxs):
        sample = ds[int(i)]
        prompt, question, refs, context = extract_context_question_refs(task, sample)
        if not refs:
            continue
        prompt = truncate_prompt(tokenizer, prompt, max_input_tokens)

        pred = ""
        try:
            pred = generate_text(model, tokenizer, prompt, max_new_tokens=max_new)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                short_prompt = truncate_prompt(tokenizer, prompt, max_input_tokens // 2)
                pred = generate_text(model, tokenizer, short_prompt, max_new_tokens=max_new)
            else:
                raise

        if metric_name == "f1":
            sc = token_f1(pred, refs)
        else:
            sc = rouge_l_f1(pred, refs)
        scores.append(sc)

        if len(records) < 3:
            records.append(
                {
                    "index": int(i),
                    "question": question[:200],
                    "prediction": pred[:500],
                    "reference_0": refs[0][:500],
                    "score": sc,
                    "context_preview": context[:300],
                }
            )

        if (k + 1) % 10 == 0:
            LOG.info("task=%s progress=%d/%d running_%s=%.4f", task, k + 1, n, metric_name, float(np.mean(scores)))

    return {
        "task": task,
        "metric": metric_name,
        "num_scored": len(scores),
        "score": float(np.mean(scores)) if scores else None,
        "examples": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare base vs hybrid_lora on LongBench subset.")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--hybrid_adapter_path",
        type=str,
        required=True,
        help="Path to hybrid_lora adapter dir (contains adapter_model.safetensors/bin).",
    )
    ap.add_argument("--tasks", type=str, default="qasper,hotpotqa,gov_report")
    ap.add_argument("--max_samples_per_task", type=int, default=100)
    ap.add_argument("--max_input_tokens", type=int, default=16384)
    ap.add_argument("--max_new_tokens_qa", type=int, default=64)
    ap.add_argument("--max_new_tokens_sum", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    enforce_offline_mode()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(out_path.parent / "eval_longbench.log")

    tasks = parse_csv(args.tasks)
    LOG.info("Tasks=%s", tasks)

    results: Dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
            "base_model_path": args.base_model_path,
            "hybrid_adapter_path": args.hybrid_adapter_path,
            "tasks": tasks,
            "max_samples_per_task": args.max_samples_per_task,
            "max_input_tokens": args.max_input_tokens,
            "attn_implementation": args.attn_implementation,
        },
        "models": {},
    }

    model_specs = [
        ("base_unfinetuned", None),
        ("hybrid_lora", args.hybrid_adapter_path),
    ]

    for model_name, adapter_path in model_specs:
        LOG.info("=== Evaluate model: %s ===", model_name)
        model, tokenizer, attn_used = load_model_and_tokenizer(
            base_model_path=args.base_model_path,
            adapter_path=adapter_path,
            merge_lora=args.merge_lora,
            attn_mode=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        mres = {"attn_used": attn_used, "tasks": {}}

        for task in tasks:
            try:
                tres = evaluate_task(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    max_samples=args.max_samples_per_task,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens_qa=args.max_new_tokens_qa,
                    max_new_tokens_sum=args.max_new_tokens_sum,
                    seed=args.seed,
                )
            except Exception as e:
                tres = {
                    "task": task,
                    "error": f"{type(e).__name__}: {e}",
                    "score": None,
                }
            mres["tasks"][task] = tres

        results["models"][model_name] = mres

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Flattened comparison table for quick reading
    compare = {}
    for task in tasks:
        base_score = results["models"].get("base_unfinetuned", {}).get("tasks", {}).get(task, {}).get("score")
        hyb_score = results["models"].get("hybrid_lora", {}).get("tasks", {}).get(task, {}).get("score")
        compare[task] = {
            "base_unfinetuned": base_score,
            "hybrid_lora": hyb_score,
            "delta_hybrid_minus_base": (hyb_score - base_score) if (base_score is not None and hyb_score is not None) else None,
        }
    results["comparison"] = compare

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Saved %s", out_path)


if __name__ == "__main__":
    main()

