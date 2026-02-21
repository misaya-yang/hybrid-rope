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
import math
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def infer_rope_theta(base_model_path: str, trust_remote_code: bool) -> float:
    theta: Optional[float] = None
    try:
        cfg_dict, _ = AutoConfig.get_config_dict(
            base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        raw_theta = cfg_dict.get("rope_theta")
        if raw_theta is not None:
            theta = float(raw_theta)
    except Exception:
        pass

    if theta is None:
        cfg_json = Path(base_model_path) / "config.json"
        if cfg_json.exists():
            try:
                raw = json.loads(cfg_json.read_text(encoding="utf-8", errors="ignore"))
                raw_theta = raw.get("rope_theta")
                if raw_theta is not None:
                    theta = float(raw_theta)
            except Exception:
                pass
    if theta is None:
        theta = 10000.0
    return float(theta)


def rope_scaling_candidates(variant: str, factor: float, orig_ctx: int, rope_theta: float) -> List[Optional[dict]]:
    factor = float(factor)
    rope_theta = float(rope_theta)
    if variant == "yarn":
        return [
            {
                "rope_type": "yarn",
                "factor": factor,
                "rope_theta": rope_theta,
                "original_max_position_embeddings": int(orig_ctx),
            },
            {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta},
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
        ]
    if variant == "pi":
        return [
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
            {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta},
        ]
    if variant == "pi_soft":
        return [
            {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta},
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
        ]
    return [None]


def compute_hybrid_inv_freq(
    head_dim: int,
    theta_base: float = 500000.0,
    split_ratio: float = 0.5,
    alpha: float = 0.2,
    p: float = 3.9,
    min_freq_scale: float = 4.0,
) -> torch.Tensor:
    k = head_dim // 2
    if k <= 1:
        raise ValueError(f"Invalid head_dim: {head_dim}")
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if p <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")
    if min_freq_scale <= 0.0:
        raise ValueError(f"min_freq_scale must be > 0, got {min_freq_scale}")

    idx = torch.arange(0, k, dtype=torch.float32)
    geo_freq = 1.0 / (float(theta_base) ** (2.0 * idx / float(head_dim)))
    split_idx = int(k * split_ratio)
    split_idx = max(1, min(k - 1, split_idx))

    out = geo_freq.clone()
    tail = k - split_idx
    if tail <= 0:
        return out

    if tail == 1:
        t = torch.zeros(1, dtype=torch.float32)
    else:
        t = torch.arange(tail, dtype=torch.float32) / float(tail - 1)

    omega_anchor = float(geo_freq[split_idx].item())
    omega_min = float(geo_freq[-1].item())
    omega_min_target = omega_min * float(min_freq_scale)
    log_w = math.log(omega_anchor) + torch.pow(t, p) * (math.log(omega_min_target) - math.log(omega_anchor))
    poly_freq = torch.exp(log_w)
    out[split_idx:] = (1.0 - alpha) * geo_freq[split_idx:] + alpha * poly_freq
    return out


def patch_hybrid_rope(model: torch.nn.Module, inv_freq_cpu: torch.Tensor) -> int:
    patched = 0
    for name, module in model.named_modules():
        if not hasattr(module, "inv_freq"):
            continue
        if "rotary_emb" not in name and not name.endswith(".rotary_emb"):
            continue
        old = module.inv_freq
        new = inv_freq_cpu.to(device=old.device, dtype=old.dtype)
        if isinstance(old, torch.nn.Parameter):
            module.inv_freq = torch.nn.Parameter(new, requires_grad=False)
        else:
            module.inv_freq = new
        if hasattr(module, "max_seq_len_cached"):
            module.max_seq_len_cached = 0
        patched += 1
    if patched == 0:
        raise RuntimeError("No rotary modules patched for hybrid variant.")
    return patched


def infer_variant_and_rope_from_adapter(
    adapter_path: str,
    fallback_variant: str,
    fallback_rope_factor: float,
    fallback_orig_ctx: int,
) -> Tuple[str, Optional[dict], float, int]:
    variant = fallback_variant
    rope_cfg: Optional[dict] = None
    rope_factor = float(fallback_rope_factor)
    orig_ctx = int(fallback_orig_ctx)

    summary_path = Path(adapter_path).resolve().parent / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
            sv = str(summary.get("variant", "")).strip().lower()
            if sv:
                variant = sv
            rope = summary.get("rope", {}) if isinstance(summary.get("rope", {}), dict) else {}
            if isinstance(rope.get("rope_scaling_used"), dict):
                rope_cfg = dict(rope["rope_scaling_used"])
            if rope.get("factor") is not None:
                rope_factor = float(rope.get("factor"))
            if rope.get("orig_ctx") is not None:
                orig_ctx = int(rope.get("orig_ctx"))
        except Exception as e:
            LOG.warning("Failed to parse adapter summary %s: %s", summary_path, e)

    if variant == "auto":
        parent_name = Path(adapter_path).resolve().parent.name.strip().lower()
        variant = parent_name if parent_name in {"hybrid", "yarn", "pi", "pi_soft"} else "base"
    return variant, rope_cfg, rope_factor, orig_ctx


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: Optional[str],
    merge_lora: bool,
    attn_mode: str,
    trust_remote_code: bool,
    variant: str,
    rope_factor: float,
    orig_ctx: int,
    rope_theta: float,
    hybrid_split_ratio: float,
    hybrid_alpha: float,
    hybrid_p: float,
    hybrid_min_freq_scale: float,
) -> Tuple[torch.nn.Module, AutoTokenizer, str, str, Optional[dict]]:
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

    variant_used = variant
    rope_cfg: Optional[dict] = None
    rope_factor_used = float(rope_factor)
    orig_ctx_used = int(orig_ctx)

    if adapter_path:
        variant_used, inferred_rope_cfg, inferred_factor, inferred_orig_ctx = infer_variant_and_rope_from_adapter(
            adapter_path=adapter_path,
            fallback_variant=variant,
            fallback_rope_factor=rope_factor,
            fallback_orig_ctx=orig_ctx,
        )
        if variant != "auto":
            variant_used = variant
        if inferred_rope_cfg is not None:
            rope_cfg = inferred_rope_cfg
        rope_factor_used = inferred_factor
        orig_ctx_used = inferred_orig_ctx
    elif variant_used == "auto":
        variant_used = "base"

    if isinstance(rope_cfg, dict) and rope_cfg.get("factor") is not None:
        try:
            rope_factor_used = float(rope_cfg.get("factor"))
        except Exception:
            pass

    if variant_used in {"yarn", "pi", "pi_soft"} and rope_cfg is None:
        rope_cfg = rope_scaling_candidates(variant_used, rope_factor_used, orig_ctx_used, rope_theta)[0]

    model = None
    used_attn = "default"
    errs: List[str] = []
    for attn in attn_candidates(attn_mode):
        try:
            cfg = AutoConfig.from_pretrained(
                base_model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )

            if adapter_path and variant_used != "base":
                target_max_pos = max(
                    int(getattr(cfg, "max_position_embeddings", orig_ctx_used)),
                    int(orig_ctx_used * max(1.0, rope_factor_used)),
                )
                cfg.max_position_embeddings = target_max_pos

            if rope_cfg is not None:
                cfg.rope_scaling = dict(rope_cfg)
                if hasattr(cfg, "rope_parameters"):
                    cfg.rope_parameters = dict(rope_cfg)
                if hasattr(cfg, "rope_theta") and getattr(cfg, "rope_theta", None) is None:
                    cfg.rope_theta = float(rope_theta)

            kwargs = dict(
                config=cfg,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            if attn is not None:
                kwargs["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
            if adapter_path and variant_used == "hybrid":
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                inv = compute_hybrid_inv_freq(
                    head_dim=head_dim,
                    theta_base=rope_theta,
                    split_ratio=hybrid_split_ratio,
                    alpha=hybrid_alpha,
                    p=hybrid_p,
                    min_freq_scale=hybrid_min_freq_scale,
                )
                patched = patch_hybrid_rope(model, inv)
                LOG.info("Patched hybrid rotary layers=%d", patched)
            used_attn = attn or "default"
            break
        except Exception as e:
            errs.append(f"variant={variant_used} rope={rope_cfg} attn={attn}: {type(e).__name__}: {e}")
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
    return model, tokenizer, used_attn, variant_used, rope_cfg


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


def load_task_dataset(task: str, local_data_dir: Optional[str] = None):
    # Prefer local jsonl files when available (fully offline and deterministic).
    if local_data_dir:
        data_dir = Path(local_data_dir)
        if data_dir.exists():
            p = data_dir / f"{task}.jsonl"
            if p.exists():
                return load_dataset("json", data_files=str(p), split="train")

    # LongBench requires a config name. Do not fall back to cfg=None, otherwise
    # we may silently return a misleading "Config name is missing" error as the
    # final exception even when earlier attempts failed for a different reason.
    base = task.strip()
    candidates: List[str] = []
    for cfg in (base, base.lower(), f"{base}_e", f"{base.lower()}_e"):
        if cfg not in candidates:
            candidates.append(cfg)

    errors: List[str] = []
    for cfg in candidates:
        try:
            return load_dataset("THUDM/LongBench", cfg, split="test", trust_remote_code=True)
        except Exception as e:
            errors.append(f"cfg={cfg}: {type(e).__name__}: {e}")

    raise RuntimeError(
        f"Cannot load LongBench task={task}. Tried configs={candidates}.\n"
        + "\n".join(errors)
    )


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
    local_data_dir: Optional[str],
) -> Dict:
    ds = load_task_dataset(task, local_data_dir=local_data_dir)
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
    ap.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "base", "hybrid", "yarn", "pi", "pi_soft"],
        help="RoPE variant used during adapter training. auto=try infer from adapter summary.",
    )
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_theta", type=float, default=0.0, help="<=0 means infer from base config.")
    ap.add_argument("--hybrid_split_ratio", type=float, default=0.5)
    ap.add_argument("--hybrid_alpha", type=float, default=0.2)
    ap.add_argument("--hybrid_p", type=float, default=3.9)
    ap.add_argument("--hybrid_min_freq_scale", type=float, default=4.0)
    ap.add_argument("--tasks", type=str, default="qasper,hotpotqa,gov_report")
    ap.add_argument("--max_samples_per_task", type=int, default=100)
    ap.add_argument("--max_input_tokens", type=int, default=16384)
    ap.add_argument("--max_new_tokens_qa", type=int, default=64)
    ap.add_argument("--max_new_tokens_sum", type=int, default=256)
    ap.add_argument(
        "--longbench_local_data_dir",
        type=str,
        default=os.environ.get("LONGBENCH_LOCAL_DATA_DIR", "/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"),
        help="Directory containing local LongBench jsonl files (e.g. qasper.jsonl).",
    )
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
    rope_theta = float(args.rope_theta) if args.rope_theta > 0 else infer_rope_theta(args.base_model_path, args.trust_remote_code)
    LOG.info("Resolved rope_theta=%.1f", rope_theta)

    results: Dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
            "base_model_path": args.base_model_path,
            "hybrid_adapter_path": args.hybrid_adapter_path,
            "tasks": tasks,
            "max_samples_per_task": args.max_samples_per_task,
            "max_input_tokens": args.max_input_tokens,
            "attn_implementation": args.attn_implementation,
            "variant": args.variant,
            "rope_factor": args.rope_factor,
            "orig_ctx": args.orig_ctx,
            "rope_theta": rope_theta,
        },
        "models": {},
    }

    model_specs = [
        ("base_unfinetuned", None, "base"),
        ("hybrid_lora", args.hybrid_adapter_path, args.variant),
    ]

    for model_name, adapter_path, variant_name in model_specs:
        LOG.info("=== Evaluate model: %s ===", model_name)
        model, tokenizer, attn_used, variant_used, rope_used = load_model_and_tokenizer(
            base_model_path=args.base_model_path,
            adapter_path=adapter_path,
            merge_lora=args.merge_lora,
            attn_mode=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
            variant=variant_name,
            rope_factor=args.rope_factor,
            orig_ctx=args.orig_ctx,
            rope_theta=rope_theta,
            hybrid_split_ratio=args.hybrid_split_ratio,
            hybrid_alpha=args.hybrid_alpha,
            hybrid_p=args.hybrid_p,
            hybrid_min_freq_scale=args.hybrid_min_freq_scale,
        )
        mres = {
            "attn_used": attn_used,
            "variant_used": variant_used,
            "rope_used": rope_used,
            "tasks": {},
        }

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
                    local_data_dir=args.longbench_local_data_dir,
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
