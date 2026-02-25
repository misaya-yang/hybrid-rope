#!/usr/bin/env python3
"""
Evaluate LongBench tasks with fair-protocol controls.

Key features:
- `task_set` support (`lb6` and full `lb21`)
- official prompt/max_new_tokens parity mode
- chat template controls (`auto/on/off`)
- truncation policy controls (`middle/tail`)
- explicit task->metric mapping for all LongBench-21 tasks
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import string
import subprocess
import sys
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


LB6_TASKS = [
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "multi_news",
    "gov_report",
    "narrativeqa",
]

LB21_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

TASK_SET_MAP: Dict[str, List[str]] = {
    "lb6": LB6_TASKS,
    "lb21": LB21_TASKS,
}

# Keep task->metric mapping explicit to avoid accidental implicit defaults.
TASK_METRIC_MAP: Dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l_f1",
    "qmsum": "rouge_l_f1",
    "multi_news": "rouge_l_f1",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l_f1",
    "lsht": "classification",
    "passage_count": "count",
    "passage_retrieval_en": "retrieval_en",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}

NO_CHAT_TEMPLATE_TASKS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}

TEMPLATE_LEAK_PATTERNS = [
    r"<\|start_header_id\|>",
    r"<\|end_header_id\|>",
    r"<\|eot_id\|>",
    r"(?i)\bassistant\s*:",
    r"(?i)\buser\s*:",
    r"(?i)\bsystem\s*:",
]


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


def clip_text(text: str, max_chars: int) -> str:
    s = text or ""
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def has_template_leakage(pred: str) -> bool:
    s = pred or ""
    for pat in TEMPLATE_LEAK_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def scale_score(value_raw: float, score_scale: str) -> float:
    if score_scale == "pct":
        return float(value_raw) * 100.0
    return float(value_raw)


def attn_candidates(mode: str) -> List[Optional[str]]:
    if mode == "auto":
        return ["flash_attention_2", "sdpa", None]
    return [mode]


def _safe_float(x: object, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


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


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for tok in parse_csv(text):
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out


def rope_scaling_candidates(
    variant: str,
    factor: float,
    orig_ctx: int,
    rope_theta: float,
    longrope_short_factor: str,
    longrope_long_factor: str,
) -> List[Optional[dict]]:
    factor = float(factor)
    rope_theta = float(rope_theta)
    v = variant.strip().lower()

    if v == "yarn":
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

    if v == "pi":
        return [
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
            {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta},
        ]

    if v == "pi_soft":
        return [
            {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta},
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
        ]

    if v in {"ntk_dynamic", "ntk", "dynamic"}:
        return [
            {
                "rope_type": "dynamic",
                "factor": factor,
                "rope_theta": rope_theta,
                "original_max_position_embeddings": int(orig_ctx),
            },
            {"rope_type": "linear", "factor": factor, "rope_theta": rope_theta},
        ]

    if v == "longrope":
        short_vals = parse_float_list(longrope_short_factor)
        long_vals = parse_float_list(longrope_long_factor)
        cfg = {
            "rope_type": "longrope",
            "factor": factor,
            "rope_theta": rope_theta,
            "original_max_position_embeddings": int(orig_ctx),
        }
        if short_vals:
            cfg["short_factor"] = short_vals
        if long_vals:
            cfg["long_factor"] = long_vals
        return [cfg, {"rope_type": "dynamic", "factor": factor, "rope_theta": rope_theta}]

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
    inv_freq_cpu = inv_freq_cpu.detach().to("cpu").float().view(-1)
    patched = 0
    for name, module in model.named_modules():
        if not hasattr(module, "inv_freq"):
            continue
        if "rotary_emb" not in name and not name.endswith(".rotary_emb"):
            continue
        old = module.inv_freq
        if old.ndim != 1:
            raise RuntimeError(f"{name}.inv_freq is not 1D: shape={tuple(old.shape)}")
        if old.numel() != inv_freq_cpu.numel():
            raise RuntimeError(
                f"{name}.inv_freq size mismatch: model={old.numel()} vs provided={inv_freq_cpu.numel()}"
            )
        new = inv_freq_cpu.to(device=old.device, dtype=old.dtype)
        if isinstance(old, torch.nn.Parameter):
            module.inv_freq = torch.nn.Parameter(new, requires_grad=False)
        else:
            module.inv_freq = new
        if hasattr(module, "max_seq_len_cached"):
            module.max_seq_len_cached = 0
        patched += 1
    if patched == 0:
        raise RuntimeError("No rotary modules patched for hybrid/custom variant.")
    return patched


def load_custom_inv_freq(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"custom_inv_freq_path not found: {p}")
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().float().view(-1)
    if isinstance(obj, dict):
        for key in ("inv_freq", "custom_inv_freq", "tensor", "data"):
            val = obj.get(key)
            if isinstance(val, torch.Tensor):
                return val.detach().float().view(-1)
    raise RuntimeError(f"Unsupported custom inv_freq payload in {p}")


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
        valid = {
            "hybrid",
            "yarn",
            "pi",
            "pi_soft",
            "custom",
            "base",
            "dynamic",
            "ntk_dynamic",
            "ntk",
            "longrope",
        }
        variant = parent_name if parent_name in valid else "base"
    return variant, rope_cfg, rope_factor, orig_ctx


def load_model_and_tokenizer(
    base_model_path: str,
    adapter_path: Optional[str],
    custom_inv_freq_path: Optional[str],
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
    longrope_short_factor: str,
    longrope_long_factor: str,
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
    custom_inv = (custom_inv_freq_path or "").strip()

    if adapter_path:
        if not custom_inv:
            auto_custom = Path(adapter_path).resolve().parent / "artifacts" / "custom_inv_freq.pt"
            if auto_custom.exists():
                custom_inv = str(auto_custom)
                LOG.info("Auto-detected custom inv_freq: %s", custom_inv)
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
        if custom_inv:
            variant_used = "custom"
            rope_cfg = None
    elif variant_used == "auto":
        variant_used = "base"

    if variant_used == "custom" and not custom_inv:
        raise ValueError("variant=custom requires custom_inv_freq_path.")

    if isinstance(rope_cfg, dict) and rope_cfg.get("factor") is not None:
        rope_factor_used = _safe_float(rope_cfg.get("factor"), rope_factor_used)

    if isinstance(rope_cfg, dict):
        rope_cfg_candidates: List[Optional[dict]] = [dict(rope_cfg)]
    elif variant_used in {"yarn", "pi", "pi_soft", "dynamic", "ntk", "ntk_dynamic", "longrope"}:
        rope_cfg_candidates = rope_scaling_candidates(
            variant=variant_used,
            factor=rope_factor_used,
            orig_ctx=orig_ctx_used,
            rope_theta=rope_theta,
            longrope_short_factor=longrope_short_factor,
            longrope_long_factor=longrope_long_factor,
        )
    else:
        rope_cfg_candidates = [None]

    model = None
    used_attn = "default"
    errs: List[str] = []
    for rope_cfg_try in rope_cfg_candidates:
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

                if rope_cfg_try is not None:
                    cfg.rope_scaling = dict(rope_cfg_try)
                    if hasattr(cfg, "rope_parameters"):
                        cfg.rope_parameters = dict(rope_cfg_try)
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
                if custom_inv:
                    inv = load_custom_inv_freq(custom_inv)
                    patched = patch_hybrid_rope(model, inv)
                    rope_cfg = {"custom_inv_freq_path": custom_inv}
                    LOG.info("Patched custom inv_freq from %s into %d layers", custom_inv, patched)
                elif adapter_path and variant_used == "hybrid":
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
                else:
                    rope_cfg = dict(rope_cfg_try) if isinstance(rope_cfg_try, dict) else None
                used_attn = attn or "default"
                break
            except Exception as e:
                errs.append(f"variant={variant_used} rope={rope_cfg_try} attn={attn}: {type(e).__name__}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
        if model is not None:
            break
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


def normalize_zh_text(text: str) -> str:
    text = text.lower()
    cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc = set(string.punctuation + cn_punc)
    text = "".join(ch for ch in text if ch not in punc)
    return "".join(text.split())


def zh_tokenize(text: str) -> List[str]:
    text = normalize_zh_text(text)
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        is_cjk = "\u4e00" <= ch <= "\u9fff"
        if is_cjk:
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t]


def token_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    ref_counter: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counter[t] = ref_counter.get(t, 0) + 1

    overlap = 0
    for t, c in pred_counter.items():
        overlap += min(c, ref_counter.get(t, 0))
    if overlap == 0:
        return 0.0
    p = overlap / max(1, len(pred_tokens))
    r = overlap / max(1, len(ref_tokens))
    return float(2 * p * r / max(1e-12, p + r))


def qa_f1_score(pred: str, refs: List[str]) -> float:
    pred_tokens = normalize_text(pred).split()
    best = 0.0
    for r in refs:
        best = max(best, token_f1(pred_tokens, normalize_text(r).split()))
    return float(best)


def qa_f1_zh_score(pred: str, refs: List[str]) -> float:
    pred_tokens = zh_tokenize(pred)
    best = 0.0
    for r in refs:
        best = max(best, token_f1(pred_tokens, zh_tokenize(r)))
    return float(best)


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
        if lcs <= 0:
            continue
        p = lcs / max(1, len(pred_tokens))
        rr = lcs / max(1, len(ref_tokens))
        f1 = 2 * p * rr / max(1e-12, p + rr)
        best = max(best, f1)
    return float(best)


def rouge_l_zh(pred: str, refs: List[str]) -> float:
    pred_tokens = zh_tokenize(pred)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for r in refs:
        ref_tokens = zh_tokenize(r)
        if not ref_tokens:
            continue
        lcs = lcs_len(pred_tokens, ref_tokens)
        if lcs <= 0:
            continue
        p = lcs / max(1, len(pred_tokens))
        rr = lcs / max(1, len(ref_tokens))
        f1 = 2 * p * rr / max(1e-12, p + rr)
        best = max(best, f1)
    return float(best)


def classification_score(pred: str, refs: List[str], all_classes: List[str]) -> float:
    if not refs:
        return 0.0
    gt = refs[0]
    if not all_classes:
        return 1.0 if gt in pred else 0.0
    matches = [c for c in all_classes if isinstance(c, str) and c and c in pred]
    clean: List[str] = []
    for m in matches:
        if m in gt and m != gt:
            continue
        clean.append(m)
    if gt in clean and clean:
        return float(1.0 / len(clean))
    return 0.0


def count_score(pred: str, refs: List[str]) -> float:
    if not refs:
        return 0.0
    gt = str(refs[0])
    numbers = re.findall(r"\d+", pred)
    if not numbers:
        return 0.0
    hit = 0
    for n in numbers:
        if n == gt:
            hit += 1
    return float(hit / len(numbers))


def retrieval_score(pred: str, refs: List[str], zh: bool = False) -> float:
    if not refs:
        return 0.0
    pattern = r"段落(\d+)" if zh else r"Paragraph\s*(\d+)"
    match = re.findall(pattern, refs[0])
    if not match:
        match = re.findall(r"\d+", refs[0])
    if not match:
        return 0.0
    target = str(match[0])
    nums = re.findall(r"\d+", pred)
    if not nums:
        return 0.0
    hit = sum(1 for n in nums if str(n) == target)
    return float(hit / len(nums))


def code_sim_score(pred: str, refs: List[str]) -> float:
    if not refs:
        return 0.0
    candidate = ""
    for line in pred.lstrip("\n").split("\n"):
        if "`" in line or line.strip().startswith("#") or "//" in line:
            continue
        candidate = line
        break
    if not candidate:
        candidate = pred.strip()
    import difflib

    return float(difflib.SequenceMatcher(None, candidate, refs[0]).ratio())


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


def enrich_sample_fields(sample: Dict) -> Dict:
    out = dict(sample)
    if "context" not in out or not out.get("context"):
        for k in ("article", "document", "documents", "passage", "text"):
            if k in out and out.get(k) is not None:
                out["context"] = as_text(out.get(k))
                break
    if "input" not in out or not out.get("input"):
        for k in ("question", "query", "instruction"):
            if k in out and out.get(k) is not None:
                out["input"] = as_text(out.get(k))
                break
    return out


def extract_refs_and_classes(sample: Dict) -> Tuple[List[str], List[str], str]:
    answer_keys = ["answers", "answer", "output", "summary", "target", "label"]
    refs: List[str] = []
    used_key = ""
    for k in answer_keys:
        if k in sample and sample[k] is not None:
            v = sample[k]
            if isinstance(v, list):
                refs = [as_text(x) for x in v if as_text(x)]
            else:
                s = as_text(v)
                refs = [s] if s else []
            if refs:
                used_key = k
                break
    all_classes = sample.get("all_classes")
    classes: List[str] = []
    if isinstance(all_classes, list):
        classes = [str(x) for x in all_classes if isinstance(x, (str, int, float))]
    return refs, classes, used_key


def extract_legacy_prompt(task: str, sample: Dict) -> str:
    context = as_text(sample.get("context", ""))
    question = as_text(sample.get("input", ""))
    if task == "gov_report":
        return (
            "You are given a long report.\n"
            "Write a concise, faithful summary.\n\n"
            f"Report:\n{context}\n\nSummary:"
        )
    return (
        "Read the context and answer the question.\n"
        "Answer as briefly and accurately as possible.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def build_prompt(
    task: str,
    sample: Dict,
    prompt_source: str,
    official_prompt_map: Dict[str, str],
) -> str:
    if prompt_source == "legacy":
        return extract_legacy_prompt(task, sample)

    tmpl = official_prompt_map.get(task)
    if not tmpl:
        return extract_legacy_prompt(task, sample)
    try:
        return str(tmpl).format_map(_SafeFormatDict(sample))
    except Exception:
        return extract_legacy_prompt(task, sample)


def truncate_prompt_with_meta(
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    mode: str,
) -> Tuple[str, Dict[str, object]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    n_total = len(ids)
    meta: Dict[str, object] = {
        "input_tokens_before_trunc": int(n_total),
        "input_tokens_after_trunc": int(min(n_total, max_tokens)),
        "truncated": bool(n_total > max_tokens),
        "truncate_mode": mode,
        "truncation_keep_head_tokens": int(0),
        "truncation_keep_tail_tokens": int(min(n_total, max_tokens)),
        "truncation_dropped_tokens": int(max(0, n_total - max_tokens)),
        "truncation_dropped_span_start": None,
        "truncation_dropped_span_end": None,
    }
    if n_total <= max_tokens:
        return prompt, meta

    if mode == "middle":
        first = max_tokens // 2
        second = max_tokens - first
        kept = ids[:first] + ids[-second:]
        meta["truncation_keep_head_tokens"] = int(first)
        meta["truncation_keep_tail_tokens"] = int(second)
        meta["truncation_dropped_span_start"] = int(first)
        meta["truncation_dropped_span_end"] = int(max(first, n_total - second))
    else:
        kept = ids[-max_tokens:]
        meta["truncation_keep_head_tokens"] = int(0)
        meta["truncation_keep_tail_tokens"] = int(max_tokens)
        meta["truncation_dropped_span_start"] = int(0)
        meta["truncation_dropped_span_end"] = int(max(0, n_total - max_tokens))

    return tokenizer.decode(kept, skip_special_tokens=True), meta


def truncate_prompt(tokenizer: AutoTokenizer, prompt: str, max_tokens: int, mode: str) -> str:
    return truncate_prompt_with_meta(tokenizer=tokenizer, prompt=prompt, max_tokens=max_tokens, mode=mode)[0]


def maybe_apply_chat_template(
    tokenizer: AutoTokenizer,
    task: str,
    prompt: str,
    chat_template: str,
) -> str:
    if chat_template == "off":
        return prompt

    if chat_template == "auto" and task in NO_CHAT_TEMPLATE_TASKS:
        return prompt

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        LOG.warning("Chat template failed for task=%s: %s", task, e)

    # Fallback for chat-template-enabled path.
    return f"User: {prompt}\nAssistant:"


def pick_max_new_tokens(
    task: str,
    metric_name: str,
    policy: str,
    official_maxlen_map: Dict[str, int],
    max_new_tokens_qa: int,
    max_new_tokens_sum: int,
) -> int:
    if policy == "official":
        val = official_maxlen_map.get(task)
        if isinstance(val, int) and val > 0:
            return int(val)
    if metric_name in {"qa_f1", "qa_f1_zh", "classification", "count", "retrieval_en", "retrieval_zh", "code_sim"}:
        return int(max_new_tokens_qa)
    return int(max_new_tokens_sum)


def post_process_prediction(task: str, pred: str) -> str:
    p = pred or ""
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        p = p.lstrip("\n").split("\n")[0]
    return p.strip()


def score_prediction(task: str, metric_name: str, pred: str, refs: List[str], all_classes: List[str]) -> float:
    if metric_name == "qa_f1":
        return qa_f1_score(pred, refs)
    if metric_name == "qa_f1_zh":
        return qa_f1_zh_score(pred, refs)
    if metric_name == "rouge_l_f1":
        return rouge_l_f1(pred, refs)
    if metric_name == "rouge_l_zh":
        return rouge_l_zh(pred, refs)
    if metric_name == "classification":
        return classification_score(pred, refs, all_classes=all_classes)
    if metric_name == "count":
        return count_score(pred, refs)
    if metric_name == "retrieval_en":
        return retrieval_score(pred, refs, zh=False)
    if metric_name == "retrieval_zh":
        return retrieval_score(pred, refs, zh=True)
    if metric_name == "code_sim":
        return code_sim_score(pred, refs)
    raise ValueError(f"Unsupported metric_name={metric_name} for task={task}")


def load_task_dataset(task: str, local_data_dir: Optional[str] = None):
    # Prefer local jsonl files when available (fully offline and deterministic).
    if local_data_dir:
        data_dir = Path(local_data_dir)
        if data_dir.exists():
            p = data_dir / f"{task}.jsonl"
            if p.exists():
                return load_dataset("json", data_files=str(p), split="train")

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
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, ids["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot batch-generate with padding.")
    tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def generate_text_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    if not prompts:
        return []
    ensure_pad_token(tokenizer)

    # Decoder-only generation is safest with left padding so "last token" is real prompt, not PAD.
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        device = next(model.parameters()).device
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)
        prompt_len = int(enc["input_ids"].shape[1])
        out = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[:, prompt_len:]
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [p.strip() for p in preds]
    finally:
        tokenizer.padding_side = old_padding_side


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    task: str,
    max_samples: int,
    max_input_tokens: int,
    max_new_tokens_qa: int,
    max_new_tokens_sum: int,
    max_new_tokens_policy: str,
    prompt_source: str,
    chat_template: str,
    truncate_mode: str,
    score_scale: str,
    seed: int,
    local_data_dir: Optional[str],
    official_prompt_map: Dict[str, str],
    official_maxlen_map: Dict[str, int],
    indices: Optional[List[int]] = None,
    batch_size: int = 1,
    save_per_sample_traces: bool = True,
    trace_output_max_chars: int = 0,
) -> Dict:
    ds = load_task_dataset(task, local_data_dir=local_data_dir)
    if indices is None:
        n = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]
    else:
        idxs = [int(i) for i in indices if 0 <= int(i) < len(ds)]
        if idxs:
            n = len(idxs)
        else:
            n = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
            idxs = list(range(len(ds)))
            random.Random(seed).shuffle(idxs)
            idxs = idxs[:n]

    metric_name = TASK_METRIC_MAP.get(task)
    if metric_name is None:
        raise ValueError(
            f"Task '{task}' is missing from TASK_METRIC_MAP. "
            "Please add explicit metric mapping to avoid silent metric drift."
        )

    raw_scores: List[float] = []
    records: List[Dict] = []
    traces: List[Dict[str, object]] = []
    n_total = len(idxs)
    empty_output_count = 0
    template_leakage_count = 0
    parse_fail_count = 0
    truncation_at_question_count = 0
    generation_error_count = 0
    missing_reference_count = 0

    bs = max(1, int(batch_size))

    def chunks(seq: List[int], size: int):
        for start in range(0, len(seq), size):
            yield seq[start : start + size]

    processed = 0
    for chunk in chunks([int(x) for x in idxs], bs):
        entries: List[Dict[str, object]] = []
        for i in chunk:
            sample = enrich_sample_fields(dict(ds[int(i)]))
            refs, all_classes, ref_key = extract_refs_and_classes(sample)
            prompt = build_prompt(
                task=task,
                sample=sample,
                prompt_source=prompt_source,
                official_prompt_map=official_prompt_map,
            )
            prompt_chat = maybe_apply_chat_template(tokenizer=tokenizer, task=task, prompt=prompt, chat_template=chat_template)
            prompt_trunc, trunc_meta = truncate_prompt_with_meta(
                tokenizer=tokenizer,
                prompt=prompt_chat,
                max_tokens=max_input_tokens,
                mode=truncate_mode,
            )
            question_text = as_text(sample.get("input", "")).strip()
            q_before = bool(question_text and (question_text in prompt_chat))
            q_after = bool(question_text and (question_text in prompt_trunc))
            q_truncated = bool(trunc_meta.get("truncated")) and q_before and (not q_after)
            if q_truncated:
                truncation_at_question_count += 1

            max_new = pick_max_new_tokens(
                task=task,
                metric_name=metric_name,
                policy=max_new_tokens_policy,
                official_maxlen_map=official_maxlen_map,
                max_new_tokens_qa=max_new_tokens_qa,
                max_new_tokens_sum=max_new_tokens_sum,
            )

            entry: Dict[str, object] = {
                "index": int(i),
                "sample": sample,
                "refs": refs,
                "all_classes": all_classes,
                "ref_key": ref_key,
                "prompt_trunc": prompt_trunc,
                "trunc_meta": trunc_meta,
                "q_before": q_before,
                "q_after": q_after,
                "q_truncated": q_truncated,
                "max_new": int(max_new),
                "raw_pred": "",
                "prompt_for_eval": prompt_trunc,
                "fallback_used": False,
                "evaluator_status": "ok",
                "failure_type": "none",
                "error_message": "",
            }

            if not refs:
                missing_reference_count += 1
                parse_fail_count += 1
                entry["evaluator_status"] = "parse_fail"
                entry["failure_type"] = "missing_reference"
            entries.append(entry)

        # Batch generate for the "ok" entries. If batch fails, fall back to per-sample generation.
        gen_positions = [p for p, e in enumerate(entries) if e.get("evaluator_status") == "ok"]
        gen_prompts = [str(entries[p]["prompt_for_eval"]) for p in gen_positions]
        gen_preds: List[str] = []
        gen_failed = False
        if gen_prompts:
            try:
                gen_preds = generate_text_batch(model, tokenizer, gen_prompts, max_new_tokens=int(entries[gen_positions[0]]["max_new"]))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                gen_failed = True
            except Exception:
                gen_failed = True

        if gen_prompts and not gen_failed and len(gen_preds) == len(gen_prompts):
            for p, raw_pred in zip(gen_positions, gen_preds):
                entries[p]["raw_pred"] = raw_pred
        else:
            for p in gen_positions:
                prompt_for_eval = str(entries[p]["prompt_for_eval"])
                max_new = int(entries[p]["max_new"])
                try:
                    entries[p]["raw_pred"] = generate_text(model, tokenizer, prompt_for_eval, max_new_tokens=max_new)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        prompt_for_eval = truncate_prompt(
                            tokenizer=tokenizer,
                            prompt=prompt_for_eval,
                            max_tokens=max(512, max_input_tokens // 2),
                            mode=truncate_mode,
                        )
                        entries[p]["prompt_for_eval"] = prompt_for_eval
                        entries[p]["fallback_used"] = True
                        try:
                            entries[p]["raw_pred"] = generate_text(model, tokenizer, prompt_for_eval, max_new_tokens=max_new)
                        except Exception as e2:
                            generation_error_count += 1
                            entries[p]["evaluator_status"] = "generation_error"
                            entries[p]["failure_type"] = "generation_error"
                            entries[p]["error_message"] = f"{type(e2).__name__}: {e2}"
                    else:
                        generation_error_count += 1
                        entries[p]["evaluator_status"] = "generation_error"
                        entries[p]["failure_type"] = "generation_error"
                        entries[p]["error_message"] = f"{type(e).__name__}: {e}"
                except Exception as e:
                    generation_error_count += 1
                    entries[p]["evaluator_status"] = "generation_error"
                    entries[p]["failure_type"] = "generation_error"
                    entries[p]["error_message"] = f"{type(e).__name__}: {e}"

        for entry in entries:
            i = int(entry["index"])
            refs = entry["refs"]  # type: ignore[assignment]
            all_classes = entry["all_classes"]  # type: ignore[assignment]
            ref_key = str(entry["ref_key"])
            raw_pred = str(entry.get("raw_pred", ""))
            pred = post_process_prediction(task=task, pred=raw_pred)

            evaluator_status = str(entry.get("evaluator_status", "ok"))
            failure_type = str(entry.get("failure_type", "none"))
            error_message = str(entry.get("error_message", ""))
            fallback_used = bool(entry.get("fallback_used", False))
            prompt_for_eval = str(entry.get("prompt_for_eval", ""))
            trunc_meta = entry.get("trunc_meta") or {}
            q_before = bool(entry.get("q_before", False))
            q_after = bool(entry.get("q_after", False))
            q_truncated = bool(entry.get("q_truncated", False))
            max_new = int(entry.get("max_new", 0))

            template_leak = has_template_leakage(pred)
            if template_leak:
                template_leakage_count += 1

            score_raw_val: Optional[float] = None
            score_pct_val: Optional[float] = None
            score_val: Optional[float] = None

            if evaluator_status == "ok":
                if pred.strip() == "":
                    empty_output_count += 1
                    failure_type = "empty_output"
                elif template_leak:
                    failure_type = "template_leakage"

                try:
                    sc_raw = score_prediction(task=task, metric_name=metric_name, pred=pred, refs=refs, all_classes=all_classes)
                    raw_scores.append(sc_raw)
                    sc = scale_score(sc_raw, score_scale=score_scale)
                    score_raw_val = float(sc_raw)
                    score_pct_val = float(sc_raw) * 100.0
                    score_val = float(sc)
                except Exception as e:
                    parse_fail_count += 1
                    evaluator_status = "parse_fail"
                    failure_type = "parse_fail"
                    error_message = f"{type(e).__name__}: {e}"

            if len(records) < 3:
                sample = entry.get("sample") or {}
                records.append(
                    {
                        "index": int(i),
                        "prediction": clip_text(pred, 500),
                        "reference_0": clip_text(refs[0], 500) if refs else "",
                        "answer_key": ref_key,
                        "score_raw": score_raw_val,
                        "score_pct": score_pct_val,
                        "score": score_val,
                        "context_preview": as_text(sample.get("context", ""))[:300],
                        "input_preview": as_text(sample.get("input", ""))[:200],
                        "failure_type": failure_type,
                        "evaluator_status": evaluator_status,
                    }
                )

            prompt_eval_tokens = tokenizer.encode(prompt_for_eval, add_special_tokens=False)
            trace_obj: Dict[str, object] = {
                "index": int(i),
                "task": task,
                "answer_key": ref_key,
                "metric": metric_name,
                "prompt_sha1": sha1_text(prompt_for_eval),
                "input_tokens_before_trunc": int(getattr(trunc_meta, "get", lambda *_: 0)("input_tokens_before_trunc", 0)),
                "input_tokens_after_trunc": int(len(prompt_eval_tokens)),
                "truncated": bool(getattr(trunc_meta, "get", lambda *_: False)("truncated", False)),
                "truncate_mode": str(getattr(trunc_meta, "get", lambda *_: truncate_mode)("truncate_mode", truncate_mode)),
                "truncation_keep_head_tokens": int(getattr(trunc_meta, "get", lambda *_: 0)("truncation_keep_head_tokens", 0)),
                "truncation_keep_tail_tokens": int(getattr(trunc_meta, "get", lambda *_: 0)("truncation_keep_tail_tokens", 0)),
                "truncation_dropped_tokens": int(getattr(trunc_meta, "get", lambda *_: 0)("truncation_dropped_tokens", 0)),
                "truncation_dropped_span_start": getattr(trunc_meta, "get", lambda *_: None)("truncation_dropped_span_start"),
                "truncation_dropped_span_end": getattr(trunc_meta, "get", lambda *_: None)("truncation_dropped_span_end"),
                "question_present_before_trunc": bool(q_before),
                "question_present_after_trunc": bool(q_after),
                "question_truncated": bool(q_truncated),
                "max_new_tokens": int(max_new),
                "fallback_short_prompt_used": bool(fallback_used),
                "raw_output": clip_text(raw_pred, trace_output_max_chars),
                "prediction": clip_text(pred, trace_output_max_chars),
                "score_raw": score_raw_val,
                "score_pct": score_pct_val,
                "score": score_val,
                "evaluator_status": evaluator_status,
                "failure_type": failure_type,
                "error_message": error_message,
                "template_leakage": bool(template_leak),
            }
            if save_per_sample_traces:
                traces.append(trace_obj)

            processed += 1
            if processed % 10 == 0 and raw_scores:
                LOG.info(
                    "task=%s progress=%d/%d running_%s_raw=%.4f",
                    task,
                    processed,
                    n,
                    metric_name,
                    float(np.mean(raw_scores)),
                )

    score_raw = float(np.mean(raw_scores)) if raw_scores else None
    score_pct = (score_raw * 100.0) if score_raw is not None else None
    display_score = score_pct if score_scale == "pct" else score_raw
    per_sample_raw = [float(x) for x in raw_scores]
    per_sample_pct = [float(x) * 100.0 for x in raw_scores]
    per_sample_display = per_sample_pct if score_scale == "pct" else per_sample_raw
    denom = max(1, n_total)
    audit = {
        "num_selected": int(n_total),
        "num_scored": int(len(raw_scores)),
        "num_missing_reference": int(missing_reference_count),
        "num_generation_error": int(generation_error_count),
        "num_parse_fail": int(parse_fail_count),
        "num_empty_output": int(empty_output_count),
        "num_template_leakage": int(template_leakage_count),
        "num_truncation_at_question": int(truncation_at_question_count),
        "empty_output_rate": float(empty_output_count / denom),
        "template_leakage_rate": float(template_leakage_count / denom),
        "parse_fail_rate": float(parse_fail_count / denom),
        "truncation_at_question_rate": float(truncation_at_question_count / denom),
    }

    return {
        "task": task,
        "metric": metric_name,
        "metric_unit": "0-1",
        "score_scale": score_scale,
        "prompt_source": prompt_source,
        "chat_template": chat_template,
        "truncate_mode": truncate_mode,
        "max_new_tokens_policy": max_new_tokens_policy,
        "indices": [int(i) for i in idxs],
        "num_scored": len(raw_scores),
        "score": display_score,
        "score_raw": score_raw,
        "score_pct": score_pct,
        "per_sample_scores": per_sample_display,
        "per_sample_scores_raw": per_sample_raw,
        "per_sample_scores_pct": per_sample_pct,
        "per_sample_traces": traces,
        "audit": audit,
        "examples": records,
    }


def load_official_longbench_config(prompt_path: str, maxlen_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    p_prompt = Path(prompt_path)
    p_maxlen = Path(maxlen_path)
    if not p_prompt.exists() or not p_maxlen.exists():
        raise FileNotFoundError(
            "Official LongBench config missing. "
            f"prompt={p_prompt} exists={p_prompt.exists()} maxlen={p_maxlen} exists={p_maxlen.exists()}"
        )

    prompt_map = json.loads(p_prompt.read_text(encoding="utf-8"))
    maxlen_map_raw = json.loads(p_maxlen.read_text(encoding="utf-8"))
    maxlen_map = {str(k): int(v) for k, v in maxlen_map_raw.items()}
    if not isinstance(prompt_map, dict):
        raise RuntimeError(f"Invalid prompt config format: {p_prompt}")
    return {str(k): str(v) for k, v in prompt_map.items()}, maxlen_map


def resolve_tasks(task_set: str, tasks: str) -> List[str]:
    explicit = parse_csv(tasks)
    if explicit:
        return explicit
    if task_set not in TASK_SET_MAP:
        raise ValueError(f"Unsupported task_set={task_set}, expected one of {sorted(TASK_SET_MAP.keys())}")
    return list(TASK_SET_MAP[task_set])


def try_run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return (p.stdout or "").strip()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def write_repro_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    tasks: List[str],
    rope_theta: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    head = try_run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = try_run(["git", "status", "--short"], cwd=repo_root)

    code_hash_lines = [
        f"repo_root={repo_root}",
        f"git_head={head}",
        "git_status_short:",
        status if status else "<clean>",
    ]
    (out_dir / "code_hash.txt").write_text("\n".join(code_hash_lines) + "\n", encoding="utf-8")

    env_lines = [
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"torch_version={getattr(torch, '__version__', 'unknown')}",
        f"transformers_version={try_run([sys.executable, '-c', 'import transformers as t; print(t.__version__)'])}",
        "pip_freeze:",
        try_run([sys.executable, "-m", "pip", "freeze"]),
    ]
    nvsmi = try_run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if nvsmi and not nvsmi.startswith("<error:"):
        env_lines.extend(["nvidia_smi:", nvsmi])
    (out_dir / "env_freeze.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    baseline_gold = [
        "baseline_gold:",
        f"  base_model_path: \"{args.base_model_path}\"",
        f"  adapter_path: \"{args.adapter_path or args.hybrid_adapter_path}\"",
        f"  task_set: \"{args.task_set}\"",
        f"  tasks: {json.dumps(tasks, ensure_ascii=False)}",
        f"  max_samples_per_task: {int(args.max_samples_per_task)}",
        f"  max_input_tokens: {int(args.max_input_tokens)}",
        f"  batch_size: {int(args.batch_size)}",
        f"  max_new_tokens_policy: \"{args.max_new_tokens_policy}\"",
        f"  prompt_source: \"{args.prompt_source}\"",
        f"  chat_template: \"{args.chat_template}\"",
        f"  truncate_mode: \"{args.truncate_mode}\"",
        f"  score_scale: \"{args.score_scale}\"",
        f"  strict_parity_check: {bool(args.strict_parity_check)}",
        f"  attn_implementation: \"{args.attn_implementation}\"",
        f"  rope_theta: {float(rope_theta)}",
        f"  official_prompt_path: \"{args.official_prompt_path}\"",
        f"  official_maxlen_path: \"{args.official_maxlen_path}\"",
        f"  output_json: \"{args.output_json}\"",
    ]
    (out_dir / "baseline_gold.yaml").write_text("\n".join(baseline_gold) + "\n", encoding="utf-8")


def enforce_strict_parity(args: argparse.Namespace) -> None:
    errors: List[str] = []
    if args.prompt_source != "official":
        errors.append("prompt_source must be 'official'")
    if args.truncate_mode != "middle":
        errors.append("truncate_mode must be 'middle'")
    if args.max_new_tokens_policy != "official":
        errors.append("max_new_tokens_policy must be 'official'")
    if args.chat_template == "off":
        errors.append("chat_template must be 'auto' or 'on' (not 'off')")
    if errors:
        raise RuntimeError("strict_parity_check failed: " + "; ".join(errors))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LongBench with official-parity controls.")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--hybrid_adapter_path",
        type=str,
        default="",
        help="Backward-compatible alias of --adapter_path.",
    )
    ap.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="LoRA adapter path to evaluate against base model.",
    )
    ap.add_argument(
        "--model_alias",
        type=str,
        default="hybrid_lora",
        help="Model key name used in output for adapter-backed model.",
    )
    ap.add_argument(
        "--skip_base_unfinetuned",
        action="store_true",
        help="Skip evaluating base model and only evaluate adapter model.",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "base", "hybrid", "yarn", "pi", "pi_soft", "custom", "dynamic", "ntk", "ntk_dynamic", "longrope"],
        help="RoPE variant used during adapter training. auto=try infer from adapter summary.",
    )
    ap.add_argument(
        "--custom_inv_freq_path",
        type=str,
        default="",
        help=(
            "Optional path to a saved inv_freq tensor (.pt). "
            "When provided, this tensor is patched into rotary modules and overrides variant mapping."
        ),
    )
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_theta", type=float, default=0.0, help="<=0 means infer from base config.")
    ap.add_argument("--longrope_short_factor", type=str, default="")
    ap.add_argument("--longrope_long_factor", type=str, default="")
    ap.add_argument("--hybrid_split_ratio", type=float, default=0.5)
    ap.add_argument("--hybrid_alpha", type=float, default=0.2)
    ap.add_argument("--hybrid_p", type=float, default=3.9)
    ap.add_argument("--hybrid_min_freq_scale", type=float, default=4.0)

    ap.add_argument(
        "--task_set",
        type=str,
        default="lb6",
        choices=["lb6", "lb21"],
        help="Predefined task set. Use --tasks for explicit override.",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Explicit comma-separated task list. Overrides --task_set when non-empty.",
    )

    ap.add_argument("--max_samples_per_task", type=int, default=100)
    ap.add_argument("--max_input_tokens", type=int, default=16384)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for greedy generation. Increase on large-VRAM GPUs to improve throughput.",
    )
    ap.add_argument("--max_new_tokens_qa", type=int, default=64)
    ap.add_argument("--max_new_tokens_sum", type=int, default=256)
    ap.add_argument(
        "--max_new_tokens_policy",
        type=str,
        default="official",
        choices=["official", "manual"],
        help="official: use dataset2maxlen.json; manual: use qa/sum fallbacks.",
    )
    ap.add_argument(
        "--prompt_source",
        type=str,
        default="official",
        choices=["official", "legacy"],
        help="Prompt source for LongBench tasks.",
    )
    ap.add_argument(
        "--chat_template",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Chat template usage policy.",
    )
    ap.add_argument(
        "--truncate_mode",
        type=str,
        default="middle",
        choices=["tail", "middle"],
        help="Prompt truncation policy when input exceeds max_input_tokens.",
    )

    ap.add_argument(
        "--score_scale",
        type=str,
        default="raw",
        choices=["raw", "pct"],
        help="How to populate task-level 'score' fields: raw in [0,1] or percentage in [0,100].",
    )
    ap.add_argument(
        "--longbench_local_data_dir",
        type=str,
        default=os.environ.get("LONGBENCH_LOCAL_DATA_DIR", "/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"),
        help="Directory containing local LongBench jsonl files (e.g. qasper.jsonl).",
    )
    ap.add_argument(
        "--official_prompt_path",
        type=str,
        default=str((Path(__file__).resolve().parent / "longbench_official_config" / "dataset2prompt.json")),
    )
    ap.add_argument(
        "--official_maxlen_path",
        type=str,
        default=str((Path(__file__).resolve().parent / "longbench_official_config" / "dataset2maxlen.json")),
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_per_sample_traces",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to save full per-sample trace objects (1=yes, 0=no).",
    )
    ap.add_argument(
        "--trace_output_max_chars",
        type=int,
        default=0,
        help="Max chars stored for raw_output/prediction in per-sample traces.",
    )
    ap.add_argument(
        "--repro_manifest_dir",
        type=str,
        default="",
        help="Optional directory to emit baseline_gold.yaml / env_freeze.txt / code_hash.txt.",
    )
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument(
        "--manifest_json",
        type=str,
        default="",
        help="Optional path for paired-eval manifest (task -> fixed indices).",
    )
    ap.add_argument(
        "--strict_parity_check",
        action="store_true",
        help="Fail fast unless official parity knobs are active.",
    )
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    enforce_offline_mode()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(out_path.parent / "eval_longbench.log")

    if args.strict_parity_check:
        enforce_strict_parity(args)

    adapter_path = args.adapter_path or args.hybrid_adapter_path
    tasks = resolve_tasks(task_set=args.task_set, tasks=args.tasks)

    official_prompt_map, official_maxlen_map = load_official_longbench_config(
        prompt_path=args.official_prompt_path,
        maxlen_path=args.official_maxlen_path,
    )

    LOG.info("Tasks=%s", tasks)
    LOG.info("Prompt source=%s chat_template=%s truncate=%s max_new_tokens_policy=%s", args.prompt_source, args.chat_template, args.truncate_mode, args.max_new_tokens_policy)

    rope_theta = float(args.rope_theta) if args.rope_theta > 0 else infer_rope_theta(args.base_model_path, args.trust_remote_code)
    LOG.info("Resolved rope_theta=%.1f", rope_theta)

    results: Dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
            "base_model_path": args.base_model_path,
            "adapter_path": adapter_path,
            "custom_inv_freq_path": args.custom_inv_freq_path,
            "task_set": args.task_set,
            "tasks": tasks,
            "max_samples_per_task": args.max_samples_per_task,
            "max_input_tokens": args.max_input_tokens,
            "batch_size": int(args.batch_size),
            "max_new_tokens_policy": args.max_new_tokens_policy,
            "prompt_source": args.prompt_source,
            "chat_template": args.chat_template,
            "truncate_mode": args.truncate_mode,
            "score_scale": args.score_scale,
            "save_per_sample_traces": bool(args.save_per_sample_traces),
            "trace_output_max_chars": int(args.trace_output_max_chars),
            "score_unit_raw": "0-1",
            "score_unit_pct": "0-100",
            "attn_implementation": args.attn_implementation,
            "variant": args.variant,
            "rope_factor": args.rope_factor,
            "orig_ctx": args.orig_ctx,
            "rope_theta": rope_theta,
            "strict_parity_check": bool(args.strict_parity_check),
            "official_prompt_path": args.official_prompt_path,
            "official_maxlen_path": args.official_maxlen_path,
            "repro_manifest_dir": args.repro_manifest_dir,
        },
        "models": {},
    }
    if args.repro_manifest_dir:
        write_repro_manifest(
            out_dir=Path(args.repro_manifest_dir),
            args=args,
            tasks=tasks,
            rope_theta=rope_theta,
        )

    manifest_path = Path(args.manifest_json) if args.manifest_json else None
    manifest_obj: Dict[str, object] = {"meta": {}, "tasks": {}}
    if manifest_path and manifest_path.exists():
        try:
            manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_obj.get("tasks"), dict):
                manifest_obj["tasks"] = {}
        except Exception as e:
            LOG.warning("Failed to read manifest %s: %s", manifest_path, e)
            manifest_obj = {"meta": {}, "tasks": {}}
    task_index_map: Dict[str, List[int]] = {}
    for t, v in (manifest_obj.get("tasks") or {}).items():
        if isinstance(v, list):
            task_index_map[str(t)] = [int(i) for i in v]

    model_specs: List[Tuple[str, Optional[str], str]] = []
    if not args.skip_base_unfinetuned:
        model_specs.append(("base_unfinetuned", None, "base"))
    if adapter_path:
        model_specs.append((args.model_alias, adapter_path, args.variant))
    if not model_specs:
        raise RuntimeError("No model selected for evaluation. Provide --adapter_path or disable --skip_base_unfinetuned.")

    for model_name, model_adapter_path, variant_name in model_specs:
        LOG.info("=== Evaluate model: %s ===", model_name)
        model, tokenizer, attn_used, variant_used, rope_used = load_model_and_tokenizer(
            base_model_path=args.base_model_path,
            adapter_path=model_adapter_path,
            custom_inv_freq_path=args.custom_inv_freq_path if model_adapter_path else "",
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
            longrope_short_factor=args.longrope_short_factor,
            longrope_long_factor=args.longrope_long_factor,
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
                    max_new_tokens_policy=args.max_new_tokens_policy,
                    prompt_source=args.prompt_source,
                    chat_template=args.chat_template,
                    truncate_mode=args.truncate_mode,
                    score_scale=args.score_scale,
                    seed=args.seed,
                    local_data_dir=args.longbench_local_data_dir,
                    official_prompt_map=official_prompt_map,
                    official_maxlen_map=official_maxlen_map,
                    indices=task_index_map.get(task),
                    batch_size=int(args.batch_size),
                    save_per_sample_traces=bool(args.save_per_sample_traces),
                    trace_output_max_chars=int(args.trace_output_max_chars),
                )
                if task not in task_index_map:
                    task_index_map[task] = [int(i) for i in tres.get("indices", [])]
            except Exception as e:
                tres = {
                    "task": task,
                    "error": f"{type(e).__name__}: {e}",
                    "score": None,
                    "score_raw": None,
                    "score_pct": None,
                }
            mres["tasks"][task] = tres

        results["models"][model_name] = mres

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Flattened comparison table for quick reading when both models exist.
    compare: Dict[str, Dict[str, Optional[float]]] = {}
    model_alias = args.model_alias
    if "base_unfinetuned" in results["models"] and model_alias in results["models"]:
        for task in tasks:
            base_score = results["models"].get("base_unfinetuned", {}).get("tasks", {}).get(task, {}).get("score")
            alt_score = results["models"].get(model_alias, {}).get("tasks", {}).get(task, {}).get("score")
            compare[task] = {
                "base_unfinetuned": base_score,
                model_alias: alt_score,
                "delta_alt_minus_base": (alt_score - base_score) if (base_score is not None and alt_score is not None) else None,
            }
        results["comparison"] = compare

    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
                "seed": args.seed,
                "tasks": tasks,
                "task_set": args.task_set,
                "max_samples_per_task": args.max_samples_per_task,
                "prompt_source": args.prompt_source,
                "chat_template": args.chat_template,
                "truncate_mode": args.truncate_mode,
                "max_new_tokens_policy": args.max_new_tokens_policy,
            },
            "tasks": {t: task_index_map.get(t, []) for t in tasks},
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        results["meta"]["manifest_json"] = str(manifest_path)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
