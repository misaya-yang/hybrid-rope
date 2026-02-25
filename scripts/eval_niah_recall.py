#!/usr/bin/env python3
"""
Needle-In-A-Haystack recall evaluation (single-needle + multi-needle) for long contexts.

Highlights:
- Supports base model or base + LoRA adapter.
- Supports single needle (`--needles_per_prompt 1`) and multi needle (`>1`) in one unified script.
- Builds a depth x context-length accuracy matrix and saves a red-green heatmap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


LOG = logging.getLogger("eval_niah_recall")


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json_silent(path: Path) -> Optional[Dict]:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_inv_metadata(
    base_only: bool,
    adapter_path: str,
    custom_inv_freq_path: str,
    rope_used: Optional[dict],
) -> Tuple[str, str]:
    if base_only:
        return "", ""

    inv_path = ""
    inv_sha256 = ""
    candidates: List[Path] = []

    custom_inv = (custom_inv_freq_path or "").strip()
    if custom_inv:
        candidates.append(Path(custom_inv).expanduser())
    if isinstance(rope_used, dict):
        rope_custom = str(rope_used.get("custom_inv_freq_path", "") or "").strip()
        if rope_custom:
            candidates.append(Path(rope_custom).expanduser())

    ap = Path(adapter_path).expanduser().resolve()
    candidates.extend(
        [
            ap / "artifacts" / "custom_inv_freq.pt",
            ap / "custom_inv_freq.pt",
            ap.parent / "artifacts" / "custom_inv_freq.pt",
        ]
    )
    for summary_path in [ap / "artifacts" / "summary.json", ap.parent / "artifacts" / "summary.json"]:
        summary = read_json_silent(summary_path)
        if isinstance(summary, dict):
            inv_sha256 = str((summary.get("rope") or {}).get("inv_sha256", "") or "")
            if inv_sha256:
                break

    seen: set[str] = set()
    for p in candidates:
        key = p.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            inv_path = p.resolve().as_posix()
            inv_sha256 = sha256_file(p.resolve())
            break
    return inv_sha256, inv_path


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate NIAH single/multi needle recall and draw heatmap.")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Optional LoRA adapter path. Ignored when --base_only is set.",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "base", "hybrid", "yarn", "pi", "pi_soft", "custom"],
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
    ap.add_argument("--hybrid_split_ratio", type=float, default=0.5)
    ap.add_argument("--hybrid_alpha", type=float, default=0.2)
    ap.add_argument("--hybrid_p", type=float, default=3.9)
    ap.add_argument("--hybrid_min_freq_scale", type=float, default=4.0)
    ap.add_argument("--base_only", action="store_true", help="Use base model without LoRA adapter.")
    ap.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base model if adapter is loaded.")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--lengths", type=str, default="4096,8192,16384,32768")
    ap.add_argument("--depths", type=str, default="0,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--trials_per_cell", type=int, default=1)
    ap.add_argument("--needles_per_prompt", type=int, default=1)
    ap.add_argument(
        "--prompt_mode",
        type=str,
        default="qa",
        choices=["qa", "continuation"],
        help="qa=instruction question/answer style; continuation=forced continuation style.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--manifest_json", type=str, default="")
    return ap


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
        raise RuntimeError("No rotary modules patched for hybrid variant.")
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
        variant = parent_name if parent_name in {"hybrid", "yarn", "pi", "pi_soft"} else "base"
    return variant, rope_cfg, rope_factor, orig_ctx


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, AutoTokenizer, str, str, Optional[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.model_max_length = 10_000_000

    variant = args.variant
    if args.base_only and variant == "auto":
        variant = "base"
    rope_cfg: Optional[dict] = None
    rope_factor = float(args.rope_factor)
    orig_ctx = int(args.orig_ctx)
    custom_inv_freq_path = args.custom_inv_freq_path.strip()

    if not args.base_only:
        if not args.adapter_path:
            raise ValueError("adapter_path is empty but base_only is False.")
        if not custom_inv_freq_path:
            auto_custom = Path(args.adapter_path).resolve().parent / "artifacts" / "custom_inv_freq.pt"
            if auto_custom.exists():
                custom_inv_freq_path = str(auto_custom)
                LOG.info("Auto-detected custom inv_freq: %s", custom_inv_freq_path)
        infer_variant = args.variant if args.variant != "auto" else "auto"
        inferred_variant, inferred_rope_cfg, inferred_factor, inferred_orig_ctx = infer_variant_and_rope_from_adapter(
            adapter_path=args.adapter_path,
            fallback_variant=infer_variant,
            fallback_rope_factor=rope_factor,
            fallback_orig_ctx=orig_ctx,
        )
        if args.variant == "auto":
            variant = inferred_variant
        if inferred_rope_cfg is not None:
            rope_cfg = inferred_rope_cfg
        rope_factor = inferred_factor
        orig_ctx = inferred_orig_ctx
    elif variant == "auto":
        variant = "base"

    if custom_inv_freq_path:
        variant = "custom"
        rope_cfg = None

    if variant == "custom" and not custom_inv_freq_path:
        raise ValueError("variant=custom requires custom_inv_freq_path.")

    if variant not in {"base", "hybrid", "yarn", "pi", "pi_soft", "custom"}:
        LOG.warning("Unknown variant=%s, fallback to base.", variant)
        variant = "base"

    rope_theta = float(args.rope_theta) if args.rope_theta > 0 else infer_rope_theta(args.base_model_path, args.trust_remote_code)
    if isinstance(rope_cfg, dict) and rope_cfg.get("factor") is not None:
        try:
            rope_factor = float(rope_cfg.get("factor"))
        except Exception:
            pass
    if variant in {"yarn", "pi", "pi_soft"} and rope_cfg is None:
        rope_cfg = rope_scaling_candidates(variant, rope_factor, orig_ctx, rope_theta)[0]

    model = None
    used_attn = "default"
    errs: List[str] = []
    for attn in attn_candidates(args.attn_implementation):
        try:
            cfg = AutoConfig.from_pretrained(
                args.base_model_path,
                trust_remote_code=args.trust_remote_code,
                local_files_only=True,
            )
            if variant != "base":
                target_max_pos = max(
                    int(getattr(cfg, "max_position_embeddings", orig_ctx)),
                    int(orig_ctx * max(1.0, rope_factor)),
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
                device_map=args.device_map,
                trust_remote_code=args.trust_remote_code,
                local_files_only=True,
            )
            if attn is not None:
                kwargs["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **kwargs)
            if custom_inv_freq_path:
                inv = load_custom_inv_freq(custom_inv_freq_path)
                patched = patch_hybrid_rope(model, inv)
                rope_cfg = {"custom_inv_freq_path": custom_inv_freq_path}
                LOG.info("Patched custom inv_freq from %s into %d layers", custom_inv_freq_path, patched)
            elif variant == "hybrid":
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                inv = compute_hybrid_inv_freq(
                    head_dim=head_dim,
                    theta_base=rope_theta,
                    split_ratio=args.hybrid_split_ratio,
                    alpha=args.hybrid_alpha,
                    p=args.hybrid_p,
                    min_freq_scale=args.hybrid_min_freq_scale,
                )
                patched = patch_hybrid_rope(model, inv)
                LOG.info("Patched hybrid rotary layers=%d", patched)
            used_attn = attn or "default"
            break
        except Exception as e:
            errs.append(f"variant={variant} rope={rope_cfg} attn={attn}: {type(e).__name__}: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    if model is None:
        raise RuntimeError("Failed to load base model:\n" + "\n".join(errs))

    if args.base_only:
        LOG.info("Running in base-only mode.")
    else:
        if PeftModel is None:
            raise RuntimeError("peft is not installed in this environment.")
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        has_weight = (adapter_path / "adapter_model.safetensors").exists() or (adapter_path / "adapter_model.bin").exists()
        if not has_weight:
            raise FileNotFoundError(
                f"Adapter path exists but no weights found in {adapter_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )
        LOG.info("Loading adapter: %s", adapter_path)
        _normalize_no_split_modules_for_peft(model)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        if args.merge_lora:
            LOG.info("Merging adapter into base model.")
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, used_attn, variant, rope_cfg


def model_main_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _normalize_no_split_modules_for_peft(model: torch.nn.Module) -> None:
    """Normalize model._no_split_modules to avoid accelerate hash/type errors."""
    raw = getattr(model, "_no_split_modules", None)
    if raw is None:
        return
    if isinstance(raw, str):
        return

    normalized: List[str] = []
    try:
        items = list(raw)
    except Exception:
        return

    for item in items:
        if isinstance(item, str):
            normalized.append(item)
            continue
        if isinstance(item, (list, tuple, set)):
            for inner in item:
                if isinstance(inner, str):
                    normalized.append(inner)
        elif item is not None:
            normalized.append(str(item))

    if normalized:
        # Keep order but remove duplicates.
        model._no_split_modules = list(dict.fromkeys(normalized))


def random_passkey(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


def build_needle_sentence(needle_id: int, value: str) -> str:
    return f" [Fact {needle_id}] The secret code for id {needle_id} is {value}. "


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    context_len: int,
    depth_percent: int,
    needles_per_prompt: int,
    prompt_mode: str,
    rng: random.Random,
) -> Tuple[List[int], str, Dict]:
    if needles_per_prompt < 1:
        raise ValueError("needles_per_prompt must be >= 1")

    target_id = rng.randint(1, needles_per_prompt)
    facts = []
    fact_ids = []
    for i in range(1, needles_per_prompt + 1):
        val = random_passkey(rng)
        fact_ids.append((i, val))
        facts.append(tokenizer.encode(build_needle_sentence(i, val), add_special_tokens=False))

    target_value = next(v for i, v in fact_ids if i == target_id)
    if prompt_mode == "continuation":
        instruction = (
            "The following is a long document. Some facts are relevant, many are not.\n"
            "Continue the final sentence with the exact secret code.\n\n"
        )
        question = f"\nBased on the document, the secret code for id {target_id} is"
    else:
        instruction = (
            "Read the long context carefully.\n"
            "It contains secret codes for different ids.\n"
            "Return only the code for the asked id.\n\n"
        )
        question = f"\nQuestion: What is the secret code for id {target_id}?\nAnswer:"

    prefix_ids = tokenizer.encode(instruction, add_special_tokens=False)
    suffix_ids = tokenizer.encode(question, add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    needle_token_total = sum(len(x) for x in facts)
    non_filler = len(bos) + len(prefix_ids) + len(suffix_ids) + needle_token_total
    if non_filler >= context_len:
        raise ValueError(
            f"context_len={context_len} too small for prompt template and {needles_per_prompt} needles "
            f"(non_filler_tokens={non_filler})"
        )

    filler_budget = context_len - non_filler
    filler_unit = tokenizer.encode(
        " This is generic background content without useful codes.",
        add_special_tokens=False,
    )
    if not filler_unit:
        filler_unit = [tokenizer.eos_token_id or 0]
    filler = (filler_unit * (filler_budget // len(filler_unit) + 1))[:filler_budget]

    target_pos = int(round((depth_percent / 100.0) * max(0, filler_budget - 1)))
    all_positions = []
    for idx in range(needles_per_prompt):
        if idx + 1 == target_id:
            all_positions.append(target_pos)
        else:
            all_positions.append(rng.randint(0, max(0, filler_budget - 1)))

    merged: List[int] = []
    cursor = 0
    insert_plan = sorted([(all_positions[i], i) for i in range(len(facts))], key=lambda x: x[0])
    for pos, fact_idx in insert_plan:
        pos = max(0, min(pos, len(filler)))
        merged.extend(filler[cursor:pos])
        merged.extend(facts[fact_idx])
        cursor = pos
    merged.extend(filler[cursor:])

    prompt_ids = bos + prefix_ids + merged + suffix_ids
    if len(prompt_ids) > context_len:
        prompt_ids = prompt_ids[:context_len]
    elif len(prompt_ids) < context_len:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        prompt_ids = prompt_ids + [pad_id] * (context_len - len(prompt_ids))

    meta = {
        "target_id": target_id,
        "target_value": target_value,
        "facts": fact_ids,
        "target_depth_percent": depth_percent,
        "prompt_mode": prompt_mode,
    }
    return prompt_ids, target_value, meta


def extract_candidate_codes(text: str) -> List[str]:
    return re.findall(r"\d{5,10}", text)


@torch.no_grad()
def run_trial(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_len: int,
    depth_percent: int,
    needles_per_prompt: int,
    prompt_mode: str,
    max_new_tokens: int,
    rng: random.Random,
) -> Dict:
    prompt_ids, target_value, meta = build_prompt_ids(
        tokenizer=tokenizer,
        context_len=context_len,
        depth_percent=depth_percent,
        needles_per_prompt=needles_per_prompt,
        prompt_mode=prompt_mode,
        rng=rng,
    )
    device = model_main_device(model)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(x, dtype=torch.long)

    out = model.generate(
        input_ids=x,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen_ids = out[0, x.shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    codes = extract_candidate_codes(text)
    pred = codes[0] if codes else None
    correct = target_value in codes
    return {
        "context_len": context_len,
        "depth_percent": depth_percent,
        "needles_per_prompt": needles_per_prompt,
        "target_value": target_value,
        "pred": pred,
        "all_codes": codes,
        "generation": text,
        "correct": correct,
        "meta": meta,
    }


@torch.no_grad()
def evaluate_matrix(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    lengths: List[int],
    depths: List[int],
    trials_per_cell: int,
    needles_per_prompt: int,
    prompt_mode: str,
    max_new_tokens: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    matrix = np.full((len(depths), len(lengths)), np.nan, dtype=np.float32)
    raw: Dict = {
        "meta": {
            "lengths": lengths,
            "depths": depths,
            "trials_per_cell": trials_per_cell,
            "needles_per_prompt": needles_per_prompt,
            "prompt_mode": prompt_mode,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "cells": {},
    }

    rng = random.Random(seed)
    skip_len = False
    for col, L in enumerate(lengths):
        key_len = str(L)
        raw["cells"][key_len] = {}
        if skip_len:
            for d in depths:
                raw["cells"][key_len][str(d)] = {"status": "skipped_due_to_oom", "accuracy": None, "trials": []}
            continue

        for row, d in enumerate(depths):
            key_depth = str(d)
            raw["cells"][key_len][key_depth] = {"trials": []}
            correct = 0
            total = 0
            status = "ok"
            LOG.info("Cell start: length=%d depth=%d%% needles=%d", L, d, needles_per_prompt)

            for t in range(trials_per_cell):
                try:
                    rec = run_trial(
                        model=model,
                        tokenizer=tokenizer,
                        context_len=L,
                        depth_percent=d,
                        needles_per_prompt=needles_per_prompt,
                        prompt_mode=prompt_mode,
                        max_new_tokens=max_new_tokens,
                        rng=rng,
                    )
                    raw["cells"][key_len][key_depth]["trials"].append(rec)
                    total += 1
                    if rec["correct"]:
                        correct += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        status = "oom"
                        LOG.warning("OOM at length=%d depth=%d%% trial=%d", L, d, t)
                        torch.cuda.empty_cache()
                        skip_len = True
                        break
                    status = "error"
                    raw["cells"][key_len][key_depth]["error"] = f"{type(e).__name__}: {e}"
                    LOG.exception("RuntimeError at length=%d depth=%d%% trial=%d", L, d, t)
                    break
                except Exception as e:
                    status = "error"
                    raw["cells"][key_len][key_depth]["error"] = f"{type(e).__name__}: {e}"
                    LOG.exception("Error at length=%d depth=%d%% trial=%d", L, d, t)
                    break

            raw["cells"][key_len][key_depth]["status"] = status
            raw["cells"][key_len][key_depth]["correct"] = correct
            raw["cells"][key_len][key_depth]["total"] = total
            if total > 0:
                acc = correct / total
                raw["cells"][key_len][key_depth]["accuracy"] = acc
                matrix[row, col] = acc
                LOG.info("Cell done: length=%d depth=%d%% acc=%.3f (%d/%d)", L, d, acc, correct, total)
            else:
                raw["cells"][key_len][key_depth]["accuracy"] = None

    df = pd.DataFrame(
        matrix,
        index=[f"{d}%" for d in depths],
        columns=[str(l) for l in lengths],
    )
    return df, raw


def save_heatmap(df: pd.DataFrame, pdf_path: Path, png_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 7))
    if sns is not None:
        sns.heatmap(
            df,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Recall / Accuracy"},
        )
    else:
        ax = plt.gca()
        arr = df.values.astype(np.float32)
        im = ax.imshow(arr, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = arr[r, c]
                text = "nan" if np.isnan(v) else f"{v:.2f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Recall / Accuracy")
        LOG.warning("seaborn unavailable; used matplotlib fallback.")

    plt.title(title, fontsize=13)
    plt.xlabel("Context Length")
    plt.ylabel("Target Needle Depth")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    enforce_offline_mode()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "niah_recall.log")

    lengths = sorted(set(parse_int_list(args.lengths)))
    depths = sorted(set(parse_int_list(args.depths)))
    if any(x <= 0 for x in lengths):
        raise ValueError("All lengths must be > 0")
    if any(d < 0 or d > 100 for d in depths):
        raise ValueError("Depths must be in [0, 100]")
    if args.needles_per_prompt < 1:
        raise ValueError("needles_per_prompt must be >= 1")

    LOG.info("Loading model...")
    model, tokenizer, attn_used, variant_used, rope_used = load_model_and_tokenizer(args)
    LOG.info("Model ready. attn=%s variant=%s rope=%s", attn_used, variant_used, rope_used)
    LOG.info("Lengths=%s Depths=%s", lengths, depths)

    df, raw = evaluate_matrix(
        model=model,
        tokenizer=tokenizer,
        lengths=lengths,
        depths=depths,
        trials_per_cell=args.trials_per_cell,
        needles_per_prompt=args.needles_per_prompt,
        prompt_mode=args.prompt_mode,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    title = f"NIAH Recall (needles={args.needles_per_prompt})"
    pdf_path = output_dir / "niah_recall_heatmap.pdf"
    png_path = output_dir / "niah_recall_heatmap.png"
    save_heatmap(df, pdf_path, png_path, title=title)
    inv_sha256, inv_path = resolve_inv_metadata(
        base_only=bool(args.base_only),
        adapter_path=args.adapter_path,
        custom_inv_freq_path=args.custom_inv_freq_path,
        rope_used=rope_used if isinstance(rope_used, dict) else None,
    )
    per_sample_scores_raw: List[float] = []
    for len_map in (raw.get("cells") or {}).values():
        if not isinstance(len_map, dict):
            continue
        for depth_obj in len_map.values():
            if not isinstance(depth_obj, dict):
                continue
            for trial in depth_obj.get("trials", []):
                if not isinstance(trial, dict):
                    continue
                correct = trial.get("correct")
                if isinstance(correct, bool):
                    per_sample_scores_raw.append(1.0 if correct else 0.0)
                elif isinstance(correct, (int, float)):
                    per_sample_scores_raw.append(float(correct))

    manifest_json = Path(args.manifest_json).resolve().as_posix() if args.manifest_json else ""
    protocol_lock = {
        "base_model_path": args.base_model_path,
        "adapter_path": None if args.base_only else args.adapter_path,
        "variant": args.variant,
        "custom_inv_freq_path": args.custom_inv_freq_path,
        "attn_implementation": args.attn_implementation,
        "lengths": lengths,
        "depths": depths,
        "trials_per_cell": int(args.trials_per_cell),
        "needles_per_prompt": int(args.needles_per_prompt),
        "prompt_mode": args.prompt_mode,
        "seed": int(args.seed),
        "decode": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "use_cache": True,
        },
    }

    payload = {
        "protocol_lock": protocol_lock,
        "manifest_json": manifest_json,
        "per_sample_scores_raw": per_sample_scores_raw,
        "inv_sha256": inv_sha256,
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
            "base_model_path": args.base_model_path,
            "adapter_path": None if args.base_only else args.adapter_path,
            "base_only": args.base_only,
            "merge_lora": args.merge_lora,
            "attn_used": attn_used,
            "variant_used": variant_used,
            "rope_used": rope_used,
            "prompt_mode": args.prompt_mode,
            "lengths": lengths,
            "depths": depths,
            "trials_per_cell": args.trials_per_cell,
            "needles_per_prompt": args.needles_per_prompt,
            "seed": args.seed,
            "manifest_json": manifest_json,
            "inv_sha256": inv_sha256,
            "inv_freq_path": inv_path,
            "protocol_lock": protocol_lock,
        },
        "accuracy_matrix": df.to_dict(orient="index"),
        "raw": raw,
    }
    (output_dir / "niah_recall_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOG.info("Saved %s", pdf_path)
    LOG.info("Saved %s", png_path)
    LOG.info("Saved %s", output_dir / "niah_recall_results.json")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
