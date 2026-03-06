#!/usr/bin/env python3
"""
Train one fair Llama-3-8B LoRA variant (600-step default) and evaluate long-context PPL.

Variant choices:
- hybrid   : custom Hybrid-RoPE inv_freq patch
- yarn     : YaRN rope scaling
- pi       : position interpolation (linear rope scaling)
- pi_soft  : dynamic soft-boundary PI-style scaling

Design goals:
- offline/local only
- same LoRA/training hyperparameters across variants
- robust logging and OOM-safe evaluation
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def compute_hybrid_inv_freq(head_dim: int, theta_base: float = 100000, alpha: float = 0.2, p: float = 3.9, omf: float = 0.3) -> torch.Tensor:
    k = head_dim // 2
    if k <= 0:
        raise ValueError(f"Invalid head_dim: {head_dim}")
    idx = torch.arange(0, k, dtype=torch.float32)
    t = idx / max(1, k - 1)
    poly = torch.pow(t + 1e-8, p)
    mixed = (1.0 - alpha) * t + alpha * poly
    inv_freq = omf / (theta_base ** mixed)
    return inv_freq


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


def infer_rope_theta(
    base_model_path: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> float:
    # Llama-3 local configs in some transformers versions expose rope_theta as None on AutoConfig;
    # read raw config dict first, then fall back to a safe default.
    theta: Optional[float] = None
    try:
        cfg_dict, _ = AutoConfig.get_config_dict(
            base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
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
            # Fallback candidates for transformers variants that may not support YaRN on this checkpoint.
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


def attn_candidates(mode: str) -> List[Optional[str]]:
    if mode == "auto":
        return ["flash_attention_2", "sdpa", None]
    return [mode]


def load_text_tokens(data_file: Path, tokenizer: AutoTokenizer) -> List[int]:
    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")
    text = data_file.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 1000:
        raise RuntimeError(f"Too few tokens in {data_file}: {len(tokens)}")
    return tokens


class RandomWindowDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int, n_windows: int, seed: int):
        if len(tokens) <= seq_len + 2:
            raise RuntimeError(f"Not enough tokens ({len(tokens)}) for seq_len={seq_len}")
        self.seq_len = seq_len
        self.tokens = tokens
        self.n_windows = n_windows
        rng = random.Random(seed)
        max_start = len(tokens) - seq_len - 1
        self.starts = [rng.randint(0, max_start) for _ in range(n_windows)]

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        st = self.starts[idx]
        x = self.tokens[st : st + self.seq_len]
        ids = torch.tensor(x, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        lr = logs.get("learning_rate", 0.0)
        msg = f"[step={state.global_step}] loss={float(loss):.6f} lr={float(lr):.3e}"
        if torch.cuda.is_available():
            max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
            max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            msg += f" max_alloc_gb={max_alloc_gb:.2f} max_reserved_gb={max_reserved_gb:.2f}"
        print(msg, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


class StabilityGuardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state.global_step}: {loss}")
        if loss > 100:
            raise RuntimeError(f"Abnormal loss at step {state.global_step}: {loss}")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    attn_used: str
    rope_used: Optional[dict]


def load_model_for_variant(
    base_model_path: str,
    variant: str,
    rope_factor: float,
    orig_ctx: int,
    attn_mode: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_rope_theta = infer_rope_theta(
        base_model_path=base_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    print(f"[rope] inferred base rope_theta={base_rope_theta}", flush=True)

    load_errs: List[str] = []
    for rope_cfg in rope_scaling_candidates(variant, rope_factor, orig_ctx, base_rope_theta):
        for attn in attn_candidates(attn_mode):
            try:
                cfg = AutoConfig.from_pretrained(
                    base_model_path,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
                if rope_cfg is not None:
                    # Newer transformers internally use `rope_parameters`; keep both for compatibility.
                    cfg.rope_scaling = dict(rope_cfg)
                    if hasattr(cfg, "rope_parameters"):
                        cfg.rope_parameters = dict(rope_cfg)
                    if hasattr(cfg, "rope_theta") and getattr(cfg, "rope_theta", None) is None:
                        cfg.rope_theta = float(base_rope_theta)
                cfg.max_position_embeddings = max(
                    int(getattr(cfg, "max_position_embeddings", orig_ctx)),
                    int(orig_ctx * rope_factor),
                )

                kwargs = dict(
                    config=cfg,
                    quantization_config=bnb_cfg,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
                if attn is not None:
                    kwargs["attn_implementation"] = attn

                model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False

                if variant == "hybrid":
                    head_dim = model.config.hidden_size // model.config.num_attention_heads
                    inv = compute_hybrid_inv_freq(head_dim=head_dim)
                    patched = patch_hybrid_rope(model, inv)
                    print(f"[rope] patched hybrid rotary layers={patched}", flush=True)

                return LoadedModel(
                    model=model,
                    tokenizer=tokenizer,
                    attn_used=attn or "default",
                    rope_used=rope_cfg,
                )
            except Exception as exc:
                load_errs.append(
                    f"rope={rope_cfg} attn={attn}: {type(exc).__name__}: {exc}"
                )
                gc.collect()
                torch.cuda.empty_cache()

    raise RuntimeError("Model load failed for all rope/attn candidates:\n" + "\n".join(load_errs))


@torch.no_grad()
def eval_ppl_lengths(
    model: torch.nn.Module,
    tokens: List[int],
    lengths: List[int],
    n_chunks: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    rng = random.Random(seed)
    out: Dict[str, Dict[str, float]] = {}

    for L in lengths:
        losses: List[float] = []
        if len(tokens) <= L + 2:
            out[str(L)] = {"mean_loss": float("nan"), "ppl": float("nan"), "n": 0}
            continue

        max_start = len(tokens) - L - 1
        for _ in range(n_chunks):
            st = rng.randint(0, max_start)
            x = torch.tensor(tokens[st : st + L], dtype=torch.long, device=model.device).unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.long)
            try:
                outputs = model(input_ids=x, attention_mask=mask, labels=x, use_cache=False)
                lv = float(outputs.loss.item())
                if math.isfinite(lv):
                    losses.append(lv)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    break
                raise

        if losses:
            mean_loss = float(np.mean(losses))
            out[str(L)] = {
                "mean_loss": mean_loss,
                "ppl": safe_exp(mean_loss),
                "n": len(losses),
                "std_loss": float(np.std(losses)),
            }
        else:
            out[str(L)] = {"mean_loss": float("nan"), "ppl": float("nan"), "n": 0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one fair Llama-3-8B LoRA rope variant.")
    ap.add_argument("--variant", type=str, required=True, choices=["hybrid", "yarn", "pi", "pi_soft"])
    ap.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data_dir", type=str, default="/root/autodl-tmp/wikitext_data")
    ap.add_argument("--output_root", type=str, default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_lora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--eval_lengths", type=str, default="16384,32768,65536")
    ap.add_argument("--eval_chunks", type=int, default=5)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    enforce_offline_mode()
    set_seed(args.seed)

    run_dir = Path(args.output_root) / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    with train_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== START {time.strftime('%Y-%m-%d %H:%M:%S')} variant={args.variant} ===\n")

    loaded = load_model_for_variant(
        base_model_path=args.base_model_path,
        variant=args.variant,
        rope_factor=args.rope_factor,
        orig_ctx=args.orig_ctx,
        attn_mode=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    print(f"[load] variant={args.variant} attn={loaded.attn_used} rope={loaded.rope_used}", flush=True)

    train_tokens = load_text_tokens(Path(args.data_dir) / "train.txt", tokenizer)
    valid_tokens = load_text_tokens(Path(args.data_dir) / "valid.txt", tokenizer)

    n_windows = args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps
    train_ds = RandomWindowDataset(
        tokens=train_tokens,
        seq_len=args.seq_len,
        n_windows=n_windows,
        seed=args.seed,
    )

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,
        callbacks=[LossLoggerCallback(train_log), StabilityGuardCallback()],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_hours = (time.time() - t0) / 3600.0

    adapter_dir = run_dir / "final_lora"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_lengths = [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]
    ppl = eval_ppl_lengths(
        model=model,
        tokens=valid_tokens,
        lengths=eval_lengths,
        n_chunks=args.eval_chunks,
        seed=args.seed + 123,
    )

    metrics = train_result.metrics if isinstance(train_result.metrics, dict) else {}
    summary = {
        "variant": args.variant,
        "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S"),
        "base_model_path": args.base_model_path,
        "data_dir": args.data_dir,
        "train": {
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "train_hours": round(train_hours, 4),
            "train_loss": metrics.get("train_loss"),
            "train_runtime": metrics.get("train_runtime"),
            "train_steps_per_second": metrics.get("train_steps_per_second"),
        },
        "rope": {
            "factor": args.rope_factor,
            "orig_ctx": args.orig_ctx,
            "rope_scaling_used": loaded.rope_used,
            "attn_used": loaded.attn_used,
        },
        "eval_ppl": ppl,
        "paths": {
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "log": str(train_log),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
