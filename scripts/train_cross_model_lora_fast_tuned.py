#!/usr/bin/env python3
"""Train one cross-model LoRA run with strict fair hyper-parameters."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rope.inject import apply_inv_freq_inplace, find_rotary_modules_with_inv_freq, hash_tensor_sha256
from rope.schedules import canonical_method, infer_rope_base_from_config


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_targets(v: str) -> List[str]:
    targets = [x.strip() for x in v.split(",") if x.strip()]
    if not targets:
        raise ValueError("lora_target_modules cannot be empty.")
    return targets


def load_text_tokens(data_file: Path, tokenizer: AutoTokenizer) -> List[int]:
    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")
    text = data_file.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 2000:
        raise RuntimeError(f"Too few tokens in {data_file}: {len(tokens)}")
    return tokens


class RandomWindowDataset(Dataset):
    def __init__(self, tokens: Sequence[int], seq_len: int, n_windows: int, seed: int) -> None:
        if len(tokens) <= seq_len + 2:
            raise RuntimeError(f"Not enough tokens ({len(tokens)}) for seq_len={seq_len}")
        self.tokens = list(tokens)
        self.seq_len = int(seq_len)
        self.n_windows = int(n_windows)
        max_start = len(self.tokens) - self.seq_len - 1
        rng = random.Random(int(seed))
        self.starts = [rng.randint(0, max_start) for _ in range(self.n_windows)]

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


def attn_candidates(mode: str) -> List[Optional[str]]:
    mode = mode.strip().lower()
    if mode == "auto":
        return ["flash_attention_2", "sdpa", "eager", None]
    if mode in {"default", "none"}:
        return [None]
    return [mode]


def geometric_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    k = head_dim // 2
    idx = torch.arange(k, dtype=torch.float64)
    return 1.0 / (float(base) ** (2.0 * idx / float(head_dim)))


def evq_cosh_inv_freq(head_dim: int, base: float, tau: float) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    tau = float(tau)
    if tau < 0:
        raise ValueError(f"evq_tau must be non-negative, got {tau}")
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float64)
    u = idx / float(n)
    if tau <= 1e-4:
        # Exact degeneration to geometric grid when tau is tiny.
        phi = u
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.asinh((1.0 - u) * float(sinh_tau))
    return torch.pow(float(base), -phi)


def evq_exp_inv_freq(head_dim: int, base: float, beta: float) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"evq_beta must be positive, got {beta}")
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float64)
    u = idx / float(n)
    if beta <= 1e-6:
        phi = u
    else:
        phi = (torch.pow(1.0 + beta, u) - 1.0) / beta
    return torch.pow(float(base), -phi)


def build_inv_freq_fast_tuned(
    method: str,
    head_dim: int,
    base: float,
    max_seq_len: int,
    anchor_factor: float,
    slope_raw: float,
    center_ratio: float,
    evq_tau: float,
    evq_beta: float,
) -> torch.Tensor:
    m = canonical_method(method)
    base_inv = geometric_inv_freq(head_dim=head_dim, base=base)
    slope_raw = float(slope_raw)
    center_ratio = float(center_ratio)
    if slope_raw <= 0:
        raise ValueError(f"slope_raw must be positive, got {slope_raw}")
    if not (0.0 <= center_ratio <= 1.0):
        raise ValueError(f"center_ratio must be within [0, 1], got {center_ratio}")

    if m == "baseline":
        return base_inv
    if m == "evq_cosh":
        return evq_cosh_inv_freq(head_dim=head_dim, base=base, tau=evq_tau)
    if m == "evq_exp":
        return evq_exp_inv_freq(head_dim=head_dim, base=base, beta=evq_beta)
    if m != "anchored_sigmoid":
        raise ValueError(f"Unsupported method for fast-tuned runner: {method}")

    n = head_dim // 2
    scale = max(float(max_seq_len) / 8192.0, 1.0)
    eff_anchor = float(anchor_factor)
    if eff_anchor <= 0:
        eff_anchor = min(max(2.0, 2.5 * scale), 30.0)
    slope = slope_raw / float(head_dim)
    center = center_ratio * float(n)
    idx = torch.arange(n, dtype=torch.float64)
    sig = 1.0 / (1.0 + torch.exp(-slope * (idx - center)))
    return base_inv / (1.0 + (eff_anchor - 1.0) * sig)


def resolve_model_path(
    base_model_path: str,
    modelscope_repo: str,
    model_cache_dir: str,
    allow_modelscope_download: bool,
) -> Tuple[str, bool]:
    p = Path(base_model_path)
    if p.exists():
        return str(p), False

    if not allow_modelscope_download:
        raise FileNotFoundError(
            f"Model path does not exist: {base_model_path}. "
            "Set --allow_modelscope_download and --modelscope_repo to auto-download."
        )
    if not modelscope_repo:
        raise ValueError("modelscope_repo is required when auto-download is enabled.")

    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "ModelScope is required for auto-download. Install with `pip install modelscope`."
        ) from exc

    cache_dir = str(Path(model_cache_dir).expanduser())
    try:
        dl_path = snapshot_download(
            modelscope_repo,
            cache_dir=cache_dir,
            ignore_patterns=["original/*", "*.pth"],
        )
    except TypeError:
        dl_path = snapshot_download(model_id=modelscope_repo, cache_dir=cache_dir)
    return str(dl_path), True


def load_model_and_tokenizer(
    model_path: str,
    load_in_4bit: bool,
    bf16: bool,
    attn_implementation: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_cfg = None
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    errors: List[str] = []
    for attn in attn_candidates(attn_implementation):
        kwargs = dict(
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            device_map="auto",
        )
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
        if attn is not None:
            kwargs["attn_implementation"] = attn
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            return model, tokenizer, attn or "default"
        except Exception as exc:
            errors.append(f"attn={attn}: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("Failed to load model with all attention candidates:\n" + "\n".join(errors))


def build_and_inject_inv_freq(
    model: torch.nn.Module,
    method: str,
    model_path: str,
    max_seq_len: int,
    rope_base: float,
    anchor_factor: float,
    slope_raw: float,
    center_ratio: float,
    evq_tau: float,
    evq_beta: float,
) -> Dict[str, object]:
    modules = find_rotary_modules_with_inv_freq(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found.")

    head_dim = int(modules[0][1].inv_freq.numel()) * 2
    base = float(rope_base)
    if base <= 0:
        base = infer_rope_base_from_config(model_path, fallback=500000.0)

    inv = build_inv_freq_fast_tuned(
        method=method,
        head_dim=head_dim,
        base=base,
        max_seq_len=max_seq_len,
        anchor_factor=anchor_factor,
        slope_raw=slope_raw,
        center_ratio=center_ratio,
        evq_tau=evq_tau,
        evq_beta=evq_beta,
    )
    inject = apply_inv_freq_inplace(model=model, inv_freq=inv)
    schedule_params = {
        "anchor_factor_requested": float(anchor_factor),
        "slope_raw": float(slope_raw),
        "center_ratio": float(center_ratio),
        "evq_tau": float(evq_tau),
        "evq_beta": float(evq_beta),
    }
    if method == "anchored_sigmoid":
        scale = max(float(max_seq_len) / 8192.0, 1.0)
        eff_anchor = float(anchor_factor)
        if eff_anchor <= 0:
            eff_anchor = min(max(2.0, 2.5 * scale), 30.0)
        schedule_params["anchor_factor_effective"] = float(eff_anchor)

    return {
        "method": method,
        "head_dim": head_dim,
        "rope_base": base,
        "inv_sha256": hash_tensor_sha256(inv),
        "inv_min": float(inv.min().item()),
        "inv_max": float(inv.max().item()),
        "inject_info": inject,
        "schedule_params": schedule_params,
        "inv_freq": inv.to(torch.float64),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train one cross-model LoRA run.")
    ap.add_argument("--method", type=str, required=True, choices=["baseline", "anchored_sigmoid", "evq_cosh", "evq_exp"])
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--modelscope_repo", type=str, default="")
    ap.add_argument("--allow_modelscope_download", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--model_cache_dir", type=str, default="/root/autodl-tmp/dfrope/ms_models")
    ap.add_argument("--data_dir", type=str, default="/root/autodl-tmp/wikitext_data")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_seq_len", type=int, default=16384)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--optim", type=str, default="paged_adamw_8bit")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--rope_base", type=float, default=0.0)
    ap.add_argument("--anchor_factor", type=float, default=4.0)
    ap.add_argument("--slope_raw", type=float, default=20.0)
    ap.add_argument("--center_ratio", type=float, default=0.70)
    ap.add_argument("--evq_tau", type=float, default=0.5)
    ap.add_argument("--evq_beta", type=float, default=3.0)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--attn_implementation", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.method = canonical_method(args.method)
    if args.method not in {"baseline", "anchored_sigmoid", "evq_cosh", "evq_exp"}:
        raise ValueError(f"Unsupported method for this runner: {args.method}")

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    artifacts_dir = out_dir / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path, downloaded = resolve_model_path(
        base_model_path=args.base_model_path,
        modelscope_repo=args.modelscope_repo,
        model_cache_dir=args.model_cache_dir,
        allow_modelscope_download=args.allow_modelscope_download,
    )

    model, tokenizer, attn_used = load_model_and_tokenizer(
        model_path=model_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    rope_info = build_and_inject_inv_freq(
        model=model,
        method=args.method,
        model_path=model_path,
        max_seq_len=args.max_seq_len,
        rope_base=args.rope_base,
        anchor_factor=args.anchor_factor,
        slope_raw=args.slope_raw,
        center_ratio=args.center_ratio,
        evq_tau=args.evq_tau,
        evq_beta=args.evq_beta,
    )
    torch.save(rope_info["inv_freq"], artifacts_dir / "custom_inv_freq.pt")

    train_tokens = load_text_tokens(Path(args.data_dir) / "train.txt", tokenizer)
    n_windows = int(args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps)
    train_ds = RandomWindowDataset(
        tokens=train_tokens,
        seq_len=args.max_seq_len,
        n_windows=n_windows,
        seed=args.seed,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
        )
    elif bool(args.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_targets(args.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    run_name = args.run_name or Path(args.output_dir).name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_hours = (time.time() - t0) / 3600.0

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Keep backward compatibility (root adapter files) while enforcing the
    # newer contract: <run_dir>/final_lora as the canonical adapter directory.
    final_lora_dir = out_dir / "final_lora"
    final_lora_dir.mkdir(parents=True, exist_ok=True)
    final_lora_export_mode = "save_pretrained"
    try:
        model.save_pretrained(str(final_lora_dir))
    except Exception as exc:
        final_lora_export_mode = f"fallback_copy:{type(exc).__name__}"
    for name in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = out_dir / name
        dst = final_lora_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    has_root_adapter = (out_dir / "adapter_config.json").exists()
    has_final_lora = (final_lora_dir / "adapter_config.json").exists()
    if has_root_adapter and has_final_lora:
        adapter_layout = "dual_root_and_final_lora"
    elif has_final_lora:
        adapter_layout = "final_lora_only"
    elif has_root_adapter:
        adapter_layout = "root_adapter_only"
    else:
        adapter_layout = "none"
    adapter_resolved_path = str(final_lora_dir if has_final_lora else out_dir) if adapter_layout != "none" else ""

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    summary = {
        "timestamp": now(),
        "method": args.method,
        "seed": int(args.seed),
        "base_model_path_input": args.base_model_path,
        "base_model_path_resolved": model_path,
        "downloaded_by_modelscope": bool(downloaded),
        "attn_used": attn_used,
        "data_dir": args.data_dir,
        "hyperparams": {
            "max_steps": int(args.max_steps),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "max_seq_len": int(args.max_seq_len),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "lora_target_modules": parse_targets(args.lora_target_modules),
            "bf16": bool(args.bf16),
            "lr_scheduler_type": args.lr_scheduler_type,
            "load_in_4bit": bool(args.load_in_4bit),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "optim": args.optim,
        },
        "train": {
            "train_loss": train_result.metrics.get("train_loss"),
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "train_hours": round(train_hours, 4),
            "peak_cuda_allocated_gb": round(float(peak_mem_gb), 4) if peak_mem_gb is not None else None,
        },
        "rope": {
            "method": rope_info["method"],
            "head_dim": rope_info["head_dim"],
            "rope_base": rope_info["rope_base"],
            "inv_sha256": rope_info["inv_sha256"],
            "inv_min": rope_info["inv_min"],
            "inv_max": rope_info["inv_max"],
            "inject_info": rope_info["inject_info"],
            "schedule_params": rope_info["schedule_params"],
        },
        "artifacts_contract": {
            "adapter_layout": adapter_layout,
            "adapter_resolved_path": adapter_resolved_path,
            "root_adapter_path": str(out_dir),
            "final_lora_path": str(final_lora_dir),
            "final_lora_export_mode": final_lora_export_mode,
        },
        "output_dir": str(out_dir),
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
