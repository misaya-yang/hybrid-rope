#!/usr/bin/env python3
"""
EVQ-Cosh LoRA Training for LLaMA-3-8B-Instruct
================================================
Supporting experiment: validates LoRA r=64 with EVQ-cosh τ=1.414 against
the phase-transition collapse observed at r=16 (PPL 77.1).

Usage:
    # Full training (requires GPU)
    python train_evq_lora.py --output_dir ./checkpoints/evq_r64

    # Dry-run (no GPU, validates config only)
    python train_evq_lora.py --dry_run --output_dir ./checkpoints/evq_r64_dry

    # Custom tau / rank
    python train_evq_lora.py --tau 1.0 --lora_r 32 --output_dir ./checkpoints/evq_r32
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.rope.schedules import (
    evq_cosh_inv_freq as _canonical_evq_cosh_inv_freq,
    geometric_inv_freq as _canonical_geometric_inv_freq,
)


# ---------------------------------------------------------------------------
# EVQ-Cosh frequency computation (from theory)
# ---------------------------------------------------------------------------

def compute_evq_cosh_inv_freq(
    head_dim: int,
    base: float,
    tau: float,
    midpoint: bool = True,
) -> torch.Tensor:
    """Compute EVQ-cosh inverse frequencies.

    φ_k(τ) = 1 - (1/τ) * arcsinh((1 - u_k) * sinh(τ))
    inv_freq_k = base^{-φ_k}

    Args:
        head_dim: dimension per attention head (e.g. 128)
        base: RoPE theta base (e.g. 500000)
        tau: temperature parameter (theory: d_head/√L)
        midpoint: if True, use u_k = (2k-1)/(2K) (midpoint quantization)
                  if False, use u_k = k/K (boundary quantization)
    """
    return _canonical_evq_cosh_inv_freq(
        head_dim=head_dim,
        tau=tau,
        base=base,
        midpoint=midpoint,
        dtype=torch.float64,
    )


def compute_geometric_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """Standard geometric RoPE inverse frequencies."""
    return _canonical_geometric_inv_freq(head_dim=head_dim, base=base, dtype=torch.float64)


# ---------------------------------------------------------------------------
# RoPE injection
# ---------------------------------------------------------------------------

def find_rotary_modules(model: torch.nn.Module):
    """Find all modules with inv_freq buffer."""
    out = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and torch.is_tensor(module.inv_freq):
            out.append((name, module))
    return out


def inject_inv_freq(model: torch.nn.Module, inv_freq: torch.Tensor) -> Dict[str, Any]:
    """Inject custom inv_freq into all rotary modules."""
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found in model.")

    expected = inv_freq.detach().cpu().view(-1)
    changed = []

    for name, module in modules:
        old = module.inv_freq
        if old.numel() != expected.numel():
            raise RuntimeError(
                f"Shape mismatch at {name}: old={tuple(old.shape)} new={tuple(expected.shape)}"
            )
        with torch.no_grad():
            old.copy_(expected.to(device=old.device, dtype=old.dtype))
        # Clear cached cos/sin
        for attr in ("_cos_cached", "_sin_cached", "cos_cached", "sin_cached",
                      "_cos_cache", "_sin_cache", "max_seq_len_cached"):
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, (int, float)):
                    setattr(module, attr, 0)
                else:
                    setattr(module, attr, None)
        changed.append(name)

    return {"patched_count": len(modules), "changed_modules": changed}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    tokenizer,
    dataset_name: str = "THUDM/LongAlign-10k",
    max_seq_len: int = 8192,
    max_samples: int = 8000,
    val_ratio: float = 0.02,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and tokenize long-context training data. Caches tokenized result to disk."""
    import hashlib, random
    from datasets import load_dataset

    # Check cache
    if cache_dir:
        cache_key = hashlib.md5(f"{dataset_name}_{max_seq_len}_{max_samples}".encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"tokenized_{cache_key}.pt")
        if os.path.exists(cache_path):
            print(f"[DATA] Loading cached tokenized data from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[DATA] Train: {len(cached['train'])}, Val: {len(cached['val'])}")
            return cached

    print(f"[DATA] Loading dataset: {dataset_name}")
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    print(f"[DATA] Raw samples: {len(ds)}")

    # Normalize to messages format
    processed = []
    for item in ds:
        messages = None

        # Format 1: messages array (LongAlign style)
        if "messages" in item and item["messages"]:
            messages = item["messages"]
        # Format 2: instruction/input/output
        elif "instruction" in item:
            user_text = item.get("instruction", "")
            if item.get("input"):
                user_text = f"{user_text}\n\n{item['input']}"
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item.get("output", "")},
            ]
        # Format 3: question/context/answer
        elif "question" in item:
            ctx = item.get("context", "")
            q = item.get("question", "")
            user_text = f"{ctx}\n\n{q}" if ctx else q
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item.get("answer", item.get("answers", ""))},
            ]

        if messages:
            processed.append(messages)

        if len(processed) >= max_samples:
            break

    print(f"[DATA] Processed samples: {len(processed)}")

    # Tokenize
    tokenized = []
    skipped = 0
    for i, msgs in enumerate(processed):
        if i % 1000 == 0 and i > 0:
            print(f"[DATA] Tokenizing {i}/{len(processed)}...")
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            text = "<|begin_of_text|>" + "".join(parts)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        if len(enc["input_ids"]) < 64:
            skipped += 1
            continue
        tokenized.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    print(f"[DATA] Tokenized: {len(tokenized)}, skipped (too short): {skipped}")

    # Token length distribution
    lengths = [len(t["input_ids"]) for t in tokenized]
    print(f"[DATA] Token lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    print(f"[DATA] Samples at max_seq_len: {sum(1 for l in lengths if l >= max_seq_len - 10)}")

    # Split train/val
    n_val = max(1, int(len(tokenized) * val_ratio))
    random.seed(42)
    indices = list(range(len(tokenized)))
    random.shuffle(indices)
    val_indices = set(indices[:n_val])
    train_data = [tokenized[i] for i in indices if i not in val_indices]
    val_data = [tokenized[i] for i in val_indices]

    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")

    result = {"train": train_data, "val": val_data}

    # Save cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(result, cache_path)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"[DATA] Cached to {cache_path} ({size_mb:.0f}MB)")

    return result


class TokenizedDataset(torch.utils.data.Dataset):
    """Simple dataset from pre-tokenized data."""
    def __init__(self, data: List[Dict], max_seq_len: int):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"][:self.max_seq_len]
        attention_mask = item["attention_mask"][:self.max_seq_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": list(input_ids),
        }


class PaddingCollator:
    """Pad variable-length samples to the longest in the batch."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EVQ-Cosh LoRA Training")

    # Model
    p.add_argument("--model_name", type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--output_dir", type=str, required=True)

    # EVQ-Cosh parameters
    p.add_argument("--tau", type=float, default=1.414,
                   help="EVQ-cosh temperature (theory: d_head/sqrt(L))")
    p.add_argument("--rope_base", type=float, default=500000.0,
                   help="RoPE theta base")
    p.add_argument("--head_dim", type=int, default=128,
                   help="Attention head dimension")

    # LoRA
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank (theory requires r >= K = d_head/2)")
    p.add_argument("--lora_alpha", type=int, default=128,
                   help="LoRA alpha (default: 2 * lora_r)")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj")

    # Training (bf16 full-precision LoRA — lr lower than QLoRA's 2e-4)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=60)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=8192)

    # Data
    p.add_argument("--dataset_name", type=str,
                   default="THUDM/LongAlign-10k",
                   help="HuggingFace dataset name")
    p.add_argument("--max_samples", type=int, default=8000)
    p.add_argument("--local_data_path", type=str, default=None,
                   help="Path to local JSONL data (overrides --dataset_name)")

    # Quantization (96GB GPU: default bf16 full precision, no quantization needed)
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit QLoRA (only if GPU < 40GB)")
    p.add_argument("--no_4bit", action="store_true", default=False)

    # Control
    p.add_argument("--dry_run", action="store_true",
                   help="Validate config without training (no GPU required)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true", default=True)

    return p.parse_args()


def validate_theory(args) -> Dict[str, Any]:
    """Validate experiment parameters against theoretical predictions."""
    K = args.head_dim // 2
    tau_theory = args.head_dim / math.sqrt(args.max_seq_len)
    r_ratio = args.lora_r / K

    checks = {
        "head_dim": args.head_dim,
        "K_channels": K,
        "lora_r": args.lora_r,
        "r_over_K": r_ratio,
        "tau_set": args.tau,
        "tau_theory": round(tau_theory, 4),
        "tau_match": abs(args.tau - tau_theory) < 0.1,
        "phase_transition_safe": args.lora_r >= K,
    }

    print("\n" + "=" * 60)
    print("THEORETICAL VALIDATION")
    print("=" * 60)
    print(f"  head_dim       = {args.head_dim}")
    print(f"  K (channels)   = {K}")
    print(f"  LoRA rank r    = {args.lora_r}")
    print(f"  r / K          = {r_ratio:.2f} {'✅' if r_ratio >= 1.0 else '⚠️' if r_ratio >= 0.5 else '❌'}")
    print(f"  τ (set)        = {args.tau}")
    print(f"  τ* (theory)    = {tau_theory:.4f}")
    print(f"  τ match        = {'✅' if checks['tau_match'] else '⚠️'}")
    print(f"  Phase-safe     = {'✅' if checks['phase_transition_safe'] else '❌ DANGER'}")

    if r_ratio < 0.5:
        print(f"\n  ⚠️  WARNING: r/K = {r_ratio:.2f} < 0.5")
        print(f"  Theory predicts EVQ will FAIL (phase transition at r_c ≈ K = {K})")
        print(f"  Recommend: r >= {K} (r/K >= 1.0)")

    print("=" * 60 + "\n")
    return checks


def compute_and_save_inv_freq(args) -> torch.Tensor:
    """Compute EVQ-cosh frequencies and save for reproducibility."""
    inv_freq_evq = compute_evq_cosh_inv_freq(
        head_dim=args.head_dim,
        base=args.rope_base,
        tau=args.tau,
        midpoint=True,
    )
    inv_freq_geo = compute_geometric_inv_freq(args.head_dim, args.rope_base)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    freq_path = os.path.join(args.output_dir, "custom_inv_freq.pt")
    torch.save({
        "inv_freq": inv_freq_evq,
        "tau": args.tau,
        "head_dim": args.head_dim,
        "base": args.rope_base,
        "method": "evq_cosh",
        "midpoint": True,
    }, freq_path)
    print(f"[FREQ] Saved to {freq_path}")

    # Diagnostic comparison
    K = args.head_dim // 2
    print(f"\n[FREQ] EVQ-cosh vs Geometric comparison (τ={args.tau}):")
    print(f"  {'Chan':>4s}  {'EVQ':>12s}  {'Geo':>12s}  {'Ratio':>8s}")
    for k in [0, K//4, K//2, 3*K//4, K-1]:
        e = inv_freq_evq[k].item()
        g = inv_freq_geo[k].item()
        print(f"  {k:4d}  {e:12.6f}  {g:12.6f}  {e/g:8.4f}")

    return inv_freq_evq


def main():
    args = parse_args()
    if args.no_4bit:
        args.load_in_4bit = False

    # 1. Theoretical validation
    theory_checks = validate_theory(args)

    # 2. Compute EVQ-cosh frequencies
    inv_freq = compute_and_save_inv_freq(args)

    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting without training.")
        config = {
            "model": args.model_name,
            "tau": args.tau,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_steps": args.max_steps,
            "max_seq_len": args.max_seq_len,
            "dataset": args.dataset_name,
            "theory": theory_checks,
        }
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[DRY RUN] Config saved to {config_path}")
        return

    # ---- Below requires GPU ----
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(args.seed)

    # 3. Load tokenizer
    print(f"\n[MODEL] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. Load model
    precision = "4-bit QLoRA" if args.load_in_4bit else "bf16 full precision"
    print(f"[MODEL] Loading model ({precision})")
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    # 5. Inject EVQ-cosh frequencies
    print("[ROPE] Injecting EVQ-cosh frequencies...")
    inject_result = inject_inv_freq(model, inv_freq)
    print(f"[ROPE] Patched {inject_result['patched_count']} modules: "
          f"{inject_result['changed_modules'][:3]}...")

    # Verify injection
    modules = find_rotary_modules(model)
    if modules:
        actual = modules[0][1].inv_freq.detach().cpu().to(torch.float64)
        expected = inv_freq.cpu().to(torch.float64)
        max_err = (actual - expected).abs().max().item()
        print(f"[ROPE] Injection verification: max_error = {max_err:.2e} "
              f"{'✅' if max_err < 1e-5 else '❌ MISMATCH'}")

    # 6. Prepare for LoRA
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[LORA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 7. Re-verify inv_freq not overwritten by PEFT
    modules_after = find_rotary_modules(model)
    if modules_after:
        actual_after = modules_after[0][1].inv_freq.detach().cpu().to(torch.float64)
        max_err_after = (actual_after - inv_freq.cpu().to(torch.float64)).abs().max().item()
        print(f"[ROPE] Post-PEFT verification: max_error = {max_err_after:.2e} "
              f"{'✅' if max_err_after < 1e-5 else '❌ PEFT OVERWROTE INV_FREQ!'}")

    # 8. Load data
    data = load_training_data(
        tokenizer=tokenizer,
        dataset_name=args.local_data_path or args.dataset_name,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        cache_dir=args.output_dir,
    )

    train_dataset = TokenizedDataset(data["train"], args.max_seq_len)
    val_dataset = TokenizedDataset(data["val"], args.max_seq_len)

    # 9. Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="no",
        evaluation_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"\n{'=' * 60}")
    print(f"TRAINING START")
    print(f"  Model:     {args.model_name}")
    print(f"  RoPE:      EVQ-cosh τ={args.tau}")
    print(f"  LoRA:      r={args.lora_r}, α={args.lora_alpha}")
    print(f"  Steps:     {args.max_steps}")
    print(f"  Seq len:   {args.max_seq_len}")
    print(f"  Data:      {args.dataset_name} ({len(train_dataset)} samples)")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    print(f"\n[DONE] Training completed in {train_time/3600:.2f} hours")

    # 10. Save
    print("[SAVE] Saving adapter + custom inv_freq...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save experiment metadata
    meta = {
        "model": args.model_name,
        "rope_method": "evq_cosh",
        "tau": args.tau,
        "rope_base": args.rope_base,
        "head_dim": args.head_dim,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_targets": target_modules,
        "max_steps": args.max_steps,
        "max_seq_len": args.max_seq_len,
        "dataset": args.dataset_name,
        "train_samples": len(train_dataset),
        "train_time_hours": round(train_time / 3600, 3),
        "theory_checks": theory_checks,
        "train_loss_final": trainer.state.log_history[-1].get("train_loss")
                           if trainer.state.log_history else None,
    }
    meta_path = os.path.join(args.output_dir, "experiment_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[SAVE] Metadata saved to {meta_path}")
    print(f"[SAVE] Adapter saved to {args.output_dir}")
    print(f"\n✅ Training complete. Next: run eval_evq_lora.py --adapter_dir {args.output_dir}")


if __name__ == "__main__":
    main()
