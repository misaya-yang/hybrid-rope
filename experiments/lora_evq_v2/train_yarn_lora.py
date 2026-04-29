#!/usr/bin/env python3
"""
YaRN LoRA Training — 与 EVQ-LoRA 同一套 LoRA config，只换频率方法为 YaRN。
用于 PE baseline comparison 实验。

Usage:
    python train_yarn_lora.py --yarn_factor 2.0 --seed 42 --output_dir ./checkpoints/yarn_s42
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

# Reuse data loading and collator from train_evq_lora
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_evq_lora import load_training_data, TokenizedDataset, PaddingCollator

LOCAL_BASE = Path(__file__).resolve().parent / "local"
DEFAULT_MODEL = os.environ.get(
    "EVQ_LORA_MODEL",
    str(LOCAL_BASE / "models" / "Meta-Llama-3-8B-Instruct"),
)
DEFAULT_TRAIN_DATA = os.environ.get(
    "EVQ_LORA_TRAIN_DATA",
    str(LOCAL_BASE / "data" / "longalign_10k" / "longalign_10k.jsonl"),
)


def parse_args():
    p = argparse.ArgumentParser(description="YaRN LoRA Training")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--yarn_factor", type=float, default=2.0)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=60)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--local_data_path", type=str, default=DEFAULT_TRAIN_DATA)
    p.add_argument("--dataset_name", type=str, default="THUDM/LongAlign-10k")
    p.add_argument("--max_samples", type=int, default=8000)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load model WITH YaRN scaling
    print(f"[MODEL] Loading with YaRN factor={args.yarn_factor}")
    rope_scaling = {
        "type": "yarn",
        "factor": args.yarn_factor,
        "original_max_position_embeddings": 8192,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
        rope_scaling=rope_scaling,
    )

    # 3. LoRA (same config as EVQ experiment)
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LORA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 4. Data (reuse cached tokenized data if available)
    data = load_training_data(
        tokenizer=tokenizer,
        dataset_name=args.local_data_path or args.dataset_name,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        cache_dir=args.output_dir,
    )
    train_dataset = TokenizedDataset(data["train"], args.max_seq_len)
    val_dataset = TokenizedDataset(data["val"], args.max_seq_len)

    # 5. Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
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

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\n{'='*60}")
    print(f"TRAINING: YaRN LoRA (factor={args.yarn_factor})")
    print(f"  LoRA: r={args.lora_r}, α={args.lora_alpha}")
    print(f"  Steps: {args.max_steps}, Seed: {args.seed}")
    print(f"  Data: {len(train_dataset)} samples")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    print(f"\n[DONE] Training completed in {train_time/60:.1f} min")

    # 6. Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    meta = {
        "method": "yarn",
        "yarn_factor": args.yarn_factor,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "train_time_min": round(train_time / 60, 2),
        "model": args.model_name,
    }
    with open(os.path.join(args.output_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] Adapter → {args.output_dir}")


if __name__ == "__main__":
    main()
