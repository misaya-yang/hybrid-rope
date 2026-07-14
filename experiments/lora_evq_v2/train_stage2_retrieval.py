#!/usr/bin/env python3
"""
Stage 2: Teach matched Geo/EVQ LoRA adapters retrieval skills.

Loads the 300-step EVQ-LoRA adapter and continues training for ~50 steps
on a mix of retrieval data (gen_retrieval_mix.py output) + original LongAlpaca
to prevent catastrophic forgetting.

Key differences from stage 1:
  - Lower learning rate (2e-5 vs 1e-4) to preserve existing adaptation
  - Short training (50 steps)
  - Explicit substrate identity for matched Geo/EVQ continuation
  - Assistant-answer-only labels instead of prompt-dominated full-token loss

Usage:
    # Generate retrieval data first
    python gen_retrieval_mix.py --output retrieval_mix.jsonl --n_samples 500

    # Stage 2 training
    python train_stage2_retrieval.py \
        --adapter_dir ./checkpoints/evq_r64 \
        --substrate evq_cosh \
        --retrieval_data retrieval_mix.jsonl \
        --output_dir ./checkpoints/evq_r64_stage2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_evq_lora import (
    build_training_inv_freq,
    evaluation_strategy_kwargs,
    inject_inv_freq,
    load_frequency_artifact,
    public_artifact_identifier,
    public_model_identifier,
    resolve_model_rope_geometry,
    sha256_file,
    verify_model_inv_freq,
)


def load_and_merge_data(
    tokenizer,
    retrieval_path: str,
    original_data_path: Optional[str],
    max_seq_len: int = 8192,
    retrieval_ratio: float = 0.7,
    max_total: int = 1000,
) -> Dict[str, List]:
    """Load retrieval + original data, merge with specified ratio."""

    def tokenize_messages(messages_list, label: str):
        def chat_ids(messages, *, add_generation_prompt: bool):
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            if hasattr(encoded, "keys"):
                encoded = encoded["input_ids"]
            return [int(token_id) for token_id in encoded]

        tokenized = []
        for msgs in messages_list:
            if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                raise ValueError(f"{label} row does not end with an assistant answer")
            prompt_ids = chat_ids(msgs[:-1], add_generation_prompt=True)
            full_ids = chat_ids(msgs, add_generation_prompt=False)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"{label} chat template does not preserve the answer boundary")
            answer_start = len(prompt_ids)
            if len(full_ids) > max_seq_len:
                trim = len(full_ids) - max_seq_len
                if trim >= answer_start:
                    continue
                full_ids = full_ids[trim:]
                answer_start -= trim
            if len(full_ids) >= 64 and answer_start < len(full_ids):
                tokenized.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": [-100] * answer_start + full_ids[answer_start:],
                })
        print(f"[DATA] {label}: {len(tokenized)} samples tokenized")
        return tokenized

    # 1. Load retrieval data
    print(f"[DATA] Loading retrieval data: {retrieval_path}")
    retrieval_messages = []
    with open(retrieval_path) as f:
        for line in f:
            item = json.loads(line)
            if "messages" in item:
                retrieval_messages.append(item["messages"])
    retrieval_tok = tokenize_messages(retrieval_messages, "Retrieval")

    # 2. Load original data (for anti-forgetting)
    original_tok = []
    if original_data_path and os.path.exists(original_data_path):
        from datasets import load_dataset

        print(f"[DATA] Loading original data: {original_data_path}")
        if original_data_path.endswith((".jsonl", ".json")):
            ds = load_dataset("json", data_files=original_data_path, split="train")
        else:
            ds = load_dataset(original_data_path, split="train", trust_remote_code=True)

        orig_messages = []
        for item in ds:
            if "messages" in item and item["messages"]:
                orig_messages.append(item["messages"])
            elif "instruction" in item:
                user_text = item.get("instruction", "")
                if item.get("input"):
                    user_text = f"{user_text}\n\n{item['input']}"
                orig_messages.append([
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": item.get("output", "")},
                ])
            if len(orig_messages) >= 2000:
                break
        original_tok = tokenize_messages(orig_messages, "Original")

    # 3. Merge with ratio
    n_retrieval = int(max_total * retrieval_ratio)
    n_original = max_total - n_retrieval

    random.seed(42)
    if len(retrieval_tok) > n_retrieval:
        retrieval_tok = random.sample(retrieval_tok, n_retrieval)
    if len(original_tok) > n_original:
        original_tok = random.sample(original_tok, n_original)

    merged = retrieval_tok + original_tok
    random.shuffle(merged)

    # Split val
    n_val = max(1, int(len(merged) * 0.02))
    val_data = merged[:n_val]
    train_data = merged[n_val:]

    print(f"[DATA] Final mix: {len(retrieval_tok)} retrieval + {len(original_tok)} original = {len(merged)} total")
    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")

    return {"train": train_data, "val": val_data}


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"][:self.max_seq_len]
        attention_mask = item["attention_mask"][:self.max_seq_len]
        labels = item["labels"][:self.max_seq_len]
        return {
            "input_ids": list(input_ids),
            "attention_mask": list(attention_mask),
            "labels": list(labels),
        }


class PaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Retrieval adaptation")

    p.add_argument("--adapter_dir", type=str, required=True,
                    help="Path to stage1 EVQ-LoRA adapter (300 steps)")
    p.add_argument("--substrate", choices=("native_geo", "evq_cosh"), required=True)
    p.add_argument("--model_name", type=str,
                    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--output_dir", type=str, required=True)

    # Data
    p.add_argument("--retrieval_data", type=str, required=True,
                    help="JSONL from gen_retrieval_mix.py")
    p.add_argument("--original_data", type=str, default=None,
                    help="Original LongAlpaca JSONL for anti-forgetting")
    p.add_argument("--retrieval_ratio", type=float, default=0.7,
                    help="Ratio of retrieval vs original data")

    # Training — conservative to preserve stage1
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Lower LR than stage1 to preserve adaptation")
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=8192)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    if args.max_steps != 50:
        raise ValueError("registered stage2 comparison requires exactly 50 optimizer steps")
    torch.manual_seed(args.seed)

    base_meta_path = Path(args.adapter_dir) / "experiment_meta.json"
    if not base_meta_path.is_file():
        raise FileNotFoundError(base_meta_path)
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    if base_meta.get("status") != "complete" or int(base_meta.get("global_step", -1)) != 300:
        raise ValueError("stage2 source must be a completed step-300 adapter")
    if base_meta.get("protocol", {}).get("method") != args.substrate:
        raise ValueError("stage2 source adapter substrate mismatch")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import PeftModel

    # 1. Load tokenizer
    print(f"[MODEL] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Materialize the deterministic data mix before reserving the GPU.
    data = load_and_merge_data(
        tokenizer=tokenizer,
        retrieval_path=args.retrieval_data,
        original_data_path=args.original_data,
        max_seq_len=args.max_seq_len,
        retrieval_ratio=args.retrieval_ratio,
    )
    train_dataset = TokenizedDataset(data["train"], args.max_seq_len)
    val_dataset = TokenizedDataset(data["val"], args.max_seq_len)

    # 3. Load base model + stage1 adapter
    print(f"[MODEL] Loading base model (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Load the registered inv_freq from stage1
    freq_path = os.path.join(args.adapter_dir, "custom_inv_freq.pt")
    inv_freq, _, frequency_provenance = load_frequency_artifact(
        freq_path,
        expected_method=args.substrate,
    )
    geometry = resolve_model_rope_geometry(model.config)
    canonical, _ = build_training_inv_freq(
        rope_method=args.substrate,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    if not torch.allclose(inv_freq.to(torch.float64), canonical, rtol=0.0, atol=1e-12):
        raise ValueError("stage1 frequency artifact differs from the canonical substrate")
    result = inject_inv_freq(model, inv_freq)
    verification = verify_model_inv_freq(model, inv_freq)
    print(
        f"[ROPE] Injected {args.substrate} frequencies into {result['patched_count']} modules; "
        f"verified={verification['verified_count']}"
    )

    # 5. Load stage1 LoRA adapter
    print(f"[LORA] Loading stage1 adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=True)
    verify_model_inv_freq(model, inv_freq)

    # Make all LoRA params trainable for continued training
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LORA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 6. Train
    training_kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "logging_steps": 5,
        "save_strategy": "no",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
    }
    training_kwargs.update(evaluation_strategy_kwargs(TrainingArguments))
    training_args = TrainingArguments(**training_kwargs)

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"\n{'='*60}")
    print(f"STAGE 2: RETRIEVAL ADAPTATION")
    print(f"  Base adapter: {args.adapter_dir}")
    print(f"  Steps:        {args.max_steps}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Data mix:     {args.retrieval_ratio*100:.0f}% retrieval + {(1-args.retrieval_ratio)*100:.0f}% original")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    print(f"\n[DONE] Stage 2 completed in {train_time/60:.1f} minutes")

    # 7. Save
    print("[SAVE] Saving stage2 adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Copy inv_freq from stage1
    import shutil
    shutil.copy2(freq_path, os.path.join(args.output_dir, "custom_inv_freq.pt"))

    meta = {
        "schema": "evq_cosh.lora_stage2_retrieval.v1",
        "status": "complete",
        "stage": 2,
        "substrate": args.substrate,
        "model": public_model_identifier(args.model_name),
        "base_adapter": public_artifact_identifier(args.adapter_dir),
        "base_adapter_sha256": sha256_file(os.path.join(args.adapter_dir, "adapter_model.safetensors")),
        "adapter_sha256": sha256_file(os.path.join(args.output_dir, "adapter_model.safetensors")),
        "model_manifest_sha256": base_meta.get("model_manifest_sha256"),
        "training_manifest_sha256": base_meta.get("data_manifest_sha256"),
        "frequency_provenance": frequency_provenance,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "retrieval_ratio": args.retrieval_ratio,
        "seed": args.seed,
        "objective": "answer_only_chat_causal_lm",
        "retrieval_data_sha256": sha256_file(args.retrieval_data),
        "original_data_sha256": sha256_file(args.original_data) if args.original_data else None,
        "train_samples": len(train_dataset),
        "train_time_min": round(train_time / 60, 2),
    }
    with open(os.path.join(args.output_dir, "stage2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] Stage2 adapter → {args.output_dir}")
    print(f"\n✅ Done. Next: run eval with --adapter_dir {args.output_dir}")


if __name__ == "__main__":
    main()
