#!/usr/bin/env python3
"""
Offline local training for a 700M checkpoint.

Key guarantees:
- No HuggingFace external network calls.
- Uses only local model + local WikiText text files.
- Enforces numeric stability checks during training.
- Verifies post-training loss drops (unless explicitly disabled).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)


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


def resolve_wikitext_files(data_dir: Path) -> Dict[str, Path]:
    files = {
        "train": data_dir / "train.txt",
        "valid": data_dir / "valid.txt",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing local dataset files: "
            + ", ".join(missing)
            + ". Please sync local WikiText files first."
        )
    return files


class LocalTextDataset(Dataset):
    def __init__(
        self,
        filepath: Path,
        tokenizer: AutoTokenizer,
        seq_length: int = 2048,
        stride: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else max(1, seq_length // 2)

        text = filepath.read_text(encoding="utf-8", errors="ignore")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 4:
            raise RuntimeError(f"Not enough tokens in {filepath} (got {len(tokens)}).")

        self.samples: List[List[int]] = []
        max_start = max(1, len(tokens) - 2)
        for start in range(0, max_start, self.stride):
            end = min(start + seq_length + 1, len(tokens))
            chunk = tokens[start:end]
            if len(chunk) >= 2:
                self.samples.append(chunk)
            if end >= len(tokens):
                break

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(f"No training windows built from {filepath}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        input_ids = chunk[:-1]
        labels = chunk[1:]
        real_len = len(input_ids)

        if real_len > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            labels = labels[: self.seq_length]
            real_len = self.seq_length

        pad_len = self.seq_length - real_len
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        attention_mask = [1] * real_len + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LossPrintingCallback(TrainerCallback):
    def __init__(self, log_file: Path | None = None) -> None:
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        msg = (
            f"[Step {state.global_step}] "
            f"loss={float(logs['loss']):.6f}, "
            f"lr={float(logs.get('learning_rate', 0.0)):.2e}"
        )
        print(msg, flush=True)
        if self.log_file is not None:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")


class StabilityGuardCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at step {state.global_step}: {loss}")
        if loss > 100:
            raise RuntimeError(f"Abnormally large loss at step {state.global_step}: {loss}")


def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_ppl(model, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(outputs.loss.item())
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite eval loss detected.")

        num_tokens = int((labels != -100).sum().item())
        total_loss += loss * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        raise RuntimeError("No valid eval tokens found.")

    avg_loss = total_loss / total_tokens
    ppl = safe_exp(avg_loss)
    return avg_loss, ppl


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline local training for 700M model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/train_freq_comparison/700m_orig_20260214_140024/model",
        help="Local model path only.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/wikitext_data",
        help="Directory that contains train.txt / valid.txt.",
    )
    parser.add_argument("--output_dir", type=str, default="./results/train_700m_local")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=128)

    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_non_improving",
        action="store_true",
        help="Do not fail when post_loss >= pre_loss.",
    )
    args = parser.parse_args()

    enforce_offline_mode()
    set_seed(args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")

    data_files = resolve_wikitext_files(Path(args.data_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Data dir: {args.data_dir}")

    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    train_dataset = LocalTextDataset(
        filepath=data_files["train"],
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
        max_samples=args.max_train_samples,
    )
    eval_dataset = LocalTextDataset(
        filepath=data_files["valid"],
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
        max_samples=args.max_eval_samples,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    eval_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, collate_fn=default_data_collator
    )
    pre_loss, pre_ppl = compute_ppl(model, eval_loader, device)
    print(f"Pre-training: loss={pre_loss:.6f}, ppl={pre_ppl:.3f}")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[LossPrintingCallback(log_file), StabilityGuardCallback()],
    )

    train_result = trainer.train()
    post_loss, post_ppl = compute_ppl(model, eval_loader, device)
    print(f"Post-training: loss={post_loss:.6f}, ppl={post_ppl:.3f}")

    improved = bool(post_loss < pre_loss and math.isfinite(post_loss))
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "data_dir": args.data_dir,
        "total_params": total_params,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "max_steps": args.max_steps,
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "pre_training": {"loss": pre_loss, "ppl": pre_ppl},
        "post_training": {"loss": post_loss, "ppl": post_ppl},
        "improvement": {
            "loss_drop": pre_loss - post_loss,
            "loss_drop_percent": (pre_loss - post_loss) / pre_loss * 100 if pre_loss > 0 else None,
            "ppl_ratio": pre_ppl / post_ppl if post_ppl > 0 and math.isfinite(post_ppl) else None,
            "improved": improved,
        },
        "trainer": {
            "train_runtime": getattr(train_result.metrics, "get", lambda *_: None)("train_runtime"),
            "train_loss": getattr(train_result.metrics, "get", lambda *_: None)("train_loss"),
            "train_steps_per_second": getattr(train_result.metrics, "get", lambda *_: None)(
                "train_steps_per_second"
            ),
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model.save_pretrained(run_dir / "final_model")
    tokenizer.save_pretrained(run_dir / "final_model")
    print(f"Saved model to: {run_dir / 'final_model'}")

    if not improved and not args.allow_non_improving:
        raise RuntimeError(
            "Training finished but post-training loss did not improve. "
            "Use --allow_non_improving to bypass this guard."
        )


if __name__ == "__main__":
    main()

