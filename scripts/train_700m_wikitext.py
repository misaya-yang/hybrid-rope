#!/usr/bin/env python3
"""
Offline WikiText training entry for 700M checkpoints.

This script is intentionally local-only:
- model/tokenizer must be available on local disk
- dataset must be local text files (train.txt / valid.txt)
- no external HuggingFace network dependency
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
            "Missing local WikiText files: "
            + ", ".join(missing)
            + ". Please sync local dataset first."
        )
    return files


class TextWindowDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        tokenizer: AutoTokenizer,
        seq_length: int,
        stride: int,
        max_samples: int | None = None,
    ) -> None:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 4:
            raise RuntimeError(f"Not enough tokens in {file_path} (got {len(tokens)}).")

        self.seq_length = seq_length
        self.pad_id = tokenizer.pad_token_id
        self.samples: List[List[int]] = []

        max_start = max(1, len(tokens) - 2)
        for start in range(0, max_start, stride):
            end = min(start + seq_length + 1, len(tokens))
            chunk = tokens[start:end]
            if len(chunk) >= 2:
                self.samples.append(chunk)
            if end >= len(tokens):
                break

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No windows built from {file_path}.")

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
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        attention_mask = [1] * real_len + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LossPrinter(TrainerCallback):
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


class StabilityGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state.global_step}: {loss}")
        if loss > 100:
            raise RuntimeError(f"Abnormal loss at step {state.global_step}: {loss}")


def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_eval_loss_ppl(model, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
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

        token_count = int((labels != -100).sum().item())
        total_loss += loss * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No valid eval tokens.")

    avg_loss = total_loss / total_tokens
    return avg_loss, safe_exp(avg_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline WikiText training for local checkpoints.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/train_freq_comparison/700m_orig_20260214_140024/model",
        help="Local model path.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/wikitext_data",
        help="Local data dir with train.txt / valid.txt.",
    )
    parser.add_argument("--output_dir", type=str, default="./results/train_700m_wikitext")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_non_improving", action="store_true")
    args = parser.parse_args()

    enforce_offline_mode()
    set_seed(args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    files = resolve_wikitext_files(Path(args.data_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Data path: {args.data_dir}")

    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_ds = TextWindowDataset(
        files["train"],
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
        max_samples=args.max_train_samples,
    )
    eval_ds = TextWindowDataset(
        files["valid"],
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
        max_samples=args.max_eval_samples,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples: {len(eval_ds)}")

    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=default_data_collator)
    pre_loss, pre_ppl = compute_eval_loss_ppl(model, eval_loader, device)
    print(f"Pre-training: loss={pre_loss:.6f}, ppl={pre_ppl:.3f}")

    targs = TrainingArguments(
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
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[LossPrinter(log_file), StabilityGuard()],
    )
    train_result = trainer.train()

    post_loss, post_ppl = compute_eval_loss_ppl(model, eval_loader, device)
    print(f"Post-training: loss={post_loss:.6f}, ppl={post_ppl:.3f}")
    improved = bool(post_loss < pre_loss and math.isfinite(post_loss))

    metrics = train_result.metrics if isinstance(train_result.metrics, dict) else {}
    out = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "data_dir": args.data_dir,
        "seq_length": args.seq_length,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "pre_training": {"loss": pre_loss, "ppl": pre_ppl},
        "post_training": {"loss": post_loss, "ppl": post_ppl},
        "improvement": {
            "loss_drop": pre_loss - post_loss,
            "loss_drop_percent": (pre_loss - post_loss) / pre_loss * 100 if pre_loss > 0 else None,
            "ppl_ratio": pre_ppl / post_ppl if post_ppl > 0 and math.isfinite(post_ppl) else None,
            "improved": improved,
        },
        "trainer": {
            "train_runtime": metrics.get("train_runtime"),
            "train_loss": metrics.get("train_loss"),
            "train_steps_per_second": metrics.get("train_steps_per_second"),
        },
    }
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    model.save_pretrained(run_dir / "final_model")
    tokenizer.save_pretrained(run_dir / "final_model")
    print(f"Saved to: {run_dir / 'final_model'}")

    if not improved and not args.allow_non_improving:
        raise RuntimeError(
            "Training completed but post-training loss did not drop. "
            "Use --allow_non_improving to bypass this guard."
        )


if __name__ == "__main__":
    main()

