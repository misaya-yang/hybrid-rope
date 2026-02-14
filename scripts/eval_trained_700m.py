#!/usr/bin/env python3
"""
Offline evaluation for trained 700M models on local WikiText text files.

Features:
- local model loading only
- optional RoPE frequency patch (orig/geometric/hybrid/sigmoid)
- sequence-length sweep and robust PPL computation
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def compute_freq_curve(freq_type: str, head_dim: int, theta_base: float = 10000, **kwargs) -> torch.Tensor:
    dim = head_dim // 2
    if dim <= 0:
        raise ValueError(f"Invalid head_dim: {head_dim}")

    idx = torch.arange(0, dim, dtype=torch.float32)
    t = idx / max(1, dim - 1)

    if freq_type == "orig":
        return 1.0 / (theta_base ** (2.0 * idx / head_dim))
    if freq_type == "geometric":
        scale = float(kwargs.get("scale", 1.0))
        return 1.0 / ((theta_base * scale) ** (2.0 * idx / head_dim))
    if freq_type == "hybrid":
        alpha = float(kwargs.get("alpha", 0.2))
        p = float(kwargs.get("p", 3.9))
        omf = float(kwargs.get("omf", 0.3))
        poly = torch.pow(t + 1e-8, p)
        mixed = (1 - alpha) * t + alpha * poly
        return omf / (theta_base ** mixed)
    if freq_type == "sigmoid":
        steepness = float(kwargs.get("steepness", 8.0))
        midpoint = float(kwargs.get("midpoint", 0.5))
        omf = float(kwargs.get("omf", 0.3))
        sigmoid_t = torch.sigmoid(steepness * (t - midpoint))
        return omf / (theta_base ** sigmoid_t)
    raise ValueError(f"Unknown freq_type: {freq_type}")


def patch_rope_freq(model, freq_type: str, **kwargs) -> List[str]:
    patched: List[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and ("rotary_emb" in name or name.endswith(".rotary_emb")):
            head_dim = int(module.inv_freq.shape[0]) * 2
            new_inv_freq = compute_freq_curve(freq_type, head_dim, **kwargs).to(
                device=module.inv_freq.device, dtype=module.inv_freq.dtype
            )
            if isinstance(module.inv_freq, nn.Parameter):
                module.inv_freq = nn.Parameter(new_inv_freq, requires_grad=False)
            else:
                module.inv_freq = new_inv_freq
            patched.append(name)
    return patched


def load_eval_tokens(data_file: Path, tokenizer: AutoTokenizer) -> List[int]:
    if not data_file.exists():
        raise FileNotFoundError(f"Local eval file not found: {data_file}")
    text = data_file.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 8:
        raise RuntimeError(f"Too few tokens in {data_file}: {len(tokens)}")
    return tokens


class EvalWindowDataset(Dataset):
    def __init__(self, tokens: List[int], seq_length: int, stride: int, max_samples: int | None = None, pad_id: int = 0):
        self.seq_length = seq_length
        self.pad_id = pad_id
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
            raise RuntimeError("No eval windows built.")

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


@torch.no_grad()
def evaluate_ppl(model, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
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
            raise RuntimeError("Non-finite loss detected in eval.")

        token_count = int((labels != -100).sum().item())
        total_loss += loss * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No valid eval tokens.")

    avg_loss = total_loss / total_tokens
    return avg_loss, safe_exp(avg_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained 700M model offline.")
    parser.add_argument("--model_path", type=str, required=True, help="Local trained model path.")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/wikitext_data")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test", "train"])
    parser.add_argument("--freq_type", type=str, default="orig", choices=["orig", "geometric", "hybrid", "sigmoid"])
    parser.add_argument("--seq_lengths", type=str, default="2048,4096,8192,16384")
    parser.add_argument("--stride_ratio", type=float, default=1.0, help="Stride = max(1, int(seq_length * stride_ratio)).")
    parser.add_argument("--max_samples", type=int, default=64, help="Per-length max windows.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results/eval_700m")

    parser.add_argument("--theta_base", type=float, default=100000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=3.9)
    parser.add_argument("--omf", type=float, default=0.3)
    parser.add_argument("--steepness", type=float, default=8.0)
    parser.add_argument("--midpoint", type=float, default=0.5)
    args = parser.parse_args()

    enforce_offline_mode()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    seq_lengths = sorted({int(x.strip()) for x in args.seq_lengths.split(",") if x.strip()})
    if not seq_lengths:
        raise ValueError("No valid sequence lengths.")

    run_dir = Path(args.output_dir) / f"eval_{args.freq_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
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

    freq_kwargs = {
        "theta_base": args.theta_base,
        "alpha": args.alpha,
        "p": args.p,
        "omf": args.omf,
        "steepness": args.steepness,
        "midpoint": args.midpoint,
    }
    patched_layers: List[str] = []
    if args.freq_type != "orig":
        patched_layers = patch_rope_freq(model, args.freq_type, **freq_kwargs)
        print(f"Patched rotary layers: {len(patched_layers)}")

    data_file = Path(args.data_dir) / f"{args.split}.txt"
    tokens = load_eval_tokens(data_file, tokenizer)
    print(f"Loaded eval tokens from {data_file}: {len(tokens)}")

    results: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "data_file": str(data_file),
        "freq_type": args.freq_type,
        "freq_kwargs": freq_kwargs,
        "patched_layers": patched_layers,
        "seq_lengths": {},
    }

    for seq_len in seq_lengths:
        stride = max(1, int(seq_len * args.stride_ratio))
        print(f"\n--- Evaluating length={seq_len}, stride={stride} ---")
        dataset = EvalWindowDataset(
            tokens=tokens,
            seq_length=seq_len,
            stride=stride,
            max_samples=args.max_samples,
            pad_id=tokenizer.pad_token_id,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        loss, ppl = evaluate_ppl(model, dataloader, device)
        results["seq_lengths"][str(seq_len)] = {
            "loss": loss,
            "ppl": ppl,
            "num_samples": len(dataset),
            "stride": stride,
        }
        print(f"loss={loss:.6f}, ppl={ppl:.3f}, samples={len(dataset)}")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved eval results to: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()

