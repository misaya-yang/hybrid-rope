#!/usr/bin/env python3
"""
Definitive LoRA training pipeline for Hybrid-RoPE NeurIPS experiments.

Two modes are supported via `--attention_mode`:
1) static: CE-only fallback with Anchored-Sigmoid RoPE injection.
2) dynamic_penalty: CE + phase-collision regularization from attention maps.

Design goals:
- Keep base model locked to Meta-Llama-3-8B-Instruct (8K native).
- Inject RoPE frequencies strictly through rotary_emb.inv_freq.copy_().
- Use TRL SFTTrainer for robust supervised fine-tuning.
- Log CE loss, attention penalty, and LR to Weights & Biases.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from trl import SFTTrainer as _SFTTrainer

    BaseTrainer = _SFTTrainer
    HAS_TRL = True
except Exception:  # pragma: no cover - fallback when TRL is unavailable
    BaseTrainer = Trainer
    HAS_TRL = False

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency at runtime
    wandb = None


MODEL_LOCK_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_LOCK_MAX_POS = 8192
MODEL_LOCK_THETA = 500000.0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_model_lock(base_model: str) -> Dict[str, object]:
    low = base_model.lower()
    if "meta-llama-3-8b-instruct" not in low and "meta-llama/Meta-Llama-3-8B-Instruct".lower() not in low:
        raise RuntimeError(
            f"Model lock violation: require Meta-Llama-3-8B-Instruct, got: {base_model}"
        )
    if "3.1" in low or "128k" in low:
        raise RuntimeError("Model lock violation: Llama-3.1 or 128K model is forbidden.")

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    max_pos = int(getattr(cfg, "max_position_embeddings", -1))
    theta = float(getattr(cfg, "rope_theta", MODEL_LOCK_THETA))
    if max_pos != MODEL_LOCK_MAX_POS:
        raise RuntimeError(
            f"Model lock violation: max_position_embeddings must be {MODEL_LOCK_MAX_POS}, got {max_pos}"
        )
    if abs(theta - MODEL_LOCK_THETA) > 1e-6:
        raise RuntimeError(
            f"Theta lock violation: rope_theta must be {MODEL_LOCK_THETA}, got {theta}"
        )
    return {
        "base_model": base_model,
        "max_position_embeddings": max_pos,
        "rope_theta": theta,
    }


def _read_json_or_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _normalize_instruction_record(obj: Dict) -> Optional[Tuple[str, str]]:
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        user_text = ""
        assistant_text = ""
        for m in msgs:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role == "user" and content and not user_text:
                user_text = content
            elif role == "assistant" and content and not assistant_text:
                assistant_text = content
        if user_text and assistant_text:
            return user_text, assistant_text

    instruction = str(obj.get("instruction", "")).strip()
    inp = str(obj.get("input", "")).strip()
    output = str(obj.get("output", obj.get("response", obj.get("answer", "")))).strip()
    if instruction and output:
        user = instruction if not inp else f"{instruction}\n\n{inp}"
        return user, output

    question = str(obj.get("question", "")).strip()
    context = str(obj.get("context", obj.get("document", ""))).strip()
    answer = str(obj.get("answer", "")).strip()
    if question and answer:
        user = question if not context else f"{context}\n\nQuestion: {question}"
        return user, answer
    return None


def _synthetic_retrieval_example(rng: random.Random, long_tokens: int) -> Tuple[str, str]:
    vocab = [
        "context",
        "signal",
        "attention",
        "query",
        "document",
        "retrieval",
        "evidence",
        "hybrid",
        "variational",
        "frequency",
        "resonance",
    ]
    filler = " ".join(vocab[rng.randrange(len(vocab))] for _ in range(long_tokens))
    key = f"KEY-{rng.randint(100000, 999999)}"
    needle = f"The exact retrieval key is {key}."
    depth = rng.randint(max(64, long_tokens // 10), max(128, (long_tokens * 8) // 10))
    prefix = filler[: min(len(filler), depth * 6)]
    suffix = filler[min(len(filler), depth * 6) :]
    user = (
        f"{prefix}\n\n{needle}\n\n{suffix}\n\n"
        "Question: What is the exact retrieval key? Return only the key."
    )
    assistant = key
    return user, assistant


def _synthetic_instruction_example(rng: random.Random) -> Tuple[str, str]:
    prompts = [
        "Summarize why controlled protocol matters in one sentence.",
        "Explain overfitting risk in low-step LoRA tuning in two sentences.",
        "Give a concise checklist for reproducible long-context experiments.",
    ]
    answers = [
        "Controlled protocol isolates the causal effect of schedule choices from confounders.",
        "Low-step LoRA can overfit formatting artifacts; validate with held-out tasks and fixed decoding.",
        "Lock model/version, fix prompt/decode settings, save manifests/hashes, and report uncertainty.",
    ]
    idx = rng.randrange(len(prompts))
    return prompts[idx], answers[idx]


def _render_chat(tokenizer: AutoTokenizer, user: str, assistant: str) -> str:
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return str(text).strip()


def load_long_short_mixture(
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    rng = random.Random(args.seed)
    long_pairs: List[Tuple[str, str]] = []
    short_pairs: List[Tuple[str, str]] = []

    # Long-context source
    if args.long_data_path:
        p = Path(args.long_data_path)
        rows = _read_json_or_jsonl(p)
        for row in rows:
            pair = _normalize_instruction_record(row)
            if pair:
                long_pairs.append(pair)
    elif args.long_hf_dataset:
        ds = load_dataset(args.long_hf_dataset, split=args.long_hf_split)
        for row in ds:
            pair = _normalize_instruction_record(dict(row))
            if pair:
                long_pairs.append(pair)

    # Short-context source
    if args.short_data_path:
        p = Path(args.short_data_path)
        rows = _read_json_or_jsonl(p)
        for row in rows:
            pair = _normalize_instruction_record(row)
            if pair:
                short_pairs.append(pair)
    elif args.short_hf_dataset:
        ds = load_dataset(args.short_hf_dataset, split=args.short_hf_split)
        for row in ds:
            pair = _normalize_instruction_record(dict(row))
            if pair:
                short_pairs.append(pair)

    # Robust fallback to synthetic examples to keep script executable in constrained environments.
    while len(long_pairs) < args.min_long_samples:
        long_pairs.append(_synthetic_retrieval_example(rng, long_tokens=rng.randint(2500, 5200)))
    while len(short_pairs) < args.min_short_samples:
        short_pairs.append(_synthetic_instruction_example(rng))

    rng.shuffle(long_pairs)
    rng.shuffle(short_pairs)

    n_total = min(args.max_total_samples, len(long_pairs) + len(short_pairs))
    n_long = int(round(n_total * args.long_ratio))
    n_short = n_total - n_long
    n_long = min(n_long, len(long_pairs))
    n_short = min(n_short, len(short_pairs))

    selected = long_pairs[:n_long] + short_pairs[:n_short]
    rng.shuffle(selected)
    rendered = [_render_chat(tokenizer, user=u, assistant=a) for u, a in selected]
    rendered = [t for t in rendered if t]

    ds = Dataset.from_dict({"text": rendered}).shuffle(seed=args.seed)
    n = len(ds)
    n_val = max(32, int(n * args.val_ratio))
    n_train = max(1, n - n_val)
    train_ds = ds.select(range(0, n_train))
    val_ds = ds.select(range(n_train, n))
    return train_ds, val_ds, {"long_samples": n_long, "short_samples": n_short, "total_samples": n}


def find_rotary_inv_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if hasattr(module, "inv_freq") and isinstance(getattr(module, "inv_freq"), torch.Tensor):
            modules.append(module)
    return modules


def build_anchored_inv_freq(head_dim: int, base: float, anchor_factor: float, slope_raw: float, center_ratio: float) -> torch.Tensor:
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float32)
    base_inv = 1.0 / (float(base) ** (idx / float(n)))
    slope = float(slope_raw) / float(head_dim)
    center = float(center_ratio) * float(n)
    sig = torch.sigmoid(slope * (idx - center))
    # Anchored-Sigmoid: higher-frequency channels keep more resolution under extension.
    return base_inv / (1.0 + (float(anchor_factor) - 1.0) * sig)


def inject_anchored_inv_freq(model: torch.nn.Module, base: float, anchor_factor: float, slope_raw: float, center_ratio: float) -> Dict[str, object]:
    modules = find_rotary_inv_modules(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found for strict copy_ injection.")
    ref = modules[0].inv_freq
    inv = build_anchored_inv_freq(
        head_dim=int(ref.numel()) * 2,
        base=float(base),
        anchor_factor=float(anchor_factor),
        slope_raw=float(slope_raw),
        center_ratio=float(center_ratio),
    ).to(dtype=ref.dtype)
    with torch.no_grad():
        for m in modules:
            m.inv_freq.copy_(inv.to(device=m.inv_freq.device, dtype=m.inv_freq.dtype))
    return {
        "module_count": len(modules),
        "inv_numel": int(inv.numel()),
        "inv_mean": float(inv.mean().item()),
        "inv_std": float(inv.std().item()),
    }


@dataclass
class PenaltyCache:
    s2: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor]

    def __init__(self) -> None:
        self.s2 = {}

    def get_s2(self, q: int, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(q), int(k), str(device), dtype)
        if key in self.s2:
            return self.s2[key]
        q_idx = torch.arange(q, device=device)
        k_idx = torch.arange(k, device=device)
        dist = (q_idx[:, None] - k_idx[None, :]).abs().to(torch.float32)
        # TODO(theory): replace this exponential toy kernel with exact Anchored-Sigmoid S^2(Delta).
        s_sq = torch.exp(-0.01 * dist).to(dtype=dtype)
        self.s2[key] = s_sq
        return s_sq


class HybridRopeTrainer(BaseTrainer):
    def __init__(self, *args, attention_mode: str, lambda_weight: float, penalty_layer_stride: int, penalty_head_max: int, penalty_query_stride: int, penalty_key_stride: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mode = str(attention_mode)
        self.lambda_weight = float(lambda_weight)
        self.penalty_layer_stride = max(1, int(penalty_layer_stride))
        self.penalty_head_max = max(1, int(penalty_head_max))
        self.penalty_query_stride = max(1, int(penalty_query_stride))
        self.penalty_key_stride = max(1, int(penalty_key_stride))
        self.penalty_cache = PenaltyCache()
        self.last_attention_penalty = 0.0
        self.oom_skip_count = 0

    def _compute_attention_penalty(self, attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not attentions:
            return torch.tensor(0.0, device=self.model.device)
        penalties: List[torch.Tensor] = []
        for li, att in enumerate(attentions):
            if li % self.penalty_layer_stride != 0:
                continue
            # att shape: [bs, heads, q, k]
            bs, h, q, k = att.shape
            head_take = min(h, self.penalty_head_max)
            att_sel = att[:, :head_take, :: self.penalty_query_stride, :: self.penalty_key_stride]
            q2 = att_sel.shape[-2]
            k2 = att_sel.shape[-1]
            s_sq = self.penalty_cache.get_s2(q2, k2, att_sel.device, att_sel.dtype)
            tau = s_sq.mean(dim=-1, keepdim=True)  # dynamic per-query baseline
            spike = torch.relu(s_sq - tau)  # penalize only abnormal collision spikes
            penalties.append(torch.sum(att_sel * spike.view(1, 1, q2, k2)))
        if not penalties:
            return torch.tensor(0.0, device=attentions[0].device)
        return self.lambda_weight * torch.stack(penalties).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        use_dynamic = self.attention_mode == "dynamic_penalty"
        model_inputs = dict(inputs)
        if use_dynamic:
            model_inputs["output_attentions"] = True

        outputs = model(**model_inputs)
        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        penalty = torch.tensor(0.0, device=ce_loss.device)
        if use_dynamic and getattr(outputs, "attentions", None):
            penalty = self._compute_attention_penalty(outputs.attentions)
        total_loss = ce_loss + penalty
        self.last_attention_penalty = float(penalty.detach().item())
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            self.oom_skip_count += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            self.log({"oom_skipped_batches": float(self.oom_skip_count)})
            return torch.zeros((), device=self.args.device, requires_grad=True)

    def log(self, logs: Dict[str, float]) -> None:  # type: ignore[override]
        logs = dict(logs)
        logs["attention_penalty"] = float(self.last_attention_penalty)
        super().log(logs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hybrid-RoPE LoRA training (static / dynamic_penalty).")
    ap.add_argument("--base_model", type=str, default=MODEL_LOCK_ID)
    ap.add_argument("--output_dir", type=str, default="artifacts/attn_integrated_lora")
    ap.add_argument("--run_name", type=str, default="llama3_8b_attn_integrated")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--attention_mode", type=str, choices=["static", "dynamic_penalty"], default="static")
    ap.add_argument("--lambda_weight", type=float, default=1e-3)

    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--max_seq_length", type=int, default=16384)
    ap.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "eager", "sdpa", "flash_attention_2"])

    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    ap.add_argument("--rope_base", type=float, default=MODEL_LOCK_THETA)
    ap.add_argument("--anchor_factor", type=float, default=4.0)
    ap.add_argument("--slope_raw", type=float, default=20.0)
    ap.add_argument("--center_ratio", type=float, default=0.70)

    # Data mixture (60% long retrieval, 40% short instruction)
    ap.add_argument("--long_data_path", type=str, default="")
    ap.add_argument("--short_data_path", type=str, default="")
    ap.add_argument("--long_hf_dataset", type=str, default="")
    ap.add_argument("--short_hf_dataset", type=str, default="")
    ap.add_argument("--long_hf_split", type=str, default="train")
    ap.add_argument("--short_hf_split", type=str, default="train")
    ap.add_argument("--long_ratio", type=float, default=0.60)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--max_total_samples", type=int, default=12000)
    ap.add_argument("--min_long_samples", type=int, default=2000)
    ap.add_argument("--min_short_samples", type=int, default=1400)

    # Attention-penalty memory controls
    ap.add_argument("--penalty_layer_stride", type=int, default=4)
    ap.add_argument("--penalty_head_max", type=int, default=4)
    ap.add_argument("--penalty_query_stride", type=int, default=16)
    ap.add_argument("--penalty_key_stride", type=int, default=16)

    # Logging
    ap.add_argument("--wandb_project", type=str, default="hybrid-rope-neurips")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--disable_wandb", action="store_true")
    return ap.parse_args()


def tokenize_for_causal_lm(ds: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    def _tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    return ds.map(_tok, batched=True, remove_columns=["text"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_info = assert_model_lock(args.base_model)

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.disable_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=args.run_name,
            config=vars(args),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, data_stats = load_long_short_mixture(tokenizer, args)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation=None if args.attn_implementation == "auto" else args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    rope_stats = inject_anchored_inv_freq(
        model=model,
        base=args.rope_base,
        anchor_factor=args.anchor_factor,
        slope_raw=args.slope_raw,
        center_ratio=args.center_ratio,
    )
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(find_rotary_inv_modules(model)[0].inv_freq.detach().cpu(), artifacts_dir / "custom_inv_freq.pt")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=parse_csv(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs_kwargs = {
        "output_dir": str(run_dir),
        "run_name": args.run_name,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "lr_scheduler_type": "cosine",
        "report_to": [] if args.disable_wandb else ["wandb"],
        "dataloader_num_workers": 2,
        "gradient_checkpointing": True,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        targs_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        targs_kwargs["eval_strategy"] = "steps"
    if "save_strategy" in ta_params:
        targs_kwargs["save_strategy"] = "steps"

    targs = TrainingArguments(**targs_kwargs)

    trainer_kwargs: Dict[str, object] = {
        "model": model,
        "args": targs,
        "attention_mode": args.attention_mode,
        "lambda_weight": args.lambda_weight,
        "penalty_layer_stride": args.penalty_layer_stride,
        "penalty_head_max": args.penalty_head_max,
        "penalty_query_stride": args.penalty_query_stride,
        "penalty_key_stride": args.penalty_key_stride,
    }
    if HAS_TRL:
        trainer_kwargs.update(
            {
                "train_dataset": train_ds,
                "eval_dataset": val_ds,
                "dataset_text_field": "text",
                "max_seq_length": args.max_seq_length,
            }
        )
        # TRL has minor API drift across versions; support both names.
        trainer_kwargs["processing_class"] = tokenizer
    else:
        train_tok = tokenize_for_causal_lm(train_ds, tokenizer, args.max_seq_length)
        val_tok = tokenize_for_causal_lm(val_ds, tokenizer, args.max_seq_length)
        trainer_kwargs.update(
            {
                "train_dataset": train_tok,
                "eval_dataset": val_tok,
                "tokenizer": tokenizer,
                "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            }
        )
    if HAS_TRL:
        try:
            trainer = HybridRopeTrainer(**trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("processing_class", None)
            trainer_kwargs["tokenizer"] = tokenizer
            trainer = HybridRopeTrainer(**trainer_kwargs)
    else:
        trainer = HybridRopeTrainer(**trainer_kwargs)

    result = trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    summary = {
        "timestamp": now(),
        "model_lock": model_info,
        "attention_mode": args.attention_mode,
        "lambda_weight": args.lambda_weight,
        "data_stats": data_stats,
        "rope_stats": rope_stats,
        "custom_inv_freq_path": str(artifacts_dir / "custom_inv_freq.pt"),
        "has_trl": bool(HAS_TRL),
        "train_metrics": dict(result.metrics),
        "output_dir": str(run_dir / "adapter"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] training complete")
    print(f"[DONE] adapter: {run_dir / 'adapter'}")
    print(f"[DONE] summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
