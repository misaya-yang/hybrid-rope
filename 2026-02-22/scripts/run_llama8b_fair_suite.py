#!/usr/bin/env python3
"""
Fair Llama-3-8B-Instruct long-context LoRA training (no forward monkey patch).

Core hard rules implemented:
1) NEVER monkey-patch any forward().
2) NEVER modify model.config.rope_scaling (must stay None).
3) RoPE changes are done ONLY by in-place overwrite of rotary inv_freq buffers.

Methods:
- baseline
- pi
- yarn
- anchored_hybrid

Default hyper-parameters are locked per request:
- max_steps=400
- per_device_train_batch_size=2
- gradient_accumulation_steps=2
- learning_rate=2e-4
- bf16=True
- LoRA target modules: q_proj,k_proj,v_proj,o_proj
- LoRA rank=64, alpha=128
"""

from __future__ import annotations

import argparse
import inspect
import json
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
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


def coerce_chat_template_ids(tokenizer: AutoTokenizer, obj: object) -> List[int]:
    """
    Normalize tokenizer.apply_chat_template outputs across transformers versions.
    Accepts list/tensor/dict/BatchEncoding/string and returns flat List[int].
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], list):
            return [int(x) for x in obj[0]]
        return [int(x) for x in obj]

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 0:
            return []
        if obj.ndim == 2:
            obj = obj[0]
        return [int(x) for x in obj.detach().cpu().tolist()]

    if isinstance(obj, dict) or hasattr(obj, "keys"):
        try:
            input_ids = obj["input_ids"]  # type: ignore[index]
        except Exception:
            input_ids = getattr(obj, "input_ids", None)
        if input_ids is not None:
            return coerce_chat_template_ids(tokenizer, input_ids)

    if isinstance(obj, str):
        return tokenizer.encode(obj, add_special_tokens=False)

    return []


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_targets(v: str) -> List[str]:
    items = [x.strip() for x in v.split(",") if x.strip()]
    if not items:
        raise ValueError("lora_target_modules cannot be empty")
    return items


def build_training_args_compat(raw_kwargs: Dict[str, object]) -> TrainingArguments:
    """
    Build TrainingArguments in a version-tolerant way (transformers 4.x/5.x).
    Drops unsupported kwargs and handles eval/evaluation strategy naming differences.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs = dict(raw_kwargs)
    if "evaluation_strategy" in supported and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in supported and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(k for k in kwargs.keys() if k not in supported)
    if dropped:
        print(f"[compat] dropped unsupported TrainingArguments keys: {dropped}", flush=True)

    return TrainingArguments(**filtered)


def infer_model_rope_base(config: AutoConfig, fallback: float = 10000.0) -> float:
    # transformers <=4.x: config.rope_theta = 500000.0
    # transformers >=5.x: config.rope_scaling = {'rope_theta': 500000.0, 'rope_type': 'default'}
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        # Try extracting from rope_scaling dict (transformers 5.x+)
        rs = getattr(config, "rope_scaling", None)
        if isinstance(rs, dict):
            theta = rs.get("rope_theta", None)
    if theta is None:
        return fallback
    theta_f = safe_float(theta, fallback)
    if theta_f <= 0:
        return fallback
    return theta_f


def rope_scaling_is_effectively_default(obj: object) -> bool:
    """
    Compatibility for transformers variants where default RoPE may be serialized as:
    {"rope_theta": ..., "rope_type": "default"} instead of None.
    transformers 5.x puts rope_theta inside rope_scaling dict by default.
    """
    rope_scaling = getattr(obj, "rope_scaling", None)
    if rope_scaling is None:
        return True
    if not isinstance(rope_scaling, dict):
        return False

    rope_type = str(rope_scaling.get("rope_type", rope_scaling.get("type", ""))).lower()
    allowed_types = {"", "default"}
    if rope_type not in allowed_types:
        return False

    # In transformers 5.x, the default config may include additional keys like
    # 'original_max_position_embeddings', 'attention_factor', etc.
    # We only reject if rope_type is a non-default scaling method.
    return True


def geometric_inv_freq(head_dim: int, base: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    k = head_dim // 2
    idx = torch.arange(k, dtype=dtype)
    return 1.0 / (float(base) ** (2.0 * idx / float(head_dim)))


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


def get_custom_inv_freq(
    method: str,
    head_dim: int = 128,
    base: float = 10000.0,
    max_seq_len: int = 8192,
    rigid_j0: int = 12,
    anchor_factor: float = 0.0,
) -> torch.Tensor:
    """
    Return inv_freq tensor of shape (head_dim // 2,).
    """
    method = method.lower()
    base_inv = geometric_inv_freq(head_dim=head_dim, base=base, dtype=torch.float64)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

    if method == "baseline":
        return base_inv

    scale = float(max_seq_len) / 8192.0
    scale = max(scale, 1.0)

    if method == "pi":
        # Position interpolation: compress positions by scale -> freq / scale.
        return base_inv / scale

    if method == "yarn":
        # YaRN-style progressive ramp: high-frequency core protected, low-frequency tail interpolated.
        # This implementation avoids config.rope_scaling and computes inv_freq directly.
        k = head_dim // 2
        idx = torch.arange(k, dtype=torch.float64)
        start = int(0.20 * k)
        end = int(0.90 * k)
        if end <= start:
            end = min(k - 1, start + 1)
        ramp = (idx - start) / float(max(1, end - start))
        ramp = torch.clamp(ramp, 0.0, 1.0)
        ramp = smoothstep(ramp)

        # Temperature term (mild) to match YaRN's "soft" interpolation flavor.
        temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
        yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
        return base_inv / yarn_scale

    if method == "anchored_hybrid":
        # Rigid high-frequency core (first j0 freq pairs) exactly baseline.
        k = head_dim // 2
        if rigid_j0 < 0:
            raise ValueError(f"rigid_j0 must be >=0, got {rigid_j0}")
        rigid_j0 = min(rigid_j0, k)

        if scale <= 1.0:
            # At native context, keep native frequencies.
            return base_inv

        # tail_base must be STRICTLY greater than base to avoid blend degenerating to identity.
        # For LLaMA-3 (rope_theta=500000), the old `max(base, 500000)` collapsed to base.
        tail_base = float(base) * (scale ** 2)
        tail_base = max(tail_base, float(base) * 4.0)  # safety floor: at least 4x base
        tail_inv = geometric_inv_freq(head_dim=head_dim, base=tail_base, dtype=torch.float64)

        # Smooth blend in mid/low frequencies only.
        out = base_inv.clone()
        if rigid_j0 < k:
            t = torch.arange(k - rigid_j0, dtype=torch.float64)
            if t.numel() == 1:
                ramp = torch.ones_like(t)
            else:
                t = t / float(t.numel() - 1)
                ramp = 0.5 - 0.5 * torch.cos(math.pi * t)

            # alpha<1 and increases mildly with extrapolation ratio.
            alpha = min(0.40, max(0.08, 0.16 * math.log2(scale)))
            blend = alpha * ramp
            out[rigid_j0:] = (1.0 - blend) * base_inv[rigid_j0:] + blend * tail_inv[rigid_j0:]

        # Hard exact overwrite for rigid core (bitwise equal in same dtype).
        out[:rigid_j0] = base_inv[:rigid_j0]
        if rigid_j0 > 0:
            assert torch.equal(out[:rigid_j0], base_inv[:rigid_j0]), "Rigid core is not exact baseline."
        return out

    if method == "sigmoid":
        # Sigmoid-RoPE: remap frequency allocation with normalized sigmoid.
        n = head_dim // 2
        slope = 16.05 / float(head_dim)
        center = 0.47 * float(n)

        idx = torch.arange(n, dtype=torch.float64)
        sigmoid_values = 1.0 / (1.0 + torch.exp(-slope * (idx - center)))
        s_min = sigmoid_values[0]
        s_max = sigmoid_values[-1]
        denom = s_max - s_min
        if torch.abs(denom) < 1e-18:
            raise RuntimeError("sigmoid normalization collapsed (s_max == s_min).")
        s_normalized = (sigmoid_values - s_min) / denom
        inv_freq = 1.0 / (float(base) ** s_normalized)
        return inv_freq

    if method == "anchored_sigmoid":
        # Anchored Sigmoid:
        # theta_i = b^{-2i/d} * [1 + (alpha-1) * sigma(k(i-j0))]
        # inv_freq_i = original_inv / [1 + (alpha-1) * sigma(...)]
        n = head_dim // 2
        slope = 16.05 / float(head_dim)
        j0 = 0.47 * float(n)

        eff_anchor = float(anchor_factor)
        if eff_anchor <= 0:
            eff_anchor = max(2.0, 2.5 * scale)
            eff_anchor = min(eff_anchor, 30.0)

        idx = torch.arange(n, dtype=torch.float64)
        sigmoid_w = 1.0 / (1.0 + torch.exp(-slope * (idx - j0)))
        scale_factor = 1.0 + (eff_anchor - 1.0) * sigmoid_w
        inv_freq = base_inv / scale_factor
        return inv_freq

    raise ValueError(f"Unknown method: {method}")


def find_rotary_modules_with_inv_freq(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    out: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq"):
            inv = getattr(module, "inv_freq")
            if torch.is_tensor(inv):
                out.append((name, module))
    return out


def _clear_rotary_cache(module: torch.nn.Module) -> None:
    # Best-effort cleanup across transformers versions.
    for attr in [
        "max_seq_len_cached",
        "_cos_cached",
        "_sin_cached",
        "cos_cached",
        "sin_cached",
        "_cos_cache",
        "_sin_cache",
    ]:
        if not hasattr(module, attr):
            continue
        try:
            cur = getattr(module, attr)
            if isinstance(cur, (int, float)):
                setattr(module, attr, 0)
            else:
                setattr(module, attr, None)
        except Exception:
            pass


def overwrite_inv_freq_inplace(
    model: torch.nn.Module,
    custom_inv_freq: torch.Tensor,
    baseline_inv_freq: torch.Tensor,
    method: str,
    rigid_j0: int,
) -> Dict[str, object]:
    modules = find_rotary_modules_with_inv_freq(model)
    if not modules:
        raise RuntimeError("No modules with inv_freq found. Cannot apply custom RoPE.")

    patched: List[Dict[str, object]] = []
    for name, module in modules:
        old = module.inv_freq
        if old.ndim != 1:
            raise RuntimeError(f"{name}.inv_freq is not 1D: shape={tuple(old.shape)}")
        if old.shape != custom_inv_freq.shape:
            raise RuntimeError(
                f"Shape mismatch at {name}: old={tuple(old.shape)} vs new={tuple(custom_inv_freq.shape)}"
            )

        print(f"  [{name}] inv_freq.dtype={old.dtype}, device={old.device}", flush=True)
        new = custom_inv_freq.to(device=old.device, dtype=old.dtype)
        baseline_new = baseline_inv_freq.to(device=old.device, dtype=old.dtype)

        with torch.no_grad():
            old.copy_(new)

        # Defensive checks.
        if module.inv_freq.shape != old.shape:
            raise RuntimeError(f"{name}.inv_freq shape changed unexpectedly after copy_")
        if module.inv_freq.dtype != old.dtype:
            raise RuntimeError(f"{name}.inv_freq dtype changed unexpectedly after copy_")

        if method == "anchored_hybrid" and rigid_j0 > 0:
            j0 = min(int(rigid_j0), old.numel())
            if not torch.equal(module.inv_freq[:j0], baseline_new[:j0]):
                diff = torch.max(torch.abs(module.inv_freq[:j0] - baseline_new[:j0])).item()
                raise RuntimeError(
                    f"Rigid core mismatch at {name}: first {j0} pairs not exact baseline (max diff={diff:.3e})"
                )

        _clear_rotary_cache(module)
        patched.append(
            {
                "name": name,
                "shape": list(old.shape),
                "dtype": str(old.dtype),
                "device": str(old.device),
                "min": float(module.inv_freq.min().item()),
                "max": float(module.inv_freq.max().item()),
            }
        )

    return {
        "patched_count": len(patched),
        "patched_modules": patched,
    }


@dataclass
class Sample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class SyntheticLongQADataset(Dataset):
    """
    Dummy long QA data with chat template.
    Format:
      <|start_header_id|>user<|end_header_id|>
      [LONG TEXT] ... What is the answer?
      <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      [ANSWER]
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int,
        max_seq_len: int,
        seed: int,
        depth_choices: Sequence[float],
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = int(num_samples)
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.depth_choices = list(depth_choices)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_id is None:
            self.pad_id = 0

        self.filler_text = (
            "This document section discusses long context memory, retrieval, and instruction fidelity. "
            "Careful reading is required to answer the final question correctly. "
        )
        self.filler_ids = tokenizer.encode(self.filler_text, add_special_tokens=False)
        if len(self.filler_ids) == 0:
            raise RuntimeError("Tokenizer produced empty ids for filler text.")

    def __len__(self) -> int:
        return self.num_samples

    def _build_context_text(self, rng: random.Random, passkey: str, depth: float) -> str:
        needle = f"The special magic number is {passkey}. Keep this exact number in memory. "
        needle_ids = self.tokenizer.encode(needle, add_special_tokens=False)

        # Reserve room for chat template + question + assistant answer.
        context_budget = max(256, self.max_seq_len - 700)
        filler_budget = max(0, context_budget - len(needle_ids))

        repeats = filler_budget // len(self.filler_ids) + 1
        context_ids = (self.filler_ids * repeats)[:filler_budget]
        pos = int(round(float(depth) * len(context_ids)))
        pos = min(max(pos, 0), len(context_ids))
        merged = context_ids[:pos] + needle_ids + context_ids[pos:]
        return self.tokenizer.decode(merged, skip_special_tokens=True)

    def _make_one(self, idx: int) -> Sample:
        rng = random.Random(self.seed * 1000003 + idx)
        passkey = f"{rng.randint(10000, 99999)}"
        depth = rng.choice(self.depth_choices)
        context_text = self._build_context_text(rng, passkey, depth)

        user_text = (
            "Read the long document and answer with only the exact number.\n\n"
            f"{context_text}\n\n"
            "Question: What is the special magic number?"
        )
        answer_text = passkey

        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer_text},
        ]
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = coerce_chat_template_ids(self.tokenizer, full_ids)
        if len(full_ids) == 0:
            raise RuntimeError("apply_chat_template returned empty sequence.")

        orig_len = len(full_ids)
        if len(full_ids) > self.max_seq_len:
            # Keep BOS (first token) + tail to preserve both BOS anchor and question+assistant region.
            full_ids = [full_ids[0]] + full_ids[-(self.max_seq_len - 1) :]
        seq_len = len(full_ids)

        # Pad to fixed length.
        if seq_len < self.max_seq_len:
            full_ids = full_ids + [self.pad_id] * (self.max_seq_len - seq_len)
        attn = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        # Supervise only assistant tail region (robust against template internals).
        ans_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        tail_k = min(max(6, len(ans_ids) + 4), max(1, seq_len - 1))
        labels = [-100] * self.max_seq_len
        start = seq_len - tail_k
        for i in range(start, seq_len):
            labels[i] = full_ids[i]

        if orig_len < 16:
            raise RuntimeError("Template sequence too short, unexpected dataset construction.")
        return Sample(input_ids=full_ids, attention_mask=attn, labels=labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self._make_one(idx)
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
        }


class RealTextPackingDataset(Dataset):
    """
    Pack real text into fixed-length sequences for causal LM training/eval.
    Falls back to structured synthetic text only when no real source is available.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        num_samples: int,
        seed: int = 42,
        data_source: str = "auto",
        split: str = "train",
        split_ratio: float = 0.95,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.num_samples = int(num_samples)
        self.split = str(split)
        self.split_ratio = float(split_ratio)
        if self.split not in {"train", "eval"}:
            raise ValueError(f"split must be 'train' or 'eval', got {self.split}")
        if not (0.5 <= self.split_ratio < 1.0):
            raise ValueError(f"split_ratio must be in [0.5,1.0), got {self.split_ratio}")

        all_text, source_used = self._load_text(data_source)
        self.data_source = source_used
        self.text_sha256 = hashlib.sha256(all_text.encode("utf-8", errors="ignore")).hexdigest()

        print(f"[data] tokenizing {len(all_text)} chars...", flush=True)
        all_ids = tokenizer.encode(all_text, add_special_tokens=False)
        print(f"[data] total tokens: {len(all_ids)}", flush=True)
        self.total_tokens = int(len(all_ids))

        all_chunks: List[List[int]] = []
        for start in range(0, max(0, len(all_ids) - self.max_seq_len), self.max_seq_len):
            all_chunks.append(all_ids[start : start + self.max_seq_len])
        self.total_chunks = int(len(all_chunks))
        if len(all_chunks) == 0:
            raise RuntimeError(
                f"[data] no valid chunks produced (tokens={len(all_ids)}, max_seq_len={self.max_seq_len})"
            )

        split_at = max(1, int(len(all_chunks) * self.split_ratio))
        if self.split == "train":
            pool = all_chunks[:split_at]
        else:
            pool = all_chunks[split_at:]
        if len(pool) == 0:
            raise RuntimeError(
                f"[data] split '{self.split}' has 0 chunks. "
                f"total_chunks={len(all_chunks)}, split_ratio={self.split_ratio}"
            )
        self.available_chunks = int(len(pool))
        self.chunks = list(pool)

        if len(self.chunks) < self.num_samples:
            print(
                f"[data] WARNING: only {len(self.chunks)} chunks available, cycling to {self.num_samples}",
                flush=True,
            )
            base_chunks = list(self.chunks)
            while len(self.chunks) < self.num_samples:
                idx = (len(self.chunks) - len(base_chunks)) % len(base_chunks)
                self.chunks.append(base_chunks[idx])

        rng = random.Random(int(seed))
        rng.shuffle(self.chunks)
        self.chunks = self.chunks[: self.num_samples]
        print(
            f"[data] source={self.data_source} split={self.split} prepared {len(self.chunks)} chunks "
            f"of length {self.max_seq_len} (pool={self.available_chunks}, total_chunks={self.total_chunks})",
            flush=True,
        )

    def _load_text(self, source: str) -> Tuple[str, str]:
        # Try 1: local text files.
        local_paths = [
            "/root/autodl-tmp/data/long_text.txt",
            "/root/autodl-tmp/data/redpajama_sample.txt",
            "/root/autodl-tmp/data/slimpajama_sample.txt",
            "./data/long_text.txt",
        ]
        for p in local_paths:
            if os.path.isfile(p):
                print(f"[data] loading from local file: {p}", flush=True)
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
                if len(text) > 10000:
                    return text, f"local:{p}"

        # Try 2: HF datasets.
        try:
            from datasets import load_dataset

            print("[data] trying to load wikitext-103-v1...", flush=True)
            ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
            text = "\n".join([x["text"] for x in ds if len(x["text"]) > 100])
            if len(text) > 10000:
                print(f"[data] loaded wikitext-103: {len(text)} chars", flush=True)
                return text, "hf:wikitext-103-v1"
        except Exception as e:
            print(f"[data] wikitext failed: {e}", flush=True)

        # Try 3: structured synthetic fallback.
        print("[data] FALLBACK: generating structured synthetic text", flush=True)
        paragraphs: List[str] = []
        topics = [
            "The history of artificial intelligence spans several decades of research and development.",
            "Climate change represents one of the most significant challenges facing humanity today.",
            "Modern financial systems rely on complex networks of institutions and regulations.",
            "The development of quantum computing promises to revolutionize computational capabilities.",
            "Advances in medical research continue to improve human health outcomes worldwide.",
            "The evolution of programming languages reflects changing computational paradigms.",
            "Space exploration has entered a new era with private companies joining government agencies.",
            "The study of neuroscience reveals the remarkable complexity of the human brain.",
            "Renewable energy technologies are transforming the global energy landscape.",
            "The field of materials science enables the creation of novel substances with unique properties.",
        ]
        rng = random.Random(42)
        target_chars = self.num_samples * self.max_seq_len * 5
        text = ""
        while len(text) < target_chars:
            topic = rng.choice(topics)
            para = (topic + " ") * rng.randint(50, 100)
            paragraphs.append(para)
            text = "\n\n".join(paragraphs)
        return text, "synthetic:fallback_structured"

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.chunks[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones(len(ids), dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fixed(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
        "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
        "labels": torch.stack([f["labels"] for f in features], dim=0),
    }


class StdoutLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if "loss" not in logs:
            return
        msg = {
            "time": now(),
            "step": int(state.global_step),
            "loss": float(logs["loss"]),
            "learning_rate": float(logs.get("learning_rate", 0.0)),
        }
        line = json.dumps(msg, ensure_ascii=False)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_template_sanity(tokenizer: AutoTokenizer) -> Dict[str, object]:
    msg = [{"role": "user", "content": "Template sanity test."}]
    ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
    ids = coerce_chat_template_ids(tokenizer, ids)
    if len(ids) == 0:
        raise RuntimeError("apply_chat_template sanity failed: empty ids.")

    start_header_id = None
    end_header_id = None
    try:
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    except Exception:
        pass
    return {
        "template_ids_len": len(ids),
        "start_header_id": int(start_header_id) if isinstance(start_header_id, int) and start_header_id >= 0 else None,
        "end_header_id": int(end_header_id) if isinstance(end_header_id, int) and end_header_id >= 0 else None,
    }


def probe_baseline_reinject_stability(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    baseline_inv_freq: torch.Tensor,
    rigid_j0: int,
) -> Dict[str, object]:
    """
    Extra calibration gate:
    If method=baseline, reinjecting baseline inv_freq should not alter outputs materially.
    """
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "State only the number 42."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = coerce_chat_template_ids(tokenizer, prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_before = model(input_ids=x).logits[:, -1, :].float().cpu()

    info = overwrite_inv_freq_inplace(
        model=model,
        custom_inv_freq=baseline_inv_freq,
        baseline_inv_freq=baseline_inv_freq,
        method="baseline",
        rigid_j0=rigid_j0,
    )
    with torch.no_grad():
        logits_after = model(input_ids=x).logits[:, -1, :].float().cpu()

    max_abs = float(torch.max(torch.abs(logits_before - logits_after)).item())
    return {
        "patched_count": int(info["patched_count"]),
        "max_abs_logit_diff": max_abs,
    }


def evaluate_ppl_at_lengths(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    eval_chunks: List[List[int]],
    lengths: Sequence[int],
    device: torch.device,
) -> Dict[int, Dict[str, object]]:
    """Evaluate Tail-PPL at multiple lengths on packed text chunks."""
    del tokenizer  # tokenizer kept for signature consistency
    model.eval()
    results: Dict[int, Dict[str, object]] = {}

    for seq_len in lengths:
        seq_len = int(seq_len)
        valid_chunks = [c[:seq_len] for c in eval_chunks if len(c) >= seq_len]
        if not valid_chunks:
            results[seq_len] = {"ppl": float("inf"), "note": "insufficient data"}
            continue

        losses: List[float] = []
        n_eval = min(10, len(valid_chunks))
        for i in range(n_eval):
            chunk = valid_chunks[i]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            labels = input_ids.clone()
            
            # Predict only the second half (tail PPL) to measure true long-range dependency
            half = seq_len // 2
            labels[:, :half] = -100
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                losses.append(float(outputs.loss.item()))

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        try:
            ppl = math.exp(min(avg_loss, 20.0)) if math.isfinite(avg_loss) else float("inf")
        except OverflowError:
            ppl = float("inf")
            
        results[seq_len] = {
            "ppl": round(float(ppl), 4) if math.isfinite(ppl) else float("inf"),
            "loss": round(float(avg_loss), 4) if math.isfinite(avg_loss) else float("inf"),
            "n_chunks": int(len(losses)),
        }
        print(
            f"  Tail-PPL@{seq_len}: {ppl:.4f} (loss={avg_loss:.4f}, chunks={len(losses)})",
            flush=True,
        )
    return results


def evaluate_passkey(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    lengths: Sequence[int],
    n_trials: int = 20,
    device: torch.device = torch.device("cuda"),
) -> Dict[int, Dict[str, object]]:
    """Passkey retrieval using instruction-tuned chat template."""
    model.eval()
    results: Dict[int, Dict[str, object]] = {}

    filler = "The quick brown fox jumps over the lazy dog. " * 10
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(filler_ids) == 0:
        raise RuntimeError("Passkey filler tokenization returned empty ids.")

    old_use_cache = getattr(model.config, "use_cache", None)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    try:
        for seq_len in lengths:
            correct = 0
            total = 0

            for trial in range(int(n_trials)):
                rng = random.Random(42 * 1000 + int(seq_len) * 31 + trial)
                passkey = str(rng.randint(10000, 99999))

                needle = f"IMPORTANT: The special magic number is {passkey}. Keep this exact number in memory."
                needle_ids = tokenizer.encode(needle, add_special_tokens=False)

                # Determine structural overhead of the chat template
                dummy_msg = [{"role": "user", "content": "Question: What is the special magic number?"}]
                template_ids = tokenizer.apply_chat_template(dummy_msg, tokenize=True, add_generation_prompt=True)
                template_ids = coerce_chat_template_ids(tokenizer, template_ids)
                
                # Budgets
                filler_budget = int(seq_len) - len(needle_ids) - len(template_ids) - 10
                if filler_budget <= 0:
                    continue

                n_filler = (filler_ids * ((filler_budget // len(filler_ids)) + 1))[:filler_budget]
                insert_pos = rng.randint(0, len(n_filler))
                context_ids = n_filler[:insert_pos] + needle_ids + n_filler[insert_pos:]
                
                context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
                user_text = f"Read the long document and answer with only the exact number.\n\n{context_text}\n\nQuestion: What is the special magic number?"
                
                messages = [{"role": "user", "content": user_text}]
                full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                full_ids = coerce_chat_template_ids(tokenizer, full_ids)
                
                if len(full_ids) == 0:
                    continue

                # Ensure exact length constraint
                if len(full_ids) > int(seq_len):
                    # Keep first token (BOS) + end
                    full_ids = [full_ids[0]] + full_ids[-(int(seq_len) - 1):]

                input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)

                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_tensor,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                generated = output[0][len(full_ids) :]
                generated_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                if passkey in generated_text:
                    correct += 1
                total += 1

            acc = (correct / total) if total > 0 else 0.0
            results[int(seq_len)] = {"accuracy": round(acc, 4), "correct": int(correct), "total": int(total)}
            print(f"  Passkey@{seq_len}: {acc:.1%} ({correct}/{total})", flush=True)
    finally:
        if hasattr(model.config, "use_cache") and old_use_cache is not None:
            model.config.use_cache = old_use_cache

    return results


def chunk_fingerprint(chunk: Sequence[int]) -> str:
    arr = np.asarray(chunk, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fair Llama-3-8B LoRA suite (single method per run).")
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "pi", "yarn", "anchored_hybrid", "sigmoid", "anchored_sigmoid"],
    )
    ap.add_argument("--run_name", type=str, required=True)

    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument("--output_root", type=str, default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_suite")
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)

    # Locked core training hyperparams (defaults).
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--warmup_steps", type=int, default=20)

    # LoRA settings (locked defaults).
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Long context and frequency settings.
    ap.add_argument("--max_seq_len", type=int, default=16384)
    ap.add_argument("--rope_base", type=float, default=0.0, help="<=0 means auto-infer from model.config.rope_theta")
    ap.add_argument("--rigid_j0", type=int, default=12)
    ap.add_argument(
        "--anchor_factor",
        type=float,
        default=0.0,
        help="Anchor factor for anchored_sigmoid. <=0 means auto-compute from scale.",
    )
    ap.add_argument("--data_split_ratio", type=float, default=0.95)
    ap.add_argument("--passkey_trials", type=int, default=20)
    ap.add_argument("--ppl_eval_chunks", type=int, default=20)
    ap.add_argument(
        "--allow_synthetic_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow synthetic fallback text when no real corpus is found.",
    )

    # Synthetic data controls.
    ap.add_argument("--num_train_samples", type=int, default=2048)
    ap.add_argument("--num_eval_samples", type=int, default=128)

    # Calibration controls.
    ap.add_argument("--baseline_probe_tolerance", type=float, default=1e-4)
    ap.add_argument("--calibration_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    enforce_offline_mode()
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_start": now(),
        "method": args.method,
        "run_name": args.run_name,
        "args": vars(args),
    }
    (run_dir / "artifacts" / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # Load tokenizer first (needed for dataset + template sanity).
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_report = run_template_sanity(tokenizer)
    print(f"[template] {template_report}", flush=True)

    # Load config and enforce rope_scaling stays default-equivalent.
    cfg = AutoConfig.from_pretrained(
        args.base_model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if not rope_scaling_is_effectively_default(cfg):
        raise RuntimeError(
            f"Config rope_scaling is non-default ({cfg.rope_scaling}). "
            "This script requires default RoPE scaling for fairness."
        )

    rope_base = float(args.rope_base)
    if rope_base <= 0:
        rope_base = infer_model_rope_base(cfg, fallback=10000.0)
    print(f"[rope] using rope_base={rope_base}", flush=True)

    # Load model with SDPA only (no monkey patch).
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if not rope_scaling_is_effectively_default(model.config):
        raise RuntimeError(
            f"Loaded model rope_scaling is non-default ({model.config.rope_scaling}). "
            "Refusing to proceed."
        )

    # Infer head_dim from first rotary module (defensive).
    rotary_modules = find_rotary_modules_with_inv_freq(model)
    if not rotary_modules:
        raise RuntimeError("No rotary modules with inv_freq found in loaded model.")
    first_name, first_mod = rotary_modules[0]
    inferred_pairs = int(first_mod.inv_freq.numel())
    inferred_head_dim = inferred_pairs * 2
    print(f"[rope] first rotary module={first_name}, head_dim={inferred_head_dim}", flush=True)

    baseline_inv = get_custom_inv_freq(
        method="baseline",
        head_dim=inferred_head_dim,
        base=rope_base,
        max_seq_len=args.max_seq_len,
        rigid_j0=args.rigid_j0,
        anchor_factor=args.anchor_factor,
    )
    custom_inv = get_custom_inv_freq(
        method=args.method,
        head_dim=inferred_head_dim,
        base=rope_base,
        max_seq_len=args.max_seq_len,
        rigid_j0=args.rigid_j0,
        anchor_factor=args.anchor_factor,
    )
    if custom_inv.shape != first_mod.inv_freq.shape:
        raise RuntimeError(
            f"Generated inv_freq shape mismatch: {tuple(custom_inv.shape)} vs expected {tuple(first_mod.inv_freq.shape)}"
        )

    # Optional strong calibration for baseline method.
    baseline_probe = None
    if args.method == "baseline":
        baseline_probe = probe_baseline_reinject_stability(
            model=model,
            tokenizer=tokenizer,
            baseline_inv_freq=baseline_inv,
            rigid_j0=args.rigid_j0,
        )
        print(f"[calibration] baseline reinject probe: {baseline_probe}", flush=True)
        if baseline_probe["max_abs_logit_diff"] > float(args.baseline_probe_tolerance):
            raise RuntimeError(
                f"Baseline reinject logit drift too large: {baseline_probe['max_abs_logit_diff']:.6e} "
                f"> {args.baseline_probe_tolerance:.6e}"
            )

    # Apply required in-place overwrite.
    inject_info = overwrite_inv_freq_inplace(
        model=model,
        custom_inv_freq=custom_inv,
        baseline_inv_freq=baseline_inv,
        method=args.method,
        rigid_j0=args.rigid_j0,
    )
    print(f"[rope] patched modules={inject_info['patched_count']}", flush=True)

    # ---- Fatal #1 fix: verify inv_freq injection is actually consumed by forward ----
    if args.method != "baseline":
        print("[safety] verifying inv_freq injection is active...", flush=True)
        _probe_device = next(model.parameters()).device
        _probe_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=_probe_device)
        model.eval()
        with torch.no_grad():
            _logits_custom = model(input_ids=_probe_ids).logits[:, -1, :].float().cpu()

        # Temporarily revert to baseline inv_freq
        _backup = {}
        for _n, _m in rotary_modules:
            _backup[_n] = _m.inv_freq.clone()
            _bl = baseline_inv.to(device=_m.inv_freq.device, dtype=_m.inv_freq.dtype)
            with torch.no_grad():
                _m.inv_freq.copy_(_bl)
            _clear_rotary_cache(_m)
        with torch.no_grad():
            _logits_baseline = model(input_ids=_probe_ids).logits[:, -1, :].float().cpu()

        # Restore custom inv_freq
        for _n, _m in rotary_modules:
            with torch.no_grad():
                _m.inv_freq.copy_(_backup[_n])
            _clear_rotary_cache(_m)

        _diff = torch.max(torch.abs(_logits_custom - _logits_baseline)).item()
        if _diff < 1e-6:
            raise RuntimeError(
                f"CRITICAL: inv_freq injection appears INERT (logit diff={_diff:.3e}). "
                "The current transformers version likely ignores the inv_freq buffer in forward(). "
                "Aborting to prevent silent baseline training."
            )
        print(f"[safety] inv_freq injection verified ACTIVE (logit diff={_diff:.3e})", flush=True)
        model.train()

    # Re-check no rope_scaling mutation to non-default state.
    if not rope_scaling_is_effectively_default(model.config):
        raise RuntimeError("model.config.rope_scaling mutated to non-default state after injection.")

    # Save inv_freq tensors for downstream eval (bitwise reproducibility).
    torch.save(custom_inv.to(torch.float64), run_dir / "artifacts" / "custom_inv_freq.pt")
    torch.save(baseline_inv.to(torch.float64), run_dir / "artifacts" / "baseline_inv_freq.pt")
    print(f"[rope] saved inv_freq tensors to {run_dir / 'artifacts'}", flush=True)

    calibration_report = {
        "timestamp": now(),
        "method": args.method,
        "inferred_head_dim": inferred_head_dim,
        "rope_base": rope_base,
        "max_seq_len": args.max_seq_len,
        "template_report": template_report,
        "baseline_probe": baseline_probe,
        "inject_info": inject_info,
        "custom_inv_stats": {
            "shape": list(custom_inv.shape),
            "min": float(custom_inv.min().item()),
            "max": float(custom_inv.max().item()),
            "mean": float(custom_inv.mean().item()),
        },
    }
    (run_dir / "artifacts" / "calibration_report.json").write_text(
        json.dumps(calibration_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.calibration_only:
        print("[done] calibration_only=True, exiting before training.", flush=True)
        return

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "peft is required for LoRA training. Install peft in this environment before full run."
        ) from exc

    # Build LoRA after RoPE injection.
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=parse_targets(args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Training/eval data: packed real text chunks.
    train_ds = RealTextPackingDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_train_samples,
        seed=args.seed,
        split="train",
        split_ratio=args.data_split_ratio,
    )
    eval_ds = RealTextPackingDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_eval_samples,
        seed=args.seed + 999,
        split="eval",
        split_ratio=args.data_split_ratio,
    )
    if train_ds.text_sha256 != eval_ds.text_sha256:
        raise RuntimeError(
            "Train/eval text source mismatch: "
            f"train_sha={train_ds.text_sha256[:12]} eval_sha={eval_ds.text_sha256[:12]}"
        )
    if train_ds.data_source.startswith("synthetic:") and not args.allow_synthetic_fallback:
        raise RuntimeError(
            "Refusing to train on synthetic fallback text by default. "
            "Provide real corpus file or pass --allow_synthetic_fallback."
        )
    train_fp = {chunk_fingerprint(c) for c in train_ds.chunks}
    eval_fp = {chunk_fingerprint(c) for c in eval_ds.chunks}
    overlap_count = len(train_fp.intersection(eval_fp))
    if overlap_count > 0:
        raise RuntimeError(
            f"Train/eval chunk leakage detected: overlap_count={overlap_count}. "
            "Refusing to continue to protect evidence validity."
        )
    print(
        f"[data] train/eval split verified: overlap=0, "
        f"source={train_ds.data_source}, sha256={train_ds.text_sha256[:12]}...",
        flush=True,
    )

    training_args = build_training_args_compat(
        {
            "output_dir": str(run_dir / "trainer"),
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": True,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "bf16": bool(args.bf16 and torch.cuda.is_available()),
            "fp16": False,
            "eval_strategy": "steps",
            "eval_steps": max(50, args.logging_steps),
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "save_total_limit": 2,
            "warmup_steps": args.warmup_steps,
            "report_to": [],
            "remove_unused_columns": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": 0,
        }
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collate_fixed,
        "callbacks": [StdoutLoggerCallback(run_dir / "logs" / "train.log")],
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        print("[compat] neither tokenizer nor processing_class accepted by Trainer; proceeding without one.", flush=True)

    trainer = Trainer(**trainer_kwargs)

    t0 = time.time()
    train_out = trainer.train()
    train_sec = float(time.time() - t0)
    metrics = dict(train_out.metrics)
    metrics["train_seconds"] = train_sec

    eval_metrics = trainer.evaluate()

    # Save LoRA adapter.
    adapter_dir = run_dir / "final_lora"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ============ Post-training evaluation ============
    print("\n" + "=" * 60, flush=True)
    print("  POST-TRAINING EVALUATION", flush=True)
    print("=" * 60, flush=True)

    eval_device = next(trainer.model.parameters()).device

    # Held-out PPL text comes from eval chunks only (avoid train leakage).
    # We pass the unflattened chunks directly to evaluate_ppl_at_lengths.
    eval_chunks = eval_ds.chunks

    print("\n[eval] Tail-PPL at multiple lengths:", flush=True)
    ppl_lengths: List[int] = [1024, 2048, 4096, 8192]
    if args.max_seq_len >= 16384:
        ppl_lengths.append(16384)
    if args.max_seq_len >= 32768:
        ppl_lengths.append(32768)

    ppl_results: Dict[int, Dict[str, object]] = {}
    if eval_chunks and max(len(c) for c in eval_chunks) >= min(ppl_lengths):
        ppl_results = evaluate_ppl_at_lengths(
            model=trainer.model,
            tokenizer=tokenizer,
            eval_chunks=eval_chunks,
            lengths=ppl_lengths,
            device=eval_device,
        )
    else:
        print("[eval] insufficient text length for Tail-PPL evaluation, skipping", flush=True)

    print("\n[eval] Passkey Retrieval:", flush=True)
    passkey_lengths: List[int] = [1024, 2048, 4096, 8192]
    if args.max_seq_len >= 16384:
        passkey_lengths.append(16384)
    passkey_results = evaluate_passkey(
        model=trainer.model,
        tokenizer=tokenizer,
        lengths=passkey_lengths,
        n_trials=args.passkey_trials,
        device=eval_device,
    )

    summary = {
        "timestamp_end": now(),
        "method": args.method,
        "run_name": args.run_name,
        "rope_base": rope_base,
        "head_dim": inferred_head_dim,
        "max_seq_len": args.max_seq_len,
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "paths": {
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "calibration_report": str(run_dir / "artifacts" / "calibration_report.json"),
            "train_log": str(run_dir / "logs" / "train.log"),
        },
        "ppl_results": ppl_results,
        "passkey_results": passkey_results,
        "data_provenance": {
            "source": train_ds.data_source,
            "text_sha256": train_ds.text_sha256,
            "max_seq_len": args.max_seq_len,
            "split_ratio": args.data_split_ratio,
            "train_chunks": len(train_ds.chunks),
            "eval_chunks": len(eval_ds.chunks),
            "train_pool_chunks": train_ds.available_chunks,
            "eval_pool_chunks": eval_ds.available_chunks,
            "split_overlap_count": overlap_count,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
