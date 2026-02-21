#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, env_info, get_device, save_json, set_seed
from src.visualization import save_fig_both, set_plot_style

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def ensure_dependencies() -> None:
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("seaborn", "seaborn"),
    ]
    missing: List[str] = []
    for mod, pip_name in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    for pip_name in missing:
        print(f"[deps] installing {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


class TokenizerBase:
    vocab_size: int
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


class ByteTokenizer(TokenizerBase):
    def __init__(self) -> None:
        self.vocab_size = 258
        self.bos_token_id = 256
        self.eos_token_id = 257

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: Sequence[int]) -> str:
        b = bytes([i for i in ids if 0 <= int(i) < 256])
        return b.decode("utf-8", errors="ignore")


class HFTokenizer(TokenizerBase):
    def __init__(self, path: str) -> None:
        from transformers import AutoTokenizer

        self.path = path
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        # Use full tokenizer length (includes added/special tokens) to avoid id overflow.
        try:
            self.vocab_size = int(len(self.tok))
        except Exception:
            self.vocab_size = int(self.tok.vocab_size)
        self.bos_token_id = self.tok.bos_token_id
        self.eos_token_id = self.tok.eos_token_id

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(map(int, ids)), skip_special_tokens=True)


def resolve_tokenizer(mode: str, tokenizer_path: str, local_model_candidates: str) -> Tuple[TokenizerBase, str]:
    if mode.lower() == "byte":
        return ByteTokenizer(), "byte"

    tried: List[str] = []
    candidates: List[str] = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    for p in [x.strip() for x in local_model_candidates.split(",") if x.strip()]:
        candidates.append(p)
    # prioritize small vocab gpt-neox if present, then others
    candidates.extend([
        "/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b",
        "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
        "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct",
    ])

    if mode.lower() in {"hf", "auto"}:
        for p in candidates:
            if not p or p in tried:
                continue
            tried.append(p)
            if not os.path.exists(p):
                continue
            try:
                tok = HFTokenizer(p)
                # Avoid extremely large vocab for this controlled experiment.
                if tok.vocab_size <= 70000:
                    return tok, f"hf:{p}"
            except Exception:
                continue

    # final fallback
    return ByteTokenizer(), "byte"


def extract_text_from_longbench_jsonl(path: Path, max_docs: int) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            parts: List[str] = []
            for k in ["context", "input"]:
                v = obj.get(k, "")
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            ans = obj.get("answers", None)
            if isinstance(ans, list) and ans:
                a0 = ans[0]
                if isinstance(a0, str) and a0.strip():
                    parts.append("Answer: " + a0.strip())
            if parts:
                texts.append("\n\n".join(parts))
            if len(texts) >= max_docs:
                break
    return texts


def generate_synthetic_texts(num_docs: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    base_sent = [
        "The research team discussed sequence modeling and long-context reasoning.",
        "This synthetic paragraph contains neutral scientific prose for language modeling.",
        "Careful optimization and reproducibility are essential for experimental reliability.",
        "Token-level prediction can reveal distributional differences across positions.",
    ]
    texts: List[str] = []
    for i in range(num_docs):
        key = rng.randint(10000, 99999)
        filler = " ".join(rng.choice(base_sent) for _ in range(180))
        doc = (
            f"Document {i}. {filler}\n"
            f"Important fact: the pass key is {key}. Please remember this key.\n"
            f"{filler}\n"
            f"Question: what is the pass key? Answer: {key}."
        )
        texts.append(doc)
    return texts


def build_token_buffer(
    tokenizer: TokenizerBase,
    texts: Iterable[str],
    target_tokens: int,
    add_bos: bool = True,
) -> np.ndarray:
    all_ids: List[int] = []
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    for txt in texts:
        ids = tokenizer.encode(txt)
        if add_bos and bos is not None:
            all_ids.append(int(bos))
        all_ids.extend(ids)
        if eos is not None:
            all_ids.append(int(eos))
        if len(all_ids) >= target_tokens:
            break
    if len(all_ids) < 100000:
        raise RuntimeError(f"Too few tokens collected: {len(all_ids)}")
    arr = np.array(all_ids[:target_tokens], dtype=np.int32)
    return arr


def load_training_tokens(
    root_dir: Path,
    tokenizer: TokenizerBase,
    target_tokens: int,
    max_docs: int,
    seed: int,
) -> Tuple[np.ndarray, str]:
    longbench_dir = Path("/root/autodl-tmp/dfrope/ms_datasets/LongBench/data")
    texts: List[str] = []
    dataset_name = ""

    if longbench_dir.exists():
        files = sorted(longbench_dir.glob("*_e.jsonl"))
        for p in files:
            if len(texts) >= max_docs:
                break
            remain = max_docs - len(texts)
            try:
                cur = extract_text_from_longbench_jsonl(p, max_docs=remain)
                texts.extend(cur)
            except Exception:
                continue
        if texts:
            dataset_name = "LongBench-local"

    if not texts:
        texts = generate_synthetic_texts(num_docs=max_docs, seed=seed)
        dataset_name = "Synthetic-Passkey"

    ids = build_token_buffer(tokenizer=tokenizer, texts=texts, target_tokens=target_tokens)
    out_meta = {
        "dataset_name": dataset_name,
        "num_text_docs": len(texts),
        "num_tokens": int(ids.size),
        "target_tokens": int(target_tokens),
    }
    save_json(root_dir / "data" / "phase4_dataset_meta.json", out_meta)
    return ids, dataset_name

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x_rot


class FixedRoPE(nn.Module):
    def __init__(self, inv_freq: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.detach().clone().float(), persistent=False)
        self._cache: Dict[Tuple[int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(seq_len), str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        self._cache[key] = (cos, sin)
        # keep cache small
        if len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache.keys())))
        return cos, sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, T, H, D]
        seq_len = q.size(1)
        cos, sin = self.get_cos_sin(seq_len, q.device, q.dtype)
        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin
        return q_out, k_out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: FixedRoPE):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B,T,H,D]
        q, k = self.rope.apply_rotary(q, k)

        q = q.transpose(1, 2)  # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if need_weights:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
            out = torch.matmul(attn, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = None

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        out = self.proj(out)
        return out, attn


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rope: FixedRoPE):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_with_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, attn = self.attn(self.norm1(x), need_weights=True)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        assert attn is not None
        return x, attn


class GPTSmall(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        inv_freq: torch.Tensor,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = FixedRoPE(inv_freq=inv_freq)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, rope=self.rope) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = gradient_checkpointing
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm_f(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.forward_hidden(input_ids)
        logits = self.lm_head(h)
        return logits

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        pred = logits[:, :-1, :].contiguous()
        tgt = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), tgt.view(-1))
        return loss

    def compute_per_token_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        pred = logits[:, :-1, :].contiguous()
        tgt = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), tgt.view(-1), reduction="none")
        return loss.view(input_ids.size(0), input_ids.size(1) - 1)

    def extract_attention_slice(self, input_ids: torch.Tensor, layer_idx: int, head_idx: int, query_tail: int) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        attn_map = None
        for i, blk in enumerate(self.blocks):
            if i == layer_idx:
                x, attn = blk.forward_with_attn(x)
                attn_map = attn
            else:
                x = blk(x)
        if attn_map is None:
            raise RuntimeError("No attention captured")
        # attn_map: [B,H,T,T]
        sel = attn_map[0, head_idx, -query_tail:, :].detach().float().cpu()
        return sel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def split_train_val(tokens: np.ndarray, val_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    n = int(tokens.size)
    n_val = max(1024, int(n * val_ratio))
    n_val = min(n_val, n // 3)
    train_ids = tokens[:-n_val]
    val_ids = tokens[-n_val:]
    return train_ids, val_ids


def sample_batch(token_ids: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    max_start = int(token_ids.size - seq_len - 1)
    if max_start <= 1:
        raise RuntimeError("Token buffer too short for seq_len")
    starts = rng.integers(low=0, high=max_start, size=batch_size, endpoint=False)
    arr = np.stack([token_ids[s : s + seq_len + 1] for s in starts], axis=0)
    x = torch.tensor(arr, dtype=torch.long, device=device)
    return x


def build_fixed_eval_batches(token_ids: np.ndarray, seq_len: int, num_batches: int, batch_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    max_start = int(token_ids.size - seq_len - 1)
    starts_all = rng.integers(low=0, high=max_start, size=num_batches * batch_size, endpoint=False)
    out: List[np.ndarray] = []
    idx = 0
    for _ in range(num_batches):
        starts = starts_all[idx : idx + batch_size]
        idx += batch_size
        arr = np.stack([token_ids[s : s + seq_len + 1] for s in starts], axis=0)
        out.append(arr)
    return out

@dataclass
class TrainConfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    head_dim: int = 64
    max_seq_len: int = 8192
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    min_steps_if_slow: int = 5000
    max_hours_budget: float = 8.0
    grad_clip: float = 1.0


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def evaluate_val_loss(
    model: GPTSmall,
    val_batches: List[np.ndarray],
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for arr in val_batches:
            x = torch.tensor(arr, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(x)
            losses.append(float(loss.detach().item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    model: GPTSmall,
    optimizer: torch.optim.Optimizer,
    step: int,
    ckpt_dir: Path,
    tag: str,
) -> Path:
    out_dir = ckpt_dir / f"{tag}_step{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    path = out_dir / "checkpoint.pt"
    torch.save(payload, path)
    return out_dir


def save_final_model(model: GPTSmall, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


def autotune_micro_batch(
    model: GPTSmall,
    token_ids: np.ndarray,
    seq_len: int,
    candidates: Sequence[int],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed)
    model.train()
    for bsz in candidates:
        try:
            x = sample_batch(token_ids=token_ids, seq_len=seq_len, batch_size=int(bsz), rng=rng, device=device)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(x)
            loss.backward()
            del x, loss
            model.zero_grad(set_to_none=True)
            cleanup_cuda()
            print(f"[autotune] micro_batch={bsz} OK")
            return int(bsz)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                print(f"[autotune] micro_batch={bsz} OOM")
                cleanup_cuda()
                continue
            raise
    return 1


def train_two_models(
    root_dir: Path,
    standard_model: GPTSmall,
    sigmoid_model: GPTSmall,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    cfg: TrainConfig,
    seed: int,
    effective_batch_target: int,
) -> Dict[str, str]:
    device = next(standard_model.parameters()).device
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    micro_batch = autotune_micro_batch(
        model=standard_model,
        token_ids=train_ids,
        seq_len=cfg.max_seq_len,
        candidates=[16, 12, 8, 6, 4, 3, 2, 1],
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        seed=seed,
    )
    # We train two models in one process; single-model autotune can overestimate safe batch.
    dual_safe_batch = max(1, micro_batch // 2)
    if dual_safe_batch < micro_batch:
        print(f"[autotune] dual-model safety: reducing micro_batch {micro_batch} -> {dual_safe_batch}")
        micro_batch = dual_safe_batch
    grad_accum = max(1, int(math.ceil(float(effective_batch_target) / float(max(1, micro_batch)))))
    eff_batch = micro_batch * grad_accum
    print(f"[train] micro_batch={micro_batch}, grad_accum={grad_accum}, effective_batch={eff_batch}")

    opt_std = torch.optim.AdamW(standard_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    opt_sig = torch.optim.AdamW(sigmoid_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    val_batches = build_fixed_eval_batches(
        token_ids=val_ids,
        seq_len=cfg.max_seq_len,
        num_batches=8,
        batch_size=max(1, min(2, micro_batch)),
        seed=seed + 7,
    )

    logs_std: List[Dict] = []
    logs_sig: List[Dict] = []
    data_rng = np.random.default_rng(seed + 123)
    ckpt_root = root_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    max_steps = int(cfg.max_steps)
    shortened = False
    t0 = time.time()
    pbar = tqdm(range(1, max_steps + 1), desc="phase4-train", dynamic_ncols=True)
    for step in pbar:
        standard_model.train()
        sigmoid_model.train()

        opt_std.zero_grad(set_to_none=True)
        opt_sig.zero_grad(set_to_none=True)

        step_loss_std = 0.0
        step_loss_sig = 0.0

        lr = cosine_lr(step=step - 1, max_steps=max_steps, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps)
        set_optimizer_lr(opt_std, lr)
        set_optimizer_lr(opt_sig, lr)

        for _ in range(grad_accum):
            x = sample_batch(train_ids, seq_len=cfg.max_seq_len, batch_size=micro_batch, rng=data_rng, device=device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_std = standard_model.compute_loss(x)
            step_loss_std += float(loss_std.detach().item())
            loss_std = loss_std / grad_accum

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_sig = sigmoid_model.compute_loss(x)
            step_loss_sig += float(loss_sig.detach().item())
            loss_sig = loss_sig / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss_std).backward()
                scaler.scale(loss_sig).backward()
            else:
                loss_std.backward()
                loss_sig.backward()

        if scaler.is_enabled():
            scaler.unscale_(opt_std)
            scaler.unscale_(opt_sig)
            torch.nn.utils.clip_grad_norm_(standard_model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(sigmoid_model.parameters(), cfg.grad_clip)
            scaler.step(opt_std)
            scaler.step(opt_sig)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(standard_model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(sigmoid_model.parameters(), cfg.grad_clip)
            opt_std.step()
            opt_sig.step()

        train_loss_std = step_loss_std / grad_accum
        train_loss_sig = step_loss_sig / grad_accum

        elapsed = time.time() - t0
        steps_done = step
        sec_per_step = elapsed / max(1, steps_done)

        # auto shorten only once after warmup profile
        if (not shortened) and step == 100 and max_steps > cfg.min_steps_if_slow:
            projected_hours = sec_per_step * max_steps / 3600.0
            if projected_hours > cfg.max_hours_budget:
                old_max = max_steps
                max_steps = cfg.min_steps_if_slow
                shortened = True
                pbar.total = max_steps
                print(f"\n[train] projected {projected_hours:.2f}h > {cfg.max_hours_budget:.1f}h, shortening {old_max} -> {max_steps}")
                if step >= max_steps:
                    break

        if step % 100 == 0 or step == 1:
            val_loss_std = evaluate_val_loss(standard_model, val_batches, device, amp_dtype, use_amp)
            val_loss_sig = evaluate_val_loss(sigmoid_model, val_batches, device, amp_dtype, use_amp)

            eta_hours = (max_steps - step) * sec_per_step / 3600.0
            print(
                f"[train] step={step}/{max_steps} lr={lr:.3e} "
                f"std_train={train_loss_std:.4f} sig_train={train_loss_sig:.4f} "
                f"std_val={val_loss_std:.4f} sig_val={val_loss_sig:.4f} eta={eta_hours:.2f}h"
            )

            logs_std.append(
                {
                    "step": step,
                    "train_loss": train_loss_std,
                    "val_loss": val_loss_std,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                }
            )
            logs_sig.append(
                {
                    "step": step,
                    "train_loss": train_loss_sig,
                    "val_loss": val_loss_sig,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                }
            )

            pd.DataFrame(logs_std).to_csv(root_dir / "data" / "training_log_standard.csv", index=False, encoding="utf-8")
            pd.DataFrame(logs_sig).to_csv(root_dir / "data" / "training_log_sigmoid.csv", index=False, encoding="utf-8")

        if step % 2000 == 0:
            save_checkpoint(standard_model, opt_std, step, ckpt_root, "standard")
            save_checkpoint(sigmoid_model, opt_sig, step, ckpt_root, "sigmoid")

        pbar.set_postfix(
            step=step,
            lr=f"{lr:.2e}",
            std=f"{train_loss_std:.3f}",
            sig=f"{train_loss_sig:.3f}",
        )

        if step >= max_steps:
            break

    pbar.close()

    save_final_model(standard_model, root_dir / "checkpoints" / "standard_final")
    save_final_model(sigmoid_model, root_dir / "checkpoints" / "sigmoid_final")

    return {
        "std_log": str(root_dir / "data" / "training_log_standard.csv"),
        "sig_log": str(root_dir / "data" / "training_log_sigmoid.csv"),
        "max_steps": str(max_steps),
        "micro_batch": str(micro_batch),
        "grad_accum": str(grad_accum),
        "effective_batch": str(eff_batch),
        "amp_dtype": str(amp_dtype),
    }

def greedy_generate(model: GPTSmall, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    x = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
    return x


def build_passkey_prompt(
    tokenizer: TokenizerBase,
    context_length: int,
    passkey: str,
    ratio: float,
) -> Optional[List[int]]:
    filler = "The grass is green. The sky is blue. The sun is yellow. "
    filler_ids = tokenizer.encode(filler)
    ask_ids = tokenizer.encode("\nWhat is the pass key? The pass key is ")
    needle_ids = tokenizer.encode(f" The pass key is {passkey}. Remember it. {passkey} is the pass key. ")

    budget = context_length - len(ask_ids) - len(needle_ids) - 4
    if budget < 32:
        return None
    rep = budget // max(1, len(filler_ids)) + 2
    body = (filler_ids * rep)[:budget]
    pos = int(ratio * max(1, len(body) - 1))
    seq = body[:pos] + needle_ids + body[pos:] + ask_ids
    seq = seq[:context_length]
    if tokenizer.bos_token_id is not None:
        seq = [int(tokenizer.bos_token_id)] + seq
    return seq


def run_passkey_eval(
    root_dir: Path,
    tokenizer: TokenizerBase,
    standard_model: GPTSmall,
    sigmoid_model: GPTSmall,
    lengths: List[int],
    ratios: List[float],
    repeats: int,
    max_new_tokens: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict] = []
    digit_re = re.compile(r"\d+")

    models = [("Standard", standard_model), ("Sigmoid", sigmoid_model)]
    for name, mdl in models:
        pbar = tqdm(total=len(lengths) * len(ratios) * repeats, desc=f"passkey-{name}", dynamic_ncols=True)
        for lval in lengths:
            for rval in ratios:
                for rep_idx in range(repeats):
                    key = f"{rng.randint(10000, 99999)}"
                    prompt = build_passkey_prompt(tokenizer=tokenizer, context_length=lval, passkey=key, ratio=rval)
                    if prompt is None:
                        rows.append(
                            {
                                "model": name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": 0,
                                "status": "invalid",
                            }
                        )
                        pbar.update(1)
                        continue

                    x = torch.tensor([prompt], dtype=torch.long, device=next(mdl.parameters()).device)
                    try:
                        out = greedy_generate(mdl, x, max_new_tokens=max_new_tokens)
                        gen_ids = out[0, x.size(1) :].detach().cpu().tolist()
                        gen_txt = tokenizer.decode(gen_ids)
                        pred_digits = "".join(digit_re.findall(gen_txt))
                        ok = int(key in pred_digits)
                        rows.append(
                            {
                                "model": name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": ok,
                                "status": "ok",
                                "passkey": key,
                                "pred_digits": pred_digits,
                            }
                        )
                    except RuntimeError as ex:
                        rows.append(
                            {
                                "model": name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": 0,
                                "status": "oom" if "out of memory" in str(ex).lower() else "error",
                            }
                        )
                        cleanup_cuda()
                    pbar.update(1)
        pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "data" / "passkey_trained_results.csv", index=False, encoding="utf-8")
    return df


def run_positional_ppl(
    root_dir: Path,
    model: GPTSmall,
    token_ids: np.ndarray,
    seq_lens: List[int],
    num_samples_each: int,
    seed: int,
    out_csv: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    rows: List[Dict] = []
    model.eval()
    for sl in seq_lens:
        if token_ids.size <= sl + 2:
            continue
        max_start = int(token_ids.size - sl - 1)
        starts = rng.integers(low=0, high=max_start, size=num_samples_each, endpoint=False)
        pos_losses = np.zeros((num_samples_each, sl), dtype=np.float64)
        ok_count = 0
        for i, st in enumerate(starts):
            arr = token_ids[st : st + sl + 1]
            x = torch.tensor(arr[None, :], dtype=torch.long, device=device)
            try:
                with torch.no_grad():
                    loss_tok = model.compute_per_token_loss(x).detach().float().cpu().numpy()[0]
                pos_losses[i, : sl] = loss_tok[:sl]
                ok_count += 1
            except RuntimeError:
                cleanup_cuda()
                continue
        if ok_count == 0:
            continue
        mean_loss = pos_losses[:ok_count].mean(axis=0)
        for p, lv in enumerate(mean_loss.tolist()):
            rows.append({"seq_len": sl, "position": p + 1, "loss": float(lv)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_df


def plot_training_curves(root_dir: Path) -> None:
    std_df = pd.read_csv(root_dir / "data" / "training_log_standard.csv")
    sig_df = pd.read_csv(root_dir / "data" / "training_log_sigmoid.csv")

    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(std_df["step"], std_df["train_loss"], color="#d62728", label="Standard")
    axes[0].plot(sig_df["step"], sig_df["train_loss"], color="#1f77b4", label="Sigmoid")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend(frameon=True)

    axes[1].plot(std_df["step"], std_df["val_loss"], color="#d62728", label="Standard")
    axes[1].plot(sig_df["step"], sig_df["val_loss"], color="#1f77b4", label="Sigmoid")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_xlabel("Step")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "training_curves")
    plt.close(fig)


def plot_passkey_results(root_dir: Path, df: pd.DataFrame, train_len_boundary: int) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    set_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True)
    for ax, model_name in zip(axes, ["Standard", "Sigmoid"]):
        sub = ok_df[ok_df["model"] == model_name]
        piv = sub.groupby(["context_length", "position_ratio"], as_index=False)["correct"].mean().pivot(
            index="context_length", columns="position_ratio", values="correct"
        )
        if piv.empty:
            ax.axis("off")
            ax.set_title(model_name)
            continue
        im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_title(model_name)
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(int(v)) for v in piv.index])
        ax.set_ylabel("Context Length")
        ax.set_xlabel("Passkey Position Ratio")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Accuracy")

    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "passkey_trained")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.8))
    mean_len = ok_df.groupby(["model", "context_length"], as_index=False)["correct"].mean()
    for model_name, color in [("Standard", "#d62728"), ("Sigmoid", "#1f77b4")]:
        sub = mean_len[mean_len["model"] == model_name].sort_values("context_length")
        ax2.plot(sub["context_length"], sub["correct"], marker="o", color=color, label=model_name)
    ax2.axvline(train_len_boundary, linestyle="--", color="black", alpha=0.7, label=f"train max len={train_len_boundary}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, root_dir / "results" / "passkey_trained_by_length")
    plt.close(fig2)


def plot_positional_ppl(root_dir: Path, std_df: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=False)
    for ax, sl in zip(axes, [8192, 16384]):
        s1 = std_df[std_df["seq_len"] == sl]
        s2 = sig_df[sig_df["seq_len"] == sl]
        if not s1.empty:
            ax.plot(s1["position"], s1["loss"], color="#d62728", label="Standard")
        if not s2.empty:
            ax.plot(s2["position"], s2["loss"], color="#1f77b4", label="Sigmoid")
        ax.set_title(f"Sequence Length = {sl}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "positional_ppl")
    plt.close(fig)


def plot_attention_heatmap(
    root_dir: Path,
    tokenizer: TokenizerBase,
    standard_model: GPTSmall,
    sigmoid_model: GPTSmall,
    token_ids: np.ndarray,
    seq_len: int,
    layer_idx: int,
    head_idx: int,
    query_tail: int,
) -> None:
    device = next(standard_model.parameters()).device
    if token_ids.size < seq_len + 1:
        seq_len = min(seq_len, int(token_ids.size - 1))
    x = torch.tensor(token_ids[:seq_len][None, :], dtype=torch.long, device=device)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, name, model in [(axes[0], "Standard", standard_model), (axes[1], "Sigmoid", sigmoid_model)]:
        try:
            att = model.extract_attention_slice(x, layer_idx=layer_idx, head_idx=head_idx, query_tail=query_tail)
            im = ax.imshow(att.numpy(), aspect="auto", origin="lower", cmap="magma")
            ax.set_title(name)
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Last Query Positions")
            cbar = fig.colorbar(im, ax=ax, shrink=0.9)
            cbar.set_label("Attention")
        except RuntimeError as ex:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Failed: {type(ex).__name__}", ha="center", va="center")
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "attention_heatmap")
    plt.close(fig)


def safe_mean(v: pd.Series) -> float:
    return float(v.mean()) if not v.empty else float("nan")


def summarize_phase4(
    root_dir: Path,
    dataset_name: str,
    total_tokens: int,
    model_params: int,
    max_steps: int,
    wall_hours: float,
    train_log_std: pd.DataFrame,
    train_log_sig: pd.DataFrame,
    passkey_df: pd.DataFrame,
    ppl_std: pd.DataFrame,
    ppl_sig: pd.DataFrame,
) -> None:
    std_final_train = float(train_log_std["train_loss"].iloc[-1])
    sig_final_train = float(train_log_sig["train_loss"].iloc[-1])
    std_final_val = float(train_log_std["val_loss"].iloc[-1])
    sig_final_val = float(train_log_sig["val_loss"].iloc[-1])

    ok = passkey_df[passkey_df["status"] == "ok"]
    in_train = ok[ok["context_length"] <= 8192]
    extrap = ok[ok["context_length"] > 8192]

    std_in = safe_mean(in_train[in_train["model"] == "Standard"]["correct"]) * 100.0
    sig_in = safe_mean(in_train[in_train["model"] == "Sigmoid"]["correct"]) * 100.0
    std_ex = safe_mean(extrap[extrap["model"] == "Standard"]["correct"]) * 100.0
    sig_ex = safe_mean(extrap[extrap["model"] == "Sigmoid"]["correct"]) * 100.0

    def end_gap(df: pd.DataFrame, seq_len: int) -> float:
        sub = df[df["seq_len"] == seq_len]
        if sub.empty:
            return float("nan")
        n = len(sub)
        h = max(1, n // 10)
        a = float(sub.iloc[:h]["loss"].mean())
        b = float(sub.iloc[-h:]["loss"].mean())
        return b - a

    std_gap = end_gap(ppl_std, 8192)
    sig_gap = end_gap(ppl_sig, 8192)

    print("\n====================================================================")
    print("          Sigmoid-RoPE Training-Time Validation Results")
    print("====================================================================")
    print(f"\n模型规模: ~{model_params/1e6:.1f}M params, 12 layers, d=768, head_dim=64")
    print(f"训练数据: {dataset_name}, {total_tokens/1e6:.1f}M tokens")
    print(f"训练步数: {max_steps}, 训练时长: {wall_hours:.2f}h")
    print("\n1. 训练 Loss:")
    print(f"   Standard final loss: {std_final_train:.4f}")
    print(f"   Sigmoid  final loss: {sig_final_train:.4f} ({(std_final_train-sig_final_train)/max(1e-9,std_final_train)*100:.2f}%)")
    print("\n2. 验证 Perplexity:")
    print(f"   Standard: {math.exp(std_final_val):.4f}")
    print(f"   Sigmoid:  {math.exp(sig_final_val):.4f} ({(math.exp(std_final_val)-math.exp(sig_final_val))/max(1e-9,math.exp(std_final_val))*100:.2f}%)")
    print("\n3. Passkey Retrieval (训练长度内, <=8192):")
    print(f"   Standard 平均准确率: {std_in:.2f}%")
    print(f"   Sigmoid  平均准确率: {sig_in:.2f}%")
    print("\n4. Passkey Retrieval (外推, >8192):")
    print(f"   Standard 平均准确率: {std_ex:.2f}%")
    print(f"   Sigmoid  平均准确率: {sig_ex:.2f}%")
    print("\n5. 位置 PPL (序列末端 vs 开头的 loss 差异):")
    print(f"   Standard: 末端 loss 比开头高 {std_gap:.4f}")
    print(f"   Sigmoid:  末端 loss 比开头高 {sig_gap:.4f}")
    print("\n====================================================================")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sigmoid-RoPE Phase-4: training-time validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "hf", "byte"])
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument(
        "--local_model_candidates",
        type=str,
        default=(
            "/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b,"
            "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct,"
            "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
        ),
    )
    ap.add_argument("--target_tokens", type=int, default=12000000)
    ap.add_argument("--max_docs", type=int, default=4000)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--min_steps_if_slow", type=int, default=5000)
    ap.add_argument("--effective_batch_target", type=int, default=8)
    ap.add_argument("--passkey_repeats", type=int, default=5)
    ap.add_argument("--passkey_max_new_tokens", type=int, default=8)
    ap.add_argument("--passkey_lengths", type=str, default="1024,2048,4096,8192,16384")
    ap.add_argument("--max_hours_budget", type=float, default=8.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    set_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    (root_dir / "data").mkdir(parents=True, exist_ok=True)
    (root_dir / "results").mkdir(parents=True, exist_ok=True)
    (root_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[phase4] device:", device)

    tokenizer, tok_name = resolve_tokenizer(
        mode=args.tokenizer_mode,
        tokenizer_path=args.tokenizer_path,
        local_model_candidates=args.local_model_candidates,
    )
    print(f"[phase4] tokenizer: {tok_name}, vocab_size={tokenizer.vocab_size}")

    token_ids, dataset_name = load_training_tokens(
        root_dir=root_dir,
        tokenizer=tokenizer,
        target_tokens=int(args.target_tokens),
        max_docs=int(args.max_docs),
        seed=int(args.seed),
    )
    print(f"[phase4] dataset={dataset_name}, tokens={token_ids.size}")
    token_min = int(token_ids.min()) if token_ids.size else 0
    token_max = int(token_ids.max()) if token_ids.size else 0
    model_vocab_size = max(int(tokenizer.vocab_size), token_max + 1)
    print(
        f"[phase4] token_id_range=[{token_min}, {token_max}], "
        f"tokenizer_vocab={tokenizer.vocab_size}, model_vocab={model_vocab_size}"
    )

    train_ids, val_ids = split_train_val(token_ids, val_ratio=0.05)
    print(f"[phase4] train_tokens={train_ids.size}, val_tokens={val_ids.size}")

    cfg = TrainConfig(
        max_steps=int(args.max_steps),
        min_steps_if_slow=int(args.min_steps_if_slow),
        max_hours_budget=float(args.max_hours_budget),
    )

    allocator = RoPEFrequencyAllocator(d=cfg.head_dim, base=10000.0)
    inv_std = allocator.standard()
    k_formula = 16.05 / cfg.head_dim
    x0_formula = 0.47 * cfg.head_dim + 0.19 * math.log(cfg.max_seq_len)
    inv_sig = allocator.sigmoid(k=k_formula, x0=x0_formula)

    torch.manual_seed(args.seed)
    standard_model = GPTSmall(
        vocab_size=model_vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        inv_freq=inv_std,
        gradient_checkpointing=True,
    ).to(device)

    torch.manual_seed(args.seed)
    sigmoid_model = GPTSmall(
        vocab_size=model_vocab_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        inv_freq=inv_sig,
        gradient_checkpointing=True,
    ).to(device)

    param_count = count_parameters(standard_model)
    print(f"[phase4] model params: {param_count/1e6:.2f}M")
    print(f"[phase4] sigmoid config: k={k_formula:.6f}, x0={x0_formula:.4f}")

    train_t0 = time.time()
    train_meta = train_two_models(
        root_dir=root_dir,
        standard_model=standard_model,
        sigmoid_model=sigmoid_model,
        train_ids=train_ids,
        val_ids=val_ids,
        cfg=cfg,
        seed=args.seed,
        effective_batch_target=int(args.effective_batch_target),
    )
    train_hours = (time.time() - train_t0) / 3600.0

    plot_training_curves(root_dir)

    passkey_lengths = sorted({int(x.strip()) for x in str(args.passkey_lengths).split(",") if x.strip()})
    passkey_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    passkey_df = run_passkey_eval(
        root_dir=root_dir,
        tokenizer=tokenizer,
        standard_model=standard_model,
        sigmoid_model=sigmoid_model,
        lengths=passkey_lengths,
        ratios=passkey_ratios,
        repeats=int(args.passkey_repeats),
        max_new_tokens=int(args.passkey_max_new_tokens),
        seed=args.seed + 999,
    )
    plot_passkey_results(root_dir, passkey_df, train_len_boundary=cfg.max_seq_len)

    ppl_std = run_positional_ppl(
        root_dir=root_dir,
        model=standard_model,
        token_ids=val_ids,
        seq_lens=[8192, 16384],
        num_samples_each=8,
        seed=args.seed + 101,
        out_csv=root_dir / "data" / "positional_ppl_standard.csv",
    )
    ppl_sig = run_positional_ppl(
        root_dir=root_dir,
        model=sigmoid_model,
        token_ids=val_ids,
        seq_lens=[8192, 16384],
        num_samples_each=8,
        seed=args.seed + 202,
        out_csv=root_dir / "data" / "positional_ppl_sigmoid.csv",
    )
    plot_positional_ppl(root_dir, ppl_std, ppl_sig)

    plot_attention_heatmap(
        root_dir=root_dir,
        tokenizer=tokenizer,
        standard_model=standard_model,
        sigmoid_model=sigmoid_model,
        token_ids=val_ids,
        seq_len=4096,
        layer_idx=6,
        head_idx=0,
        query_tail=256,
    )

    train_log_std = pd.read_csv(root_dir / "data" / "training_log_standard.csv")
    train_log_sig = pd.read_csv(root_dir / "data" / "training_log_sigmoid.csv")

    summary = {
        "dataset_name": dataset_name,
        "tokenizer": tok_name,
        "total_tokens": int(token_ids.size),
        "param_count": int(param_count),
        "train_hours": float(train_hours),
        "train_meta": train_meta,
        "sigmoid_formula": {"k": float(k_formula), "x0": float(x0_formula)},
    }
    save_json(root_dir / "data" / "phase4_summary.json", summary)

    summarize_phase4(
        root_dir=root_dir,
        dataset_name=dataset_name,
        total_tokens=int(token_ids.size),
        model_params=int(param_count),
        max_steps=int(train_meta["max_steps"]),
        wall_hours=float(train_hours),
        train_log_std=train_log_std,
        train_log_sig=train_log_sig,
        passkey_df=passkey_df,
        ppl_std=ppl_std,
        ppl_sig=ppl_sig,
    )


if __name__ == "__main__":
    main()
