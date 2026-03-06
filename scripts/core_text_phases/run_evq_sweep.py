#!/usr/bin/env python3
"""EVQ τ-sweep on Apple M4 Max — theory validation for NeurIPS v5.

Usage:
    conda activate aidemo
    python scripts/core_text_phases/run_evq_sweep.py --tier 50m
    python scripts/core_text_phases/run_evq_sweep.py --tier 50m --taus 0.0,0.8 --dry_run

Validates:
  1. Theorem 2 (degradation): τ=0 ≡ geometric RoPE
  2. Waterbed inequality: long-context PPL ↓ ⟹ short-context PPL ↑
  3. Phase collision reduction with increasing τ
  4. Optimal τ range for downstream longinst experiments
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Use HuggingFace mirror for Chinese mainland servers
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# Fix: hf-mirror.com returns pagination Link headers pointing to huggingface.co,
# causing "Network is unreachable" on page 2+. Patch _get_next_page to rewrite URLs.
try:
    import huggingface_hub.utils._pagination as _hf_pag
    _orig_get_next_page = _hf_pag._get_next_page
    def _patched_get_next_page(response):
        url = _orig_get_next_page(response)
        if url and "huggingface.co" in url:
            mirror = os.environ.get("HF_ENDPOINT", "").rstrip("/")
            if mirror and mirror != "https://huggingface.co":
                url = url.replace("https://huggingface.co", mirror)
        return url
    _hf_pag._get_next_page = _patched_get_next_page
except Exception:
    pass
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device & dtype detection
# ---------------------------------------------------------------------------

def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32
print(f"[init] device={DEVICE}  dtype={DTYPE}  autocast={USE_AUTOCAST}")

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

TIER_CONFIGS = {
    "50m": {
        "vocab_size": 50304,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "max_position_embeddings": 2048,
        "batch_size": 32,       # CUDA bf16: 32 fits easily; MPS: auto-reduced below
        "train_tokens": 50_000_000,
        "seq_len": 2048,
        "lr": 6e-4,
        "eval_lengths": [2048, 4096, 8192, 16384],
        "eval_chunks": 10,
    },
    "125m": {
        "vocab_size": 50304,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "max_position_embeddings": 2048,
        "batch_size": 16,
        "train_tokens": 100_000_000,
        "seq_len": 2048,
        "lr": 3e-4,
        "eval_lengths": [2048, 4096, 8192, 16384],
        "eval_chunks": 10,
    },
    "350m": {
        "vocab_size": 50304,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
        "batch_size": 2,
        "train_tokens": 100_000_000,
        "seq_len": 2048,
        "lr": 2e-4,
        "eval_lengths": [2048, 4096, 8192],
        "eval_chunks": 8,
    },
    "500m": {
        "vocab_size": 50304,
        "hidden_size": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
        "batch_size": 4,
        "train_tokens": 500_000_000,
        "seq_len": 2048,
        "lr": 1.5e-4,
        "eval_lengths": [2048, 4096, 8192],
        "eval_chunks": 10,
    },
}

# ---------------------------------------------------------------------------
# EVQ-cosh frequency builder (from rope/schedules.py)
# ---------------------------------------------------------------------------

def evq_cosh_inv_freq(
    head_dim: int, tau: float, base: float = 500000.0
) -> torch.Tensor:
    """φ_k(τ) = 1 - (1/τ) arcsinh((1-u_k) sinh(τ)), u_k = (k+0.5)/K

    Returns inv_freq of shape (head_dim // 2,) in float32.
    """
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)  # midpoint quantization (paper formula 9)
    if abs(tau) < 1e-8:
        phi = u  # geometric limit (Theorem 2), half-step shifted
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
    inv = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    return inv.float()


def hybrid_evq_inv_freq(
    head_dim: int, tau: float, r: int, base: float = 500000.0
) -> torch.Tensor:
    """Hybrid frequency allocation: first r channels stay Geometric, rest use EVQ warp.

    Args:
        head_dim: head dimension (e.g. 64)
        tau: EVQ warp temperature
        r: number of high-frequency channels to keep as Geometric
           r=0 → full EVQ, r=K → full Geometric
        base: RoPE base frequency

    Returns:
        inv_freq of shape (head_dim // 2,) in float32
    """
    K = head_dim // 2
    if r <= 0:
        return evq_cosh_inv_freq(head_dim, tau, base)
    if r >= K:
        return evq_cosh_inv_freq(head_dim, 0.0, base)  # geometric

    # Geometric part: channels 0..r-1 (high-frequency)
    geo_full = evq_cosh_inv_freq(head_dim, 0.0, base)
    geo_part = geo_full[:r]

    # EVQ part: channels r..K-1 (low-frequency), warp within [theta_r, theta_{K-1}]
    n_evq = K - r
    theta_max = geo_full[r].item()      # highest freq in EVQ region
    theta_min = geo_full[K - 1].item()  # lowest freq in EVQ region

    idx = torch.arange(n_evq, dtype=torch.float64)
    u = idx / max(n_evq - 1, 1)  # 0..1 within EVQ region

    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)

    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))

    return torch.cat([geo_part, evq_part.float()])


# ---------------------------------------------------------------------------
# Phase collision metric
# ---------------------------------------------------------------------------

def phase_collision_score(
    inv_freq: torch.Tensor,
    distances: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute average cos(θ·Δ) phase collision across distance ranges.

    Lower is better (less collision = more orthogonal frequencies).
    """
    if distances is None:
        distances = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    inv = inv_freq.double()
    theta = inv  # angular frequencies
    results = {}
    short_scores, mid_scores, long_scores = [], [], []

    for d in distances:
        # cos(θ_k · Δ) averaged over all frequency channels
        collision = torch.cos(theta * d).mean().item()
        results[f"d={d}"] = round(collision, 6)
        if d <= 100:
            short_scores.append(collision)
        elif d <= 5000:
            mid_scores.append(collision)
        else:
            long_scores.append(collision)

    results["short_avg"] = round(np.mean(short_scores), 6) if short_scores else 0.0
    results["mid_avg"] = round(np.mean(mid_scores), 6) if mid_scores else 0.0
    results["long_avg"] = round(np.mean(long_scores), 6) if long_scores else 0.0
    results["total"] = round(
        0.2 * results["short_avg"] + 0.3 * results["mid_avg"] + 0.5 * results["long_avg"],
        6,
    )
    return results


# ---------------------------------------------------------------------------
# Waterbed analysis
# ---------------------------------------------------------------------------

def waterbed_analysis(
    ppl_baseline: Dict[str, float], ppl_tau: Dict[str, float]
) -> Dict[str, object]:
    """Compare PPL at different lengths against τ=0 baseline.

    Waterbed holds if: long-context PPL improves AND short-context PPL degrades.
    """
    lengths = sorted(int(k) for k in ppl_baseline.keys())
    if len(lengths) < 2:
        return {"error": "need at least 2 eval lengths"}

    short_L = str(lengths[0])
    long_L = str(lengths[-1])

    short_change = (ppl_tau[short_L] / ppl_baseline[short_L] - 1.0) * 100
    long_change = (ppl_tau[long_L] / ppl_baseline[long_L] - 1.0) * 100

    per_length = {}
    for L in lengths:
        sL = str(L)
        change = (ppl_tau[sL] / ppl_baseline[sL] - 1.0) * 100
        per_length[sL] = round(change, 3)

    return {
        f"short_{short_L}_change_%": round(short_change, 3),
        f"long_{long_L}_change_%": round(long_change, 3),
        "per_length_change_%": per_length,
        "waterbed_holds": long_change < -0.5 and short_change > 0.5,
        "long_improves": long_change < -0.5,
        "short_degrades": short_change > 0.5,
    }


# ---------------------------------------------------------------------------
# Model components (from run_50m_yarn_compare.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x)
            * self.weight
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int, inv_freq: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("inv_freq", inv_freq)
        self._build(max_seq)

    def _build(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)
        self._max = seq_len

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if L > self._max:
            self._build(L)
        return self.cos_c[:L], self.sin_c[:L]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.hd = cfg["head_dim"]
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h, m = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate = nn.Linear(h, m, bias=False)
        self.up = nn.Linear(h, m, bias=False)
        self.down = nn.Linear(m, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = Attention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: dict, inv_freq: torch.Tensor) -> None:
        super().__init__()
        self._num_layers = cfg["num_layers"]
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope = RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList(
            [Block(cfg, rope) for _ in range(cfg["num_layers"])]
        )
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self.apply(self._init)
        # Depth-scaled init for residual projections (GPT-2 / nanoGPT convention)
        residual_scale = 1.0 / math.sqrt(2 * self._num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.down.weight, std=0.02 * residual_scale)
        n = sum(p.numel() for p in self.parameters())
        print(f"  Model params: {n / 1e6:.1f}M")

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def extend_rope(self, L: int) -> None:
        self.blocks[0].attn.rope._build(L)


# ---------------------------------------------------------------------------
# Data loading with fallback
# ---------------------------------------------------------------------------

# Dataset candidates for fallback (priority order)
_DATASET_CANDIDATES = {
    "fineweb-edu": [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        ("cerebras/SlimPajama-627B", None, "text"),
        ("roneneldan/TinyStories", None, "text"),
    ],
    "tinystories": [
        ("roneneldan/TinyStories", None, "text"),
    ],
}


def _stream_tokenize(
    ds_name: str,
    config: Optional[str],
    text_key: str,
    tokenizer,
    max_tokens: int,
    shuffle_seed: Optional[int] = None,
    print_samples: int = 0,
) -> List[int]:
    """Stream-tokenize from a HF dataset. Returns list of token IDs."""
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if config:
        ds = load_dataset(ds_name, name=config, **kwargs)
    else:
        ds = load_dataset(ds_name, **kwargs)

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=10000)

    ids: List[int] = []
    n_docs = 0
    _next_report = 10_000_000  # progress every 10M tokens
    t0 = time.time()
    for x in ds:
        txt = x.get(text_key)
        if not txt:
            continue
        if print_samples > 0 and n_docs < print_samples:
            print(f"    [sample {n_docs}] {txt[:100]!r}...")
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        n_docs += 1
        if len(ids) >= _next_report:
            elapsed = time.time() - t0
            rate = len(ids) / elapsed
            eta = (max_tokens - len(ids)) / rate if rate > 0 else 0
            print(f"    [{len(ids)/1e6:.0f}M/{max_tokens/1e6:.0f}M tokens] "
                  f"{n_docs} docs, {rate/1e6:.1f}M tok/s, ETA {eta:.0f}s")
            _next_report += 10_000_000
        if len(ids) >= max_tokens:
            break

    elapsed = time.time() - t0
    print(f"    Tokenized {len(ids)/1e6:.1f}M tokens from {n_docs} docs in {elapsed:.0f}s")
    return ids


def load_data(
    tokenizer, max_tokens: int, seq_len: int, dataset: str = "fineweb-edu",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Load training data with automatic fallback and disk cache."""
    # Check disk cache first
    if cache_dir:
        cache_path = Path(cache_dir) / f"train_{dataset}_{max_tokens}_{seq_len}.pt"
        if cache_path.exists():
            print(f"  [data] Loading from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            print(f"  [data] Cached: {data.shape[0]} chunks ({data.numel()/1e6:.1f}M tokens)")
            return data

        # Try to slice from a larger cached file
        import glob as _glob
        prefix = f"train_{dataset}_"
        suffix = f"_{seq_len}.pt"
        for p in sorted(Path(cache_dir).glob(f"{prefix}*{suffix}"), reverse=True):
            fname = p.name
            try:
                cached_tokens = int(fname[len(prefix):-len(suffix)])
            except ValueError:
                continue
            if cached_tokens > max_tokens:
                print(f"  [data] Found larger cache: {p} ({cached_tokens/1e6:.0f}M tokens)")
                data = torch.load(p, weights_only=True)
                need_chunks = max_tokens // seq_len
                if data.shape[0] >= need_chunks:
                    data = data[:need_chunks]
                    print(f"  [data] Sliced to {data.shape[0]} chunks ({data.numel()/1e6:.1f}M tokens)")
                    return data

    candidates = _DATASET_CANDIDATES.get(dataset, _DATASET_CANDIDATES["tinystories"])

    for ds_name, config, text_key in candidates:
        try:
            print(f"  [data] Trying {ds_name} (config={config})...")

            # Quick connectivity check
            test_ids = _stream_tokenize(
                ds_name, config, text_key, tokenizer,
                max_tokens=1000, print_samples=0,
            )
            if len(test_ids) < 1000:
                raise RuntimeError(f"Only got {len(test_ids)} tokens from {ds_name}")
            print(f"  [data] Connection OK, streaming {max_tokens/1e6:.0f}M tokens...")

            # Full load
            ids = _stream_tokenize(
                ds_name, config, text_key, tokenizer,
                max_tokens=max_tokens, print_samples=3,
            )
            n = len(ids) // seq_len
            print(f"  [data] Got {n} chunks ({len(ids)/1e6:.1f}M tokens) from {ds_name}")
            data = torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)

            # Save to disk cache
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                torch.save(data, cache_path)
                print(f"  [data] Saved cache: {cache_path} ({cache_path.stat().st_size/1e6:.1f}MB)")

            return data

        except Exception as e:
            print(f"  [data] FAILED: {ds_name} — {e}")
            print(f"  [data] Falling back to next candidate...")
            continue

    raise RuntimeError("All dataset candidates failed!")


def load_val(
    tokenizer, max_tokens: int = 5_000_000, dataset: str = "fineweb-edu",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Load validation data (shuffled split for fineweb-edu)."""
    # Check disk cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"val_{dataset}_{max_tokens}.pt"
        if cache_path.exists():
            print(f"  [val] Loading from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            print(f"  [val] Cached: {data.numel()/1e6:.1f}M tokens")
            return data

    if dataset == "fineweb-edu":
        candidates = [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            ("cerebras/SlimPajama-627B", None, "text"),
            ("roneneldan/TinyStories", None, "text"),
        ]
        shuffle_seed = 99999  # different from train for sample-level split
    else:
        candidates = [("roneneldan/TinyStories", None, "text")]
        shuffle_seed = None  # TinyStories has a validation split

    for ds_name, config, text_key in candidates:
        try:
            print(f"  [val] Trying {ds_name}...")
            # For TinyStories, use the validation split directly
            if ds_name == "roneneldan/TinyStories" and shuffle_seed is None:
                from datasets import load_dataset
                vds = load_dataset(ds_name, split="validation", streaming=True)
                ids: List[int] = []
                for x in vds:
                    ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
                    if len(ids) >= max_tokens:
                        break
            else:
                ids = _stream_tokenize(
                    ds_name, config, text_key, tokenizer,
                    max_tokens=max_tokens, shuffle_seed=shuffle_seed,
                )
            print(f"  [val] Got {len(ids)/1e6:.1f}M tokens from {ds_name}")
            data = torch.tensor(ids, dtype=torch.long)

            # Save to disk cache
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                torch.save(data, cache_path)
                print(f"  [val] Saved cache: {cache_path} ({cache_path.stat().st_size/1e6:.1f}MB)")

            return data
        except Exception as e:
            print(f"  [val] FAILED: {ds_name} — {e}")
            continue

    raise RuntimeError("All validation dataset candidates failed!")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random as _random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


def resolve_passkey_mix_ratio(default: float = 0.005) -> float:
    """Resolve passkey mix ratio from env var (supports plain float or percent)."""
    raw = os.environ.get("PASSKEY_MIX_RATIO")
    if raw is None or raw == "":
        ratio = default
    else:
        text = raw.strip()
        if text.endswith("%"):
            ratio = float(text[:-1]) / 100.0
        else:
            ratio = float(text)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"PASSKEY_MIX_RATIO must be within [0,1], got {ratio}")
    return ratio


def get_batch_from_data(data, indices: torch.Tensor) -> torch.Tensor:
    """Fetch a batch from either a tensor dataset or a torch Dataset."""
    if isinstance(data, torch.Tensor):
        return data[indices]
    index_list = indices.tolist()
    return torch.stack([data[i] for i in index_list], dim=0)


def maybe_wrap_with_passkey_mix(
    train_data: torch.Tensor,
    filler_tokens: torch.Tensor,
    tokenizer,
    seq_len: int,
    passkey_ratio: float,
    sample_check_n: int = 2000,
):
    """Wrap train_data with MixedDataset if passkey_ratio > 0, otherwise return input."""
    if passkey_ratio <= 0.0:
        print("  [passkey-train] mix disabled (ratio=0.00%)")
        return train_data
    if tokenizer is None:
        raise ValueError("tokenizer is required when passkey_ratio > 0")

    from eval_passkey_scratch import MixedDataset
    mixed_data = MixedDataset(
        lm_data=train_data,
        filler_tokens=filler_tokens,
        tokenizer=tokenizer,
        passkey_ratio=passkey_ratio,
        seq_len=seq_len,
    )

    # Deterministic sanity check on the hash-based sampler logic.
    import random as _random
    n = min(sample_check_n, len(mixed_data))
    pk_count = sum(
        1
        for i in range(n)
        if _random.Random(i * 6364136223846793005 + 1).random() < passkey_ratio
    )
    print(
        f"  [passkey-train] mix target={passkey_ratio:.2%}, "
        f"sample_check={pk_count}/{n} ({(pk_count / max(n, 1)):.2%})"
    )
    return mixed_data


def train_model(
    model: GPT,
    data,
    cfg: dict,
    seed: int = 42,
) -> GPT:
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1  # Cosine schedule floor (standard practice)
    batch_size = cfg["batch_size"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    steps = len(data) // batch_size
    warmup = int(steps * 0.02)
    set_seed(seed)
    perm = torch.randperm(len(data))
    t0 = time.time()

    for s in range(steps):
        batch = get_batch_from_data(
            data, perm[s * batch_size : (s + 1) * batch_size]
        ).to(DEVICE)

        # Cosine LR with warmup + min_lr floor
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
            )
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Forward pass — autocast on CUDA (flash attention + bf16), plain on MPS/CPU
        ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
        with ctx:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(
                f"    step {s}/{steps}  loss={loss.item():.4f}  "
                f"lr={cur_lr:.2e}  ETA={eta / 60:.0f}min"
            )

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed / 60:.1f} min")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_model(
    model: GPT,
    val_data: torch.Tensor,
    eval_lengths: List[int],
    eval_chunks: int = 10,
    eval_seed: int = 9999,
) -> Dict[str, float]:
    model.eval()
    model.extend_rope(max(eval_lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(eval_seed)
    results = {}
    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            print(f"    L={L}: val_data too short, skipping")
            continue
        offsets = sorted(rng.choice(max_start, size=min(eval_chunks, max_start // L), replace=False))
        for offset in offsets:
            chunk = val_data[offset : offset + L].unsqueeze(0).to(DEVICE)
            try:
                with ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                    )
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM on offset {offset}, stopping this length")
                    # Free memory
                    del chunk
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE == "mps":
                        torch.mps.empty_cache()
                    break
                raise
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L}: PPL={ppl:.3f}  ({len(losses)} chunks)")
        else:
            print(f"    L={L}: skipped (no valid chunks)")
    return results


# ---------------------------------------------------------------------------
# Single run: train → eval → metrics
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: str
    tau: float
    seed: int
    tier: str
    base: float
    ppl: Dict[str, float] = field(default_factory=dict)
    phase_collision: Dict[str, float] = field(default_factory=dict)
    inv_freq_stats: Dict[str, float] = field(default_factory=dict)
    train_time_sec: float = 0.0
    eval_time_sec: float = 0.0


def run_single(
    tau: float,
    seed: int,
    cfg: dict,
    tier: str,
    base: float,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    work_dir: Path,
    dry_run: bool = False,
    tokenizer=None,
    eval_16k: bool = False,
    override_inv_freq: Optional[torch.Tensor] = None,
    override_run_id: Optional[str] = None,
) -> RunResult:
    run_id = override_run_id or f"{tier}_tau{tau:.2f}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}  (base={base:.0f})")
    print(f"{'='*60}")

    # Build inv_freq
    if override_inv_freq is not None:
        inv_freq = override_inv_freq
    else:
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)
    import hashlib
    inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]
    print(f"  inv_freq: shape={inv_freq.shape}  max={inv_freq.max().item():.6f}  "
          f"min={inv_freq.min().item():.8f}  hash={inv_hash}")

    # Phase collision
    pc = phase_collision_score(inv_freq)
    print(f"  Phase collision total={pc['total']:.6f}")

    # Inv_freq stats
    inv_stats = {
        "max": round(inv_freq.max().item(), 8),
        "min": round(inv_freq.min().item(), 8),
        "ratio_max_min": round((inv_freq.max() / inv_freq.min()).item(), 2),
        "mean": round(inv_freq.mean().item(), 8),
        "std": round(inv_freq.std().item(), 8),
        "sha256_16": inv_hash,
    }

    if dry_run:
        print("  [DRY RUN] skipping training & eval")
        return RunResult(
            run_id=run_id, tau=tau, seed=seed, tier=tier, base=base,
            phase_collision=pc, inv_freq_stats=inv_stats,
        )

    passkey_mix_ratio = float(cfg.get("passkey_mix_ratio", 0.005))
    mixed_data = maybe_wrap_with_passkey_mix(
        train_data=train_data,
        filler_tokens=val_data[:50000],
        tokenizer=tokenizer,
        seq_len=cfg["seq_len"],
        passkey_ratio=passkey_mix_ratio,
    )

    # Train
    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)

    # Parameter count validation for 500m
    n_params = sum(p.numel() for p in model.parameters())
    if tier == "500m":
        assert 450_000_000 <= n_params <= 550_000_000, \
            f"500M tier: expected 450-550M params, got {n_params/1e6:.1f}M"

    t0 = time.time()
    model = train_model(model, mixed_data, cfg, seed=seed)
    train_time = time.time() - t0

    # Eval: PPL
    eval_lengths = list(cfg["eval_lengths"])
    if eval_16k and 16384 not in eval_lengths:
        eval_lengths.append(16384)
    t1 = time.time()
    ppl = eval_model(model, val_data, eval_lengths, cfg["eval_chunks"])
    eval_time = time.time() - t1

    # Eval: Passkey NLL gap
    passkey_results = {}
    if tokenizer is not None:
        print(f"\n  [passkey] NLL-gap evaluation for {run_id}")
        from eval_passkey_scratch import eval_passkey_nll_gap
        pk_lengths = [L for L in eval_lengths if L >= cfg["seq_len"]][:3]
        passkey_results = eval_passkey_nll_gap(
            model, tokenizer, val_data,
            lengths=pk_lengths,
            depths=[0.10, 0.25, 0.50, 0.75, 0.90],
            num_trials=10,
        )
        g = passkey_results.get("global", {})
        print(f"  [passkey] Retrieval rate: {g.get('retrieval_rate', 'N/A')}")
        print(f"  [passkey] Mean NLL gap: {g.get('mean_nll_gap', 'N/A')}")

    # Save checkpoint
    run_dir = work_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.numpy())
    if passkey_results:
        import json as _json
        with open(run_dir / "passkey_nll.json", "w") as f:
            _json.dump(passkey_results, f, indent=2, ensure_ascii=False, default=str)

    # PI baseline: inference-time only, on geometric model (τ=0)
    pi_results = {}
    if abs(tau) < 1e-8:
        print(f"\n  [PI] Inference-time PI baseline (zero-cost, no extra training)")
        pi_scale = max(max(eval_lengths) / cfg["seq_len"], 1.0)
        pi_inv_freq = inv_freq / pi_scale
        pi_hash = hashlib.sha256(pi_inv_freq.numpy().tobytes()).hexdigest()[:16]
        print(f"  [PI] scale={pi_scale:.1f}  inv_freq hash={pi_hash}")

        # Replace inv_freq in model
        orig_inv = model.blocks[0].attn.rope.inv_freq.clone()
        model.blocks[0].attn.rope.inv_freq.copy_(pi_inv_freq)
        model.blocks[0].attn.rope._build(max(eval_lengths) + 100)

        # PPL eval with PI
        pi_ppl = eval_model(model, val_data, eval_lengths, cfg["eval_chunks"])
        pi_results["ppl"] = pi_ppl

        # Passkey eval with PI
        if tokenizer is not None:
            print(f"  [PI] Passkey NLL-gap evaluation")
            from eval_passkey_scratch import eval_passkey_nll_gap
            pi_passkey = eval_passkey_nll_gap(
                model, tokenizer, val_data,
                lengths=[2048, 4096, 8192],
                depths=[0.10, 0.25, 0.50, 0.75, 0.90],
                num_trials=10,
            )
            pi_results["passkey"] = pi_passkey

        # Restore original inv_freq
        model.blocks[0].attn.rope.inv_freq.copy_(orig_inv)
        model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])

    # Cleanup model to free memory
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

    result = RunResult(
        run_id=run_id,
        tau=tau,
        seed=seed,
        tier=tier,
        base=base,
        ppl=ppl,
        phase_collision=pc,
        inv_freq_stats=inv_stats,
        train_time_sec=round(train_time, 1),
        eval_time_sec=round(eval_time, 1),
    )
    return result, passkey_results, pi_results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EVQ τ-sweep on M4 Max")
    parser.add_argument("--tier", choices=["50m", "125m", "350m", "500m"], default="50m")
    parser.add_argument("--taus", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--base", type=float, default=500000.0,
                        help="RoPE base (500000=Llama-3, 10000=classic)")
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="fineweb-edu",
                        choices=["fineweb-edu", "tinystories"],
                        help="Training data source")
    parser.add_argument("--eval_16k", action="store_true",
                        help="Include 16384 in eval lengths (may OOM)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build models & inv_freq only, skip training")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have results")
    parser.add_argument(
        "--passkey_mix_ratio",
        type=float,
        default=None,
        help="Optional passkey training mix ratio (e.g. 0.02 for 2%%). "
             "If omitted, uses PASSKEY_MIX_RATIO env or defaults to 0.005.",
    )
    parser.add_argument("--r_values", type=str, default=None,
                        help="Comma-separated Hybrid r values to sweep (e.g. '0,8,14,16,24,32'). "
                             "If provided, runs r-sweep instead of tau-sweep. "
                             "r=0 means full EVQ, r=32 means full Geometric.")
    parser.add_argument("--fixed_tau", type=float, default=None,
                        help="Fixed tau for r-sweep (required when --r_values is set)")
    parser.add_argument("--train_tokens", type=int, default=None,
                        help="Override tier's default train_tokens (e.g. 50000000 for 50M)")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override training sequence length (default: from tier config). "
                             "Batch size auto-adjusts to keep tokens/step constant.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (bypasses all auto-adjustment)")
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = TIER_CONFIGS[args.tier].copy()

    # Override train_tokens if specified
    if args.train_tokens is not None:
        cfg["train_tokens"] = args.train_tokens
        print(f"  [override] train_tokens={cfg['train_tokens']/1e6:.0f}M")

    # Override seq_len if specified (adjust batch_size to keep tokens/step constant)
    if args.seq_len is not None:
        orig_seq = cfg["seq_len"]
        cfg["seq_len"] = args.seq_len
        cfg["max_position_embeddings"] = args.seq_len
        scale_factor = orig_seq / args.seq_len
        cfg["batch_size"] = max(1, int(cfg["batch_size"] * scale_factor))
        cfg["eval_lengths"] = [args.seq_len * m for m in [1, 2, 4, 8, 16]
                               if args.seq_len * m <= 16384]
        print(f"  [override] seq_len={args.seq_len}  batch_size={cfg['batch_size']}  "
              f"eval_lengths={cfg['eval_lengths']}")

    # r-sweep mode setup
    r_sweep_mode = args.r_values is not None
    r_values = []
    fixed_tau = None
    if r_sweep_mode:
        r_values = [int(r) for r in args.r_values.split(",")]
        fixed_tau = args.fixed_tau
        if fixed_tau is None:
            fixed_tau = cfg["head_dim"] / math.sqrt(cfg["seq_len"])
            print(f"  [r-sweep] No --fixed_tau provided, using tau*={fixed_tau:.2f}")
        print(f"  [r-sweep] r_values={r_values}, fixed_tau={fixed_tau:.2f}")

    cfg["passkey_mix_ratio"] = (
        args.passkey_mix_ratio
        if args.passkey_mix_ratio is not None
        else resolve_passkey_mix_ratio(default=0.005)
    )
    if not (0.0 <= cfg["passkey_mix_ratio"] <= 1.0):
        raise ValueError(
            f"passkey_mix_ratio must be within [0,1], got {cfg['passkey_mix_ratio']}"
        )

    # Device-specific adjustments (skip batch_size override if --seq_len was set,
    # since batch_size was already scaled to keep tokens/step constant)
    seq_len_overridden = args.seq_len is not None
    if DEVICE == "mps":
        # MPS float32: no flash attention, full attention matrix materialised
        # Reduce batch + cap eval length to avoid OOM
        if not seq_len_overridden:
            cfg["batch_size"] = min(cfg["batch_size"], 8)
        cfg["eval_lengths"] = [L for L in cfg["eval_lengths"] if L <= 8192]
        if args.tier == "350m" and not seq_len_overridden:
            cfg["batch_size"] = 1
            cfg["eval_lengths"] = [2048, 4096]
        print(f"  [MPS] Adjusted: batch={cfg['batch_size']}, eval_lengths={cfg['eval_lengths']}")
    elif DEVICE == "cuda":
        # CUDA bf16 + flash attention: can handle full config
        # Check VRAM and adjust if < 48GB
        props = torch.cuda.get_device_properties(0)
        # Prefer modern field name, then fallback to legacy alias.
        total_mem = getattr(props, "total_memory", None)
        if total_mem is None:
            total_mem = getattr(props, "total_mem")
        vram_gb = total_mem / (1024 ** 3)
        print(f"  [CUDA] GPU: {props.name}, VRAM: {vram_gb:.1f} GB")
        if vram_gb >= 80:
            if not seq_len_overridden:
                if args.tier == "500m":
                    cfg["batch_size"] = 16
                elif args.tier == "350m":
                    cfg["batch_size"] = 8
            if args.eval_16k and 16384 not in cfg["eval_lengths"]:
                cfg["eval_lengths"].append(16384)
            print(f"  [CUDA >=80GB] Adjusted {args.tier}: batch={cfg['batch_size']}, eval={cfg['eval_lengths']}")
        elif vram_gb < 40:
            if not seq_len_overridden:
                if args.tier == "50m":
                    cfg["batch_size"] = 8
                elif args.tier == "125m":
                    cfg["batch_size"] = 4
                elif args.tier in ("350m", "500m"):
                    cfg["batch_size"] = 2
            cfg["eval_lengths"] = [L for L in cfg["eval_lengths"] if L <= 8192]
            print(f"  [CUDA <40GB] Adjusted {args.tier}: batch={cfg['batch_size']}, eval={cfg['eval_lengths']}")

    # Manual batch_size override (highest priority)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
        print(f"  [override] batch_size={cfg['batch_size']}")

    if not args.work_dir:
        args.work_dir = str(Path.home() / "evq_m4_sweep")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    if r_sweep_mode:
        print(f"  EVQ r-SWEEP  |  tier={args.tier}  |  r_values={r_values}  |  tau={fixed_tau:.2f}")
    else:
        print(f"  EVQ τ-SWEEP  |  tier={args.tier}  |  taus={taus}")
    print(f"  device={DEVICE}  dtype={DTYPE}  base={args.base}")
    print(f"  train_tokens={cfg['train_tokens']/1e6:.0f}M")
    print(f"  passkey_mix_ratio={cfg['passkey_mix_ratio']:.2%}")
    print(f"  work_dir={work_dir}")
    print(f"{'#'*60}\n")

    # Load tokenizer & data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Passkey tokenizer sanity check
    from eval_passkey_scratch import sanity_check_tokenizer
    print("\n  [check] Passkey tokenizer sanity check:")
    tok_ok = sanity_check_tokenizer(tok)
    if not tok_ok:
        print("  WARNING: tokenizer sanity check failed!")

    if not args.dry_run:
        train_data = load_data(tok, cfg["train_tokens"], cfg["seq_len"], args.dataset,
                               cache_dir=str(work_dir))
        val_data = load_val(tok, dataset=args.dataset, cache_dir=str(work_dir))
    else:
        train_data = torch.zeros(10, cfg["seq_len"], dtype=torch.long)
        val_data = torch.zeros(50000, dtype=torch.long)

    # Load existing results if resuming
    results_path = work_dir / "results_checkpoint.json"
    if args.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Resumed {len(all_results.get('experiments', {}))} existing results")
    else:
        metadata = {
            "tier": args.tier,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "base": args.base,
            "seeds": seeds,
            "dataset": args.dataset,
            "train_tokens": cfg["train_tokens"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if r_sweep_mode:
            metadata["mode"] = "r-sweep"
            metadata["r_values"] = r_values
            metadata["fixed_tau"] = fixed_tau
        else:
            metadata["mode"] = "tau-sweep"
            metadata["taus"] = taus
        all_results = {
            "metadata": metadata,
            "experiments": {},
            "waterbed": {},
        }

    # Run sweep
    t_total = time.time()

    def _process_run_result(ret, run_id):
        """Extract result tuple and store in all_results."""
        if isinstance(ret, tuple):
            result, passkey_res, pi_res = ret
        else:
            result, passkey_res, pi_res = ret, {}, {}
        all_results["experiments"][run_id] = asdict(result)
        if passkey_res:
            all_results["experiments"][run_id]["passkey"] = passkey_res
        if pi_res:
            pi_run_id = f"{args.tier}_PI_seed{seed}"
            all_results["experiments"][pi_run_id] = {
                "method": "PI_inference_time",
                "source_run": run_id,
                **pi_res,
            }
        # Checkpoint after each run
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  [checkpoint] saved to {results_path}")

    if r_sweep_mode:
        for seed in seeds:
            for r_val in r_values:
                run_id = f"{args.tier}_r{r_val}_tau{fixed_tau:.2f}_seed{seed}"

                if args.resume and run_id in all_results["experiments"]:
                    print(f"\n  [SKIP] {run_id} already completed")
                    continue

                inv_freq = hybrid_evq_inv_freq(cfg["head_dim"], fixed_tau, r_val, args.base)

                # Print inv_freq continuity check at the r boundary
                K = cfg["head_dim"] // 2
                if 0 < r_val < K:
                    geo_full = evq_cosh_inv_freq(cfg["head_dim"], 0.0, args.base)
                    print(f"  [r={r_val}] boundary check: geo[{r_val-1}]={inv_freq[r_val-1].item():.6e}, "
                          f"evq[0]={inv_freq[r_val].item():.6e}, "
                          f"ratio={inv_freq[r_val-1].item()/inv_freq[r_val].item():.3f}")

                ret = run_single(
                    tau=fixed_tau, seed=seed, cfg=cfg, tier=args.tier, base=args.base,
                    train_data=train_data, val_data=val_data,
                    work_dir=work_dir, dry_run=args.dry_run,
                    tokenizer=tok, eval_16k=args.eval_16k,
                    override_inv_freq=inv_freq,
                    override_run_id=run_id,
                )
                _process_run_result(ret, run_id)
    else:
        for seed in seeds:
            for tau in taus:
                run_id = f"{args.tier}_tau{tau:.2f}_seed{seed}"

                if args.resume and run_id in all_results["experiments"]:
                    print(f"\n  [SKIP] {run_id} already completed")
                    continue

                ret = run_single(
                    tau=tau, seed=seed, cfg=cfg, tier=args.tier, base=args.base,
                    train_data=train_data, val_data=val_data,
                    work_dir=work_dir, dry_run=args.dry_run,
                    tokenizer=tok, eval_16k=args.eval_16k,
                )
                _process_run_result(ret, run_id)

    # Waterbed analysis
    if r_sweep_mode:
        # Compare each r<32 against r=32 (geometric baseline)
        K = cfg["head_dim"] // 2
        for seed in seeds:
            baseline_id = f"{args.tier}_r{K}_tau{fixed_tau:.2f}_seed{seed}"
            baseline_ppl = all_results["experiments"].get(baseline_id, {}).get("ppl", {})
            if not baseline_ppl:
                continue
            for r_val in r_values:
                if r_val >= K:
                    continue
                run_id = f"{args.tier}_r{r_val}_tau{fixed_tau:.2f}_seed{seed}"
                run_ppl = all_results["experiments"].get(run_id, {}).get("ppl", {})
                if not run_ppl:
                    continue
                wb = waterbed_analysis(baseline_ppl, run_ppl)
                all_results["waterbed"][run_id] = wb
    else:
        # Compare each τ>0 against τ=0
        for seed in seeds:
            baseline_id = f"{args.tier}_tau0.00_seed{seed}"
            baseline_ppl = all_results["experiments"].get(baseline_id, {}).get("ppl", {})
            if not baseline_ppl:
                continue
            for tau in taus:
                if tau == 0.0:
                    continue
                run_id = f"{args.tier}_tau{tau:.2f}_seed{seed}"
                run_ppl = all_results["experiments"].get(run_id, {}).get("ppl", {})
                if not run_ppl:
                    continue
                wb = waterbed_analysis(baseline_ppl, run_ppl)
                all_results["waterbed"][run_id] = wb

    # Final save
    total_time = time.time() - t_total
    all_results["metadata"]["total_time_min"] = round(total_time / 60, 1)
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    final_path = work_dir / "results_final.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE  |  {total_time / 60:.1f} min total")
    print(f"  Results: {final_path}")
    print(f"{'='*60}")

    if all_results["waterbed"]:
        print("\n  Waterbed Analysis:")
        for rid, wb in sorted(all_results["waterbed"].items()):
            holds = "✓" if wb.get("waterbed_holds") else "✗"
            short_key = [k for k in wb if "short" in k and "change" in k]
            long_key = [k for k in wb if "long" in k and "change" in k]
            sk = wb.get(short_key[0], "?") if short_key else "?"
            lk = wb.get(long_key[0], "?") if long_key else "?"
            print(f"    {rid}: short={sk:+.1f}%  long={lk:+.1f}%  waterbed={holds}")

    print("\n  Next: summarize the sweep with the curated report path:")
    print("        docs/exp/2026-02-27_evq_tau_sweep_results.md")


if __name__ == "__main__":
    main()
