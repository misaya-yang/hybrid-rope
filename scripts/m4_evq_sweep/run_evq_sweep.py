#!/usr/bin/env python3
"""EVQ τ-sweep on Apple M4 Max — theory validation for NeurIPS v5.

Usage:
    conda activate aidemo
    python scripts/m4_evq_sweep/run_evq_sweep.py --tier 50m
    python scripts/m4_evq_sweep/run_evq_sweep.py --tier 50m --taus 0.0,0.8 --dry_run

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


def yarn_inv_freq(
    head_dim: int, base: float = 500000.0,
    original_max_position: int = 128, target_max_position: int = 8192,
    beta_fast: float = 32.0, beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN frequency scaling (Peng et al. 2023).

    Three regions: high-freq (no scaling), low-freq (PI scaling),
    mid-freq (smooth interpolation).

    Returns inv_freq of shape (head_dim // 2,) in float32.
    """
    dim = head_dim
    s = target_max_position / original_max_position

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    # Wavelength boundaries
    low = math.floor(dim * math.log(original_max_position / (beta_fast * 2 * math.pi))
                     / (2 * math.log(base)))
    high = math.ceil(dim * math.log(original_max_position / (beta_slow * 2 * math.pi))
                     / (2 * math.log(base)))
    low = max(low, 0)
    high = min(high, dim // 2 - 1)

    scaled = inv_freq.clone()
    for i in range(dim // 2):
        if i < low:
            # High frequency: no scaling
            pass
        elif i > high:
            # Low frequency: PI-like linear scaling
            scaled[i] = inv_freq[i] / s
        else:
            # Middle: smooth interpolation
            t = (i - low) / max(high - low, 1)
            gamma = 1 - t
            scaled[i] = inv_freq[i] / (1 + (s - 1) * (1 - gamma))

    return scaled.float()


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


class LearnableRotaryEmbedding(nn.Module):
    """RoPE adapter wrapping LearnableEVQRoPE — recomputes every forward for gradient flow."""

    def __init__(self, head_dim: int, max_seq: int, base: float,
                 tau_init: float = 1.0, tau_lr_multiplier: float = 10.0) -> None:
        super().__init__()
        # Add project root so rope/ package is importable
        _proj_root = str(Path(__file__).resolve().parents[2])
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)
        from rope.learnable_evq import LearnableEVQRoPE
        self.evq = LearnableEVQRoPE(
            dim=head_dim, max_seq_len=max_seq, base=base,
            tau_init=tau_init, tau_lr_multiplier=tau_lr_multiplier,
        )
        self._max = max_seq

    def _extend(self, seq_len: int) -> None:
        """Extend position buffer for eval at longer sequences."""
        if seq_len > self._max:
            device = self.evq.pos.device
            new_pos = torch.arange(seq_len, dtype=torch.float64, device=device)
            self.evq.register_buffer("pos", new_pos)
            self.evq.max_seq_len = seq_len
            self._max = seq_len

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._extend(L)
        cos_half, sin_half = self.evq.get_cos_sin(L)  # (L, n_freqs=head_dim//2)
        # Double to match (L, head_dim) expected by apply_rope
        cos = torch.cat([cos_half, cos_half], dim=-1).float()
        sin = torch.cat([sin_half, sin_half], dim=-1).float()
        return cos, sin


class DAPERotaryEmbedding(nn.Module):
    """DAPE-style: d/2 independent learnable log-frequency parameters.

    Each of the d/2 frequency channels is an independent nn.Parameter (log-scale).
    Recomputes cos/sin every forward for gradient flow.
    """

    def __init__(self, head_dim: int, max_seq: int, base: float = 500000.0,
                 lr_multiplier: float = 10.0) -> None:
        super().__init__()
        n_freqs = head_dim // 2
        # Initialise as geometric RoPE: inv_freq_k = base^{-(k+0.5)/K}
        u = (torch.arange(n_freqs, dtype=torch.float64) + 0.5) / n_freqs
        log_inv_freq = -u * math.log(base)  # log(inv_freq)
        self.log_inv_freq = nn.Parameter(log_inv_freq)  # d/2 learnable params
        self.lr_multiplier = lr_multiplier
        self._max = max_seq

    def get_inv_freq(self) -> torch.Tensor:
        return torch.exp(self.log_inv_freq).float()

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = torch.exp(self.log_inv_freq)  # stay in float64 for precision
        t = torch.arange(L, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().float(), emb.sin().float()


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
    def __init__(self, cfg: dict, inv_freq: torch.Tensor,
                 learnable_rope: Optional[LearnableRotaryEmbedding] = None) -> None:
        super().__init__()
        self._num_layers = cfg["num_layers"]
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        if learnable_rope is not None:
            rope = learnable_rope
        else:
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
        rope = self.blocks[0].attn.rope
        if isinstance(rope, LearnableRotaryEmbedding):
            rope._extend(L)
        elif isinstance(rope, DAPERotaryEmbedding):
            rope._max = max(rope._max, L)  # no cache to rebuild
        else:
            rope._build(L)


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


def train_model(
    model: GPT,
    data: torch.Tensor,
    cfg: dict,
    seed: int = 42,
    tau_lr_multiplier: Optional[float] = None,
    tau_logger=None,
) -> GPT:
    model.train()
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]

    # Optimizer: separate param group for learnable PE params (raw_tau or log_inv_freq)
    # Store base_lr per group for cosine schedule
    _pe_param_names = ("raw_tau", "log_inv_freq")
    if tau_lr_multiplier is not None:
        pe_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and any(k in n for k in _pe_param_names)]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(k in n for k in _pe_param_names)]
        pe_base_lr = lr * tau_lr_multiplier
        opt = torch.optim.AdamW([
            {"params": other_params, "lr": lr, "weight_decay": 0.1},
            {"params": pe_params, "lr": pe_base_lr, "weight_decay": 0.0},
        ], betas=(0.9, 0.95))
        group_base_lrs = [lr, pe_base_lr]
        print(f"  [PE] Separate param group: lr_pe={pe_base_lr:.2e}, "
              f"n_pe_params={sum(p.numel() for p in pe_params)}")
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        group_base_lrs = [lr]

    steps = len(data) // batch_size
    warmup = int(steps * 0.02)
    set_seed(seed)
    perm = torch.randperm(len(data))
    t0 = time.time()

    # Find EVQ module for tau logging (if any)
    evq_module = None
    if tau_logger is not None:
        rope = model.blocks[0].attn.rope
        if isinstance(rope, LearnableRotaryEmbedding):
            evq_module = rope.evq

    for s in range(steps):
        batch = data[perm[s * batch_size : (s + 1) * batch_size]].to(DEVICE)

        # Cosine LR with warmup + min_lr floor (applied per group)
        for gi, g in enumerate(opt.param_groups):
            blr = group_base_lrs[gi]
            blr_min = blr * 0.1
            if s < warmup:
                g["lr"] = blr * s / max(warmup, 1)
            else:
                g["lr"] = blr_min + (blr - blr_min) * 0.5 * (
                    1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
                )

        # Forward pass — autocast on CUDA (flash attention + bf16), plain on MPS/CPU
        ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
        with ctx:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Log τ (after backward, before clip — captures raw gradient)
        if tau_logger is not None and evq_module is not None:
            tau_logger.log(s, evq_module, loss=loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            tau_str = ""
            if evq_module is not None:
                tau_str = f"  tau={evq_module.get_tau_value():.4f}"
            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"    step {s}/{steps}  loss={loss.item():.4f}  "
                f"lr={cur_lr:.2e}{tau_str}  ETA={eta / 60:.0f}min"
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
                    print(f"    L={L}: OOM on chunk {i}, stopping this length")
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
    learnable: bool = False,
    tau_init: float = 1.0,
    tau_lr_multiplier: float = 10.0,
    dape: bool = False,
    yarn: bool = False,
    yarn_target: int = 8192,
) -> RunResult:
    if yarn:
        run_id = f"{tier}_yarn_target{yarn_target}_seed{seed}"
    elif dape:
        run_id = f"{tier}_dape_lrmult{tau_lr_multiplier:.0f}_seed{seed}"
    elif learnable:
        run_id = f"{tier}_learnable_init{tau_init:.2f}_seed{seed}"
    else:
        run_id = f"{tier}_tau{tau:.2f}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}  (base={base:.0f})")
    print(f"{'='*60}")

    import hashlib

    if yarn:
        # YaRN mode: compute YaRN-scaled frequencies
        inv_freq = yarn_inv_freq(
            cfg["head_dim"], base,
            original_max_position=cfg["seq_len"],
            target_max_position=yarn_target,
        )
        print(f"  [yarn] original={cfg['seq_len']}  target={yarn_target}  scale={yarn_target/cfg['seq_len']:.0f}x")
    elif dape:
        # DAPE mode: init frequencies are geometric (τ=0)
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], 0.0, base)
        print(f"  [dape] d/2={cfg['head_dim']//2} learnable freqs  lr_mult={tau_lr_multiplier}")
    elif learnable:
        # Learnable mode: inv_freq computed from initial tau for stats/collision only
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau_init, base)
        print(f"  [learnable] tau_init={tau_init:.4f}  lr_mult={tau_lr_multiplier}")
    else:
        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)

    inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]
    print(f"  inv_freq: shape={inv_freq.shape}  max={inv_freq.max().item():.6f}  "
          f"min={inv_freq.min().item():.8f}  hash={inv_hash}")

    # Phase collision (on initial frequencies)
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
            run_id=run_id, tau=tau if not learnable else tau_init,
            seed=seed, tier=tier, base=base,
            phase_collision=pc, inv_freq_stats=inv_stats,
        )

    # Wrap training data with MixedDataset for passkey mixing (skip for short seq_len)
    if cfg["seq_len"] >= 2048:
        from eval_passkey_scratch import MixedDataset
        mixed_data = MixedDataset(
            lm_data=train_data,
            filler_tokens=val_data[:50000],
            tokenizer=tokenizer,
            passkey_ratio=0.005,
            seq_len=cfg["seq_len"],
        )
        # Log passkey mixing ratio
        import random as _rnd
        pk_count = sum(1 for i in range(min(1000, len(mixed_data)))
                       if _rnd.Random(i * 6364136223846793005 + 1).random() < 0.005)
        print(f"  [passkey] Mix ratio check: {pk_count}/1000 = {pk_count/10:.1f}%")
    else:
        mixed_data = train_data  # plain LM data for short sequences

    # Build model
    set_seed(seed)
    tau_logger = None
    if dape:
        dape_rope = DAPERotaryEmbedding(
            head_dim=cfg["head_dim"],
            max_seq=cfg["max_position_embeddings"],
            base=base,
            lr_multiplier=tau_lr_multiplier,
        ).to(DEVICE)
        model = GPT(cfg, inv_freq, learnable_rope=dape_rope).to(DEVICE)
    elif learnable:
        learnable_rope = LearnableRotaryEmbedding(
            head_dim=cfg["head_dim"],
            max_seq=cfg["max_position_embeddings"],
            base=base,
            tau_init=tau_init,
            tau_lr_multiplier=tau_lr_multiplier,
        ).to(DEVICE)
        model = GPT(cfg, inv_freq, learnable_rope=learnable_rope).to(DEVICE)
        # Import TauLogger
        _proj_root = str(Path(__file__).resolve().parents[2])
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)
        from rope.learnable_evq import TauLogger
        tau_logger = TauLogger(log_interval=50)
    else:
        model = GPT(cfg, inv_freq).to(DEVICE)

    # Parameter count validation for 500m
    n_params = sum(p.numel() for p in model.parameters())
    if tier == "500m":
        assert 450_000_000 <= n_params <= 550_000_000, \
            f"500M tier: expected 450-550M params, got {n_params/1e6:.1f}M"

    t0 = time.time()
    model = train_model(
        model, mixed_data, cfg, seed=seed,
        tau_lr_multiplier=tau_lr_multiplier if (learnable or dape) else None,
        tau_logger=tau_logger,
    )
    train_time = time.time() - t0

    # Eval: PPL
    eval_lengths = list(cfg["eval_lengths"])
    if eval_16k and 16384 not in eval_lengths:
        eval_lengths.append(16384)
    t1 = time.time()
    ppl = eval_model(model, val_data, eval_lengths, cfg["eval_chunks"])
    eval_time = time.time() - t1

    # Eval: Passkey NLL gap (skip for short seq_len — passkey requires ≥2048)
    passkey_results = {}
    if tokenizer is not None and cfg["seq_len"] >= 2048:
        print(f"\n  [passkey] NLL-gap evaluation for {run_id}")
        from eval_passkey_scratch import eval_passkey_nll_gap
        passkey_results = eval_passkey_nll_gap(
            model, tokenizer, val_data,
            lengths=[2048, 4096, 8192],
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

    # Learnable-specific saves
    final_tau = tau
    if dape:
        dape_mod = model.blocks[0].attn.rope
        learned_inv = dape_mod.get_inv_freq().detach().cpu().numpy()
        np.save(run_dir / "dape_learned_inv_freq.npy", learned_inv)
        # Compute distance from geometric init for analysis
        K = cfg["head_dim"] // 2
        u = (np.arange(K) + 0.5) / K
        geo_inv = np.float_power(base, -u)
        delta = np.abs(learned_inv - geo_inv) / (geo_inv + 1e-30)
        print(f"  [dape] Learned inv_freq: max_rel_change={delta.max():.4f}, "
              f"mean_rel_change={delta.mean():.4f}")
    elif learnable:
        evq_mod = model.blocks[0].attn.rope.evq
        final_tau = evq_mod.get_tau_value()
        # Save tau trajectory
        if tau_logger is not None:
            tau_logger.save(str(run_dir / "tau_trajectory.json"))
        # Save learned inv_freq
        learned_freqs = evq_mod.get_frequencies().detach().cpu().numpy()
        np.save(run_dir / "learned_inv_freq.npy", learned_freqs)
        # Save learned phi schedule
        learned_phi = evq_mod.get_phi_schedule().detach().cpu().numpy()
        np.save(run_dir / "learned_phi.npy", learned_phi)
        print(f"  [learnable] Final tau: {final_tau:.6f}")
        if tau_logger is not None:
            conv_std = tau_logger.get_convergence_std()
            print(f"  [learnable] Convergence std (last 20%): {conv_std:.6f}")
    else:
        np.save(run_dir / "inv_freq.npy", inv_freq.numpy())

    if passkey_results:
        import json as _json
        with open(run_dir / "passkey_nll.json", "w") as f:
            _json.dump(passkey_results, f, indent=2, ensure_ascii=False, default=str)

    # PI baseline: inference-time only, on geometric model (τ=0) — skip for learnable/dape
    pi_results = {}
    if not learnable and not dape and abs(tau) < 1e-8:
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

        # Passkey eval with PI (skip for short seq_len — passkey requires ≥2048)
        if tokenizer is not None and cfg["seq_len"] >= 2048:
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
        tau=final_tau,
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
    parser.add_argument("--learnable", action="store_true",
                        help="Use learnable τ (LearnableEVQRoPE)")
    parser.add_argument("--tau_init", type=float, default=1.0,
                        help="Initial τ for learnable mode (default: 1.0)")
    parser.add_argument("--tau_lr_mult", type=float, default=10.0,
                        help="LR multiplier for τ parameter (default: 10.0)")
    parser.add_argument("--train_tokens", type=int, default=None,
                        help="Override train_tokens (e.g. 15000000 for quick sanity check)")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override training seq_len (e.g. 128 for DAPE-style PE quality test)")
    parser.add_argument("--eval_lengths", type=str, default=None,
                        help="Override eval lengths (comma-separated, e.g. 128,256,512,1024,2048,4096,8192)")
    parser.add_argument("--dape", action="store_true",
                        help="Use DAPE-style d/2 independent learnable frequencies")
    parser.add_argument("--yarn", action="store_true",
                        help="Use YaRN frequency scaling (from-scratch training)")
    parser.add_argument("--yarn_target", type=int, default=8192,
                        help="YaRN target max position (default: 8192)")
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = TIER_CONFIGS[args.tier].copy()
    if args.train_tokens is not None:
        cfg["train_tokens"] = args.train_tokens
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len
        cfg["max_position_embeddings"] = args.seq_len
    if args.eval_lengths is not None:
        cfg["eval_lengths"] = [int(x) for x in args.eval_lengths.split(",")]

    # Device-specific adjustments (applied BEFORE short-seq scaling so the
    # auto-scale multiplier stacks on top of the device-adjusted base batch)
    if DEVICE == "mps":
        # MPS float32: no flash attention, full attention matrix materialised
        # Reduce batch + cap eval length to avoid OOM
        cfg["batch_size"] = min(cfg["batch_size"], 8)
        cfg["eval_lengths"] = [L for L in cfg["eval_lengths"] if L <= 8192]
        if args.tier == "350m":
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
            if args.tier == "500m":
                cfg["batch_size"] = 16
            elif args.tier == "350m":
                cfg["batch_size"] = 8
            if args.eval_16k and 16384 not in cfg["eval_lengths"]:
                cfg["eval_lengths"].append(16384)
            print(f"  [CUDA >=80GB] Adjusted {args.tier}: batch={cfg['batch_size']}, eval={cfg['eval_lengths']}")
        elif vram_gb < 40:
            if args.tier == "50m":
                cfg["batch_size"] = 8
            elif args.tier == "125m":
                cfg["batch_size"] = 4
            elif args.tier in ("350m", "500m"):
                cfg["batch_size"] = 2
                cfg["eval_lengths"] = [2048, 4096, 8192]
            print(f"  [CUDA <40GB] Adjusted {args.tier}: batch={cfg['batch_size']}, eval={cfg['eval_lengths']}")

    # Auto-scale batch_size for short sequences (AFTER device adjustments).
    # 128 tok uses ~16x less memory than 2048 → scale batch proportionally.
    if cfg["seq_len"] < 2048:
        scale = 2048 // cfg["seq_len"]  # e.g. 16x for 128 tokens
        cfg["batch_size"] = cfg["batch_size"] * scale
        print(f"  [short-seq] seq_len={cfg['seq_len']}, batch_size scaled {scale}x to {cfg['batch_size']}")

    if not args.work_dir:
        args.work_dir = str(Path.home() / "evq_m4_sweep")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    if args.dape:
        print(f"  DAPE (d/2 learnable freqs)  |  tier={args.tier}")
        print(f"  lr_mult={args.tau_lr_mult}  seeds={seeds}")
    elif args.learnable:
        print(f"  EVQ LEARNABLE τ  |  tier={args.tier}  |  tau_init={args.tau_init}")
        print(f"  lr_mult={args.tau_lr_mult}  seeds={seeds}")
    else:
        print(f"  EVQ τ-SWEEP  |  tier={args.tier}  |  taus={taus}")
    print(f"  seq_len={cfg['seq_len']}  train_tokens={cfg['train_tokens']/1e6:.0f}M")
    print(f"  eval_lengths={cfg['eval_lengths']}")
    print(f"  device={DEVICE}  dtype={DTYPE}  base={args.base}")
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
        all_results = {
            "metadata": {
                "tier": args.tier,
                "device": DEVICE,
                "dtype": str(DTYPE),
                "base": args.base,
                "seq_len": cfg["seq_len"],
                "train_tokens": cfg["train_tokens"],
                "taus": taus,
                "seeds": seeds,
                "dataset": args.dataset,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "experiments": {},
            "waterbed": {},
        }

    # Run sweep
    t_total = time.time()

    if args.yarn:
        # YaRN mode: single run per seed with YaRN-scaled frequencies
        run_configs = [(seed, None, False, False, True) for seed in seeds]
    elif args.dape:
        # DAPE mode: single run per seed
        run_configs = [(seed, None, False, True, False) for seed in seeds]
    elif args.learnable:
        # Learnable mode: single run per seed with given tau_init
        run_configs = [(seed, None, True, False, False) for seed in seeds]
    else:
        # Fixed-τ sweep
        run_configs = [(seed, tau, False, False, False) for seed in seeds for tau in taus]

    for seed, tau, is_learnable, is_dape, is_yarn in run_configs:
        if is_yarn:
            run_id = f"{args.tier}_yarn_target{args.yarn_target}_seed{seed}"
        elif is_dape:
            run_id = f"{args.tier}_dape_lrmult{args.tau_lr_mult:.0f}_seed{seed}"
        elif is_learnable:
            run_id = f"{args.tier}_learnable_init{args.tau_init:.2f}_seed{seed}"
        else:
            run_id = f"{args.tier}_tau{tau:.2f}_seed{seed}"

        if args.resume and run_id in all_results["experiments"]:
            print(f"\n  [SKIP] {run_id} already completed")
            continue

        ret = run_single(
            tau=tau if tau is not None else 0.0,
            seed=seed, cfg=cfg, tier=args.tier, base=args.base,
            train_data=train_data, val_data=val_data,
            work_dir=work_dir, dry_run=args.dry_run,
            tokenizer=tok, eval_16k=args.eval_16k,
            learnable=is_learnable,
            tau_init=args.tau_init,
            tau_lr_multiplier=args.tau_lr_mult,
            dape=is_dape,
            yarn=is_yarn,
            yarn_target=args.yarn_target,
        )
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

    # Waterbed analysis (compare each τ>0 against τ=0)
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

    print(f"\n  Next: python scripts/m4_evq_sweep/evq_analysis.py --input {final_path}")


if __name__ == "__main__":
    main()
