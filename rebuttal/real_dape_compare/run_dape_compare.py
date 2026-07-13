#!/usr/bin/env python3
"""Honest PE-dominant comparison: Geo / EVQ / free-inv_freq / Kerple / DAPE-ish.

Identity rules (hard):
  - free_inv_freq  == historical paper Table-4 row mislabeled "DAPE"
  - dape_kerple_mlp == Zheng-inspired Kerple + attention-score MLP
  - NEVER write free_inv_freq results under the name DAPE

This script is self-contained for model+train+eval. EVQ schedule prefers the
canonical API in scripts.lib.rope.schedules when importable.

Usage:
  python rebuttal/real_dape_compare/run_dape_compare.py --smoke
  python rebuttal/real_dape_compare/run_dape_compare.py --dry-run
  python rebuttal/real_dape_compare/run_dape_compare.py --protocol p1 --seeds 42 \\
      --train-cache ... --val-cache ... --work /tmp/dape_p1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHOD_IDENTITY = {
    "geo": {
        "identity": "geometric_or_midpoint_geometric_rope",
        "extra_params": 0,
        "not": ["dape", "free_inv_freq"],
    },
    "evq": {
        "identity": "evq_cosh_fixed_tau",
        "extra_params": 0,
        "not": ["dape", "learnable_operator"],
    },
    "free_inv_freq": {
        "identity": "learnable_inv_freq_32",
        "extra_params": 32,
        "not": [
            "zheng2024_dape",
            "kerple_mlp",
            "paper_table4_name_dape_is_wrong",
        ],
        "historical_paper_label": "DAPE (incorrect)",
    },
    "kerple": {
        "identity": "kerple_bias_only",
        "extra_params": "2*n_heads (p,a per head)",
        "not": ["full_dape_mlp"],
    },
    "dape_kerple_mlp": {
        "identity": "zheng_inspired_kerple_plus_attn_mlp",
        "extra_params": "kerple + 2-layer MLP on (attn,kerple) features",
        "not": [
            "zheng2024_official_gpt_neox",
            "free_inv_freq_32",
            "paper_table4_row_dape",
        ],
    },
}

ALL_METHODS = list(METHOD_IDENTITY.keys())


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device_dtype() -> Tuple[str, torch.dtype, bool]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16, True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32, False
    return "cpu", torch.float32, False


DEVICE, DTYPE, USE_AUTOCAST = get_device_dtype()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha12(t: torch.Tensor) -> str:
    arr = t.detach().cpu().float().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:12]


def git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def geometric_inv_freq(head_dim: int = 64, base: float = 500_000.0) -> torch.Tensor:
    try:
        from scripts.lib.rope.schedules import geometric_inv_freq as _g

        return _g(head_dim, base).float()
    except Exception:
        n = head_dim // 2
        return torch.tensor(
            [1.0 / (base ** (2 * i / head_dim)) for i in range(n)],
            dtype=torch.float32,
        )


def midpoint_geometric_inv_freq(head_dim: int = 64, base: float = 500_000.0) -> torch.Tensor:
    """Match EVQ τ→0 midpoint grid used in core text experiments."""
    return evq_cosh_inv_freq(head_dim=head_dim, tau=0.0, base=base)


def evq_cosh_inv_freq(
    head_dim: int = 64, tau: float = 5.0, base: float = 500_000.0
) -> torch.Tensor:
    try:
        from scripts.lib.rope.schedules import evq_cosh_inv_freq as _e

        return _e(head_dim=head_dim, tau=tau, base=base).float()
    except Exception:
        if abs(tau) < 1e-8:
            k = head_dim // 2
            u = (torch.arange(k, dtype=torch.float64) + 0.5) / float(k)
            return (1.0 / (base ** u)).float()
        k = head_dim // 2
        u = (torch.arange(k, dtype=torch.float64) + 0.5) / float(k)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
        return (1.0 / (base ** phi)).float()


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(orig)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_pos: int,
        inv_freq: torch.Tensor,
        learnable: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.learnable = learnable
        if learnable:
            # Parameter lives on GPT; this module only stores a handle via set_inv_freq
            self.inv_freq = inv_freq  # type: ignore[assignment]
        else:
            self.register_buffer("inv_freq", inv_freq.float(), persistent=True)
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._built_for = 0
        # Defer cache build until first forward (correct device after .to())

    def set_inv_freq(self, inv_freq: torch.Tensor) -> None:
        self.inv_freq = inv_freq
        self._built_for = 0
        self._cos = None
        self._sin = None

    def _build(self, seq_len: int) -> None:
        if (
            not self.learnable
            and seq_len <= self._built_for
            and self._cos is not None
            and self._cos.device == self.inv_freq.device
        ):
            return
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.float())
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._built_for = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.learnable:
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.float())
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos(), emb.sin()
        self._build(seq_len)
        assert self._cos is not None and self._sin is not None
        return self._cos[:seq_len], self._sin[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, D)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos_h, sin_h = cos[..., : d // 2], sin[..., : d // 2]
    o1 = x1 * cos_h - x2 * sin_h
    o2 = x1 * sin_h + x2 * cos_h
    return torch.cat([o1, o2], dim=-1)


class MLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.up = nn.Linear(h, i, bias=False)
        self.gate = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class KerpleBias(nn.Module):
    """Kerple: -p * log(1 + a * |m-n|), per-head p, a."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.log_p = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.log_a = nn.Parameter(torch.zeros(n_heads, 1, 1))
        nn.init.normal_(self.log_p, mean=0.0, std=0.02)
        nn.init.normal_(self.log_a, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, device=self.log_p.device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        p = F.softplus(self.log_p)
        a = F.softplus(self.log_a)
        return -p * torch.log1p(a * dist.unsqueeze(0))


class DAPERefine(nn.Module):
    """Zheng-inspired: MLP(concat(attn, kerple)) residual + kerple add."""

    def __init__(self, n_heads: int, hidden_mult: int = 2):
        super().__init__()
        self.kerple = KerpleBias(n_heads)
        hidden = n_heads * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(2 * n_heads, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, h, l, _ = attn_scores.shape
        kerple = self.kerple(seq_len).unsqueeze(0).expand(b, -1, -1, -1)
        attn_t = attn_scores.permute(0, 2, 3, 1)
        kerple_t = kerple.permute(0, 2, 3, 1)
        refined = self.net(torch.cat([attn_t, kerple_t], dim=-1)).permute(0, 3, 1, 2)
        return attn_scores + kerple + refined


class Attention(nn.Module):
    def __init__(
        self,
        cfg: dict,
        rope: RotaryEmbedding,
        mode: str = "sdpa",
    ):
        super().__init__()
        h, n, d = cfg["hidden_size"], cfg["num_heads"], cfg["head_dim"]
        self.n_heads = n
        self.head_dim = d
        self.mode = mode  # sdpa | kerple | dape
        self.qkv = nn.Linear(h, 3 * n * d, bias=False)
        self.o = nn.Linear(n * d, h, bias=False)
        self.rope = rope
        if mode == "kerple":
            self.kerple = KerpleBias(n)
        elif mode == "dape":
            self.dape = DAPERefine(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(l)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.mode == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.mode == "kerple":
                scores = scores + self.kerple(l).unsqueeze(0)
            elif self.mode == "dape":
                scores = self.dape(scores, l)
            mask = torch.triu(
                torch.ones(l, l, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask[None, None], float("-inf"))
            out = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(b, l, -1))


class Block(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding, attn_mode: str):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = Attention(cfg, rope, mode=attn_mode)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        cfg: dict,
        inv_freq: torch.Tensor,
        attn_mode: str = "sdpa",
        learnable_inv_freq: bool = False,
        free_inv_lr_mult: float = 100.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.attn_mode = attn_mode
        self.free_inv_lr_mult = free_inv_lr_mult
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.learnable_inv_freq = learnable_inv_freq
        if learnable_inv_freq:
            self.inv_freq_param = nn.Parameter(inv_freq.float().clone())
            self.rope = RotaryEmbedding(
                cfg["head_dim"],
                cfg["max_position_embeddings"],
                self.inv_freq_param,
                learnable=True,
            )
        else:
            self.register_buffer(
                "inv_freq_buf", inv_freq.float().clone(), persistent=True
            )
            self.rope = RotaryEmbedding(
                cfg["head_dim"],
                cfg["max_position_embeddings"],
                self.inv_freq_buf,
                learnable=False,
            )
        self.blocks = nn.ModuleList(
            [Block(cfg, self.rope, attn_mode) for _ in range(cfg["num_layers"])]
        )
        for blk in self.blocks:
            blk.attn.rope = self.rope
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)
        scale = 1.0 / math.sqrt(2 * cfg["num_layers"])
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o.weight, std=0.02 * scale)
            nn.init.normal_(blk.mlp.down.weight, std=0.02 * scale)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_inv_freq:
            self.rope.set_inv_freq(self.inv_freq_param)
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def extend_rope(self, l: int) -> None:
        if self.learnable_inv_freq:
            self.rope.set_inv_freq(self.inv_freq_param)
        else:
            self.rope._build(l)

    def param_groups(self, lr: float) -> List[dict]:
        if not self.learnable_inv_freq:
            return [
                {"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}
            ]
        decay, no_decay, inv = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n == "inv_freq_param":
                inv.append(p)
            elif p.ndim == 1 or n.endswith("bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "lr": lr, "weight_decay": 0.01},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
        if inv:
            groups.append(
                {
                    "params": inv,
                    "lr": lr * self.free_inv_lr_mult,
                    "weight_decay": 0.0,
                }
            )
        return groups


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def protocol_p1(tokens: Optional[int] = None) -> dict:
    return {
        "name": "p1_pe_dominant_l128",
        "vocab_size": 50304,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "seq_len": 128,
        "max_position_embeddings": 128,
        "train_tokens": int(tokens or 15_000_000),
        "lr": 6e-4,
        "batch_size": 64,
        "base": 500_000.0,
        "evq_tau": 5.0,
        "eval_lengths": [128, 256, 512, 1024, 2048, 4096, 8192],
        "eval_chunks": 8,
        "free_inv_lr_mult": 100.0,
    }


def protocol_smoke() -> dict:
    p = protocol_p1(tokens=8_192)
    p["name"] = "smoke"
    p["hidden_size"] = 128
    p["num_layers"] = 2
    p["num_heads"] = 4
    p["head_dim"] = 32
    p["intermediate_size"] = 256
    p["seq_len"] = 32
    p["max_position_embeddings"] = 32
    p["batch_size"] = 4
    p["eval_lengths"] = [32, 64]
    p["eval_chunks"] = 2
    p["vocab_size"] = 256
    return p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"cache must be Tensor, got {type(data)}")
    return data


def synthetic_data(n_tokens: int, seq_len: int, vocab: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    flat = torch.from_numpy(rng.randint(0, vocab, size=n_tokens, dtype=np.int64))
    n = n_tokens // seq_len
    return flat[: n * seq_len].view(n, seq_len)


def rechunk(data: torch.Tensor, seq_len: int, max_tokens: int) -> torch.Tensor:
    flat = data.reshape(-1)[:max_tokens]
    n = len(flat) // seq_len
    if n < 1:
        raise ValueError("not enough tokens for seq_len")
    return flat[: n * seq_len].view(n, seq_len)


def get_batch(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return data[indices].long()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def eval_ppl(
    model: GPT,
    val_data: torch.Tensor,
    eval_lengths: Sequence[int],
    n_chunks: int,
) -> Dict[str, float]:
    model.eval()
    model.extend_rope(max(eval_lengths) + 8)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)
    out: Dict[str, float] = {}
    flat = val_data.reshape(-1)
    for l in eval_lengths:
        losses = []
        max_start = len(flat) - l
        if max_start <= 0:
            continue
        n = min(n_chunks, max(1, max_start // max(l, 1)))
        offsets = sorted(rng.choice(max_start, size=n, replace=False).tolist())
        for off in offsets:
            chunk = flat[off : off + l].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                    )
                losses.append(float(loss.item()))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                raise
        if losses:
            out[str(l)] = round(math.exp(sum(losses) / len(losses)), 4)
    return out


def train_model(
    model: GPT,
    train_data: torch.Tensor,
    cfg: dict,
    seed: int,
    micro_batch: int,
) -> float:
    set_seed(seed)
    bs = cfg["batch_size"]
    ga = max(1, bs // micro_batch)
    seq_len = cfg["seq_len"]
    lr = cfg["lr"]
    total_tokens = cfg["train_tokens"]
    tokens_per_step = bs * seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup = min(200, max(1, total_steps // 10))

    groups = model.param_groups(lr)
    for g in groups:
        g["_base_lr"] = g["lr"]
    try:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=(DEVICE == "cuda"))
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.1)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))

    model.train()
    n_samples = len(train_data)
    perm = torch.randperm(n_samples)
    ptr = 0
    t0 = time.time()
    log_every = max(1, total_steps // 20)

    for step in range(1, total_steps + 1):
        if step <= warmup:
            warm = step / float(warmup)
            for g in opt.param_groups:
                g["lr"] = g["_base_lr"] * warm

        opt.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(ga):
            if ptr + micro_batch > n_samples:
                perm = torch.randperm(n_samples)
                ptr = 0
            idx = perm[ptr : ptr + micro_batch]
            ptr += micro_batch
            batch = get_batch(train_data, idx).to(DEVICE)
            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
                )
                loss_s = loss / ga
            scaler.scale(loss_s).backward()
            accum += float(loss.item()) / ga
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step > warmup:
            sched.step()
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            tps = (step * tokens_per_step) / max(elapsed, 1e-6)
            print(
                f"    step {step}/{total_steps} loss={accum:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} {tps/1e6:.3f}Mtok/s",
                flush=True,
            )
    return time.time() - t0


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def method_to_model_cfg(method: str) -> Tuple[str, bool]:
    """Return (attn_mode, learnable_inv_freq)."""
    if method in ("geo", "evq"):
        return "sdpa", False
    if method == "free_inv_freq":
        return "sdpa", True
    if method == "kerple":
        return "kerple", False
    if method == "dape_kerple_mlp":
        return "dape", False
    raise ValueError(method)


def build_inv_freq(method: str, cfg: dict) -> torch.Tensor:
    base = cfg["base"]
    d = cfg["head_dim"]
    if method == "geo":
        # midpoint geometric to match EVQ τ=0 family used in core sweeps
        return midpoint_geometric_inv_freq(d, base)
    if method == "evq":
        return evq_cosh_inv_freq(d, tau=float(cfg["evq_tau"]), base=base)
    if method == "free_inv_freq":
        # init from geometric; learn freely
        return geometric_inv_freq(d, base)
    if method in ("kerple", "dape_kerple_mlp"):
        return midpoint_geometric_inv_freq(d, base)
    raise ValueError(method)


def run_one(
    work: Path,
    method: str,
    seed: int,
    cfg: dict,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    micro_batch: int,
) -> dict:
    run_id = f"{cfg['name']}_{method}_seed{seed}"
    run_dir = work / "runs" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"  SKIP existing {run_id}")
        return json.loads(result_path.read_text())

    attn_mode, learnable = method_to_model_cfg(method)
    inv = build_inv_freq(method, cfg)
    inv_hash = sha12(inv)

    print(f"\n{'='*72}\n  RUN {run_id}\n  identity={METHOD_IDENTITY[method]['identity']}\n"
          f"  inv_freq_hash={inv_hash} attn={attn_mode} learnable_inv={learnable}\n{'='*72}",
          flush=True)

    set_seed(seed)
    model = GPT(
        cfg,
        inv,
        attn_mode=attn_mode,
        learnable_inv_freq=learnable,
        free_inv_lr_mult=float(cfg.get("free_inv_lr_mult", 100.0)),
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params/1e6:.2f}M device={DEVICE}", flush=True)

    train_time = train_model(model, train_data, cfg, seed=seed, micro_batch=micro_batch)
    print("  eval...", flush=True)
    ppl = eval_ppl(model, val_data, cfg["eval_lengths"], cfg["eval_chunks"])

    run_dir.mkdir(parents=True, exist_ok=True)
    # save inv_freq snapshot
    if learnable:
        learned = model.inv_freq_param.detach().cpu()
        torch.save(learned, run_dir / "learned_inv_freq.pt")
        final_hash = sha12(learned)
    else:
        torch.save(inv.cpu(), run_dir / "inv_freq.pt")
        final_hash = inv_hash

    result = {
        "run_id": run_id,
        "method_id": method,
        "identity": METHOD_IDENTITY[method]["identity"],
        "identity_not": METHOD_IDENTITY[method]["not"],
        "historical_paper_label": METHOD_IDENTITY[method].get("historical_paper_label"),
        "seed": seed,
        "protocol": cfg["name"],
        "ppl": ppl,
        "train_time_sec": round(train_time, 2),
        "inv_freq_hash_init": inv_hash,
        "inv_freq_hash_final": final_hash,
        "n_params": n_params,
        "attn_mode": attn_mode,
        "learnable_inv_freq": learnable,
        "evq_tau": cfg.get("evq_tau"),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "git_sha": git_sha(),
        "platform": platform.platform(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  done ppl={ppl} time={train_time:.1f}s", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(results: List[dict], work: Path) -> dict:
    by_method: Dict[str, List[dict]] = {}
    for r in results:
        by_method.setdefault(r["method_id"], []).append(r)

    rows = []
    for m, rs in by_method.items():
        # prefer 8192 then max length
        def pick(r):
            p = r.get("ppl") or {}
            if "8192" in p:
                return p["8192"]
            if not p:
                return None
            return p[max(p.keys(), key=lambda x: int(x))]

        vals = [pick(r) for r in rs]
        vals = [v for v in vals if v is not None]
        rows.append(
            {
                "method_id": m,
                "identity": METHOD_IDENTITY[m]["identity"],
                "n_seeds": len(rs),
                "ppl_long_mean": round(sum(vals) / len(vals), 4) if vals else None,
                "ppl_long_all": vals,
                "seeds": [r["seed"] for r in rs],
            }
        )

    summary = {
        "n_results": len(results),
        "rows": rows,
        "claim_boundary": (
            "free_inv_freq is NOT Zheng DAPE. dape_kerple_mlp is Zheng-inspired "
            "Kerple+MLP, not official GPT-NeoX DAPE. Absolute win vs dape_kerple_mlp "
            "is NOT expected based on phase11b."
        ),
        "existing_phase11b_note": (
            "At L=256/100M, Geo+DAPE mean PPL@8K≈55.9 vs EVQ4+DAPE≈56.8; "
            "plain EVQ4≈254.7. See FINDINGS.md."
        ),
    }
    agg = work / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# DAPE-compare summary",
        "",
        summary["claim_boundary"],
        "",
        "| method | identity | seeds | long-PPL mean |",
        "|--------|----------|-------|---------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_id']} | {row['identity']} | {row['n_seeds']} | {row['ppl_long_mean']} |"
        )
    lines += ["", "## phase11b prior", summary["existing_phase11b_note"], ""]
    (agg / "summary.md").write_text("\n".join(lines))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", choices=["p1", "smoke"], default="p1")
    p.add_argument("--seeds", default="42")
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--work", type=Path, default=None)
    p.add_argument("--train-cache", type=Path, default=None)
    p.add_argument("--val-cache", type=Path, default=None)
    p.add_argument("--tokens", type=int, default=None)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--tau", type=float, default=None, help="Override EVQ tau (default 5.0 for p1)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny synthetic CPU/GPU run")
    p.add_argument("--list-existing-findings", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_existing_findings:
        print((PKG_DIR / "FINDINGS.md").read_text())
        return 0

    if args.smoke:
        args.protocol = "smoke"

    cfg = protocol_smoke() if args.protocol == "smoke" else protocol_p1(args.tokens)
    if args.tau is not None:
        cfg["evq_tau"] = float(args.tau)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_IDENTITY:
            print(f"Unknown method {m}; choose from {ALL_METHODS}", file=sys.stderr)
            return 2
        if m == "DAPE" or m.lower() == "dape":
            print("ERROR: method name 'DAPE' forbidden. Use free_inv_freq or dape_kerple_mlp.", file=sys.stderr)
            return 2

    work = args.work or Path(
        os.environ.get("EVQ_DAPE_WORK", str(PKG_DIR / "work" / cfg["name"]))
    )
    work = work.expanduser().resolve()

    plan = [f"{cfg['name']}_{m}_seed{s}" for m in methods for s in seeds]
    print(f"[plan] device={DEVICE} dtype={DTYPE} work={work}")
    print(f"[plan] protocol={cfg['name']} tokens={cfg['train_tokens']} L={cfg['seq_len']}")
    print(f"[plan] methods={methods} seeds={seeds}")
    print(f"[plan] runs ({len(plan)}):")
    for r in plan:
        print(f"  - {r}")

    if args.dry_run:
        print("[dry-run] exit before data/train")
        return 0

    # data
    if args.protocol == "smoke":
        train_data = synthetic_data(
            cfg["train_tokens"] + cfg["seq_len"] * 4,
            cfg["seq_len"],
            cfg["vocab_size"],
            seed=0,
        )
        val_data = synthetic_data(cfg["seq_len"] * 64, cfg["seq_len"], cfg["vocab_size"], seed=1)
        val_data = val_data.reshape(-1)
    else:
        if args.train_cache is None or args.val_cache is None:
            print(
                "ERROR: --train-cache and --val-cache required for p1 "
                "(prepare data before GPU). For wiring test use --smoke.",
                file=sys.stderr,
            )
            return 2
        train_raw = load_cache(args.train_cache)
        val_data = load_cache(args.val_cache).reshape(-1)
        train_data = rechunk(
            train_raw, cfg["seq_len"], cfg["train_tokens"] + 5 * cfg["seq_len"]
        )
        print(f"[data] train={tuple(train_data.shape)} val_flat={len(val_data)}")

    micro = args.micro_batch
    if micro is None:
        # DAPE/kerple need smaller micro due to materialised attention
        heavy = any(m in ("kerple", "dape_kerple_mlp") for m in methods)
        micro = 16 if heavy else 32
        if args.protocol == "smoke":
            micro = 4
    micro = min(micro, cfg["batch_size"])

    work.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_unix": time.time(),
        "git_sha": git_sha(),
        "cfg": cfg,
        "methods": methods,
        "seeds": seeds,
        "plan": plan,
        "method_identity": METHOD_IDENTITY,
        "train_cache": str(args.train_cache) if args.train_cache else "synthetic",
        "val_cache": str(args.val_cache) if args.val_cache else "synthetic",
        "device": DEVICE,
    }
    man_dir = work / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = []
    for method in methods:
        for seed in seeds:
            results.append(
                run_one(work, method, seed, cfg, train_data, val_data, micro)
            )

    summary = summarize(results, work)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {work / 'aggregate' / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
