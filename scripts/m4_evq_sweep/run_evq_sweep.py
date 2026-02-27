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
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
}

# ---------------------------------------------------------------------------
# EVQ-cosh frequency builder (from rope/schedules.py)
# ---------------------------------------------------------------------------

def evq_cosh_inv_freq(
    head_dim: int, tau: float, base: float = 500000.0
) -> torch.Tensor:
    """φ_k(τ) = 1 - (1/τ) arcsinh((1-u_k) sinh(τ))

    Returns inv_freq of shape (head_dim // 2,) in float32.
    """
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = idx / float(K)
    if abs(tau) < 1e-8:
        phi = u  # geometric limit (Theorem 2)
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
    inv = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    return inv.float()


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
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope = RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList(
            [Block(cfg, rope) for _ in range(cfg["num_layers"])]
        )
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self.apply(self._init)
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
# Data loading (TinyStories from HuggingFace)
# ---------------------------------------------------------------------------

def load_data(tokenizer, max_tokens: int, seq_len: int) -> torch.Tensor:
    from datasets import load_dataset

    print(f"  Loading TinyStories train ({max_tokens / 1e6:.0f}M tokens)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ids: List[int] = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    n = len(ids) // seq_len
    print(f"  Got {n} chunks ({len(ids) / 1e6:.1f}M tokens)")
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def load_val(tokenizer, max_tokens: int = 5_000_000) -> torch.Tensor:
    from datasets import load_dataset

    print("  Loading validation data...")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids: List[int] = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    print(f"  Val tokens: {len(ids) / 1e6:.1f}M")
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: GPT,
    data: torch.Tensor,
    cfg: dict,
    seed: int = 42,
) -> GPT:
    model.train()
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    steps = len(data) // batch_size
    warmup = int(steps * 0.02)
    torch.manual_seed(seed)
    perm = torch.randperm(len(data))
    t0 = time.time()

    for s in range(steps):
        batch = data[perm[s * batch_size : (s + 1) * batch_size]].to(DEVICE)

        # Cosine LR with warmup
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = lr * 0.5 * (
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
) -> Dict[str, float]:
    model.eval()
    model.extend_rope(max(eval_lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    results = {}
    for L in eval_lengths:
        losses = []
        for i in range(eval_chunks):
            if (i + 1) * L > len(val_data):
                break
            chunk = val_data[i * L : (i + 1) * L].unsqueeze(0).to(DEVICE)
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
) -> RunResult:
    run_id = f"{tier}_tau{tau:.2f}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}  (base={base:.0f})")
    print(f"{'='*60}")

    # Build inv_freq
    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)
    print(f"  inv_freq: max={inv_freq.max().item():.6f}  min={inv_freq.min().item():.8f}")

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
    }

    if dry_run:
        print("  [DRY RUN] skipping training & eval")
        return RunResult(
            run_id=run_id, tau=tau, seed=seed, tier=tier, base=base,
            phase_collision=pc, inv_freq_stats=inv_stats,
        )

    # Train
    torch.manual_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    t0 = time.time()
    model = train_model(model, train_data, cfg, seed=seed)
    train_time = time.time() - t0

    # Eval
    t1 = time.time()
    ppl = eval_model(model, val_data, cfg["eval_lengths"], cfg["eval_chunks"])
    eval_time = time.time() - t1

    # Save checkpoint
    run_dir = work_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.numpy())

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
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EVQ τ-sweep on M4 Max")
    parser.add_argument("--tier", choices=["50m", "125m", "350m"], default="50m")
    parser.add_argument("--taus", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--base", type=float, default=500000.0,
                        help="RoPE base (500000=Llama-3, 10000=classic)")
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build models & inv_freq only, skip training")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have results")
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = TIER_CONFIGS[args.tier].copy()

    # Device-specific adjustments
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
        if vram_gb < 40:
            if args.tier == "50m":
                cfg["batch_size"] = 8
            elif args.tier == "125m":
                cfg["batch_size"] = 4
            elif args.tier == "350m":
                cfg["batch_size"] = 2
                cfg["eval_lengths"] = [2048, 4096, 8192]
            print(f"  [CUDA <40GB] Adjusted {args.tier}: batch={cfg['batch_size']}, eval={cfg['eval_lengths']}")

    if not args.work_dir:
        args.work_dir = str(Path.home() / "evq_m4_sweep")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  EVQ τ-SWEEP  |  tier={args.tier}  |  taus={taus}")
    print(f"  device={DEVICE}  dtype={DTYPE}  base={args.base}")
    print(f"  work_dir={work_dir}")
    print(f"{'#'*60}\n")

    # Load tokenizer & data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    if not args.dry_run:
        train_data = load_data(tok, cfg["train_tokens"], cfg["seq_len"])
        val_data = load_val(tok)
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
                "taus": taus,
                "seeds": seeds,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "experiments": {},
            "waterbed": {},
        }

    # Run sweep
    t_total = time.time()
    for seed in seeds:
        for tau in taus:
            run_id = f"{args.tier}_tau{tau:.2f}_seed{seed}"

            if args.resume and run_id in all_results["experiments"]:
                print(f"\n  [SKIP] {run_id} already completed")
                continue

            result = run_single(
                tau=tau, seed=seed, cfg=cfg, tier=args.tier, base=args.base,
                train_data=train_data, val_data=val_data,
                work_dir=work_dir, dry_run=args.dry_run,
            )
            all_results["experiments"][run_id] = asdict(result)

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
