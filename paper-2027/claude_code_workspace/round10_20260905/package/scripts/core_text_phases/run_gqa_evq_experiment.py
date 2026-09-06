#!/usr/bin/env python3
"""GQA + MLA + EVQ training entry-point.

Self-contained reconstruction of the runtime-patched script invoked by:
    scripts/core_text_phases/run_350m_mla32_500m.sh    (350M MLA, L=8192, 500M tok)
    scripts/core_text_phases/run_125m_mla_v2_500m.sh   (125M MLA-v2, L=4096, 500M tok)
    scripts/core_text_phases/run_50m_mla_v2_tau_sweep.sh (50M MLA-v2 sweep)
    scripts/core_text_phases/run_125m_gqa_experiment.sh
    scripts/core_text_phases/run_phase6_gqa2_tau1p5.sh

References:
    scripts/core_text_phases/run_evq_sweep.py
        Canonical MHA training pipeline. Imported here for shared utilities
        (RMSNorm, RotaryEmbedding, MLP, evq_cosh_inv_freq, train_model,
        eval_model, load_data, etc.).
    scripts/core_text_phases/mla_patch.py
        Original runtime monkey-patch blueprint. The MLAttention class below
        is a verbatim consolidation of that patch.

Reproducibility note (Q5):
    This file is a clean consolidation of the runtime-patched pipeline that
    produced Table 19 in the paper (432M MLA, 3 seeds 42/43/88, 500M tokens,
    L_train=8192, FineWeb-Edu, base=500K). The model architecture, EVQ inv_freq
    generation, and training hyperparameters are unchanged from the originally
    trained checkpoints; only the file layout is consolidated for distribution
    (no monkey-patching needed at runtime).

Smoke test (no training, ~30s on M4 Max):
    python scripts/core_text_phases/run_gqa_evq_experiment.py \\
        --tier 350m --taus 0.0,1.414 --seeds 42 \\
        --attn_type mla --d_rope 32 --seq_len 8192 \\
        --batch_size 6 --train_tokens 5000000 --dry_run

Full primary reproduction (paper Table 19):
    bash scripts/core_text_phases/run_350m_mla32_500m.sh
    # invokes this script with --tier 350m --taus 0.0,1.414 --seeds 42
    # --attn_type mla --d_rope 32 --seq_len 8192 --train_tokens 500000000
    # (multi-seed sweep was launched separately; per-seed runs ~22h on H100)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shared MHA infrastructure: import from run_evq_sweep.py.
# This includes config templates, EVQ inv_freq builder, base model classes,
# data loading, training loop, evaluation, and seeding.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_evq_sweep import (  # noqa: E402
    DEVICE,
    TIER_CONFIGS,
    Attention,                 # MHA reference attention
    MLP,
    RMSNorm,
    RotaryEmbedding,
    RunResult,
    apply_rope,
    eval_model,
    evq_cosh_inv_freq,
    load_data,
    load_val,
    maybe_wrap_with_passkey_mix,
    set_seed,
    train_model,
)


# ---------------------------------------------------------------------------
# GQA attention: standard multi-head attention with n_kv_heads != n_heads.
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    """Grouped-query attention; query heads outnumber KV heads by an integer ratio."""

    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.nkv = int(cfg.get("n_kv_heads") or self.nh)
        if self.nh % self.nkv != 0:
            raise ValueError(
                f"num_heads ({self.nh}) must be a multiple of n_kv_heads ({self.nkv})"
            )
        self.hd = cfg["head_dim"]
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


# ---------------------------------------------------------------------------
# MLA attention (DeepSeek-V2 style decoupled-RoPE multi-head latent attention).
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    """Decoupled-RoPE multi-head latent attention.

    Architecture (matches paper Table 19):
        Q : hidden_size -> num_heads * (d_nope + d_rope), split per head
        KV latent compression: hidden_size -> kv_lora_rank
        K_nope : kv_lora_rank -> num_heads * d_nope
        V      : kv_lora_rank -> num_heads * v_head_dim
        K_rope : hidden_size -> num_heads * d_rope (decoupled, bypasses latent)
        RoPE applied only to the d_rope dimensions of Q and K.
        EVQ inv_freq has length d_rope/2 (fewer frequencies than MHA d_head/2).
    """

    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.hd = cfg["head_dim"]
        self.d_rope = int(cfg.get("d_rope", 32))
        self.d_nope = int(cfg.get("d_nope", self.hd - self.d_rope))
        self.v_hd = int(cfg.get("v_head_dim", self.hd))
        self.d_c = int(cfg.get("kv_lora_rank") or (h // 4))

        self.q_proj = nn.Linear(h, self.nh * (self.d_nope + self.d_rope), bias=False)
        self.kv_down = nn.Linear(h, self.d_c, bias=False)
        self.k_nope_up = nn.Linear(self.d_c, self.nh * self.d_nope, bias=False)
        self.v_up = nn.Linear(self.d_c, self.nh * self.v_hd, bias=False)
        self.k_rope_proj = nn.Linear(h, self.nh * self.d_rope, bias=False)
        self.o = nn.Linear(self.nh * self.v_hd, h, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.d_nope + self.d_rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)

        c_kv = self.kv_down(x)
        k_nope = self.k_nope_up(c_kv).view(B, L, self.nh, self.d_nope).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, L, self.nh, self.d_rope).transpose(1, 2)
        v = self.v_up(c_kv).view(B, L, self.nh, self.v_hd).transpose(1, 2)

        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


# ---------------------------------------------------------------------------
# Block + GPT with attn_type dispatch.
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        attn_type = cfg.get("attn_type", "mha")
        if attn_type == "mla":
            self.attn = MLAttention(cfg, rope)
        elif attn_type == "gqa" or cfg.get("n_kv_heads"):
            self.attn = GQAttention(cfg, rope)
        else:
            self.attn = Attention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """GPT body with selectable attention type and decoupled-RoPE for MLA."""

    def __init__(self, cfg: dict, inv_freq: torch.Tensor) -> None:
        super().__init__()
        self._num_layers = cfg["num_layers"]
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        # MLA's RoPE acts on d_rope dims; MHA/GQA on full head_dim.
        rope_dim = (
            cfg["d_rope"] if cfg.get("attn_type") == "mla" else cfg["head_dim"]
        )
        rope = RotaryEmbedding(rope_dim, cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList(
            [Block(cfg, rope) for _ in range(cfg["num_layers"])]
        )
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self.apply(self._init)
        # Depth-scaled init for residual projections (GPT-2 / nanoGPT convention).
        residual_scale = 1.0 / math.sqrt(2 * self._num_layers)
        for block in self.blocks:
            o = getattr(block.attn, "o", None)
            if o is not None:
                nn.init.normal_(o.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.down.weight, std=0.02 * residual_scale)
        n = sum(p.numel() for p in self.parameters())
        print(
            f"  Model params: {n / 1e6:.1f}M  "
            f"(attn={cfg.get('attn_type', 'mha')})"
        )

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
# Single-run driver (replaces run_evq_sweep.run_single to use this GPT class).
# ---------------------------------------------------------------------------

def _run_single_with_arch(
    *,
    tau: float,
    seed: int,
    cfg: dict,
    tier: str,
    base: float,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    work_dir: Path,
    dry_run: bool,
    eval_16k: bool,
    tokenizer=None,
):
    attn_type = cfg.get("attn_type", "mha")
    run_id = f"{tier}_{attn_type}_tau{tau:.3f}_seed{seed}"
    print(f"\n{'='*60}\n  RUN: {run_id}  (base={base:.0f})\n{'='*60}")

    # EVQ inv_freq: MLA uses d_rope; MHA/GQA use head_dim.
    rope_dim = cfg["d_rope"] if attn_type == "mla" else cfg["head_dim"]
    inv_freq = evq_cosh_inv_freq(rope_dim, tau, base)
    import hashlib
    inv_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]
    print(
        f"  inv_freq: rope_dim={rope_dim}  K={rope_dim // 2}  "
        f"shape={tuple(inv_freq.shape)}  hash={inv_hash}"
    )
    inv_stats = {
        "max": round(inv_freq.max().item(), 8),
        "min": round(inv_freq.min().item(), 8),
        "ratio_max_min": round((inv_freq.max() / inv_freq.min()).item(), 2),
        "sha256_16": inv_hash,
    }

    if dry_run:
        set_seed(seed)
        model = GPT(cfg, inv_freq).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  [DRY RUN] model built; params={n_params/1e6:.1f}M; "
            "skipping training & eval"
        )
        return RunResult(
            run_id=run_id,
            tau=tau,
            seed=seed,
            tier=tier,
            base=base,
            inv_freq_stats=inv_stats,
        )

    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)

    # Training (with optional passkey-mix wrapping; ratio in cfg).
    passkey_mix_ratio = float(cfg.get("passkey_mix_ratio", 0.0))
    mixed_data = maybe_wrap_with_passkey_mix(
        train_data=train_data,
        filler_tokens=val_data[:50000],
        tokenizer=tokenizer,
        seq_len=cfg["seq_len"],
        passkey_ratio=passkey_mix_ratio,
    )
    t0 = time.time()
    model = train_model(model, mixed_data, cfg, seed=seed)
    train_time = time.time() - t0

    # Multi-length PPL eval.
    eval_lengths = list(cfg["eval_lengths"])
    if eval_16k and 16384 not in eval_lengths:
        eval_lengths.append(16384)
    t1 = time.time()
    ppl = eval_model(model, val_data, eval_lengths, cfg["eval_chunks"])
    eval_time = time.time() - t1

    # Persist.
    run_dir = work_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    import numpy as _np
    _np.save(run_dir / "inv_freq.npy", inv_freq.numpy())
    with open(run_dir / "results.json", "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "tau": tau,
                "seed": seed,
                "tier": tier,
                "base": base,
                "ppl": ppl,
                "attn_type": attn_type,
                "d_rope": cfg.get("d_rope"),
                "n_kv_heads": cfg.get("n_kv_heads"),
                "kv_lora_rank": cfg.get("kv_lora_rank"),
                "train_time_sec": round(train_time, 1),
                "eval_time_sec": round(eval_time, 1),
                "inv_freq_stats": inv_stats,
            },
            f,
            indent=2,
        )

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return RunResult(
        run_id=run_id,
        tau=tau,
        seed=seed,
        tier=tier,
        base=base,
        ppl=ppl,
        inv_freq_stats=inv_stats,
        train_time_sec=round(train_time, 1),
        eval_time_sec=round(eval_time, 1),
    )


# ---------------------------------------------------------------------------
# Main / CLI.
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="GQA + MLA + EVQ training entry-point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tier",
        choices=list(TIER_CONFIGS.keys()),
        default="50m",
        help="Model size config (overridden per-arg below).",
    )
    p.add_argument(
        "--taus",
        type=str,
        default="0.0",
        help="Comma-separated EVQ τ values, e.g. '0.0,1.414'.",
    )
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--base", type=float, default=500000.0)

    # Attention selector + per-type knobs.
    p.add_argument(
        "--attn_type",
        choices=["mha", "gqa", "mla"],
        default="mha",
        help="mha (standard MHA, default), gqa (n_kv_heads), or mla (decoupled-RoPE).",
    )
    p.add_argument(
        "--n_kv_heads",
        type=int,
        default=None,
        help="GQA: number of KV heads (must divide num_heads).",
    )
    p.add_argument(
        "--d_rope",
        type=int,
        default=32,
        help="MLA: dims of decoupled RoPE per head; K = d_rope/2 frequencies.",
    )
    p.add_argument(
        "--d_nope",
        type=int,
        default=None,
        help="MLA: content dims per head (default head_dim - d_rope).",
    )
    p.add_argument(
        "--v_head_dim",
        type=int,
        default=None,
        help="MLA: V head dim (default head_dim).",
    )
    p.add_argument(
        "--kv_lora_rank",
        type=int,
        default=None,
        help="MLA: KV latent rank (default hidden_size // 4).",
    )

    # Training overrides.
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--train_tokens", type=int, default=None)
    p.add_argument(
        "--dataset",
        choices=["fineweb-edu", "tinystories"],
        default="fineweb-edu",
    )
    p.add_argument("--passkey_mix_ratio", type=float, default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile (forwarded via cfg; train_model checks cfg['compile']).",
    )
    p.add_argument("--eval_16k", action="store_true")

    # Workflow.
    p.add_argument("--work_dir", type=str, default="")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Build model + inv_freq only; skip data loading and training.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose results.json already exists.",
    )

    args = p.parse_args()

    cfg = TIER_CONFIGS[args.tier].copy()
    # Mutable list copy.
    cfg["eval_lengths"] = list(cfg["eval_lengths"])

    # CLI overrides.
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len
        cfg["max_position_embeddings"] = max(
            args.seq_len, cfg["max_position_embeddings"]
        )
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.train_tokens is not None:
        cfg["train_tokens"] = args.train_tokens
    if args.passkey_mix_ratio is not None:
        cfg["passkey_mix_ratio"] = args.passkey_mix_ratio
    cfg["compile"] = args.compile
    cfg["attn_type"] = args.attn_type

    if args.attn_type == "mla":
        cfg["d_rope"] = args.d_rope
        cfg["d_nope"] = (
            args.d_nope if args.d_nope is not None else cfg["head_dim"] - args.d_rope
        )
        cfg["v_head_dim"] = (
            args.v_head_dim if args.v_head_dim is not None else cfg["head_dim"]
        )
        cfg["kv_lora_rank"] = args.kv_lora_rank or (cfg["hidden_size"] // 4)
        print(
            f"  [MLA] d_rope={args.d_rope}  d_nope={cfg['d_nope']}  "
            f"v_head_dim={cfg['v_head_dim']}  kv_lora_rank={cfg['kv_lora_rank']}  "
            f"K={args.d_rope // 2}"
        )
    elif args.attn_type == "gqa":
        cfg["n_kv_heads"] = args.n_kv_heads or (cfg["num_heads"] // 2)
        print(
            f"  [GQA] n_kv_heads={cfg['n_kv_heads']} / "
            f"num_heads={cfg['num_heads']}"
        )

    if args.eval_16k and 16384 not in cfg["eval_lengths"]:
        cfg["eval_lengths"].append(16384)

    work_dir = Path(
        args.work_dir or f".runs/{args.tier}_{args.attn_type}_b{int(args.base)}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Data.
    tokenizer = None
    if args.dry_run:
        print("[dry_run] skipping data loading")
        train_data = torch.zeros(1, dtype=torch.long)
        val_data = torch.zeros(1, dtype=torch.long)
    else:
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"  [warn] tokenizer load failed: {e}; "
                  "using None (data loading may fail)")
        train_data = load_data(
            tokenizer,
            cfg["train_tokens"],
            cfg["seq_len"],
            args.dataset,
            cache_dir=str(work_dir),
        )
        val_data = load_val(
            tokenizer,
            dataset=args.dataset,
            cache_dir=str(work_dir),
        )

    # Sweep.
    results = []
    for tau in taus:
        for seed in seeds:
            run_id = f"{args.tier}_{args.attn_type}_tau{tau:.3f}_seed{seed}"
            results_file = work_dir / run_id / "results.json"
            if args.resume and results_file.exists():
                print(f"[resume] skipping {run_id} (results.json exists)")
                continue
            r = _run_single_with_arch(
                tau=tau,
                seed=seed,
                cfg=cfg,
                tier=args.tier,
                base=args.base,
                train_data=train_data,
                val_data=val_data,
                work_dir=work_dir,
                dry_run=args.dry_run,
                eval_16k=args.eval_16k,
                tokenizer=tokenizer,
            )
            results.append(r)

    # Summary.
    summary = work_dir / "summary.json"
    with open(summary, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[done] {len(results)} run(s); summary -> {summary}")


if __name__ == "__main__":
    main()
