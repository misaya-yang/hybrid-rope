#!/usr/bin/env python3
"""YaRN fine-tuning + evaluation for 350M MLA models.

Implements proper YaRN (official recipe):
  - NTK-by-parts frequency scaling (beta_fast=32, beta_slow=1)
  - mscale temperature correction on cos/sin embeddings
  - Short full-parameter fine-tuning on TARGET-LENGTH data
  - Then PPL evaluation at extended lengths

Three conditions per model: baseline / YaRN inference-only / YaRN + FT

Usage:
    # 8K models, extend to 16K (s=2)
    python yarn_finetune_eval.py --scale 2 --train_seq_len 8192 \
        --work_dir /root/autodl-tmp/350m_mla32_500m \
        --ft_data /root/autodl-tmp/data/train_750m_clean/test_16384.pt

    # 8K models, extend to 32K (s=4)
    python yarn_finetune_eval.py --scale 4 --train_seq_len 8192 \
        --ft_data /root/autodl-tmp/data/train_750m_clean/test_32768.pt
"""
import os, math, json, time, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from pathlib import Path

DEVICE = "cuda"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Model (MLA architecture, mirrors run_gqa_evq_experiment.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq)
        self.mscale = 1.0
        self._build(max_seq)
    def _build(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos() * self.mscale, persistent=False)
        self.register_buffer("sin_c", emb.sin() * self.mscale, persistent=False)
        self._max = seq_len
    def forward(self, L):
        if L > self._max: self._build(L)
        return self.cos_c[:L], self.sin_c[:L]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class MLAttention(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]; self.hd = cfg["head_dim"]
        self.d_rope = cfg.get("d_rope", 32); self.d_nope = self.hd - self.d_rope
        self.d_c = cfg.get("kv_lora_rank", h // 4)
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.kv_down = nn.Linear(h, self.d_c, bias=False)
        self.k_nope_up = nn.Linear(self.d_c, self.nh * self.d_nope, bias=False)
        self.v_up = nn.Linear(self.d_c, self.nh * self.hd, bias=False)
        self.k_rope_proj = nn.Linear(h, self.nh * self.d_rope, bias=False)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rope = rope
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        c_kv = self.kv_down(x)
        k_nope = self.k_nope_up(c_kv).view(B, L, self.nh, self.d_nope).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, L, self.nh, self.d_rope).transpose(1, 2)
        v = self.v_up(c_kv).view(B, L, self.nh, self.hd).transpose(1, 2)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, m = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate = nn.Linear(h, m, bias=False)
        self.up = nn.Linear(h, m, bias=False)
        self.down = nn.Linear(m, h, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = MLAttention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, cfg, inv_freq):
        super().__init__()
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope_dim = cfg.get("d_rope", cfg["head_dim"])
        rope = RotaryEmbedding(rope_dim, cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList([Block(cfg, rope) for _ in range(cfg["num_layers"])])
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
        self.use_grad_ckpt = False
    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = grad_ckpt(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.head(self.ln(x))
    def extend_rope(self, L):
        self.blocks[0].attn.rope._build(L)
    def get_rope(self):
        return self.blocks[0].attn.rope

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ---------------------------------------------------------------------------
# YaRN: NTK-by-parts + mscale (matches official jquesnelle/yarn)
# ---------------------------------------------------------------------------

def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Find RoPE dimension index for a given number of rotations within max_pos."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings):
    """Dimensions [low, high]: low=many rotations (keep), high=few rotations (interpolate)."""
    low = math.floor(find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high

def linear_ramp_mask(low, high, dim):
    """Linear ramp from 0 to 1 between low and high."""
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear, 0, 1)

def get_mscale(scale):
    """YaRN attention temperature: sqrt(1/t) where t = 0.1*ln(s)+1."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def apply_yarn_scaling(orig_inv_freq, scale, original_max_pos, base,
                       beta_fast=32, beta_slow=1):
    """Apply YaRN NTK-by-parts scaling. Returns (new_inv_freq, mscale)."""
    dim = len(orig_inv_freq) * 2  # full RoPE dim

    inv_freq_extrapolation = orig_inv_freq.clone()
    inv_freq_interpolation = orig_inv_freq / scale

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_pos)

    # mask: 1 = keep original (high freq), 0 = interpolate (low freq)
    inv_freq_mask = 1.0 - linear_ramp_mask(low, high, dim // 2).to(orig_inv_freq.device)

    new_inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    mscale = get_mscale(scale)

    # Debug: print which channels are interpolated vs kept
    n_interp = (inv_freq_mask < 0.5).sum().item()
    n_keep = (inv_freq_mask >= 0.5).sum().item()
    print(f"  [YaRN] scale={scale}, low={low}, high={high}, "
          f"interpolated={n_interp}, kept={n_keep}, mscale={mscale:.4f}")

    return new_inv_freq, mscale

# ---------------------------------------------------------------------------
# EVQ-Cosh inv_freq
# ---------------------------------------------------------------------------

def evq_cosh_inv_freq(head_dim, tau, base=500000.0):
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = idx / (K - 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
    return (base ** phi).reciprocal().float()

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, val_data, lengths, n_chunks=8, seed=9999):
    model.eval()
    model.extend_rope(max(lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE)
    rng = np.random.RandomState(seed)
    results = {}
    for L in lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            print(f"  L={L}: val_data too short ({len(val_data)} tokens)")
            continue
        n_avail = max(1, max_start // L)
        offsets = sorted(rng.choice(max_start, size=min(n_chunks, n_avail), replace=False))
        for offset in offsets:
            chunk = val_data[offset:offset+L].unsqueeze(0).to(DEVICE)
            try:
                with ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  L={L}: OOM"); del chunk; torch.cuda.empty_cache(); break
                raise
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[L] = round(ppl, 3)
            print(f"  L={L}: PPL={ppl:.3f} ({len(losses)} chunks)")
    return results

# ---------------------------------------------------------------------------
# Fine-tuning (YaRN official recipe adapted for 350M)
# ---------------------------------------------------------------------------

def finetune_yarn(model, ft_data, target_seq_len, steps=500, lr=2e-6,
                  warmup=50, batch_size=2, short_data=None, mix_ratio=0.5):
    """Full-parameter fine-tuning on target-length data after YaRN scaling.

    Follows YaRN paper: AdamW, short schedule, full params.
    ft_data: pre-chunked tensor of shape (n_chunks, target_seq_len) or flat 1D.
    short_data: original training-length data to mix in (prevents catastrophic forgetting).
    mix_ratio: fraction of batches sampled from short_data (default 0.5 = 50/50 mix).
    """
    model.train()
    model.extend_rope(target_seq_len + 100)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Handle both 2D (pre-chunked) and 1D (flat) data
    if ft_data.dim() == 2 and ft_data.shape[1] == target_seq_len:
        chunks = ft_data.long()
        print(f"  FT data: {chunks.shape[0]} pre-chunked sequences @ {target_seq_len}")
    elif ft_data.dim() == 1:
        n_chunks = ft_data.numel() // target_seq_len
        chunks = ft_data[:n_chunks * target_seq_len].long().view(n_chunks, target_seq_len)
        print(f"  FT data: reshaped to {n_chunks} chunks @ {target_seq_len}")
    else:
        # 2D but different seq_len — flatten and rechunk
        flat = ft_data.reshape(-1)
        n_chunks = flat.numel() // target_seq_len
        chunks = flat[:n_chunks * target_seq_len].long().view(n_chunks, target_seq_len)
        print(f"  FT data: rechunked to {n_chunks} chunks @ {target_seq_len}")

    # Prepare short (in-distribution) data for mixing
    short_chunks = None
    if short_data is not None and mix_ratio > 0:
        train_seq = short_data.shape[1] if short_data.dim() == 2 else target_seq_len // 2
        if short_data.dim() == 2:
            short_chunks = short_data.long()
        else:
            n_sc = short_data.numel() // train_seq
            short_chunks = short_data[:n_sc * train_seq].long().view(n_sc, train_seq)
        print(f"  Mix data: {short_chunks.shape[0]} short sequences @ {short_chunks.shape[1]} "
              f"(mix_ratio={mix_ratio:.0%})")

    n_chunks = chunks.shape[0]
    if n_chunks == 0:
        print("  ERROR: no FT data available!")
        return model

    if n_chunks < batch_size:
        print(f"  WARNING: only {n_chunks} chunks, reducing batch to {n_chunks}")
        batch_size = n_chunks

    t0 = time.time()
    rng = torch.Generator().manual_seed(42)

    for s in range(steps):
        # Cosine LR with warmup
        if s < warmup:
            cur_lr = lr * (s + 1) / warmup  # avoid lr=0 at step 0
        else:
            progress = (s - warmup) / max(steps - warmup, 1)
            cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Random sample batch (mix long + short data)
        use_short = (short_chunks is not None and
                     torch.rand(1, generator=rng).item() < mix_ratio)
        if use_short:
            idx = torch.randint(0, short_chunks.shape[0], (batch_size,), generator=rng)
            batch = short_chunks[idx].to(DEVICE)
        else:
            idx = torch.randint(0, n_chunks, (batch_size,), generator=rng)
            batch = chunks[idx].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=DTYPE):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 50 == 0 or s == steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"    step {s}/{steps}  loss={loss.item():.4f}  "
                  f"lr={cur_lr:.2e}  mem={mem:.1f}GB  ETA={eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"  Fine-tuning done in {elapsed/60:.1f} min ({steps} steps)")
    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YaRN fine-tuning + eval for 350M MLA")
    parser.add_argument("--scale", type=float, default=2.0, help="YaRN scale factor (2=2x ctx)")
    parser.add_argument("--ft_steps", type=int, default=500, help="Fine-tuning steps")
    parser.add_argument("--ft_lr", type=float, default=2e-6, help="Fine-tuning peak LR")
    parser.add_argument("--ft_batch", type=int, default=2, help="Fine-tuning batch size")
    parser.add_argument("--ft_warmup", type=int, default=50, help="Fine-tuning warmup steps")
    parser.add_argument("--ft_mix", type=float, default=0.5,
                        help="Fraction of FT batches from original-length data (0=no mix)")
    parser.add_argument("--train_seq_len", type=int, default=8192, help="Original training seq len")
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/350m_mla32_500m")
    parser.add_argument("--ft_data", type=str, default=None,
                        help="Path to target-length FT data (.pt). "
                             "Should be (n, target_seq_len) or flat 1D tensor. "
                             "If not provided, uses work_dir training data reshaped.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_suffix", type=str, default="",
                        help="Checkpoint suffix, e.g. '_50pct' loads model_50pct.pt instead of model.pt")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for FT")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Gradient checkpointing during FT (saves ~40%% VRAM, allows larger batch)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    base = 500000.0
    scale = args.scale
    train_seq = args.train_seq_len
    target_seq = int(train_seq * scale)

    # Eval at: original, target, 2x target, 4x target (capped at 65536)
    eval_lengths = sorted(set([train_seq, target_seq, target_seq * 2, min(target_seq * 4, 65536)]))
    eval_lengths = [L for L in eval_lengths if L <= 65536]

    cfg = {
        "vocab_size": 50304, "hidden_size": 1024, "num_layers": 24,
        "num_heads": 16, "head_dim": 64, "intermediate_size": 4096,
        "max_position_embeddings": train_seq, "attn_type": "mla",
        "d_rope": 32, "kv_lora_rank": 256,
    }

    print(f"{'='*70}")
    print(f"  YaRN Fine-tuning Experiment")
    print(f"  scale={scale}, train_seq={train_seq} -> target_seq={target_seq}")
    print(f"  ft_steps={args.ft_steps}, ft_lr={args.ft_lr}, ft_batch={args.ft_batch}, "
          f"ft_warmup={args.ft_warmup}, ft_mix={args.ft_mix}")
    print(f"  eval_lengths={eval_lengths}")
    print(f"  seed={args.seed}, compile={args.compile}")
    print(f"  ft_data={'auto' if args.ft_data is None else args.ft_data}")
    print(f"{'='*70}")

    # --- Load eval data ---
    val_path = work_dir / "val_fineweb-edu_5000000.pt"
    val_data = torch.load(val_path, weights_only=True).long()
    print(f"Val data: {val_data.numel()/1e6:.1f}M tokens (for eval)")

    # --- Load FT data (target length) ---
    if args.ft_data and Path(args.ft_data).exists():
        ft_raw = torch.load(args.ft_data, weights_only=True)
        print(f"FT data loaded: {args.ft_data}, shape={ft_raw.shape}, dtype={ft_raw.dtype}")
    else:
        # Fallback: load training data and let finetune_yarn reshape it
        train_path = work_dir / f"train_fineweb-edu_500000000_{train_seq}.pt"
        if not train_path.exists():
            # Try 1B version
            train_path = work_dir / f"train_fineweb-edu_1000000000_{train_seq}.pt"
        if train_path.exists():
            ft_raw = torch.load(train_path, weights_only=True)
            print(f"FT data (from train): shape={ft_raw.shape}")
        else:
            print(f"WARNING: no FT data found, using val data")
            ft_raw = val_data

    # --- Load short (in-distribution) data for mixing ---
    short_data = None
    if args.ft_mix > 0:
        short_path = work_dir / f"train_fineweb-edu_500000000_{train_seq}.pt"
        if not short_path.exists():
            short_path = work_dir / f"train_fineweb-edu_1000000000_{train_seq}.pt"
        if short_path.exists():
            short_data = torch.load(short_path, weights_only=True)
            print(f"Short data loaded for mixing: shape={short_data.shape}")
        else:
            print(f"WARNING: short data not found, skipping mix")

    geo_inv = evq_cosh_inv_freq(32, 0.0, base)
    evq_inv = evq_cosh_inv_freq(32, 1.414, base)

    all_results = {}

    for tau_name, inv_freq, tau_str in [("GEO", geo_inv, "0.00"), ("EVQ", evq_inv, "1.41")]:
        ckpt_name = f"model{args.ckpt_suffix}.pt"
        ckpt_path = work_dir / f"350m_tau{tau_str}_seed{args.seed}" / ckpt_name
        if not ckpt_path.exists():
            print(f"\n  [SKIP] {ckpt_path} not found")
            continue

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        print(f"\n{'='*70}")
        print(f"  {tau_name} seed={args.seed}")
        print(f"{'='*70}")

        # --- 1. Baseline (no YaRN, no FT) ---
        print(f"\n--- {tau_name} baseline ---")
        cleanup_gpu()
        model = GPT(cfg, inv_freq)
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE)
        ppl_base = eval_ppl(model, val_data, eval_lengths)
        all_results[f"{tau_name}_baseline"] = ppl_base
        del model; cleanup_gpu()

        # --- 2. YaRN inference-only (no FT) ---
        print(f"\n--- {tau_name}+YaRN(s={scale:.0f}) inference-only ---")
        model = GPT(cfg, inv_freq)
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE)
        yarn_inv, mscale = apply_yarn_scaling(inv_freq, scale, train_seq, base)
        rope = model.get_rope()
        rope.inv_freq.copy_(yarn_inv)
        rope.mscale = mscale
        rope._build(max(eval_lengths) + 100)
        ppl_yarn_noft = eval_ppl(model, val_data, eval_lengths)
        all_results[f"{tau_name}+YaRN_no_ft"] = ppl_yarn_noft
        del model; cleanup_gpu()

        # --- 3. YaRN + fine-tuning (proper recipe) ---
        print(f"\n--- {tau_name}+YaRN(s={scale:.0f})+FT({args.ft_steps}steps) ---")
        model = GPT(cfg, inv_freq)
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE)

        # Apply YaRN scaling before FT
        yarn_inv, mscale = apply_yarn_scaling(inv_freq, scale, train_seq, base)
        rope = model.get_rope()
        rope.inv_freq.copy_(yarn_inv)
        rope.mscale = mscale

        if args.grad_ckpt:
            model.use_grad_ckpt = True
            print("  [grad_ckpt] Gradient checkpointing enabled")

        if args.compile:
            print("  [compile] Applying torch.compile...")
            model = torch.compile(model, mode="default")

        # Fine-tune
        model = finetune_yarn(model, ft_raw, target_seq,
                              steps=args.ft_steps, lr=args.ft_lr,
                              warmup=args.ft_warmup, batch_size=args.ft_batch,
                              short_data=short_data, mix_ratio=args.ft_mix)

        # Eval (disable grad_ckpt, unwrap compile for extend_rope)
        eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        eval_model.use_grad_ckpt = False
        ppl_yarn_ft = eval_ppl(eval_model, val_data, eval_lengths)
        all_results[f"{tau_name}+YaRN+FT"] = ppl_yarn_ft
        del model, eval_model; cleanup_gpu()

        del sd

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  RESULTS: YaRN FT (seed={args.seed}, scale={scale}x, "
          f"{train_seq}->{target_seq})")
    print(f"{'='*80}\n")

    header = f"{'Method':<30}" + "".join(f"{'PPL@'+str(L//1024)+'K':>12}" for L in eval_lengths)
    print(header)
    print("-" * len(header))
    for method, ppls in all_results.items():
        row = f"{method:<30}"
        for L in eval_lengths:
            v = ppls.get(L, float("nan"))
            if not math.isnan(v):
                row += f"{v:>12.1f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # Relative to GEO baseline
    geo_base = all_results.get("GEO_baseline", {})
    if geo_base:
        print(f"\n--- Relative to GEO baseline (%) ---")
        header2 = f"{'Method':<30}" + "".join(f"{'d@'+str(L//1024)+'K':>12}" for L in eval_lengths)
        print(header2)
        print("-" * len(header2))
        for method, ppls in all_results.items():
            if method == "GEO_baseline": continue
            row = f"{method:<30}"
            for L in eval_lengths:
                g = geo_base.get(L)
                v = ppls.get(L)
                if g and v and not math.isnan(v):
                    delta = (v / g - 1) * 100
                    row += f"{delta:>+11.1f}%"
                else:
                    row += f"{'—':>12}"
            print(row)

    # Key comparison: FT synergy
    geo_ft = all_results.get("GEO+YaRN+FT", {})
    evq_ft = all_results.get("EVQ+YaRN+FT", {})
    if geo_ft and evq_ft and target_seq in geo_ft and target_seq in evq_ft:
        synergy = (evq_ft[target_seq] / geo_ft[target_seq] - 1) * 100
        print(f"\n  ** EVQ+YaRN+FT vs GEO+YaRN+FT @ {target_seq//1024}K: {synergy:+.1f}% **")

    # Save
    ckpt_tag = args.ckpt_suffix.lstrip("_") if args.ckpt_suffix else "100pct"
    out_path = work_dir / f"yarn_ft_s{scale:.0f}_seed{args.seed}_{ckpt_tag}_results.json"
    out = {k: {str(kk): vv for kk, vv in v.items()} for k, v in all_results.items()}
    out["config"] = {
        "scale": scale, "ft_steps": args.ft_steps, "ft_lr": args.ft_lr,
        "ft_batch": args.ft_batch, "train_seq": train_seq, "target_seq": target_seq,
        "seed": args.seed, "ft_data": str(args.ft_data),
        "beta_fast": 32, "beta_slow": 1,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
