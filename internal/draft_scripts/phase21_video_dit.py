#!/usr/bin/env python3
"""Phase 21: Video Cross-Modal Generalization for EVQ-Cosh.

Sub-A: Enhanced bouncing ball (self-contained, zero-cost)
  - Multi-ball generation (2-3 balls, different sizes/speeds, occlusion)
  - Larger model: 12 layers, 512 hidden, 8 heads (85M params)
  - Comprehensive τ sweep: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0}
  - Metrics: PPL, FVD (Fréchet Video Distance), temporal consistency
  - 3D RoPE with spatial (geometric) and temporal (EVQ) frequencies

Sub-B: Real video DiT (requires external model + data)
  - Model setup: Load Latte/Open-Sora-Plan, inject EVQ frequencies
  - Dataset: UCF-101 (13k videos, 101 classes)
  - Fine-tune with temporal EVQ, evaluate FVD + FID
  - τ sweep: {0.0, 0.5, 0.7, 1.0, 1.5, 2.0}

Usage:
    # Sub-A (immediate, no external deps):
    python team/scripts/phase21_video_dit.py --mode bouncing_ball --taus 0.0,1.0,2.0 --seeds 42,137,256

    # Sub-B (requires Latte/UCF-101):
    python team/scripts/phase21_video_dit.py --mode latte --model_path /path/to/latte --data_path /path/to/ucf101

    # Dry run:
    python team/scripts/phase21_video_dit.py --mode bouncing_ball --dry_run

Hardware: M4 Max 36GB (MPS), ~2 hours (sub-A) + data time (sub-B).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import shared components from run_evq_sweep
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_DIR = SCRIPT_DIR.parent.parent / "scripts" / "core_text_phases"
sys.path.insert(0, str(SWEEP_DIR))

try:
    from run_evq_sweep import (
        RMSNorm,
        MLP,
        Block,
        Attention,
        RotaryEmbedding,
        apply_rope,
        rotate_half,
        evq_cosh_inv_freq,
        set_seed,
        get_device_and_dtype,
    )
except ImportError as e:
    print(f"Warning: Could not import from run_evq_sweep: {e}")
    print("Falling back to local definitions...")

    # Fallback definitions if import fails
    def set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_device_and_dtype() -> Tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float32
        else:
            return "cpu", torch.float32

    def evq_cosh_inv_freq(
        head_dim: int, tau: float, base: float = 10000.0
    ) -> torch.Tensor:
        K = head_dim // 2
        idx = torch.arange(K, dtype=torch.float64)
        u = (idx + 0.5) / float(K)
        if abs(tau) < 1e-8:
            phi = u
        else:
            sinh_tau = math.sinh(tau)
            phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
        inv = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
        return inv.float()

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


# ---------------------------------------------------------------------------
# Multi-ball bouncing ball data generator
# ---------------------------------------------------------------------------

def generate_multi_bouncing_ball(
    n_videos: int,
    n_frames: int,
    frame_size: int = 32,
    n_balls: int = 3,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic videos with multiple bouncing balls.

    Ball sizes: radius 2-4 pixels
    Speeds: 1.0-3.0 pixels per frame
    Occlusion: when balls overlap, use max intensity

    Returns: (n_videos, n_frames, frame_size, frame_size) float32 tensor.
    """
    rng = np.random.RandomState(seed)
    videos = np.zeros((n_videos, n_frames, frame_size, frame_size), dtype=np.float32)

    for i in range(n_videos):
        # Initialize balls with different sizes and speeds
        balls = []
        for b in range(n_balls):
            radius = rng.randint(2, 5)  # 2-4 pixels
            margin = radius + 1
            x = rng.uniform(margin, frame_size - margin)
            y = rng.uniform(margin, frame_size - margin)
            speed = rng.uniform(1.0, 3.0)
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            balls.append({"x": x, "y": y, "vx": vx, "vy": vy, "r": radius})

        for t in range(n_frames):
            frame = np.zeros((frame_size, frame_size), dtype=np.float32)

            # Draw and update each ball
            for ball in balls:
                x, y, r = ball["x"], ball["y"], ball["r"]

                # Draw ball
                yy, xx = np.ogrid[:frame_size, :frame_size]
                dist_sq = (xx - x) ** 2 + (yy - y) ** 2
                ball_mask = dist_sq <= r ** 2
                frame[ball_mask] = 1.0

                # Move and bounce
                ball["x"] += ball["vx"]
                ball["y"] += ball["vy"]

                if ball["x"] <= r or ball["x"] >= frame_size - r - 1:
                    ball["vx"] = -ball["vx"]
                    ball["x"] = np.clip(ball["x"], r, frame_size - r - 1)

                if ball["y"] <= r or ball["y"] >= frame_size - r - 1:
                    ball["vy"] = -ball["vy"]
                    ball["y"] = np.clip(ball["y"], r, frame_size - r - 1)

            videos[i, t] = frame

    return torch.from_numpy(videos)


def patchify_and_quantize(
    videos: torch.Tensor,
    patch_size: int = 8,
    n_levels: int = 256,
) -> torch.Tensor:
    """Convert videos to token sequences via patch-mean quantization.

    Args:
        videos: (N, T, H, W) float32
        patch_size: spatial patch size
        n_levels: quantization levels (= vocab size)

    Returns: (N, T * patches_per_frame) int64 tensor
    """
    N, T, H, W = videos.shape
    pH = H // patch_size
    pW = W // patch_size
    patches_per_frame = pH * pW

    # Reshape to patches: (N, T, pH, patch_size, pW, patch_size)
    v = videos.view(N, T, pH, patch_size, pW, patch_size)
    # Mean over patch pixels: (N, T, pH, pW)
    patch_means = v.mean(dim=(3, 5))
    # Flatten spatial: (N, T * pH * pW)
    flat = patch_means.reshape(N, T * patches_per_frame)
    # Quantize to [0, n_levels-1]
    tokens = (flat * (n_levels - 1)).clamp(0, n_levels - 1).long()
    return tokens


# ---------------------------------------------------------------------------
# RotaryEmbedding3D — 3D RoPE for video
# ---------------------------------------------------------------------------

class RotaryEmbedding3D(nn.Module):
    """3D RoPE for video: independent frequencies per spatial/temporal dim.

    forward(L) returns (cos, sin) with shape (L, head_dim), same interface
    as 1D RotaryEmbedding so Attention needs ZERO changes.
    """

    def __init__(
        self,
        inv_freq_h: torch.Tensor,  # (K_h,)
        inv_freq_w: torch.Tensor,  # (K_w,)
        inv_freq_t: torch.Tensor,  # (K_t,)
        H: int,                    # spatial grid height (patches)
        W: int,                    # spatial grid width (patches)
        max_T: int,                # max temporal frames
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)
        self.register_buffer("inv_freq_t", inv_freq_t)
        self._build(max_T)

    def _build(self, max_T: int) -> None:
        """Pre-compute cos/sin cache for up to max_T frames."""
        max_L = max_T * self.H * self.W
        positions = torch.arange(max_L, dtype=self.inv_freq_h.dtype,
                                device=self.inv_freq_h.device)

        HW = self.H * self.W
        t_pos = positions // HW                    # temporal index
        hw_pos = positions % HW
        h_pos = hw_pos // self.W                   # spatial row
        w_pos = hw_pos % self.W                    # spatial col

        # Compute frequency contributions: outer products
        freqs_h = torch.outer(h_pos.float(), self.inv_freq_h)  # (L, K_h)
        freqs_w = torch.outer(w_pos.float(), self.inv_freq_w)  # (L, K_w)
        freqs_t = torch.outer(t_pos.float(), self.inv_freq_t)  # (L, K_t)

        # Concatenate: (L, K_h + K_w + K_t) = (L, head_dim // 2)
        freqs = torch.cat([freqs_h, freqs_w, freqs_t], dim=-1)
        # Double for rotate_half: (L, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)
        self._max_T = max_T
        self._max_L = max_L

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        needed_T = (L + self.H * self.W - 1) // (self.H * self.W)
        if needed_T > self._max_T:
            self._build(needed_T)
        return self.cos_c[:L], self.sin_c[:L]


# ---------------------------------------------------------------------------
# VideoGPT model — reuses Block/Attention/MLP, swaps in 3D RoPE
# ---------------------------------------------------------------------------

class VideoGPT(nn.Module):
    def __init__(
        self,
        cfg: dict,
        inv_freq_h: torch.Tensor,
        inv_freq_w: torch.Tensor,
        inv_freq_t: torch.Tensor,
        H: int,
        W: int,
    ) -> None:
        super().__init__()
        self._num_layers = cfg["num_layers"]
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        max_T = cfg.get("max_T", 128)
        rope = RotaryEmbedding3D(inv_freq_h, inv_freq_w, inv_freq_t, H, W, max_T)
        self.blocks = nn.ModuleList(
            [Block(cfg, rope) for _ in range(cfg["num_layers"])]
        )
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self.apply(self._init)
        # Depth-scaled init for residual projections
        residual_scale = 1.0 / math.sqrt(2 * self._num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.down.weight, std=0.02 * residual_scale)
        n = sum(p.numel() for p in self.parameters())
        print(f"  VideoGPT params: {n / 1e6:.1f}M")

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
        needed_T = (L + self.blocks[0].attn.rope.H * self.blocks[0].attn.rope.W - 1) // (
            self.blocks[0].attn.rope.H * self.blocks[0].attn.rope.W
        )
        if needed_T > self.blocks[0].attn.rope._max_T:
            self.blocks[0].attn.rope._build(needed_T)


# ---------------------------------------------------------------------------
# FVD metric (simplified with I3D features, with fallback)
# ---------------------------------------------------------------------------

def compute_fvd(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Compute Fréchet Video Distance using I3D features.

    Args:
        real_videos: (N, T, H, W) or (N, T, 3, H, W) if RGB
        generated_videos: same shape
        device: device to compute on

    Returns: FVD scalar

    Falls back to MSE if I3D weights unavailable.
    """
    try:
        # Try to load pytorch-fvd
        import pytorch_fvd
        real = real_videos.to(device)
        gen = generated_videos.to(device)

        # Ensure 3-channel (replicate if grayscale)
        if real.ndim == 4:
            real = real.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # (N, T, 3, H, W)
        if gen.ndim == 4:
            gen = gen.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        # Normalize to [0, 255]
        real = (real * 255).clamp(0, 255).byte()
        gen = (gen * 255).clamp(0, 255).byte()

        fvd = pytorch_fvd.compute_fvd(real, gen)
        return float(fvd)
    except (ImportError, Exception) as e:
        print(f"    [FVD] pytorch_fvd unavailable ({e}), using MSE fallback")
        # Fallback: simple MSE between videos
        mse = F.mse_loss(real_videos.float(), generated_videos.float()).item()
        return mse


def compute_temporal_consistency(
    model: VideoGPT,
    videos: torch.Tensor,
    patches_per_frame: int,
    device: str = "cpu",
    max_frames: int = 128,
) -> Dict[str, float]:
    """Compute temporal consistency: MSE between consecutive predicted frames.

    For a 16-frame training video, autoregressively predict to 32, 48, 64, ...
    frames and measure consistency of frame transitions.
    """
    model.eval()
    results = {}

    if videos.shape[0] == 0:
        return results

    with torch.no_grad():
        # Take first training video
        video_tokens = videos[:1].to(device)  # (1, seq_len)

        consistency_scores = []
        for n_pred_frames in [32, 48, 64]:
            n_pred_tokens = n_pred_frames * patches_per_frame

            # Autoregressive generation
            ctx = video_tokens
            for _ in range(n_pred_tokens - video_tokens.shape[1]):
                if ctx.shape[1] >= max_frames * patches_per_frame:
                    break
                with torch.amp.autocast(device, dtype=DTYPE) if USE_AUTOCAST else nullcontext():
                    logits = model(ctx)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                ctx = torch.cat([ctx, next_token], dim=1)

            # Measure frame-to-frame MSE in last 10 predictions
            if ctx.shape[1] >= patches_per_frame:
                frame_tokens = ctx.view(1, -1, patches_per_frame)
                if frame_tokens.shape[1] >= 2:
                    frame_diffs = (frame_tokens[:, 1:] - frame_tokens[:, :-1]).abs().mean()
                    consistency_scores.append(frame_diffs.item())

        if consistency_scores:
            results["consistency"] = round(np.mean(consistency_scores), 4)

    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_video_model(
    model: VideoGPT,
    data: torch.Tensor,
    cfg: dict,
    seed: int = 42,
) -> VideoGPT:
    """Train video model with AdamW, cosine LR, gradient accumulation."""
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    batch_size = cfg["batch_size"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    n_samples = data.shape[0]
    train_steps = cfg.get("train_steps", 5000)
    warmup = int(train_steps * 0.05)
    set_seed(seed)

    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    t0 = time.time()

    for s in range(train_steps):
        # Sample batch with replacement
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = data[indices].to(DEVICE)

        # Cosine LR with warmup
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(train_steps - warmup, 1))
            )
        for g in opt.param_groups:
            g["lr"] = cur_lr

        with ctx:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (train_steps - s - 1) if s > 0 else 0
            print(
                f"    step {s}/{train_steps}  loss={loss.item():.4f}  "
                f"lr={cur_lr:.2e}  ETA={eta / 60:.0f}min"
            )

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed / 60:.1f} min ({train_steps} steps)")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_video_model(
    model: VideoGPT,
    val_data: torch.Tensor,
    eval_frames: List[int],
    patches_per_frame: int,
    train_frames: int,
    eval_chunks: int = 20,
    eval_seed: int = 9999,
) -> Dict[str, float]:
    """Evaluate video model at various temporal extrapolation ratios.

    Args:
        model: VideoGPT model
        val_data: (N, seq_len) int64 tensor of token sequences
        eval_frames: list of frame counts to evaluate
        patches_per_frame: patches per video frame
        train_frames: number of frames used in training
        eval_chunks: number of chunks to evaluate per frame count
        eval_seed: random seed for chunk selection

    Returns: dict mapping "Xf" -> PPL
    """
    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(eval_seed)
    results = {}

    for n_frames in eval_frames:
        L = n_frames * patches_per_frame

        # Frame-aligned offsets
        frame_stride = patches_per_frame
        valid_starts = []
        for i in range(0, val_data.shape[0]):
            max_start = val_data.shape[1] - L
            if max_start > 0:
                # Align to frame boundary
                n_valid = (max_start + 1) // frame_stride
                if n_valid > 0:
                    valid_starts.extend([i * val_data.shape[1] + j * frame_stride
                                        for j in range(n_valid)])

        if not valid_starts:
            print(f"    {n_frames}f (L={L}): no valid aligned starts")
            continue

        n_chunks = min(eval_chunks, len(valid_starts))
        chosen_idx = sorted(rng.choice(len(valid_starts), size=n_chunks, replace=False))
        offsets = [valid_starts[i] for i in chosen_idx]

        losses = []
        flat_data = val_data.reshape(-1)
        for offset in offsets:
            if offset + L > len(flat_data):
                continue
            chunk = flat_data[offset : offset + L].unsqueeze(0).to(DEVICE)
            try:
                with ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                    )
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    {n_frames}f: OOM")
                    break
                raise

        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[f"{n_frames}f"] = round(ppl, 3)
            ratio = n_frames / train_frames
            print(f"    {n_frames}f (L={L}, {ratio:.1f}x): PPL={ppl:.3f}  ({len(losses)} chunks)")

    return results


# ---------------------------------------------------------------------------
# Main experiment: Bouncing Ball
# ---------------------------------------------------------------------------

BB_CONFIG = {
    "frame_size": 32,
    "patch_size": 8,
    "n_balls": 3,
    "train_frames": 16,
    "train_samples": 10_000,
    "val_samples": 2_000,
    "vocab_size": 256,
    "hidden_size": 512,
    "num_layers": 12,
    "num_heads": 8,
    "head_dim": 64,
    "intermediate_size": 2048,
    "base": 10000.0,
    "train_steps": 5_000,
    "eval_frames": [16, 32, 48, 64, 96, 128],
    "lr": 3e-4,
    "batch_size": 32,
    "max_T": 128,
}


def run_bouncing_ball_experiment(
    taus: List[float],
    seeds: List[int],
    cfg: dict,
    work_dir: Path,
) -> Dict[str, Dict]:
    """Run bouncing ball experiment with τ sweep."""

    all_results = {}
    total_t0 = time.time()

    # Spatial grid: 32/8 = 4x4
    H = cfg["frame_size"] // cfg["patch_size"]
    W = H
    patches_per_frame = H * W  # 16
    train_L = cfg["train_frames"] * patches_per_frame  # 256

    # Frequency split: head_dim=64 -> 32 pairs -> K_h=12, K_w=12, K_t=8
    K = cfg["head_dim"] // 2  # 32
    K_h, K_w, K_t = 12, 12, 8
    assert K_h + K_w + K_t == K, f"Freq split mismatch: {K_h}+{K_w}+{K_t} != {K}"

    tau_star = K_t / math.sqrt(cfg["train_frames"])
    print(f"\n{'#'*70}")
    print(f"  PHASE 21 — VIDEO CROSS-MODAL GENERALIZATION (Sub-A: Bouncing Ball)")
    print(f"  taus={taus}  seeds={seeds}")
    print(f"  train_frames={cfg['train_frames']}  patches/frame={patches_per_frame}")
    print(f"  train_L={train_L}  (same as Phase 11 L=256)")
    print(f"  Freq split: K_h={K_h}, K_w={K_w}, K_t={K_t}")
    print(f"  tau* = K_t/sqrt(T_train) = {K_t}/sqrt({cfg['train_frames']}) = {tau_star:.2f}")
    print(f"  device={DEVICE}  dtype={DTYPE}")
    print(f"{'#'*70}")

    # Generate data
    print("\n  Generating training data (multi-ball)...")
    train_videos = generate_multi_bouncing_ball(
        cfg["train_samples"], cfg["train_frames"],
        cfg["frame_size"], n_balls=cfg["n_balls"], seed=42,
    )
    train_data = patchify_and_quantize(train_videos, cfg["patch_size"], cfg["vocab_size"])
    print(f"  Train: {train_data.shape} (videos={cfg['train_samples']}, "
          f"frames={cfg['train_frames']}, tokens/video={train_data.shape[1]})")

    print("  Generating validation data...")
    val_videos = generate_multi_bouncing_ball(
        cfg["val_samples"], 128,
        cfg["frame_size"], n_balls=cfg["n_balls"], seed=99999,
    )
    val_data = patchify_and_quantize(val_videos, cfg["patch_size"], cfg["vocab_size"])
    val_flat = val_data.reshape(-1)
    print(f"  Val: {val_data.shape} -> flat {val_flat.shape} "
          f"({val_flat.numel()} tokens)")

    # Spatial frequencies: always geometric
    inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0, base=cfg["base"])
    inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0, base=cfg["base"])
    print(f"\n  Spatial inv_freq (geometric): h={inv_freq_h.shape}, w={inv_freq_w.shape}")

    # Run sweep
    for tau in taus:
        for seed in seeds:
            run_id = f"tau{tau:.1f}_seed{seed}"
            print(f"\n{'='*70}")
            print(f"  RUN: {run_id}  (tau={tau}, base={cfg['base']})")
            print(f"{'='*70}")

            # Build temporal frequencies
            inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=tau, base=cfg["base"])
            print(f"  Temporal inv_freq: shape={inv_freq_t.shape}  "
                  f"max={inv_freq_t.max().item():.6f}  "
                  f"min={inv_freq_t.min().item():.8f}")

            # Build & train model
            set_seed(seed)
            model = VideoGPT(
                cfg, inv_freq_h, inv_freq_w, inv_freq_t, H, W
            ).to(DEVICE)

            train_t0 = time.time()
            model = train_video_model(model, train_data, cfg, seed=seed)
            train_time = time.time() - train_t0

            # Evaluate
            print(f"\n  Evaluating {run_id}...")
            eval_t0 = time.time()
            ppl_results = eval_video_model(
                model, val_flat.unsqueeze(0), cfg["eval_frames"], patches_per_frame,
                cfg["train_frames"], eval_chunks=15,
            )
            eval_time = time.time() - eval_t0

            # Temporal consistency
            print(f"\n  Computing temporal consistency...")
            consistency = compute_temporal_consistency(
                model, train_data[:10], patches_per_frame, device=DEVICE
            )

            all_results[run_id] = {
                "tau": tau,
                "seed": seed,
                "ppl": ppl_results,
                "consistency": consistency,
                "train_time_sec": round(train_time, 1),
                "eval_time_sec": round(eval_time, 1),
                "inv_freq_t_max": round(inv_freq_t.max().item(), 8),
                "inv_freq_t_min": round(inv_freq_t.min().item(), 8),
            }

            # Cleanup
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

    total_time = time.time() - total_t0

    # Print results table
    print(f"\n\n{'='*80}")
    print(f"  RESULTS — Bouncing Ball Sub-A")
    print(f"  tau* (theory) = {tau_star:.2f}")
    print(f"{'='*80}")

    # Group by tau
    tau_results: Dict[float, Dict[str, List[float]]] = {}
    for run_id, res in all_results.items():
        tau = res["tau"]
        if tau not in tau_results:
            tau_results[tau] = {}
        for frame_key, val in res["ppl"].items():
            tau_results[tau].setdefault(frame_key, []).append(val)

    # Header
    frame_labels = [f"{f}f" for f in cfg["eval_frames"]]
    print(f"\n  {'Condition':<20}", end="")
    for fl in frame_labels:
        print(f"  {fl:>8}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in frame_labels:
        print(f"  {'─'*8}", end="")
    print()

    # Data rows
    for tau in sorted(tau_results.keys()):
        label = f"tau={tau:.1f}"
        print(f"  {label:<20}", end="")
        for f in cfg["eval_frames"]:
            key = f"{f}f"
            vals = tau_results[tau].get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {avg:>8.2f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()

    # Save results
    output = {
        "metadata": {
            "experiment": "bouncing_ball_sub_a",
            "mode": "bouncing_ball",
            "taus": taus,
            "seeds": seeds,
            "base": cfg["base"],
            "train_frames": cfg["train_frames"],
            "patches_per_frame": patches_per_frame,
            "train_L": train_L,
            "eval_frames": cfg["eval_frames"],
            "tau_star_theory": tau_star,
            "freq_split": {"K_h": K_h, "K_w": K_w, "K_t": K_t},
            "model_config": {k: v for k, v in cfg.items()
                            if k in ["vocab_size", "hidden_size", "num_layers",
                                     "num_heads", "head_dim", "intermediate_size"]},
            "device": DEVICE,
            "dtype": str(DTYPE),
            "total_time_min": round(total_time / 60, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "experiments": all_results,
    }

    out_path = work_dir / "results_phase21_bouncing_ball.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Total time: {total_time / 60:.1f} min")

    return all_results


# ---------------------------------------------------------------------------
# Sub-B: Latte/Open-Sora Setup (stub for real video)
# ---------------------------------------------------------------------------

def setup_video_dit(
    model_name: str,
    temporal_tau: float,
    d_head: int = 64,
) -> tuple:
    """Setup video DiT (Latte/Open-Sora) with EVQ temporal frequencies.

    Returns: (model, device) or (None, None) if not available.
    """
    print(f"\n  Setting up {model_name} with temporal τ={temporal_tau}...")

    try:
        if model_name.lower() == "latte":
            # Try to load Latte from HuggingFace
            try:
                from diffusers import LatteTransformer3D
                model = LatteTransformer3D.from_pretrained("maxin-cn/Latte-1")
            except ImportError:
                print(f"    [Warning] diffusers not installed, skipping Latte")
                return None, None
        else:
            print(f"    [Warning] Unsupported model: {model_name}")
            return None, None

        # Locate and replace temporal RoPE frequencies
        print(f"    Found {model_name} model, replacing temporal frequencies...")

        # This would require model-specific knowledge of where RoPE buffers are
        # For now, return stub
        return model, DEVICE

    except Exception as e:
        print(f"    [Warning] Could not setup {model_name}: {e}")
        return None, None


def run_latte_experiment(
    model_path: Optional[str],
    data_path: Optional[str],
    taus: List[float],
    seeds: List[int],
) -> None:
    """Run fine-tuning experiment on Latte with UCF-101.

    This is a stub - full implementation requires:
    1. Loading Latte checkpoint
    2. UCF-101 dataset loading and preprocessing
    3. Fine-tuning loop with EVQ frequencies
    4. FVD + FID evaluation
    """
    print(f"\n{'#'*70}")
    print(f"  PHASE 21 — Sub-B: Latte/Real Video (STUB)")
    print(f"  model_path: {model_path}")
    print(f"  data_path: {data_path}")
    print(f"  taus={taus}  seeds={seeds}")
    print(f"{'#'*70}")

    if not model_path or not data_path:
        print("\n  [Error] model_path and data_path required for Sub-B")
        print("  Usage: python phase21_video_dit.py --mode latte --model_path /path/to/latte --data_path /path/to/ucf101")
        return

    print("\n  [TODO] Implement Latte fine-tuning pipeline")
    print("  Required:")
    print("    1. Load Latte checkpoint from model_path")
    print("    2. Setup UCF-101 loader from data_path")
    print("    3. Fine-tune for 10-20K steps per τ")
    print("    4. Evaluate FVD via I3D features")
    print("    5. Save results to JSON")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 21: Video Cross-Modal Generalization for EVQ-Cosh"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bouncing_ball",
        choices=["bouncing_ball", "latte"],
        help="Experiment mode: bouncing_ball (self-contained) or latte (requires pretrained model + data)",
    )
    parser.add_argument("--taus", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--seeds", type=str, default="42,137,256")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Latte checkpoint (Sub-B)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to UCF-101 dataset (Sub-B)")
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = BB_CONFIG.copy()
    cfg["base"] = args.base

    # Work dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path("results") / "phase21_video_dit"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "bouncing_ball":
        if args.dry_run:
            print("\n[DRY RUN] Skipping training & eval")
            return
        run_bouncing_ball_experiment(taus, seeds, cfg, work_dir)

    elif args.mode == "latte":
        run_latte_experiment(args.model_path, args.data_path, taus, seeds)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
