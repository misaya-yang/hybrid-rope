#!/usr/bin/env python3
"""Video temporal extrapolation experiment — EVQ-cosh on 3D RoPE.

Tests whether EVQ-cosh applied to temporal channels improves temporal
extrapolation in a video transformer, validating EVQ as a unified theory
across text and video (analogous to Phase 11 L=256 language model).

Usage:
    conda activate aidemo
    python scripts/video_temporal/run_video_temporal.py
    python scripts/video_temporal/run_video_temporal.py --taus 0.0,2.0 --seeds 42

Hardware: M4 Max 36GB (MPS), ~90 min total.
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
SWEEP_DIR = SCRIPT_DIR.parent / "m4_evq_sweep"
sys.path.insert(0, str(SWEEP_DIR))

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

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32


# ---------------------------------------------------------------------------
# Bouncing ball data generator
# ---------------------------------------------------------------------------

def generate_bouncing_ball(
    n_samples: int,
    n_frames: int,
    frame_size: int = 32,
    ball_radius: int = 3,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic bouncing ball videos.

    Returns: (n_samples, n_frames, frame_size, frame_size) float32 tensor.
    """
    rng = np.random.RandomState(seed)
    videos = np.zeros((n_samples, n_frames, frame_size, frame_size), dtype=np.float32)

    for i in range(n_samples):
        # Random initial position (inside bounds)
        margin = ball_radius + 1
        x = rng.uniform(margin, frame_size - margin)
        y = rng.uniform(margin, frame_size - margin)

        # Random velocity (pixels per frame)
        speed = rng.uniform(1.0, 3.0)
        angle = rng.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        for t in range(n_frames):
            # Draw ball
            yy, xx = np.ogrid[:frame_size, :frame_size]
            dist_sq = (xx - x) ** 2 + (yy - y) ** 2
            ball_mask = dist_sq <= ball_radius ** 2
            videos[i, t][ball_mask] = 1.0

            # Move and bounce
            x += vx
            y += vy
            if x <= ball_radius or x >= frame_size - ball_radius - 1:
                vx = -vx
                x = np.clip(x, ball_radius, frame_size - ball_radius - 1)
            if y <= ball_radius or y >= frame_size - ball_radius - 1:
                vy = -vy
                y = np.clip(y, ball_radius, frame_size - ball_radius - 1)

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
# RotaryEmbedding3D — 3D RoPE with same forward() interface as 1D
# ---------------------------------------------------------------------------

class RotaryEmbedding3D(nn.Module):
    """3D RoPE for video: independent frequencies per spatial/temporal dim.

    forward(L) returns (cos, sin) with shape (L, head_dim), same interface
    as the 1D RotaryEmbedding so Attention needs ZERO changes.
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
# Training loop (adapted from run_evq_sweep.train_model)
# ---------------------------------------------------------------------------

def train_video_model(
    model: VideoGPT,
    data: torch.Tensor,
    cfg: dict,
    seed: int = 42,
) -> VideoGPT:
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    batch_size = cfg["batch_size"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    n_samples = data.shape[0]
    steps = (n_samples * cfg.get("epochs", 40)) // batch_size
    warmup = int(steps * 0.02)
    set_seed(seed)

    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    t0 = time.time()

    for s in range(steps):
        # Sample batch with replacement across epochs
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = data[indices].to(DEVICE)

        # Cosine LR with warmup
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
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
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(
                f"    step {s}/{steps}  loss={loss.item():.4f}  "
                f"lr={cur_lr:.2e}  ETA={eta / 60:.0f}min"
            )

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed / 60:.1f} min ({steps} steps)")
    return model


# ---------------------------------------------------------------------------
# Evaluation — frame-aligned chunks + optional temporal YaRN
# ---------------------------------------------------------------------------

def build_temporal_yarn_inv_freq(
    inv_freq_t: torch.Tensor, scale: float
) -> torch.Tensor:
    """Apply YaRN-style smoothstep interpolation to temporal frequencies only."""
    K = len(inv_freq_t)
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (inv_freq_t.cpu().double() / yarn_scale).float()


@torch.no_grad()
def eval_video_model(
    model: VideoGPT,
    val_data: torch.Tensor,
    eval_frames: List[int],
    patches_per_frame: int,
    train_frames: int,
    eval_chunks: int = 20,
    eval_seed: int = 9999,
    yarn: bool = False,
) -> Dict[str, float]:
    """Evaluate video model at various temporal extrapolation ratios.

    val_data: flat 1D tensor of tokens (all val videos concatenated)
    eval_frames: list of frame counts to evaluate
    """
    model.eval()
    rope = model.blocks[0].attn.rope

    # Save original inv_freq_t for restore
    orig_inv_freq_t = rope.inv_freq_t.clone()

    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(eval_seed)
    results = {}

    for n_frames in eval_frames:
        L = n_frames * patches_per_frame
        scale = n_frames / train_frames

        # Apply temporal YaRN if requested
        if yarn and scale > 1.0:
            yarn_inv_t = build_temporal_yarn_inv_freq(orig_inv_freq_t, scale)
            rope.inv_freq_t.copy_(yarn_inv_t.to(rope.inv_freq_t.device))

        # Rebuild cache for this length
        rope._build(n_frames + 4)

        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            print(f"    {n_frames}f (L={L}): val_data too short, skipping")
            continue

        # Frame-aligned offsets: must start on frame boundary
        frame_stride = patches_per_frame
        valid_starts = list(range(0, max_start + 1, frame_stride))
        if not valid_starts:
            print(f"    {n_frames}f (L={L}): no valid aligned starts")
            continue
        n_chunks = min(eval_chunks, len(valid_starts))
        chosen = sorted(rng.choice(len(valid_starts), size=n_chunks, replace=False))
        offsets = [valid_starts[c] for c in chosen]

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
                    print(f"    {n_frames}f: OOM, stopping")
                    del chunk
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE == "mps":
                        torch.mps.empty_cache()
                    break
                raise

        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            label = f"{'yarn_' if yarn else ''}{n_frames}f"
            results[label] = round(ppl, 3)
            ratio = n_frames / train_frames
            print(f"    {label} (L={L}, {ratio:.1f}x): PPL={ppl:.3f}  ({len(losses)} chunks)")

        # Restore original temporal freqs
        if yarn and scale > 1.0:
            rope.inv_freq_t.copy_(orig_inv_freq_t)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VIDEO_CFG = {
    "vocab_size": 256,
    "hidden_size": 384,
    "num_layers": 6,
    "num_heads": 6,
    "head_dim": 64,
    "intermediate_size": 1536,
    "max_T": 128,
    # Training
    "lr": 6e-4,
    "batch_size": 64,
    "epochs": 40,
    # Data
    "frame_size": 32,
    "patch_size": 8,
    "train_frames": 16,
    "train_samples": 10000,
    "val_samples": 500,
    "val_frames": 128,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video temporal extrapolation — EVQ on 3D RoPE"
    )
    parser.add_argument("--taus", type=str, default="0.0,2.0,4.0")
    parser.add_argument("--seeds", type=str, default="42,137")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--train_frames", type=int, default=16)
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = VIDEO_CFG.copy()
    cfg["train_frames"] = args.train_frames

    # Spatial grid: 32/8 = 4x4
    H = cfg["frame_size"] // cfg["patch_size"]
    W = H
    patches_per_frame = H * W  # 16
    train_L = cfg["train_frames"] * patches_per_frame  # 256

    # Frequency split: head_dim=64 -> 32 pairs -> K_h=12, K_w=12, K_t=8
    K = cfg["head_dim"] // 2  # 32
    K_h, K_w, K_t = 12, 12, 8
    assert K_h + K_w + K_t == K, f"Freq split mismatch: {K_h}+{K_w}+{K_t} != {K}"

    # Theory prediction: tau* = K_t / sqrt(T_train)
    tau_star = K_t / math.sqrt(cfg["train_frames"])
    print(f"\n{'#'*60}")
    print(f"  VIDEO TEMPORAL EXTRAPOLATION — EVQ on 3D RoPE")
    print(f"  taus={taus}  seeds={seeds}  base={args.base}")
    print(f"  train_frames={cfg['train_frames']}  patches/frame={patches_per_frame}")
    print(f"  train_L={train_L}  (same as Phase 11 L=256)")
    print(f"  Freq split: K_h={K_h}, K_w={K_w}, K_t={K_t}")
    print(f"  tau* = K_t/sqrt(T_train) = {K_t}/sqrt({cfg['train_frames']}) = {tau_star:.2f}")
    print(f"  device={DEVICE}  dtype={DTYPE}")
    print(f"{'#'*60}")

    # Work dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path("results") / "video_temporal"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Generate data
    # ---------------------------------------------------------------------------
    print("\n  Generating training data...")
    train_videos = generate_bouncing_ball(
        cfg["train_samples"], cfg["train_frames"],
        cfg["frame_size"], seed=42,
    )
    train_data = patchify_and_quantize(train_videos, cfg["patch_size"], cfg["vocab_size"])
    print(f"  Train: {train_data.shape} (videos={cfg['train_samples']}, "
          f"frames={cfg['train_frames']}, tokens/video={train_data.shape[1]})")

    print("  Generating validation data...")
    val_videos = generate_bouncing_ball(
        cfg["val_samples"], cfg["val_frames"],
        cfg["frame_size"], seed=99999,
    )
    val_data = patchify_and_quantize(val_videos, cfg["patch_size"], cfg["vocab_size"])
    # Flatten val data to 1D for eval chunking
    val_flat = val_data.reshape(-1)
    print(f"  Val: {val_data.shape} -> flat {val_flat.shape} "
          f"({val_flat.numel()} tokens)")

    # Eval frame counts: 1x to 8x temporal extrapolation
    eval_frames = [16, 32, 48, 64, 96, 128]

    # Spatial frequencies: always geometric (controlled variable)
    inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0, base=args.base)
    inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0, base=args.base)
    print(f"\n  Spatial inv_freq (geometric): h={inv_freq_h.shape}, w={inv_freq_w.shape}")

    # ---------------------------------------------------------------------------
    # Run sweep
    # ---------------------------------------------------------------------------
    all_results = {}
    total_t0 = time.time()

    for tau in taus:
        for seed in seeds:
            run_id = f"tau{tau:.1f}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"  RUN: {run_id}  (tau={tau}, base={args.base})")
            print(f"{'='*60}")

            # Build temporal frequencies
            inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=tau, base=args.base)
            print(f"  Temporal inv_freq: shape={inv_freq_t.shape}  "
                  f"max={inv_freq_t.max().item():.6f}  "
                  f"min={inv_freq_t.min().item():.8f}")

            if args.dry_run:
                print("  [DRY RUN] skipping training & eval")
                continue

            # Build & train model
            set_seed(seed)
            model = VideoGPT(
                cfg, inv_freq_h, inv_freq_w, inv_freq_t, H, W
            ).to(DEVICE)

            train_t0 = time.time()
            model = train_video_model(model, train_data, cfg, seed=seed)
            train_time = time.time() - train_t0

            # Eval: raw (no YaRN)
            print(f"\n  Evaluating {run_id} (raw)...")
            eval_t0 = time.time()
            ppl_raw = eval_video_model(
                model, val_flat, eval_frames, patches_per_frame,
                cfg["train_frames"], eval_chunks=20, yarn=False,
            )
            eval_time_raw = time.time() - eval_t0

            # Eval: with temporal YaRN
            print(f"\n  Evaluating {run_id} (temporal YaRN)...")
            eval_t1 = time.time()
            ppl_yarn = eval_video_model(
                model, val_flat, eval_frames, patches_per_frame,
                cfg["train_frames"], eval_chunks=20, yarn=True,
            )
            eval_time_yarn = time.time() - eval_t1

            total_eval_time = eval_time_raw + eval_time_yarn

            all_results[run_id] = {
                "tau": tau,
                "seed": seed,
                "ppl_raw": ppl_raw,
                "ppl_yarn": ppl_yarn,
                "train_time_sec": round(train_time, 1),
                "eval_time_sec": round(total_eval_time, 1),
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

    if args.dry_run:
        print("\n  [DRY RUN] No results to report.")
        return

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print(f"  RESULTS — Video Temporal Extrapolation")
    print(f"  tau* (theory) = {tau_star:.2f}")
    print(f"{'='*80}")

    # Group by tau, average over seeds
    tau_results: Dict[float, Dict[str, List[float]]] = {}
    for run_id, res in all_results.items():
        tau = res["tau"]
        if tau not in tau_results:
            tau_results[tau] = {}
        for key in ["ppl_raw", "ppl_yarn"]:
            for frame_key, val in res[key].items():
                full_key = f"{key}_{frame_key}"
                tau_results[tau].setdefault(full_key, []).append(val)

    # Header
    frame_labels = [f"{f}f" for f in eval_frames]
    print(f"\n  {'Condition':<20}", end="")
    for fl in frame_labels:
        print(f"  {fl:>8}", end="")
    print()
    print(f"  {'─'*20}", end="")
    for _ in frame_labels:
        print(f"  {'─'*8}", end="")
    print()

    # Raw rows
    for tau in sorted(tau_results.keys()):
        label = f"tau={tau:.1f} (raw)"
        print(f"  {label:<20}", end="")
        for f in eval_frames:
            key = f"ppl_raw_{f}f"
            vals = tau_results[tau].get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {avg:>8.2f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()

    print()

    # YaRN rows
    for tau in sorted(tau_results.keys()):
        label = f"tau={tau:.1f} (YaRN)"
        print(f"  {label:<20}", end="")
        for f in eval_frames:
            key = f"ppl_yarn_yarn_{f}f"
            vals = tau_results[tau].get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {avg:>8.2f}", end="")
            else:
                print(f"  {'N/A':>8}", end="")
        print()

    # ---------------------------------------------------------------------------
    # Falsification check
    # ---------------------------------------------------------------------------
    print(f"\n\n  FALSIFICATION CHECK")
    print(f"  {'─'*50}")

    # Check 1: EVQ tau=2.0 vs Geo at 4x (64 frames)
    geo_raw_64 = tau_results.get(0.0, {}).get("ppl_raw_64f", [])
    evq_raw_64 = tau_results.get(2.0, {}).get("ppl_raw_64f", [])
    if geo_raw_64 and evq_raw_64:
        geo_avg = sum(geo_raw_64) / len(geo_raw_64)
        evq_avg = sum(evq_raw_64) / len(evq_raw_64)
        improvement = (1 - evq_avg / geo_avg) * 100
        status = "PASS" if improvement >= 15 else "FAIL"
        print(f"  1. EVQ tau=2.0 vs Geo at 4x: {improvement:+.1f}%  [{status}]"
              f"  (Geo={geo_avg:.2f}, EVQ={evq_avg:.2f})")

    # Check 2: tau=2.0 better than tau=4.0
    evq2_raw = tau_results.get(2.0, {})
    evq4_raw = tau_results.get(4.0, {})
    if evq2_raw and evq4_raw:
        wins_2 = 0
        total = 0
        for f in eval_frames[1:]:  # skip 1x
            k = f"ppl_raw_{f}f"
            v2 = evq2_raw.get(k, [])
            v4 = evq4_raw.get(k, [])
            if v2 and v4:
                total += 1
                if sum(v2) / len(v2) <= sum(v4) / len(v4):
                    wins_2 += 1
        status = "PASS" if wins_2 > total / 2 else "FAIL"
        print(f"  2. tau=2.0 <= tau=4.0: {wins_2}/{total} ratios  [{status}]")

    # Check 3: EVQ+YaRN synergy
    geo_yarn_64 = tau_results.get(0.0, {}).get("ppl_yarn_yarn_64f", [])
    evq_yarn_64 = tau_results.get(2.0, {}).get("ppl_yarn_yarn_64f", [])
    if geo_raw_64 and evq_raw_64 and geo_yarn_64 and evq_yarn_64:
        geo_raw_avg = sum(geo_raw_64) / len(geo_raw_64)
        geo_yarn_avg = sum(geo_yarn_64) / len(geo_yarn_64)
        evq_raw_avg = sum(evq_raw_64) / len(evq_raw_64)
        evq_yarn_avg = sum(evq_yarn_64) / len(evq_yarn_64)
        geo_yarn_gain = (1 - geo_yarn_avg / geo_raw_avg) * 100
        evq_yarn_gain = (1 - evq_yarn_avg / evq_raw_avg) * 100
        combined_gain = (1 - evq_yarn_avg / geo_raw_avg) * 100
        additive = geo_yarn_gain + (1 - evq_raw_avg / geo_raw_avg) * 100
        synergy = combined_gain - additive
        status = "PASS" if synergy > 0 else "FAIL"
        print(f"  3. EVQ+YaRN synergy at 4x: {synergy:+.1f}%  [{status}]"
              f"  (combined={combined_gain:.1f}%, additive={additive:.1f}%)")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    output = {
        "metadata": {
            "experiment": "video_temporal_extrapolation",
            "taus": taus,
            "seeds": seeds,
            "base": args.base,
            "train_frames": cfg["train_frames"],
            "patches_per_frame": patches_per_frame,
            "train_L": train_L,
            "eval_frames": eval_frames,
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

    out_path = work_dir / "results_video_temporal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Total time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
