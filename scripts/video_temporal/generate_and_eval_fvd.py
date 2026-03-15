#!/usr/bin/env python3
"""Video generation + FVD/MSE/SSIM evaluation for temporal allocation experiments.

This module adds the missing FVD pipeline to the video temporal sweep:
  1. Autoregressive video generation (teacher-forced start + AR continuation)
  2. Token -> pixel decoding
  3. FVD computation (I3D-based or pixel-space fallback)
  4. MSE / SSIM against ground-truth future frames

Usage as standalone:
    python generate_and_eval_fvd.py \
        --checkpoint results/.../geo_k16_seed42.pt \
        --data-dir data/video_temporal/generated/moving_mnist_medium \
        --n-generate 1024

Usage as library:
    from generate_and_eval_fvd import generate_videos, compute_fvd, compute_prediction_metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_video_temporal import DEVICE, DTYPE, USE_AUTOCAST, VideoGPT, apply_rope  # noqa: E402
from contextlib import nullcontext


# ---------------------------------------------------------------------------
# KV-cached forward pass helpers (avoids O(n²) per-token generation cost)
# ---------------------------------------------------------------------------

def _attention_forward_cached(attn, x, past_kv, start_pos):
    """Attention forward with KV cache. x can be (B, L, D) for prefill or (B, 1, D) for decode."""
    B, L, _ = x.shape
    qkv = attn.qkv(x).view(B, L, 3, attn.nh, attn.hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # RoPE for positions [start_pos, start_pos + L)
    cos_full, sin_full = attn.rope(start_pos + L)
    cos = cos_full[start_pos:start_pos + L][None, None]
    sin = sin_full[start_pos:start_pos + L][None, None]
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    # Concatenate with cached K, V
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.o(out.transpose(1, 2).reshape(B, L, -1)), (k, v)


def _model_forward_cached(model, tokens, past_kv_list, start_pos):
    """VideoGPT forward with per-layer KV cache. Returns (logits, new_kv_list)."""
    x = model.emb(tokens)
    new_kv_list = []
    for i, block in enumerate(model.blocks):
        past = past_kv_list[i] if past_kv_list else None
        h = block.ln1(x)
        attn_out, new_kv = _attention_forward_cached(block.attn, h, past, start_pos)
        x = x + attn_out
        x = x + block.mlp(block.ln2(x))
        new_kv_list.append(new_kv)
    return model.head(model.ln(x)), new_kv_list


# ---------------------------------------------------------------------------
# Token <-> Pixel conversion
# ---------------------------------------------------------------------------

def decode_tokens_to_frames(
    tokens: torch.Tensor,
    patch_size: int = 8,
    grid_h: int = 8,
    grid_w: int = 8,
) -> torch.Tensor:
    """Decode token sequence to pixel frames.

    Args:
        tokens: (N, T * grid_h * grid_w) int64 token IDs in [0, 255]
        patch_size: spatial size of each patch
        grid_h, grid_w: number of patches per spatial dim

    Returns:
        frames: (N, T, H, W) float32 in [0, 1]
        where H = grid_h * patch_size, W = grid_w * patch_size
    """
    N = tokens.shape[0]
    total_tokens = tokens.shape[1]
    patches_per_frame = grid_h * grid_w
    T = total_tokens // patches_per_frame

    # (N, T, grid_h, grid_w) float values
    values = tokens.float().reshape(N, T, grid_h, grid_w) / 255.0

    # Expand each patch: (N, T, grid_h, grid_w) -> (N, T, H, W)
    # Use repeat_interleave for clean upsampling
    frames = values.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return frames


def encode_frames_to_tokens(
    frames: torch.Tensor,
    patch_size: int = 8,
) -> torch.Tensor:
    """Encode pixel frames back to tokens (inverse of decode).

    Args:
        frames: (N, T, H, W) float32 in [0, 1]
        patch_size: spatial size of each patch

    Returns:
        tokens: (N, T * patches_per_frame) int64 in [0, 255]
    """
    N, T, H, W = frames.shape
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Average pool to patch means
    # Reshape: (N, T, grid_h, patch_size, grid_w, patch_size)
    v = frames.reshape(N, T, grid_h, patch_size, grid_w, patch_size)
    patch_means = v.mean(dim=(3, 5))  # (N, T, grid_h, grid_w)
    flat = patch_means.reshape(N, T * grid_h * grid_w)
    tokens = (flat * 255).clamp(0, 255).long()
    return tokens


# ---------------------------------------------------------------------------
# Autoregressive video generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_videos(
    model: VideoGPT,
    context_tokens: torch.Tensor,
    train_frames: int,
    target_frames: int,
    patches_per_frame: int,
    n_generate: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Generate videos via teacher-forced start + autoregressive continuation.

    Uses KV cache for O(n) per-token cost instead of O(n²).

    Args:
        model: trained VideoGPT
        context_tokens: (N_available, total_tokens_per_video) test/val token sequences
        train_frames: number of context frames to teacher-force
        target_frames: total frames to generate (including context)
        patches_per_frame: tokens per frame (e.g. 64 for 8x8 grid)
        n_generate: how many videos to generate
        temperature: sampling temperature
        top_k: top-k sampling
        batch_size: generation batch size (controls GPU memory)
        seed: random seed for generation

    Returns:
        generated: (n_generate, target_frames * patches_per_frame) int64 tokens
    """
    model.eval()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    context_len = train_frames * patches_per_frame
    target_len = target_frames * patches_per_frame
    tokens_to_generate = target_len - context_len

    if tokens_to_generate <= 0:
        # No extrapolation needed, just return context
        return context_tokens[:n_generate, :target_len].clone()

    # Select n_generate context sequences
    n_available = context_tokens.shape[0]
    if n_generate <= n_available:
        indices = torch.randperm(n_available, generator=rng)[:n_generate]
    else:
        indices = torch.randint(0, n_available, (n_generate,), generator=rng)

    # Extend RoPE cache for the full target length
    model.extend_rope(target_len + 1)

    all_generated = []
    n_batches = (n_generate + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, n_generate)
        batch_indices = indices[start:end]
        batch_context = context_tokens[batch_indices, :context_len].to(DEVICE)

        # Prefill: process full context, build KV cache
        with ctx:
            logits, kv_cache = _model_forward_cached(
                model, batch_context, None, start_pos=0
            )

        generated = []
        for tok_idx in range(tokens_to_generate):
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0 and top_k < next_logits.size(-1):
                topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.tensor(float('-inf'), device=next_logits.device, dtype=next_logits.dtype),
                    next_logits,
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token)

            # Decode: single-token forward with KV cache
            if tok_idx < tokens_to_generate - 1:
                with ctx:
                    logits, kv_cache = _model_forward_cached(
                        model, next_token, kv_cache,
                        start_pos=context_len + tok_idx,
                    )

            if (tok_idx + 1) % (patches_per_frame * 4) == 0:
                frames_done = (tok_idx + 1) // patches_per_frame
                total_gen_frames = tokens_to_generate // patches_per_frame
                print(f"    batch {b_idx+1}/{n_batches}: "
                      f"generated {frames_done}/{total_gen_frames} frames")

        current = torch.cat([batch_context] + generated, dim=1)
        all_generated.append(current.cpu())

        # Free KV cache and GPU memory
        del kv_cache, generated, current, batch_context, logits
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(all_generated, dim=0)


@torch.no_grad()
def generate_videos_chunked(
    model: VideoGPT,
    context_tokens: torch.Tensor,
    train_frames: int,
    target_frames: int,
    patches_per_frame: int,
    n_generate: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    batch_size: int = 16,
    seed: int = 42,
    max_context_window: int = 0,
) -> torch.Tensor:
    """Memory-efficient generation using sliding window.

    For very long sequences (128 frames = 8192 tokens), full attention
    over the entire sequence is expensive. This version uses a sliding
    context window for generation while keeping the full sequence.

    If max_context_window <= 0, uses the full context (same as generate_videos).
    """
    if max_context_window <= 0:
        return generate_videos(
            model, context_tokens, train_frames, target_frames,
            patches_per_frame, n_generate, temperature, top_k,
            batch_size, seed,
        )

    model.eval()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    context_len = train_frames * patches_per_frame
    target_len = target_frames * patches_per_frame
    tokens_to_generate = target_len - context_len

    n_available = context_tokens.shape[0]
    if n_generate <= n_available:
        indices = torch.randperm(n_available, generator=rng)[:n_generate]
    else:
        indices = torch.randint(0, n_available, (n_generate,), generator=rng)

    all_generated = []
    n_batches = (n_generate + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, n_generate)
        batch_indices = indices[start:end]
        # Keep full sequence on CPU, only move window to GPU
        full_seq = context_tokens[batch_indices, :context_len].clone()

        model.extend_rope(max_context_window + 1)

        for tok_idx in range(tokens_to_generate):
            # Use last max_context_window tokens as context
            if full_seq.shape[1] <= max_context_window:
                window = full_seq.to(DEVICE)
            else:
                window = full_seq[:, -max_context_window:].to(DEVICE)

            with ctx:
                logits = model(window)
                next_logits = logits[:, -1, :] / temperature

            if top_k > 0 and top_k < next_logits.size(-1):
                topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float('-inf')),
                    next_logits,
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).cpu()
            full_seq = torch.cat([full_seq, next_token], dim=1)

            del window
            if (tok_idx + 1) % (patches_per_frame * 8) == 0:
                frames_done = (tok_idx + 1) // patches_per_frame
                total_gen_frames = tokens_to_generate // patches_per_frame
                print(f"    batch {b_idx+1}/{n_batches}: "
                      f"generated {frames_done}/{total_gen_frames} frames")

        all_generated.append(full_seq)
        del full_seq
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(all_generated, dim=0)


# ---------------------------------------------------------------------------
# FVD computation
# ---------------------------------------------------------------------------

def compute_pixel_fvd(
    real_frames: torch.Tensor,
    gen_frames: torch.Tensor,
    feature_dim: int = 2048,
) -> float:
    """Compute Frechet Video Distance using pixel-space features.

    This is a fallback when I3D is not available. We flatten each video
    to a feature vector and compute the Frechet distance.

    For robustness, we downsample temporally and spatially, then PCA
    to feature_dim dimensions.

    Args:
        real_frames: (N, T, H, W) float32 [0,1]
        gen_frames:  (N, T, H, W) float32 [0,1]
        feature_dim: dimension after PCA

    Returns:
        FVD score (lower is better)
    """
    from scipy.linalg import sqrtm

    def extract_features(frames: torch.Tensor) -> np.ndarray:
        N, T, H, W = frames.shape
        # Subsample temporally to max 16 frames for manageable size
        if T > 16:
            indices = torch.linspace(0, T - 1, 16).long()
            frames = frames[:, indices]
            T = 16
        # Downsample spatially to 32x32
        if H > 32 or W > 32:
            frames = F.interpolate(
                frames.reshape(N * T, 1, H, W),
                size=(32, 32), mode='bilinear', align_corners=False,
            ).reshape(N, T, 32, 32)
        # Flatten: (N, T * 32 * 32)
        flat = frames.reshape(N, -1).numpy()
        # PCA via SVD if dimension is too high
        if flat.shape[1] > feature_dim:
            mean = flat.mean(axis=0, keepdims=True)
            flat_c = flat - mean
            U, S, Vt = np.linalg.svd(flat_c, full_matrices=False)
            flat = flat_c @ Vt[:feature_dim].T
        return flat

    real_feat = extract_features(real_frames)
    gen_feat = extract_features(gen_frames)

    # Frechet distance
    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    sigma_r = np.cov(real_feat, rowvar=False)
    sigma_g = np.cov(gen_feat, rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fvd


def try_load_i3d(i3d_path: str = "data/video_temporal/external/i3d_torchscript.pt") -> Optional[torch.jit.ScriptModule]:
    """Try to load I3D TorchScript model. Returns None if unavailable."""
    p = Path(i3d_path)
    if not p.exists():
        print(f"  [FVD] I3D model not found at {p}, will use pixel-space FVD")
        return None
    try:
        model = torch.jit.load(str(p)).eval()
        if DEVICE == "cuda":
            model = model.cuda()
        print(f"  [FVD] Loaded I3D model from {p}")
        return model
    except Exception as e:
        print(f"  [FVD] Failed to load I3D: {e}, will use pixel-space FVD")
        return None


def _i3d_forward_features(model: torch.jit.ScriptModule, x: torch.Tensor) -> torch.Tensor:
    """Run I3D forward up to avg_pool to get 1024-dim features (skip classification head).

    Standard FVD uses pre-logit features, NOT softmax probabilities.
    """
    x = model.conv3d_1a_7x7(x)
    x = model.maxPool3d_2a_3x3(x)
    x = model.conv3d_2b_1x1(x)
    x = model.conv3d_2c_3x3(x)
    x = model.maxPool3d_3a_3x3(x)
    x = model.mixed_3b(x)
    x = model.mixed_3c(x)
    x = model.maxPool3d_4a_3x3(x)
    x = model.mixed_4b(x)
    x = model.mixed_4c(x)
    x = model.mixed_4d(x)
    x = model.mixed_4e(x)
    x = model.mixed_4f(x)
    x = model.maxPool3d_5a_2x2(x)
    x = model.mixed_5b(x)
    x = model.mixed_5c(x)
    x = model.avg_pool(x)
    return x.flatten(1)  # (B, 1024)


@torch.no_grad()
def extract_i3d_features(
    videos: torch.Tensor,
    i3d_model: torch.jit.ScriptModule,
    batch_size: int = 8,
    target_frames: int = 16,
) -> torch.Tensor:
    """Extract I3D features from video clips.

    Args:
        videos: (N, T, H, W) float32 [0,1] grayscale
        i3d_model: loaded I3D TorchScript model
        batch_size: processing batch size
        target_frames: I3D temporal window (usually 16)

    Returns:
        features: (N, feature_dim) float32
    """
    N, T, H, W = videos.shape
    all_features = []

    for i in range(0, N, batch_size):
        batch = videos[i:i+batch_size]
        B = batch.shape[0]

        # Temporal: subsample or pad to target_frames
        if T > target_frames:
            indices = torch.linspace(0, T - 1, target_frames).long()
            batch = batch[:, indices]
        elif T < target_frames:
            # Pad by repeating last frame
            pad = batch[:, -1:].expand(-1, target_frames - T, -1, -1)
            batch = torch.cat([batch, pad], dim=1)

        # Grayscale -> RGB: (B, T, H, W) -> (B, T, 3, H, W)
        batch = batch.unsqueeze(2).expand(-1, -1, 3, -1, -1)

        # Resize to 224x224
        B2, T2, C, _, _ = batch.shape
        batch = batch.reshape(B2 * T2, C, H, W)
        batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
        batch = batch.reshape(B2, T2, C, 224, 224)

        # I3D expects (B, C, T, H, W)
        batch = batch.permute(0, 2, 1, 3, 4).to(DEVICE)

        try:
            feat = _i3d_forward_features(i3d_model, batch)
            all_features.append(feat.cpu())
        except Exception as e:
            print(f"  [I3D] Error at batch {i}: {e}")
            # Return what we have so far
            break

        del batch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not all_features:
        return None
    return torch.cat(all_features, dim=0)


def compute_i3d_fvd(
    real_frames: torch.Tensor,
    gen_frames: torch.Tensor,
    i3d_model: torch.jit.ScriptModule,
    batch_size: int = 8,
) -> Optional[float]:
    """Compute FVD using I3D features.

    Returns None if extraction fails.
    """
    from scipy.linalg import sqrtm

    real_feat = extract_i3d_features(real_frames, i3d_model, batch_size)
    gen_feat = extract_i3d_features(gen_frames, i3d_model, batch_size)

    if real_feat is None or gen_feat is None:
        return None

    real_feat = real_feat.numpy()
    gen_feat = gen_feat.numpy()

    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    sigma_r = np.cov(real_feat, rowvar=False)
    sigma_g = np.cov(gen_feat, rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fvd


def compute_fvd(
    real_frames: torch.Tensor,
    gen_frames: torch.Tensor,
    i3d_model: Optional[torch.jit.ScriptModule] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Compute FVD using best available method.

    Always computes pixel-space FVD. If I3D is available, also computes I3D FVD.
    Also computes Temporal FVD (FVD-T) on frame differences to measure motion quality.

    Returns dict with 'pixel_fvd', optionally 'i3d_fvd', and temporal variants.
    """
    results = {}

    # Always compute pixel-space FVD (fast, reliable)
    t0 = time.time()
    results["pixel_fvd"] = compute_pixel_fvd(real_frames, gen_frames)
    print(f"    pixel_fvd = {results['pixel_fvd']:.2f}  ({time.time()-t0:.1f}s)")

    # Try I3D FVD if model available
    if i3d_model is not None:
        t0 = time.time()
        i3d_fvd = compute_i3d_fvd(real_frames, gen_frames, i3d_model, batch_size)
        if i3d_fvd is not None:
            results["i3d_fvd"] = i3d_fvd
            print(f"    i3d_fvd = {results['i3d_fvd']:.2f}  ({time.time()-t0:.1f}s)")

    # Temporal FVD: FVD on frame differences (motion fields)
    # This directly measures temporal/motion quality, not just spatial appearance
    if real_frames.shape[1] > 1:
        real_diffs = real_frames[:, 1:] - real_frames[:, :-1]
        gen_diffs = gen_frames[:, 1:] - gen_frames[:, :-1]
        # Shift to [0,1] range for FVD computation (diffs are in [-1,1])
        real_diffs = (real_diffs + 1.0) / 2.0
        gen_diffs = (gen_diffs + 1.0) / 2.0

        t0 = time.time()
        results["temporal_pixel_fvd"] = compute_pixel_fvd(real_diffs, gen_diffs)
        print(f"    temporal_pixel_fvd = {results['temporal_pixel_fvd']:.2f}  ({time.time()-t0:.1f}s)")

        if i3d_model is not None:
            t0 = time.time()
            temporal_i3d = compute_i3d_fvd(real_diffs, gen_diffs, i3d_model, batch_size)
            if temporal_i3d is not None:
                results["temporal_i3d_fvd"] = temporal_i3d
                print(f"    temporal_i3d_fvd = {results['temporal_i3d_fvd']:.2f}  ({time.time()-t0:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Prediction quality metrics (MSE, SSIM)
# ---------------------------------------------------------------------------

def compute_prediction_metrics(
    real_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    train_frames: int,
    patches_per_frame: int,
    patch_size: int = 8,
    grid_h: int = 8,
    grid_w: int = 8,
) -> Dict[str, float]:
    """Compare generated future frames against ground-truth.

    Only compares the EXTRAPOLATED portion (frames beyond train_frames).

    Returns dict with mse, ssim, token_accuracy metrics.
    """
    context_len = train_frames * patches_per_frame
    total_len = min(real_tokens.shape[1], gen_tokens.shape[1])

    if total_len <= context_len:
        return {"mse": float('nan'), "token_accuracy": float('nan')}

    # Extract extrapolated portion only
    real_extrap = real_tokens[:, context_len:total_len]
    gen_extrap = gen_tokens[:, context_len:total_len]

    # Token-level accuracy
    n_correct = (real_extrap == gen_extrap).float().mean().item()

    # Decode to pixels for MSE
    real_frames = decode_tokens_to_frames(real_extrap, patch_size, grid_h, grid_w)
    gen_frames = decode_tokens_to_frames(gen_extrap, patch_size, grid_h, grid_w)

    mse = F.mse_loss(gen_frames, real_frames).item()

    # Simple SSIM approximation (mean-based, not full SSIM but fast)
    # Full SSIM is expensive; use a per-frame L1 + structure term
    l1 = F.l1_loss(gen_frames, real_frames).item()

    return {
        "mse": round(mse, 6),
        "l1": round(l1, 6),
        "token_accuracy": round(n_correct, 4),
    }


# ---------------------------------------------------------------------------
# Temporal coherence score
# ---------------------------------------------------------------------------

def compute_temporal_coherence(
    frames: torch.Tensor,
) -> float:
    """Measure temporal smoothness: average L2 of frame-to-frame differences.

    Lower = smoother (more temporally coherent).

    Args:
        frames: (N, T, H, W) float32 [0,1]

    Returns:
        mean frame-difference L2 norm
    """
    diffs = frames[:, 1:] - frames[:, :-1]  # (N, T-1, H, W)
    per_frame_l2 = diffs.pow(2).mean(dim=(2, 3)).sqrt()  # (N, T-1)
    return float(per_frame_l2.mean().item())


# ---------------------------------------------------------------------------
# Temporal quality metrics (inspired by FVMD, STREAM-T, RIFLEx)
# ---------------------------------------------------------------------------

def compute_temporal_quality(
    real_frames: torch.Tensor,
    gen_frames: torch.Tensor,
) -> Dict[str, float]:
    """Compute temporal-specific quality metrics beyond FVD.

    These metrics are designed to be sensitive to temporal quality differences
    that FVD misses (FVD is known to be dominated by spatial/per-frame quality).

    Metrics:
      - fvmd_lite: Simplified Frechet Video Motion Distance (velocity histograms)
      - dynamic_degree: Motion intensity in extrapolated frames
      - norepeat_score: Absence of temporal repetition (higher = less repetition)
      - temporal_autocorr_*: Autocorrelation at key lags (motion periodicity)
      - motion_fft_divergence: FFT spectrum divergence of temporal changes (STREAM-T inspired)

    Args:
        real_frames: (N, T, H, W) float32 [0,1]
        gen_frames:  (N, T, H, W) float32 [0,1]
    """
    results = {}

    # Frame differences (velocity proxy)
    real_vel = (real_frames[:, 1:] - real_frames[:, :-1]).numpy()  # (N, T-1, H, W)
    gen_vel = (gen_frames[:, 1:] - gen_frames[:, :-1]).numpy()

    # 1. Dynamic Degree: mean absolute motion (higher = more dynamic)
    real_dynamic = float(np.abs(real_vel).mean())
    gen_dynamic = float(np.abs(gen_vel).mean())
    results["dynamic_degree_real"] = round(real_dynamic, 6)
    results["dynamic_degree_gen"] = round(gen_dynamic, 6)
    results["dynamic_degree_ratio"] = round(gen_dynamic / max(real_dynamic, 1e-9), 4)

    # 2. FVMD-lite: velocity magnitude histogram Frechet distance
    real_mag = np.sqrt((real_vel ** 2).mean(axis=(2, 3)))  # (N, T-1)
    gen_mag = np.sqrt((gen_vel ** 2).mean(axis=(2, 3)))

    # Build 16-bin velocity magnitude histograms per video, average
    v_max = max(real_mag.max(), gen_mag.max(), 0.01)
    bins = np.linspace(0, v_max, 17)
    real_hist = np.zeros(16)
    gen_hist = np.zeros(16)
    for i in range(real_mag.shape[0]):
        h, _ = np.histogram(real_mag[i], bins=bins, density=True)
        real_hist += h
        h, _ = np.histogram(gen_mag[i], bins=bins, density=True)
        gen_hist += h
    real_hist /= real_mag.shape[0]
    gen_hist /= gen_mag.shape[0]

    # Frechet-like distance between 1D histograms (Wasserstein-1)
    fvmd = float(np.abs(np.cumsum(real_hist - gen_hist) * (bins[1] - bins[0])).sum())
    results["fvmd_lite"] = round(fvmd, 6)

    # 3. NoRepeat Score: detect temporal repetition in generated videos
    # Compare 16-frame windows, measure max cosine similarity between non-overlapping windows
    N, Tm1, H, W = gen_vel.shape
    window_size = min(16, Tm1 // 4)
    if window_size > 0 and Tm1 > window_size * 2:
        n_windows = Tm1 // window_size
        windows = []
        for w in range(n_windows):
            start = w * window_size
            chunk = gen_frames[:, start:start + window_size].reshape(N, -1).numpy()
            windows.append(chunk)

        # Compute max cosine similarity between non-adjacent windows
        max_sims = []
        for i in range(len(windows)):
            for j in range(i + 2, len(windows)):  # skip adjacent
                # Cosine similarity per video, then average
                norm_i = np.linalg.norm(windows[i], axis=1, keepdims=True) + 1e-9
                norm_j = np.linalg.norm(windows[j], axis=1, keepdims=True) + 1e-9
                cos_sim = ((windows[i] / norm_i) * (windows[j] / norm_j)).sum(axis=1)
                max_sims.append(cos_sim.mean())

        if max_sims:
            # NoRepeat = 1 - max_repetition. Higher = less repetition = better
            results["norepeat_score"] = round(float(1.0 - max(max_sims)), 4)
        # Also compute for real videos as reference
        windows_r = []
        for w in range(n_windows):
            start = w * window_size
            chunk = real_frames[:, start:start + window_size].reshape(N, -1).numpy()
            windows_r.append(chunk)
        max_sims_r = []
        for i in range(len(windows_r)):
            for j in range(i + 2, len(windows_r)):
                norm_i = np.linalg.norm(windows_r[i], axis=1, keepdims=True) + 1e-9
                norm_j = np.linalg.norm(windows_r[j], axis=1, keepdims=True) + 1e-9
                cos_sim = ((windows_r[i] / norm_i) * (windows_r[j] / norm_j)).sum(axis=1)
                max_sims_r.append(float(cos_sim.mean()))
        if max_sims_r:
            results["norepeat_score_real"] = round(float(1.0 - max(max_sims_r)), 4)

    # 4. Motion FFT divergence (STREAM-T inspired)
    # Compute temporal power spectrum of motion, compare distribution shape
    motion_energy_real = np.abs(real_vel).mean(axis=(2, 3))  # (N, T-1)
    motion_energy_gen = np.abs(gen_vel).mean(axis=(2, 3))

    # Average FFT power spectrum across videos
    fft_real = np.abs(np.fft.rfft(motion_energy_real - motion_energy_real.mean(axis=1, keepdims=True), axis=1)) ** 2
    fft_gen = np.abs(np.fft.rfft(motion_energy_gen - motion_energy_gen.mean(axis=1, keepdims=True), axis=1)) ** 2
    avg_psd_real = fft_real.mean(axis=0)
    avg_psd_gen = fft_gen.mean(axis=0)

    # Normalize to probability distributions
    avg_psd_real_n = avg_psd_real / (avg_psd_real.sum() + 1e-9)
    avg_psd_gen_n = avg_psd_gen / (avg_psd_gen.sum() + 1e-9)

    # Jensen-Shannon divergence of spectral shapes
    m = 0.5 * (avg_psd_real_n + avg_psd_gen_n)
    kl_rm = float(np.sum(avg_psd_real_n * np.log((avg_psd_real_n + 1e-12) / (m + 1e-12))))
    kl_gm = float(np.sum(avg_psd_gen_n * np.log((avg_psd_gen_n + 1e-12) / (m + 1e-12))))
    jsd = 0.5 * kl_rm + 0.5 * kl_gm
    results["motion_fft_jsd"] = round(jsd, 6)

    # 5. Temporal autocorrelation at key lags (oscillation fidelity)
    # For each video, compute autocorrelation of per-frame motion energy
    # Check lags 8, 16, 24, 32 (typical oscillation periods)
    for lag in [8, 16, 24, 32]:
        if lag >= Tm1:
            continue
        autocorr_real = []
        autocorr_gen = []
        for i in range(min(N, 256)):
            r = motion_energy_real[i]
            g = motion_energy_gen[i]
            r_centered = r - r.mean()
            g_centered = g - g.mean()
            r_var = max(r_centered.var(), 1e-9)
            g_var = max(g_centered.var(), 1e-9)
            ac_r = float(np.mean(r_centered[:-lag] * r_centered[lag:]) / r_var)
            ac_g = float(np.mean(g_centered[:-lag] * g_centered[lag:]) / g_var)
            autocorr_real.append(ac_r)
            autocorr_gen.append(ac_g)
        results[f"autocorr_lag{lag}_real"] = round(float(np.mean(autocorr_real)), 4)
        results[f"autocorr_lag{lag}_gen"] = round(float(np.mean(autocorr_gen)), 4)

    return results


# ---------------------------------------------------------------------------
# Main standalone evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    model: VideoGPT,
    test_tokens: torch.Tensor,
    train_frames: int,
    eval_frame_counts: List[int],
    patches_per_frame: int,
    patch_size: int,
    grid_h: int,
    grid_w: int,
    n_generate: int = 1024,
    temperature: float = 0.9,
    top_k: int = 50,
    gen_batch_size: int = 16,
    i3d_model=None,
    seed: int = 42,
    max_context_window: int = 0,
) -> Dict[str, dict]:
    """Full FVD evaluation for one model checkpoint.

    Returns dict keyed by frame count, each containing FVD + prediction metrics.
    """
    results = {}

    for target_frames in eval_frame_counts:
        print(f"\n  --- Evaluating {target_frames}f ({target_frames/train_frames:.1f}x extrap) ---")
        target_len = target_frames * patches_per_frame

        # Check if test data is long enough
        if test_tokens.shape[1] < target_len:
            print(f"    SKIP: test data only has {test_tokens.shape[1]} tokens, "
                  f"need {target_len}")
            continue

        t0 = time.time()

        # 1. Generate videos
        print(f"    Generating {n_generate} videos ({target_frames} frames each)...")
        gen_tokens = generate_videos_chunked(
            model=model,
            context_tokens=test_tokens,
            train_frames=train_frames,
            target_frames=target_frames,
            patches_per_frame=patches_per_frame,
            n_generate=n_generate,
            temperature=temperature,
            top_k=top_k,
            batch_size=gen_batch_size,
            seed=seed,
            max_context_window=max_context_window,
        )
        gen_time = time.time() - t0
        print(f"    Generated {gen_tokens.shape[0]} videos in {gen_time:.1f}s")

        # 2. Get matching real videos
        n_real = min(n_generate, test_tokens.shape[0])
        real_tokens_subset = test_tokens[:n_real, :target_len]

        # 3. Decode both to pixel frames
        real_frames = decode_tokens_to_frames(
            real_tokens_subset, patch_size, grid_h, grid_w
        )
        gen_frames = decode_tokens_to_frames(
            gen_tokens[:n_real, :target_len], patch_size, grid_h, grid_w
        )

        # 4. Compute FVD
        print(f"    Computing FVD...")
        fvd_results = compute_fvd(real_frames, gen_frames, i3d_model)

        # 5. Compute prediction metrics
        print(f"    Computing prediction metrics...")
        pred_metrics = compute_prediction_metrics(
            real_tokens_subset, gen_tokens[:n_real],
            train_frames, patches_per_frame, patch_size, grid_h, grid_w,
        )

        # 6. Temporal coherence
        real_coherence = compute_temporal_coherence(real_frames)
        gen_coherence = compute_temporal_coherence(gen_frames)

        # 7. Temporal quality metrics (FVMD-lite, NoRepeat, FFT, autocorrelation)
        print(f"    Computing temporal quality metrics...")
        temporal_quality = compute_temporal_quality(real_frames, gen_frames)

        frame_result = {
            "target_frames": target_frames,
            "extrap_ratio": round(target_frames / train_frames, 2),
            "n_generated": gen_tokens.shape[0],
            "generation_time_sec": round(gen_time, 1),
            **fvd_results,
            **pred_metrics,
            "temporal_coherence_real": round(real_coherence, 6),
            "temporal_coherence_gen": round(gen_coherence, 6),
            **temporal_quality,
        }
        results[f"{target_frames}f"] = frame_result

        # Print comprehensive results
        print(f"    {target_frames}f: pixel_fvd={fvd_results.get('pixel_fvd', 'N/A'):.2f}  "
              f"mse={pred_metrics.get('mse', 'N/A')}  "
              f"tok_acc={pred_metrics.get('token_accuracy', 'N/A')}")
        print(f"    temporal: fvmd={temporal_quality.get('fvmd_lite', 'N/A')}  "
              f"norepeat={temporal_quality.get('norepeat_score', 'N/A')}  "
              f"fft_jsd={temporal_quality.get('motion_fft_jsd', 'N/A')}  "
              f"dyn_ratio={temporal_quality.get('dynamic_degree_ratio', 'N/A')}")

        # Cleanup
        del gen_tokens, real_frames, gen_frames
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    print("generate_and_eval_fvd.py: module loaded successfully")
    print(f"  device={DEVICE}, dtype={DTYPE}")
    print("  Use as library: from generate_and_eval_fvd import evaluate_checkpoint")
