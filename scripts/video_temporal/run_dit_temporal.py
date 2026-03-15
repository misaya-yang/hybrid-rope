#!/usr/bin/env python3
"""Video DiT temporal extrapolation experiment — EVQ-Cosh on 3D RoPE.

Why DiT instead of AR VideoGPT?
  The AR architecture + top-k sampling compresses distributional differences:
  EVQ has 27% better PPL but only 1.5% better FVD because both models share
  similar top-k token sets. DiT avoids this entirely — denoising uses the
  FULL learned distribution, so frequency allocation quality directly
  affects generation quality.

  If EVQ's PPL advantage is real (it is), DiT should show a MUCH larger
  FVD gap than the 1.5% we see with AR VideoGPT.

Experiment design:
  1. Train DiT on 32-frame oscillating MNIST (pixel space, 64x64)
  2. Generate 128-frame videos (4x temporal extrapolation)
  3. Compare geo vs EVQ temporal frequency allocations
  4. Evaluate with VideoMAE FVD + I3D FVD

Usage:
    # Full experiment (both methods, 1 seed each for quick verification)
    python scripts/video_temporal/run_dit_temporal.py --seed 42

    # Quick test (fewer steps)
    python scripts/video_temporal/run_dit_temporal.py --seed 42 --quick

    # EVQ only
    python scripts/video_temporal/run_dit_temporal.py --method evq --seed 42

Hardware: NVIDIA RTX PRO 6000 Blackwell 96GB (~2h per method)
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

# MUST set before any torch import — CUDA allocator config is read at init time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from video_dit import (
    VideoDiT,
    RectifiedFlowScheduler,
    RotaryEmbedding3D,
    evq_cosh_inv_freq,
    power_shift_inv_freq,
    generate_oscillating_mnist_pixels,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dit(
    model: VideoDiT,
    train_videos: torch.Tensor,  # (N, T, C, H, W) in [-1, 1]
    cfg: dict,
    seed: int = 42,
) -> Dict[str, float]:
    """Train DiT with Rectified Flow (flow matching) objective."""
    model.train()
    set_seed(seed)

    scheduler = RectifiedFlowScheduler()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    batch_size = cfg["batch_size"]
    n_samples = train_videos.shape[0]
    steps = cfg.get("steps", (n_samples * cfg.get("epochs", 100)) // batch_size)
    warmup = int(steps * 0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)

    use_autocast = DEVICE == "cuda" and DTYPE != torch.float32
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if use_autocast else nullcontext()
    # bf16 has same exponent range as fp32 → GradScaler not needed (only for fp16)
    use_fp16_scaler = use_autocast and DTYPE == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_fp16_scaler else None

    losses = []
    t0 = time.time()

    for step in range(steps):
        # Cosine LR with warmup
        if step < warmup:
            cur_lr = lr * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        # Sample batch
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = train_videos[indices].to(DEVICE)  # (B, T, C, H, W)

        # Patchify
        patches = model.patchify(batch)  # (B, N, patch_dim)

        # Flow matching: sample t ∈ [0, 1], interpolate, predict velocity
        t = torch.rand(batch_size, device=DEVICE)
        noise = torch.randn_like(patches)
        x_t = scheduler.interpolate(patches, noise, t)
        v_target = scheduler.velocity(patches, noise)

        with ctx:
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        losses.append(loss.item())

        if step % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (steps - step - 1) if step > 0 else 0
            print(f"    step {step}/{steps}  loss={avg_loss:.6f}  lr={cur_lr:.2e}  "
                  f"ETA={eta / 60:.0f}min")

    elapsed = time.time() - t0
    final_loss = sum(losses[-500:]) / len(losses[-500:])
    print(f"  Training done in {elapsed / 60:.1f}min  final_loss={final_loss:.6f}")

    return {"final_loss": final_loss, "elapsed_min": elapsed / 60, "steps": steps}


# ---------------------------------------------------------------------------
# Generation with temporal extrapolation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_videos(
    model: VideoDiT,
    n_videos: int,
    n_frames: int,
    train_frames: int,
    sampling_steps: int = 50,
    batch_size: int = 8,
    apply_yarn: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate videos using Euler ODE (Rectified Flow) with optional temporal YaRN.

    Args:
        model: Trained VideoDiT
        n_videos: Number of videos to generate
        n_frames: Frames per video (can be > train_frames for extrapolation)
        train_frames: Number of frames used during training
        sampling_steps: Number of Euler ODE steps (50 is typically sufficient)
        batch_size: Generation batch size
        apply_yarn: Whether to apply YaRN for temporal extrapolation
        device: Device

    Returns:
        videos: (n_videos, n_frames, C, H, W) in [-1, 1]
    """
    model.eval()
    scheduler = RectifiedFlowScheduler()
    rope = model.blocks[0].attn.rope
    orig_inv_freq_t = rope.inv_freq_t.clone()

    scale = n_frames / train_frames
    if apply_yarn and scale > 1.0:
        # YaRN-style interpolation on temporal frequencies
        K = len(orig_inv_freq_t)
        idx = torch.arange(K, dtype=torch.float64)
        start = int(0.20 * K)
        end = int(0.90 * K)
        if end <= start:
            end = min(K - 1, start + 1)
        ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
        temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
        yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
        yarn_freq = (orig_inv_freq_t.cpu().double() / yarn_scale).float()
        rope.inv_freq_t.copy_(yarn_freq.to(rope.inv_freq_t.device))

    # Rebuild RoPE cache for extrapolated length
    rope._build(n_frames + 4)

    pH, pW = model.pH, model.pW
    N = n_frames * pH * pW  # total tokens per video
    patch_dim = model.in_channels * model.patch_size * model.patch_size

    all_videos = []
    for start_idx in range(0, n_videos, batch_size):
        B = min(batch_size, n_videos - start_idx)
        shape = (B, N, patch_dim)

        # Euler ODE sampling: t=1 (noise) → t=0 (clean)
        # Pass DTYPE so scheduler wraps forward passes in autocast → FlashAttention2
        x = scheduler.sample(model, shape, num_steps=sampling_steps,
                             device=device, dtype=DTYPE if DTYPE != torch.float32 else None)

        # Unpatchify to pixel space
        videos = model.unpatchify(x, n_frames)  # (B, T, C, H, W)
        all_videos.append(videos.cpu())

        if (start_idx // batch_size) % 5 == 0:
            print(f"    generated {start_idx + B}/{n_videos} videos")

    # Restore original temporal frequencies
    rope.inv_freq_t.copy_(orig_inv_freq_t)

    return torch.cat(all_videos, dim=0)


# ---------------------------------------------------------------------------
# FVD evaluation
# ---------------------------------------------------------------------------

def compute_fvd(
    real_videos: torch.Tensor,
    gen_videos: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute FVD using cd-fvd library (VideoMAE + I3D).

    Falls back to skip if model weights can't be downloaded (no internet).
    """
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("FVD model download timed out")

    try:
        from cdfvd.fvd import cdfvd
    except ImportError:
        print("  WARNING: cd-fvd not installed, skipping FVD")
        return {"videomae_fvd": -1.0, "i3d_fvd": -1.0}

    results = {}

    def preprocess(videos: torch.Tensor, n_frames: int = 16) -> np.ndarray:
        N, T, C, H, W = videos.shape
        indices = np.linspace(0, T - 1, n_frames, dtype=int)
        v = videos[:, indices]
        if C == 1:
            v = v.repeat(1, 1, 3, 1, 1)
        v_flat = v.reshape(-1, 3, H, W)
        v_resized = F.interpolate(v_flat, size=(224, 224), mode="bilinear", align_corners=False)
        v_out = v_resized.reshape(N, n_frames, 3, 224, 224)
        v_out = (v_out + 1.0) / 2.0
        v_out = (v_out.permute(0, 1, 3, 4, 2).clamp(0, 1) * 255).to(torch.uint8)
        return v_out.numpy()

    real_np = preprocess(real_videos)
    gen_np = preprocess(gen_videos)

    for name, extractor in [("videomae", "videomae"), ("i3d", "i3d")]:
        try:
            # 30s timeout for model download — skip if no internet
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(600)
            fvd_calc = cdfvd()
            fvd_calc.load_feature_extractor(extractor, ckpt_path=None, device=device)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            real_feats = fvd_calc.compute_features(real_np)
            gen_feats = fvd_calc.compute_features(gen_np)
            val = float(fvd_calc.compute_fvd(real_feats, gen_feats))
            results[f"{name}_fvd"] = round(val, 2)
            print(f"    {name} FVD: {val:.2f}")
        except Exception as e:
            signal.alarm(0)
            print(f"    {name} FVD skipped: {e}")
            results[f"{name}_fvd"] = -1.0

    return results


def compute_pixel_metrics(
    real_videos: torch.Tensor,
    gen_videos: torch.Tensor,
) -> Dict[str, float]:
    """Compute pixel-space metrics: MSE, PSNR, temporal consistency."""
    # Use min of the two counts
    N = min(len(real_videos), len(gen_videos))
    real = real_videos[:N]
    gen = gen_videos[:N]

    # MSE and PSNR (on [-1,1] scale)
    mse = F.mse_loss(gen, real).item()
    psnr = -10 * math.log10(mse + 1e-10)

    # Temporal consistency: mean absolute frame difference
    def temporal_diff(v):
        return (v[:, 1:] - v[:, :-1]).abs().mean().item()

    real_tc = temporal_diff(real)
    gen_tc = temporal_diff(gen)

    return {
        "mse": round(mse, 6),
        "psnr": round(psnr, 2),
        "temporal_consistency_real": round(real_tc, 4),
        "temporal_consistency_gen": round(gen_tc, 4),
    }


# ---------------------------------------------------------------------------
# Denoising precision (DiT equivalent of teacher-forced evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_denoising_precision(
    model: VideoDiT,
    val_videos: torch.Tensor,  # (N, T, C, H, W) in [-1, 1]
    train_frames: int,
    noise_level: float = 0.5,
    apply_yarn: bool = True,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, float]:
    """Per-frame denoising MSE — DiT's teacher-forced evaluation.

    Add noise at level t to real videos, predict velocity, measure per-frame
    prediction quality. Like teacher-forced eval: model sees (noisy) ground truth,
    directly measures per-position prediction quality without sampling accumulation.
    """
    model.eval()
    scheduler = RectifiedFlowScheduler()

    rope = model.blocks[0].attn.rope
    orig_inv_freq_t = rope.inv_freq_t.clone()
    n_frames = val_videos.shape[1]

    scale = n_frames / train_frames
    if apply_yarn and scale > 1.0:
        K = len(orig_inv_freq_t)
        idx = torch.arange(K, dtype=torch.float64)
        start = int(0.20 * K)
        end = int(0.90 * K)
        if end <= start:
            end = min(K - 1, start + 1)
        ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
        yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
        yarn_freq = (orig_inv_freq_t.cpu().double() / yarn_scale).float()
        rope.inv_freq_t.copy_(yarn_freq.to(rope.inv_freq_t.device))
    rope._build(n_frames + 4)

    use_autocast = device == "cuda" and DTYPE != torch.float32
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if use_autocast else nullcontext()

    pH, pW = model.pH, model.pW
    ppf = pH * pW
    N = val_videos.shape[0]

    # Accumulate per-frame MSE
    frame_mse_sum = torch.zeros(n_frames, device="cpu")
    frame_count = 0

    for start in range(0, N, batch_size):
        B = min(batch_size, N - start)
        batch = val_videos[start:start + B].to(device)
        patches = model.patchify(batch)  # (B, T*ppf, patch_dim)

        noise = torch.randn_like(patches)
        t = torch.full((B,), noise_level, device=device)
        x_t = scheduler.interpolate(patches, noise, t)
        v_target = scheduler.velocity(patches, noise)

        with ctx:
            v_pred = model(x_t, t)

        # Per-token squared error, reshape to (B, T, ppf, patch_dim)
        se = (v_pred.float() - v_target.float()).pow(2)
        se = se.reshape(B, n_frames, ppf, -1)
        # Mean over spatial patches and patch_dim → (B, T)
        frame_se = se.mean(dim=(2, 3))
        frame_mse_sum += frame_se.sum(dim=0).cpu()
        frame_count += B

    frame_mse = frame_mse_sum / frame_count

    # Region statistics
    tf = train_frames
    result = {
        "noise_level": noise_level,
        "n_videos": N,
        "per_frame_mse": frame_mse.tolist(),
        "train_mse": float(frame_mse[:tf].mean()),
    }
    if n_frames > tf:
        result["all_extrap_mse"] = float(frame_mse[tf:].mean())
    ne = min(2 * tf, n_frames)
    if ne > tf:
        result["near_extrap_mse"] = float(frame_mse[tf:ne].mean())
    me = min(3 * tf, n_frames)
    if me > ne:
        result["mid_extrap_mse"] = float(frame_mse[ne:me].mean())
    if n_frames > me:
        result["far_extrap_mse"] = float(frame_mse[me:].mean())

    # Restore
    rope.inv_freq_t.copy_(orig_inv_freq_t)

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_one_method(
    method: str,
    seed: int,
    cfg: dict,
    train_videos: torch.Tensor,
    val_videos_32f: torch.Tensor,
    val_videos_128f: torch.Tensor,
    work_dir: Path,
    tau_override: Optional[float] = None,
    alpha_override: Optional[float] = None,
) -> dict:
    """Run a single method (geo/evq/custom tau/power-shift alpha) and return results."""
    print(f"\n{'='*60}")
    print(f"  Method: {method}  Seed: {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # Frequency allocation
    K_t = cfg["K_t"]
    K_h = cfg["K_h"]
    K_w = cfg["K_w"]
    train_frames = cfg["train_frames"]

    # Spatial: always geometric
    inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0, base=cfg["base"])
    inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0, base=cfg["base"])

    # Temporal frequency allocation
    if alpha_override is not None:
        # Route B: power-shift family (DiT-optimized)
        inv_freq_t = power_shift_inv_freq(K_t * 2, alpha=alpha_override, base=cfg["base"])
        tau = -1.0  # sentinel: not using cosh
        print(f"  family = power-shift  alpha = {alpha_override:.4f}")
    else:
        # Route A: cosh family (custom tau, geo, or evq)
        if tau_override is not None:
            tau = tau_override
        elif method == "geo":
            tau = 0.0
        elif method == "evq":
            tau = K_t / math.sqrt(train_frames)
        else:
            raise ValueError(f"Unknown method: {method}")
        inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=tau, base=cfg["base"])
        print(f"  family = cosh  tau = {tau:.4f}")
    print(f"  inv_freq_t range: [{inv_freq_t.min().item():.8f}, {inv_freq_t.max().item():.6f}]")

    # Build model
    model = VideoDiT(
        in_channels=1,
        patch_size=cfg["patch_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        frame_size=cfg["frame_size"],
        max_T=cfg.get("max_T", 256),
        inv_freq_h=inv_freq_h,
        inv_freq_w=inv_freq_w,
        inv_freq_t=inv_freq_t,
    ).to(DEVICE)
    # gradient checkpointing only for match profile on large GPUs
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing = True

    # Train (or load existing checkpoint)
    run_id = f"{method}_seed{seed}"
    ckpt_path = work_dir / f"{run_id}.pt"

    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        train_result = ckpt.get("train_result", {"final_loss": -1, "elapsed_min": 0, "steps": 0})
    else:
        train_result = train_dit(model, train_videos, cfg, seed=seed)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "method": method,
            "tau": tau,
            "seed": seed,
            "train_result": train_result,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # Generate videos at train length (32f) and extrapolated (128f)
    n_gen = cfg["n_gen_videos"]
    gen_batch = cfg.get("gen_batch_size", 4)
    sampling_steps = cfg.get("sampling_steps", 50)

    # Generate videos (or load existing)
    gen_32f_path = work_dir / f"{run_id}_gen_32f.npy"
    gen_128f_path = work_dir / f"{run_id}_gen_128f.npy"
    gen_128f_ny_path = work_dir / f"{run_id}_gen_128f_noyarn.npy"

    if gen_32f_path.exists():
        print(f"\n  Loading existing generated videos...")
        gen_32f = torch.from_numpy(np.load(gen_32f_path))
        gen_128f = torch.from_numpy(np.load(gen_128f_path))
        gen_128f_noyarn = torch.from_numpy(np.load(gen_128f_ny_path))
    else:
        print(f"\n  Generating {n_gen} videos at {train_frames}f (in-distribution)...")
        gen_32f = generate_videos(
            model, n_gen, train_frames, train_frames,
            sampling_steps=sampling_steps, batch_size=gen_batch,
            apply_yarn=False, device=DEVICE,
        )
        print(f"\n  Generating {n_gen} videos at 128f (4x extrapolation, YaRN)...")
        gen_128f = generate_videos(
            model, n_gen, 128, train_frames,
            sampling_steps=sampling_steps, batch_size=gen_batch,
            apply_yarn=True, device=DEVICE,
        )
        print(f"\n  Generating {n_gen} videos at 128f (4x extrapolation, no YaRN)...")
        gen_128f_noyarn = generate_videos(
            model, n_gen, 128, train_frames,
            sampling_steps=sampling_steps, batch_size=gen_batch,
            apply_yarn=False, device=DEVICE,
        )
        np.save(gen_32f_path, gen_32f.numpy())
        np.save(gen_128f_path, gen_128f.numpy())
        np.save(gen_128f_ny_path, gen_128f_noyarn.numpy())

    # Evaluate FVD
    results = {"method": method, "seed": seed, "tau": tau, **train_result}

    print(f"\n  Computing FVD at 32f (in-distribution)...")
    fvd_32f = compute_fvd(val_videos_32f, gen_32f, device=DEVICE)
    results["fvd_32f"] = fvd_32f

    print(f"\n  Computing FVD at 128f (4x extrapolation, YaRN)...")
    fvd_128f = compute_fvd(val_videos_128f, gen_128f, device=DEVICE)
    results["fvd_128f_yarn"] = fvd_128f

    print(f"\n  Computing FVD at 128f (4x extrapolation, no YaRN)...")
    fvd_128f_ny = compute_fvd(val_videos_128f, gen_128f_noyarn, device=DEVICE)
    results["fvd_128f_noyarn"] = fvd_128f_ny

    # Pixel metrics
    pix_32f = compute_pixel_metrics(val_videos_32f[:n_gen], gen_32f)
    pix_128f = compute_pixel_metrics(val_videos_128f[:n_gen], gen_128f)
    results["pixel_32f"] = pix_32f
    results["pixel_128f_yarn"] = pix_128f

    # Denoising precision (DiT equivalent of teacher-forced evaluation)
    print(f"\n  Computing denoising precision at 128f (YaRN, t=0.5)...")
    denoise_128f = compute_denoising_precision(
        model, val_videos_128f, train_frames,
        noise_level=0.5, apply_yarn=True,
        batch_size=gen_batch, device=DEVICE,
    )
    results["denoise_128f_yarn"] = denoise_128f

    print(f"\n  Computing denoising precision at 128f (no YaRN, t=0.5)...")
    denoise_128f_ny = compute_denoising_precision(
        model, val_videos_128f, train_frames,
        noise_level=0.5, apply_yarn=False,
        batch_size=gen_batch, device=DEVICE,
    )
    results["denoise_128f_noyarn"] = denoise_128f_ny

    # Print denoising summary
    for label, d in [("YaRN", denoise_128f), ("no-YaRN", denoise_128f_ny)]:
        print(f"    Denoising MSE ({label}): train={d['train_mse']:.6f}  "
              f"extrap={d.get('all_extrap_mse', -1):.6f}")

    # Save results
    result_path = work_dir / f"{run_id}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    return results


def print_comparison(all_results: List[dict]):
    """Print comparison table — supports arbitrary method names (tau sweep)."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")

    # Group by method (preserve order)
    method_order = []
    by_method: Dict[str, list] = {}
    for r in all_results:
        m = r["method"]
        if m not in by_method:
            by_method[m] = []
            method_order.append(m)
        by_method[m].append(r)

    if len(method_order) < 2:
        print("  Need at least 2 methods for comparison")
        return

    def avg_metric(results, *keys):
        vals = []
        for r in results:
            v = r
            for k in keys:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and isinstance(v, (int, float)) and v >= 0:
                vals.append(v)
        return sum(vals) / len(vals) if vals else -1

    metrics = [
        ("Train loss", "final_loss"),
        ("Denoise train (YaRN)", "denoise_128f_yarn", "train_mse"),
        ("Denoise extrap (YaRN)", "denoise_128f_yarn", "all_extrap_mse"),
        ("Denoise near (YaRN)", "denoise_128f_yarn", "near_extrap_mse"),
        ("Denoise mid (YaRN)", "denoise_128f_yarn", "mid_extrap_mse"),
        ("Denoise far (YaRN)", "denoise_128f_yarn", "far_extrap_mse"),
        ("Denoise extrap (no YaRN)", "denoise_128f_noyarn", "all_extrap_mse"),
    ]

    # Header
    col_w = max(10, max(len(m) for m in method_order) + 2)
    header = f"  {'Metric':<30}"
    for m in method_order:
        header += f" {m:>{col_w}}"
    header += f" {'Best':>{col_w}}"
    print(f"\n{header}")
    print(f"  {'-'*(30 + (col_w+1) * (len(method_order) + 1))}")

    for metric_info in metrics:
        name = metric_info[0]
        keys = metric_info[1:]
        vals = {}
        for m in method_order:
            vals[m] = avg_metric(by_method[m], *keys)

        valid = {m: v for m, v in vals.items() if v >= 0}
        if len(valid) >= 2:
            best_m = min(valid, key=valid.get)  # lower is better for all these metrics
            row = f"  {name:<30}"
            for m in method_order:
                v = vals[m]
                marker = " *" if m == best_m and len(valid) > 1 else ""
                row += f" {v:>{col_w}.6f}" if v < 0.1 else f" {v:>{col_w}.4f}"
            row += f" {best_m:>{col_w}}"
            print(row)
        else:
            row = f"  {name:<30}"
            for m in method_order:
                row += f" {vals[m]:>{col_w}.2f}"
            row += f" {'N/A':>{col_w}}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Video DiT temporal extrapolation")
    parser.add_argument("--method", type=str, default="both",
                        choices=["geo", "evq", "both"],
                        help="Which frequency allocation to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated seeds (overrides --seed)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer steps and videos")
    parser.add_argument("--work_dir", type=str, default="")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load checkpoints and evaluate")
    parser.add_argument("--n_gen", type=int, default=128,
                        help="Number of videos to generate for FVD")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="Euler ODE sampling steps (50 is usually enough)")
    parser.add_argument("--profile", type=str, default="default",
                        choices=["default", "medium", "match"],
                        help="Model profile: 'default' (~50M, fast) or 'match' (~250M, needs 80GB+)")
    parser.add_argument("--tau", type=str, default="",
                        help="Comma-separated tau values to sweep (overrides --method). "
                             "Example: --tau 0.0,0.3,0.7,1.5,2.83")
    parser.add_argument("--alpha", type=str, default="",
                        help="Comma-separated alpha values for power-shift family (Route B). "
                             "Example: --alpha 0.0,0.3,0.5,1.0")
    args = parser.parse_args()

    # Seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    # Methods / tau sweep / alpha sweep
    tau_sweep = []
    alpha_sweep = []
    if args.alpha:
        alpha_sweep = [float(a) for a in args.alpha.split(",")]
        methods = [f"alpha{a:.2f}" for a in alpha_sweep]
    elif args.tau:
        tau_sweep = [float(t) for t in args.tau.split(",")]
        methods = [f"tau{t:.2f}" for t in tau_sweep]
    elif args.method == "both":
        methods = ["geo", "evq"]
    else:
        methods = [args.method]

    # Model profiles
    PROFILES = {
        "default": {
            # ~50M params, fast training. 8×8 patches = 64 tokens/frame (same as VideoGPT)
            # geo vs evq comparison is relative — small model shows same relative differences
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "head_dim": 64,
            "patch_size": 8,    # 64/8 = 8×8 = 64 patches/frame
            "lr": 2e-4,
            "batch_size": 64,
            "epochs": 30,
            "gen_batch_size": 16,
        },
        "medium": {
            # ~125M params — larger model to confirm same geo vs evq trends
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "head_dim": 64,
            "patch_size": 8,
            "lr": 1.5e-4,
            "batch_size": 16,
            "epochs": 15,
            "gen_batch_size": 8,
        },
        "match": {
            # Match VideoGPT capacity (~250M) — only use on 80GB+ GPU
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "head_dim": 64,
            "patch_size": 8,
            "lr": 1e-4,
            "batch_size": 16,
            "epochs": 50,
            "gen_batch_size": 4,
        },
    }

    profile = PROFILES[args.profile]
    cfg = {
        **profile,
        "frame_size": 64,
        "max_T": 256,

        # Frequency allocation (same as VideoGPT oscillating MNIST)
        "K_h": 8,
        "K_w": 8,
        "K_t": 16,
        "base": 10000.0,

        # Data
        "train_frames": 32,
        "train_samples": 16000,  # match VideoGPT
        "val_samples": 256,
        "n_gen_videos": args.n_gen,

        # Sampling
        "sampling_steps": args.sampling_steps,
    }

    if args.quick:
        cfg["epochs"] = 10
        cfg["train_samples"] = 1000
        cfg["val_samples"] = 64
        cfg["n_gen_videos"] = 16
        cfg["sampling_steps"] = 20
    pH = cfg["frame_size"] // cfg["patch_size"]
    ppf = pH * pH
    train_tokens = cfg["train_frames"] * ppf
    tau_star = cfg["K_t"] / math.sqrt(cfg["train_frames"])
    print(f"\n{'#'*60}")
    print(f"  VIDEO DiT — RECTIFIED FLOW + 3D RoPE")
    print(f"  Profile: {args.profile}  methods={methods}  seeds={seeds}")
    print(f"  Architecture: {cfg['num_layers']}L {cfg['hidden_size']}D {cfg['num_heads']}H")
    print(f"  Patch: {cfg['patch_size']}×{cfg['patch_size']} → {pH}×{pH} = {ppf} tokens/frame")
    print(f"  Train: {cfg['train_frames']}f × {ppf} = {train_tokens} tokens")
    print(f"  Freq split: K_h={cfg['K_h']}, K_w={cfg['K_w']}, K_t={cfg['K_t']}")
    print(f"  τ* = {cfg['K_t']}/√{cfg['train_frames']} = {tau_star:.4f}")
    print(f"  Sampling: Euler ODE × {cfg['sampling_steps']} steps")
    print(f"  Device: {DEVICE}  dtype: {DTYPE}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'#'*60}")

    # Work dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        work_dir = Path("results") / "video_dit" / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Work dir: {work_dir}")

    # Save config
    with open(work_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Generate data
    print(f"\n  Generating training data ({cfg['train_samples']} videos, {cfg['train_frames']}f)...")
    train_videos = generate_oscillating_mnist_pixels(
        cfg["train_samples"], cfg["train_frames"],
        cfg["frame_size"], seed=42,
    )
    print(f"  Train shape: {train_videos.shape}  range: [{train_videos.min():.1f}, {train_videos.max():.1f}]")

    print(f"  Generating validation data (32f)...")
    val_videos_32f = generate_oscillating_mnist_pixels(
        cfg["val_samples"], cfg["train_frames"],
        cfg["frame_size"], seed=99999,
    )

    print(f"  Generating validation data (128f for extrapolation eval)...")
    val_videos_128f = generate_oscillating_mnist_pixels(
        cfg["val_samples"], 128,
        cfg["frame_size"], seed=99999,
    )

    # Run experiments
    all_results = []

    for i, method in enumerate(methods):
        tau_override = tau_sweep[i] if tau_sweep else None
        alpha_override = alpha_sweep[i] if alpha_sweep else None
        for seed in seeds:
            result = run_one_method(
                method, seed, cfg,
                train_videos, val_videos_32f, val_videos_128f,
                work_dir,
                tau_override=tau_override,
                alpha_override=alpha_override,
            )
            all_results.append(result)

    # Comparison
    print_comparison(all_results)

    # Save summary
    summary_path = work_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results: {work_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
