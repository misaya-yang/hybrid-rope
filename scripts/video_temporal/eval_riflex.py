#!/usr/bin/env python3
"""Standalone eval: RIFLEx + multi-timestep denoising precision on existing checkpoints.

Runs alongside training (uses ~3GB VRAM with eval_batch=2).
Tests EVQ×{YaRN, RIFLEx} × {t=0.2, 0.5, 0.8} matrix.

Usage:
    python eval_riflex.py --ckpt_dir results/video_dit/20260316_head2head
    python eval_riflex.py --ckpt_dir results/video_dit/20260316_h2h_seed137
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from video_dit import (
    VideoDiT,
    RectifiedFlowScheduler,
    evq_cosh_inv_freq,
    generate_oscillating_mnist_pixels,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def identify_k_riflex(base: float, dim: int, N: int):
    """Official RIFLEx k identification (thu-ml/RIFLEx/riflex_utils.py).

    Uses standard geometric formula to compute periods, finds the one
    closest to the repetition frame N.

    Args:
        base: RoPE base frequency
        dim: Full temporal dimension (K_t * 2)
        N: First repetition frame (= train_frames for our setup)
    Returns:
        k: 0-indexed intrinsic frequency index
        N_k: period of the selected frequency
    """
    n_freqs = dim // 2
    periods = []
    for j in range(1, n_freqs + 1):
        theta_j = 1.0 / (base ** (2 * (j - 1) / dim))
        N_j = round(2 * math.pi / theta_j)
        periods.append(N_j)
    diffs = [abs(N_j - N) for N_j in periods]
    k = diffs.index(min(diffs))  # 0-indexed
    return k, periods[k]


def apply_riflex(inv_freq_t: torch.Tensor, train_frames: int, target_frames: int,
                 base: float = 10000.0, dim: int = 32) -> torch.Tensor:
    """RIFLEx (ICML 2025): modify only the intrinsic frequency.

    Official implementation: thu-ml/RIFLEx
    1. Identify intrinsic frequency k via standard geometric periods
    2. Set freq[k] = 0.9 * 2pi / L_test (with safety margin)
    """
    freqs = inv_freq_t.clone().double()
    k, N_k = identify_k_riflex(base, dim, train_frames)
    # Official formula with 0.9 safety factor
    freqs[k] = 0.9 * 2.0 * math.pi / target_frames
    print(f"    RIFLEx: k={k}, original_period={N_k}, new_freq={freqs[k]:.6f}, "
          f"old_freq={inv_freq_t[k]:.6f}")
    return freqs.float()


@torch.no_grad()
def compute_denoising_precision(
    model: VideoDiT,
    val_videos: torch.Tensor,
    train_frames: int,
    noise_level: float = 0.5,
    scaling_method: str = "yarn",  # "yarn", "riflex", "none"
    batch_size: int = 2,
    device: str = "cuda",
    base: float = 10000.0,
) -> dict:
    """Per-frame denoising MSE with configurable scaling method."""
    model.eval()
    scheduler = RectifiedFlowScheduler()

    rope = model.blocks[0].attn.rope
    orig_inv_freq_t = rope.inv_freq_t.clone()
    n_frames = val_videos.shape[1]
    scale = n_frames / train_frames

    if scale > 1.0:
        if scaling_method == "yarn":
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
            new_freq = (orig_inv_freq_t.cpu().double() / yarn_scale).float()
            rope.inv_freq_t.copy_(new_freq.to(rope.inv_freq_t.device))
        elif scaling_method == "riflex":
            K_t = len(orig_inv_freq_t)
            new_freq = apply_riflex(orig_inv_freq_t, train_frames, n_frames,
                                    base=base, dim=K_t * 2)
            rope.inv_freq_t.copy_(new_freq.to(rope.inv_freq_t.device))
        # "none" = no scaling

    rope._build(n_frames + 4)

    use_autocast = device == "cuda" and DTYPE != torch.float32
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if use_autocast else nullcontext()

    pH, pW = model.pH, model.pW
    ppf = pH * pW
    N = val_videos.shape[0]

    frame_mse_sum = torch.zeros(n_frames, device="cpu")
    frame_count = 0

    for start in range(0, N, batch_size):
        B = min(batch_size, N - start)
        batch = val_videos[start:start + B].to(device)
        patches = model.patchify(batch)

        noise = torch.randn_like(patches)
        t = torch.full((B,), noise_level, device=device)
        x_t = scheduler.interpolate(patches, noise, t)
        v_target = scheduler.velocity(patches, noise)

        with ctx:
            v_pred = model(x_t, t)

        se = (v_pred.float() - v_target.float()).pow(2)
        se = se.reshape(B, n_frames, ppf, -1)
        frame_se = se.mean(dim=(2, 3))
        frame_mse_sum += frame_se.sum(dim=0).cpu()
        frame_count += B

    frame_mse = frame_mse_sum / frame_count

    tf = train_frames
    result = {
        "noise_level": noise_level,
        "scaling_method": scaling_method,
        "n_videos": N,
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


def load_model_from_ckpt(ckpt_path: str, device: str = "cuda") -> VideoDiT:
    """Load a VideoDiT from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    K_h, K_w, K_t = cfg["K_h"], cfg["K_w"], cfg["K_t"]
    tau = ckpt.get("tau", 0.0)
    base = cfg.get("base", 10000.0)

    inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0, base=base)
    inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0, base=base)
    # Use tau from checkpoint for temporal
    inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=tau, base=base)

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
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg, tau


def main():
    parser = argparse.ArgumentParser(description="RIFLEx + multi-timestep eval")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing checkpoint files")
    parser.add_argument("--eval_batch", type=int, default=2,
                        help="Eval batch size (keep small to share GPU)")
    parser.add_argument("--out", type=str, default="",
                        help="Output JSON path (default: ckpt_dir/riflex_eval.json)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out) if args.out else ckpt_dir / "riflex_eval.json"

    # Find all checkpoint files
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"No .pt files in {ckpt_dir}")
        return

    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    # Generate validation data (must match training data generation)
    print("Generating 128f validation data...")
    val_videos_128f = generate_oscillating_mnist_pixels(
        256, 128, 64, seed=99999,
    )

    scaling_methods = ["yarn", "riflex", "none"]
    noise_levels = [0.2, 0.5, 0.8]
    all_results = {}

    for ckpt_path in ckpt_files:
        name = ckpt_path.stem  # e.g. "tau0.00_seed42"
        print(f"\n{'='*50}")
        print(f"  Checkpoint: {name}")
        print(f"{'='*50}")

        model, cfg, tau = load_model_from_ckpt(str(ckpt_path), DEVICE)
        train_frames = cfg.get("train_frames", 32)
        all_results[name] = {"tau": tau, "base": cfg.get("base", 10000.0)}

        for method in scaling_methods:
            for t in noise_levels:
                key = f"{method}_t{t}"
                print(f"  {name} | {method} | t={t} ...", end=" ", flush=True)
                t0 = time.time()

                result = compute_denoising_precision(
                    model, val_videos_128f, train_frames,
                    noise_level=t,
                    scaling_method=method,
                    batch_size=args.eval_batch,
                    device=DEVICE,
                    base=cfg.get("base", 10000.0),
                )
                elapsed = time.time() - t0
                print(f"train={result['train_mse']:.6f} far={result.get('far_extrap_mse', -1):.6f} ({elapsed:.1f}s)")
                all_results[name][key] = result

        # Release model memory
        model.cpu()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("  COMPARISON: YaRN vs RIFLEx vs None (t=0.5)")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<25} {'Method':<10} {'Train MSE':>12} {'Far MSE':>12}")
    print("-" * 65)
    for name, data in all_results.items():
        for method in scaling_methods:
            key = f"{method}_t0.5"
            if key in data:
                r = data[key]
                print(f"{name:<25} {method:<10} {r['train_mse']:>12.6f} {r.get('far_extrap_mse', -1):>12.6f}")
        print()

    print(f"\n{'='*80}")
    print("  COMPARISON: Multi-timestep (YaRN)")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<25} {'t':>5} {'Train MSE':>12} {'Far MSE':>12}")
    print("-" * 55)
    for name, data in all_results.items():
        for t in noise_levels:
            key = f"yarn_t{t}"
            if key in data:
                r = data[key]
                print(f"{name:<25} {t:>5.1f} {r['train_mse']:>12.6f} {r.get('far_extrap_mse', -1):>12.6f}")
        print()


if __name__ == "__main__":
    main()
