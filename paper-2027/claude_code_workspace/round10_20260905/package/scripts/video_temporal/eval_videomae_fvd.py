#!/usr/bin/env python3
"""VideoMAE-v2 FVD evaluation — temporal-sensitive video quality metric.

Motivation (CVPR 2024 "On the Content Bias in Fréchet Video Distance"):
  - I3D features are heavily biased toward per-frame spatial quality
  - Even shuffling frames only increases I3D FVD by 3.6%
  - VideoMAE-v2 features are 5x more sensitive to temporal distortion
  - This explains why our 13-34% PPL advantage only shows as <1% I3D FVD

This script:
  1. Loads trained checkpoints (geo_k16 and evq_k16)
  2. Generates videos at multiple temperatures (0.9 original + 0.1 low-temp)
  3. Computes FVD using both I3D and VideoMAE-v2 features via cd-fvd
  4. Saves per-frame accuracy for each temperature

Usage:
    python scripts/video_temporal/eval_videomae_fvd.py \
        --data-dir data/video_temporal/generated/oscillating_mnist \
        --ckpt-dir results/supporting_video/oscillating_fvd/20260315_093550 \
        --work-dir results/supporting_video/videomae_fvd

Time estimate: ~1h on Blackwell 96GB (generation dominates).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_video_temporal import (  # noqa: E402
    DEVICE, DTYPE, USE_AUTOCAST, VideoGPT, set_seed,
    build_temporal_yarn_inv_freq,
)
from run_video_temporal_allocation_sweep import (  # noqa: E402
    MODEL_PROFILES,
    build_variant_inv_freqs,
    load_manifest,
    load_tokens,
    parse_variants,
)
from run_phase23_fvd_verify import load_checkpoint  # noqa: E402
from generate_and_eval_fvd import (  # noqa: E402
    generate_videos,
    decode_tokens_to_frames,
)


def apply_yarn(model: VideoGPT, train_frames: int, eval_frames: int) -> torch.Tensor:
    """Apply YaRN temporal scaling. Returns original inv_freq_t for restore."""
    rope = model.blocks[0].attn.rope
    orig = rope.inv_freq_t.clone()
    scale = eval_frames / train_frames
    if scale > 1.0:
        yarn_inv_t = build_temporal_yarn_inv_freq(orig, scale)
        rope.inv_freq_t.copy_(yarn_inv_t.to(rope.inv_freq_t.device))
        rope._build(eval_frames + 4)
    return orig


def restore_rope(model: VideoGPT, orig_inv_freq_t: torch.Tensor) -> None:
    rope = model.blocks[0].attn.rope
    rope.inv_freq_t.copy_(orig_inv_freq_t)


def frames_to_uint8_video(frames: torch.Tensor) -> np.ndarray:
    """Convert (N, T, H, W) float32 [0,1] grayscale to (N, T, H, W, 3) uint8 RGB.

    cd-fvd expects (N, T, H, W, C) uint8 format.
    We need to resize to at least 224x224 for VideoMAE.
    """
    N, T, H, W = frames.shape
    # Resize to 224x224 if needed
    if H < 224 or W < 224:
        frames = frames.reshape(N * T, 1, H, W)
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
        frames = frames.reshape(N, T, 224, 224)

    # Grayscale -> RGB, float -> uint8
    video = (frames.unsqueeze(-1).expand(-1, -1, -1, -1, 3) * 255).clamp(0, 255).byte().numpy()
    return video


def compute_videomae_fvd(
    real_videos: np.ndarray,
    fake_videos: np.ndarray,
) -> dict:
    """Compute FVD using both I3D and VideoMAE-v2 features via cd-fvd.

    Args:
        real_videos: (N, T, H, W, C) uint8
        fake_videos: (N, T, H, W, C) uint8

    Returns:
        dict with i3d_fvd and videomae_fvd
    """
    from cdfvd import fvd

    results = {}

    # VideoMAE FVD (temporal-sensitive)
    print("    Computing VideoMAE FVD...")
    t0 = time.time()
    try:
        evaluator = fvd.cdfvd(model='videomae', ckpt_path=None, device='cuda')
        vmae_fvd = evaluator.compute_fvd(real_videos, fake_videos)
        results["videomae_fvd"] = round(float(vmae_fvd), 4)
        print(f"    VideoMAE FVD = {vmae_fvd:.4f}  ({time.time()-t0:.1f}s)")
        del evaluator
    except Exception as e:
        print(f"    VideoMAE FVD failed: {e}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # I3D FVD (spatial-biased, for comparison)
    print("    Computing I3D FVD (cd-fvd)...")
    t0 = time.time()
    try:
        evaluator = fvd.cdfvd(model='i3d', ckpt_path=None, device='cuda')
        i3d_fvd = evaluator.compute_fvd(real_videos, fake_videos)
        results["i3d_fvd_cdfvd"] = round(float(i3d_fvd), 4)
        print(f"    I3D FVD (cd-fvd) = {i3d_fvd:.4f}  ({time.time()-t0:.1f}s)")
        del evaluator
    except Exception as e:
        print(f"    I3D FVD (cd-fvd) failed: {e}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


def compute_perframe_accuracy(
    real_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    train_frames: int,
    patches_per_frame: int,
) -> dict:
    """Per-frame token accuracy in extrapolation region."""
    context_len = train_frames * patches_per_frame
    total_len = min(real_tokens.shape[1], gen_tokens.shape[1])
    n = min(real_tokens.shape[0], gen_tokens.shape[0])

    real = real_tokens[:n, context_len:total_len]
    gen = gen_tokens[:n, context_len:total_len]

    matches = (real == gen).float().numpy()
    n_extrap_tokens = real.shape[1]
    n_extrap_frames = n_extrap_tokens // patches_per_frame

    per_frame = matches[:, :n_extrap_frames * patches_per_frame]
    per_frame = per_frame.reshape(n, n_extrap_frames, patches_per_frame)
    per_frame_acc = per_frame.mean(axis=(0, 2))

    mid = len(per_frame_acc) // 2
    return {
        "per_frame_acc": per_frame_acc.tolist(),
        "mean_acc": round(float(per_frame_acc.mean()), 4),
        "early_acc": round(float(per_frame_acc[:mid].mean()), 4) if mid > 0 else 0,
        "late_acc": round(float(per_frame_acc[mid:].mean()), 4) if mid > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VideoMAE-v2 FVD evaluation (temporal-sensitive)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default="blackwell96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=str, default="geo_k16,evq_k16")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--n-generate", type=int, default=256)
    parser.add_argument("--gen-batch-size", type=int, default=48)
    parser.add_argument("--yarn", action="store_true",
                        help="Apply YaRN temporal scaling during generation")
    # Temperatures to evaluate
    parser.add_argument("--temperatures", type=str, default="0.9,0.1",
                        help="Comma-separated temperatures to evaluate")
    parser.add_argument("--save-videos", action="store_true", default=True,
                        help="Save generated videos to disk for reuse")
    parser.add_argument("--reuse-videos", action="store_true",
                        help="Load previously saved videos instead of regenerating")
    args = parser.parse_args()

    temps = [float(t) for t in args.temperatures.split(",")]

    if args.work_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.work_dir = Path(f"results/supporting_video/videomae_fvd/{stamp}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.data_dir)
    train_frames = int(manifest["train_frames"])
    eval_frames = int(manifest.get("eval_frames", train_frames * 4))
    patches_per_frame = int(manifest["patches_per_frame"])
    image_size = int(manifest["image_size"])
    patch_size = int(manifest["patch_size"])
    grid_h = grid_w = image_size // patch_size

    variants = parse_variants(args.variants, train_frames=train_frames)
    profile = MODEL_PROFILES[args.profile].copy()
    cfg = {
        "vocab_size": int(manifest["vocab_size"]),
        "hidden_size": profile["hidden_size"],
        "num_layers": profile["num_layers"],
        "num_heads": profile["num_heads"],
        "head_dim": profile["head_dim"],
        "intermediate_size": profile["intermediate_size"],
        "max_T": profile["max_T"],
    }

    print(f"\n{'#' * 72}")
    print(f"  VideoMAE-v2 FVD EVALUATION")
    print(f"  variants: {[v['name'] for v in variants]}")
    print(f"  train={train_frames}f eval={eval_frames}f ppf={patches_per_frame}")
    print(f"  temperatures: {temps}")
    print(f"  n_generate={args.n_generate} yarn={args.yarn}")
    print(f"  device={DEVICE} dtype={DTYPE}")
    print(f"  work_dir={args.work_dir}")
    print(f"{'#' * 72}\n")

    # Load test data
    print("Loading test data...")
    test_tokens = load_tokens(args.data_dir, "test")
    print(f"  test: {test_tokens.shape}")

    # Prepare real videos (same for all variants)
    target_len = eval_frames * patches_per_frame
    n_real = min(args.n_generate, test_tokens.shape[0])
    real_tokens_subset = test_tokens[:n_real, :target_len]
    real_frames = decode_tokens_to_frames(real_tokens_subset, patch_size, grid_h, grid_w)
    # Subsample to 16 frames BEFORE resize to 224x224 to save memory
    # (256, 128, H, W) -> (256, 16, H, W) then resize
    T_real = real_frames.shape[1]
    if T_real > 16:
        sub_indices = np.linspace(0, T_real - 1, 16).astype(int)
        real_frames_sub = real_frames[:, sub_indices]
    else:
        real_frames_sub = real_frames
    real_uint8 = frames_to_uint8_video(real_frames_sub)
    del real_frames, real_frames_sub  # free memory
    print(f"  Real videos (16f subsample): {real_uint8.shape} ({real_uint8.dtype})")

    all_results = {}
    t0_total = time.time()

    for variant in variants:
        inv_h, inv_w, inv_t, split_info = build_variant_inv_freqs(
            head_dim=cfg["head_dim"],
            base=args.base,
            train_frames=train_frames,
            variant=variant,
        )
        run_id = f"{variant['name']}_seed{args.seed}"
        ckpt_path = args.ckpt_dir / f"{run_id}.pt"

        if not ckpt_path.exists():
            print(f"  SKIP {run_id}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {run_id}")
        print(f"{'=' * 60}")

        set_seed(args.seed)
        model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)
        load_checkpoint(model, ckpt_path)

        orig_inv_freq_t = None
        if args.yarn:
            orig_inv_freq_t = apply_yarn(model, train_frames, eval_frames)

        variant_result = {"variant": variant["name"], "yarn": args.yarn}

        for temp in temps:
            top_k = 50 if temp >= 0.5 else 0
            print(f"\n  --- temp={temp}, top_k={top_k} ---")

            # Check for cached generated tokens
            cache_path = args.work_dir / f"{run_id}_gen_tokens_t{temp}.npy"
            if cache_path.exists():
                print(f"  Loading cached gen tokens: {cache_path}")
                gen_tokens = torch.from_numpy(np.load(cache_path))
                gen_time = 0.0
            else:
                t0 = time.time()
                gen_tokens = generate_videos(
                    model=model,
                    context_tokens=test_tokens,
                    train_frames=train_frames,
                    target_frames=eval_frames,
                    patches_per_frame=patches_per_frame,
                    n_generate=args.n_generate,
                    temperature=temp,
                    top_k=top_k,
                    batch_size=args.gen_batch_size,
                    seed=args.seed,
                )
                gen_time = time.time() - t0
                # Save generated tokens to disk
                np.save(cache_path, gen_tokens.numpy())
                print(f"  Saved gen tokens: {cache_path}")

            print(f"  Generated {gen_tokens.shape[0]} videos in {gen_time:.1f}s")

            # Per-frame accuracy
            pf_acc = compute_perframe_accuracy(
                real_tokens_subset, gen_tokens[:n_real],
                train_frames, patches_per_frame,
            )
            print(f"  Token acc: mean={pf_acc['mean_acc']:.4f} "
                  f"early={pf_acc['early_acc']:.4f} late={pf_acc['late_acc']:.4f}")

            # Convert to uint8 video for FVD (subsample to 16f before resize)
            gen_frames = decode_tokens_to_frames(
                gen_tokens[:n_real, :target_len], patch_size, grid_h, grid_w
            )
            T_gen = gen_frames.shape[1]
            if T_gen > 16:
                gen_frames_sub = gen_frames[:, sub_indices]
            else:
                gen_frames_sub = gen_frames
            gen_uint8 = frames_to_uint8_video(gen_frames_sub)
            del gen_frames, gen_frames_sub
            print(f"  Gen videos (16f subsample): {gen_uint8.shape}")

            # Free GPU memory before loading VideoMAE model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Compute FVD with both I3D and VideoMAE
            fvd_results = compute_videomae_fvd(real_uint8, gen_uint8)

            temp_key = f"temp_{temp}"
            variant_result[temp_key] = {
                "temperature": temp,
                "top_k": top_k,
                "generation_time_sec": round(gen_time, 1),
                **pf_acc,
                **fvd_results,
            }

            del gen_tokens, gen_uint8
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        all_results[run_id] = variant_result

        # Save per-variant result
        (args.work_dir / f"{run_id}_videomae.json").write_text(
            json.dumps(variant_result, indent=2)
        )

        if orig_inv_freq_t is not None:
            restore_rope(model, orig_inv_freq_t)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - t0_total

    # === Comparison summary ===
    print(f"\n\n{'=' * 72}")
    print(f"  VideoMAE FVD COMPARISON")
    print(f"{'=' * 72}")

    variant_names = [v["name"] for v in variants]
    if len(variant_names) >= 2:
        r1_key = f"{variant_names[0]}_seed{args.seed}"
        r2_key = f"{variant_names[1]}_seed{args.seed}"
        r1 = all_results.get(r1_key, {})
        r2 = all_results.get(r2_key, {})

        for temp in temps:
            tk = f"temp_{temp}"
            if tk in r1 and tk in r2:
                t1, t2 = r1[tk], r2[tk]
                print(f"\n  Temperature = {temp}:")
                print(f"    {'Metric':<25} {variant_names[0]:>12} {variant_names[1]:>12} {'Delta':>10} {'Winner':>8}")
                print(f"    {'-'*67}")

                for metric in ["videomae_fvd", "i3d_fvd_cdfvd", "mean_acc", "early_acc", "late_acc"]:
                    v1 = t1.get(metric)
                    v2 = t2.get(metric)
                    if v1 is not None and v2 is not None:
                        if "fvd" in metric:
                            delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
                            winner = variant_names[1] if v2 < v1 else variant_names[0]
                        else:
                            delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
                            winner = variant_names[1] if v2 > v1 else variant_names[0]
                        print(f"    {metric:<25} {v1:>12.4f} {v2:>12.4f} {delta:>+9.2f}% {winner:>8}")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    summary = {
        "results": all_results,
        "config": {
            "data_dir": str(args.data_dir),
            "ckpt_dir": str(args.ckpt_dir),
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "temperatures": temps,
            "n_generate": args.n_generate,
            "yarn": args.yarn,
        },
        "total_time_sec": round(total_time, 1),
    }
    summary_path = args.work_dir / "videomae_fvd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
