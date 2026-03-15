#!/usr/bin/env python3
"""Per-frame top-1 accuracy + low-temperature generation evaluation.

Approach B: Teacher-forced top-1 accuracy per frame position.
  - No sampling, just argmax vs ground truth per position.
  - Shows where EVQ predicts better (especially in extrapolation zone).

Approach E: Low-temperature (0.1) generation + per-frame token accuracy.
  - Near-greedy decoding amplifies distribution differences.
  - Per-frame accuracy curve shows where the gap manifests.

Outputs:
  - perframe_top1.json: per-frame top-1 accuracy (Approach B)
  - perframe_lowtemp.json: per-frame token accuracy at temp=0.1 (Approach E)
  - Combined plot data for paper figure.

Usage:
    python scripts/video_temporal/eval_perframe_accuracy.py \
        --data-dir data/video_temporal/generated/oscillating_mnist \
        --ckpt-dir results/supporting_video/oscillating_fvd/20260315_093550 \
        --work-dir results/supporting_video/perframe_eval
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
    VARIANT_SPECS,
    build_variant_inv_freqs,
    load_manifest,
    load_tokens,
    parse_variants,
)
from run_phase23_fvd_verify import load_checkpoint  # noqa: E402
from generate_and_eval_fvd import generate_videos  # noqa: E402
from contextlib import nullcontext


# ---------------------------------------------------------------------------
# YaRN helper: apply temporal frequency scaling for extrapolation
# ---------------------------------------------------------------------------

def apply_yarn_to_model(model: VideoGPT, train_frames: int, eval_frames: int) -> torch.Tensor:
    """Apply YaRN temporal frequency scaling. Returns original inv_freq_t for restore."""
    rope = model.blocks[0].attn.rope
    orig_inv_freq_t = rope.inv_freq_t.clone()
    scale = eval_frames / train_frames
    if scale > 1.0:
        yarn_inv_t = build_temporal_yarn_inv_freq(orig_inv_freq_t, scale)
        rope.inv_freq_t.copy_(yarn_inv_t.to(rope.inv_freq_t.device))
        rope._build(eval_frames + 4)
        print(f"    [YaRN] Applied temporal scaling: {scale:.1f}x")
    return orig_inv_freq_t


def restore_yarn(model: VideoGPT, orig_inv_freq_t: torch.Tensor) -> None:
    """Restore original temporal frequencies after YaRN."""
    rope = model.blocks[0].attn.rope
    rope.inv_freq_t.copy_(orig_inv_freq_t)


# ---------------------------------------------------------------------------
# Approach B: Teacher-forced top-1 accuracy per position
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_forced_accuracy(
    model: VideoGPT,
    test_tokens: torch.Tensor,
    eval_frames: int,
    patches_per_frame: int,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute per-position top-1 accuracy in teacher-forced mode.

    For each position t, checks if argmax(logits[t]) == ground_truth[t+1].
    This directly measures the model's prediction quality without sampling noise.

    Args:
        model: trained VideoGPT
        test_tokens: (N, total_tokens) ground truth token sequences
        eval_frames: total number of frames to evaluate
        patches_per_frame: tokens per frame
        batch_size: processing batch size

    Returns:
        per_position_accuracy: (total_tokens - 1,) float array
    """
    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    total_tokens = eval_frames * patches_per_frame
    total_tokens = min(total_tokens, test_tokens.shape[1])
    N = test_tokens.shape[0]

    # Accumulate correct counts per position
    correct = np.zeros(total_tokens - 1, dtype=np.float64)
    count = np.zeros(total_tokens - 1, dtype=np.float64)

    # Extend RoPE for full sequence
    model.extend_rope(total_tokens + 1)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = test_tokens[start:end, :total_tokens].to(DEVICE)

        with ctx:
            logits = model(batch)  # (B, total_tokens, vocab)

        # Top-1 predictions: logits[:, t, :] predicts token at t+1
        preds = logits[:, :-1, :].argmax(dim=-1).cpu()  # (B, total_tokens-1)
        targets = batch[:, 1:].cpu()  # (B, total_tokens-1)

        matches = (preds == targets).float().numpy()
        correct += matches.sum(axis=0)
        count += matches.shape[0]

        del batch, logits, preds, targets
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (start // batch_size) % 10 == 0:
            print(f"    teacher-forced: {end}/{N} videos processed")

    per_pos_acc = correct / np.maximum(count, 1)
    return per_pos_acc


def aggregate_to_per_frame(
    per_pos_acc: np.ndarray,
    patches_per_frame: int,
) -> np.ndarray:
    """Average per-position accuracy into per-frame accuracy."""
    n_tokens = len(per_pos_acc)
    n_frames = n_tokens // patches_per_frame
    # Trim to exact frame boundary
    trimmed = per_pos_acc[:n_frames * patches_per_frame]
    per_frame = trimmed.reshape(n_frames, patches_per_frame).mean(axis=1)
    return per_frame


# ---------------------------------------------------------------------------
# Approach E: Low-temperature generation + per-frame accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_lowtemp_perframe_accuracy(
    model: VideoGPT,
    test_tokens: torch.Tensor,
    train_frames: int,
    eval_frames: int,
    patches_per_frame: int,
    n_generate: int = 256,
    temperature: float = 0.1,
    top_k: int = 0,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """Generate at low temperature, compute per-frame token accuracy.

    Returns dict with per-frame accuracy arrays and aggregate stats.
    """
    # Generate videos at low temperature
    print(f"    Generating {n_generate} videos at temp={temperature}, top_k={top_k}...")
    t0 = time.time()
    gen_tokens = generate_videos(
        model=model,
        context_tokens=test_tokens,
        train_frames=train_frames,
        target_frames=eval_frames,
        patches_per_frame=patches_per_frame,
        n_generate=n_generate,
        temperature=temperature,
        top_k=top_k,
        batch_size=batch_size,
        seed=seed,
    )
    gen_time = time.time() - t0
    print(f"    Generation done in {gen_time:.1f}s")

    # Compare against ground truth
    target_len = eval_frames * patches_per_frame
    n_compare = min(n_generate, test_tokens.shape[0])
    real = test_tokens[:n_compare, :target_len]
    gen = gen_tokens[:n_compare, :target_len]

    # Per-position accuracy (extrapolation portion only)
    context_len = train_frames * patches_per_frame
    extrap_real = real[:, context_len:]
    extrap_gen = gen[:, context_len:]

    matches = (extrap_real == extrap_gen).float().numpy()  # (N, extrap_tokens)

    # Per-frame accuracy
    n_extrap_tokens = extrap_real.shape[1]
    n_extrap_frames = n_extrap_tokens // patches_per_frame
    per_frame_matches = matches[:, :n_extrap_frames * patches_per_frame]
    per_frame_matches = per_frame_matches.reshape(n_compare, n_extrap_frames, patches_per_frame)
    per_frame_acc = per_frame_matches.mean(axis=(0, 2))  # (n_extrap_frames,)

    # Also compute per-frame accuracy including context (should be 100%)
    full_matches = (real == gen).float().numpy()
    n_total_frames = target_len // patches_per_frame
    full_per_frame = full_matches[:, :n_total_frames * patches_per_frame]
    full_per_frame = full_per_frame.reshape(n_compare, n_total_frames, patches_per_frame)
    full_per_frame_acc = full_per_frame.mean(axis=(0, 2))

    return {
        "extrap_per_frame_acc": per_frame_acc.tolist(),
        "full_per_frame_acc": full_per_frame_acc.tolist(),
        "extrap_mean_acc": float(per_frame_acc.mean()),
        "generation_time_sec": round(gen_time, 1),
        "temperature": temperature,
        "top_k": top_k,
        "n_generate": n_generate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame accuracy evaluation (Approach B + E)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help="Directory with geo_k16_seed42.pt and evq_k16_seed42.pt")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default="blackwell96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=str, default="geo_k16,evq_k16")
    parser.add_argument("--base", type=float, default=10000.0)
    # Approach B options
    parser.add_argument("--tf-batch-size", type=int, default=8,
                        help="Batch size for teacher-forced evaluation")
    # Approach E options
    parser.add_argument("--n-generate", type=int, default=256)
    parser.add_argument("--gen-batch-size", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k for low-temp generation (0=no filtering, greedy-ish)")
    # YaRN
    parser.add_argument("--yarn", action="store_true",
                        help="Apply YaRN temporal scaling during evaluation/generation")
    # Control
    parser.add_argument("--skip-approach-b", action="store_true")
    parser.add_argument("--skip-approach-e", action="store_true")
    args = parser.parse_args()

    if args.work_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.work_dir = Path(f"results/supporting_video/perframe_eval/{stamp}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
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
    print(f"  PER-FRAME ACCURACY EVALUATION")
    print(f"  variants: {[v['name'] for v in variants]}")
    print(f"  train_frames={train_frames}  eval_frames={eval_frames}")
    print(f"  patches_per_frame={patches_per_frame}")
    print(f"  device={DEVICE} dtype={DTYPE} yarn={args.yarn}")
    print(f"  temp={args.temperature} top_k={args.top_k} n_gen={args.n_generate}")
    print(f"  work_dir={args.work_dir}")
    print(f"{'#' * 72}\n")

    # Load test data
    print("Loading test data...")
    test_tokens = load_tokens(args.data_dir, "test")
    print(f"  test: {test_tokens.shape}")

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

        # Build and load model
        set_seed(args.seed)
        model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)
        load_checkpoint(model, ckpt_path)
        print(f"  Loaded checkpoint: {ckpt_path}")

        # Apply YaRN if requested
        orig_inv_freq_t = None
        if args.yarn:
            orig_inv_freq_t = apply_yarn_to_model(model, train_frames, eval_frames)

        result = {"variant": variant["name"], "seed": args.seed, "yarn": args.yarn}

        # --- Approach B: Teacher-forced top-1 accuracy ---
        if not args.skip_approach_b:
            print(f"\n  [Approach B] Teacher-forced top-1 accuracy...")
            t0 = time.time()
            per_pos_acc = compute_teacher_forced_accuracy(
                model, test_tokens, eval_frames, patches_per_frame,
                batch_size=args.tf_batch_size,
            )
            tf_time = time.time() - t0
            print(f"  Teacher-forced done in {tf_time:.1f}s")

            per_frame_acc = aggregate_to_per_frame(per_pos_acc, patches_per_frame)

            # Compute stats for different regions
            context_frames = train_frames - 1  # first frame has no prediction
            extrap_start = context_frames
            extrap_acc = per_frame_acc[extrap_start:] if len(per_frame_acc) > extrap_start else []

            # Split extrapolation into early (1-2x) and late (3-4x)
            mid_point = len(extrap_acc) // 2 if len(extrap_acc) > 0 else 0
            early_acc = extrap_acc[:mid_point] if mid_point > 0 else []
            late_acc = extrap_acc[mid_point:] if mid_point > 0 else []

            result["approach_b"] = {
                "per_frame_acc": per_frame_acc.tolist(),
                "context_mean_acc": float(per_frame_acc[:context_frames].mean()) if context_frames > 0 else 0,
                "extrap_mean_acc": float(np.mean(extrap_acc)) if len(extrap_acc) > 0 else 0,
                "early_extrap_acc": float(np.mean(early_acc)) if len(early_acc) > 0 else 0,
                "late_extrap_acc": float(np.mean(late_acc)) if len(late_acc) > 0 else 0,
                "time_sec": round(tf_time, 1),
            }

            print(f"    Context acc: {result['approach_b']['context_mean_acc']:.4f}")
            print(f"    Extrap acc:  {result['approach_b']['extrap_mean_acc']:.4f}")
            print(f"    Early extrap (frame {train_frames}-{train_frames + mid_point}): "
                  f"{result['approach_b']['early_extrap_acc']:.4f}")
            print(f"    Late extrap  (frame {train_frames + mid_point}-{eval_frames}): "
                  f"{result['approach_b']['late_extrap_acc']:.4f}")

        # --- Approach E: Low-temperature generation ---
        if not args.skip_approach_e:
            print(f"\n  [Approach E] Low-temp generation (temp={args.temperature})...")
            lowtemp_result = compute_lowtemp_perframe_accuracy(
                model=model,
                test_tokens=test_tokens,
                train_frames=train_frames,
                eval_frames=eval_frames,
                patches_per_frame=patches_per_frame,
                n_generate=args.n_generate,
                temperature=args.temperature,
                top_k=args.top_k,
                batch_size=args.gen_batch_size,
                seed=args.seed,
            )
            result["approach_e"] = lowtemp_result

            extrap_acc = lowtemp_result["extrap_per_frame_acc"]
            mid = len(extrap_acc) // 2
            early = extrap_acc[:mid] if mid > 0 else []
            late = extrap_acc[mid:] if mid > 0 else []
            print(f"    Extrap mean acc: {lowtemp_result['extrap_mean_acc']:.4f}")
            print(f"    Early extrap: {float(np.mean(early)):.4f}" if early else "")
            print(f"    Late extrap:  {float(np.mean(late)):.4f}" if late else "")

        all_results[run_id] = result

        # Save per-variant result
        (args.work_dir / f"{run_id}_perframe.json").write_text(
            json.dumps(result, indent=2)
        )

        # Restore and cleanup
        if orig_inv_freq_t is not None:
            restore_yarn(model, orig_inv_freq_t)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - t0_total

    # --- Comparison summary ---
    print(f"\n\n{'=' * 72}")
    print(f"  PER-FRAME ACCURACY COMPARISON")
    print(f"{'=' * 72}")

    variant_names = [v["name"] for v in variants]
    if len(variant_names) >= 2:
        r1_key = f"{variant_names[0]}_seed{args.seed}"
        r2_key = f"{variant_names[1]}_seed{args.seed}"
        r1 = all_results.get(r1_key, {})
        r2 = all_results.get(r2_key, {})

        if "approach_b" in r1 and "approach_b" in r2:
            b1 = r1["approach_b"]
            b2 = r2["approach_b"]
            print(f"\n  Approach B (Teacher-forced top-1):")
            print(f"    {'Metric':<30} {variant_names[0]:>12} {variant_names[1]:>12} {'Delta':>10}")
            print(f"    {'-'*64}")
            for key in ["context_mean_acc", "extrap_mean_acc", "early_extrap_acc", "late_extrap_acc"]:
                v1, v2 = b1[key], b2[key]
                delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
                winner = "**" if abs(delta) > 0.5 else ""
                label = key.replace("_", " ").title()
                print(f"    {label:<30} {v1:>12.4f} {v2:>12.4f} {delta:>+9.2f}% {winner}")

        if "approach_e" in r1 and "approach_e" in r2:
            e1 = r1["approach_e"]
            e2 = r2["approach_e"]
            print(f"\n  Approach E (Low-temp generation, temp={args.temperature}):")
            v1, v2 = e1["extrap_mean_acc"], e2["extrap_mean_acc"]
            delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
            print(f"    Extrap mean acc: {variant_names[0]}={v1:.4f}  "
                  f"{variant_names[1]}={v2:.4f}  delta={delta:+.2f}%")

            # Per-frame comparison for early vs late
            acc1 = e1["extrap_per_frame_acc"]
            acc2 = e2["extrap_per_frame_acc"]
            mid = len(acc1) // 2
            if mid > 0:
                early1, early2 = np.mean(acc1[:mid]), np.mean(acc2[:mid])
                late1, late2 = np.mean(acc1[mid:]), np.mean(acc2[mid:])
                d_early = (early2 - early1) / max(abs(early1), 1e-9) * 100
                d_late = (late2 - late1) / max(abs(late1), 1e-9) * 100
                print(f"    Early extrap:   {variant_names[0]}={early1:.4f}  "
                      f"{variant_names[1]}={early2:.4f}  delta={d_early:+.2f}%")
                print(f"    Late extrap:    {variant_names[0]}={late1:.4f}  "
                      f"{variant_names[1]}={late2:.4f}  delta={d_late:+.2f}%")

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save combined results
    summary = {
        "results": all_results,
        "config": {
            "data_dir": str(args.data_dir),
            "ckpt_dir": str(args.ckpt_dir),
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "patches_per_frame": patches_per_frame,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "n_generate": args.n_generate,
        },
        "total_time_sec": round(total_time, 1),
    }
    summary_path = args.work_dir / "perframe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
