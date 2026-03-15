#!/usr/bin/env python3
"""Temporal Precision Evaluation: teacher-forced per-frame accuracy.

Directly measures how precisely each RoPE variant tracks temporal position
during extrapolation, without autoregressive error accumulation or FVD
aggregation compressing the signal.

Decomposition dimensions:
  1. Frame regions: train(0-32f), near(32-64f), mid(64-96f), far(96-128f)
  2. Per-frame accuracy curve (every frame + 4-frame binned for plotting)
  3. Top-1 and Top-5 accuracy (top-5 captures near-miss predictions)
  4. YaRN vs no-YaRN modes
  5. Frequency decomposition: per-period (P=16/24/32) accuracy via spatial FFT

Usage:
    python scripts/video_temporal/eval_temporal_precision.py \
        --data-dir data/video_temporal/generated/oscillating_mnist \
        --ckpt-dir checkpoints/oscillating_fvd \
        --work-dir results/supporting_video/temporal_precision

    # Both YaRN and no-YaRN:
    python scripts/video_temporal/eval_temporal_precision.py \
        --data-dir ... --ckpt-dir ... --both-modes

    # Frequency decomposition (YaRN only):
    python scripts/video_temporal/eval_temporal_precision.py \
        --data-dir ... --ckpt-dir ... --freq-decompose

    # Quick test with fewer videos:
    python scripts/video_temporal/eval_temporal_precision.py \
        --data-dir ... --ckpt-dir ... --n-videos 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
from contextlib import nullcontext


# ---------------------------------------------------------------------------
# YaRN helpers (same as eval_perframe_accuracy.py)
# ---------------------------------------------------------------------------

def apply_yarn_to_model(model: VideoGPT, train_frames: int, eval_frames: int) -> torch.Tensor:
    """Apply YaRN temporal scaling. Returns original inv_freq_t for restore."""
    rope = model.blocks[0].attn.rope
    orig = rope.inv_freq_t.clone()
    scale = eval_frames / train_frames
    if scale > 1.0:
        yarn = build_temporal_yarn_inv_freq(orig, scale)
        rope.inv_freq_t.copy_(yarn.to(rope.inv_freq_t.device))
        rope._build(eval_frames + 4)
    return orig


def restore_yarn(model: VideoGPT, orig: torch.Tensor) -> None:
    model.blocks[0].attn.rope.inv_freq_t.copy_(orig)


# ---------------------------------------------------------------------------
# Core evaluation: teacher-forced top-1 and top-5 per position
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perframe_precision(
    model: VideoGPT,
    test_tokens: torch.Tensor,
    eval_frames: int,
    patches_per_frame: int,
    batch_size: int = 8,
    n_videos: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced per-position top-1 and top-5 accuracy.

    Returns (top1_per_pos, top5_per_pos), each shape (total_tokens - 1,).
    """
    model.eval()
    ctx = (torch.amp.autocast("cuda", dtype=DTYPE)
           if USE_AUTOCAST else nullcontext())

    total_tokens = min(eval_frames * patches_per_frame, test_tokens.shape[1])
    N = test_tokens.shape[0] if n_videos is None else min(n_videos, test_tokens.shape[0])

    correct_top1 = np.zeros(total_tokens - 1, dtype=np.float64)
    correct_top5 = np.zeros(total_tokens - 1, dtype=np.float64)
    count = np.zeros(total_tokens - 1, dtype=np.float64)

    model.extend_rope(total_tokens + 1)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = test_tokens[start:end, :total_tokens].to(DEVICE)

        with ctx:
            logits = model(batch)  # (B, L, V)

        targets = batch[:, 1:].cpu()

        # Top-1
        preds = logits[:, :-1, :].argmax(dim=-1).cpu()
        top1_match = (preds == targets).float().numpy()
        correct_top1 += top1_match.sum(axis=0)

        # Top-5
        topk = logits[:, :-1, :].topk(5, dim=-1).indices.cpu()
        top5_match = (topk == targets.unsqueeze(-1)).any(dim=-1).float().numpy()
        correct_top5 += top5_match.sum(axis=0)

        count += top1_match.shape[0]

        del batch, logits, preds, targets, topk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (start // batch_size) % 10 == 0:
            print(f"    {end}/{N} videos")

    return (
        correct_top1 / np.maximum(count, 1),
        correct_top5 / np.maximum(count, 1),
    )


# ---------------------------------------------------------------------------
# Frequency decomposition: classify patches by dominant oscillation period
# ---------------------------------------------------------------------------

def classify_patches_by_period(
    test_tokens: torch.Tensor,
    eval_frames: int,
    ppf: int,
    periods: list[int],
    n_videos: int | None = None,
    var_threshold: float = 0.5,
) -> np.ndarray:
    """Classify each (video, spatial_position) by dominant oscillation period.

    Uses FFT of each patch's temporal trace to find the dominant frequency,
    then assigns to the nearest target period.

    Returns: (N, ppf) int array, values from periods or -1 (background).
    """
    N = test_tokens.shape[0] if n_videos is None else min(n_videos, test_tokens.shape[0])
    tokens_np = test_tokens[:N].numpy().reshape(N, eval_frames, ppf).astype(np.float32)

    patch_periods = np.full((N, ppf), -1, dtype=int)

    for p in range(ppf):
        traces = tokens_np[:, :, p]  # (N, eval_frames)
        traces = traces - traces.mean(axis=1, keepdims=True)  # detrend
        variances = traces.var(axis=1)  # (N,)
        active = variances > var_threshold

        if not active.any():
            continue

        spectra = np.abs(np.fft.rfft(traces[active], axis=1)) ** 2

        # Power at each target period
        period_powers = np.zeros((active.sum(), len(periods)))
        for j, P in enumerate(periods):
            k = eval_frames / P
            k_int = int(round(k))
            k_lo = max(1, k_int - 1)
            k_hi = min(eval_frames // 2 + 1, k_int + 2)
            period_powers[:, j] = spectra[:, k_lo:k_hi].sum(axis=1)

        dominant = np.argmax(period_powers, axis=1)
        active_indices = np.where(active)[0]
        for idx, dom in zip(active_indices, dominant):
            patch_periods[idx, p] = periods[dom]

    return patch_periods


@torch.no_grad()
def compute_pervideo_predictions(
    model: VideoGPT,
    test_tokens: torch.Tensor,
    eval_frames: int,
    ppf: int,
    batch_size: int = 8,
    n_videos: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced evaluation returning per-video, per-position correctness.

    Returns (top1_correct, top5_correct), each bool array of shape (N, L-1).
    """
    model.eval()
    ctx = (torch.amp.autocast("cuda", dtype=DTYPE)
           if USE_AUTOCAST else nullcontext())

    total_tokens = min(eval_frames * ppf, test_tokens.shape[1])
    N = test_tokens.shape[0] if n_videos is None else min(n_videos, test_tokens.shape[0])

    top1_correct = np.zeros((N, total_tokens - 1), dtype=np.bool_)
    top5_correct = np.zeros((N, total_tokens - 1), dtype=np.bool_)

    model.extend_rope(total_tokens + 1)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = test_tokens[start:end, :total_tokens].to(DEVICE)

        with ctx:
            logits = model(batch)

        targets = batch[:, 1:].cpu()
        preds = logits[:, :-1, :].argmax(dim=-1).cpu()
        top1_correct[start:end] = (preds == targets).numpy()

        topk = logits[:, :-1, :].topk(5, dim=-1).indices.cpu()
        top5_correct[start:end] = (topk == targets.unsqueeze(-1)).any(dim=-1).numpy()

        del batch, logits, preds, targets, topk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (start // batch_size) % 10 == 0:
            print(f"    {end}/{N} videos")

    return top1_correct, top5_correct


def decompose_by_frequency(
    top1_correct: np.ndarray,
    top5_correct: np.ndarray,
    patch_periods: np.ndarray,
    ppf: int,
    train_frames: int,
    periods: list[int],
) -> dict:
    """Decompose accuracy by frequency band and region.

    For each period P, selects only positions belonging to patches classified
    as period P, then computes per-frame accuracy and region stats.
    """
    N, total_pos = top1_correct.shape
    n_frames = total_pos // ppf

    # Reshape to (N, n_frames, ppf)
    top1 = top1_correct[:, :n_frames * ppf].reshape(N, n_frames, ppf)
    top5 = top5_correct[:, :n_frames * ppf].reshape(N, n_frames, ppf)

    results = {}
    for P in periods:
        mask = (patch_periods == P)  # (N, ppf)
        n_patches_total = mask.sum()

        per_frame_top1 = np.zeros(n_frames)
        per_frame_top5 = np.zeros(n_frames)

        for f in range(n_frames):
            selected_top1 = top1[:, f, :][mask]
            selected_top5 = top5[:, f, :][mask]
            if len(selected_top1) > 0:
                per_frame_top1[f] = selected_top1.mean()
                per_frame_top5[f] = selected_top5.mean()

        results[P] = {
            "n_patches": int(n_patches_total),
            "top1_regions": region_stats(per_frame_top1, train_frames),
            "top5_regions": region_stats(per_frame_top5, train_frames),
        }

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def to_per_frame(per_pos: np.ndarray, ppf: int) -> np.ndarray:
    """Average per-position accuracy into per-frame accuracy."""
    n = len(per_pos) // ppf
    return per_pos[:n * ppf].reshape(n, ppf).mean(axis=1)


def bin_every_k(per_frame: np.ndarray, k: int = 4) -> np.ndarray:
    """Bin per-frame accuracy into k-frame groups."""
    n = len(per_frame) // k * k
    return per_frame[:n].reshape(-1, k).mean(axis=1)


def region_stats(per_frame: np.ndarray, train_frames: int) -> dict:
    """Accuracy for train / near / mid / far extrapolation regions."""
    # Frame 0 has no prediction target, so train region is frames 0..(tf-2)
    tf = train_frames - 1
    total = len(per_frame)
    regions = {}

    if tf > 0:
        regions["train"] = float(per_frame[:tf].mean())

    # Near: 1x-2x training length
    ne = min(2 * train_frames - 1, total)
    if ne > tf:
        regions["near_extrap"] = float(per_frame[tf:ne].mean())

    # Mid: 2x-3x
    me = min(3 * train_frames - 1, total)
    if me > ne:
        regions["mid_extrap"] = float(per_frame[ne:me].mean())

    # Far: 3x-4x
    if total > me:
        regions["far_extrap"] = float(per_frame[me:].mean())

    # All extrapolation
    if total > tf:
        regions["all_extrap"] = float(per_frame[tf:].mean())

    return regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal precision evaluation (teacher-forced)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default="blackwell96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=str, default="geo_k16,evq_k16")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-videos", type=int, default=None,
                        help="Number of test videos (default: all 2000)")
    parser.add_argument("--no-yarn", action="store_true",
                        help="Run WITHOUT YaRN temporal scaling")
    parser.add_argument("--both-modes", action="store_true",
                        help="Run both YaRN and no-YaRN")
    parser.add_argument("--freq-decompose", action="store_true",
                        help="Frequency decomposition: per-period accuracy (YaRN only)")
    args = parser.parse_args()

    if args.work_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.work_dir = Path(f"results/supporting_video/temporal_precision/{stamp}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and config
    manifest = load_manifest(args.data_dir)
    train_frames = int(manifest["train_frames"])
    eval_frames = int(manifest.get("eval_frames", train_frames * 4))
    ppf = int(manifest["patches_per_frame"])
    image_size = int(manifest["image_size"])
    patch_size = int(manifest["patch_size"])
    grid_h = grid_w = image_size // patch_size
    periods = manifest.get("oscillation_periods", [16, 24, 32])

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

    # Determine modes
    if args.both_modes:
        yarn_modes = [("yarn", True), ("noyarn", False)]
    elif args.no_yarn:
        yarn_modes = [("noyarn", False)]
    else:
        yarn_modes = [("yarn", True)]

    print(f"\n{'#' * 72}")
    print(f"  TEMPORAL PRECISION EVALUATION")
    print(f"  variants: {[v['name'] for v in variants]}")
    print(f"  train={train_frames}f  eval={eval_frames}f  ppf={ppf}")
    print(f"  oscillation periods={periods}")
    print(f"  modes: {[m[0] for m in yarn_modes]}")
    print(f"  device={DEVICE}  n_videos={args.n_videos or 'all'}")
    print(f"  work_dir={args.work_dir}")
    print(f"{'#' * 72}\n")

    # Load test data
    print("Loading test data...")
    test_tokens = load_tokens(args.data_dir, "test")
    print(f"  test: {test_tokens.shape}")

    all_results = {}
    t0_total = time.time()

    for mode_tag, use_yarn in yarn_modes:
        print(f"\n{'=' * 60}")
        print(f"  MODE: {mode_tag}")
        print(f"{'=' * 60}")

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
                print(f"  SKIP {run_id}: {ckpt_path} not found")
                continue

            result_key = f"{run_id}_{mode_tag}"
            print(f"\n  --- {result_key} ---")

            set_seed(args.seed)
            model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)
            load_checkpoint(model, ckpt_path)
            print(f"    Loaded: {ckpt_path}")

            orig = None
            if use_yarn:
                orig = apply_yarn_to_model(model, train_frames, eval_frames)
                print(f"    YaRN: {eval_frames / train_frames:.0f}x")

            t0 = time.time()
            top1_pos, top5_pos = compute_perframe_precision(
                model, test_tokens, eval_frames, ppf,
                batch_size=args.batch_size,
                n_videos=args.n_videos,
            )
            elapsed = time.time() - t0

            # Per-frame aggregation
            top1_frame = to_per_frame(top1_pos, ppf)
            top5_frame = to_per_frame(top5_pos, ppf)

            # Binned (4-frame groups for plotting)
            top1_binned = bin_every_k(top1_frame, 4)
            top5_binned = bin_every_k(top5_frame, 4)

            # Region stats
            top1_reg = region_stats(top1_frame, train_frames)
            top5_reg = region_stats(top5_frame, train_frames)

            result = {
                "variant": variant["name"],
                "seed": args.seed,
                "yarn": use_yarn,
                "mode": mode_tag,
                "eval_frames": eval_frames,
                "train_frames": train_frames,
                "n_videos": args.n_videos or int(test_tokens.shape[0]),
                "time_sec": round(elapsed, 1),
                "split": split_info,
                "top1": {
                    "per_frame": top1_frame.tolist(),
                    "binned_4f": top1_binned.tolist(),
                    "regions": top1_reg,
                },
                "top5": {
                    "per_frame": top5_frame.tolist(),
                    "binned_4f": top5_binned.tolist(),
                    "regions": top5_reg,
                },
            }

            all_results[result_key] = result

            # Save per-variant
            out_path = args.work_dir / f"{result_key}_precision.json"
            out_path.write_text(json.dumps(result, indent=2))

            print(f"    Time: {elapsed:.1f}s")
            print(f"    Top-1: {json.dumps({k: round(v, 4) for k, v in top1_reg.items()})}")
            print(f"    Top-5: {json.dumps({k: round(v, 4) for k, v in top5_reg.items()})}")

            if orig is not None:
                restore_yarn(model, orig)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    total_time = time.time() - t0_total

    # --- Comparison table ---
    print(f"\n\n{'=' * 72}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 72}")

    variant_names = [v["name"] for v in variants]
    for mode_tag, _ in yarn_modes:
        keys = [f"{v}_seed{args.seed}_{mode_tag}" for v in variant_names]
        pair = [all_results.get(k) for k in keys]
        if not all(pair):
            continue

        r1, r2 = pair
        print(f"\n  [{mode_tag.upper()}] {variant_names[0]} vs {variant_names[1]}")
        print(f"  {'Metric':<30} {variant_names[0]:>10} {variant_names[1]:>10} "
              f"{'Delta':>10} {'Winner':>8}")
        print(f"  {'-' * 70}")

        for metric_label, metric_key in [("Top-1", "top1"), ("Top-5", "top5")]:
            for region in ["train", "near_extrap", "mid_extrap",
                           "far_extrap", "all_extrap"]:
                v1 = r1[metric_key]["regions"].get(region)
                v2 = r2[metric_key]["regions"].get(region)
                if v1 is None or v2 is None:
                    continue
                delta_pct = (v2 - v1) / max(abs(v1), 1e-9) * 100
                delta_abs = v2 - v1
                winner = (variant_names[1] if v2 > v1 + 1e-6
                          else variant_names[0] if v1 > v2 + 1e-6
                          else "tie")
                label = f"{metric_label} {region.replace('_', ' ')}"
                print(f"  {label:<30} {v1:>10.4f} {v2:>10.4f} "
                      f"{delta_pct:>+9.2f}% {winner:>8}")

    # --- Frequency decomposition ---
    freq_results = {}
    if args.freq_decompose:
        print(f"\n\n{'=' * 72}")
        print(f"  FREQUENCY DECOMPOSITION (per-period accuracy)")
        print(f"{'=' * 72}")

        n_eval = args.n_videos or int(test_tokens.shape[0])
        print(f"\n  Classifying {n_eval} x {ppf} patches by dominant period...")
        t0_class = time.time()
        patch_periods_arr = classify_patches_by_period(
            test_tokens, eval_frames, ppf, periods,
            n_videos=args.n_videos,
        )
        class_time = time.time() - t0_class
        print(f"  Classification done in {class_time:.1f}s")

        for P in periods:
            n_p = (patch_periods_arr == P).sum()
            print(f"    P={P}: {n_p} patches ({n_p / patch_periods_arr.size * 100:.1f}%)")
        n_bg = (patch_periods_arr == -1).sum()
        print(f"    bg: {n_bg} patches ({n_bg / patch_periods_arr.size * 100:.1f}%)")

        # Run per-video evaluation for each variant (YaRN mode only)
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
                continue

            result_key = f"{run_id}_freq"
            print(f"\n  --- {result_key} ---")

            set_seed(args.seed)
            model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)
            load_checkpoint(model, ckpt_path)
            orig = apply_yarn_to_model(model, train_frames, eval_frames)

            t0 = time.time()
            top1_c, top5_c = compute_pervideo_predictions(
                model, test_tokens, eval_frames, ppf,
                batch_size=args.batch_size,
                n_videos=args.n_videos,
            )
            elapsed = time.time() - t0

            freq_result = decompose_by_frequency(
                top1_c, top5_c, patch_periods_arr, ppf, train_frames, periods,
            )
            freq_results[run_id] = freq_result

            print(f"    Time: {elapsed:.1f}s")
            for P in periods:
                r = freq_result[P]
                print(f"    P={P}: top1={r['top1_regions']}, n={r['n_patches']}")

            restore_yarn(model, orig)
            del model, top1_c, top5_c
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # Frequency comparison table
        if len(variant_names) >= 2:
            r1_freq = freq_results.get(f"{variant_names[0]}_seed{args.seed}", {})
            r2_freq = freq_results.get(f"{variant_names[1]}_seed{args.seed}", {})
            if r1_freq and r2_freq:
                print(f"\n  FREQUENCY DECOMPOSITION: {variant_names[0]} vs {variant_names[1]}")
                print(f"  {'Period':<8} {'Region':<15} {variant_names[0]:>10} "
                      f"{variant_names[1]:>10} {'Delta':>10}")
                print(f"  {'-' * 60}")
                for P in periods:
                    for region in ["train", "near_extrap", "mid_extrap",
                                   "far_extrap", "all_extrap"]:
                        v1 = r1_freq[P]["top1_regions"].get(region)
                        v2 = r2_freq[P]["top1_regions"].get(region)
                        if v1 is None or v2 is None:
                            continue
                        delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
                        print(f"  P={P:<5} {region:<15} {v1:>10.4f} "
                              f"{v2:>10.4f} {delta:>+9.2f}%")
                    print()

                # Top-5 table
                print(f"\n  FREQUENCY DECOMPOSITION (Top-5): "
                      f"{variant_names[0]} vs {variant_names[1]}")
                print(f"  {'Period':<8} {'Region':<15} {variant_names[0]:>10} "
                      f"{variant_names[1]:>10} {'Delta':>10}")
                print(f"  {'-' * 60}")
                for P in periods:
                    for region in ["train", "near_extrap", "mid_extrap",
                                   "far_extrap", "all_extrap"]:
                        v1 = r1_freq[P]["top5_regions"].get(region)
                        v2 = r2_freq[P]["top5_regions"].get(region)
                        if v1 is None or v2 is None:
                            continue
                        delta = (v2 - v1) / max(abs(v1), 1e-9) * 100
                        print(f"  P={P:<5} {region:<15} {v1:>10.4f} "
                              f"{v2:>10.4f} {delta:>+9.2f}%")
                    print()

    print(f"\n  Total: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # Save summary
    summary = {
        "results": all_results,
        "config": {
            "data_dir": str(args.data_dir),
            "ckpt_dir": str(args.ckpt_dir),
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "patches_per_frame": ppf,
            "oscillation_periods": periods,
            "n_videos": args.n_videos,
            "batch_size": args.batch_size,
            "yarn_modes": [m[0] for m in yarn_modes],
        },
        "total_time_sec": round(total_time, 1),
    }
    if freq_results:
        # Convert int keys to strings for JSON
        summary["freq_results"] = {
            variant_id: {str(P): data for P, data in period_data.items()}
            for variant_id, period_data in freq_results.items()
        }
    summary_path = args.work_dir / "temporal_precision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
