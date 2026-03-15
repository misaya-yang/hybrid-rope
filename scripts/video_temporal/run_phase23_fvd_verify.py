#!/usr/bin/env python3
"""Single-seed FVD verification: geo_k16 vs evq_k16 on Moving MNIST.

This is the VERIFICATION-FIRST script: one seed, one decisive pair,
full FVD pipeline. Only scale to multi-seed after this confirms
FVD(EVQ) < FVD(Geo).

Usage:
    # Full verification run (train + generate + FVD)
    python scripts/video_temporal/run_phase23_fvd_verify.py

    # Quick test with fewer videos (for debugging pipeline)
    python scripts/video_temporal/run_phase23_fvd_verify.py \
        --n-generate 64 --gen-batch-size 8 --epochs 2 --quick

    # Dry run (no training, just validate config)
    python scripts/video_temporal/run_phase23_fvd_verify.py --dry-run

    # Skip training, only run FVD on existing checkpoints
    python scripts/video_temporal/run_phase23_fvd_verify.py --eval-only \
        --work-dir results/supporting_video/phase23_fvd_verify/<timestamp>

Time estimate on R6000 Blackwell 96GB:
    Training: ~2.5h per arm (5h total)
    Generation: ~1.5h (1024 videos × 4 frame counts × 2 arms)
    FVD: ~0.5h
    Total: ~7h (well within 12h budget, leaving buffer for debugging)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_video_temporal import (  # noqa: E402
    DEVICE,
    DTYPE,
    VideoGPT,
    eval_video_model,
    evq_cosh_inv_freq,
    set_seed,
    train_video_model,
)
from run_video_temporal_allocation_sweep import (  # noqa: E402
    MODEL_PROFILES,
    VARIANT_SPECS,
    build_variant_inv_freqs,
    load_manifest,
    load_tokens,
    parse_variants,
    summarize_variant,
)
from generate_and_eval_fvd import (  # noqa: E402
    compute_fvd,
    compute_prediction_metrics,
    compute_temporal_coherence,
    decode_tokens_to_frames,
    evaluate_checkpoint,
    try_load_i3d,
)


def save_checkpoint(model: VideoGPT, path: Path, metadata: dict) -> None:
    """Save model checkpoint with metadata."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)
    print(f"  Checkpoint saved: {path} ({path.stat().st_size / 1e6:.1f}MB)")


def load_checkpoint(
    model: VideoGPT, path: Path
) -> dict:
    """Load model checkpoint, return metadata."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("metadata", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-seed FVD verification: geo_k16 vs evq_k16"
    )
    # Data
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/video_temporal/generated/moving_mnist_medium"),
    )
    # Model
    parser.add_argument("--profile", type=str, default="blackwell96")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument(
        "--variants", type=str, default="geo_k16,evq_k16",
        help="Variants to compare (default: decisive pair only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Training overrides
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--eval-chunks", type=int, default=16)
    # Generation
    parser.add_argument("--n-generate", type=int, default=1024,
                        help="Number of videos to generate per arm per frame count")
    parser.add_argument("--gen-batch-size", type=int, default=16,
                        help="Batch size for autoregressive generation")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-context-window", type=int, default=0,
                        help="Sliding window for generation (0=full context)")
    # FVD
    parser.add_argument("--eval-frames", type=str, default="",
                        help="Frame counts to evaluate (default: 1x,2x,3x,4x)")
    parser.add_argument("--i3d-path", type=str,
                        default="data/video_temporal/external/i3d_torchscript.pt")
    # Control
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load existing checkpoints")
    parser.add_argument("--finetune-from", type=Path, default=None,
                        help="Directory with base checkpoints to fine-tune from "
                             "(loads weights before training on new data)")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.3,
                        help="LR multiplier for fine-tuning (default: 0.3x base LR)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs, fewer videos for pipeline test")
    args = parser.parse_args()

    # --- Config ---
    manifest = load_manifest(args.data_dir)
    train_frames = int(manifest["train_frames"])
    patches_per_frame = int(manifest["patches_per_frame"])
    image_size = int(manifest["image_size"])
    patch_size = int(manifest["patch_size"])
    grid_h = grid_w = image_size // patch_size

    eval_frames = (
        [int(x) for x in args.eval_frames.split(",") if x.strip()]
        if args.eval_frames
        else [train_frames, train_frames * 2, train_frames * 3, train_frames * 4]
    )
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
        "lr": profile["lr"],
        "batch_size": profile["batch_size"],
        "epochs": args.epochs,
    }
    if args.batch_size > 0:
        cfg["batch_size"] = args.batch_size
    if args.lr > 0.0:
        cfg["lr"] = args.lr

    if args.quick:
        cfg["epochs"] = min(cfg["epochs"], 2)
        args.n_generate = min(args.n_generate, 64)
        args.gen_batch_size = min(args.gen_batch_size, 8)
        eval_frames = [train_frames, train_frames * 2]  # only 1x and 2x
        print("[QUICK MODE] Reduced epochs/videos/frames for pipeline testing")

    # Work directory
    if args.work_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.work_dir = Path(f"results/supporting_video/phase23_fvd_verify/{stamp}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 72}")
    print(f"  FVD VERIFICATION: {[v['name'] for v in variants]} @ seed={args.seed}")
    print(f"  device={DEVICE} dtype={DTYPE} profile={args.profile}")
    print(f"  train_frames={train_frames} eval_frames={eval_frames}")
    print(f"  n_generate={args.n_generate} temperature={args.temperature}")
    print(f"  work_dir={args.work_dir}")
    print(f"{'#' * 72}\n")

    if args.dry_run:
        print(json.dumps({
            "cfg": cfg,
            "variants": variants,
            "eval_frames": eval_frames,
            "n_generate": args.n_generate,
        }, indent=2))
        print("\n[DRY RUN] Config validated. Exiting.")
        return

    # --- Load data ---
    print("Loading data...")
    train_tokens = load_tokens(args.data_dir, "train")
    test_tokens = load_tokens(args.data_dir, "test")
    val_tokens = load_tokens(args.data_dir, "val")
    val_flat = val_tokens.reshape(-1)
    print(f"  train: {train_tokens.shape}, test: {test_tokens.shape}, val: {val_tokens.shape}")

    # --- Try loading I3D ---
    i3d_model = try_load_i3d(args.i3d_path)

    # --- Main loop ---
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
        ckpt_path = args.work_dir / f"{run_id}.pt"

        print(f"\n{'=' * 72}")
        print(f"  VARIANT: {run_id}")
        print(f"  temporal_pairs={variant['temporal_pairs']} tau={variant['tau']:.4f}")
        print(f"  split={split_info}")
        print(f"{'=' * 72}")

        # Build model
        set_seed(args.seed)
        model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)

        # --- Phase 1: Train (or load checkpoint) ---
        if args.eval_only and ckpt_path.exists():
            print(f"\n  [EVAL-ONLY] Loading checkpoint: {ckpt_path}")
            meta = load_checkpoint(model, ckpt_path)
            train_time = meta.get("train_time_sec", 0)
        else:
            # Fine-tune: load base checkpoint first, then train on new data
            ft_cfg = cfg.copy()
            if args.finetune_from is not None:
                base_ckpt = args.finetune_from / f"{run_id}.pt"
                if base_ckpt.exists():
                    print(f"\n  [FINETUNE] Loading base weights: {base_ckpt}")
                    load_checkpoint(model, base_ckpt)
                    ft_cfg["lr"] = cfg["lr"] * args.finetune_lr_factor
                    print(f"  [FINETUNE] LR: {cfg['lr']:.2e} → {ft_cfg['lr']:.2e} "
                          f"(×{args.finetune_lr_factor})")
                else:
                    print(f"  [FINETUNE] Warning: base checkpoint not found: {base_ckpt}")
                    print(f"  [FINETUNE] Training from scratch instead")

            print(f"\n  Phase 1: Training {run_id}...")
            t_train = time.time()
            model = train_video_model(model, train_tokens, ft_cfg, seed=args.seed)
            train_time = time.time() - t_train
            print(f"  Training done in {train_time / 60:.1f} min")

            # Save checkpoint
            save_checkpoint(model, ckpt_path, {
                "variant": variant["name"],
                "seed": args.seed,
                "tau": variant["tau"],
                "temporal_pairs": variant["temporal_pairs"],
                "split": split_info,
                "epochs": ft_cfg["epochs"],
                "train_time_sec": round(train_time, 1),
                "finetuned_from": str(args.finetune_from) if args.finetune_from else None,
            })

        # --- Phase 2: PPL evaluation (quick, for comparison with existing results) ---
        print(f"\n  Phase 2: PPL evaluation...")
        ppl_raw = eval_video_model(
            model, val_flat, eval_frames,
            patches_per_frame=patches_per_frame,
            train_frames=train_frames,
            eval_chunks=args.eval_chunks,
            yarn=False,
        )
        ppl_yarn = eval_video_model(
            model, val_flat, eval_frames,
            patches_per_frame=patches_per_frame,
            train_frames=train_frames,
            eval_chunks=args.eval_chunks,
            yarn=True,
        )

        # --- Phase 3: FVD evaluation ---
        print(f"\n  Phase 3: FVD evaluation...")
        fvd_results = evaluate_checkpoint(
            model=model,
            test_tokens=test_tokens,
            train_frames=train_frames,
            eval_frame_counts=eval_frames,
            patches_per_frame=patches_per_frame,
            patch_size=patch_size,
            grid_h=grid_h,
            grid_w=grid_w,
            n_generate=args.n_generate,
            temperature=args.temperature,
            top_k=args.top_k,
            gen_batch_size=args.gen_batch_size,
            i3d_model=i3d_model,
            seed=args.seed,
            max_context_window=args.max_context_window,
        )

        # --- Save per-variant results ---
        result = {
            "variant": variant["name"],
            "seed": args.seed,
            "temporal_pairs": variant["temporal_pairs"],
            "tau": round(float(variant["tau"]), 6),
            "split": split_info,
            "train_time_sec": round(train_time, 1),
            "ppl_raw": ppl_raw,
            "ppl_yarn": ppl_yarn,
            "fvd": fvd_results,
        }
        all_results[run_id] = result
        (args.work_dir / f"{run_id}_fvd.json").write_text(
            json.dumps(result, indent=2)
        )

        # Cleanup
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - t0_total

    # --- Comparison summary ---
    print(f"\n\n{'=' * 72}")
    print(f"  FVD VERIFICATION SUMMARY")
    print(f"{'=' * 72}")

    # Print comparison table
    variant_names = [v["name"] for v in variants]
    if len(variant_names) >= 2:
        v1, v2 = variant_names[0], variant_names[1]
        r1 = all_results.get(f"{v1}_seed{args.seed}", {})
        r2 = all_results.get(f"{v2}_seed{args.seed}", {})

        print(f"\n  {'Metric':<25} {'geo_k16':>12} {'evq_k16':>12} {'Delta':>12} {'Winner':>8}")
        print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

        # PPL comparison
        for key in sorted(set(list(r1.get("ppl_raw", {}).keys()) +
                               list(r2.get("ppl_raw", {}).keys()))):
            val1 = r1.get("ppl_raw", {}).get(key)
            val2 = r2.get("ppl_raw", {}).get(key)
            if val1 is not None and val2 is not None:
                delta = (val2 - val1) / val1 * 100
                winner = "EVQ" if val2 < val1 else "Geo"
                print(f"  PPL raw {key:<16} {val1:>12.2f} {val2:>12.2f} {delta:>+11.1f}% {winner:>8}")

        # FVD comparison
        for frame_key in eval_frames:
            fk = f"{frame_key}f"
            fvd1 = r1.get("fvd", {}).get(fk, {})
            fvd2 = r2.get("fvd", {}).get(fk, {})

            for metric in ["pixel_fvd", "i3d_fvd", "mse", "token_accuracy"]:
                val1 = fvd1.get(metric)
                val2 = fvd2.get(metric)
                if val1 is not None and val2 is not None:
                    if metric == "token_accuracy":
                        # Higher is better
                        delta = (val2 - val1) / max(val1, 1e-8) * 100
                        winner = "EVQ" if val2 > val1 else "Geo"
                    else:
                        # Lower is better
                        delta = (val2 - val1) / max(abs(val1), 1e-8) * 100
                        winner = "EVQ" if val2 < val1 else "Geo"
                    print(f"  {metric}@{fk:<12} {val1:>12.4f} {val2:>12.4f} {delta:>+11.1f}% {winner:>8}")

    # --- Verdict ---
    print(f"\n  VERDICT:")
    evq_wins_fvd = 0
    geo_wins_fvd = 0
    for fk in eval_frames:
        frame_key = f"{fk}f"
        r1_fvd = all_results.get(f"geo_k16_seed{args.seed}", {}).get("fvd", {}).get(frame_key, {})
        r2_fvd = all_results.get(f"evq_k16_seed{args.seed}", {}).get("fvd", {}).get(frame_key, {})
        pf1 = r1_fvd.get("pixel_fvd")
        pf2 = r2_fvd.get("pixel_fvd")
        if pf1 is not None and pf2 is not None:
            if pf2 < pf1:
                evq_wins_fvd += 1
            else:
                geo_wins_fvd += 1

    total_compared = evq_wins_fvd + geo_wins_fvd
    if total_compared > 0:
        if evq_wins_fvd > geo_wins_fvd:
            print(f"  EVQ wins FVD at {evq_wins_fvd}/{total_compared} frame counts")
            print(f"  --> PROCEED to multi-seed replication")
        elif evq_wins_fvd == geo_wins_fvd:
            print(f"  TIE: EVQ wins {evq_wins_fvd}/{total_compared} frame counts")
            print(f"  --> CHECK extrapolation ratios (expect EVQ better at higher ratios)")
        else:
            print(f"  Geo wins FVD at {geo_wins_fvd}/{total_compared} frame counts")
            print(f"  --> INVESTIGATE: try different temperature, check training convergence")

    # --- Save summary ---
    summary = {
        "experiment": "phase23_fvd_verification",
        "profile": args.profile,
        "seed": args.seed,
        "variants": [v["name"] for v in variants],
        "eval_frames": eval_frames,
        "n_generate": args.n_generate,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "total_time_min": round(total_time / 60, 2),
        "device": DEVICE,
        "dtype": str(DTYPE),
        "results": all_results,
        "evq_fvd_wins": evq_wins_fvd,
        "geo_fvd_wins": geo_wins_fvd,
    }
    summary_path = args.work_dir / "fvd_verify_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Total time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
