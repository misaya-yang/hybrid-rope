#!/usr/bin/env python3
"""Per-frame teacher-forced accuracy analysis.

No sampling, no generation — just forward pass on test data.
Produces per-frame top-1 accuracy + per-frame PPL curves that
directly visualize where EVQ outperforms Geo.

This is THE most compelling visualization for a frequency allocation
paper: it shows the PPL advantage as a function of temporal distance,
without any sampling noise.

Usage:
    # From existing checkpoints (no retraining):
    python scripts/video_temporal/run_perframe_accuracy.py \
        --ckpt-dir results/supporting_video/phase23_fvd_verify/<timestamp>

    # Quick test:
    python scripts/video_temporal/run_perframe_accuracy.py \
        --ckpt-dir results/supporting_video/phase23_fvd_verify/<timestamp> \
        --max-videos 64

Time: ~10 min on R6000 (forward pass only, no training).

Output:
    - perframe_accuracy.json: raw data
    - perframe_accuracy.csv: for plotting
    - perframe_summary.txt: human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_video_temporal import (  # noqa: E402
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    VideoGPT,
    evq_cosh_inv_freq,
    set_seed,
)
from run_video_temporal_allocation_sweep import (  # noqa: E402
    MODEL_PROFILES,
    VARIANT_SPECS,
    build_variant_inv_freqs,
    load_manifest,
    load_tokens,
    parse_variants,
)
from contextlib import nullcontext


@torch.no_grad()
def compute_perframe_metrics(
    model: VideoGPT,
    data: torch.Tensor,
    patches_per_frame: int,
    max_videos: int = 0,
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """Compute per-frame teacher-forced metrics on data.

    For each token position, compute:
      - top-1 accuracy (argmax == target)
      - top-5 accuracy
      - cross-entropy loss (= log PPL)
      - mean rank of correct token

    Returns dict of arrays, each shaped (total_tokens,), which can be
    reshaped to (n_frames, patches_per_frame) for per-frame analysis.
    """
    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    N = data.shape[0]
    if max_videos > 0:
        N = min(N, max_videos)
    L = data.shape[1]

    # Extend RoPE
    model.extend_rope(L + 1)

    # Accumulate per-position metrics
    total_tokens = L - 1  # predict position 1..L from 0..L-1
    acc_top1 = np.zeros(total_tokens, dtype=np.float64)
    acc_top5 = np.zeros(total_tokens, dtype=np.float64)
    loss_sum = np.zeros(total_tokens, dtype=np.float64)
    rank_sum = np.zeros(total_tokens, dtype=np.float64)
    count = np.zeros(total_tokens, dtype=np.float64)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.time()

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, N)
        batch = data[start:end].to(DEVICE)  # (B, L)
        B = batch.shape[0]

        with ctx:
            logits = model(batch[:, :-1])  # (B, L-1, V)
            targets = batch[:, 1:]  # (B, L-1)

            # Per-position cross-entropy (no reduction)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none',
            ).reshape(B, -1)  # (B, L-1)

            # Top-1 accuracy
            preds = logits.argmax(dim=-1)  # (B, L-1)
            correct_top1 = (preds == targets).float()  # (B, L-1)

            # Top-5 accuracy
            topk = torch.topk(logits, k=5, dim=-1).indices  # (B, L-1, 5)
            correct_top5 = (topk == targets.unsqueeze(-1)).any(dim=-1).float()

            # Rank of correct token (0-indexed)
            # Sort logits descending, find where target appears
            sorted_indices = logits.argsort(dim=-1, descending=True)  # (B, L-1, V)
            ranks = (sorted_indices == targets.unsqueeze(-1)).float().argmax(dim=-1)  # (B, L-1)

        # Accumulate (move to CPU)
        ce_np = ce.cpu().numpy()
        top1_np = correct_top1.cpu().numpy()
        top5_np = correct_top5.cpu().numpy()
        ranks_np = ranks.cpu().float().numpy()

        for i in range(B):
            acc_top1 += top1_np[i]
            acc_top5 += top5_np[i]
            loss_sum += ce_np[i]
            rank_sum += ranks_np[i]
            count += 1.0

        if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (b_idx + 1) * (n_batches - b_idx - 1)
            print(f"    batch {b_idx+1}/{n_batches}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        del batch, logits, targets, ce, preds, topk, sorted_indices, ranks
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Average
    mask = count > 0
    acc_top1[mask] /= count[mask]
    acc_top5[mask] /= count[mask]
    loss_sum[mask] /= count[mask]
    rank_sum[mask] /= count[mask]

    return {
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "cross_entropy": loss_sum,
        "mean_rank": rank_sum,
        "count": count,
    }


def aggregate_per_frame(
    metrics: Dict[str, np.ndarray],
    patches_per_frame: int,
) -> Dict[str, np.ndarray]:
    """Aggregate per-token metrics to per-frame averages."""
    total = len(metrics["top1_accuracy"])
    n_frames = total // patches_per_frame
    # Trim to exact frame boundary
    trim = n_frames * patches_per_frame

    result = {}
    for key in ["top1_accuracy", "top5_accuracy", "cross_entropy", "mean_rank"]:
        arr = metrics[key][:trim].reshape(n_frames, patches_per_frame)
        result[key] = arr.mean(axis=1)  # (n_frames,)

    # Also compute per-frame PPL from cross-entropy
    result["ppl"] = np.exp(result["cross_entropy"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame teacher-forced accuracy analysis"
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help="Directory with model checkpoints (.pt files)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/video_temporal/generated/moving_mnist_medium"))
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"])
    parser.add_argument("--variants", type=str, default="geo_k16,evq_k16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=str, default="blackwell96")
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Limit number of test videos (0=all)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.data_dir)
    train_frames = int(manifest["train_frames"])
    patches_per_frame = int(manifest["patches_per_frame"])
    image_size = int(manifest["image_size"])
    patch_size = int(manifest["patch_size"])
    grid_h = grid_w = image_size // patch_size

    profile = MODEL_PROFILES[args.profile]
    cfg = {
        "vocab_size": int(manifest["vocab_size"]),
        "hidden_size": profile["hidden_size"],
        "num_layers": profile["num_layers"],
        "num_heads": profile["num_heads"],
        "head_dim": profile["head_dim"],
        "intermediate_size": profile["intermediate_size"],
        "max_T": profile["max_T"],
    }

    variants = parse_variants(args.variants, train_frames=train_frames)
    output_dir = args.output_dir or args.ckpt_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  PER-FRAME ACCURACY ANALYSIS")
    print(f"  ckpt_dir={args.ckpt_dir}")
    print(f"  split={args.split}  train_frames={train_frames}")
    print(f"  patches_per_frame={patches_per_frame}")
    print(f"{'#' * 60}\n")

    # Load data
    data = load_tokens(args.data_dir, args.split)
    print(f"  Data: {data.shape} ({args.split})")

    all_results = {}

    for variant in variants:
        run_id = f"{variant['name']}_seed{args.seed}"
        ckpt_path = args.ckpt_dir / f"{run_id}.pt"

        if not ckpt_path.exists():
            print(f"\n  SKIP {run_id}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {run_id}")
        print(f"{'=' * 60}")

        inv_h, inv_w, inv_t, split_info = build_variant_inv_freqs(
            head_dim=cfg["head_dim"],
            base=args.base,
            train_frames=train_frames,
            variant=variant,
        )

        set_seed(args.seed)
        model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path}")

        # Compute per-position metrics
        t0 = time.time()
        raw_metrics = compute_perframe_metrics(
            model, data, patches_per_frame,
            max_videos=args.max_videos,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Aggregate to per-frame
        frame_metrics = aggregate_per_frame(raw_metrics, patches_per_frame)
        n_frames = len(frame_metrics["top1_accuracy"])

        all_results[run_id] = {
            "variant": variant["name"],
            "n_frames": n_frames,
            "train_frames": train_frames,
            "per_frame": {
                k: v.tolist() for k, v in frame_metrics.items()
            },
        }

        # Print summary by region
        for region_name, start_f, end_f in [
            ("train (1-32f)", 0, train_frames),
            ("2x (33-64f)", train_frames, train_frames * 2),
            ("3x (65-96f)", train_frames * 2, train_frames * 3),
            ("4x (97-128f)", train_frames * 3, train_frames * 4),
        ]:
            if start_f >= n_frames:
                break
            end_f = min(end_f, n_frames)
            acc = frame_metrics["top1_accuracy"][start_f:end_f].mean()
            ppl = frame_metrics["ppl"][start_f:end_f].mean()
            print(f"    {region_name}: top1={acc:.4f}  PPL={ppl:.3f}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- Comparison ---
    variant_names = [v["name"] for v in variants]
    if len(variant_names) >= 2 and all(
        f"{v}_seed{args.seed}" in all_results for v in variant_names
    ):
        v1, v2 = variant_names[0], variant_names[1]
        r1 = all_results[f"{v1}_seed{args.seed}"]
        r2 = all_results[f"{v2}_seed{args.seed}"]

        print(f"\n{'=' * 60}")
        print(f"  COMPARISON: {v1} vs {v2}")
        print(f"{'=' * 60}")

        for region_name, start_f, end_f in [
            ("train (1-32f)", 0, train_frames),
            ("2x (33-64f)", train_frames, train_frames * 2),
            ("3x (65-96f)", train_frames * 2, train_frames * 3),
            ("4x (97-128f)", train_frames * 3, train_frames * 4),
        ]:
            n1 = len(r1["per_frame"]["top1_accuracy"])
            n2 = len(r2["per_frame"]["top1_accuracy"])
            if start_f >= min(n1, n2):
                break
            ef = min(end_f, n1, n2)
            a1 = np.array(r1["per_frame"]["top1_accuracy"][start_f:ef]).mean()
            a2 = np.array(r2["per_frame"]["top1_accuracy"][start_f:ef]).mean()
            p1 = np.array(r1["per_frame"]["ppl"][start_f:ef]).mean()
            p2 = np.array(r2["per_frame"]["ppl"][start_f:ef]).mean()
            acc_delta = (a2 - a1) / a1 * 100
            ppl_delta = (p2 - p1) / p1 * 100
            winner_acc = "EVQ" if a2 > a1 else "Geo"
            winner_ppl = "EVQ" if p2 < p1 else "Geo"
            print(f"  {region_name}:")
            print(f"    top1: {v1}={a1:.4f}  {v2}={a2:.4f}  delta={acc_delta:+.2f}%  ({winner_acc})")
            print(f"    PPL:  {v1}={p1:.3f}  {v2}={p2:.3f}  delta={ppl_delta:+.1f}%  ({winner_ppl})")

        # Export CSV for plotting
        csv_path = output_dir / "perframe_accuracy.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "extrap_ratio",
                f"{v1}_top1", f"{v2}_top1", "top1_delta",
                f"{v1}_ppl", f"{v2}_ppl", "ppl_delta",
                f"{v1}_top5", f"{v2}_top5",
                f"{v1}_rank", f"{v2}_rank",
            ])
            n = min(len(r1["per_frame"]["top1_accuracy"]),
                    len(r2["per_frame"]["top1_accuracy"]))
            for i in range(n):
                a1 = r1["per_frame"]["top1_accuracy"][i]
                a2 = r2["per_frame"]["top1_accuracy"][i]
                p1 = r1["per_frame"]["ppl"][i]
                p2 = r2["per_frame"]["ppl"][i]
                t5_1 = r1["per_frame"]["top5_accuracy"][i]
                t5_2 = r2["per_frame"]["top5_accuracy"][i]
                rk1 = r1["per_frame"]["mean_rank"][i]
                rk2 = r2["per_frame"]["mean_rank"][i]
                writer.writerow([
                    i + 1, round((i + 1) / train_frames, 3),
                    round(a1, 6), round(a2, 6), round(a2 - a1, 6),
                    round(p1, 4), round(p2, 4), round(p2 - p1, 4),
                    round(t5_1, 6), round(t5_2, 6),
                    round(rk1, 4), round(rk2, 4),
                ])
        print(f"\n  CSV saved: {csv_path}")
        print(f"  (Import to matplotlib / Excel for plotting)")

    # Save JSON
    json_path = output_dir / "perframe_accuracy.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"  JSON saved: {json_path}")

    # --- Plotting hint ---
    print(f"""
  PLOTTING (copy-paste to generate figure):
  ─────────────────────────────────────────
  import pandas as pd
  import matplotlib.pyplot as plt

  df = pd.read_csv("{csv_path}")
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

  ax1.plot(df['frame'], df['geo_k16_top1'], label='Geo', alpha=0.7)
  ax1.plot(df['frame'], df['evq_k16_top1'], label='EVQ-Cosh', alpha=0.7)
  ax1.axvline(x={train_frames}, color='red', ls='--', label='Train boundary')
  ax1.set_ylabel('Top-1 Accuracy')
  ax1.legend()
  ax1.set_title('Per-Frame Prediction Accuracy (Teacher-Forced)')

  ax2.plot(df['frame'], df['geo_k16_ppl'], label='Geo')
  ax2.plot(df['frame'], df['evq_k16_ppl'], label='EVQ-Cosh')
  ax2.axvline(x={train_frames}, color='red', ls='--', label='Train boundary')
  ax2.set_ylabel('Per-Frame PPL')
  ax2.set_xlabel('Frame Index')
  ax2.legend()

  plt.tight_layout()
  plt.savefig('perframe_accuracy.pdf', dpi=150)
  plt.show()
""")


if __name__ == "__main__":
    main()
