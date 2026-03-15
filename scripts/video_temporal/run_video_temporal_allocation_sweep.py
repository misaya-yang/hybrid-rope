#!/usr/bin/env python3
"""Temporal-allocation sweep for video RoPE on tokenized Moving MNIST.

Goal:
  Test whether the temporal axis benefits from:
  1. more channel budget (`K_t` larger), and
  2. EVQ-Cosh reshaping within that temporal subspace.

This is the cleanest medium-cost bridge from the text results to VideoRoPE's
LTA claim before touching a full video VLM stack.
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


MODEL_PROFILES = {
    "medium": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "lr": 4e-4,
        "batch_size": 24,
        "epochs": 20,
        "max_T": 256,
    },
    "blackwell96": {
        "hidden_size": 1024,
        "num_layers": 16,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4096,
        "lr": 3e-4,
        "batch_size": 24,
        "epochs": 24,
        "max_T": 256,
    },
}

VARIANT_SPECS = {
    "geo_k8": {"temporal_pairs": 8, "tau_mode": "zero"},
    "geo_k12": {"temporal_pairs": 12, "tau_mode": "zero"},
    "geo_k16": {"temporal_pairs": 16, "tau_mode": "zero"},
    "evq_k12": {"temporal_pairs": 12, "tau_mode": "auto"},
    "evq_k16": {"temporal_pairs": 16, "tau_mode": "auto"},
}


def load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def load_tokens(data_dir: Path, split: str) -> torch.Tensor:
    path = data_dir / f"{split}_tokens.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing split tokens: {path}")
    arr = np.load(path)
    return torch.from_numpy(arr.astype(np.int64))


def parse_variants(variant_names: str, train_frames: int) -> List[dict]:
    variants = []
    for name in [v.strip() for v in variant_names.split(",") if v.strip()]:
        if name not in VARIANT_SPECS:
            raise ValueError(f"unknown variant {name}; choices={sorted(VARIANT_SPECS)}")
        spec = VARIANT_SPECS[name].copy()
        temporal_pairs = spec["temporal_pairs"]
        tau = 0.0 if spec["tau_mode"] == "zero" else temporal_pairs / math.sqrt(train_frames)
        spec.update({"name": name, "tau": tau})
        variants.append(spec)
    return variants


def split_pairs(head_dim: int, temporal_pairs: int) -> tuple[int, int, int]:
    total_pairs = head_dim // 2
    if temporal_pairs <= 0 or temporal_pairs >= total_pairs:
        raise ValueError(f"invalid temporal_pairs={temporal_pairs} for head_dim={head_dim}")
    remaining = total_pairs - temporal_pairs
    spatial_h = remaining // 2
    spatial_w = remaining - spatial_h
    if min(spatial_h, spatial_w) <= 0:
        raise ValueError(
            f"not enough spatial pairs after temporal allocation: "
            f"{spatial_h=}, {spatial_w=}, {temporal_pairs=}"
        )
    return spatial_h, spatial_w, temporal_pairs


def build_variant_inv_freqs(head_dim: int, base: float, train_frames: int, variant: dict):
    k_h, k_w, k_t = split_pairs(head_dim, variant["temporal_pairs"])
    inv_freq_h = evq_cosh_inv_freq(k_h * 2, tau=0.0, base=base)
    inv_freq_w = evq_cosh_inv_freq(k_w * 2, tau=0.0, base=base)
    inv_freq_t = evq_cosh_inv_freq(k_t * 2, tau=variant["tau"], base=base)
    return inv_freq_h, inv_freq_w, inv_freq_t, {"K_h": k_h, "K_w": k_w, "K_t": k_t}


def summarize_variant(results: Dict[str, dict], eval_frames: List[int]) -> List[dict]:
    table = []
    for name, res in results.items():
        row = {"variant": name}
        raw = res["ppl_raw"]
        yarn = res["ppl_yarn"]
        for n_frames in eval_frames:
            row[f"raw_{n_frames}f"] = raw.get(f"{n_frames}f")
            row[f"yarn_{n_frames}f"] = yarn.get(f"yarn_{n_frames}f")
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Video temporal allocation sweep.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/video_temporal/generated/moving_mnist_medium"),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="geo_k8,geo_k12,geo_k16,evq_k12,evq_k16",
    )
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--profile", type=str, default="blackwell96", choices=sorted(MODEL_PROFILES))
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--eval-frames", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0, help="override profile epochs when > 0")
    parser.add_argument("--batch-size", type=int, default=0, help="override profile batch size when > 0")
    parser.add_argument("--lr", type=float, default=0.0, help="override profile learning rate when > 0")
    parser.add_argument("--eval-chunks", type=int, default=24)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/supporting_video/phase23_video_temporal_blackwell"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.data_dir)
    train_frames = int(manifest["train_frames"])
    patches_per_frame = int(manifest["patches_per_frame"])
    image_size = int(manifest["image_size"])
    patch_size = int(manifest["patch_size"])
    grid_h = grid_w = image_size // patch_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
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
        "epochs": profile["epochs"],
    }
    if args.epochs > 0:
        cfg["epochs"] = args.epochs
    if args.batch_size > 0:
        cfg["batch_size"] = args.batch_size
    if args.lr > 0.0:
        cfg["lr"] = args.lr

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[video-temporal] device={DEVICE} dtype={DTYPE} profile={args.profile}")
    print(f"[video-temporal] data_dir={args.data_dir}")
    print(f"[video-temporal] train_frames={train_frames} eval_frames={eval_frames}")
    print(f"[video-temporal] variants={[v['name'] for v in variants]}")

    if args.dry_run:
        print(json.dumps({
            "manifest": manifest,
            "cfg": cfg,
            "variants": variants,
            "eval_frames": eval_frames,
            "work_dir": str(work_dir),
        }, indent=2))
        return

    train_tokens = load_tokens(args.data_dir, "train")
    val_tokens = load_tokens(args.data_dir, "val")
    val_flat = val_tokens.reshape(-1)

    all_results = {}
    t0 = time.time()

    for variant in variants:
        inv_h, inv_w, inv_t, split_info = build_variant_inv_freqs(
            head_dim=cfg["head_dim"],
            base=args.base,
            train_frames=train_frames,
            variant=variant,
        )
        for seed in seeds:
            set_seed(seed)
            run_id = f"{variant['name']}_seed{seed}"
            print(f"\n{'=' * 72}\nRUN {run_id}\n{'=' * 72}")
            print(
                f"temporal_pairs={variant['temporal_pairs']} tau={variant['tau']:.4f} "
                f"split={split_info}"
            )

            model = VideoGPT(cfg, inv_h, inv_w, inv_t, grid_h, grid_w).to(DEVICE)
            train_start = time.time()
            model = train_video_model(model, train_tokens, cfg, seed=seed)
            train_time = time.time() - train_start

            raw = eval_video_model(
                model,
                val_flat,
                eval_frames,
                patches_per_frame=patches_per_frame,
                train_frames=train_frames,
                eval_chunks=args.eval_chunks,
                yarn=False,
            )
            yarn = eval_video_model(
                model,
                val_flat,
                eval_frames,
                patches_per_frame=patches_per_frame,
                train_frames=train_frames,
                eval_chunks=args.eval_chunks,
                yarn=True,
            )

            result = {
                "variant": variant["name"],
                "seed": seed,
                "temporal_pairs": variant["temporal_pairs"],
                "tau": round(float(variant["tau"]), 6),
                "split": split_info,
                "ppl_raw": raw,
                "ppl_yarn": yarn,
                "train_time_sec": round(train_time, 1),
            }
            all_results[run_id] = result
            (work_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2))

            # Save checkpoint for downstream FVD evaluation
            ckpt_path = work_dir / f"{run_id}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "metadata": result,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Aggregate across seeds.
    aggregate: Dict[str, dict] = {}
    for variant in variants:
        runs = [res for res in all_results.values() if res["variant"] == variant["name"]]
        variant_out = {
            "variant": variant["name"],
            "temporal_pairs": variant["temporal_pairs"],
            "tau": round(float(variant["tau"]), 6),
            "ppl_raw": {},
            "ppl_yarn": {},
        }
        for n_frames in eval_frames:
            raw_vals = [run["ppl_raw"].get(f"{n_frames}f") for run in runs if f"{n_frames}f" in run["ppl_raw"]]
            yarn_key = f"yarn_{n_frames}f"
            yarn_vals = [run["ppl_yarn"].get(yarn_key) for run in runs if yarn_key in run["ppl_yarn"]]
            if raw_vals:
                variant_out["ppl_raw"][f"{n_frames}f"] = round(sum(raw_vals) / len(raw_vals), 3)
            if yarn_vals:
                variant_out["ppl_yarn"][yarn_key] = round(sum(yarn_vals) / len(yarn_vals), 3)
        aggregate[variant["name"]] = variant_out

    summary = {
        "metadata": {
            "experiment": "video_temporal_allocation_sweep",
            "profile": args.profile,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "data_dir": str(args.data_dir),
            "base": args.base,
            "train_frames": train_frames,
            "eval_frames": eval_frames,
            "eval_chunks": args.eval_chunks,
            "seeds": seeds,
            "total_time_min": round((time.time() - t0) / 60.0, 2),
        },
        "aggregate": aggregate,
        "table": summarize_variant(aggregate, eval_frames),
        "runs": all_results,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {work_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
