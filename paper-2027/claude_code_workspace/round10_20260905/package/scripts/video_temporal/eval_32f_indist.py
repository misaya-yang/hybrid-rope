#!/usr/bin/env python3
"""Pure in-distribution eval: 32f denoising precision (NO extrapolation, NO YaRN).

Disentangles training quality from extrapolation ability.
If τ=1.2 is fine at 32f but bad at 128f → problem is extrapolation.
If τ=1.2 is bad even at 32f → problem is training dynamics.

Usage:
    python eval_32f_indist.py --ckpt_dir results/video_dit/20260316_head2head
    python eval_32f_indist.py --ckpt_dir results/video_dit/base_sweep_b10000
"""
import os, sys, json, math, time, argparse, gc
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from contextlib import nullcontext

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from video_dit import (
    VideoDiT, RectifiedFlowScheduler, evq_cosh_inv_freq,
    generate_oscillating_mnist_pixels,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


@torch.no_grad()
def eval_denoising(model, val_videos, noise_level=0.5, batch_size=4):
    """Denoising precision on videos AT TRAINING LENGTH (no extrapolation)."""
    model.eval()
    scheduler = RectifiedFlowScheduler()
    n_frames = val_videos.shape[1]
    pH, pW = model.pH, model.pW
    ppf = pH * pW
    N = val_videos.shape[0]

    ctx = torch.amp.autocast("cuda", dtype=DTYPE)
    frame_mse_sum = torch.zeros(n_frames, device="cpu")
    frame_count = 0

    for start in range(0, N, batch_size):
        B = min(batch_size, N - start)
        batch = val_videos[start:start + B].to(DEVICE)
        patches = model.patchify(batch)
        noise = torch.randn_like(patches)
        t = torch.full((B,), noise_level, device=DEVICE)
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
    return {"mean_mse": float(frame_mse.mean()), "per_frame": frame_mse.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--eval_batch", type=int, default=8)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"No .pt in {ckpt_dir}")
        return

    # Config
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        cfg = {"frame_size": 64, "train_frames": 32, "val_samples": 256,
               "patch_size": 8, "hidden_size": 768, "num_layers": 12,
               "num_heads": 12, "head_dim": 64, "K_h": 8, "K_w": 8, "K_t": 16,
               "base": 10000.0}

    # Generate 32f validation data (same seed as training eval)
    print("Generating 32f validation data...")
    val_32f = generate_oscillating_mnist_pixels(
        cfg.get("val_samples", 256), cfg.get("train_frames", 32),
        cfg.get("frame_size", 64), seed=99999,
    )
    print(f"  Shape: {val_32f.shape}")

    results = {}
    for ckpt_path in ckpt_files:
        name = ckpt_path.stem
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        tau = ckpt.get("tau", 0.0)
        base = cfg.get("base", 10000.0)

        inv_freq_h = evq_cosh_inv_freq(cfg["K_h"] * 2, tau=0.0, base=base)
        inv_freq_w = evq_cosh_inv_freq(cfg["K_w"] * 2, tau=0.0, base=base)
        inv_freq_t = evq_cosh_inv_freq(cfg["K_t"] * 2, tau=tau, base=base)

        model = VideoDiT(
            in_channels=cfg.get("in_channels", 1),
            patch_size=cfg["patch_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            frame_size=cfg["frame_size"],
            max_T=cfg.get("max_T", 256),
            inv_freq_h=inv_freq_h, inv_freq_w=inv_freq_w, inv_freq_t=inv_freq_t,
        ).to(DEVICE)
        # Handle torch.compile _orig_mod. prefix
        state = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)

        for t in [0.2, 0.5, 0.8]:
            r = eval_denoising(model, val_32f, noise_level=t, batch_size=args.eval_batch)
            print(f"  t={t}: 32f MSE = {r['mean_mse']:.6f}")
            results[f"{name}_t{t}"] = {"tau": tau, "noise": t, **r}

        model.cpu(); del model, ckpt; gc.collect(); torch.cuda.empty_cache()

    out = ckpt_dir / "eval_32f_indist.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
