#!/usr/bin/env python3
"""Wan2.1 temporal extrapolation evaluation.

Generates videos at training length (81f) and extrapolated lengths (161f, 241f).
Computes temporal quality metrics and compares GEO vs EVQ.

Usage:
  python -u wan21_eval_extrap.py --method geo --lora_dir results/wan21_evq_288/geo_tau0.0/lora
  python -u wan21_eval_extrap.py --method evq --tau 3.2 --lora_dir results/wan21_evq_288/evq_tau3.2/lora
  python -u wan21_eval_extrap.py --method both  # runs both sequentially
"""
import os, sys, gc, json, math, time, argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/autodl-tmp/wan21/Wan2.1")

import torch
import torch.nn.functional as F
import numpy as np

MODEL_DIR = "/root/autodl-tmp/wan21"
RESULTS_DIR = "/root/autodl-tmp/results/wan21_evq_288"

# ============================================================
# EVQ-Cosh (same as training script)
# ============================================================
def evq_cosh_inv_freq(dim, tau, base=10000.0):
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)

def geometric_inv_freq(dim, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

def patch_temporal_freqs(model, inv_freq, t_dim=44):
    K_t = t_dim // 2
    positions = torch.arange(model.freqs.shape[0], dtype=torch.float64)
    raw = torch.outer(positions, inv_freq.double())
    new_freqs = torch.polar(torch.ones_like(raw), raw)
    model.freqs[:, :K_t] = new_freqs
    print(f"  Patched temporal RoPE: {K_t} pairs")

# ============================================================
# Temporal quality metrics
# ============================================================
def compute_metrics(frames_np):
    """Compute temporal metrics from numpy array [T, H, W, C] in [0,255] uint8."""
    frames = frames_np.astype(np.float32) / 255.0
    T = len(frames)
    if T < 2:
        return {}

    # Temporal MSE between consecutive frames
    diffs = frames[1:] - frames[:-1]
    per_frame_mse = (diffs ** 2).mean(axis=(1, 2, 3))

    # Motion magnitude
    motion_mag = np.abs(diffs).mean(axis=(1, 2, 3))

    # Temporal jitter (second-order smoothness)
    jitter = np.abs(np.diff(per_frame_mse)).mean() if T >= 3 else 0.0

    # Cosine similarity at different distances (pixel space)
    frames_flat = frames.reshape(T, -1)
    norms = np.linalg.norm(frames_flat, axis=1, keepdims=True) + 1e-8
    frames_normed = frames_flat / norms

    cos_sims = {}
    for d in [1, 2, 4, 8, 16, 32]:
        if d >= T:
            break
        sims = (frames_normed[:-d] * frames_normed[d:]).sum(axis=1)
        cos_sims[f"cos_d{d}"] = float(sims.mean())

    return {
        "temporal_mse": float(per_frame_mse.mean()),
        "temporal_mse_std": float(per_frame_mse.std()),
        "motion_mag": float(motion_mag.mean()),
        "jitter": float(jitter),
        "num_frames": T,
        **cos_sims,
    }


# ============================================================
# Generate and evaluate
# ============================================================
def run_eval(method, tau, lora_dir, frame_counts, prompts, output_dir, seed=42):
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  Eval: {method.upper()} tau={tau}")
    print(f"  Frames: {frame_counts}, Videos: {len(prompts)}")
    print(f"{'='*60}\n")

    # Load pipeline
    from wan.text2video import WanT2V
    print("[1] Loading WanT2V pipeline...")
    t2v = WanT2V(
        config=__import__('wan.configs', fromlist=['WAN_CONFIGS']).WAN_CONFIGS['t2v-1.3B'],
        checkpoint_dir=MODEL_DIR,
        device_id=0,
        t5_cpu=False,  # GPU for speed
    )

    # Patch temporal RoPE
    t_dim = 44
    if method == "evq":
        inv_freq = evq_cosh_inv_freq(t_dim, tau=tau)
    else:
        inv_freq = geometric_inv_freq(t_dim)
    patch_temporal_freqs(t2v.model, inv_freq.float(), t_dim=t_dim)
    t2v.model.freqs = t2v.model.freqs.to(device)

    # Load LoRA if available
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        t2v.model = PeftModel.from_pretrained(t2v.model, lora_dir)
        print(f"  Loaded LoRA from {lora_dir}")
        t2v.model = t2v.model.merge_and_unload()  # Merge for faster inference
        print("  LoRA merged")

    out_path = Path(output_dir) / f"eval_{method}_tau{tau}"
    out_path.mkdir(parents=True, exist_ok=True)

    all_metrics = {"method": method, "tau": tau, "per_video": {}, "per_length": {}}

    for nf in frame_counts:
        label = f"{nf}f"
        extrap = nf / frame_counts[0]
        print(f"\n  --- {nf} frames ({extrap:.1f}x) ---")

        length_metrics = []
        for i, prompt in enumerate(prompts):
            vid_key = f"{label}/video{i}"
            try:
                gen_start = time.time()
                video = t2v.generate(
                    input_prompt=prompt,
                    size=(288, 288),
                    frame_num=nf,
                    sampling_steps=30,  # Faster for eval
                    guide_scale=5.0,
                    seed=seed + i,
                    offload_model=False,
                )
                gen_time = time.time() - gen_start

                # video is a list of PIL images or tensor
                if isinstance(video, list):
                    frames_np = np.stack([np.array(f) for f in video])
                elif isinstance(video, torch.Tensor):
                    # VAE decode returns [-1, 1], convert to [0, 255]
                    # [C,T,H,W] or [T,H,W,C]
                    if video.shape[0] == 3:
                        video = video.permute(1, 2, 3, 0)
                    frames_np = ((video.cpu().float().clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)
                else:
                    print(f"    video {i}: unknown output type {type(video)}")
                    continue

                # Save as frames
                save_dir = out_path / label
                save_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                for j in range(min(len(frames_np), 10)):  # Save first 10 frames
                    Image.fromarray(frames_np[j]).save(save_dir / f"v{i}_f{j:04d}.png")

                # Compute metrics
                m = compute_metrics(frames_np)
                m["gen_time_s"] = gen_time
                m["extrap_ratio"] = extrap
                all_metrics["per_video"][vid_key] = m
                length_metrics.append(m)

                print(f"    video {i}: {len(frames_np)}f, {gen_time:.1f}s, "
                      f"mse={m['temporal_mse']:.4f}, jitter={m['jitter']:.4f}")

            except Exception as e:
                print(f"    video {i}: FAILED ({e})")
                import traceback; traceback.print_exc()
                all_metrics["per_video"][vid_key] = {"error": str(e)}

        # Aggregate
        if length_metrics:
            agg = {}
            for k in length_metrics[0]:
                if isinstance(length_metrics[0][k], (int, float)):
                    vals = [m[k] for m in length_metrics if k in m]
                    agg[f"{k}_mean"] = sum(vals) / len(vals)
            agg["n_videos"] = len(length_metrics)
            agg["extrap_ratio"] = extrap
            all_metrics["per_length"][label] = agg

    # Summary
    print(f"\n{'='*60}")
    print(f"  EXTRAPOLATION SUMMARY: {method.upper()} tau={tau}")
    print(f"{'='*60}")
    print(f"  {'Length':>8s}  {'MSE':>8s}  {'Jitter':>8s}  {'cos_d1':>8s}")
    base_label = f"{frame_counts[0]}f"
    for nf in frame_counts:
        label = f"{nf}f"
        if label in all_metrics["per_length"]:
            lm = all_metrics["per_length"][label]
            print(f"  {label:>8s}  {lm.get('temporal_mse_mean',0):8.4f}  "
                  f"{lm.get('jitter_mean',0):8.4f}  {lm.get('cos_d1_mean',0):8.4f}")

    eval_path = out_path / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: {eval_path}")

    del t2v
    torch.cuda.empty_cache(); gc.collect()
    return all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["geo", "evq", "both"], default="both")
    p.add_argument("--tau", type=float, default=3.2)
    p.add_argument("--num_videos", type=int, default=4)
    p.add_argument("--frame_counts", type=int, nargs="+", default=[81, 161, 241])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prompts = [
        "A boat sailing slowly across a calm lake at sunset",
        "Ocean waves crashing rhythmically onto a rocky shore",
        "A drone flying slowly over a dense tropical rainforest",
        "Colorful ink drops swirling and mixing in clear water",
        "A flag waving steadily in strong wind against a clear sky",
        "Clouds drifting across a blue sky over snow-capped mountains",
        "A candle flame flickering gently in a dark room",
        "Rain falling on a still pond creating expanding ripples",
    ][:args.num_videos]

    methods = [("geo", 0.0), ("evq", args.tau)] if args.method == "both" else \
              [("geo", 0.0)] if args.method == "geo" else [("evq", args.tau)]

    all_results = {}
    for method, tau in methods:
        lora_dir = str(Path(RESULTS_DIR) / f"{method}_tau{tau}" / "lora")
        result = run_eval(method, tau, lora_dir, args.frame_counts, prompts, RESULTS_DIR, args.seed)
        all_results[f"{method}_tau{tau}"] = result

    # Cross-method comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD: TEMPORAL EXTRAPOLATION")
        print(f"{'='*60}")
        keys = list(all_results.keys())
        print(f"  {'Length':>8s}", end="")
        for k in keys:
            name = k.split("_")[0].upper()
            print(f"  {name+'_mse':>10s}  {name+'_jit':>10s}", end="")
        print()
        for nf in args.frame_counts:
            label = f"{nf}f"
            line = f"  {label:>8s}"
            for k in keys:
                pl = all_results[k].get("per_length", {}).get(label, {})
                mse = pl.get("temporal_mse_mean", float('nan'))
                jit = pl.get("jitter_mean", float('nan'))
                line += f"  {mse:10.4f}  {jit:10.4f}"
            print(line)

    with open(Path(RESULTS_DIR) / "eval_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full comparison: {RESULTS_DIR}/eval_comparison.json")


if __name__ == "__main__":
    main()
