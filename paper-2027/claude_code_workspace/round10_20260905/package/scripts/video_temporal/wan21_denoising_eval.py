#!/usr/bin/env python3
"""Wan2.1 denoising MSE evaluation — per-frame temporal decomposition.

Adapted from DiT compute_denoising_precision() in run_dit_temporal.py.
Measures velocity prediction quality at each temporal position.

Pre-requisites (already on server from training run):
  - Model: /root/autodl-tmp/wan21/ (raw WanModel)
  - LoRA: /root/autodl-tmp/results/wan21_evq_288/{geo,evq}_tau*/lora/
  - Cache 81f: /root/autodl-tmp/data/wan21_288/ (latents.pt, text_embeds.pt)
  - Videos: /root/autodl-tmp/data/wan21_encoded/raw_videos/ (for 161f encoding)

Usage:
  python -u wan21_denoising_eval.py           # full eval: 81f + 161f, GEO + EVQ
  python -u wan21_denoising_eval.py --quick   # 81f only, fast sanity check
"""
import os, sys, gc, json, math, time, argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/autodl-tmp/wan21/Wan2.1")

import torch
import torch.nn.functional as F

# ============================================================
# Config (matches training script wan21_raw_train.py)
# ============================================================
MODEL_DIR    = "/root/autodl-tmp/wan21"
CACHE_81F    = "/root/autodl-tmp/data/wan21_288"
CACHE_161F   = "/root/autodl-tmp/data/wan21_288_161f"
VIDEO_DIR    = "/root/autodl-tmp/data/wan21_encoded/raw_videos"
RESULTS_DIR  = "/root/autodl-tmp/results/wan21_evq_288"
HEIGHT, WIDTH = 288, 288
T_DIM        = 44   # 22 temporal freq pairs
K_T          = 22

# ============================================================
# EVQ-Cosh (identical to wan21_raw_train.py)
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

def patch_temporal_freqs(model, inv_freq):
    """Replace temporal columns in model.freqs (complex [1024, 64])."""
    positions = torch.arange(model.freqs.shape[0], dtype=torch.float64)
    raw = torch.outer(positions, inv_freq.double())
    model.freqs[:, :K_T] = torch.polar(torch.ones_like(raw), raw)

# ============================================================
# Encode 161-frame validation latents (reuses training encode logic)
# ============================================================
def encode_161f():
    """Encode same videos at 161 frames for extrapolation eval."""
    cache = Path(CACHE_161F)
    if (cache / "latents.pt").exists() and (cache / "text_embeds.pt").exists():
        print("[Encode 161f] Cache exists, skipping.")
        return
    cache.mkdir(parents=True, exist_ok=True)

    # Reuse text embeddings from 81f cache (same videos, same prompts)
    text_81f = torch.load(Path(CACHE_81F) / "text_embeds.pt", weights_only=True)
    torch.save(text_81f, cache / "text_embeds.pt")
    print(f"[Encode 161f] Copied text embeddings: {text_81f.shape}")

    # Encode videos at 161 frames
    from wan.modules.vae import WanVAE
    import decord
    decord.bridge.set_bridge("torch")

    device = torch.device("cuda")
    num_frames = 161

    print("[Encode 161f] Loading VAE (GPU)...")
    vae = WanVAE(vae_pth=os.path.join(MODEL_DIR, "Wan2.1_VAE.pth"), device=device)

    captions_file = Path(VIDEO_DIR) / "captions.json"
    with open(captions_file) as f:
        captions = json.load(f)
    video_files = sorted(Path(VIDEO_DIR).glob("*.mp4"))[:20]

    all_latents = []
    for i, vf in enumerate(video_files):
        vr = decord.VideoReader(str(vf))
        total = len(vr)
        if total >= num_frames:
            indices = torch.linspace(0, total - 1, num_frames).long().tolist()
        else:
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        frames = vr.get_batch(indices).float() / 255.0  # [T,H,W,C]
        frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]
        frames = F.interpolate(frames, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
        frames = frames.permute(1, 0, 2, 3)  # [C,T,H,W]
        frames = (frames * 2.0 - 1.0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            latent = vae.encode([frames])[0]  # [C,F,H,W]
        all_latents.append(latent.cpu().float())
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Encoded {i+1}/{len(video_files)}: {latent.shape}")

    del vae; torch.cuda.empty_cache(); gc.collect()

    latents = torch.stack(all_latents)
    torch.save(latents, cache / "latents.pt")
    meta = {"n": len(latents), "shape": list(latents.shape), "num_frames": num_frames}
    with open(cache / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Encode 161f] Saved: {latents.shape}")

# ============================================================
# Core: denoising MSE evaluation (adapted from DiT)
# ============================================================
@torch.no_grad()
def compute_denoising_mse(model, latents, text_embeds, noise_level=0.5, device="cuda"):
    """Per-frame denoising MSE — Wan2.1 teacher-forced evaluation.

    Args:
        model: WanModel (with or without LoRA, on device)
        latents: [N, C=16, F, H, W] float32 validation latents
        text_embeds: [N, L, D] float32 text embeddings
        noise_level: sigma for noise injection (0.5 = 50% noise)

    Returns:
        frame_mse: [F] tensor, per-frame average MSE
    """
    model.eval()
    N = latents.shape[0]
    F_lat = latents.shape[2]
    H_lat, W_lat = latents.shape[3], latents.shape[4]

    # seq_len for RoPE indexing (must match training)
    patch_h, patch_w = 2, 2  # Wan2.1 patch_size = (1, 2, 2)
    seq_len = F_lat * (H_lat // patch_h) * (W_lat // patch_w)

    frame_mse_sum = torch.zeros(F_lat, device="cpu")
    frame_count = 0

    sigma = noise_level
    timestep_val = sigma * 1000.0  # Wan2.1 uses 0-1000 range

    for i in range(N):
        clean = latents[i].to(device=device, dtype=torch.bfloat16)  # [C, F, H, W]
        text = text_embeds[i].to(device=device, dtype=torch.bfloat16)  # [L, D]

        noise = torch.randn_like(clean)
        # Flow matching interpolation: x_t = (1-sigma)*clean + sigma*noise
        x_t = (1 - sigma) * clean + sigma * noise
        # Velocity target: v = noise - clean
        v_target = noise - clean

        # Wan2.1 forward: List[Tensor] API
        t_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_pred_list = model([x_t], t_tensor, [text], seq_len)
        v_pred = v_pred_list[0]  # [C, F, H, W]

        # Per-frame squared error: average over C, H, W
        se = (v_pred.float() - v_target.float()).pow(2)  # [C, F, H, W]
        frame_se = se.mean(dim=(0, 2, 3))  # [F]

        frame_mse_sum += frame_se.cpu()
        frame_count += 1

        if (i + 1) % 5 == 0:
            print(f"    Eval {i+1}/{N}, frame_mse_avg={frame_se.mean():.6f}")

    frame_mse = frame_mse_sum / frame_count
    return frame_mse

# ============================================================
# Region decomposition (same as DiT)
# ============================================================
def decompose_regions(frame_mse, train_frames_latent):
    """Split per-frame MSE into train/near/mid/far regions."""
    tf = train_frames_latent
    F_lat = len(frame_mse)

    result = {
        "per_frame_mse": frame_mse.tolist(),
        "train_mse": float(frame_mse[:tf].mean()),
    }
    if F_lat > tf:
        result["all_extrap_mse"] = float(frame_mse[tf:].mean())
    ne = min(2 * tf, F_lat)
    if ne > tf:
        result["near_extrap_mse"] = float(frame_mse[tf:ne].mean())
    me = min(3 * tf, F_lat)
    if me > ne:
        result["mid_extrap_mse"] = float(frame_mse[ne:me].mean())
    if F_lat > me:
        result["far_extrap_mse"] = float(frame_mse[me:].mean())
    return result

# ============================================================
# Run eval for one method
# ============================================================
def eval_method(method, tau, lora_dir, latents, text_embeds, train_F, noise_level, device):
    """Load model, patch RoPE, load LoRA, run denoising eval."""
    print(f"\n{'='*60}")
    print(f"  Denoising eval: {method.upper()} tau={tau}")
    print(f"  Latents: {latents.shape}, train_F={train_F}")
    print(f"{'='*60}")

    from wan.modules.model import WanModel

    # Load fresh model each time (clean state)
    model = WanModel.from_pretrained(MODEL_DIR)
    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Patch temporal RoPE
    if method == "evq":
        inv_freq = evq_cosh_inv_freq(T_DIM, tau=tau)
    else:
        inv_freq = geometric_inv_freq(T_DIM)
    patch_temporal_freqs(model, inv_freq.float())
    model.freqs = model.freqs.to(device)
    print(f"  RoPE patched: {method}, inv_freq range=[{inv_freq.min():.6f}, {inv_freq.max():.6f}]")

    # Load LoRA
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        print(f"  LoRA loaded + merged from {lora_dir}")
    else:
        print(f"  WARNING: No LoRA at {lora_dir}, using base model")

    model.eval()

    # Run denoising eval
    t0 = time.time()
    frame_mse = compute_denoising_mse(model, latents, text_embeds, noise_level, device)
    dt = time.time() - t0

    result = decompose_regions(frame_mse, train_F)
    result["method"] = method
    result["tau"] = tau
    result["noise_level"] = noise_level
    result["eval_time_s"] = dt
    result["n_videos"] = latents.shape[0]
    result["latent_F"] = latents.shape[2]
    result["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n  Results ({dt:.0f}s, VRAM={result['vram_gb']:.1f}GB):")
    print(f"    train_mse:      {result['train_mse']:.6f}")
    if "all_extrap_mse" in result:
        print(f"    all_extrap_mse: {result['all_extrap_mse']:.6f}")
    if "near_extrap_mse" in result:
        print(f"    near_extrap_mse: {result['near_extrap_mse']:.6f}")

    del model; torch.cuda.empty_cache(); gc.collect()
    return result

# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="81f only, skip 161f")
    p.add_argument("--noise_level", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=3.2)
    args = p.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}")

    # ---- Step 1: Load 81f data (already cached from training) ----
    print("\n[1] Loading 81f validation data...")
    lat_81 = torch.load(Path(CACHE_81F) / "latents.pt", weights_only=True)
    txt = torch.load(Path(CACHE_81F) / "text_embeds.pt", weights_only=True)
    train_F = lat_81.shape[2]  # 21
    print(f"  81f latents: {lat_81.shape}, train_F={train_F}")
    print(f"  text: {txt.shape}")

    # ---- Step 2: Encode 161f data (if needed) ----
    lat_161 = None
    if not args.quick:
        print("\n[2] Preparing 161f extrapolation data...")
        encode_161f()
        lat_161 = torch.load(Path(CACHE_161F) / "latents.pt", weights_only=True)
        print(f"  161f latents: {lat_161.shape}, extrap_F={lat_161.shape[2]}")

    # ---- Step 3: Eval GEO + EVQ on 81f ----
    all_results = {}
    for method in ["geo", "evq"]:
        tau = 0.0 if method == "geo" else args.tau
        lora_dir = str(Path(RESULTS_DIR) / f"{method}_tau{tau}" / "lora")

        # 81f (in-distribution)
        key_81 = f"{method}_81f"
        all_results[key_81] = eval_method(
            method, tau, lora_dir, lat_81, txt, train_F, args.noise_level, device)

        # 161f (extrapolation)
        if lat_161 is not None:
            key_161 = f"{method}_161f"
            all_results[key_161] = eval_method(
                method, tau, lora_dir, lat_161, txt, train_F, args.noise_level, device)

    # ---- Step 4: Summary ----
    print(f"\n{'='*60}")
    print(f"  DENOISING MSE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'':>20s}  {'GEO':>10s}  {'EVQ':>10s}  {'Delta':>10s}")

    for region in ["train_mse", "all_extrap_mse", "near_extrap_mse"]:
        geo_81 = all_results.get("geo_81f", {}).get(region)
        evq_81 = all_results.get("evq_81f", {}).get(region)
        if geo_81 is not None and evq_81 is not None:
            delta = (evq_81 - geo_81) / geo_81 * 100
            label = f"81f {region.replace('_mse','')}"
            print(f"  {label:>20s}  {geo_81:10.6f}  {evq_81:10.6f}  {delta:+9.1f}%")

    if lat_161 is not None:
        for region in ["train_mse", "all_extrap_mse", "near_extrap_mse"]:
            geo_161 = all_results.get("geo_161f", {}).get(region)
            evq_161 = all_results.get("evq_161f", {}).get(region)
            if geo_161 is not None and evq_161 is not None:
                delta = (evq_161 - geo_161) / geo_161 * 100
                label = f"161f {region.replace('_mse','')}"
                print(f"  {label:>20s}  {geo_161:10.6f}  {evq_161:10.6f}  {delta:+9.1f}%")

    # Save
    out_path = Path(RESULTS_DIR) / "denoising_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
