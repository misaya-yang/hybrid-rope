#!/usr/bin/env python3
"""Prepare training data for Wan2.1 EVQ fine-tuning.

Downloads real videos and encodes them through Wan2.1's T5 + VAE.
Outputs: latents.pt + text_embeds.pt (ready for training without T5/VAE).

Run on AWS (or any machine with the model + enough RAM):
    python wan21_prepare_data.py \
        --model_dir /path/to/Wan2.1-T2V-1.3B \
        --video_dir /path/to/videos \
        --output_dir /path/to/encoded_data \
        --num_samples 500

If --video_dir is empty, downloads a small subset of Pexels/Mixkit free videos.
"""
import os, sys, gc, json, math, time, argparse, subprocess
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np


# ============================================================
# Download free CC0 videos (no auth needed)
# ============================================================

# Mixkit free stock videos (CC0, direct download)
FREE_VIDEO_URLS = [
    ("https://assets.mixkit.co/videos/1171/1171-720.mp4", "Aerial view of ocean waves hitting rocky shore"),
    ("https://assets.mixkit.co/videos/4367/4367-720.mp4", "Sun setting behind mountains and clouds"),
    ("https://assets.mixkit.co/videos/4640/4640-720.mp4", "Stars moving in night sky time lapse"),
    ("https://assets.mixkit.co/videos/3784/3784-720.mp4", "Raindrops falling on green leaves"),
    ("https://assets.mixkit.co/videos/4898/4898-720.mp4", "Candle flame flickering in darkness"),
    ("https://assets.mixkit.co/videos/2611/2611-720.mp4", "Waves crashing on sandy beach at sunset"),
    ("https://assets.mixkit.co/videos/3117/3117-720.mp4", "Person walking through autumn forest"),
    ("https://assets.mixkit.co/videos/1234/1234-720.mp4", "Smoke rising from incense stick"),
    ("https://assets.mixkit.co/videos/5021/5021-720.mp4", "Traffic moving on city highway at night"),
    ("https://assets.mixkit.co/videos/4690/4690-720.mp4", "Clouds moving over mountain landscape"),
    ("https://assets.mixkit.co/videos/2551/2551-720.mp4", "School of fish swimming underwater"),
    ("https://assets.mixkit.co/videos/3124/3124-720.mp4", "Waterfall flowing in tropical forest"),
    ("https://assets.mixkit.co/videos/5198/5198-720.mp4", "Snow falling on pine trees in winter"),
    ("https://assets.mixkit.co/videos/4873/4873-720.mp4", "Lightning striking during thunderstorm"),
    ("https://assets.mixkit.co/videos/2790/2790-720.mp4", "Cat stretching and yawning on sofa"),
    ("https://assets.mixkit.co/videos/3483/3483-720.mp4", "Hot air balloons rising at dawn"),
    ("https://assets.mixkit.co/videos/5303/5303-720.mp4", "Bees collecting pollen from flowers"),
    ("https://assets.mixkit.co/videos/4012/4012-720.mp4", "River flowing through rocky canyon"),
    ("https://assets.mixkit.co/videos/2971/2971-720.mp4", "Person skateboarding in empty parking lot"),
    ("https://assets.mixkit.co/videos/5453/5453-720.mp4", "Jellyfish floating in dark ocean water"),
]


def download_videos(output_dir: str, num: int = 20):
    """Download free CC0 videos from Mixkit."""
    vdir = Path(output_dir) / "raw_videos"
    vdir.mkdir(parents=True, exist_ok=True)

    captions = {}
    downloaded = 0

    for i, (url, caption) in enumerate(FREE_VIDEO_URLS[:num]):
        out_path = vdir / f"video_{i:04d}.mp4"
        if out_path.exists():
            captions[out_path.name] = caption
            downloaded += 1
            continue

        try:
            print(f"  Downloading {i+1}/{num}: {caption[:50]}...")
            subprocess.run(
                ["wget", "-q", "-O", str(out_path), url],
                timeout=60, check=True
            )
            captions[out_path.name] = caption
            downloaded += 1
        except Exception as e:
            print(f"    Failed: {e}")

    # Save captions
    with open(vdir / "captions.json", "w") as f:
        json.dump(captions, f, indent=2)

    print(f"  Downloaded {downloaded}/{num} videos to {vdir}")
    return vdir, captions


# ============================================================
# Encode videos through Wan2.1 VAE + T5
# ============================================================

def encode_dataset(
    model_dir: str,
    video_dir: str,
    output_dir: str,
    num_frames: int = 81,  # Wan2.1 default (4n+1)
    height: int = 480,
    width: int = 832,  # Wan2.1 default for 1.3B
    device: str = "cuda",
):
    """Encode videos → VAE latents, text → T5 embeddings."""

    # Try multiple possible locations for Wan2.1 repo
    for candidate in [
        Path(model_dir) / "Wan2.1",        # model_dir/Wan2.1
        Path(model_dir).parent / "Wan2.1",  # model_dir/../Wan2.1
        Path(model_dir) / ".." / "Wan2.1",
    ]:
        if (candidate / "wan").exists():
            sys.path.insert(0, str(candidate.resolve()))
            break
    else:
        wan_repo = os.environ.get("WAN_REPO", str(Path(model_dir) / "Wan2.1"))
        sys.path.insert(0, wan_repo)

    from wan.modules.vae import WanVAE
    from wan.modules.t5 import T5EncoderModel
    from wan.configs.wan_t2v_1_3B import t2v_1_3B as cfg

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if (out_path / "latents.pt").exists() and (out_path / "text_embeds.pt").exists():
        print(f"[Encode] Cache exists at {out_path}, skipping")
        return

    # Load captions
    captions_path = Path(video_dir) / "captions.json"
    if captions_path.exists():
        with open(captions_path) as f:
            captions = json.load(f)
    else:
        captions = {}

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"[Encode] Found {len(video_files)} videos")

    # --- T5 encoding (GPU) ---
    print("[Encode] Loading T5 encoder (GPU)...")
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device(device),
        checkpoint_path=os.path.join(model_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(model_dir, cfg.t5_tokenizer),
    )

    all_text_embeds = []
    for vf in video_files:
        caption = captions.get(vf.name, "A video clip")
        with torch.no_grad():
            embed = t5([caption], torch.device(device))
        all_text_embeds.append(embed[0].cpu())  # [text_len, dim]

    print(f"  Encoded {len(all_text_embeds)} text prompts")

    # Free T5
    del t5
    gc.collect()

    # --- VAE encoding (GPU) ---
    print(f"[Encode] Loading VAE to {device}...")
    vae = WanVAE(
        vae_pth=os.path.join(model_dir, cfg.vae_checkpoint),
        device=torch.device(device),
    )

    try:
        import decord
        decord.bridge.set_bridge("torch")
    except ImportError:
        print("[ERROR] decord needed: pip install decord")
        sys.exit(1)

    all_latents = []
    vae_stride = cfg.vae_stride  # (4, 8, 8)

    for i, vf in enumerate(video_files):
        try:
            vr = decord.VideoReader(str(vf))
            total = len(vr)

            # Sample num_frames evenly
            if total >= num_frames:
                indices = np.linspace(0, total - 1, num_frames, dtype=int)
            else:
                indices = np.arange(total)
                pad = np.full(num_frames - total, total - 1, dtype=int)
                indices = np.concatenate([indices, pad])

            frames = vr.get_batch(indices.tolist())  # [T, H, W, C] uint8
            frames = frames.float() / 255.0  # [T, H, W, C]
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]

            # Resize
            from torch.nn.functional import interpolate
            frames = frames.unsqueeze(0)  # [1, C, T, H, W]
            # Resize spatial dims
            B, C, T, H, W = frames.shape
            frames = frames.reshape(B * C, T, H, W).permute(0, 2, 1, 3)  # hack for spatial resize
            # Actually, let's do it properly
            frames = frames.permute(0, 2, 1, 3)  # back
            frames = frames.reshape(B, C, T, H, W)

            # Simple approach: resize frame by frame
            frames_resized = []
            for t in range(T):
                f = frames[0, :, t, :, :]  # [C, H, W]
                f = interpolate(f.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
                frames_resized.append(f.squeeze(0))
            frames = torch.stack(frames_resized, dim=1).unsqueeze(0)  # [1, C, T, H, W]

            # Normalize to [-1, 1]
            frames = frames * 2.0 - 1.0
            frames = frames.to(device=device, dtype=torch.float32)

            # VAE encode
            with torch.no_grad():
                # WanVAE.encode expects [B, C, T, H, W]
                latent = vae.encode([frames.squeeze(0)])[0]  # [C_z, T', H', W']

            all_latents.append(latent.cpu().float())

            if (i + 1) % 5 == 0:
                print(f"  Encoded {i+1}/{len(video_files)} videos, latent shape: {latent.shape}")
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed {vf.name}: {e}")
            import traceback; traceback.print_exc()
            continue

    if len(all_latents) == 0:
        print("[ERROR] No videos encoded successfully!")
        sys.exit(1)

    # Stack and save
    # Note: latents may have different shapes if videos differ in length
    # For simplicity, only keep latents with matching shape
    target_shape = all_latents[0].shape
    matched = [l for l in all_latents if l.shape == target_shape]
    matched_embeds = [all_text_embeds[i] for i, l in enumerate(all_latents) if l.shape == target_shape]

    print(f"[Encode] {len(matched)}/{len(all_latents)} videos matched shape {target_shape}")

    latents_tensor = torch.stack(matched)
    # Pad text embeddings to max length (T5 outputs variable length)
    max_text_len = max(e.shape[0] for e in matched_embeds)
    text_dim = matched_embeds[0].shape[1]
    padded_embeds = []
    for e in matched_embeds:
        if e.shape[0] < max_text_len:
            pad = torch.zeros(max_text_len - e.shape[0], text_dim, dtype=e.dtype)
            e = torch.cat([e, pad], dim=0)
        padded_embeds.append(e)
    text_tensor = torch.stack(padded_embeds)

    torch.save(latents_tensor, out_path / "latents.pt")
    torch.save(text_tensor, out_path / "text_embeds.pt")

    # Save metadata
    meta = {
        "num_samples": len(matched),
        "latent_shape": list(target_shape),
        "text_shape": list(text_tensor.shape[1:]),
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "vae_stride": list(vae_stride),
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Encode] Saved to {out_path}:")
    print(f"  latents: {latents_tensor.shape} ({latents_tensor.numel()*4/1e9:.2f} GB)")
    print(f"  text_embeds: {text_tensor.shape}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wan2.1 training data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Wan2.1-T2V-1.3B directory")
    parser.add_argument("--video_dir", type=str, default="",
                        help="Directory with .mp4 files (will download if empty)")
    parser.add_argument("--output_dir", type=str, default="data/wan21_encoded")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Number of videos to download (if --video_dir empty)")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Download videos if needed
    if not args.video_dir or not Path(args.video_dir).exists():
        print("[1/2] Downloading free CC0 videos...")
        video_dir, _ = download_videos(args.output_dir, num=args.num_videos)
        args.video_dir = str(video_dir)

    # Encode
    print("[2/2] Encoding videos through T5 + VAE...")
    encode_dataset(
        model_dir=args.model_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        device=args.device,
    )

    print("\nDone! Transfer the output directory to GPU server for training.")


if __name__ == "__main__":
    main()
