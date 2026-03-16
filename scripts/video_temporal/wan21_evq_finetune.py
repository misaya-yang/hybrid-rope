#!/usr/bin/env python3
"""
Wan2.1-T2V-1.3B EVQ-Cosh Temporal RoPE Fine-tuning Experiment

Only modifies the TEMPORAL dimension (22 frequency pairs, t_dim=44) of the 3D RoPE.
Spatial dimensions (height/width, 42 dims each) remain untouched.

Architecture (verified from HF config):
  Wan2.1-T2V-1.3B: 30 layers, 12 heads, head_dim=128, hidden=1536, ffn=8960
  3D RoPE split: t_dim=44 (22 pairs), h_dim=42 (21 pairs), w_dim=42 (21 pairs)
  theta=10000, patch_size=(1,2,2), VAE: 4x temporal + 8x spatial

Dead channel analysis (base=10000, T=13):
  10/22 temporal channels dead/near-dead (45.5%)
  EVQ redistributes frequencies away from dead zones

Head-to-head design:
  GEO baseline (tau=0) and EVQ run on SAME data, SAME seed, in SAME script.
  This eliminates CUDA non-determinism per REPORT_FINAL guidelines.

Hardware target: RTX 6000 Pro (96GB), bfloat16
Estimated VRAM: ~25-35GB for 1.3B LoRA + grad checkpointing

Usage:
  python wan21_evq_finetune.py --model_path /path/to/Wan2.1-T2V-1.3B-Diffusers --data_dir /path/to/videos
  python wan21_evq_finetune.py --model_path /path/to/model --pilot  # 5-step verify
"""

import os
import sys
import gc
import json
import math
import time
import argparse
from pathlib import Path
from typing import Optional

# CUDA config BEFORE torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# EVQ-Cosh frequency computation (from AIHANDOFF Part 2)
# ============================================================

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    """Compute EVQ-Cosh inverse frequencies.

    Args:
        dim: Number of dimensions (NOT pairs). E.g., t_dim=44 means 22 pairs.
        tau: Concentration parameter. tau=0 -> geometric.
        base: RoPE base frequency.

    Returns:
        inv_freq tensor of shape [dim//2] in float32 (computed in float64).
    """
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)  # midpoint quantiles per AIHANDOFF

    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))

    inv_freq = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    return inv_freq.float()


def geometric_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Standard geometric (GEO) inverse frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return freqs.float()


# ============================================================
# Patch Wan2.1 WanRotaryPosEmbed temporal frequencies
# ============================================================

def patch_wan_rope_temporal(rope_module, temporal_inv_freq: torch.Tensor, method_name: str = "EVQ"):
    """Replace temporal portion of WanRotaryPosEmbed's precomputed cos/sin buffers.

    WanRotaryPosEmbed stores:
      freqs_cos: [max_seq_len, t_dim + h_dim + w_dim]  (all concatenated)
      freqs_sin: [max_seq_len, t_dim + h_dim + w_dim]

    The first t_dim columns are temporal. We replace ONLY those columns
    with cos/sin from our custom inv_freq values.

    Args:
        rope_module: WanRotaryPosEmbed module instance
        temporal_inv_freq: Tensor of shape [t_dim // 2] with custom frequencies
        method_name: Name for logging
    """
    t_dim = rope_module.t_dim  # 44 for 1.3B
    max_seq_len = rope_module.max_seq_len  # 1024

    K_t = t_dim // 2  # 22 pairs
    assert temporal_inv_freq.shape[0] == K_t, \
        f"Expected {K_t} temporal freqs, got {temporal_inv_freq.shape[0]}"

    # Compute positions [0, 1, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, dtype=torch.float64)

    # Compute raw frequencies: [max_seq_len, K_t]
    inv_freq_64 = temporal_inv_freq.double()
    freqs_raw = torch.outer(positions, inv_freq_64)  # [S, K_t]

    # repeat_interleave(2) to match Wan2.1 convention: [S, t_dim]
    freqs_cos_new = freqs_raw.cos().repeat_interleave(2, dim=1).float()
    freqs_sin_new = freqs_raw.sin().repeat_interleave(2, dim=1).float()

    # Replace temporal columns (first t_dim columns) in existing buffers
    # Keep spatial columns (h_dim + w_dim) untouched
    freqs_cos = rope_module.freqs_cos.clone()
    freqs_sin = rope_module.freqs_sin.clone()

    freqs_cos[:, :t_dim] = freqs_cos_new
    freqs_sin[:, :t_dim] = freqs_sin_new

    # Update buffers
    rope_module.freqs_cos = freqs_cos
    rope_module.freqs_sin = freqs_sin

    print(f"  [{method_name}] Patched temporal RoPE: {K_t} frequency pairs")
    print(f"  [{method_name}] inv_freq range: [{temporal_inv_freq.min():.6f}, {temporal_inv_freq.max():.6f}]")


# ============================================================
# Dataset: load real video latents encoded through VAE
# ============================================================

class VideoLatentDataset(Dataset):
    """Dataset of pre-encoded video latents + text embeddings.

    Expects a cache directory with:
      latents.pt:     [N, C, T, H, W] float32 tensor
      text_embeds.pt: [N, seq_len, dim] float32 tensor

    Created by encode_videos() below.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.latents = None
        self.text_embeds = None

    def load(self) -> bool:
        latents_path = self.cache_dir / "latents.pt"
        text_path = self.cache_dir / "text_embeds.pt"
        if latents_path.exists() and text_path.exists():
            self.latents = torch.load(latents_path, weights_only=True)
            self.text_embeds = torch.load(text_path, weights_only=True)
            print(f"[Dataset] Loaded {len(self.latents)} samples from {self.cache_dir}")
            print(f"  latents shape: {self.latents.shape}")
            print(f"  text_embeds shape: {self.text_embeds.shape}")
            return True
        return False

    def __len__(self):
        return len(self.latents) if self.latents is not None else 0

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "text_embeds": self.text_embeds[idx],
        }


def encode_videos(
    pipe,
    video_dir: str,
    cache_dir: str,
    num_samples: int = 200,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
):
    """Encode real videos through Wan2.1's VAE to create training latents.

    If video_dir contains .mp4 files, encode them.
    Otherwise, use the pipeline to generate a self-distillation dataset
    (generate videos with the base model, encode them back).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    latents_path = cache_path / "latents.pt"
    text_path = cache_path / "text_embeds.pt"

    if latents_path.exists() and text_path.exists():
        print(f"[Dataset] Cache found at {cache_path}")
        return

    video_files = []
    if video_dir and Path(video_dir).exists():
        video_files = sorted(Path(video_dir).glob("*.mp4"))
        print(f"[Dataset] Found {len(video_files)} videos in {video_dir}")

    device = torch.device("cuda")
    vae = pipe.vae.to(device)

    if len(video_files) > 0:
        # Encode real videos
        _encode_real_videos(vae, pipe, video_files, cache_path, num_samples,
                           num_frames, height, width, device)
    else:
        # Self-distillation: generate then encode
        print(f"[Dataset] No videos found. Using self-distillation (generate + encode)...")
        _self_distill(pipe, cache_path, num_samples, num_frames, height, width, device)

    vae.cpu()
    torch.cuda.empty_cache()
    gc.collect()


def _encode_real_videos(vae, pipe, video_files, cache_path, num_samples,
                        num_frames, height, width, device):
    """Encode real .mp4 files through VAE."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
    except ImportError:
        print("[ERROR] decord not installed. Run: pip install decord --break-system-packages")
        sys.exit(1)

    all_latents = []
    all_text_embeds = []

    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer

    prompts = [
        "A video clip",  # Generic prompt for real video encoding
    ]

    # Encode a single text prompt (reused for all videos)
    with torch.no_grad():
        tokens = tokenizer(
            prompts[0], padding="max_length",
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512,
            truncation=True, return_tensors="pt",
        ).to(device)
        base_text_embed = text_encoder(tokens.input_ids)[0].cpu()  # [1, seq_len, dim]

    text_encoder.cpu()
    torch.cuda.empty_cache()

    for i, vf in enumerate(video_files[:num_samples]):
        try:
            vr = decord.VideoReader(str(vf))
            total_frames = len(vr)

            # Sample num_frames evenly
            if total_frames >= num_frames:
                indices = torch.linspace(0, total_frames - 1, num_frames).long()
            else:
                # Pad by repeating last frame
                indices = torch.arange(total_frames)
                pad = torch.full((num_frames - total_frames,), total_frames - 1, dtype=torch.long)
                indices = torch.cat([indices, pad])

            frames = vr.get_batch(indices.tolist())  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

            # Resize to target resolution
            frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)

            # Normalize to [-1, 1]
            frames = frames * 2.0 - 1.0

            # Add batch dim: [1, C, T, H, W]
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                latent = vae.encode(frames).latent_dist.sample()
                latent = latent * vae.config.scaling_factor

            all_latents.append(latent.cpu().float())
            all_text_embeds.append(base_text_embed)

            if (i + 1) % 10 == 0:
                print(f"  Encoded {i+1}/{min(len(video_files), num_samples)} videos")

        except Exception as e:
            print(f"  Warning: Failed to encode {vf.name}: {e}")
            continue

    if len(all_latents) == 0:
        print("[ERROR] No videos could be encoded. Falling back to self-distillation.")
        _self_distill_latents(vae, pipe, cache_path, num_samples, device)
        return

    all_latents = torch.cat(all_latents, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0).expand(len(all_latents), -1, -1)

    torch.save(all_latents, cache_path / "latents.pt")
    torch.save(all_text_embeds, cache_path / "text_embeds.pt")
    print(f"[Dataset] Saved {len(all_latents)} encoded videos to {cache_path}")


def _self_distill(pipe, cache_path, num_samples, num_frames, height, width, device):
    """Generate videos with the base model, encode back to latents."""
    prompts_pool = [
        "A cat walking gracefully across a garden",
        "Ocean waves rolling onto a sandy beach",
        "A person riding a bicycle through a park",
        "Fireworks lighting up the night sky",
        "A dog playing with a ball in the yard",
        "Rain falling on a quiet city street",
        "Flowers blooming in a time-lapse sequence",
        "Birds flying across a golden sunset",
        "A river flowing through a dense forest",
        "Clouds drifting across a blue sky",
        "A candle flame flickering in the wind",
        "Leaves falling from autumn trees",
        "A train moving along mountain tracks",
        "Fish swimming in clear ocean water",
        "Snow falling on a peaceful village",
    ]

    pipe.to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    all_latents = []
    all_text_embeds = []

    print(f"[Self-distill] Generating {num_samples} videos for training data...")

    for i in range(num_samples):
        prompt = prompts_pool[i % len(prompts_pool)]

        # Encode text
        with torch.no_grad():
            tokens = tokenizer(
                prompt, padding="max_length",
                max_length=getattr(tokenizer, 'model_max_length', 512),
                truncation=True, return_tensors="pt",
            ).to(device)
            text_embed = text_encoder(tokens.input_ids)[0].cpu()

        # Generate video
        generator = torch.Generator(device).manual_seed(42 + i)
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=20,  # Fewer steps for speed (just need data)
                generator=generator,
                output_type="latent",  # Get latents directly, skip VAE decode
            )

        if hasattr(output, 'frames'):
            # If output_type="latent" returns latents directly
            latent = output.frames if isinstance(output.frames, torch.Tensor) else output.frames[0]
        else:
            latent = output[0]

        if isinstance(latent, torch.Tensor):
            all_latents.append(latent.cpu().float())
        else:
            # Fallback: generate pixel output and encode through VAE
            output = pipe(
                prompt=prompt, num_frames=num_frames, height=height, width=width,
                num_inference_steps=20, generator=torch.Generator(device).manual_seed(42 + i),
            )
            frames = output.frames[0]  # list of PIL images
            # Convert to tensor and encode
            import torchvision.transforms as T
            transform = T.Compose([T.Resize((height, width)), T.ToTensor(), T.Normalize([0.5], [0.5])])
            frames_tensor = torch.stack([transform(f) for f in frames])  # [T, C, H, W]
            frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = vae.encode(frames_tensor).latent_dist.sample() * vae.config.scaling_factor
            all_latents.append(latent.cpu().float())

        all_text_embeds.append(text_embed)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{num_samples}")
            torch.cuda.empty_cache()

    all_latents = torch.cat(all_latents, dim=0) if all_latents[0].dim() == 4 else torch.stack(all_latents)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    torch.save(all_latents, cache_path / "latents.pt")
    torch.save(all_text_embeds, cache_path / "text_embeds.pt")
    print(f"[Dataset] Saved {len(all_latents)} self-distilled samples to {cache_path}")

    pipe.cpu()
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Training loop (head-to-head)
# ============================================================

def train_lora(
    pipe,
    method: str,  # "geo" or "evq"
    tau: float = 3.2,
    theta_t: float = 10000.0,
    num_steps: int = 500,
    lora_rank: int = 16,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    output_dir: str = "results",
    dataset: VideoLatentDataset = None,
    use_compile: bool = True,
    seed: int = 42,
):
    """Fine-tune Wan2.1 transformer with LoRA, using specified temporal RoPE."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  Wan2.1-1.3B LoRA Fine-tuning: {method.upper()}")
    print(f"  tau={tau}, theta_t={theta_t}, steps={num_steps}")
    print(f"  LoRA rank={lora_rank}, lr={lr}, bs={batch_size}x{grad_accum}")
    print(f"{'='*60}\n")

    transformer = pipe.transformer

    # ---- Compute and patch temporal inv_freq ----
    t_dim = transformer.rope.t_dim  # 44

    if method == "evq":
        temporal_inv_freq = evq_cosh_inv_freq(t_dim, tau=tau, base=theta_t)
        patch_wan_rope_temporal(transformer.rope, temporal_inv_freq, f"EVQ(τ={tau})")
    else:
        temporal_inv_freq = geometric_inv_freq(t_dim, base=theta_t)
        patch_wan_rope_temporal(transformer.rope, temporal_inv_freq, "GEO")

    # ---- Apply LoRA ----
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    # Remove any existing LoRA before applying new one
    if hasattr(transformer, 'peft_config'):
        transformer = transformer.base_model.model

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device)
    transformer.train()

    # Move RoPE buffers to device
    transformer.base_model.model.rope.freqs_cos = transformer.base_model.model.rope.freqs_cos.to(device)
    transformer.base_model.model.rope.freqs_sin = transformer.base_model.model.rope.freqs_sin.to(device)

    # Enable gradient checkpointing
    if hasattr(transformer, 'enable_gradient_checkpointing'):
        transformer.enable_gradient_checkpointing()

    # ---- torch.compile (optional) ----
    compile_model = None
    if use_compile:
        print("[Compile] Compiling model with torch.compile(mode='default')...")
        compile_model = torch.compile(transformer, mode="default")
    else:
        compile_model = transformer

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    # ---- Scheduler for noise ----
    noise_scheduler = pipe.scheduler

    # ---- DataLoader ----
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Training ----
    out_path = Path(output_dir) / f"wan21_{method}_tau{tau}_base{int(theta_t)}"
    out_path.mkdir(parents=True, exist_ok=True)

    step = 0
    losses = []
    start_time = time.time()

    print(f"[Train] Starting {num_steps} steps...")

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            if use_compile:
                torch.compiler.cudagraph_mark_step_begin()

            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            text_embeds = batch["text_embeds"].to(device=device, dtype=torch.bfloat16)

            bsz = latents.shape[0]

            # Sample random timesteps (flow matching: t in [0, 1])
            # Wan2.1 uses FlowMatchEulerDiscreteScheduler
            sigmas = torch.rand(bsz, device=device, dtype=torch.float32)
            timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).long()

            # Add noise: noisy = (1-sigma) * latents + sigma * noise
            noise = torch.randn_like(latents)
            sigmas_bc = sigmas.view(bsz, 1, 1, 1, 1).to(dtype=latents.dtype)
            noisy_latents = (1 - sigmas_bc) * latents + sigmas_bc * noise

            # Forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_output = compile_model(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=text_embeds,
                    timestep=timesteps.float(),
                    return_dict=False,
                )[0]

                # Flow matching loss: predict velocity v = noise - latents
                target = noise - latents
                loss = F.mse_loss(model_output, target) / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or step == num_steps - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_val = loss.item() * grad_accum
            losses.append(loss_val)

            if step % 50 == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step {step}/{num_steps}  loss={loss_val:.6f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"VRAM={vram:.1f}GB  elapsed={elapsed:.0f}s")

            step += 1

    total_time = time.time() - start_time
    avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)

    print(f"\n[Done] {method.upper()}: {num_steps} steps in {total_time:.0f}s")
    print(f"  Average loss (last 100): {avg_loss:.6f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    # ---- Save results ----
    result = {
        "method": method,
        "tau": tau,
        "theta_t": theta_t,
        "num_steps": num_steps,
        "lora_rank": lora_rank,
        "final_loss": avg_loss,
        "all_losses": losses,
        "total_time_s": total_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "temporal_inv_freq": temporal_inv_freq.tolist(),
        "seed": seed,
    }

    with open(out_path / "train_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save LoRA weights
    lora_path = out_path / "lora_weights"
    transformer.save_pretrained(str(lora_path))
    print(f"  LoRA weights: {lora_path}")

    # ---- Clean up: unwrap LoRA for next run ----
    unwrapped = transformer.base_model.model
    pipe.transformer = unwrapped
    del transformer, compile_model
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    pipe,
    lora_dir: str,
    method: str,
    tau: float,
    theta_t: float,
    output_dir: str,
    num_videos: int = 4,
    seed: int = 42,
):
    """Generate videos at standard and extrapolated lengths, measure quality."""
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  Evaluation: {method.upper()} tau={tau} base={int(theta_t)}")
    print(f"{'='*60}\n")

    transformer = pipe.transformer

    # Patch temporal RoPE
    t_dim = transformer.rope.t_dim
    if method == "evq":
        temporal_inv_freq = evq_cosh_inv_freq(t_dim, tau=tau, base=theta_t)
    else:
        temporal_inv_freq = geometric_inv_freq(t_dim, base=theta_t)
    patch_wan_rope_temporal(transformer.rope, temporal_inv_freq, method.upper())

    # Load LoRA
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(transformer, lora_dir)
        print(f"  Loaded LoRA from {lora_dir}")

    pipe.to(device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prompts = [
        "A cat walking gracefully across a garden",
        "Ocean waves rolling onto a sandy beach at sunset",
        "A time-lapse of flowers blooming in spring",
        "A person skateboarding through a city park",
    ][:num_videos]

    results = {"method": method, "tau": tau, "theta_t": theta_t}

    # Standard generation (49 frames)
    print("  Generating standard-length videos (49 frames)...")
    try:
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device).manual_seed(seed + i)
            output = pipe(
                prompt=prompt, num_frames=49, height=480, width=720,
                num_inference_steps=50, generator=generator,
            )
            # Save frames for FVD computation
            frames = output.frames[0]  # list of PIL images
            save_dir = out_path / f"videos_{method}" / f"standard_49f"
            save_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(frames):
                frame.save(save_dir / f"video{i}_frame{j:04d}.png")
            print(f"    Video {i}: {len(frames)} frames")
        results["standard_49f"] = "completed"
    except Exception as e:
        print(f"    Standard generation failed: {e}")
        results["standard_49f"] = f"failed: {e}"

    # Extrapolated generation (97 frames = ~2x temporal)
    print("  Generating extrapolated videos (97 frames)...")
    try:
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device).manual_seed(seed + i)
            output = pipe(
                prompt=prompt, num_frames=97, height=480, width=720,
                num_inference_steps=50, generator=generator,
            )
            frames = output.frames[0]
            save_dir = out_path / f"videos_{method}" / f"extrap_97f"
            save_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(frames):
                frame.save(save_dir / f"video{i}_frame{j:04d}.png")
            print(f"    Video {i}: {len(frames)} frames (extrapolated)")
        results["extrapolated_97f"] = "completed"
    except Exception as e:
        print(f"    Extrapolation failed: {e}")
        results["extrapolated_97f"] = f"failed: {e}"

    with open(out_path / f"eval_{method}_tau{tau}_base{int(theta_t)}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Clean up
    if hasattr(pipe.transformer, 'base_model'):
        pipe.transformer = pipe.transformer.base_model.model
    pipe.cpu()
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B EVQ Fine-tuning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path to Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory of .mp4 video files (optional, will self-distill if empty)")
    parser.add_argument("--method", type=str, choices=["geo", "evq", "both"], default="both",
                        help="Which method to run. 'both' = head-to-head comparison")
    parser.add_argument("--tau", type=float, default=3.2,
                        help="EVQ tau parameter (default: 3.2 = τ*_DiT for Wan2.1)")
    parser.add_argument("--theta_t", type=float, default=10000.0,
                        help="Temporal RoPE base frequency")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/wan21_evq")
    parser.add_argument("--cache_dir", type=str, default="data/wan21_cache")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="5 steps to verify setup")
    parser.add_argument("--num_data", type=int, default=200,
                        help="Number of training samples to encode/generate")

    args = parser.parse_args()

    if args.pilot:
        args.num_steps = 5
        args.skip_eval = True
        args.num_data = 10
        print("\n*** PILOT MODE: 5 steps, 10 samples, no eval ***\n")

    # ---- Load pipeline ----
    print("[1/5] Loading Wan2.1-T2V-1.3B pipeline...")
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    # Print architecture info
    transformer = pipe.transformer
    t_dim = transformer.rope.t_dim
    h_dim = transformer.rope.h_dim
    w_dim = transformer.rope.w_dim
    print(f"  Architecture: {sum(p.numel() for p in transformer.parameters())/1e6:.1f}M params")
    print(f"  3D RoPE: t_dim={t_dim} ({t_dim//2} pairs), h_dim={h_dim} ({h_dim//2} pairs), w_dim={w_dim} ({w_dim//2} pairs)")
    print(f"  head_dim={transformer.rope.attention_head_dim}, theta=10000")

    # τ* calculation
    K_t = t_dim // 2
    T_est = 13  # 49 frames / 4x VAE compression ≈ 13
    tau_star_ar = K_t / math.sqrt(T_est)
    tau_star_dit = 0.53 * tau_star_ar
    print(f"\n  τ* estimates (T_train≈{T_est}, K_t={K_t}):")
    print(f"    τ*_AR  = {tau_star_ar:.3f}")
    print(f"    τ*_DiT = {tau_star_dit:.3f} (γ=0.53)")

    # ---- Prepare dataset ----
    if not args.eval_only:
        print(f"\n[2/5] Preparing dataset (cache: {args.cache_dir})...")
        encode_videos(
            pipe, args.data_dir, args.cache_dir,
            num_samples=args.num_data,
        )
        dataset = VideoLatentDataset(args.cache_dir)
        if not dataset.load():
            print("[ERROR] Failed to load dataset. Check cache_dir.")
            sys.exit(1)
    else:
        dataset = None

    # ---- Run experiments ----
    methods = ["geo", "evq"] if args.method == "both" else [args.method]
    all_results = {}

    for method in methods:
        tau = 0.0 if method == "geo" else args.tau

        if not args.eval_only:
            print(f"\n[3/5] Training {method.upper()}...")
            # Reload fresh transformer for each method (clean LoRA state)
            if method != methods[0]:
                # Reload pipeline to get clean transformer
                del pipe
                torch.cuda.empty_cache()
                gc.collect()
                pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

            result = train_lora(
                pipe=pipe,
                method=method,
                tau=tau,
                theta_t=args.theta_t,
                num_steps=args.num_steps,
                lora_rank=args.lora_rank,
                lr=args.lr,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                output_dir=args.output_dir,
                dataset=dataset,
                use_compile=not args.no_compile,
                seed=args.seed,
            )
            all_results[f"{method}_tau{tau}_base{int(args.theta_t)}"] = result

        if not args.skip_eval:
            lora_dir = str(Path(args.output_dir) /
                          f"wan21_{method}_tau{tau}_base{int(args.theta_t)}" / "lora_weights")
            evaluate(
                pipe=pipe,
                lora_dir=lora_dir,
                method=method,
                tau=tau,
                theta_t=args.theta_t,
                output_dir=args.output_dir,
                seed=args.seed,
            )

    # ---- Summary ----
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"{'='*60}")
        geo_key = [k for k in all_results if k.startswith("geo")][0]
        evq_key = [k for k in all_results if k.startswith("evq")][0]
        geo_loss = all_results[geo_key]["final_loss"]
        evq_loss = all_results[evq_key]["final_loss"]
        delta = (evq_loss - geo_loss) / geo_loss * 100

        print(f"  GEO: loss={geo_loss:.6f}")
        print(f"  EVQ: loss={evq_loss:.6f}")
        print(f"  Delta: {delta:+.1f}% ({'EVQ wins' if delta < 0 else 'GEO wins'})")
        print(f"{'='*60}")

        for name, r in all_results.items():
            print(f"  {name}: loss={r['final_loss']:.6f}, time={r['total_time_s']:.0f}s, "
                  f"VRAM={r['peak_vram_gb']:.1f}GB")

    summary_path = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results: {summary_path}")


if __name__ == "__main__":
    main()
