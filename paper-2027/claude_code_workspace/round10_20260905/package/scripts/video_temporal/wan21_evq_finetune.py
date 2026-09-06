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

    # NOTE: No gradient checkpointing — it conflicts with torch.compile on Blackwell
    # and slows per-step throughput. We compensate with larger batch_size.

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

    # ---- VRAM auto-scaling (find max batch_size) ----
    if batch_size == 0:
        # Auto mode: profile bs=1 then scale up
        print("[VRAM] Auto-detecting optimal batch_size...")
        torch.cuda.reset_peak_memory_stats()

        sample_lat = dataset[0]["latents"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        sample_txt = dataset[0]["text_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        sample_sig = torch.rand(1, device=device)
        sample_ts = (sample_sig * noise_scheduler.config.num_train_timesteps).long()
        sample_noise = torch.randn_like(sample_lat)
        sig_bc = sample_sig.view(1, 1, 1, 1, 1).to(dtype=sample_lat.dtype)
        sample_noisy = (1 - sig_bc) * sample_lat + sig_bc * sample_noise

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = transformer(hidden_states=sample_noisy, encoder_hidden_states=sample_txt,
                              timestep=sample_ts.float(), return_dict=False)[0]
            probe_loss = F.mse_loss(out, sample_noise - sample_lat)
        probe_loss.backward()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        peak_bs1 = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        baseline = sum(p.nelement() * p.element_size() for p in transformer.parameters()) / 1e9
        act_per_sample = peak_bs1 - baseline
        safety = 4.0  # GB margin
        batch_size = max(1, int((total_vram - safety - baseline) / act_per_sample))
        print(f"  bs=1 peak: {peak_bs1:.1f}GB, act/sample: {act_per_sample:.1f}GB, "
              f"total VRAM: {total_vram:.0f}GB -> auto bs={batch_size}")

        del out, probe_loss, sample_lat, sample_txt, sample_noisy, sample_noise
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- DataLoader ----
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Training ----
    out_path = Path(output_dir) / f"wan21_{method}_tau{tau}_base{int(theta_t)}"
    out_path.mkdir(parents=True, exist_ok=True)

    step = 0
    losses = []
    compile_time = None
    start_time = time.time()

    print(f"[Train] Starting {num_steps} steps (bs={batch_size})...")

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            step_start = time.time()

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

            torch.cuda.synchronize()
            step_time = time.time() - step_start
            loss_val = loss.item() * grad_accum
            losses.append(loss_val)

            if step == 0:
                compile_time = step_time
                print(f"  step 0 (includes compile): {compile_time:.1f}s  loss={loss_val:.6f}  "
                      f"VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            elif step <= 3 or step % 50 == 0 or step == num_steps - 1:
                elapsed = time.time() - start_time
                vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step {step}/{num_steps}  loss={loss_val:.6f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"{step_time:.2f}s/step  VRAM={vram:.1f}GB  elapsed={elapsed:.0f}s")

            step += 1

    total_time = time.time() - start_time
    train_time = total_time - (compile_time or 0)
    avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)

    print(f"\n[Done] {method.upper()}: {num_steps} steps in {total_time:.0f}s "
          f"(compile: {compile_time:.0f}s, train: {train_time:.0f}s)")
    if num_steps > 1:
        print(f"  Avg step time (excl compile): {train_time / (num_steps - 1):.2f}s/step")
    print(f"  Average loss (last 100): {avg_loss:.6f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print(f"  Batch size: {batch_size}")

    # ---- Save results ----
    result = {
        "method": method,
        "tau": tau,
        "theta_t": theta_t,
        "num_steps": num_steps,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "final_loss": avg_loss,
        "all_losses": losses,
        "total_time_s": total_time,
        "compile_time_s": compile_time,
        "train_time_s": train_time,
        "avg_step_time_s": train_time / max(num_steps - 1, 1),
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
# Evaluation: temporal extrapolation quality
# ============================================================

# Diverse prompts covering different motion types
EVAL_PROMPTS = [
    # Smooth continuous motion
    "A boat sailing slowly across a calm lake at sunset",
    "Clouds drifting across a blue sky over snow-capped mountains",
    # Dynamic fast motion
    "A cheetah running at full speed across the African savanna",
    "Fireworks exploding in colorful patterns over a city skyline",
    # Periodic repetitive motion
    "Ocean waves crashing rhythmically onto a rocky shore",
    "A flag waving steadily in strong wind against a clear sky",
    # Complex multi-object
    "Children playing in a park while dogs chase each other",
    "Busy city traffic with cars and pedestrians at an intersection",
    # Camera motion
    "A drone flying slowly over a dense tropical rainforest",
    "Walking through a narrow cobblestone alley in an old European town",
    # Nature organic
    "A flower blooming in time-lapse with petals slowly unfolding",
    "Rain falling on a still pond creating expanding ripples",
    # Human motion
    "A dancer performing a graceful ballet routine on stage",
    "A person riding a bicycle along a tree-lined country road",
    # Abstract
    "Colorful ink drops swirling and mixing in clear water",
    "Northern lights dancing across a dark starry sky over mountains",
]


def compute_video_metrics(frames_tensor: torch.Tensor) -> dict:
    """Compute temporal quality metrics from a video tensor.

    Args:
        frames_tensor: [T, C, H, W] float32 tensor in [0, 1]

    Returns:
        Dict of scalar metrics.
    """
    T = frames_tensor.shape[0]
    if T < 2:
        return {}

    # --- Temporal smoothness (pixel-level) ---
    # MSE between consecutive frames
    diffs = frames_tensor[1:] - frames_tensor[:-1]  # [T-1, C, H, W]
    per_frame_mse = (diffs ** 2).mean(dim=(1, 2, 3))  # [T-1]

    # --- Temporal consistency at multiple distances ---
    # Cosine similarity between frames at distance d (in feature space = flattened pixels)
    frames_flat = frames_tensor.view(T, -1)  # [T, C*H*W]
    norms = frames_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    frames_normed = frames_flat / norms

    consistency_by_distance = {}
    for d in [1, 2, 4, 8, 16]:
        if d >= T:
            break
        cos_sims = (frames_normed[:-d] * frames_normed[d:]).sum(dim=1)  # [T-d]
        consistency_by_distance[f"cos_sim_d{d}"] = cos_sims.mean().item()
        consistency_by_distance[f"cos_sim_d{d}_std"] = cos_sims.std().item()

    # --- Motion magnitude ---
    motion_mag = diffs.abs().mean(dim=(1, 2, 3))  # [T-1]

    # --- Temporal jitter (second-order smoothness) ---
    # High jitter = inconsistent motion (frame-to-frame speed changes)
    if T >= 3:
        accel = per_frame_mse[1:] - per_frame_mse[:-1]
        jitter = accel.abs().mean().item()
    else:
        jitter = 0.0

    return {
        "temporal_mse_mean": per_frame_mse.mean().item(),
        "temporal_mse_std": per_frame_mse.std().item(),
        "motion_magnitude_mean": motion_mag.mean().item(),
        "motion_magnitude_std": motion_mag.std().item(),
        "temporal_jitter": jitter,
        **consistency_by_distance,
    }


def compute_clip_metrics(frames_pil: list, prompt: str, clip_model, clip_processor, device) -> dict:
    """Compute CLIP-based temporal quality metrics.

    Args:
        frames_pil: List of PIL images
        prompt: Text prompt
        clip_model: CLIPModel instance
        clip_processor: CLIPProcessor instance
        device: torch device

    Returns:
        Dict of CLIP metrics.
    """
    import numpy as np

    T = len(frames_pil)
    if T < 2:
        return {}

    # Encode all frames in batches of 16
    all_feats = []
    for i in range(0, T, 16):
        batch_frames = frames_pil[i:i+16]
        inputs = clip_processor(images=batch_frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
    frame_feats = torch.cat(all_feats, dim=0)  # [T, D]

    # --- CLIP frame consistency at multiple distances ---
    clip_consistency = {}
    for d in [1, 2, 4, 8, 16]:
        if d >= T:
            break
        cos_sims = (frame_feats[:-d] * frame_feats[d:]).sum(dim=1)
        clip_consistency[f"clip_cos_d{d}"] = cos_sims.mean().item()
        clip_consistency[f"clip_cos_d{d}_std"] = cos_sims.std().item()

    # --- CLIP text-video alignment ---
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**text_inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu()

    text_sims = (frame_feats * text_feat).sum(dim=1)  # [T]
    # Decay curve: average text alignment in first half vs second half
    mid = T // 2
    first_half_sim = text_sims[:mid].mean().item()
    second_half_sim = text_sims[mid:].mean().item()

    return {
        "clip_text_align_mean": text_sims.mean().item(),
        "clip_text_align_std": text_sims.std().item(),
        "clip_text_align_first_half": first_half_sim,
        "clip_text_align_second_half": second_half_sim,
        "clip_text_align_decay": second_half_sim - first_half_sim,
        **clip_consistency,
    }


def evaluate(
    pipe,
    lora_dir: str,
    method: str,
    tau: float,
    theta_t: float,
    output_dir: str,
    num_videos: int = 8,
    frame_counts: list = None,
    num_inference_steps: int = 50,
    seed: int = 42,
):
    """Evaluate temporal extrapolation quality.

    Core idea: EVQ's advantage is extrapolation. Generate videos at training
    length (49f) and beyond (97f, 145f), measure how temporal quality degrades.

    Metrics per video per length:
      - Pixel-level: temporal_mse, motion_magnitude, temporal_jitter
      - Cosine similarity decay curve: cos_sim at distance 1/2/4/8/16
      - CLIP (if available): frame consistency, text alignment, decay

    Output: JSON with per-video metrics + aggregated per-length summary.
    """
    if frame_counts is None:
        frame_counts = [49, 97, 145]  # 1x, 2x, 3x training length

    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  Evaluation: {method.upper()} tau={tau} base={int(theta_t)}")
    print(f"  Videos: {num_videos}, Lengths: {frame_counts}")
    print(f"  Inference steps: {num_inference_steps}")
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

    # Try loading CLIP for semantic metrics
    clip_model, clip_processor = None, None
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                                torch_dtype=torch.float16).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("  [OK] CLIP loaded for semantic metrics")
    except Exception as e:
        print(f"  [WARN] CLIP not available ({e}), using pixel-only metrics")

    prompts = EVAL_PROMPTS[:num_videos]

    all_metrics = {
        "method": method, "tau": tau, "theta_t": theta_t,
        "num_videos": num_videos, "frame_counts": frame_counts,
        "per_video": {},  # {f"{nf}f/video{i}": metrics}
        "per_length": {},  # {f"{nf}f": aggregated metrics}
    }

    for nf in frame_counts:
        label = f"{nf}f"
        extrap_ratio = nf / frame_counts[0]
        print(f"\n  --- Generating {nf} frames ({extrap_ratio:.1f}x) ---")

        length_metrics = []

        for i, prompt in enumerate(prompts):
            vid_key = f"{label}/video{i}"
            try:
                generator = torch.Generator(device).manual_seed(seed + i)
                gen_start = time.time()
                output = pipe(
                    prompt=prompt,
                    num_frames=nf,
                    height=480,
                    width=720,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
                gen_time = time.time() - gen_start

                frames_pil = output.frames[0]  # list of PIL images

                # Save video frames
                save_dir = out_path / f"videos_{method}_tau{tau}" / label
                save_dir.mkdir(parents=True, exist_ok=True)
                for j, frame in enumerate(frames_pil):
                    frame.save(save_dir / f"video{i}_frame{j:04d}.png")

                # Convert to tensor for pixel metrics
                import numpy as np
                frames_np = np.stack([np.array(f).astype(np.float32) / 255.0 for f in frames_pil])
                frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # [T, C, H, W]

                # Compute pixel-level metrics
                vid_metrics = compute_video_metrics(frames_tensor)
                vid_metrics["gen_time_s"] = gen_time
                vid_metrics["num_frames"] = len(frames_pil)
                vid_metrics["extrap_ratio"] = extrap_ratio
                vid_metrics["prompt"] = prompt

                # Compute CLIP metrics if available
                if clip_model is not None:
                    clip_m = compute_clip_metrics(frames_pil, prompt, clip_model, clip_processor, device)
                    vid_metrics.update(clip_m)

                all_metrics["per_video"][vid_key] = vid_metrics
                length_metrics.append(vid_metrics)

                print(f"    video {i}: {len(frames_pil)}f, {gen_time:.1f}s, "
                      f"mse={vid_metrics['temporal_mse_mean']:.4f}, "
                      f"jitter={vid_metrics['temporal_jitter']:.4f}"
                      + (f", clip_d1={vid_metrics.get('clip_cos_d1', 'N/A'):.4f}"
                         if 'clip_cos_d1' in vid_metrics else ""))

            except Exception as e:
                print(f"    video {i}: FAILED ({e})")
                all_metrics["per_video"][vid_key] = {"error": str(e)}
                continue

        # Aggregate per-length metrics
        if length_metrics:
            agg = {}
            # Collect all numeric keys
            numeric_keys = [k for k in length_metrics[0] if isinstance(length_metrics[0][k], (int, float))]
            for k in numeric_keys:
                vals = [m[k] for m in length_metrics if k in m and isinstance(m.get(k), (int, float))]
                if vals:
                    agg[f"{k}_mean"] = sum(vals) / len(vals)
                    if len(vals) > 1:
                        mean = agg[f"{k}_mean"]
                        agg[f"{k}_std"] = (sum((v - mean)**2 for v in vals) / (len(vals) - 1)) ** 0.5
            agg["n_videos"] = len(length_metrics)
            agg["extrap_ratio"] = extrap_ratio
            all_metrics["per_length"][label] = agg

    # --- Summary: extrapolation degradation ---
    print(f"\n{'='*60}")
    print(f"  EXTRAPOLATION SUMMARY: {method.upper()} tau={tau}")
    print(f"{'='*60}")

    base_label = f"{frame_counts[0]}f"
    if base_label in all_metrics["per_length"]:
        base_mse = all_metrics["per_length"][base_label].get("temporal_mse_mean_mean", 0)
        base_jitter = all_metrics["per_length"][base_label].get("temporal_jitter_mean", 0)
        base_clip_d1 = all_metrics["per_length"][base_label].get("clip_cos_d1_mean", None)

        print(f"  {'Length':>8s}  {'MSE':>8s}  {'Jitter':>8s}  {'MSE_degr':>10s}"
              + ("  CLIP_d1  CLIP_degr" if base_clip_d1 is not None else ""))

        for nf in frame_counts:
            label = f"{nf}f"
            if label not in all_metrics["per_length"]:
                continue
            lm = all_metrics["per_length"][label]
            mse = lm.get("temporal_mse_mean_mean", 0)
            jitter = lm.get("temporal_jitter_mean", 0)
            mse_degr = (mse - base_mse) / max(base_mse, 1e-8) * 100

            line = f"  {label:>8s}  {mse:8.4f}  {jitter:8.4f}  {mse_degr:+9.1f}%"
            if base_clip_d1 is not None:
                clip_d1 = lm.get("clip_cos_d1_mean", 0)
                clip_degr = (clip_d1 - base_clip_d1) / max(abs(base_clip_d1), 1e-8) * 100
                line += f"  {clip_d1:.4f}  {clip_degr:+8.1f}%"
            print(line)

    # Save all metrics
    eval_path = out_path / f"eval_{method}_tau{tau}_base{int(theta_t)}.json"
    with open(eval_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved: {eval_path}")

    # Clean up
    del clip_model, clip_processor
    if hasattr(pipe.transformer, 'base_model'):
        pipe.transformer = pipe.transformer.base_model.model
    pipe.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return all_metrics


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
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = auto-detect max batch_size to fill VRAM")
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
    parser.add_argument("--num_eval_videos", type=int, default=8,
                        help="Number of videos to generate per method per length")
    parser.add_argument("--frame_counts", type=int, nargs="+", default=[49, 97, 145],
                        help="Frame counts for eval: training length + extrapolation")

    args = parser.parse_args()

    if args.pilot:
        args.num_steps = 5
        args.skip_eval = True
        args.num_data = 10
        print("\n*** PILOT MODE: 5 steps, 10 samples, no eval ***\n")

    # ---- Startup diagnostics ----
    print("[0/5] System check...")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    cc = torch.cuda.get_device_capability(0)
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f}GB), compute capability: {cc[0]}.{cc[1]}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Check SDPA backends
    from torch.nn.functional import scaled_dot_product_attention
    dummy_q = torch.randn(1, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
    backends = []
    for name, check in [
        ("flash", torch.backends.cuda.flash_sdp_enabled),
        ("mem_efficient", torch.backends.cuda.mem_efficient_sdp_enabled),
        ("math", torch.backends.cuda.math_sdp_enabled),
    ]:
        if callable(check) and check():
            backends.append(name)
    print(f"  SDPA backends enabled: {backends}")
    del dummy_q

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
            # Reload pipeline fresh for evaluation (no compile baggage)
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

            lora_dir = str(Path(args.output_dir) /
                          f"wan21_{method}_tau{tau}_base{int(args.theta_t)}" / "lora_weights")
            eval_result = evaluate(
                pipe=pipe,
                lora_dir=lora_dir,
                method=method,
                tau=tau,
                theta_t=args.theta_t,
                output_dir=args.output_dir,
                num_videos=args.num_eval_videos,
                frame_counts=args.frame_counts,
                seed=args.seed,
            )
            all_results[f"{method}_tau{tau}_eval"] = eval_result

    # ---- Summary ----
    # Training loss comparison
    train_keys = [k for k in all_results if not k.endswith("_eval")]
    if len(train_keys) > 1:
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD: TRAINING LOSS")
        print(f"{'='*60}")
        geo_key = [k for k in train_keys if k.startswith("geo")][0]
        evq_key = [k for k in train_keys if k.startswith("evq")][0]
        geo_loss = all_results[geo_key]["final_loss"]
        evq_loss = all_results[evq_key]["final_loss"]
        delta = (evq_loss - geo_loss) / geo_loss * 100

        print(f"  GEO: loss={geo_loss:.6f}")
        print(f"  EVQ: loss={evq_loss:.6f}")
        print(f"  Delta: {delta:+.1f}% ({'EVQ wins' if delta < 0 else 'GEO wins'})")
        print(f"  NOTE: Training loss favors GEO (model pretrained with GEO).")
        print(f"        The real test is extrapolation quality below.")

        for name in train_keys:
            r = all_results[name]
            print(f"  {name}: loss={r['final_loss']:.6f}, time={r['total_time_s']:.0f}s, "
                  f"VRAM={r['peak_vram_gb']:.1f}GB")

    # Extrapolation comparison (the real test)
    eval_keys = [k for k in all_results if k.endswith("_eval")]
    if len(eval_keys) > 1:
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD: TEMPORAL EXTRAPOLATION")
        print(f"{'='*60}")

        # Compare degradation at each length
        frame_counts_used = args.frame_counts
        base_nf = frame_counts_used[0]

        header = f"  {'Length':>8s}"
        for ek in eval_keys:
            method_name = ek.replace("_eval", "").split("_")[0].upper()
            header += f"  {method_name+'_mse':>10s}  {method_name+'_jit':>10s}"
        print(header)

        for nf in frame_counts_used:
            label = f"{nf}f"
            line = f"  {label:>8s}"
            for ek in eval_keys:
                per_len = all_results[ek].get("per_length", {}).get(label, {})
                mse = per_len.get("temporal_mse_mean_mean", float('nan'))
                jitter = per_len.get("temporal_jitter_mean", float('nan'))
                line += f"  {mse:10.4f}  {jitter:10.4f}"
            print(line)

    summary_path = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results: {summary_path}")


if __name__ == "__main__":
    main()
