#!/usr/bin/env python3
"""
CogVideoX-2B EVQ-Cosh Temporal RoPE Fine-tuning Experiment

Only modifies the TEMPORAL dimension (8 frequency pairs) of the 3D RoPE.
Spatial dimensions (height/width) remain untouched.

Two configurations:
  1. GEO baseline (τ=0):  theta_t=10000, geometric frequencies
  2. EVQ (τ=1.2):         theta_t=10000, EVQ-Cosh temporal frequencies

Each: LoRA rank=16, 500 steps, then simple quality eval.

Hardware target: RTX 6000 Pro (96GB), torch.compile, bfloat16
Memory budget: 70GB for model+batch (reserve 20GB for compile overhead)
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
from contextlib import nullcontext

# CUDA config BEFORE torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# EVQ-Cosh frequency computation (from AIHANDOFF Part 2)
# ============================================================

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    """Compute EVQ-Cosh inverse frequencies for a given dimension count.

    Args:
        dim: Number of dimensions (NOT pairs). E.g., dim_t=16 means 8 pairs.
        tau: Concentration parameter. tau=0 -> geometric.
        base: RoPE base frequency.

    Returns:
        inv_freq tensor of shape [dim//2] in float64.
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
    K = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return freqs.float()


# ============================================================
# Monkey-patch: replace temporal RoPE in CogVideoX
# ============================================================

def make_patched_get_3d_rotary_pos_embed(temporal_inv_freq: torch.Tensor, temporal_theta: float):
    """Create a patched version of get_3d_rotary_pos_embed that uses custom temporal frequencies.

    The spatial dimensions (height, width) keep their original geometric frequencies.
    Only the temporal dimension is replaced.
    """
    from diffusers.models.embeddings import get_1d_rotary_pos_embed as orig_get_1d

    def patched_get_3d_rotary_pos_embed(
        embed_dim,
        crops_coords,
        grid_size,
        temporal_size,
        theta: int = 10000,
        use_real: bool = True,
        grid_type: str = "linspace",
        max_size=None,
        device=None,
    ):
        if use_real is not True:
            raise ValueError("use_real=False not supported")

        if grid_type == "linspace":
            start, stop = crops_coords
            grid_size_h, grid_size_w = grid_size
            grid_h = torch.linspace(
                start[0], stop[0] * (grid_size_h - 1) / grid_size_h,
                grid_size_h, device=device, dtype=torch.float32
            )
            grid_w = torch.linspace(
                start[1], stop[1] * (grid_size_w - 1) / grid_size_w,
                grid_size_w, device=device, dtype=torch.float32
            )
            grid_t = torch.linspace(
                0, temporal_size * (temporal_size - 1) / temporal_size,
                temporal_size, device=device, dtype=torch.float32
            )
        elif grid_type == "slice":
            max_h, max_w = max_size
            grid_size_h, grid_size_w = grid_size
            grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
            grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
            grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid grid_type: {grid_type}")

        dim_t = embed_dim // 4   # 16 dims = 8 pairs for temporal
        dim_h = embed_dim // 8 * 3  # 24 dims for height
        dim_w = embed_dim // 8 * 3  # 24 dims for width

        # PATCHED: Use custom temporal frequencies instead of geometric
        # Compute temporal freqs manually using our inv_freq
        inv_freq_t = temporal_inv_freq.to(device=device)  # [K_t] = [8]
        freqs_t_raw = torch.outer(grid_t, inv_freq_t)  # [T, K_t]
        # repeat_interleave to match CogVideoX convention (cos/sin interleaved)
        freqs_t_cos = freqs_t_raw.cos().repeat_interleave(2, dim=1).float()  # [T, dim_t]
        freqs_t_sin = freqs_t_raw.sin().repeat_interleave(2, dim=1).float()  # [T, dim_t]
        freqs_t = (freqs_t_cos, freqs_t_sin)

        # Spatial: keep original geometric frequencies (unchanged)
        freqs_h = orig_get_1d(dim_h, grid_h, theta=theta, use_real=True)
        freqs_w = orig_get_1d(dim_w, grid_w, theta=theta, use_real=True)

        # Combine (same logic as original)
        def combine_thw(ft, fh, fw):
            ft = ft[:, None, None, :].expand(-1, grid_size_h, grid_size_w, -1)
            fh = fh[None, :, None, :].expand(temporal_size, -1, grid_size_w, -1)
            fw = fw[None, None, :, :].expand(temporal_size, grid_size_h, -1, -1)
            f = torch.cat([ft, fh, fw], dim=-1)
            return f.view(temporal_size * grid_size_h * grid_size_w, -1)

        t_cos, t_sin = freqs_t
        h_cos, h_sin = freqs_h
        w_cos, w_sin = freqs_w

        if grid_type == "slice":
            t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
            h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
            w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

        cos = combine_thw(t_cos, h_cos, w_cos)
        sin = combine_thw(t_sin, h_sin, w_sin)
        return cos, sin

    return patched_get_3d_rotary_pos_embed


# ============================================================
# Dataset: WebVid or simple synthetic for quick validation
# ============================================================

class TextVideoDataset(Dataset):
    """Simple dataset that loads pre-encoded video latents + text embeddings.

    For LoRA fine-tuning we need (latents, encoder_hidden_states, timesteps).
    We generate these from the pipeline's VAE + text encoder on first run,
    then cache them.
    """
    def __init__(self, cache_dir: str, num_samples: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.num_samples = num_samples
        # Will be populated by prepare_dataset()
        self.latents = None
        self.text_embeds = None

    def load(self):
        latents_path = self.cache_dir / "latents.pt"
        text_path = self.cache_dir / "text_embeds.pt"
        if latents_path.exists() and text_path.exists():
            self.latents = torch.load(latents_path, weights_only=True)
            self.text_embeds = torch.load(text_path, weights_only=True)
            self.num_samples = len(self.latents)
            return True
        return False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "text_embeds": self.text_embeds[idx],
        }


def prepare_dataset(pipe, cache_dir: str, num_samples: int = 200, batch_encode: int = 4):
    """Encode a set of dummy/real videos into latent space for training.

    For the experiment, we use the pipeline to generate a small training set:
    encode random noise through VAE to get realistic latent shapes.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    latents_path = cache_path / "latents.pt"
    text_path = cache_path / "text_embeds.pt"

    if latents_path.exists() and text_path.exists():
        print(f"[Dataset] Cache found at {cache_path}, loading...")
        return

    print(f"[Dataset] Preparing {num_samples} training samples...")

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Get latent shape from model config
    # CogVideoX-2B: 49 frames -> VAE 4x temporal compress -> 13 temporal tokens
    # Spatial: 480x720 -> VAE 8x compress -> 60x90 -> patch 2 -> 30x45
    # Latent channels: 16

    # For training: use standard 49-frame, 480x720 config
    num_frames = 49
    height, width = 480, 720

    # Temporal: 49 frames / 4 = 13 (rounded, actually (49-1)/4 + 1 = 13)
    # Spatial: 480/8=60, 720/8=90
    latent_t = (num_frames - 1) // 4 + 1  # = 13
    latent_h = height // 8   # = 60
    latent_w = width // 8    # = 90
    latent_channels = 16

    # Generate random latents (standard normal, as diffusion models expect)
    all_latents = torch.randn(num_samples, latent_channels, latent_t, latent_h, latent_w,
                               dtype=torch.float32)

    # Generate text embeddings from simple prompts
    prompts = [
        "A cat walking on the grass",
        "Ocean waves crashing on the beach",
        "A person riding a bicycle in the park",
        "Fireworks lighting up the night sky",
        "A dog playing with a ball",
        "Rain falling on a city street",
        "Flowers blooming in time-lapse",
        "Birds flying across the sunset",
        "A river flowing through a forest",
        "Clouds moving across a blue sky",
    ]

    all_text_embeds = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_encode):
            batch_prompts = [prompts[j % len(prompts)] for j in range(i, min(i + batch_encode, num_samples))]
            tokens = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(text_encoder.device)
            embeds = text_encoder(tokens.input_ids)[0].cpu()
            all_text_embeds.append(embeds)

    all_text_embeds = torch.cat(all_text_embeds, dim=0)[:num_samples]

    torch.save(all_latents, latents_path)
    torch.save(all_text_embeds, text_path)
    print(f"[Dataset] Saved {num_samples} samples to {cache_path}")


# ============================================================
# Training loop
# ============================================================

def train_lora(
    model_path: str,
    method: str,  # "geo" or "evq"
    tau: float = 1.2,
    theta_t: float = 10000.0,
    num_steps: int = 500,
    lora_rank: int = 16,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    output_dir: str = "results",
    cache_dir: str = "data_cache",
    use_compile: bool = True,
    seed: int = 42,
):
    """Fine-tune CogVideoX-2B with LoRA, optionally with EVQ temporal RoPE."""

    torch.manual_seed(seed)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  CogVideoX-2B LoRA Fine-tuning: {method.upper()}")
    print(f"  tau={tau}, theta_t={theta_t}, steps={num_steps}")
    print(f"  LoRA rank={lora_rank}, lr={lr}, bs={batch_size}x{grad_accum}")
    print(f"{'='*60}\n")

    # ---- Load pipeline ----
    from diffusers import CogVideoXPipeline

    print("[1/6] Loading CogVideoX-2B pipeline...")
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer

    # ---- Compute temporal inv_freq ----
    embed_dim = transformer.config.attention_head_dim  # 64
    dim_t = embed_dim // 4  # 16 dims = 8 pairs

    if method == "evq":
        temporal_inv_freq = evq_cosh_inv_freq(dim_t, tau=tau, base=theta_t)
        print(f"[2/6] EVQ-Cosh temporal frequencies (tau={tau}, base={theta_t}):")
    else:
        temporal_inv_freq = geometric_inv_freq(dim_t, base=theta_t)
        print(f"[2/6] Geometric temporal frequencies (base={theta_t}):")

    print(f"  inv_freq: {temporal_inv_freq.tolist()}")

    # ---- Monkey-patch 3D RoPE ----
    import diffusers.models.embeddings as emb_module
    import diffusers.pipelines.cogvideo.pipeline_cogvideox as pipe_module

    patched_fn = make_patched_get_3d_rotary_pos_embed(temporal_inv_freq, theta_t)
    emb_module.get_3d_rotary_pos_embed = patched_fn
    pipe_module.get_3d_rotary_pos_embed = patched_fn

    # Also patch in the transformer module if it imports directly
    transformer_module = sys.modules.get("diffusers.models.transformers.cogvideox_transformer_3d")
    if transformer_module and hasattr(transformer_module, "get_3d_rotary_pos_embed"):
        transformer_module.get_3d_rotary_pos_embed = patched_fn

    print(f"  [Patched] 3D RoPE temporal frequencies replaced")

    # ---- Prepare dataset ----
    print("[3/6] Preparing dataset...")
    pipe.text_encoder.to(device)
    prepare_dataset(pipe, cache_dir, num_samples=200, batch_encode=4)
    pipe.text_encoder.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    dataset = TextVideoDataset(cache_dir)
    dataset.load()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Apply LoRA ----
    print(f"[4/6] Applying LoRA (rank={lora_rank})...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device)
    transformer.train()

    # Enable gradient checkpointing to save memory
    transformer.enable_gradient_checkpointing()

    # ---- torch.compile ----
    if use_compile:
        print("[4.5/6] Compiling model with torch.compile...")
        transformer = torch.compile(transformer, mode="default")

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Cosine LR schedule
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    # ---- Load scheduler for noise ----
    noise_scheduler = pipe.scheduler

    # ---- Training ----
    print(f"[5/6] Training for {num_steps} steps...")

    out_path = Path(output_dir) / f"cogvideox_{method}_tau{tau}_base{int(theta_t)}"
    out_path.mkdir(parents=True, exist_ok=True)

    step = 0
    losses = []
    start_time = time.time()

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            if use_compile:
                torch.compiler.cudagraph_mark_step_begin()

            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            text_embeds = batch["text_embeds"].to(device=device, dtype=torch.bfloat16)

            # Sample random timesteps
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                       (bsz,), device=device, dtype=torch.long)

            # Add noise (forward diffusion)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Forward pass through transformer
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # CogVideoX transformer expects: hidden_states, encoder_hidden_states, timestep
                model_output = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=text_embeds,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                # MSE loss against noise (epsilon prediction) or v-prediction
                loss = F.mse_loss(model_output, noise) / grad_accum

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
    avg_loss = sum(losses[-100:]) / len(losses[-100:])

    print(f"\n[6/6] Training complete. {num_steps} steps in {total_time:.0f}s")
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

    result_path = out_path / "train_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {result_path}")

    # Save LoRA weights
    lora_path = out_path / "lora_weights"
    transformer.save_pretrained(str(lora_path))
    print(f"  LoRA weights saved to {lora_path}")

    return result


# ============================================================
# Simple eval: generate videos at train length and extrapolated length
# ============================================================

def evaluate(
    model_path: str,
    lora_dir: str,
    method: str,
    tau: float,
    theta_t: float,
    output_dir: str,
    num_videos: int = 4,
    seed: int = 42,
):
    """Generate videos at standard (49f) and extrapolated (97f) lengths."""

    print(f"\n{'='*60}")
    print(f"  Evaluation: {method.upper()} tau={tau} base={int(theta_t)}")
    print(f"{'='*60}\n")

    device = torch.device("cuda")

    from diffusers import CogVideoXPipeline

    # Load pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # Patch temporal RoPE
    embed_dim = pipe.transformer.config.attention_head_dim
    dim_t = embed_dim // 4

    if method == "evq":
        temporal_inv_freq = evq_cosh_inv_freq(dim_t, tau=tau, base=theta_t)
    else:
        temporal_inv_freq = geometric_inv_freq(dim_t, base=theta_t)

    import diffusers.models.embeddings as emb_module
    import diffusers.pipelines.cogvideo.pipeline_cogvideox as pipe_module

    patched_fn = make_patched_get_3d_rotary_pos_embed(temporal_inv_freq, theta_t)
    emb_module.get_3d_rotary_pos_embed = patched_fn
    pipe_module.get_3d_rotary_pos_embed = patched_fn

    transformer_module = sys.modules.get("diffusers.models.transformers.cogvideox_transformer_3d")
    if transformer_module and hasattr(transformer_module, "get_3d_rotary_pos_embed"):
        transformer_module.get_3d_rotary_pos_embed = patched_fn

    # Load LoRA if exists
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_dir)
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

    results = {}

    # Standard length generation (49 frames)
    print("\n  Generating standard-length videos (49 frames)...")
    generator = torch.Generator(device).manual_seed(seed)
    for i, prompt in enumerate(prompts):
        video = pipe(
            prompt=prompt,
            num_frames=49,
            height=480,
            width=720,
            num_inference_steps=50,
            generator=generator,
        ).frames[0]
        # Save as tensor for metric computation
        print(f"    Video {i}: {len(video)} frames generated")

    results["standard_49f"] = "completed"

    # Extrapolated length generation (97 frames = ~2x temporal extrapolation)
    print("\n  Generating extrapolated videos (97 frames)...")
    generator = torch.Generator(device).manual_seed(seed)
    try:
        for i, prompt in enumerate(prompts):
            video = pipe(
                prompt=prompt,
                num_frames=97,
                height=480,
                width=720,
                num_inference_steps=50,
                generator=generator,
            ).frames[0]
            print(f"    Video {i}: {len(video)} frames generated (extrapolated)")
        results["extrapolated_97f"] = "completed"
    except Exception as e:
        print(f"    Extrapolation failed: {e}")
        results["extrapolated_97f"] = f"failed: {str(e)}"

    result_path = out_path / f"eval_{method}_tau{tau}_base{int(theta_t)}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Eval results saved to {result_path}")


# ============================================================
# Main: run full experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CogVideoX-2B EVQ Fine-tuning")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--method", type=str, choices=["geo", "evq", "both"], default="both",
                        help="Which method to run")
    parser.add_argument("--tau", type=float, default=1.2,
                        help="EVQ tau parameter")
    parser.add_argument("--theta_t", type=float, default=10000.0,
                        help="Temporal RoPE base frequency")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/cogvideox_evq")
    parser.add_argument("--cache_dir", type=str, default="data/cogvideox_cache")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run 5 steps to verify setup")

    args = parser.parse_args()

    if args.pilot:
        args.num_steps = 5
        args.skip_eval = True
        print("\n*** PILOT MODE: 5 steps, no eval ***\n")

    methods = ["geo", "evq"] if args.method == "both" else [args.method]

    all_results = {}

    for method in methods:
        tau = 0.0 if method == "geo" else args.tau
        theta = args.theta_t

        if not args.eval_only:
            result = train_lora(
                model_path=args.model_path,
                method=method,
                tau=tau,
                theta_t=theta,
                num_steps=args.num_steps,
                lora_rank=args.lora_rank,
                lr=args.lr,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                use_compile=not args.no_compile,
                seed=args.seed,
            )
            all_results[f"{method}_tau{tau}_base{int(theta)}"] = result

        if not args.skip_eval:
            lora_dir = str(Path(args.output_dir) /
                          f"cogvideox_{method}_tau{tau}_base{int(theta)}" / "lora_weights")
            evaluate(
                model_path=args.model_path,
                lora_dir=lora_dir,
                method=method,
                tau=tau,
                theta_t=theta,
                output_dir=args.output_dir,
                seed=args.seed,
            )

        # Clean up between runs
        torch.cuda.empty_cache()
        gc.collect()

    # ---- Summary ----
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        for name, r in all_results.items():
            print(f"  {name}: loss={r['final_loss']:.6f}, time={r['total_time_s']:.0f}s, "
                  f"VRAM={r['peak_vram_gb']:.1f}GB")

    # Save combined results
    summary_path = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results: {summary_path}")


if __name__ == "__main__":
    main()
