#!/usr/bin/env python3
"""
CogVideoX-2B EVQ-Cosh Temporal RoPE Fine-tuning Experiment

Only modifies the TEMPORAL dimension (8 frequency pairs) of the 3D RoPE.
Spatial dimensions (height/width) remain untouched.

Head-to-head design:
  GEO baseline (tau=0) and EVQ run on SAME data, SAME seed.
  Eliminates CUDA non-determinism per REPORT_FINAL guidelines.

Data: Disney-VideoGeneration-Dataset (69 real videos)
  Videos → CogVideoX VAE → latents (cached to disk)
  Prompts → T5 text encoder → embeddings (cached to disk)

Server paths:
  Model:   /root/autodl-tmp/cogvideoX/model/ZhipuAI/CogVideoX-2b
  Dataset: /root/autodl-tmp/cogvideoX/dataset
  Cache:   /root/autodl-tmp/cogvideoX/cache

Usage:
  python cogvideox_evq_finetune.py --pilot                    # 5-step verify
  python cogvideox_evq_finetune.py                             # full: GEO + EVQ(τ=1.2), 500 steps
  python cogvideox_evq_finetune.py --tau 1.5 --method evq      # single EVQ run
"""

import os
import sys
import gc
import json
import math
import time
import argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# EVQ-Cosh frequency computation (from AIHANDOFF Part 2)
# ============================================================

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    inv_freq = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    return inv_freq.float()


def geometric_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return freqs.float()


# ============================================================
# Monkey-patch: replace temporal RoPE in CogVideoX
# ============================================================

def make_patched_get_3d_rotary_pos_embed(temporal_inv_freq: torch.Tensor, temporal_theta: float):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed as orig_get_1d

    def patched_get_3d_rotary_pos_embed(
        embed_dim, crops_coords, grid_size, temporal_size,
        theta: int = 10000, use_real: bool = True,
        grid_type: str = "linspace", max_size=None, device=None,
    ):
        if use_real is not True:
            raise ValueError("use_real=False not supported")

        if grid_type == "linspace":
            start, stop = crops_coords
            grid_size_h, grid_size_w = grid_size
            grid_h = torch.linspace(start[0], stop[0] * (grid_size_h - 1) / grid_size_h,
                                    grid_size_h, device=device, dtype=torch.float32)
            grid_w = torch.linspace(start[1], stop[1] * (grid_size_w - 1) / grid_size_w,
                                    grid_size_w, device=device, dtype=torch.float32)
            grid_t = torch.linspace(0, temporal_size * (temporal_size - 1) / temporal_size,
                                    temporal_size, device=device, dtype=torch.float32)
        elif grid_type == "slice":
            max_h, max_w = max_size
            grid_size_h, grid_size_w = grid_size
            grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
            grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
            grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid grid_type: {grid_type}")

        dim_t = embed_dim // 4
        dim_h = embed_dim // 8 * 3
        dim_w = embed_dim // 8 * 3

        # PATCHED: custom temporal frequencies
        inv_freq_t = temporal_inv_freq.to(device=device)
        freqs_t_raw = torch.outer(grid_t, inv_freq_t)
        freqs_t_cos = freqs_t_raw.cos().repeat_interleave(2, dim=1).float()
        freqs_t_sin = freqs_t_raw.sin().repeat_interleave(2, dim=1).float()
        freqs_t = (freqs_t_cos, freqs_t_sin)

        # Spatial: original geometric (unchanged)
        freqs_h = orig_get_1d(dim_h, grid_h, theta=theta, use_real=True)
        freqs_w = orig_get_1d(dim_w, grid_w, theta=theta, use_real=True)

        def combine_thw(ft, fh, fw):
            ft = ft[:, None, None, :].expand(-1, grid_size_h, grid_size_w, -1)
            fh = fh[None, :, None, :].expand(temporal_size, -1, grid_size_w, -1)
            fw = fw[None, None, :, :].expand(temporal_size, grid_size_h, -1, -1)
            return torch.cat([ft, fh, fw], dim=-1).view(temporal_size * grid_size_h * grid_size_w, -1)

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
# Dataset: real video latents from Disney dataset
# ============================================================

class CachedLatentDataset(Dataset):
    """Pre-encoded video latents + text embeddings from disk cache."""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.latents = None
        self.text_embeds = None

    def load(self) -> bool:
        lp = self.cache_dir / "latents.pt"
        tp = self.cache_dir / "text_embeds.pt"
        if lp.exists() and tp.exists():
            self.latents = torch.load(lp, weights_only=True)
            self.text_embeds = torch.load(tp, weights_only=True)
            print(f"[Dataset] Loaded {len(self.latents)} samples")
            print(f"  latents: {self.latents.shape}, text: {self.text_embeds.shape}")
            return True
        return False

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {"latents": self.latents[idx], "text_embeds": self.text_embeds[idx]}


def encode_dataset(
    model_path: str,
    data_dir: str,
    cache_dir: str,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
):
    """Encode Disney videos through CogVideoX VAE + T5 to create training cache.

    Reads .mp4 files from data_dir/videos/ and prompts from data_dir/prompt.txt.
    Saves latents.pt and text_embeds.pt to cache_dir.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if (cache_path / "latents.pt").exists() and (cache_path / "text_embeds.pt").exists():
        print(f"[Encode] Cache exists at {cache_path}, skipping.")
        return

    data_path = Path(data_dir)

    # Load video-prompt pairs
    videos_txt = data_path / "videos.txt"
    prompt_txt = data_path / "prompt.txt"

    video_files = [data_path / line.strip() for line in videos_txt.read_text().splitlines() if line.strip()]
    prompts = [line.strip() for line in prompt_txt.read_text().splitlines() if line.strip()]

    assert len(video_files) == len(prompts), \
        f"Mismatch: {len(video_files)} videos vs {len(prompts)} prompts"
    print(f"[Encode] Found {len(video_files)} video-prompt pairs")

    device = torch.device("cuda")

    # --- Load VAE ---
    from diffusers import AutoencoderKLCogVideoX
    print("[Encode] Loading VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16)
    vae.to(device).eval()
    vae.enable_slicing()
    vae.enable_tiling()

    # --- Load text encoder ---
    from transformers import T5EncoderModel, T5Tokenizer
    print("[Encode] Loading T5 text encoder...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder.to(device).eval()

    # --- Load video decoder ---
    try:
        import decord
        decord.bridge.set_bridge("torch")
        use_decord = True
    except ImportError:
        print("[Encode] decord not found, trying imageio...")
        import imageio.v3 as iio
        use_decord = False

    all_latents = []
    all_text_embeds = []
    skipped = 0

    for i, (vf, prompt) in enumerate(zip(video_files, prompts)):
        if not vf.exists():
            print(f"  Skip: {vf.name} not found")
            skipped += 1
            continue

        try:
            # --- Read video frames ---
            if use_decord:
                vr = decord.VideoReader(str(vf))
                total = len(vr)
                if total >= num_frames:
                    indices = torch.linspace(0, total - 1, num_frames).long().tolist()
                else:
                    indices = list(range(total)) + [total - 1] * (num_frames - total)
                frames = vr.get_batch(indices)  # [T, H, W, C] uint8
                frames = frames.float() / 255.0  # [T, H, W, C]
                frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
            else:
                raw_frames = list(iio.iter(str(vf)))
                total = len(raw_frames)
                if total >= num_frames:
                    step = total / num_frames
                    indices = [int(j * step) for j in range(num_frames)]
                else:
                    indices = list(range(total)) + [total - 1] * (num_frames - total)
                frames = torch.stack([torch.from_numpy(raw_frames[j]).float() / 255.0 for j in indices])
                frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

            # Resize
            frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
            # Normalize to [-1, 1]
            frames = frames * 2.0 - 1.0
            # [B, C, T, H, W] for VAE
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float16)

            # --- Encode video through VAE ---
            with torch.no_grad():
                latent_dist = vae.encode(frames).latent_dist
                latent = latent_dist.sample()
                latent = latent * vae.config.scaling_factor
            all_latents.append(latent.cpu().float())

            # --- Encode text through T5 ---
            with torch.no_grad():
                tokens = tokenizer(
                    prompt, padding="max_length", max_length=226,
                    truncation=True, return_tensors="pt",
                ).to(device)
                text_embed = text_encoder(tokens.input_ids)[0].cpu().float()
            all_text_embeds.append(text_embed)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Encoded {i+1}/{len(video_files)} | latent: {latent.shape} | VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        except Exception as e:
            print(f"  Error encoding {vf.name}: {e}")
            skipped += 1
            continue

    # Cleanup encoder models
    del vae, text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    if len(all_latents) == 0:
        print("[ERROR] No videos encoded!")
        sys.exit(1)

    all_latents = torch.cat(all_latents, dim=0)      # [N, C, T, H, W]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)  # [N, seq_len, dim]

    torch.save(all_latents, cache_path / "latents.pt")
    torch.save(all_text_embeds, cache_path / "text_embeds.pt")
    print(f"[Encode] Done: {len(all_latents)} videos encoded, {skipped} skipped")
    print(f"  latents: {all_latents.shape}, text: {all_text_embeds.shape}")
    print(f"  Saved to {cache_path}")


# ============================================================
# Training loop
# ============================================================

def train_lora(
    model_path: str,
    method: str,
    tau: float = 1.2,
    theta_t: float = 10000.0,
    num_steps: int = 500,
    lora_rank: int = 16,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    output_dir: str = "results",
    dataset: CachedLatentDataset = None,
    use_compile: bool = True,
    seed: int = 42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  CogVideoX-2B LoRA: {method.upper()}")
    print(f"  tau={tau}, base={theta_t}, steps={num_steps}")
    print(f"  LoRA rank={lora_rank}, lr={lr}, bs={batch_size}x{grad_accum}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    from diffusers import CogVideoXPipeline
    print("[1/5] Loading CogVideoX-2B...")
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer

    # ---- Compute and patch temporal inv_freq ----
    embed_dim = transformer.config.attention_head_dim  # 64
    dim_t = embed_dim // 4  # 16 dims = 8 pairs

    if method == "evq":
        temporal_inv_freq = evq_cosh_inv_freq(dim_t, tau=tau, base=theta_t)
        print(f"[2/5] EVQ-Cosh temporal (tau={tau}, base={theta_t}):")
    else:
        temporal_inv_freq = geometric_inv_freq(dim_t, base=theta_t)
        print(f"[2/5] Geometric temporal (base={theta_t}):")
    print(f"  inv_freq: {[f'{x:.6f}' for x in temporal_inv_freq.tolist()]}")

    # Monkey-patch all import sites
    import diffusers.models.embeddings as emb_module
    import diffusers.pipelines.cogvideo.pipeline_cogvideox as pipe_module
    patched_fn = make_patched_get_3d_rotary_pos_embed(temporal_inv_freq, theta_t)
    emb_module.get_3d_rotary_pos_embed = patched_fn
    pipe_module.get_3d_rotary_pos_embed = patched_fn
    transformer_module = sys.modules.get("diffusers.models.transformers.cogvideox_transformer_3d")
    if transformer_module and hasattr(transformer_module, "get_3d_rotary_pos_embed"):
        transformer_module.get_3d_rotary_pos_embed = patched_fn
    print("  [OK] Temporal RoPE patched")

    # ---- LoRA ----
    print(f"[3/5] Applying LoRA (rank={lora_rank})...")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.to(device).train()
    transformer.enable_gradient_checkpointing()

    if use_compile:
        print("[3.5/5] torch.compile...")
        compiled = torch.compile(transformer, mode="default")
    else:
        compiled = transformer

    # ---- Optimizer ----
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    noise_scheduler = pipe.scheduler
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Train ----
    out_path = Path(output_dir) / f"cogvideox_{method}_tau{tau}_base{int(theta_t)}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[4/5] Training {num_steps} steps...")
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

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device, dtype=torch.long
            )

            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_output = compiled(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=text_embeds,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
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
    avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)

    print(f"\n[5/5] Done: {method.upper()} {num_steps} steps in {total_time:.0f}s")
    print(f"  Avg loss (last 100): {avg_loss:.6f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    result = {
        "method": method, "tau": tau, "theta_t": theta_t,
        "num_steps": num_steps, "lora_rank": lora_rank,
        "final_loss": avg_loss, "all_losses": losses,
        "total_time_s": total_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "temporal_inv_freq": temporal_inv_freq.tolist(),
        "seed": seed,
    }
    with open(out_path / "train_result.json", "w") as f:
        json.dump(result, f, indent=2)

    lora_path = out_path / "lora_weights"
    transformer.save_pretrained(str(lora_path))
    print(f"  Results: {out_path}")

    del transformer, compiled, pipe
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CogVideoX-2B EVQ Fine-tuning")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/cogvideoX/model/ZhipuAI/CogVideoX-2b")
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/cogvideoX/dataset")
    parser.add_argument("--cache_dir", type=str,
                        default="/root/autodl-tmp/cogvideoX/cache")
    parser.add_argument("--method", type=str, choices=["geo", "evq", "both"], default="both")
    parser.add_argument("--tau", type=float, default=1.2)
    parser.add_argument("--theta_t", type=float, default=10000.0)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/cogvideoX/results")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    if args.pilot:
        args.num_steps = 5
        args.skip_eval = True
        print("\n*** PILOT MODE: 5 steps, no eval ***\n")

    # ---- Step 1: Encode dataset (once, shared between GEO and EVQ) ----
    print("[0/5] Encoding video dataset through VAE + T5...")
    encode_dataset(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )

    dataset = CachedLatentDataset(args.cache_dir)
    if not dataset.load():
        print("[ERROR] Failed to load cached dataset")
        sys.exit(1)

    # ---- Step 2: Train ----
    methods = ["geo", "evq"] if args.method == "both" else [args.method]
    all_results = {}

    for method in methods:
        tau = 0.0 if method == "geo" else args.tau
        result = train_lora(
            model_path=args.model_path,
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
        print(f"  Delta: {delta:+.1f}% ({'EVQ WINS' if delta < 0 else 'GEO WINS'})")

        for name, r in all_results.items():
            print(f"  {name}: time={r['total_time_s']:.0f}s, VRAM={r['peak_vram_gb']:.1f}GB")

    summary_path = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results: {summary_path}")


if __name__ == "__main__":
    main()
