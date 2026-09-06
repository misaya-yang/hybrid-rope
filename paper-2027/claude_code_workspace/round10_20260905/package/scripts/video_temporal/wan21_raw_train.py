#!/usr/bin/env python3
"""Wan2.1-1.3B EVQ temporal RoPE: encode + train (raw model, no diffusers).

Usage:
  # First run: encode 288x288 + pilot 5 steps
  python -u wan21_raw_train.py --pilot

  # Full experiment
  python -u wan21_raw_train.py
"""
import os, sys, gc, json, math, time, argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/root/autodl-tmp/wan21/Wan2.1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Config
# ============================================================
MODEL_DIR     = "/root/autodl-tmp/wan21"
VIDEO_DIR     = "/root/autodl-tmp/data/wan21_encoded/raw_videos"
CACHE_DIR     = "/root/autodl-tmp/data/wan21_288"
RESULTS_DIR   = "/root/autodl-tmp/results/wan21_evq_288"
HEIGHT, WIDTH = 288, 288
NUM_FRAMES    = 81

# ============================================================
# EVQ-Cosh frequency computation
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
    """Replace temporal columns in model.freqs (complex tensor [1024, 64])."""
    K_t = t_dim // 2  # 22
    assert inv_freq.shape[0] == K_t, f"Expected {K_t} freqs, got {inv_freq.shape[0]}"
    positions = torch.arange(model.freqs.shape[0], dtype=torch.float64)
    raw = torch.outer(positions, inv_freq.double())  # [1024, 22]
    new_freqs = torch.polar(torch.ones_like(raw), raw)  # complex
    model.freqs[:, :K_t] = new_freqs
    print(f"  Patched temporal RoPE: {K_t} pairs, range=[{inv_freq.min():.6f}, {inv_freq.max():.6f}]")

# ============================================================
# Encode videos (T5 on CPU, VAE on GPU)
# ============================================================
def encode_288(num_videos=20):
    cache = Path(CACHE_DIR)
    if (cache / "latents.pt").exists() and (cache / "text_embeds.pt").exists():
        print("[Encode] Cache exists, skipping.")
        return
    cache.mkdir(parents=True, exist_ok=True)

    from wan.modules.vae import WanVAE
    from wan.modules.t5 import T5EncoderModel
    import decord
    decord.bridge.set_bridge("torch")

    device = torch.device("cuda")

    # T5 on GPU (96GB plenty of room)
    print("[Encode] Loading T5 (GPU)...")
    t5 = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(MODEL_DIR, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(MODEL_DIR, "google/umt5-xxl"),
    )

    # Read captions
    captions_file = Path(VIDEO_DIR) / "captions.json"
    with open(captions_file) as f:
        captions = json.load(f)
    video_files = sorted(Path(VIDEO_DIR).glob("*.mp4"))[:num_videos]

    # Encode text
    print(f"[Encode] Encoding {len(video_files)} text prompts (CPU)...")
    all_text = []
    for vf in video_files:
        cap = captions.get(vf.name, "A video clip")
        embed = t5([cap], device)
        all_text.append(embed[0].cpu())
    del t5; gc.collect()
    print(f"  Text done: {all_text[0].shape}")

    # VAE on GPU
    print("[Encode] Loading VAE (GPU)...")
    vae = WanVAE(vae_pth=os.path.join(MODEL_DIR, "Wan2.1_VAE.pth"), device=device)

    all_latents = []
    for i, vf in enumerate(video_files):
        vr = decord.VideoReader(str(vf))
        total = len(vr)
        if total >= NUM_FRAMES:
            indices = torch.linspace(0, total - 1, NUM_FRAMES).long().tolist()
        else:
            indices = list(range(total)) + [total - 1] * (NUM_FRAMES - total)
        frames = vr.get_batch(indices).float() / 255.0  # [T,H,W,C]
        frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]
        # Resize spatially frame by frame
        frames = F.interpolate(frames, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
        frames = frames.permute(1, 0, 2, 3)  # [C,T,H,W]
        frames = (frames * 2.0 - 1.0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            latent = vae.encode([frames])[0]  # [C,F,H,W]
        all_latents.append(latent.cpu().float())
        if (i+1) % 5 == 0 or i == 0:
            print(f"  Encoded {i+1}/{len(video_files)}: {latent.shape}, VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    del vae; torch.cuda.empty_cache(); gc.collect()

    latents = torch.stack(all_latents)  # [N,C,F,H,W]
    # Pad text to max length
    max_len = max(t.shape[0] for t in all_text)
    padded = []
    for t in all_text:
        if t.shape[0] < max_len:
            pad = torch.zeros(max_len - t.shape[0], t.shape[1], dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    text_embeds = torch.stack(padded)  # [N,L,D]
    torch.save(latents, cache / "latents.pt")
    torch.save(text_embeds, cache / "text_embeds.pt")
    meta = {"n": len(latents), "latent_shape": list(latents.shape[1:]),
            "text_shape": list(text_embeds.shape[1:]), "height": HEIGHT, "width": WIDTH}
    with open(cache / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Encode] Saved {len(latents)} samples: latents={latents.shape}, text={text_embeds.shape}")

# ============================================================
# Dataset
# ============================================================
class LatentDataset(Dataset):
    def __init__(self, cache_dir):
        self.latents = torch.load(Path(cache_dir)/"latents.pt", weights_only=True)
        self.text = torch.load(Path(cache_dir)/"text_embeds.pt", weights_only=True)
        print(f"[Data] {len(self.latents)} samples, latent={self.latents.shape}, text={self.text.shape}")
    def __len__(self): return len(self.latents)
    def __getitem__(self, i): return self.latents[i], self.text[i]

# ============================================================
# Training
# ============================================================
def train(method, tau, num_steps, batch_size, lr, lora_rank, seed, use_compile):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"  {method.upper()} | tau={tau} | steps={num_steps} | bs={batch_size} | lr={lr}")
    print(f"{'='*60}")

    # Load model
    from wan.modules.model import WanModel
    model = WanModel.from_pretrained(MODEL_DIR)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Patch temporal RoPE
    t_dim = 44  # 22 pairs for 1.3B
    if method == "evq":
        inv_freq = evq_cosh_inv_freq(t_dim, tau=tau)
    else:
        inv_freq = geometric_inv_freq(t_dim)
    patch_temporal_freqs(model, inv_freq.float(), t_dim=t_dim)
    model.freqs = model.freqs.to(device)

    # LoRA
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                          target_modules=["q", "k", "v", "o"],
                          lora_dropout=0.0)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if use_compile:
        print("  torch.compile(mode='default')...")
        compiled = torch.compile(model, mode="default")
    else:
        compiled = model

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr*0.1)

    dataset = LatentDataset(CACHE_DIR)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Compute seq_len for RoPE indexing
    sample_lat = dataset[0][0]  # [C,F,H,W]
    F_lat, H_lat, W_lat = sample_lat.shape[1], sample_lat.shape[2], sample_lat.shape[3]
    seq_len = F_lat * H_lat * W_lat
    print(f"  Latent: [{sample_lat.shape}], seq_len={seq_len}")

    out_path = Path(RESULTS_DIR) / f"{method}_tau{tau}"
    out_path.mkdir(parents=True, exist_ok=True)

    step, losses = 0, []
    start = time.time()
    print(f"  Training {num_steps} steps...")

    while step < num_steps:
        for lat_batch, txt_batch in loader:
            if step >= num_steps: break
            step_t0 = time.time()

            # WanModel expects List[Tensor] inputs
            x_list = [lat_batch[i].to(device, dtype=torch.bfloat16) for i in range(lat_batch.shape[0])]
            ctx_list = [txt_batch[i].to(device, dtype=torch.bfloat16) for i in range(txt_batch.shape[0])]
            bsz = len(x_list)

            # Flow matching: sample sigma, create noisy input
            sigmas = torch.rand(bsz, device=device)
            t = sigmas * 1000.0  # Wan2.1 uses 0-1000 range

            noise_list = [torch.randn_like(x) for x in x_list]
            noisy_list = []
            for i in range(bsz):
                s = sigmas[i].to(dtype=x_list[i].dtype)
                noisy_list.append((1 - s) * x_list[i] + s * noise_list[i])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_list = compiled(noisy_list, t, ctx_list, seq_len)
                # Flow matching target: v = noise - x
                loss = 0
                for i in range(bsz):
                    target = noise_list[i] - x_list[i]
                    loss = loss + F.mse_loss(pred_list[i].float(), target.float())
                loss = loss / bsz

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()

            torch.cuda.synchronize()
            step_dt = time.time() - step_t0
            lv = loss.item()
            losses.append(lv)

            if step == 0:
                print(f"  step 0 (compile): {step_dt:.1f}s  loss={lv:.6f}  VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            elif step <= 3 or step % 50 == 0 or step == num_steps - 1:
                print(f"  step {step}/{num_steps}  loss={lv:.6f}  {step_dt:.2f}s/step  VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

            # Save checkpoint every 100 steps
            if (step + 1) % 100 == 0:
                ckpt = {"step": step + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "losses": losses}
                torch.save(ckpt, out_path / "checkpoint.pt")

            step += 1

    total = time.time() - start
    avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
    print(f"\n  Done: {method.upper()} {num_steps}steps in {total:.0f}s, avg_loss={avg:.6f}")

    result = {"method": method, "tau": tau, "num_steps": num_steps, "batch_size": batch_size,
              "final_loss": avg, "losses": losses, "time_s": total,
              "vram_gb": torch.cuda.max_memory_allocated()/1e9, "seed": seed}
    with open(out_path / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    model.save_pretrained(str(out_path / "lora"))
    print(f"  Saved to {out_path}")

    del model, compiled; torch.cuda.empty_cache(); gc.collect()
    return result

# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--method", choices=["geo","evq","both"], default="both")
    p.add_argument("--tau", type=float, default=3.2)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()

    if args.pilot:
        args.steps = 5
        print("\n*** PILOT: 5 steps ***\n")

    # System check
    print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    print(f"torch={torch.__version__}")

    # Encode if needed
    encode_288()

    # Train
    methods = ["geo", "evq"] if args.method == "both" else [args.method]
    results = {}
    for m in methods:
        tau = 0.0 if m == "geo" else args.tau
        results[m] = train(m, tau, args.steps, args.bs, args.lr, args.rank, args.seed, not args.no_compile)

    if len(results) > 1:
        g, e = results["geo"]["final_loss"], results["evq"]["final_loss"]
        d = (e - g) / g * 100
        print(f"\n  GEO={g:.6f}  EVQ={e:.6f}  delta={d:+.1f}%")

if __name__ == "__main__":
    main()
