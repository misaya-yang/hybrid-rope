#!/usr/bin/env python3
"""Wan2.1-T2V-1.3B EVQ-Cosh Fine-tuning (Official API, head-to-head).

Uses official Wan2.1 code (NOT diffusers). Modifies ONLY temporal RoPE frequencies.
Pre-encoded latents + text embeddings as input (no T5/VAE needed during training).

Three configs in one h2h run:
  1. GEO (base=10000, tau=0) - original
  2. GEO (base=1000, tau=0) - dead channel fix only
  3. EVQ (base=1000, tau=1.5) - dead channel fix + EVQ

AIHANDOFF compliance:
  - torch.compile(mode="default") [#1]
  - cudagraph_mark_step_begin() [#2]
  - expandable_segments [#3]
  - batch_size targets >90% VRAM [#4]
  - bfloat16 [#5]
  - memory cleanup between runs [#6]
  - 5-step pilot verify [#7, #12]
  - results in JSON [#9]
  - gradient accumulation [#10]
  - h2h same-run comparison [Part 3 #9]

Usage:
    # Pilot (5 steps, verify VRAM/loss)
    python wan21_finetune_evq.py --model_dir /path/to/Wan2.1-T2V-1.3B \
        --data_dir /path/to/encoded_data --pilot

    # Full h2h (500 steps each)
    python wan21_finetune_evq.py --model_dir /path/to/Wan2.1-T2V-1.3B \
        --data_dir /path/to/encoded_data --steps 500

Hardware: RTX 6000 Pro (96GB), Blackwell sm_120
"""
import os, sys, gc, json, math, time, argparse
from pathlib import Path
from typing import Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# CRITICAL: seq_len=32760 → O(L²) math_sdp would need 25.8GB/sample
# Force mem_efficient_sdp to avoid catastrophic OOM
torch.backends.cuda.enable_flash_sdp(True)     # Try flash first
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Fallback to mem-efficient
torch.backends.cuda.enable_math_sdp(False)      # NEVER use O(L²) math

DEVICE = "cuda"
DTYPE = torch.bfloat16


# ============================================================
# EVQ-Cosh (AIHANDOFF Part 2, float64, midpoint u_k)
# ============================================================

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


# ============================================================
# Patch Wan2.1 model.freqs (complex polar form)
# ============================================================

def patch_temporal_freqs(model, temporal_inv_freq: torch.Tensor, label: str = ""):
    """Replace temporal frequencies in model.freqs (complex128).

    model.freqs: [1024, 64] complex128
    Split: [K_t_complex, K_h_complex, K_w_complex] = [22, 21, 21]

    We replace the first 22 complex columns with new temporal frequencies.
    """
    d = model.dim // model.num_heads  # 128
    K_t_complex = (d - 4 * (d // 6)) // 2  # 22

    assert temporal_inv_freq.shape[0] == K_t_complex, \
        f"Expected {K_t_complex} temporal freqs, got {temporal_inv_freq.shape[0]}"

    # Build new temporal freqs in complex polar form (same as rope_params)
    max_seq_len = model.freqs.shape[0]  # 1024
    positions = torch.arange(max_seq_len, dtype=torch.float64)
    new_temporal = torch.outer(positions, temporal_inv_freq.double())
    new_temporal = torch.polar(torch.ones_like(new_temporal), new_temporal)

    # Replace temporal portion, keep spatial
    new_freqs = model.freqs.clone()
    new_freqs[:, :K_t_complex] = new_temporal

    model.freqs = new_freqs
    print(f"  [{label}] Patched temporal RoPE: K_t={K_t_complex}")
    print(f"  [{label}] inv_freq range: [{temporal_inv_freq.min():.8f}, {temporal_inv_freq.max():.6f}]")


def compute_wan_geometric_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Wan2.1's exact geometric formula: 1/base^(2k/dim) for k=0..dim//2-1."""
    return (1.0 / torch.pow(base, torch.arange(0, dim, 2, dtype=torch.float64) / dim)).float()


# ============================================================
# Dataset
# ============================================================

class EncodedVideoDataset(Dataset):
    """Pre-encoded video latents + text embeddings."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.latents = torch.load(self.data_dir / "latents.pt", weights_only=True)
        self.text_embeds = torch.load(self.data_dir / "text_embeds.pt", weights_only=True)

        with open(self.data_dir / "metadata.json") as f:
            self.meta = json.load(f)

        print(f"[Data] Loaded {len(self.latents)} samples")
        print(f"  latent shape: {self.latents.shape}")
        print(f"  text_embed shape: {self.text_embeds.shape}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.text_embeds[idx]


# ============================================================
# Training
# ============================================================

def train_one_config(
    model_dir: str,
    data_dir: str,
    method: str,
    tau: float,
    base_t: float,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    lora_rank: int,
    output_dir: str,
    use_compile: bool,
    seed: int,
    pilot: bool = False,
):
    """Train one configuration. Returns result dict."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Verify memory is clean
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"\n[Memory] GPU allocated before load: {allocated:.2f} GB")
    assert allocated < 1.0, f"GPU not clean! {allocated:.2f} GB still allocated"

    # ---- Load model ----
    wan_repo = os.environ.get("WAN_REPO", str(Path(model_dir).parent / "Wan2.1"))
    if wan_repo not in sys.path:
        sys.path.insert(0, wan_repo)
    from wan.modules.model import WanModel

    print(f"\n{'='*60}")
    print(f"  Config: {method} | tau={tau} | base_t={int(base_t)}")
    print(f"  Steps: {steps} | bs={batch_size} | accum={grad_accum} | lr={lr}")
    print(f"{'='*60}")

    model = WanModel.from_pretrained(model_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e9:.2f}B params")

    # ---- Patch temporal RoPE ----
    d = model.dim // model.num_heads  # 128
    t_dim = d - 4 * (d // 6)  # 44

    if method == "evq":
        inv_freq_t = evq_cosh_inv_freq(t_dim, tau=tau, base=base_t)
    else:
        inv_freq_t = compute_wan_geometric_inv_freq(t_dim, base=base_t)

    patch_temporal_freqs(model, inv_freq_t, f"{method}(τ={tau},b={int(base_t)})")

    # ---- LoRA ----
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[ERROR] peft not installed: pip install peft")
        sys.exit(1)

    # Target the self-attention and cross-attention linear layers
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q", "k", "v", "o"],  # WanSelfAttention layers
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(DEVICE)
    model.train()

    # Enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
    except AttributeError:
        # WanModel uses diffusers ModelMixin which needs manual setup
        model.enable_gradient_checkpointing = True
        import functools
        from torch.utils.checkpoint import checkpoint as torch_ckpt
        for block in model.base_model.model.blocks:
            block._orig_forward = block.forward
            block.forward = functools.partial(
                lambda mod, *args, **kwargs: torch_ckpt(mod._orig_forward, *args, use_reentrant=False, **kwargs),
                block)
        print("  Gradient checkpointing: manual wrap applied")

    # ---- Compile ----
    if use_compile:
        print("  torch.compile(mode='default')...")
        compiled = torch.compile(model, mode="default")
    else:
        compiled = model

    # ---- Data ----
    dataset = EncodedVideoDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- Optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Cosine schedule
    def get_lr(step):
        warmup = int(steps * 0.05)
        if step < warmup:
            return lr * step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return lr * 0.1 + lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # ---- Verify data format ----
    sample_lat, sample_txt = dataset[0]
    print(f"\n  Data verification:")
    print(f"    latent: {sample_lat.shape} {sample_lat.dtype}")
    print(f"    text: {sample_txt.shape} {sample_txt.dtype}")

    # Wan2.1 forward expects:
    #   x: List[Tensor] of [C, F, H, W] per sample
    #   t: Tensor [B] timesteps
    #   context: List[Tensor] of [text_len, dim] per sample
    #   seq_len: int (max sequence length)

    # Compute seq_len from latent shape
    C, F_lat, H_lat, W_lat = sample_lat.shape
    patch_size = (1, 2, 2)
    seq_len = F_lat * (H_lat // patch_size[1]) * (W_lat // patch_size[2])
    print(f"    F={F_lat}, H={H_lat}, W={W_lat} → seq_len={seq_len}")

    total_tokens = steps * batch_size * grad_accum * seq_len
    print(f"\n  Training plan:")
    print(f"    total_steps={steps}, tokens/step={batch_size * seq_len}")
    print(f"    total_tokens={total_tokens:,}")

    # ---- Training loop ----
    has_cudagraph = hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    step = 0
    losses = []
    t0 = time.time()
    out_path = Path(output_dir) / f"{method}_tau{tau}_base{int(base_t)}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Starting training ({steps} steps)...")
    torch.cuda.reset_peak_memory_stats()

    while step < steps:
        for latent_batch, text_batch in dataloader:
            if step >= steps:
                break

            if has_cudagraph:
                torch.compiler.cudagraph_mark_step_begin()

            # Set LR
            cur_lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            # Prepare inputs in Wan2.1 format
            latents = latent_batch.to(DEVICE, dtype=DTYPE)  # [B, C, F, H, W]
            texts = text_batch.to(DEVICE, dtype=DTYPE)       # [B, text_len, dim]

            bsz = latents.shape[0]

            # Flow matching: sample t, interpolate
            t_sample = torch.rand(bsz, device=DEVICE)
            noise = torch.randn_like(latents)
            # x_t = (1-t) * x_0 + t * noise
            t_bc = t_sample.view(bsz, 1, 1, 1, 1)
            x_t = (1 - t_bc) * latents + t_bc * noise
            # velocity target: v = noise - x_0
            v_target = noise - latents

            # Wan2.1 forward: expects list of tensors + timestep as 1000-scale
            x_list = [x_t[i] for i in range(bsz)]
            context_list = [texts[i] for i in range(bsz)]
            timestep = (t_sample * 1000).to(DTYPE)

            with torch.amp.autocast("cuda", dtype=DTYPE):
                v_pred_list = compiled(
                    x_list, t=timestep, context=context_list, seq_len=seq_len
                )
                # Compute loss on each sample
                loss = 0
                for i in range(bsz):
                    loss = loss + F.mse_loss(v_pred_list[i], v_target[i])
                loss = loss / bsz / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or step == steps - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * grad_accum
            losses.append(loss_val)

            if step % 10 == 0 or step == steps - 1:
                elapsed = time.time() - t0
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                ms_per_step = elapsed / (step + 1) * 1000
                eta = (steps - step - 1) * ms_per_step / 60000
                print(f"  step {step:4d}/{steps}  loss={loss_val:.6f}  "
                      f"lr={cur_lr:.2e}  VRAM={vram_peak:.1f}GB  "
                      f"{ms_per_step:.0f}ms/step  ETA={eta:.1f}min")

            step += 1

            # Pilot: check after 5 steps
            if pilot and step >= 5:
                vram_peak = torch.cuda.max_memory_allocated() / 1e9
                vram_pct = vram_peak / 96 * 100
                print(f"\n  *** PILOT COMPLETE ***")
                print(f"  VRAM peak: {vram_peak:.1f} GB ({vram_pct:.0f}%)")
                print(f"  Loss: {loss_val:.6f}")
                if vram_pct < 80:
                    print(f"  WARNING: VRAM < 80%, increase batch_size!")
                break

        if pilot and step >= 5:
            break

    elapsed = time.time() - t0
    avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n  Done: {step} steps in {elapsed:.0f}s, avg_loss={avg_loss:.6f}, VRAM={vram_peak:.1f}GB")

    # Save results
    result = {
        "method": method, "tau": tau, "base_t": base_t,
        "steps": step, "final_loss": avg_loss, "all_losses": losses,
        "elapsed_s": elapsed, "vram_peak_gb": vram_peak,
        "batch_size": batch_size, "grad_accum": grad_accum,
        "lora_rank": lora_rank, "lr": lr, "seed": seed,
        "inv_freq_t": inv_freq_t.tolist(),
    }
    with open(out_path / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save LoRA
    if not pilot:
        model.save_pretrained(str(out_path / "lora"))
        print(f"  Saved LoRA to {out_path / 'lora'}")

    # ---- Cleanup (AIHANDOFF #6) ----
    model.cpu()
    del model, compiled, optimizer, trainable, dataset, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  Memory released: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wan2.1 EVQ h2h fine-tuning")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Wan2.1-T2V-1.3B weights dir")
    parser.add_argument("--wan_repo", type=str, default="",
                        help="Path to Wan2.1 code repo (default: model_dir/../Wan2.1)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/hybrid-rope/results/wan21_evq")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size. 96GB+LoRA+grad_ckpt: bs=8 (~79GB, 82%)")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="5-step verification only")
    args = parser.parse_args()

    if args.pilot:
        args.steps = 5
        print("\n*** PILOT MODE: 5 steps only ***\n")

    # Set Wan2.1 repo path for model loading
    if args.wan_repo:
        os.environ["WAN_REPO"] = args.wan_repo

    # h2h configs: all in ONE script, SAME data, SAME seed
    # Primary comparison: EVQ drop-in at base=10000 (no hyperparameter change)
    configs = [
        ("geo", 0.0, 10000.0),    # Original GEO (baseline)
        ("evq", 1.5, 10000.0),    # EVQ drop-in (same base, only change freq allocation)
    ]

    all_results = {}
    for method, tau, base_t in configs:
        result = train_one_config(
            model_dir=args.model_dir,
            data_dir=args.data_dir,
            method=method, tau=tau, base_t=base_t,
            steps=args.steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            lora_rank=args.lora_rank,
            output_dir=args.output_dir,
            use_compile=not args.no_compile,
            seed=args.seed,
            pilot=args.pilot,
        )
        key = f"{method}_tau{tau}_base{int(base_t)}"
        all_results[key] = result

    # Summary
    if not args.pilot and len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD SUMMARY")
        print(f"{'='*60}")
        for name, r in all_results.items():
            print(f"  {name}: loss={r['final_loss']:.6f} ({r['elapsed_s']:.0f}s, {r['vram_peak_gb']:.1f}GB)")

        # Compare
        geo10k = all_results.get("geo_tau0.0_base10000", {}).get("final_loss", 0)
        geo1k = all_results.get("geo_tau0.0_base1000", {}).get("final_loss", 0)
        evq1k = all_results.get("evq_tau1.5_base1000", {}).get("final_loss", 0)

        if geo10k > 0 and geo1k > 0:
            print(f"\n  GEO base1000 vs base10000: {(geo1k-geo10k)/geo10k*100:+.1f}%")
        if geo1k > 0 and evq1k > 0:
            print(f"  EVQ base1000 vs GEO base1000: {(evq1k-geo1k)/geo1k*100:+.1f}%")
        if geo10k > 0 and evq1k > 0:
            print(f"  EVQ base1000 vs GEO base10000: {(evq1k-geo10k)/geo10k*100:+.1f}%")

    summary_path = Path(args.output_dir) / "h2h_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
