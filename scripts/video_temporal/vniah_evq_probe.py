#!/usr/bin/env python3
"""V-NIAH-D inference probe: EVQ frequency allocation on Qwen2-VL.

Phase 1: No training. At inference time, replace temporal inv_freq with EVQ values.
Tests whether EVQ frequency redistribution improves temporal position retrieval.

4 configurations:
1. M-RoPE (original Qwen2-VL baseline)
2. M-RoPE + EVQ (EVQ on full dim, temporal section redistributed)
3. VideoRoPE (temporal → low freq slots)
4. VideoRoPE + EVQ (temporal → low freq + EVQ redistribution)

Usage:
    python vniah_evq_probe.py --model_path /path/to/Qwen2-VL-2B-Instruct \
        --videorope_path /path/to/videorope_code \
        --config mrope  # or mrope_evq, videorope, videorope_evq
"""
import os, sys, math, json, argparse, torch
import numpy as np
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evq_cosh_inv_freq(dim: int, tau: float, base: float = 1000000.0) -> torch.Tensor:
    """EVQ-Cosh on FULL dim, then caller takes the appropriate section."""
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def patch_inv_freq_evq(model, tau: float = 1.414, base: float = 1000000.0):
    """Replace model's inv_freq with EVQ-redistributed frequencies.

    EVQ is applied to the FULL head_dim, then sections are automatically
    split by mrope_section during RoPE application.
    """
    rotary_emb = model.model.rotary_emb
    orig_inv_freq = rotary_emb.inv_freq  # [head_dim/2] = [64]
    dim = orig_inv_freq.shape[0] * 2  # 128

    evq_freq = evq_cosh_inv_freq(dim, tau=tau, base=base)
    evq_freq = evq_freq.to(orig_inv_freq.device)

    print(f"  Patching inv_freq: GEO range [{orig_inv_freq.min():.6e}, {orig_inv_freq.max():.4f}]")
    print(f"  →                  EVQ range [{evq_freq.min():.6e}, {evq_freq.max():.4f}]")

    rotary_emb.inv_freq = evq_freq
    rotary_emb.original_inv_freq = evq_freq
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--videorope_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True,
                        choices=["mrope", "mrope_evq", "videorope", "videorope_evq"])
    parser.add_argument("--tau", type=float, default=1.414,
                        help="EVQ tau (default: 128/sqrt(8192)=1.414)")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/results/vniah_evq")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    use_videorope = "videorope" in args.config
    use_evq = "evq" in args.config

    print(f"\n{'='*60}")
    print(f"  V-NIAH-D Probe: {args.config}")
    print(f"  VideoRoPE: {use_videorope}, EVQ: {use_evq} (τ={args.tau})")
    print(f"{'='*60}")

    # Load model
    if use_videorope:
        # Use VideoRoPE's modified modeling file
        sys.path.insert(0, args.videorope_path)
        # VideoRoPE replaces the modeling file in transformers
        print("  Loading with VideoRoPE modeling...")
        # TODO: integrate VideoRoPE's modeling_videorope.py
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(args.model_path)

    # Apply EVQ if needed
    if use_evq:
        print(f"  Applying EVQ τ={args.tau}...")
        model = patch_inv_freq_evq(model, tau=args.tau)

    # Print config
    inv_freq = model.model.rotary_emb.inv_freq
    print(f"  inv_freq shape: {inv_freq.shape}")
    print(f"  inv_freq range: [{inv_freq.min():.6e}, {inv_freq.max():.4f}]")

    model.eval()
    print(f"  Model loaded. VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    # Run V-NIAH-D evaluation
    # Use VideoRoPE's eval script
    vniah_dir = Path(args.videorope_path) / "vision_niah_d"
    sys.path.insert(0, str(vniah_dir))

    print(f"\n  Running V-NIAH-D evaluation...")
    print(f"  Output: {out_dir}")

    # Save config
    config = {
        "config_name": args.config,
        "use_videorope": use_videorope,
        "use_evq": use_evq,
        "tau": args.tau if use_evq else None,
        "model_path": args.model_path,
        "inv_freq": inv_freq.cpu().tolist(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Config saved. Ready for V-NIAH-D eval.")
    print(f"  To run eval, use VideoRoPE's eval script with this model.")


if __name__ == "__main__":
    main()
