#!/usr/bin/env python3
"""Phase 7A: YaRN implementation verification + hyperparameter ablation."""

import sys, os, json, math, time
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_evq_sweep import (
    GPT, RotaryEmbedding, evq_cosh_inv_freq, yarn_inv_freq,
    TIER_CONFIGS, DEVICE, DTYPE, USE_AUTOCAST, eval_model,
)
import torch

CFG = TIER_CONFIGS["125m"].copy()
CFG["seq_len"] = 128
CFG["max_position_embeddings"] = 128
BASE = 500000.0


def verify_yarn():
    """7A-1: Print and verify YaRN frequencies."""
    dim = 64
    geo = 1.0 / (BASE ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    yarn = yarn_inv_freq(dim, BASE, original_max_position=128, target_max_position=8192)
    evq = evq_cosh_inv_freq(dim, 2.5, BASE)

    s = 8192.0 / 128
    low = math.floor(dim * math.log(128 / (32 * 2 * math.pi)) / (2 * math.log(BASE)))
    high = math.ceil(dim * math.log(128 / (1 * 2 * math.pi)) / (2 * math.log(BASE)))
    low = max(low, 0)
    high = min(high, dim // 2 - 1)

    print("=" * 80)
    print("7A-1: Frequency Comparison (32 channels)")
    print("=" * 80)
    print(f"{'Ch':>3} {'Geometric':>14} {'YaRN':>14} {'EVQ2.5':>14} {'YaRN/Geo':>10} {'Region':>10}")
    print("-" * 80)
    for i in range(32):
        ratio = yarn[i].item() / geo[i].item()
        if i < low:
            region = "high-freq"
        elif i > high:
            region = "low-freq"
        else:
            region = "interp"
        print(f"{i:>3d} {geo[i].item():>14.8f} {yarn[i].item():>14.8f} "
              f"{evq[i].item():>14.8f} {ratio:>10.4f} {region:>10}")

    print(f"\nYaRN boundaries: low={low}, high={high}, scale={s:.0f}x")
    print(f"  Channels 0..{max(low-1,0)}: unchanged (high freq)")
    print(f"  Channels {low}..{high}: interpolated")
    print(f"  Channels {high+1}..31: divided by {s:.0f} (low freq / PI)")

    return {
        "geometric": geo.tolist(),
        "yarn_default": yarn.tolist(),
        "evq_2.5": evq.tolist(),
        "yarn_geo_ratio": (yarn / geo.float()).tolist(),
        "boundaries": {"low": low, "high": high, "scale": s},
    }


def yarn_ablation():
    """7A-2: Eval Geometric checkpoint with different YaRN configs."""
    ckpt_path = "/root/autodl-tmp/evq_128tok/125m_tau0.00_seed42/model.pt"
    val_path = "/root/autodl-tmp/evq_128tok/val_fineweb-edu_5000000.pt"
    val_data = torch.load(val_path, weights_only=True)

    configs = {
        "Y1_default":        {"beta_fast": 32, "beta_slow": 1},
        "Y2_more_unchanged": {"beta_fast": 16, "beta_slow": 1},
        "Y3_less_unchanged": {"beta_fast": 64, "beta_slow": 2},
    }
    results = {}

    for name, yc in configs.items():
        print(f"\n{'='*60}")
        print(f"  YaRN Ablation: {name} (beta_fast={yc['beta_fast']}, beta_slow={yc['beta_slow']})")
        print(f"{'='*60}")
        inv = yarn_inv_freq(64, BASE, 128, 8192, yc["beta_fast"], yc["beta_slow"])

        geo_inv = evq_cosh_inv_freq(64, 0.0, BASE)
        model = GPT(CFG, geo_inv).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt, strict=False)
        model.blocks[0].attn.rope.inv_freq.copy_(inv)
        model.blocks[0].attn.rope._build(CFG["max_position_embeddings"])

        ppl = eval_model(model, val_data, [128, 2048, 4096, 8192], 10)
        results[name] = {"config": yc, "ppl": ppl}
        print(f"  -> PPL@128={ppl.get('128','N/A')}  PPL@8K={ppl.get('8192','N/A')}")
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Y4: NTK-aware (inference-time)
    print(f"\n{'='*60}")
    print(f"  YaRN Ablation: Y4_NTK_aware")
    print(f"{'='*60}")
    s = 8192.0 / 128
    ntk_base = BASE * (s ** (64 / 62))
    print(f"  NTK base = {ntk_base:.0f}")
    ntk_inv = 1.0 / (ntk_base ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    ntk_inv = ntk_inv.float()

    geo_inv = evq_cosh_inv_freq(64, 0.0, BASE)
    model = GPT(CFG, geo_inv).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.blocks[0].attn.rope.inv_freq.copy_(ntk_inv)
    model.blocks[0].attn.rope._build(CFG["max_position_embeddings"])

    ppl = eval_model(model, val_data, [128, 2048, 4096, 8192], 10)
    results["Y4_NTK_aware"] = {"config": {"method": "NTK-aware", "base": ntk_base}, "ppl": ppl}
    print(f"  -> PPL@128={ppl.get('128','N/A')}  PPL@8K={ppl.get('8192','N/A')}")
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()

    return results


def main():
    out = Path("/root/autodl-tmp/evq_phase7/yarn_verification")
    out.mkdir(parents=True, exist_ok=True)
    verify = verify_yarn()
    ablation = yarn_ablation()
    with open(out / "results.json", "w") as f:
        json.dump({"verification": verify, "ablation": ablation}, f, indent=2, default=str)
    print(f"\n7A COMPLETE — saved to {out / 'results.json'}")
    for n, r in ablation.items():
        p = r.get("ppl", {})
        print(f"  {n:25s}  PPL@128={p.get('128','?'):>8}  PPL@8K={p.get('8192','?'):>8}")


if __name__ == "__main__":
    main()
