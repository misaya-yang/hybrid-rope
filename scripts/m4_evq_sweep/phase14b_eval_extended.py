#!/usr/bin/env python3
"""
Phase 14B extended: Evaluate fine-tuned 750M checkpoints at 16K and 32K.
Loads the already fine-tuned models from Phase 14B and runs passkey eval
at longer contexts (no re-training needed).

Usage:
    python phase14b_eval_extended.py
"""

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST, set_seed
from eval_passkey_scratch import eval_passkey_nll_gap

BASE = 500_000.0
DIM = 64
TAU = 1.5

CFG_750M = dict(
    vocab_size=50304,
    hidden_size=1536,
    num_layers=18,
    num_heads=24,
    head_dim=64,
    intermediate_size=6144,
    max_position_embeddings=2048,
)

WORK = Path("/root/autodl-tmp/evq_phase14b")
DATA_CACHE_DIR = Path("/root/autodl-tmp/evq_phase9/data")

# Fine-tuned checkpoints from Phase 14B
FT_GEO = WORK / "geo_750m" / "finetuned.pt"
FT_HYBRID = WORK / "hybrid_750m" / "finetuned.pt"

# Original checkpoints (for pre-FT comparison if desired)
ORIG_GEO = Path("/root/autodl-tmp/evq_phase9/seed42/geo_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt")
ORIG_HYBRID = Path("/root/autodl-tmp/evq_phase9/seed42/hybrid1.5_r16_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt")


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=TAU, r=16):
    n = dim // 2
    geo = torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float64)
    n_evq = n - r
    if n_evq <= 0:
        return geo.float()
    theta_max = geo[r].item()
    theta_min = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_model(inv_freq, ckpt_path):
    cfg = CFG_750M.copy()
    model = GPT(cfg, inv_freq)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(DEVICE)
    return model


def eval_passkey(model, tokenizer, filler_tokens, lengths, tag, num_trials=20):
    """Evaluate passkey retrieval at given lengths."""
    print(f"\n  Evaluating: {tag}  lengths={lengths}")
    model.eval()
    with torch.no_grad():
        result = eval_passkey_nll_gap(
            model, tokenizer, filler_tokens,
            lengths=lengths,
            depths=[0.5],
            num_trials=num_trials,
            seed=42,
        )
    g = result.get("global", {})
    print(f"    {tag}: retrieval={g.get('retrieval_rate', 0):.4f}, gap={g.get('mean_nll_gap', 0):.4f}")
    return {
        "label": tag,
        "global": g,
        "summary": result.get("summary", {}),
    }


def run_eval(tag, inv_freq, ft_ckpt, orig_ckpt, filler, tok, eval_lengths, num_trials):
    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")

    results = {}

    # Eval original (pre-FT) checkpoint
    print(f"\n  Loading ORIGINAL checkpoint: {orig_ckpt}")
    model = load_model(inv_freq, orig_ckpt)
    results["pre_ft"] = eval_passkey(model, tok, filler, eval_lengths, f"{tag}_pre", num_trials)
    del model
    torch.cuda.empty_cache()

    # Eval fine-tuned checkpoint
    print(f"\n  Loading FINE-TUNED checkpoint: {ft_ckpt}")
    model = load_model(inv_freq, ft_ckpt)
    results["post_ft"] = eval_passkey(model, tok, filler, eval_lengths, f"{tag}_post", num_trials)
    del model
    torch.cuda.empty_cache()

    return results


def main():
    set_seed(42)

    eval_lengths = [2048, 4096, 8192, 16384, 32768]
    num_trials = 20

    print("#" * 70)
    print("  Phase 14B Extended: Passkey eval at 16K/32K")
    print(f"  lengths={eval_lengths}, trials={num_trials}")
    print("#" * 70)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    val_path = DATA_CACHE_DIR / "val_fineweb-edu_5000000.pt"
    print(f"\n  Loading filler tokens from: {val_path}")
    val_data = torch.load(val_path, map_location="cpu", weights_only=True)
    filler = val_data.reshape(-1)[:100000]
    print(f"  Filler tokens: {filler.shape}")

    all_results = {}

    all_results["geo"] = run_eval(
        "geo_750m", geometric_inv_freq(),
        FT_GEO, ORIG_GEO, filler, tok, eval_lengths, num_trials,
    )

    all_results["hybrid"] = run_eval(
        "hybrid_750m", hybrid_evq_inv_freq(),
        FT_HYBRID, ORIG_HYBRID, filler, tok, eval_lengths, num_trials,
    )

    # Summary table
    print(f"\n{'='*70}")
    print("  PHASE 14B EXTENDED SUMMARY")
    print(f"{'='*70}")

    for key in ["geo", "hybrid"]:
        r = all_results[key]
        print(f"\n  {key}_750m:")
        for phase in ["pre_ft", "post_ft"]:
            g = r[phase]["global"]
            label = "Pre-FT " if phase == "pre_ft" else "Post-FT"
            print(f"    {label}: retrieval={g.get('retrieval_rate', 0):.4f}, gap={g.get('mean_nll_gap', 0):.4f}")
            # per-length details
            summary = r[phase].get("summary", {})
            for sk in sorted(summary.keys()):
                s = summary[sk]
                print(f"      {sk}: ret={s.get('retrieval_rate', 0):.2f}, gap={s.get('mean_nll_gap', 0):.3f}, AR={s.get('ar_exact_match_rate', 0):.2f}")

    save_json(WORK / "phase14b_extended_summary.json", all_results)
    print(f"\n  Saved: {WORK}/phase14b_extended_summary.json")


if __name__ == "__main__":
    main()
