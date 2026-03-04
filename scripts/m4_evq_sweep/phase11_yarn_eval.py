#!/usr/bin/env python3
"""
Phase 11 YaRN Post-Eval: Load trained checkpoints, apply YaRN scaling, re-evaluate PPL.
Tests EVQ+YaRN synergy vs Geo+YaRN at various extrapolation ratios.
"""

import json, math, os, sys, gc
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST
from phase11_L256_extrap import (
    geometric_inv_freq, evq_cosh_inv_freq, hybrid_evq_inv_freq,
    CFG_350M, EVAL_LENGTHS,
)

WORK = Path("/root/autodl-tmp/evq_phase11_L256")
TRAIN_LEN = 256
BASE = 500_000.0
DIM = 64


def build_yarn_inv_freq(base_inv, head_dim, scale):
    """YaRN progressive frequency scaling with smoothstep ramp."""
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()


def eval_ppl_with_inv_freq(model, val_data, inv_freq, eval_lengths, n_chunks=8):
    """Evaluate PPL after swapping in new inv_freq."""
    # Swap inv_freq in all layers
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq)
    model.blocks[0].attn.rope._build(max(eval_lengths) + 100)

    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)
    results = {}

    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            continue
        offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
        for offset in offsets:
            chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           chunk[:, 1:].reshape(-1))
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM, skipping")
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del chunk
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L:>5d}: PPL={ppl:.2f}  ({len(losses)} chunks)")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137,256")
    parser.add_argument("--methods", default="geo,evq2.0,evq4.0")
    parser.add_argument("--r", type=int, default=16)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    # Load val data
    val_path = Path("/root/autodl-tmp/evq_phase9/data/val_fineweb-edu_5000000.pt")
    print(f"Loading val data from {val_path}...")
    val_data = torch.load(val_path, weights_only=True)

    cfg = CFG_350M.copy()
    cfg["seq_len"] = TRAIN_LEN
    cfg["max_position_embeddings"] = TRAIN_LEN

    # Build base inv_freqs
    base_inv_freqs = {}
    for m in methods:
        if m == "geo":
            base_inv_freqs[m] = geometric_inv_freq()
        elif m.startswith("evq"):
            tau = float(m.replace("evq", ""))
            base_inv_freqs[m] = evq_cosh_inv_freq(tau=tau)
        elif m.startswith("hybrid"):
            tau = float(m.replace("hybrid", ""))
            base_inv_freqs[m] = hybrid_evq_inv_freq(tau=tau, r=args.r)

    all_results = {}

    for method in methods:
        for seed in seeds:
            run_id = f"350m_{method}_seed{seed}"
            model_path = WORK / run_id / "model.pt"

            if not model_path.exists():
                print(f"\n  SKIP {run_id} (no checkpoint)")
                continue

            result_path = WORK / run_id / "yarn_eval.json"
            if result_path.exists():
                print(f"\n  SKIP {run_id} (yarn eval exists)")
                with open(result_path) as f:
                    all_results[run_id] = json.load(f)
                continue

            print(f"\n{'='*70}")
            print(f"  YARN EVAL: {run_id}")
            print(f"{'='*70}")

            # Load model
            base_inv = base_inv_freqs[method]
            model = GPT(cfg, base_inv).to(DEVICE)
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)

            result = {"run_id": run_id, "method": method, "seed": seed}

            # For each eval length > TRAIN_LEN, apply YaRN with appropriate scale
            for yarn_scale_name, yarn_scales in [
                ("yarn_auto", None),  # auto scale = L_eval / L_train
            ]:
                yarn_ppl = {}
                for L in EVAL_LENGTHS:
                    if L <= TRAIN_LEN:
                        # No YaRN needed for in-distribution
                        continue
                    scale = L / TRAIN_LEN
                    yarn_inv = build_yarn_inv_freq(base_inv, DIM, scale)
                    # Eval just this length
                    ppl = eval_ppl_with_inv_freq(model, val_data, yarn_inv, [L])
                    yarn_ppl.update(ppl)

                # Also eval at TRAIN_LEN with no YaRN (baseline)
                ppl_base = eval_ppl_with_inv_freq(model, val_data, base_inv, [TRAIN_LEN])
                yarn_ppl.update(ppl_base)

                result[yarn_scale_name] = yarn_ppl
                print(f"  {yarn_scale_name}: {yarn_ppl}")

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results[run_id] = result

            del model, state
            gc.collect()
            torch.cuda.empty_cache()

    # Save aggregate
    agg_path = WORK / "yarn_eval_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*90}")
    print(f"  YARN EVAL RESULTS: L_train={TRAIN_LEN}, 350M")
    print(f"{'='*90}")

    for method in methods:
        print(f"\n  {method.upper()} + YaRN(auto):")
        header = "    " + " ".join(f"{'L='+str(L):>10s}" for L in EVAL_LENGTHS)
        print(header)

        for seed in seeds:
            r = all_results.get(f"350m_{method}_seed{seed}", {})
            yarn = r.get("yarn_auto", {})
            # Also load raw PPL for comparison
            raw_path = WORK / f"350m_{method}_seed{seed}" / "result.json"
            raw_ppl = {}
            if raw_path.exists():
                with open(raw_path) as f:
                    raw_ppl = json.load(f).get("ppl", {})

            raw_vals = " ".join(f"{raw_ppl.get(str(L), 0):>10.1f}" for L in EVAL_LENGTHS)
            yarn_vals = " ".join(f"{yarn.get(str(L), 0):>10.1f}" for L in EVAL_LENGTHS)
            print(f"    raw s={seed:>3d}: {raw_vals}")
            print(f"    yarn s={seed:>3d}: {yarn_vals}")

    # Delta: YaRN improvement
    print(f"\n  YaRN IMPROVEMENT (PPL reduction %):")
    for method in methods:
        print(f"    {method.upper():>10s}: ", end="")
        for L in EVAL_LENGTHS:
            if L <= TRAIN_LEN:
                print(f"{'--':>10s}", end=" ")
                continue
            raw_ppls = []
            yarn_ppls = []
            for seed in seeds:
                raw_path = WORK / f"350m_{method}_seed{seed}" / "result.json"
                if raw_path.exists():
                    with open(raw_path) as f:
                        rp = json.load(f).get("ppl", {}).get(str(L))
                        if rp: raw_ppls.append(rp)
                yr = all_results.get(f"350m_{method}_seed{seed}", {}).get("yarn_auto", {}).get(str(L))
                if yr: yarn_ppls.append(yr)
            if raw_ppls and yarn_ppls:
                raw_m = sum(raw_ppls) / len(raw_ppls)
                yarn_m = sum(yarn_ppls) / len(yarn_ppls)
                delta = (yarn_m / raw_m - 1) * 100
                print(f"{delta:>+9.1f}%", end=" ")
            else:
                print(f"{'N/A':>10s}", end=" ")
        print()

    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
