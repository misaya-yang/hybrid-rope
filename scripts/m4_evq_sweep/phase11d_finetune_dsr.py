#!/usr/bin/env python3
"""
Phase 11D: Fine-tune Phase 11 checkpoints with passkey data, then run DSR.

Takes existing 454M L=256 checkpoints, fine-tunes for ~100 steps with 50% passkey
mix at L=256, then evaluates DSR at L=2048 (8× extrapolation).

This isolates the RoPE frequency effect on retrieval without full retraining.
"""

import json, math, os, sys, time, gc
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, DEVICE, DTYPE, USE_AUTOCAST,
    set_seed, get_batch_from_data,
)
from eval_passkey_scratch import (
    make_passkey_training_sample, MixedDataset,
    build_passkey_eval_sequence, PASSKEY_PREFIX, PASSKEY_SUFFIX, DASH,
)
from phase11_L256_extrap import (
    geometric_inv_freq, evq_cosh_inv_freq,
    CFG_350M, EVAL_LENGTHS,
    load_validation_data,
)

WORK = Path("/root/autodl-tmp/evq_phase11d_dsr")
PHASE11_DIR = Path("/root/autodl-tmp/evq_phase11_L256")
PHASE9_DATA = Path("/root/autodl-tmp/evq_phase9/data")

SEQ_LEN = 256
BASE = 500_000.0
DIM = 64

# DSR config
DSR_EVAL_LEN = 2048  # 8× extrapolation
DSR_DISTANCES = [128, 256, 384, 512, 768, 1024, 1536, 2048]  # 0.5×-8× L_train


def load_lm_data():
    """Load LM data for fine-tuning mix."""
    cache = PHASE11_DIR / "data" / "train_fineweb-edu_105000000_256.pt"
    if cache.exists():
        print(f"  LM data from {cache}")
        return torch.load(cache, weights_only=True)
    # Fallback
    raw = torch.load(PHASE9_DATA / "train_fineweb-edu_2000000000_2048.pt", weights_only=True)
    flat = raw.reshape(-1)[:10_000_000]
    n = len(flat) // SEQ_LEN
    return flat[:n * SEQ_LEN].reshape(n, SEQ_LEN)


def finetune_with_passkey(model, lm_data, filler_tokens, tokenizer,
                          num_steps=100, lr=1e-4, passkey_ratio=0.5,
                          micro_batch=64, seed=42):
    """Fine-tune model with passkey mix for a few steps."""
    set_seed(seed)
    dataset = MixedDataset(lm_data, filler_tokens, tokenizer,
                           passkey_ratio=passkey_ratio, seq_len=SEQ_LEN)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95), fused=True)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    model.train()
    t0 = time.time()
    rng = np.random.RandomState(seed)

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)
        indices = rng.randint(0, len(dataset), size=micro_batch)
        batch = torch.stack([dataset[int(i)] for i in indices], dim=0).to(DEVICE)

        with ctx:
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"    ft step {step:>4d}/{num_steps} | loss={loss.item():.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Fine-tune done in {elapsed:.0f}s")
    return elapsed


def eval_passkey_nll(model, filler_tokens, tokenizer, eval_len, distance,
                     num_trials=40, seed=42):
    """Evaluate passkey retrieval at specific distance using NLL gap."""
    model.eval()
    model.extend_rope(eval_len + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(seed)

    # Pre-compute answer length
    sample_answer = f"0{DASH}0{DASH}0{DASH}0{DASH}0{PASSKEY_SUFFIX}"
    answer_toks = tokenizer.encode(sample_answer, add_special_tokens=False)
    n_answer = len(answer_toks)

    correct_nlls = []
    wrong_nlls = []

    for t in range(num_trials):
        # Generate passkey
        passkey = DASH.join([str(rng.randint(0, 9)) for _ in range(5)])
        wrong_pk = DASH.join([str((int(d) + rng.randint(1, 9)) % 10) for d in passkey.split(DASH)])

        # Compute depth from distance
        full_marker = f"{PASSKEY_PREFIX}{passkey}{PASSKEY_SUFFIX}"
        marker_len = len(tokenizer.encode(full_marker, add_special_tokens=False))
        probe_len = len(tokenizer.encode(PASSKEY_PREFIX, add_special_tokens=False))
        filler_budget = eval_len - marker_len - probe_len
        after_len = max(0, distance - marker_len)
        before_len = max(0, filler_budget - after_len)
        depth = before_len / max(filler_budget, 1)

        filler_offset = rng.randint(0, max(1, len(filler_tokens) - eval_len))
        trial_filler = filler_tokens[filler_offset:]

        correct_ids, pk_start, probe_start = build_passkey_eval_sequence(
            trial_filler, passkey, tokenizer, eval_len, depth
        )

        # Build wrong version
        wrong_ids = correct_ids.clone()
        wrong_marker_toks = tokenizer.encode(
            f"{PASSKEY_PREFIX}{wrong_pk}{PASSKEY_SUFFIX}", add_special_tokens=False
        )
        for i, tok in enumerate(wrong_marker_toks):
            if pk_start + i < len(wrong_ids):
                wrong_ids[pk_start + i] = tok

        # Compute NLL at probe positions
        for ids, nll_list in [(correct_ids, correct_nlls), (wrong_ids, wrong_nlls)]:
            inp = ids.unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(inp[:, :-1])
                    # NLL only at answer positions
                    n_valid = min(n_answer, logits.size(1) - probe_start)
                    if n_valid > 0:
                        logits_ans = logits[0, probe_start:probe_start + n_valid]
                        target_ans = ids[probe_start + 1:probe_start + 1 + n_valid].to(DEVICE)
                        nll = F.cross_entropy(logits_ans, target_ans).item()
                        nll_list.append(nll)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del inp

    if not correct_nlls:
        return None

    # Compute retrieval rate and mean gap
    gaps = [w - c for c, w in zip(correct_nlls, wrong_nlls)]
    retrieval_rate = sum(1 for g in gaps if g > 0) / len(gaps)
    mean_gap = sum(gaps) / len(gaps)

    return {
        "distance": distance,
        "distance_ratio": round(distance / SEQ_LEN, 2),
        "retrieval_rate": round(retrieval_rate, 4),
        "mean_nll_gap": round(mean_gap, 4),
        "num_trials": len(gaps),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="geo,evq4.0")
    parser.add_argument("--seeds", default="42,137,256")
    parser.add_argument("--ft_steps", type=int, default=100)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--passkey_ratio", type=float, default=0.5)
    parser.add_argument("--num_trials", type=int, default=40)
    args = parser.parse_args()

    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    WORK.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load data
    print("[1] Loading data...")
    lm_data = load_lm_data()
    val_data = load_validation_data()
    filler_tokens = val_data  # use val as filler

    print(f"  LM data: {lm_data.shape}")
    print(f"  Filler: {filler_tokens.shape}")

    cfg = CFG_350M.copy()
    cfg["seq_len"] = SEQ_LEN
    cfg["max_position_embeddings"] = SEQ_LEN

    # Base inv_freqs
    inv_freqs = {}
    for m in methods:
        if m == "geo":
            inv_freqs[m] = geometric_inv_freq()
        elif m.startswith("evq"):
            tau = float(m.replace("evq", ""))
            inv_freqs[m] = evq_cosh_inv_freq(tau=tau)

    all_results = {}

    for method in methods:
        for seed in seeds:
            run_id = f"454m_{method}_seed{seed}"
            src_model = PHASE11_DIR / f"350m_{method}_seed{seed}" / "model.pt"
            result_path = WORK / f"{run_id}_dsr.json"

            if result_path.exists():
                print(f"\n  SKIP {run_id} (result exists)")
                with open(result_path) as f:
                    all_results[run_id] = json.load(f)
                continue

            if not src_model.exists():
                print(f"\n  SKIP {run_id} (no checkpoint at {src_model})")
                continue

            print(f"\n{'='*70}")
            print(f"  {run_id}: Fine-tune + DSR")
            print(f"{'='*70}")

            # Load model
            inv_freq = inv_freqs[method]
            model = GPT(cfg, inv_freq).to(DEVICE)
            state = torch.load(str(src_model), map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            del state

            # Fine-tune with passkey
            print(f"\n  [1] Fine-tuning {args.ft_steps} steps (passkey_ratio={args.passkey_ratio})...")
            finetune_with_passkey(model, lm_data, filler_tokens, tok,
                                num_steps=args.ft_steps, lr=args.ft_lr,
                                passkey_ratio=args.passkey_ratio, seed=seed)

            # Save fine-tuned model
            ft_dir = WORK / run_id
            ft_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ft_dir / "model_ft.pt")

            # Quick PPL check (should be close to original)
            print(f"\n  [2] PPL check after fine-tune...")
            model.eval()
            from phase11_L256_extrap import eval_ppl
            ppl = eval_ppl(model, val_data, eval_lengths=[256, 512, 1024, 2048])

            # DSR eval at 2048 (8× extrapolation)
            print(f"\n  [3] DSR at L={DSR_EVAL_LEN} (8× extrapolation)...")
            dsr_results = {}
            for dist in DSR_DISTANCES:
                r = eval_passkey_nll(model, filler_tokens, tok,
                                     DSR_EVAL_LEN, dist,
                                     num_trials=args.num_trials, seed=seed)
                if r:
                    ratio_str = f"D={dist}_R={r['distance_ratio']:.1f}x"
                    dsr_results[ratio_str] = r
                    print(f"    Δ={dist:>5d} ({r['distance_ratio']:.1f}×): "
                          f"acc={r['retrieval_rate']*100:5.1f}%  gap={r['mean_nll_gap']:+.3f}")

            result = {
                "run_id": run_id,
                "method": method,
                "seed": seed,
                "ft_steps": args.ft_steps,
                "ppl_after_ft": ppl,
                "dsr": dsr_results,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results[run_id] = result

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  PHASE 11D: Fine-tune + DSR Summary")
    print(f"{sep}")

    for method in methods:
        print(f"\n  Method: {method}")
        for dist in DSR_DISTANCES:
            ratio = dist / SEQ_LEN
            accs = []
            gaps = []
            for seed in seeds:
                r = all_results.get(f"454m_{method}_seed{seed}", {})
                dsr = r.get("dsr", {})
                key = f"D={dist}_R={ratio:.1f}x"
                if key in dsr:
                    accs.append(dsr[key]["retrieval_rate"])
                    gaps.append(dsr[key]["mean_nll_gap"])
            if accs:
                mean_acc = sum(accs) / len(accs)
                mean_gap = sum(gaps) / len(gaps)
                print(f"    Δ={dist:>5d} ({ratio:.1f}×): "
                      f"acc={mean_acc*100:5.1f}% ± {np.std(accs)*100:.1f}  "
                      f"gap={mean_gap:+.4f}")

    # Cross-method comparison
    print(f"\n  {'─'*60}")
    print(f"  DSR Comparison: Mean Retrieval Rate (%)")
    print(f"  {'─'*60}")
    header = f"  {'Method':>12s}"
    for dist in DSR_DISTANCES:
        header += f"  {dist/SEQ_LEN:.1f}×"
    print(header)

    for method in methods:
        line = f"  {method:>12s}"
        for dist in DSR_DISTANCES:
            ratio = dist / SEQ_LEN
            accs = []
            for seed in seeds:
                r = all_results.get(f"454m_{method}_seed{seed}", {})
                dsr = r.get("dsr", {})
                key = f"D={dist}_R={ratio:.1f}x"
                if key in dsr:
                    accs.append(dsr[key]["retrieval_rate"])
            if accs:
                line += f"  {sum(accs)/len(accs)*100:4.0f}"
            else:
                line += f"  {'—':>4}"
        print(line)

    agg_path = WORK / "dsr_all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
