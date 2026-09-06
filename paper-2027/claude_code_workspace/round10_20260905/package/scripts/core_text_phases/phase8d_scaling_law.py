#!/usr/bin/env python3
"""Phase 8D: τ* Scaling Law Verification.

Verify τ*(L_train) = d_head / √L_train = 64/√L conjecture.
D1: L=256, τ sweep {0.0, 2.0, 3.0, 4.0, 5.0}
D2: L=512, τ sweep {0.0, 1.5, 2.0, 2.83, 3.5, 4.0}
Model: 125M (head_dim=64), 50M tokens, from-scratch.
"""

import sys, os, json, math, time
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_evq_sweep import (
    GPT, RotaryEmbedding, evq_cosh_inv_freq,
    TIER_CONFIGS, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, train_model, set_seed, load_data, load_val,
)

import torch
import numpy as np

BASE = 500_000.0
TRAIN_TOKENS = 50_000_000
SEED = 42
LR = 6e-4

WORK_DIR = Path("/root/autodl-tmp/evq_phase8/scaling_law")


def run_single_d(seq_len, tau, eval_lengths, work_dir, batch_size):
    """Train a single from-scratch 125M model and eval."""
    run_name = f"L{seq_len}_tau{tau:.2f}"
    run_dir = work_dir / run_name
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n  [SKIP] {run_name} already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  {run_name}: from-scratch 125M, L={seq_len}, τ={tau}")
    print(f"{'='*60}")

    cfg = TIER_CONFIGS["125m"].copy()
    cfg["seq_len"] = seq_len
    cfg["max_position_embeddings"] = seq_len
    cfg["train_tokens"] = TRAIN_TOKENS
    cfg["lr"] = LR
    cfg["batch_size"] = batch_size
    cfg["eval_lengths"] = eval_lengths

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Data
    data_dir = work_dir / f"data_L{seq_len}"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, TRAIN_TOKENS, seq_len, "fineweb-edu",
                           cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))

    # Model
    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, BASE)
    set_seed(SEED)
    model = GPT(cfg, inv_freq).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {n_params:.1f}M")

    # Train
    t0 = time.time()
    model = train_model(model, train_data, cfg, seed=SEED)
    train_time = time.time() - t0
    print(f"  Train time: {train_time/60:.1f} min")

    # Eval
    ppl = eval_model(model, val_data, eval_lengths, 10)

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "run_name": run_name,
        "seq_len": seq_len,
        "tau": tau,
        "ppl": ppl,
        "train_time_sec": round(train_time, 1),
    }
    with open(result_file, "w") as f:
        json.dump(res, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return res


def run_d1(work_dir):
    """D1: L_train=256, τ sweep {0.0, 2.0, 3.0, 4.0, 5.0}."""
    print(f"\n{'#'*60}")
    print(f"  8D-1: L_train=256, τ sweep")
    print(f"{'#'*60}")

    seq_len = 256
    eval_lengths = [256, 512, 1024, 2048, 4096, 8192]
    taus = [0.0, 2.0, 3.0, 4.0, 5.0]
    # Short sequences → can use large batch
    batch_size = 8

    results = {}
    for tau in taus:
        res = run_single_d(seq_len, tau, eval_lengths, work_dir, batch_size)
        results[f"tau_{tau}"] = res

    # Find τ* (best PPL@2048 = 8×L)
    best_tau = None
    best_ppl = float("inf")
    for tau in taus:
        key = f"tau_{tau}"
        ppl_2048 = results[key]["ppl"].get("2048", float("inf"))
        if isinstance(ppl_2048, (int, float)) and ppl_2048 < best_ppl:
            best_ppl = ppl_2048
            best_tau = tau

    print(f"\n  D1 τ* (best PPL@2048) = {best_tau}  (PPL={best_ppl:.1f})")
    return results, best_tau


def run_d2(work_dir):
    """D2: L_train=512, τ sweep {0.0, 1.5, 2.0, 2.83, 3.5, 4.0}."""
    print(f"\n{'#'*60}")
    print(f"  8D-2: L_train=512, τ sweep")
    print(f"{'#'*60}")

    seq_len = 512
    eval_lengths = [512, 1024, 2048, 4096, 8192]
    taus = [0.0, 1.5, 2.0, 64.0 / math.sqrt(512), 3.5, 4.0]  # 2.83 = 64/√512
    batch_size = 8

    results = {}
    for tau in taus:
        tau_round = round(tau, 2)
        res = run_single_d(seq_len, tau_round, eval_lengths, work_dir, batch_size)
        results[f"tau_{tau_round}"] = res

    # Find τ* (best PPL@4096 = 8×L)
    best_tau = None
    best_ppl = float("inf")
    for tau in taus:
        tau_round = round(tau, 2)
        key = f"tau_{tau_round}"
        ppl_4096 = results[key]["ppl"].get("4096", float("inf"))
        if isinstance(ppl_4096, (int, float)) and ppl_4096 < best_ppl:
            best_ppl = ppl_4096
            best_tau = tau_round

    print(f"\n  D2 τ* (best PPL@4096) = {best_tau}  (PPL={best_ppl:.1f})")
    return results, best_tau


def compute_scaling_law_fit(d1_tau_star, d2_tau_star):
    """Fit τ* vs 1/√L using all data points. Report R²."""
    import numpy as np

    # All data points: (L, τ*)
    data_points = [
        # Existing from Phase 6/7
        (128, 5.0),       # Phase 6: monotonically falling, >5.0; use 5.0 as lower bound
        (1024, 2.0),      # Phase 6 1024-tok
        (2048, 1.5),      # Phase 7F context extension
        # New from 8D
        (256, d1_tau_star),
        (512, d2_tau_star),
    ]

    # τ* = C / √L  →  τ* = C × (1/√L)  →  linear in 1/√L
    x = np.array([1.0 / np.sqrt(L) for L, _ in data_points])
    y = np.array([tau for _, tau in data_points])

    # Linear fit: y = slope * x + intercept (expect intercept ≈ 0)
    from numpy.polynomial import polynomial as P
    # Using np.polyfit for slope, intercept
    slope, intercept = np.polyfit(x, y, 1)

    # R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    # Also fit forced through origin: y = C * x
    C_forced = np.sum(x * y) / np.sum(x * x)
    y_pred_forced = C_forced * x
    ss_res_forced = np.sum((y - y_pred_forced) ** 2)
    r_squared_forced = 1.0 - ss_res_forced / ss_tot

    print(f"\n  Scaling law fit (τ* = C / √L):")
    print(f"    Free fit: τ* = {slope:.2f}/√L + {intercept:.3f},  R² = {r_squared:.4f}")
    print(f"    Forced origin: τ* = {C_forced:.2f}/√L,  R² = {r_squared_forced:.4f}")
    print(f"    Predicted C = 64 (d_head), fitted C = {C_forced:.2f}")

    return {
        "data_points": [
            {"L": L, "observed_tau_star": tau, "predicted": round(64.0/math.sqrt(L), 2)}
            for L, tau in data_points
        ],
        "free_fit": {
            "slope": round(slope, 2),
            "intercept": round(intercept, 3),
            "R2": round(r_squared, 4),
        },
        "forced_origin_fit": {
            "C": round(C_forced, 2),
            "R2": round(r_squared_forced, 4),
            "note": f"Predicted C=64 (d_head), fitted C={C_forced:.2f}",
        },
    }


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  PHASE 8D: τ* Scaling Law Verification")
    print(f"  Model: 125M, 50M tokens, from-scratch")
    print(f"  Verify τ*(L) = 64/√L")
    print(f"{'#'*60}")

    # D1: L=256
    d1_results, d1_tau_star = run_d1(WORK_DIR)

    # D2: L=512
    d2_results, d2_tau_star = run_d2(WORK_DIR)

    # Scaling law fit
    fit = compute_scaling_law_fit(d1_tau_star, d2_tau_star)

    # Build 8D JSON section
    d8_data = {
        "purpose": "Verify tau*(L) = 64/sqrt(L) conjecture",
        "model": "125M (head_dim=64)",
        "train_tokens": "50M",
        "D1_L256": {
            "predicted_tau_star": 4.0,
            "observed_tau_star": d1_tau_star,
            "results": {},
        },
        "D2_L512": {
            "predicted_tau_star": round(64.0 / math.sqrt(512), 2),
            "observed_tau_star": d2_tau_star,
            "results": {},
        },
        "scaling_law_fit": fit,
    }

    # Fill D1 results
    for key, res in d1_results.items():
        ppl = res.get("ppl", {})
        d8_data["D1_L256"]["results"][key] = {
            f"ppl_{k}": v for k, v in ppl.items()
        }

    # Fill D2 results
    for key, res in d2_results.items():
        ppl = res.get("ppl", {})
        d8_data["D2_L512"]["results"][key] = {
            f"ppl_{k}": v for k, v in ppl.items()
        }

    # Append to results_phase8.json
    results_file = Path("/root/autodl-tmp/evq_phase8/results_phase8.json")
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if "experiments" not in all_results:
        all_results["experiments"] = {}
    all_results["experiments"]["8D_scaling_law_verification"] = d8_data

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Updated: {results_file}")

    # Save standalone 8D results
    d8_file = WORK_DIR / "results_8d.json"
    with open(d8_file, "w") as f:
        json.dump(d8_data, f, indent=2, default=str)
    print(f"  Saved: {d8_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  8D SCALING LAW RESULTS")
    print(f"{'='*60}")

    print(f"\n  D1: L_train=256 (predicted τ*=4.0)")
    print(f"  {'τ':>6s} {'PPL@256':>10s} {'PPL@2048':>10s} {'PPL@4096':>10s} {'PPL@8192':>10s}")
    for key, res in sorted(d1_results.items()):
        p = res.get("ppl", {})
        print(f"  {res['tau']:6.2f} {p.get('256','?'):>10} {p.get('2048','?'):>10} "
              f"{p.get('4096','?'):>10} {p.get('8192','?'):>10}")

    print(f"\n  D2: L_train=512 (predicted τ*=2.83)")
    print(f"  {'τ':>6s} {'PPL@512':>10s} {'PPL@2048':>10s} {'PPL@4096':>10s} {'PPL@8192':>10s}")
    for key, res in sorted(d2_results.items()):
        p = res.get("ppl", {})
        print(f"  {res['tau']:6.2f} {p.get('512','?'):>10} {p.get('2048','?'):>10} "
              f"{p.get('4096','?'):>10} {p.get('8192','?'):>10}")

    print(f"\n  Observed τ*: D1={d1_tau_star}, D2={d2_tau_star}")
    print(f"  Scaling law fit: C={fit['forced_origin_fit']['C']:.2f} (predicted 64), "
          f"R²={fit['forced_origin_fit']['R2']:.4f}")
    print(f"\n  DONE!")


if __name__ == "__main__":
    main()
