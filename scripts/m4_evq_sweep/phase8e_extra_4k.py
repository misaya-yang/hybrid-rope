#!/usr/bin/env python3
"""Phase 8E: Extra from-scratch 4K runs (EVQ τ=1.0 + Hybrid τ=1.0).

Same config as 8C (350M, from-scratch, 4K seq_len, 50M tokens, lr=6e-4).
Reuses 8C's cached data.
"""

import sys, os, json, math, time, hashlib
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_evq_sweep import (
    GPT, RotaryEmbedding, evq_cosh_inv_freq,
    TIER_CONFIGS, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, train_model, set_seed, load_data, load_val,
)
from eval_passkey_scratch import eval_passkey_nll_gap

import torch
import numpy as np

BASE = 500000.0
DIM = 64
WORK = Path("/root/autodl-tmp/evq_phase8/from_scratch_4k")
CFG_350M = TIER_CONFIGS["350m"].copy()


def geometric_inv_freq(dim=DIM, base=BASE):
    return evq_cosh_inv_freq(dim, 0.0, base)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=1.0, n_geometric_high=8):
    """Hybrid EVQ: high-freq channels keep Geometric, low-freq use EVQ-cosh."""
    n = dim // 2  # 32
    geo = geometric_inv_freq(dim, base).double()
    n_evq = n - n_geometric_high  # 24
    theta_max_low = geo[n_geometric_high].item()
    theta_min_low = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / (n_evq - 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min_low ** phi) * (theta_max_low ** (1.0 - phi))
    inv_freq = torch.cat([geo[:n_geometric_high], evq_part])
    return inv_freq.float()


def _inv_hash(inv_freq):
    return hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:16]


def _run_passkey(model, tok, filler, lengths, trials, label=""):
    pk = eval_passkey_nll_gap(model, tok, filler, lengths=lengths, depths=[0.5], num_trials=trials)
    g = pk.get("global", {})
    print(f"    [{label}] retrieval={g.get('retrieval_rate','?')}  gap={g.get('mean_nll_gap','?')}")
    return pk


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_single(run_name, desc, inv_freq, work_dir, cfg, train_data, val_data, filler, tok):
    run_dir = work_dir / run_name
    result_file = run_dir / "result.json"

    EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
    PK_LENGTHS = [1024, 2048, 4096, 8192]
    PK_TRIALS = 100

    if result_file.exists():
        print(f"\n  [SKIP] {run_name} — already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'─'*60}")
    print(f"  {desc}  [{run_name}]")
    print(f"{'─'*60}")
    print(f"  inv_freq hash={_inv_hash(inv_freq)}  "
          f"max={inv_freq.max():.8f}  min={inv_freq.min():.8f}")

    set_seed(42)
    model = GPT(cfg, inv_freq).to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    t0 = time.time()
    model = train_model(model, train_data, cfg, seed=42)
    train_time = time.time() - t0
    print(f"  Train time: {train_time/60:.1f} min")

    ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

    # Passkey
    print(f"  Passkey eval ({PK_TRIALS} trials)...")
    pk = _run_passkey(model, tok, filler, PK_LENGTHS, PK_TRIALS, label=run_name)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", inv_freq.numpy())
    res = {
        "method": desc,
        "ppl": ppl,
        "passkey_global": pk.get("global", {}),
        "passkey_summary": pk.get("summary", {}),
        "train_time_sec": round(train_time, 1),
    }
    _save_json(result_file, res)

    del model
    torch.cuda.empty_cache()
    return res


def main():
    print(f"\n{'#'*60}")
    print(f"  PHASE 8E: Extra From-Scratch 4K Runs")
    print(f"  EVQ τ=1.0 + Hybrid τ=1.0 (if time)")
    print(f"{'#'*60}")

    SEQ = 4096
    TOKENS = 50_000_000
    LR = 6e-4
    BATCH = 2

    cfg = CFG_350M.copy()
    cfg["seq_len"] = SEQ
    cfg["max_position_embeddings"] = SEQ
    cfg["train_tokens"] = TOKENS
    cfg["lr"] = LR
    cfg["batch_size"] = BATCH

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    data_dir = WORK / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, TOKENS, SEQ, "fineweb-edu", cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))
    filler = val_data[:50000]

    all_results = {}

    # E1: EVQ τ=1.0
    evq10_inv = evq_cosh_inv_freq(DIM, 1.0, BASE)
    res1 = run_single("evq1.0_4k", "E1 EVQ τ=1.0 (from scratch)",
                       evq10_inv, WORK, cfg, train_data, val_data, filler, tok)
    all_results["evq1.0_4k"] = res1

    # E2: Hybrid τ=1.0
    hybrid10_inv = hybrid_evq_inv_freq(DIM, BASE, tau=1.0, n_geometric_high=8)
    res2 = run_single("hybrid1.0_4k", "E2 Hybrid EVQ τ=1.0 (from scratch)",
                       hybrid10_inv, WORK, cfg, train_data, val_data, filler, tok)
    all_results["hybrid1.0_4k"] = res2

    # Append to results_phase8.json
    results_file = Path("/root/autodl-tmp/evq_phase8/results_phase8.json")
    if results_file.exists():
        with open(results_file) as f:
            master = json.load(f)
    else:
        master = {"experiments": {}}

    if "8C_from_scratch_4k" not in master.get("experiments", {}):
        master.setdefault("experiments", {})["8C_from_scratch_4k"] = {}

    # Add 8E results under 8C section (same experiment type)
    for name, res in all_results.items():
        master["experiments"]["8C_from_scratch_4k"][name] = res

    # Also add as separate 8E section
    master["experiments"]["8E_extra_from_scratch_4k"] = all_results

    _save_json(results_file, master)
    print(f"\n  Updated: {results_file}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  8E RESULTS: Extra From-Scratch 4K")
    print(f"{'='*60}")
    for name, res in all_results.items():
        p = res.get("ppl", {})
        pk = res.get("passkey_global", {})
        print(f"  {name:20s} PPL@4K={p.get('4096','?')}  PPL@8K={p.get('8192','?')}  "
              f"PPL@16K={p.get('16384','?')}  passkey={pk.get('retrieval_rate','?')}")
    print(f"\n  DONE!")


if __name__ == "__main__":
    main()
