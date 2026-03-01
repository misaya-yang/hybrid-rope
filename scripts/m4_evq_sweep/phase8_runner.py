#!/usr/bin/env python3
"""Phase 8: Extended-Ratio Context Extension + Passkey Recovery.

8A: 512→4K (8x expansion), 7 methods including Hybrid EVQ
8B: Fine-tune ablation (passkey recovery vs continuation tokens)
8C: From-scratch 4K baseline (Geo + EVQ τ=2.0)

Usage:
    python phase8_runner.py          # run all 8A→8B→8C
    python phase8_runner.py --only 8A
    python phase8_runner.py --only 8B
    python phase8_runner.py --only 8C
"""

import sys, os, json, math, time, hashlib, argparse
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_evq_sweep import (
    GPT, RotaryEmbedding, evq_cosh_inv_freq, yarn_inv_freq,
    TIER_CONFIGS, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, train_model, set_seed, load_data, load_val,
    phase_collision_score,
)
from eval_passkey_scratch import eval_passkey_nll_gap

import torch
import numpy as np
from contextlib import nullcontext

# ─── Constants ──────────────────────────────────────────────
BASE = 500000.0
DIM = 64
PRETRAIN_CKPT = Path("/root/autodl-tmp/evq_phase7/context_extension_350m/pretrain_512tok/model.pt")
PRETRAIN_SEQ = 512
WORK = Path("/root/autodl-tmp/evq_phase8")
CFG_350M = TIER_CONFIGS["350m"].copy()


# ─── Frequency helpers ──────────────────────────────────────

def geometric_inv_freq(dim=DIM, base=BASE):
    return evq_cosh_inv_freq(dim, 0.0, base)

def pi_inv_freq(dim=DIM, base=BASE, scale=8.0):
    return geometric_inv_freq(dim, base) / scale

def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=2.0, n_geometric_high=8):
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


# ─── Generic helpers ────────────────────────────────────────

def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def _load_pretrain_state():
    return torch.load(PRETRAIN_CKPT, map_location=DEVICE, weights_only=True)

def _build_and_load(cfg, inv_freq, state_dict):
    """Build GPT, load pretrained weights, replace inv_freq."""
    set_seed(42)
    model = GPT(cfg, geometric_inv_freq()).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.blocks[0].attn.rope.inv_freq.copy_(inv_freq)
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    return model


def _run_passkey(model, tok, filler, lengths, trials, label=""):
    """Run passkey NLL-gap eval and return result dict."""
    model.eval()
    pk = eval_passkey_nll_gap(
        model, tok, filler,
        lengths=lengths, depths=[0.5], num_trials=trials,
    )
    g = pk.get("global", {})
    print(f"    [{label}] retrieval={g.get('retrieval_rate','?'):.4f}  "
          f"gap={g.get('mean_nll_gap','?'):.4f}")
    return pk


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════
#  8A: 512→4K Context Extension (8x expansion)
# ═══════════════════════════════════════════════════════════

def run_8A():
    print(f"\n{'='*70}")
    print(f"  PHASE 8A: 512→4K Context Extension (8x expansion ratio)")
    print(f"{'='*70}")

    EXT_SEQ = 4096
    EXT_TOKENS = 10_000_000
    LR = 3e-5
    BATCH = 2
    EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
    PK_LENGTHS = [1024, 2048, 4096, 8192]
    PK_TRIALS = 100

    work_dir = WORK / "ext_4k"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = CFG_350M.copy()
    cfg["seq_len"] = EXT_SEQ
    cfg["max_position_embeddings"] = EXT_SEQ
    cfg["train_tokens"] = EXT_TOKENS
    cfg["lr"] = LR
    cfg["batch_size"] = BATCH

    tok = _get_tokenizer()
    print(f"  Config: seq={EXT_SEQ}, tokens={EXT_TOKENS/1e6:.0f}M, lr={LR}, batch={BATCH}")

    # Data
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, EXT_TOKENS, EXT_SEQ, "fineweb-edu", cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))
    filler = val_data[:50000]

    # Pretrained state
    state_dict = _load_pretrain_state()

    # Define 7 methods
    s = EXT_SEQ / PRETRAIN_SEQ  # 8.0
    methods = {
        "extend_geo":         ("A1 Geometric (unchanged)",  geometric_inv_freq()),
        "extend_pi":          (f"A2 PI (/{s:.0f}x)",        pi_inv_freq(scale=s)),
        "extend_yarn":        (f"A3 YaRN (s={s:.0f})",
                               yarn_inv_freq(DIM, BASE, original_max_position=PRETRAIN_SEQ,
                                             target_max_position=EXT_SEQ)),
        "extend_evq_1.5":     ("A4 EVQ τ=1.5",              evq_cosh_inv_freq(DIM, 1.5, BASE)),
        "extend_evq_2.0":     ("A5 EVQ τ=2.0",              evq_cosh_inv_freq(DIM, 2.0, BASE)),
        "extend_evq_2.5":     ("A6 EVQ τ=2.5",              evq_cosh_inv_freq(DIM, 2.5, BASE)),
        "extend_hybrid_2.0":  ("A7 Hybrid EVQ τ=2.0",       hybrid_evq_inv_freq(tau=2.0)),
    }

    all_results = {}

    # ── Training + PPL eval ──
    for run_name, (desc, inv_freq) in methods.items():
        run_dir = work_dir / run_name
        result_file = run_dir / "result.json"

        if result_file.exists():
            print(f"\n  [SKIP] {run_name} — already done")
            with open(result_file) as f:
                all_results[run_name] = json.load(f)
            continue

        print(f"\n{'─'*60}")
        print(f"  {desc}  [{run_name}]")
        print(f"{'─'*60}")
        print(f"  inv_freq hash={_inv_hash(inv_freq)}  "
              f"max={inv_freq.max():.8f}  min={inv_freq.min():.8f}")

        model = _build_and_load(cfg, inv_freq, state_dict)
        print(f"  Loaded pretrained weights, replaced inv_freq")

        t0 = time.time()
        model = train_model(model, train_data, cfg, seed=42)
        train_time = time.time() - t0
        print(f"  Train time: {train_time/60:.1f} min")

        ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "model.pt")
        np.save(run_dir / "inv_freq.npy", inv_freq.numpy())
        res = {"method": desc, "ppl": ppl, "train_time_sec": round(train_time, 1)}
        _save_json(result_file, res)
        all_results[run_name] = res

        del model
        torch.cuda.empty_cache()

    # ── Passkey eval ──
    print(f"\n{'─'*60}")
    print(f"  8A Passkey NLL-gap evaluation ({PK_TRIALS} trials)")
    print(f"{'─'*60}")

    pk_dir = work_dir / "passkey_eval"
    pk_dir.mkdir(parents=True, exist_ok=True)
    all_pk = {}

    for run_name, (desc, inv_freq) in methods.items():
        ckpt = work_dir / run_name / "model.pt"
        pk_file = pk_dir / f"passkey_{run_name}.json"

        if pk_file.exists():
            print(f"  [SKIP] {run_name} passkey — already done")
            with open(pk_file) as f:
                all_pk[run_name] = json.load(f)
            continue

        if not ckpt.exists():
            print(f"  [SKIP] {run_name} — no checkpoint")
            continue

        print(f"\n  Passkey: {desc}")
        inv_freq_load = torch.from_numpy(np.load(work_dir / run_name / "inv_freq.npy"))
        model = GPT(cfg, inv_freq_load).to(DEVICE)
        st = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(st, strict=False)

        pk = _run_passkey(model, tok, filler, PK_LENGTHS, PK_TRIALS, label=run_name)
        _save_json(pk_file, pk)
        all_pk[run_name] = {"global": pk.get("global", {}), "summary": pk.get("summary", {})}

        del model
        torch.cuda.empty_cache()

    # ── Consolidate ──
    consolidated = {"methods": {}, "passkey": all_pk}
    for run_name in methods:
        r = all_results.get(run_name, {})
        pk = all_pk.get(run_name, {})
        consolidated["methods"][run_name] = {
            "method": r.get("method", ""),
            "ppl": r.get("ppl", {}),
            "passkey": pk.get("global", {}),
            "passkey_by_length": pk.get("summary", {}),
            "train_time_sec": r.get("train_time_sec", 0),
        }
    _save_json(work_dir / "results_8A.json", consolidated)

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"  8A RESULTS: 512→4K (8x expansion)")
    print(f"{'='*80}")
    print(f"  {'Method':28s} {'PPL@4K':>8} {'PPL@8K':>8} {'PPL@16K':>8} "
          f"{'PK@2K':>6} {'PK@4K':>6} {'PK@8K':>6}")
    print(f"  {'-'*76}")
    for run_name, (desc, _) in methods.items():
        r = all_results.get(run_name, {})
        p = r.get("ppl", {})
        pk = all_pk.get(run_name, {}).get("summary", {})
        pk2 = pk.get("L=2048_d=0.5", {}).get("retrieval_rate", "?")
        pk4 = pk.get("L=4096_d=0.5", {}).get("retrieval_rate", "?")
        pk8 = pk.get("L=8192_d=0.5", {}).get("retrieval_rate", "?")
        print(f"  {desc:28s} {p.get('4096','?'):>8} {p.get('8192','?'):>8} "
              f"{p.get('16384','?'):>8} {pk2!s:>6} {pk4!s:>6} {pk8!s:>6}")

    return consolidated


# ═══════════════════════════════════════════════════════════
#  8B: Fine-tune Ablation (Passkey Recovery)
# ═══════════════════════════════════════════════════════════

def run_8B():
    print(f"\n{'='*70}")
    print(f"  PHASE 8B: Fine-tune Ablation (512→2K, varying continuation tokens)")
    print(f"{'='*70}")

    EXT_SEQ = 2048
    LR = 3e-5
    BATCH = 4
    EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]
    PK_LENGTHS = [1024, 2048, 4096]
    PK_TRIALS = 100

    work_dir = WORK / "finetune_ablation"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = CFG_350M.copy()
    cfg["seq_len"] = EXT_SEQ
    cfg["max_position_embeddings"] = EXT_SEQ
    cfg["lr"] = LR
    cfg["batch_size"] = BATCH

    tok = _get_tokenizer()

    # Data — load max needed (20M tokens at 2K)
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data_20m = load_data(tok, 20_000_000, EXT_SEQ, "fineweb-edu", cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))
    filler = val_data[:50000]

    state_dict = _load_pretrain_state()

    # Runs: (name, method, inv_freq, tokens)
    evq20_inv = evq_cosh_inv_freq(DIM, 2.0, BASE)
    geo_inv = geometric_inv_freq()

    runs = [
        ("evq2.0_2.5M",  "EVQ τ=2.0 (2.5M)",  evq20_inv,  2_500_000),
        # B2 (EVQ 5M) = 7F reference, skip
        ("evq2.0_10M",   "EVQ τ=2.0 (10M)",   evq20_inv,  10_000_000),
        ("evq2.0_20M",   "EVQ τ=2.0 (20M)",   evq20_inv,  20_000_000),
        ("geo_10M",      "Geometric (10M)",    geo_inv,    10_000_000),
        ("geo_20M",      "Geometric (20M)",    geo_inv,    20_000_000),
    ]

    all_results = {}

    for run_name, desc, inv_freq, tokens in runs:
        run_dir = work_dir / run_name
        result_file = run_dir / "result.json"

        if result_file.exists():
            print(f"\n  [SKIP] {run_name} — already done")
            with open(result_file) as f:
                all_results[run_name] = json.load(f)
            continue

        print(f"\n{'─'*60}")
        print(f"  {desc}  [{run_name}]")
        print(f"{'─'*60}")

        # Slice train_data to required token count
        n_chunks = tokens // EXT_SEQ
        train_data = train_data_20m[:n_chunks]
        print(f"  Using {n_chunks} chunks ({tokens/1e6:.1f}M tokens)")

        cfg_run = cfg.copy()
        cfg_run["train_tokens"] = tokens

        model = _build_and_load(cfg_run, inv_freq, state_dict)
        print(f"  Loaded pretrained weights, replaced inv_freq")

        t0 = time.time()
        model = train_model(model, train_data, cfg_run, seed=42)
        train_time = time.time() - t0
        print(f"  Train time: {train_time/60:.1f} min")

        ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

        # Passkey immediately
        print(f"  Passkey eval ({PK_TRIALS} trials)...")
        pk = _run_passkey(model, tok, filler, PK_LENGTHS, PK_TRIALS, label=run_name)

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "model.pt")
        np.save(run_dir / "inv_freq.npy", inv_freq.numpy())
        res = {
            "method": desc,
            "tokens": tokens,
            "ppl": ppl,
            "passkey_global": pk.get("global", {}),
            "passkey_summary": pk.get("summary", {}),
            "train_time_sec": round(train_time, 1),
        }
        _save_json(result_file, res)
        _save_json(run_dir / "passkey.json", pk)
        all_results[run_name] = res

        del model
        torch.cuda.empty_cache()

    # Add 7F references
    all_results["evq2.0_5M_ref"] = {
        "method": "EVQ τ=2.0 (5M) — 7F ref",
        "tokens": 5_000_000,
        "ppl": {"512": 88.292, "1024": 88.856, "2048": 89.233, "4096": 95.922, "8192": 99.129},
        "passkey_global": {"retrieval_rate": 0.6467, "mean_nll_gap": 0.1086},
        "passkey_summary": {
            "L=1024_d=0.5": {"retrieval_rate": 0.82},
            "L=2048_d=0.5": {"retrieval_rate": 0.72},
            "L=4096_d=0.5": {"retrieval_rate": 0.40},
        },
    }
    all_results["geo_5M_ref"] = {
        "method": "Geometric (5M) — 7F ref",
        "tokens": 5_000_000,
        "ppl": {"512": 86.731, "1024": 86.472, "2048": 87.589, "4096": 93.170, "8192": 97.959},
        "passkey_global": {"retrieval_rate": 0.7333, "mean_nll_gap": 0.2103},
        "passkey_summary": {
            "L=1024_d=0.5": {"retrieval_rate": 0.90},
            "L=2048_d=0.5": {"retrieval_rate": 0.78},
            "L=4096_d=0.5": {"retrieval_rate": 0.52},
        },
    }

    _save_json(work_dir / "results_8B.json", all_results)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  8B RESULTS: Fine-tune Ablation (512→2K)")
    print(f"{'='*80}")
    print(f"  {'Run':22s} {'Tokens':>7} {'PPL@2K':>8} {'PPL@8K':>8} "
          f"{'PK@1K':>6} {'PK@2K':>6} {'PK@4K':>6}")
    print(f"  {'-'*65}")
    for name in ["evq2.0_2.5M", "evq2.0_5M_ref", "evq2.0_10M", "evq2.0_20M",
                  "geo_5M_ref", "geo_10M", "geo_20M"]:
        r = all_results.get(name, {})
        p = r.get("ppl", {})
        pk = r.get("passkey_summary", {})
        pk1 = pk.get("L=1024_d=0.5", {}).get("retrieval_rate", "?")
        pk2 = pk.get("L=2048_d=0.5", {}).get("retrieval_rate", "?")
        pk4 = pk.get("L=4096_d=0.5", {}).get("retrieval_rate", "?")
        tok_str = f"{r.get('tokens',0)/1e6:.1f}M"
        print(f"  {name:22s} {tok_str:>7} {p.get('2048','?'):>8} {p.get('8192','?'):>8} "
              f"{pk1!s:>6} {pk2!s:>6} {pk4!s:>6}")

    return all_results


# ═══════════════════════════════════════════════════════════
#  8C: From-Scratch 4K Baseline
# ═══════════════════════════════════════════════════════════

def run_8C():
    print(f"\n{'='*70}")
    print(f"  PHASE 8C: From-Scratch 4K Baseline (350M, 50M tokens)")
    print(f"{'='*70}")

    SEQ = 4096
    TOKENS = 50_000_000
    LR = 6e-4
    BATCH = 2
    EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
    PK_LENGTHS = [1024, 2048, 4096, 8192]
    PK_TRIALS = 100

    work_dir = WORK / "from_scratch_4k"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = CFG_350M.copy()
    cfg["seq_len"] = SEQ
    cfg["max_position_embeddings"] = SEQ
    cfg["train_tokens"] = TOKENS
    cfg["lr"] = LR
    cfg["batch_size"] = BATCH

    tok = _get_tokenizer()

    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, TOKENS, SEQ, "fineweb-edu", cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))
    filler = val_data[:50000]

    runs = [
        ("geo_4k",     "C1 Geometric (from scratch)", geometric_inv_freq()),
        ("evq2.0_4k",  "C2 EVQ τ=2.0 (from scratch)", evq_cosh_inv_freq(DIM, 2.0, BASE)),
    ]

    all_results = {}

    for run_name, desc, inv_freq in runs:
        run_dir = work_dir / run_name
        result_file = run_dir / "result.json"

        if result_file.exists():
            print(f"\n  [SKIP] {run_name} — already done")
            with open(result_file) as f:
                all_results[run_name] = json.load(f)
            continue

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
        _save_json(run_dir / "passkey.json", pk)
        all_results[run_name] = res

        del model
        torch.cuda.empty_cache()

    _save_json(work_dir / "results_8C.json", all_results)

    print(f"\n{'='*80}")
    print(f"  8C RESULTS: From-Scratch 4K")
    print(f"{'='*80}")
    for name, r in all_results.items():
        p = r.get("ppl", {})
        pk = r.get("passkey_global", {})
        print(f"  {name:15s}  PPL@4K={p.get('4096','?')}  PPL@8K={p.get('8192','?')}  "
              f"PPL@16K={p.get('16384','?')}  passkey={pk.get('retrieval_rate','?')}")

    return all_results


# ═══════════════════════════════════════════════════════════
#  Consolidate all Phase 8 results
# ═══════════════════════════════════════════════════════════

def consolidate():
    """Build results_phase8.json from sub-experiment JSONs."""
    result = {"phase": 8, "date": "2026-03-01", "hardware": "RTX 5090 32GB", "experiments": {}}

    f8a = WORK / "ext_4k" / "results_8A.json"
    if f8a.exists():
        with open(f8a) as f:
            result["experiments"]["8A_ext_4k"] = json.load(f)

    f8b = WORK / "finetune_ablation" / "results_8B.json"
    if f8b.exists():
        with open(f8b) as f:
            result["experiments"]["8B_finetune_ablation"] = json.load(f)

    f8c = WORK / "from_scratch_4k" / "results_8C.json"
    if f8c.exists():
        with open(f8c) as f:
            result["experiments"]["8C_from_scratch_4k"] = json.load(f)

    _save_json(WORK / "results_phase8.json", result)
    print(f"\n  Consolidated: {WORK / 'results_phase8.json'}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment: 8A, 8B, or 8C")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={DEVICE}  dtype={DTYPE}  autocast={USE_AUTOCAST}")
    print(f"[init] pretrain ckpt: {PRETRAIN_CKPT}  exists={PRETRAIN_CKPT.exists()}")

    t_total = time.time()

    if args.only is None or args.only.upper() == "8A":
        run_8A()
        consolidate()

    if args.only is None or args.only.upper() == "8B":
        run_8B()
        consolidate()

    if args.only is None or args.only.upper() == "8C":
        run_8C()
        consolidate()

    total_min = (time.time() - t_total) / 60
    print(f"\n{'#'*70}")
    print(f"  PHASE 8 COMPLETE — total time: {total_min:.1f} min")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
