#!/usr/bin/env python3
"""Phase 7F: 350M Context Extension (512→2K).

Stage 1: Pretrain 350M at 512-tok with Geometric RoPE (50M tokens)
Stage 2: Continue training at 2K with Geo/PI/YaRN/EVQ (5M tokens each)
Stage 3: Passkey NLL-gap evaluation
"""

import sys, os, json, math, time, hashlib
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

BASE = 500000.0
PRETRAIN_SEQ = 512
PRETRAIN_TOKENS = 50_000_000
CONTINUE_SEQ = 2048
CONTINUE_TOKENS = 5_000_000
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192]

# 350M config
CFG_350M = TIER_CONFIGS["350m"].copy()


def stage1_pretrain(work_dir: Path):
    """Stage 1: Pretrain 350M at 512 tokens with Geometric RoPE."""
    pt_dir = work_dir / "pretrain_512tok"
    ckpt_path = pt_dir / "model.pt"

    if ckpt_path.exists():
        print(f"\n[Stage 1] Pretrained checkpoint exists: {ckpt_path}")
        return ckpt_path

    print(f"\n{'#'*60}")
    print(f"  STAGE 1: Pretrain 350M @ 512-tok, {PRETRAIN_TOKENS/1e6:.0f}M tokens")
    print(f"{'#'*60}")

    cfg = CFG_350M.copy()
    cfg["seq_len"] = PRETRAIN_SEQ
    cfg["max_position_embeddings"] = PRETRAIN_SEQ
    cfg["train_tokens"] = PRETRAIN_TOKENS

    # Batch auto-scale: 350m default batch=2, scale = 2048/512 = 4 → batch=8
    # For <40GB VRAM: batch stays 2, then scale → 8
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    vram_gb = total_mem / (1024 ** 3)
    if vram_gb < 40:
        cfg["batch_size"] = 2
    scale = 2048 // cfg["seq_len"]
    cfg["batch_size"] = cfg["batch_size"] * scale
    print(f"  batch_size={cfg['batch_size']} (auto-scaled {scale}x for seq_len={cfg['seq_len']})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    train_data = load_data(tok, PRETRAIN_TOKENS, PRETRAIN_SEQ, "fineweb-edu",
                           cache_dir=str(pt_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(pt_dir))

    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], 0.0, BASE)
    set_seed(42)
    model = GPT(cfg, inv_freq).to(DEVICE)

    t0 = time.time()
    model = train_model(model, train_data, cfg, seed=42)
    train_time = time.time() - t0
    print(f"  Pretrain time: {train_time/60:.1f} min")

    # Eval
    ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

    # Save
    pt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    np.save(pt_dir / "inv_freq.npy", inv_freq.numpy())
    with open(pt_dir / "pretrain_result.json", "w") as f:
        json.dump({"ppl": ppl, "train_time_sec": round(train_time, 1)}, f, indent=2)

    del model
    torch.cuda.empty_cache()
    print(f"  Pretrain checkpoint saved: {ckpt_path}")
    return ckpt_path


def stage2_continue(work_dir: Path, pretrain_ckpt: Path):
    """Stage 2: Continue training at 2K with different frequency methods."""

    print(f"\n{'#'*60}")
    print(f"  STAGE 2: Continue Training 512→2K, {CONTINUE_TOKENS/1e6:.0f}M tokens")
    print(f"{'#'*60}")

    cfg = CFG_350M.copy()
    cfg["seq_len"] = CONTINUE_SEQ
    cfg["max_position_embeddings"] = CONTINUE_SEQ
    cfg["train_tokens"] = CONTINUE_TOKENS

    # Batch for 2K training: 350m default batch=2
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    vram_gb = total_mem / (1024 ** 3)
    if vram_gb < 40:
        cfg["batch_size"] = 2
    # No auto-scale since seq_len=2048 matches default
    print(f"  Continue: batch_size={cfg['batch_size']}, seq_len={cfg['seq_len']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load/build 2K data
    ext_dir = work_dir / "continue_data"
    ext_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, CONTINUE_TOKENS, CONTINUE_SEQ, "fineweb-edu",
                           cache_dir=str(ext_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(ext_dir))

    # Define methods
    s = CONTINUE_SEQ / PRETRAIN_SEQ  # 4.0
    geo_inv = evq_cosh_inv_freq(64, 0.0, BASE)
    pi_inv = geo_inv / s
    yarn_inv = yarn_inv_freq(64, BASE, original_max_position=PRETRAIN_SEQ,
                             target_max_position=CONTINUE_SEQ)
    evq15_inv = evq_cosh_inv_freq(64, 1.5, BASE)
    evq20_inv = evq_cosh_inv_freq(64, 2.0, BASE)

    methods = {
        "extend_geo":     ("Geometric (unchanged)", geo_inv),
        "extend_pi":      (f"PI (/{s:.0f}x)", pi_inv),
        "extend_yarn":    (f"YaRN (s={s:.0f})", yarn_inv),
        "extend_evq_1.5": ("EVQ tau=1.5", evq15_inv),
        "extend_evq_2.0": ("EVQ tau=2.0", evq20_inv),
    }

    results = {}
    for run_name, (desc, target_inv) in methods.items():
        run_dir = work_dir / run_name
        result_file = run_dir / "result.json"
        if result_file.exists():
            print(f"\n  [SKIP] {run_name} already done")
            with open(result_file) as f:
                results[run_name] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"  CONTINUE: {run_name} — {desc}")
        print(f"{'='*60}")

        inv_hash = hashlib.sha256(target_inv.numpy().tobytes()).hexdigest()[:16]
        print(f"  inv_freq hash={inv_hash}  max={target_inv.max():.8f}  min={target_inv.min():.8f}")

        # Build model, load pretrained weights, replace inv_freq
        set_seed(42)
        model = GPT(cfg, geo_inv).to(DEVICE)  # init with geo (will be overwritten)
        ckpt = torch.load(pretrain_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt, strict=False)

        # Replace inv_freq with target method
        model.blocks[0].attn.rope.inv_freq.copy_(target_inv)
        model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
        print(f"  Loaded pretrained weights, replaced inv_freq")

        # Continue training
        t0 = time.time()
        model = train_model(model, train_data, cfg, seed=42)
        train_time = time.time() - t0
        print(f"  Continue time: {train_time/60:.1f} min")

        # Eval
        ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

        # Save
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "model.pt")
        np.save(run_dir / "inv_freq.npy", target_inv.numpy())
        res = {"method": desc, "ppl": ppl, "train_time_sec": round(train_time, 1)}
        with open(result_file, "w") as f:
            json.dump(res, f, indent=2)
        results[run_name] = res

        del model
        torch.cuda.empty_cache()

    return results


def stage3_passkey(work_dir: Path):
    """Stage 3: Passkey NLL-gap evaluation on continued models."""
    print(f"\n{'#'*60}")
    print(f"  STAGE 3: Passkey NLL-gap Evaluation (350M)")
    print(f"{'#'*60}")

    cfg = CFG_350M.copy()
    cfg["seq_len"] = CONTINUE_SEQ
    cfg["max_position_embeddings"] = CONTINUE_SEQ

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    val_path = work_dir / "continue_data" / "val_fineweb-edu_5000000.pt"
    val_data = torch.load(val_path, weights_only=True)
    filler = val_data[:50000]

    methods = ["extend_geo", "extend_pi", "extend_yarn", "extend_evq_1.5", "extend_evq_2.0"]
    pk_dir = work_dir / "passkey_eval"
    pk_dir.mkdir(parents=True, exist_ok=True)

    all_pk = {}
    for run_name in methods:
        ckpt = work_dir / run_name / "model.pt"
        if not ckpt.exists():
            print(f"  [SKIP] {run_name} — no checkpoint")
            continue

        pk_file = pk_dir / f"passkey_{run_name}.json"
        if pk_file.exists():
            print(f"  [SKIP] {run_name} — passkey already done")
            with open(pk_file) as f:
                all_pk[run_name] = json.load(f)
            continue

        print(f"\n  Passkey eval: {run_name}")
        inv_freq = torch.from_numpy(np.load(work_dir / run_name / "inv_freq.npy"))
        model = GPT(cfg, inv_freq).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        pk_result = eval_passkey_nll_gap(
            model, tok, filler,
            lengths=[1024, 2048, 4096],
            depths=[0.5],
            num_trials=50,
        )
        g = pk_result.get("global", {})
        print(f"    retrieval={g.get('retrieval_rate','?')}  gap={g.get('mean_nll_gap','?')}")

        with open(pk_file, "w") as f:
            json.dump(pk_result, f, indent=2, default=str)
        all_pk[run_name] = {
            "global": pk_result.get("global", {}),
            "summary": pk_result.get("summary", {}),
        }

        del model
        torch.cuda.empty_cache()

    return all_pk


def main():
    work_dir = Path("/root/autodl-tmp/evq_phase7/context_extension_350m")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Pretrain
    pretrain_ckpt = stage1_pretrain(work_dir)

    # Stage 2: Continue training
    ext_results = stage2_continue(work_dir, pretrain_ckpt)

    # Stage 3: Passkey eval
    pk_results = stage3_passkey(work_dir)

    # Save consolidated results
    final = {
        "pretrain": {},
        "continue": ext_results,
        "passkey": pk_results,
    }
    pt_result_file = work_dir / "pretrain_512tok" / "pretrain_result.json"
    if pt_result_file.exists():
        with open(pt_result_file) as f:
            final["pretrain"] = json.load(f)

    with open(work_dir / "results_checkpoint.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  7F CONTEXT EXTENSION RESULTS (350M, 512→2K)")
    print(f"{'='*60}")
    print(f"  {'Method':25s} {'PPL@512':>8} {'PPL@2K':>8} {'PPL@4K':>8} {'PPL@8K':>8}")
    print(f"  {'-'*57}")
    for name, res in ext_results.items():
        p = res.get("ppl", {})
        print(f"  {name:25s} {p.get('512','?'):>8} {p.get('2048','?'):>8} "
              f"{p.get('4096','?'):>8} {p.get('8192','?'):>8}")


if __name__ == "__main__":
    main()
