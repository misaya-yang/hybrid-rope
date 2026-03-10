#!/usr/bin/env python3
"""
Phase 17C Extended Eval: geo + evq at longer lengths (up to 48K).

Improvements over the in-training eval:
  - RoPE extended per-length (not pre-allocated to max)
  - Aggressive memory cleanup between lengths
  - Includes both raw and +YaRN modes
  - Passkey eval at longer lengths (4K-16K)

Usage:
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/hybrid-rope
  PYTHONUNBUFFERED=1 nohup python scripts/core_text_phases/phase17c_extended_eval.py \
    > /root/autodl-tmp/evq_phase17c_2048_continue/logs/phase17c_extended_eval.log 2>&1 &
"""

import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "supporting_eval"))

from eval_passkey_scratch import eval_passkey_nll_gap  # noqa: E402
from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    evq_cosh_inv_freq,
    load_val,
    set_seed,
)

# ── Config ────────────────────────────────────────────────────────────────

BASE = 500_000.0
DIM = 64
SEQ_LEN = 2048
TAU = DIM / math.sqrt(SEQ_LEN)  # 1.414214

CFG_454M = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
)

WORK = Path("/root/autodl-tmp/evq_phase17c_2048_continue")
SEED = 42

EVAL_LENGTHS = [2048, 4096, 8192, 16384, 24576, 32768, 40960, 49152]
EVAL_CHUNKS = 10
MODES = ["raw", "yarn"]

PK_LENGTHS = [2048, 4096, 8192, 16384]
PK_TRIALS = 10
PK_DEPTHS = [0.5]

MODELS = {
    "geo_2048_cont": {
        "ckpt": str(WORK / "seed42" / "geo_454m_1024_to_2048_continue" / "model.pt"),
        "L_train": 2048,
        "tau": 0.0,
        "label": "Geo L_train=512→1024→2048",
    },
    "evq_2048_cont": {
        "ckpt": str(WORK / "seed42" / f"evq{TAU:g}_454m_1024_to_2048_continue" / "model.pt"),
        "L_train": 2048,
        "tau": TAU,
        "label": f"EVQ τ={TAU:.4f} L_train=512→1024→2048",
    },
}

RESULT_FILE = WORK / "phase17c_extended_eval.json"


# ── Helpers ───────────────────────────────────────────────────────────────

def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def build_yarn_inv_freq(base_inv, head_dim, scale):
    """YaRN progressive frequency scaling with smoothstep ramp."""
    if scale <= 1.0:
        return base_inv.clone()
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + 0.07 * math.log2(scale)
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()


def get_base_inv_freq(tau):
    if tau == 0.0:
        return geometric_inv_freq()
    return evq_cosh_inv_freq(head_dim=DIM, tau=tau, base=BASE)


def _load_state(path: str):
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint format at {path}")
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


def load_model(cfg, ckpt_path, inv_freq):
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = _load_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    return model


def swap_inv_freq(model, inv_freq, max_pos):
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    model.blocks[0].attn.rope._build(max_pos)


def eval_ppl_single_length(model, val_data, L, n_chunks=EVAL_CHUNKS):
    """Evaluate PPL at a single length, with per-chunk OOM handling."""
    model.eval()
    model.extend_rope(L + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)

    max_start = len(val_data) - L
    if max_start <= 0:
        return None
    offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
    losses = []
    for offset in offsets:
        chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
        try:
            with torch.no_grad(), ctx:
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1)
                )
            losses.append(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    L={L}: OOM on chunk offset={offset}, stopping")
                break
            raise
        finally:
            del chunk
            # Aggressive cleanup after each chunk at long lengths
            if L >= 16384 and DEVICE == "cuda":
                torch.cuda.empty_cache()

    if losses:
        return round(math.exp(sum(losses) / len(losses)), 3)
    return None


def eval_passkey_single_length(model, tokenizer, filler, L, depth=0.5, trials=10):
    """Evaluate passkey at a single length."""
    try:
        model.extend_rope(L + 100)
        result = eval_passkey_nll_gap(
            model, tokenizer, filler,
            lengths=[L],
            depths=[depth],
            num_trials=trials,
        )
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    passkey L={L}: OOM")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return None
        raise


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("#" * 72)
    print("  Phase 17C Extended Eval")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Lengths: {EVAL_LENGTHS}")
    print(f"  Modes: {MODES}")
    print(f"  Passkey lengths: {PK_LENGTHS}")
    print(f"  Device: {DEVICE}  dtype: {DTYPE}")
    print("#" * 72)

    # Check SDPA backend
    if DEVICE == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Check flash attention availability
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("  Flash SDP: enabled")
        except Exception:
            print("  Flash SDP: not available")
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("  Mem-efficient SDP: enabled")
        except Exception:
            print("  Mem-efficient SDP: not available")

    # Load or init results (for resume)
    if RESULT_FILE.exists():
        with open(RESULT_FILE) as f:
            results = json.load(f)
        print(f"\n  Resuming from {RESULT_FILE}")
    else:
        results = {}

    # Load tokenizer & val data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print("\n  Loading validation data...")
    val_cache = Path("/root/autodl-tmp/evq_phase17/val_proof-pile-2_5000000.pt")
    if val_cache.exists():
        val_data = torch.load(val_cache, map_location="cpu", weights_only=True)
    else:
        val_data = load_val(tok, 5_000_000, "proof-pile-2", cache_dir=str(WORK / "data"))
    filler = val_data[:50000]
    print(f"  val_data: {val_data.shape}, filler: {filler.shape}")

    t0_global = time.time()

    for model_key, model_info in MODELS.items():
        ckpt_path = model_info["ckpt"]
        L_train = model_info["L_train"]
        tau = model_info["tau"]
        label = model_info["label"]

        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {model_key}: checkpoint not found at {ckpt_path}")
            continue

        base_inv = get_base_inv_freq(tau)
        cfg = {**CFG_454M, "max_position_embeddings": L_train, "seq_len": L_train}

        for mode in MODES:
            mode_key = f"{model_key}__{mode}"

            if mode_key in results and results[mode_key].get("status") == "done":
                print(f"\n  [SKIP] {mode_key}: already complete")
                continue

            print(f"\n{'=' * 72}")
            print(f"  {mode_key}  ({label})")
            print(f"  ckpt: {ckpt_path}")
            print(f"  L_train={L_train}, tau={tau}, mode={mode}")
            print(f"{'=' * 72}")

            # Load model fresh for each mode to avoid memory fragmentation
            model = load_model(cfg, ckpt_path, base_inv)
            model.eval()
            print(f"  Model loaded (no torch.compile)")

            ppl_results = {}
            pk_results = {}
            t0_mode = time.time()

            # ── PPL eval ──
            for L in EVAL_LENGTHS:
                scale = L / L_train

                if mode == "yarn" and scale > 1.0:
                    inv_freq = build_yarn_inv_freq(base_inv, DIM, scale)
                else:
                    inv_freq = base_inv.clone()
                swap_inv_freq(model, inv_freq, L + 100)

                ppl = eval_ppl_single_length(model, val_data, L)
                if ppl is not None:
                    ppl_results[str(L)] = ppl
                    yarn_note = f" +YaRN({scale:.1f}×)" if mode == "yarn" and scale > 1.0 else ""
                    print(f"    L={L:>5d} ({scale:>4.1f}×): PPL={ppl:>8.3f}{yarn_note}")
                else:
                    print(f"    L={L:>5d}: OOM/skip")
                    # If OOM, remaining lengths will likely also OOM
                    break

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            # ── Passkey eval ──
            for L in PK_LENGTHS:
                scale = L / L_train

                if mode == "yarn" and scale > 1.0:
                    inv_freq = build_yarn_inv_freq(base_inv, DIM, scale)
                else:
                    inv_freq = base_inv.clone()
                swap_inv_freq(model, inv_freq, L + 100)

                pk = eval_passkey_single_length(model, tok, filler, L,
                                                 depth=0.5, trials=PK_TRIALS)
                if pk is not None:
                    g = pk.get("global", {})
                    s = pk.get("summary", {})
                    pk_results[str(L)] = {
                        "retrieval_rate": g.get("retrieval_rate"),
                        "mean_nll_gap": g.get("mean_nll_gap"),
                        "ar_exact_match": g.get("ar_exact_match_rate"),
                        "summary": s,
                    }
                    yarn_note = f" +YaRN({scale:.1f}×)" if mode == "yarn" and scale > 1.0 else ""
                    print(f"    PK L={L:>5d}: ret={g.get('retrieval_rate', 0):.0%}  "
                          f"gap={g.get('mean_nll_gap', 0):+.3f}  "
                          f"AR={g.get('ar_exact_match_rate', 0):.0%}{yarn_note}")
                else:
                    print(f"    PK L={L:>5d}: OOM")
                    break

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            elapsed = time.time() - t0_mode
            results[mode_key] = {
                "status": "done",
                "model_key": model_key,
                "mode": mode,
                "label": label,
                "L_train": L_train,
                "tau": tau,
                "ckpt": ckpt_path,
                "ppl": ppl_results,
                "passkey": pk_results,
                "elapsed_sec": round(elapsed, 1),
            }
            save_json(RESULT_FILE, results)
            print(f"\n    Saved ({elapsed:.0f}s)")

            del model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    total_elapsed = time.time() - t0_global

    # ── Summary ──
    results["_meta"] = {
        "eval_lengths": EVAL_LENGTHS,
        "eval_chunks": EVAL_CHUNKS,
        "pk_lengths": PK_LENGTHS,
        "pk_trials": PK_TRIALS,
        "pk_depths": PK_DEPTHS,
        "total_elapsed_sec": round(total_elapsed, 1),
    }
    save_json(RESULT_FILE, results)

    # ── Print tables ──
    print(f"\n{'=' * 100}")
    print("  PPL SUMMARY")
    print(f"{'=' * 100}")
    header = f"  {'Model':>20s} {'Mode':>6s}"
    for L in EVAL_LENGTHS:
        header += f"  {L//1024:>5d}K"
    print(header)
    print("  " + "─" * 90)

    for mk in MODELS:
        for mode in MODES:
            key = f"{mk}__{mode}"
            r = results.get(key, {})
            ppl = r.get("ppl", {})
            line = f"  {mk:>20s} {mode:>6s}"
            for L in EVAL_LENGTHS:
                v = ppl.get(str(L))
                if v is not None:
                    line += f"  {v:>6.2f}" if v < 100 else f"  {v:>6.1f}"
                else:
                    line += f"  {'--':>6s}"
            print(line)
        print()

    print(f"\n{'=' * 100}")
    print("  PASSKEY RETRIEVAL RATE")
    print(f"{'=' * 100}")
    header = f"  {'Model':>20s} {'Mode':>6s}"
    for L in PK_LENGTHS:
        header += f"  {L//1024:>5d}K"
    print(header)
    print("  " + "─" * 60)

    for mk in MODELS:
        for mode in MODES:
            key = f"{mk}__{mode}"
            r = results.get(key, {})
            pk = r.get("passkey", {})
            line = f"  {mk:>20s} {mode:>6s}"
            for L in PK_LENGTHS:
                v = pk.get(str(L), {})
                rr = v.get("retrieval_rate") if isinstance(v, dict) else None
                if rr is not None:
                    line += f"  {rr:>5.0%}"
                else:
                    line += f"  {'--':>6s}"
            print(line)
        print()

    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")
    print("  DONE!")


if __name__ == "__main__":
    main()
