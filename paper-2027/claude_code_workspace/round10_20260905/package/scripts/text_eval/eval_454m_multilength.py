#!/usr/bin/env python3
"""
Evaluate 454M text model checkpoints (GEO and EVQ) at multiple context lengths.

Runs two evaluation suites:
  1. Sliding-window PPL on PG-19 / proof-pile-2 test data
     Lengths: 2048, 4096, 8192, 16384, 32768
  2. Passkey retrieval (NLL-gap + autoregressive exact-match)
     Lengths: 4096, 8192, 16384, 32768, 65536

Both raw and +YaRN modes are evaluated for each checkpoint.

Usage (on server):
  export PATH=/root/miniconda3/bin:$PATH
  cd /root/autodl-tmp/hybrid-rope

  # Auto-detect checkpoints in default location:
  python scripts/text_eval/eval_454m_multilength.py

  # Explicit paths:
  python scripts/text_eval/eval_454m_multilength.py \
    --model_dir /root/autodl-tmp/evq_phase17c_2048_continue \
    --geo_ckpt /root/autodl-tmp/evq_phase17c_2048_continue/seed42/geo_454m_1024_to_2048_continue/model.pt \
    --evq_ckpt /root/autodl-tmp/evq_phase17c_2048_continue/seed42/evq1.41421_454m_1024_to_2048_continue/model.pt \
    --output_dir results/454m_multilength

  # With local validation data:
  python scripts/text_eval/eval_454m_multilength.py \
    --data_path /root/autodl-tmp/evq_phase17/val_proof-pile-2_5000000.pt

Paper Role:  Table/Figure — multi-length PPL + passkey comparison (454M, GEO vs EVQ)
Input:       Phase 17C trained checkpoints
Output:      results/454m_multilength/eval_results.json + console comparison table
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# Resolve imports from the codebase
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_TEXT_DIR = SCRIPT_DIR.parent / "core_text_phases"
SUPPORTING_EVAL_DIR = SCRIPT_DIR.parent / "supporting_eval"
sys.path.insert(0, str(CORE_TEXT_DIR))
sys.path.insert(0, str(SUPPORTING_EVAL_DIR))

from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    evq_cosh_inv_freq,
    load_val,
    set_seed,
)
from eval_passkey_scratch import eval_passkey_nll_gap  # noqa: E402

# ---------------------------------------------------------------------------
# Model architecture config (454M, matching phase17c)
# ---------------------------------------------------------------------------

BASE = 500_000.0
DIM = 64          # head_dim
L_TRAIN = 2048    # training context length for phase17c
TAU = DIM / math.sqrt(L_TRAIN)  # 1.414214...

CFG_454M = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=L_TRAIN,
    seq_len=L_TRAIN,
)

# ---------------------------------------------------------------------------
# Evaluation defaults
# ---------------------------------------------------------------------------

PPL_LENGTHS = [2048, 4096, 8192, 16384, 32768]
PPL_CHUNKS = 10

PK_LENGTHS = [4096, 8192, 16384, 32768, 65536]
PK_TRIALS = 10
PK_DEPTHS = [0.5]

MODES = ["raw", "yarn"]

# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------


def geometric_inv_freq(dim: int = DIM, base: float = BASE) -> torch.Tensor:
    """Standard geometric RoPE inverse frequencies."""
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def build_yarn_inv_freq(
    base_inv: torch.Tensor, head_dim: int, scale: float
) -> torch.Tensor:
    """YaRN progressive frequency scaling with smoothstep ramp.

    Matches the implementation used in phase17c_extended_eval.py.
    """
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


# ---------------------------------------------------------------------------
# Checkpoint loading (handles _orig_mod. prefix and rope buffer stripping)
# ---------------------------------------------------------------------------


def _load_state(path: str) -> dict:
    """Load a state dict from disk, normalising torch.compile prefixes."""
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)

    # Unwrap {"model": ...} wrapper if present
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format at {path}")

    # Strip _orig_mod. prefix from torch.compile
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }

    # Remove RoPE buffers (they will be rebuilt with the target inv_freq)
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


def load_model(
    cfg: dict, ckpt_path: str, inv_freq: torch.Tensor
) -> GPT:
    """Instantiate GPT, load checkpoint weights, and inject target inv_freq."""
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = _load_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING: missing non-rope keys: {other_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    # Force-set inv_freq on every attention block
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    return model


def swap_inv_freq(model: GPT, inv_freq: torch.Tensor, max_pos: int) -> None:
    """Replace RoPE inv_freq in-place and rebuild cos/sin cache."""
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    model.blocks[0].attn.rope._build(max_pos)


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_ppl_single_length(
    model: GPT,
    val_data: torch.Tensor,
    L: int,
    n_chunks: int = PPL_CHUNKS,
) -> Optional[float]:
    """Evaluate sliding-window PPL at a single context length.

    Returns PPL as a float, or None on OOM / insufficient data.
    """
    model.eval()
    model.extend_rope(L + 128)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)

    max_start = len(val_data) - L
    if max_start <= 0:
        return None
    actual_chunks = min(n_chunks, max(1, max_start // L))
    offsets = sorted(rng.choice(max_start, size=actual_chunks, replace=False))
    losses: List[float] = []

    for offset in offsets:
        chunk = val_data[offset : offset + L].unsqueeze(0).to(DEVICE)
        try:
            with ctx:
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    chunk[:, 1:].reshape(-1),
                )
            losses.append(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    L={L}: OOM on chunk offset={offset}, stopping")
                break
            raise
        finally:
            del chunk
            if L >= 16384 and DEVICE == "cuda":
                torch.cuda.empty_cache()

    if losses:
        return round(math.exp(sum(losses) / len(losses)), 3)
    return None


# ---------------------------------------------------------------------------
# Passkey evaluation
# ---------------------------------------------------------------------------


def eval_passkey_single_length(
    model: GPT,
    tokenizer,
    filler: torch.Tensor,
    L: int,
    depth: float = 0.5,
    trials: int = PK_TRIALS,
) -> Optional[Dict]:
    """Evaluate passkey retrieval at a single context length.

    Returns the result dict from eval_passkey_nll_gap, or None on OOM.
    """
    try:
        model.extend_rope(L + 128)
        result = eval_passkey_nll_gap(
            model,
            tokenizer,
            filler,
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


# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------


def auto_detect_checkpoints(model_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Search model_dir for geo and evq model.pt checkpoints.

    Looks inside seed42/ subdirectories following the naming convention from
    phase17c_454m_1024_to_2048_continue.py.
    """
    base = Path(model_dir)
    geo_ckpt = None
    evq_ckpt = None

    # Pattern 1: seed42/<tag>/model.pt
    seed_dir = base / "seed42"
    if seed_dir.exists():
        for sub in sorted(seed_dir.iterdir()):
            mp = sub / "model.pt"
            if mp.exists():
                name = sub.name.lower()
                if name.startswith("geo"):
                    geo_ckpt = str(mp)
                elif name.startswith("evq"):
                    evq_ckpt = str(mp)

    # Pattern 2: flat structure — geo_*/model.pt, evq_*/model.pt
    if geo_ckpt is None or evq_ckpt is None:
        for sub in sorted(base.iterdir()):
            mp = sub / "model.pt"
            if mp.exists():
                name = sub.name.lower()
                if name.startswith("geo") and geo_ckpt is None:
                    geo_ckpt = str(mp)
                elif name.startswith("evq") and evq_ckpt is None:
                    evq_ckpt = str(mp)

    # Pattern 3: direct model.pt files named geo_model.pt / evq_model.pt
    if geo_ckpt is None:
        for candidate in ["geo_model.pt", "geo.pt"]:
            p = base / candidate
            if p.exists():
                geo_ckpt = str(p)
                break
    if evq_ckpt is None:
        for candidate in ["evq_model.pt", "evq.pt"]:
            p = base / candidate
            if p.exists():
                evq_ckpt = str(p)
                break

    return geo_ckpt, evq_ckpt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_ppl_table(results: dict, lengths: List[int]) -> None:
    """Print a formatted PPL comparison table."""
    header = f"  {'Model':<25s} {'Mode':>6s}"
    for L in lengths:
        header += f"  {L // 1024:>5d}K"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_key in sorted(
        {k.rsplit("__", 1)[0] for k in results if not k.startswith("_")}
    ):
        for mode in MODES:
            key = f"{model_key}__{mode}"
            r = results.get(key, {})
            ppl = r.get("ppl", {})
            line = f"  {model_key:<25s} {mode:>6s}"
            for L in lengths:
                v = ppl.get(str(L))
                if v is not None:
                    line += f"  {v:>6.2f}" if v < 100 else f"  {v:>6.1f}"
                else:
                    line += f"  {'--':>6s}"
            print(line)
        print()


def print_passkey_table(results: dict, lengths: List[int]) -> None:
    """Print a formatted passkey retrieval table."""
    header = f"  {'Model':<25s} {'Mode':>6s}"
    for L in lengths:
        header += f"  {L // 1024:>5d}K"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_key in sorted(
        {k.rsplit("__", 1)[0] for k in results if not k.startswith("_")}
    ):
        for mode in MODES:
            key = f"{model_key}__{mode}"
            r = results.get(key, {})
            pk = r.get("passkey", {})
            line = f"  {model_key:<25s} {mode:>6s}"
            for L in lengths:
                v = pk.get(str(L), {})
                rr = v.get("retrieval_rate") if isinstance(v, dict) else None
                if rr is not None:
                    line += f"  {rr:>5.0%}"
                else:
                    line += f"  {'--':>6s}"
            print(line)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 454M checkpoints (GEO vs EVQ) at multiple context lengths"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/root/autodl-tmp/evq_phase17c_2048_continue",
        help="Root directory containing checkpoints (auto-detect geo/evq inside)",
    )
    parser.add_argument(
        "--geo_ckpt",
        type=str,
        default=None,
        help="Explicit path to GEO model.pt (overrides auto-detect)",
    )
    parser.add_argument(
        "--evq_ckpt",
        type=str,
        default=None,
        help="Explicit path to EVQ model.pt (overrides auto-detect)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to validation data .pt file (1-D token tensor). "
        "If not given, will look for cached proof-pile-2 or download.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results JSON. Defaults to <model_dir>/multilength_eval/",
    )
    parser.add_argument(
        "--ppl_lengths",
        type=str,
        default=",".join(str(x) for x in PPL_LENGTHS),
        help="Comma-separated PPL evaluation lengths",
    )
    parser.add_argument(
        "--pk_lengths",
        type=str,
        default=",".join(str(x) for x in PK_LENGTHS),
        help="Comma-separated passkey evaluation lengths",
    )
    parser.add_argument(
        "--ppl_chunks",
        type=int,
        default=PPL_CHUNKS,
        help="Number of sliding-window chunks per PPL evaluation",
    )
    parser.add_argument(
        "--pk_trials",
        type=int,
        default=PK_TRIALS,
        help="Number of passkey trials per length",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=5_000_000,
        help="Max validation tokens to load if downloading",
    )
    parser.add_argument(
        "--skip_ppl",
        action="store_true",
        help="Skip PPL evaluation",
    )
    parser.add_argument(
        "--skip_passkey",
        action="store_true",
        help="Skip passkey evaluation",
    )
    parser.add_argument(
        "--skip_geo",
        action="store_true",
        help="Skip GEO checkpoint evaluation",
    )
    parser.add_argument(
        "--skip_evq",
        action="store_true",
        help="Skip EVQ checkpoint evaluation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results JSON (skip completed model+mode combos)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ppl_lengths = [int(x) for x in args.ppl_lengths.split(",") if x.strip()]
    pk_lengths = [int(x) for x in args.pk_lengths.split(",") if x.strip()]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir) / "multilength_eval"
    result_file = output_dir / "eval_results.json"

    print("#" * 72)
    print("  454M Multi-Length Evaluation")
    print(f"  Device: {DEVICE}  dtype: {DTYPE}  autocast: {USE_AUTOCAST}")
    print(f"  PPL lengths: {ppl_lengths}")
    print(f"  Passkey lengths: {pk_lengths}")
    print(f"  PPL chunks: {args.ppl_chunks}  PK trials: {args.pk_trials}")
    print("#" * 72)

    # ── Resolve checkpoints ──────────────────────────────────────────────

    geo_ckpt = args.geo_ckpt
    evq_ckpt = args.evq_ckpt

    if geo_ckpt is None or evq_ckpt is None:
        print(f"\n  Auto-detecting checkpoints in: {args.model_dir}")
        auto_geo, auto_evq = auto_detect_checkpoints(args.model_dir)
        if geo_ckpt is None:
            geo_ckpt = auto_geo
        if evq_ckpt is None:
            evq_ckpt = auto_evq

    models_to_eval: Dict[str, dict] = {}

    if not args.skip_geo and geo_ckpt:
        if Path(geo_ckpt).exists():
            models_to_eval["geo"] = {
                "ckpt": geo_ckpt,
                "tau": 0.0,
                "label": f"Geometric (tau=0) L_train={L_TRAIN}",
            }
            print(f"  GEO checkpoint: {geo_ckpt}")
        else:
            print(f"  WARNING: GEO checkpoint not found: {geo_ckpt}")
    elif args.skip_geo:
        print("  GEO: skipped by --skip_geo")
    else:
        print("  WARNING: No GEO checkpoint found")

    if not args.skip_evq and evq_ckpt:
        if Path(evq_ckpt).exists():
            models_to_eval["evq"] = {
                "ckpt": evq_ckpt,
                "tau": TAU,
                "label": f"EVQ-Cosh (tau={TAU:.4f}) L_train={L_TRAIN}",
            }
            print(f"  EVQ checkpoint: {evq_ckpt}")
        else:
            print(f"  WARNING: EVQ checkpoint not found: {evq_ckpt}")
    elif args.skip_evq:
        print("  EVQ: skipped by --skip_evq")
    else:
        print("  WARNING: No EVQ checkpoint found")

    if not models_to_eval:
        print("\n  ERROR: No valid checkpoints found. Nothing to evaluate.")
        print("  Provide --geo_ckpt and/or --evq_ckpt, or set --model_dir to the")
        print("  directory containing seed42/<tag>/model.pt checkpoints.")
        sys.exit(1)

    # ── SDPA backend info (CUDA only) ────────────────────────────────────

    if DEVICE == "cuda":
        print(f"\n  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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

    # ── Load tokenizer & validation data ─────────────────────────────────

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print(f"\n  Tokenizer: {tok.__class__.__name__} (vocab={tok.vocab_size})")

    print("  Loading validation data...")
    if args.data_path and Path(args.data_path).exists():
        print(f"    From: {args.data_path}")
        val_data = torch.load(args.data_path, map_location="cpu", weights_only=True)
    else:
        # Try known cache locations, then download
        cache_candidates = [
            Path("/root/autodl-tmp/evq_phase17/val_proof-pile-2_5000000.pt"),
            Path(args.model_dir) / "data" / "val_fineweb-edu_5000000.pt",
        ]
        loaded = False
        for cache_path in cache_candidates:
            if cache_path.exists():
                print(f"    From cache: {cache_path}")
                val_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                loaded = True
                break
        if not loaded:
            print("    Downloading validation data (fineweb-edu)...")
            cache_dir = str(output_dir / "data")
            val_data = load_val(tok, args.val_tokens, "fineweb-edu", cache_dir=cache_dir)

    filler = val_data[:50000]
    print(f"  val_data: {val_data.shape if hasattr(val_data, 'shape') else len(val_data)} tokens")
    print(f"  filler: {filler.shape} tokens")

    # ── Load or init results (for resume) ────────────────────────────────

    if args.resume and result_file.exists():
        with open(result_file) as f:
            results = json.load(f)
        print(f"\n  Resuming from: {result_file}")
    else:
        results = {}

    t0_global = time.time()

    # ── Evaluate each model in each mode ─────────────────────────────────

    for model_key, model_info in models_to_eval.items():
        ckpt_path = model_info["ckpt"]
        tau = model_info["tau"]
        label = model_info["label"]

        # Compute base inv_freq for this model
        if tau == 0.0:
            base_inv = geometric_inv_freq()
        else:
            base_inv = evq_cosh_inv_freq(head_dim=DIM, tau=tau, base=BASE)

        for mode in MODES:
            mode_key = f"{model_key}__{mode}"

            # Skip completed if resuming
            if mode_key in results and results[mode_key].get("status") == "done":
                print(f"\n  [SKIP] {mode_key}: already complete")
                continue

            print(f"\n{'=' * 72}")
            print(f"  {mode_key}  ({label})")
            print(f"  ckpt: {ckpt_path}")
            print(f"  tau={tau}, mode={mode}")
            print(f"{'=' * 72}")

            # Load model fresh for each mode to avoid memory fragmentation
            cfg = {**CFG_454M, "max_position_embeddings": L_TRAIN, "seq_len": L_TRAIN}
            model = load_model(cfg, ckpt_path, base_inv)
            model.eval()
            print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

            ppl_results: Dict[str, float] = {}
            pk_results: Dict[str, dict] = {}
            t0_mode = time.time()

            # ── PPL evaluation ──

            if not args.skip_ppl:
                print(f"\n  --- PPL evaluation (mode={mode}) ---")
                for L in ppl_lengths:
                    scale = L / L_TRAIN

                    # Apply YaRN scaling if mode=yarn and L > L_train
                    if mode == "yarn" and scale > 1.0:
                        inv_freq = build_yarn_inv_freq(base_inv, DIM, scale)
                    else:
                        inv_freq = base_inv.clone()
                    swap_inv_freq(model, inv_freq, L + 128)

                    ppl = eval_ppl_single_length(model, val_data, L, args.ppl_chunks)
                    if ppl is not None:
                        ppl_results[str(L)] = ppl
                        yarn_note = f" +YaRN({scale:.1f}x)" if mode == "yarn" and scale > 1.0 else ""
                        print(f"    L={L:>6d} ({scale:>5.1f}x): PPL={ppl:>8.3f}{yarn_note}")
                    else:
                        print(f"    L={L:>6d}: OOM/skip")
                        # Remaining lengths will likely also OOM
                        break

                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            # ── Passkey evaluation ──

            if not args.skip_passkey:
                print(f"\n  --- Passkey evaluation (mode={mode}) ---")
                for L in pk_lengths:
                    scale = L / L_TRAIN

                    if mode == "yarn" and scale > 1.0:
                        inv_freq = build_yarn_inv_freq(base_inv, DIM, scale)
                    else:
                        inv_freq = base_inv.clone()
                    swap_inv_freq(model, inv_freq, L + 128)

                    pk = eval_passkey_single_length(
                        model, tok, filler, L,
                        depth=0.5, trials=args.pk_trials,
                    )
                    if pk is not None:
                        g = pk.get("global", {})
                        pk_results[str(L)] = {
                            "retrieval_rate": g.get("retrieval_rate"),
                            "mean_nll_gap": g.get("mean_nll_gap"),
                            "ar_exact_match": g.get("ar_exact_match_rate"),
                        }
                        yarn_note = f" +YaRN({scale:.1f}x)" if mode == "yarn" and scale > 1.0 else ""
                        print(
                            f"    PK L={L:>6d}: "
                            f"ret={g.get('retrieval_rate', 0):.0%}  "
                            f"gap={g.get('mean_nll_gap', 0):+.3f}  "
                            f"AR={g.get('ar_exact_match_rate', 0):.0%}"
                            f"{yarn_note}"
                        )
                    else:
                        print(f"    PK L={L:>6d}: OOM/skip")
                        break

                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            elapsed = time.time() - t0_mode
            results[mode_key] = {
                "status": "done",
                "model_key": model_key,
                "mode": mode,
                "label": label,
                "L_train": L_TRAIN,
                "tau": tau,
                "ckpt": ckpt_path,
                "ppl": ppl_results,
                "passkey": pk_results,
                "elapsed_sec": round(elapsed, 1),
            }

            # Incremental save after each model+mode
            save_json(result_file, results)
            print(f"\n    Saved results ({elapsed:.0f}s)")

            # Free model memory
            del model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    total_elapsed = time.time() - t0_global

    # ── Save final metadata ──────────────────────────────────────────────

    results["_meta"] = {
        "model": "454M",
        "architecture": CFG_454M,
        "base": BASE,
        "tau": TAU,
        "L_train": L_TRAIN,
        "ppl_lengths": ppl_lengths,
        "ppl_chunks": args.ppl_chunks,
        "pk_lengths": pk_lengths,
        "pk_trials": args.pk_trials,
        "pk_depths": PK_DEPTHS,
        "modes": MODES,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "total_elapsed_sec": round(total_elapsed, 1),
    }
    save_json(result_file, results)
    print(f"\n  Results saved to: {result_file}")

    # ── Print comparison tables ──────────────────────────────────────────

    print(f"\n{'=' * 100}")
    print("  PPL COMPARISON")
    print(f"{'=' * 100}")
    print_ppl_table(results, ppl_lengths)

    print(f"\n{'=' * 100}")
    print("  PASSKEY RETRIEVAL RATE")
    print(f"{'=' * 100}")
    print_passkey_table(results, pk_lengths)

    # ── Delta table (EVQ vs GEO improvement) ─────────────────────────────

    if "geo" in models_to_eval and "evq" in models_to_eval:
        print(f"\n{'=' * 100}")
        print("  EVQ vs GEO IMPROVEMENT (negative = EVQ is better)")
        print(f"{'=' * 100}")
        for mode in MODES:
            geo_key = f"geo__{mode}"
            evq_key = f"evq__{mode}"
            geo_ppl = results.get(geo_key, {}).get("ppl", {})
            evq_ppl = results.get(evq_key, {}).get("ppl", {})

            header = f"  Mode={mode:>4s}"
            for L in ppl_lengths:
                header += f"  {L // 1024:>5d}K"
            print(header)

            line = f"  {'delta%':>10s}"
            for L in ppl_lengths:
                g = geo_ppl.get(str(L))
                e = evq_ppl.get(str(L))
                if g is not None and e is not None and g > 0:
                    delta = (e / g - 1.0) * 100
                    line += f"  {delta:>+5.1f}%"
                else:
                    line += f"  {'--':>6s}"
            print(line)
            print()

    print(f"\n  Total elapsed: {total_elapsed / 60:.1f} min")
    print("  DONE!")


if __name__ == "__main__":
    main()
