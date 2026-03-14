#!/usr/bin/env python3
"""
Strict FineWeb-Edu raw + YaRN eval for Phase17H checkpoints.

Uses the repository-standard progressive YaRN implementation from
`eval_pe_baselines.py` and the standard auto scale rule:

    scale = L_eval / L_train

The validation tensor is provided explicitly to avoid any dataset fallback.
"""

import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    evq_cosh_inv_freq,
)


BASE = 500_000.0
DIM = 64
SEQ_LEN = 1024
CFG = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=64,
    intermediate_size=3072,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
)

WORK = Path("/root/autodl-tmp/evq_phase17h_125m_L1024_fixed")
VAL_PATH = WORK / "data_cache" / "val_fineweb-edu_5000000.pt"
OUT_PATH = WORK / "compare_seed7_fineweb_yarn_eval.json"
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
EVAL_CHUNKS = 32


def geometric_inv_freq(dim: int = DIM, base: float = BASE) -> torch.Tensor:
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(dim // 2)],
        dtype=torch.float32,
    )


def build_yarn_inv_freq(base_inv: torch.Tensor, head_dim: int, scale: float) -> torch.Tensor:
    """Exact progressive YaRN used in eval_pe_baselines.py."""
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1.0 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        key = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        if ".rope." in key:
            continue
        cleaned[key] = value.detach().cpu()
    return cleaned


def load_clean_state(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return clean_state_dict(ckpt)


def apply_state(model: GPT, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing or unexpected:
        raise RuntimeError(f"missing={other_missing} unexpected={unexpected}")


def build_model(inv_freq: torch.Tensor, state):
    model = GPT(CFG, inv_freq).to(DEVICE)
    apply_state(model, state)
    return model


def set_inv_freq(model: GPT, inv_freq: torch.Tensor, max_len: int):
    inv_freq = inv_freq.to(DEVICE)
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq)
        block.attn.rope._build(max_len + 100)


def eval_ppl_single(model: GPT, val_data: torch.Tensor, inv_freq: torch.Tensor, L: int) -> float:
    set_inv_freq(model, inv_freq, L)
    model.eval()
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)

    max_start = len(val_data) - L
    if max_start <= 0:
        raise RuntimeError(f"val_data too short for L={L}")
    offsets = sorted(
        rng.choice(max_start, size=min(EVAL_CHUNKS, max_start // L), replace=False)
    )
    losses = []
    for offset in offsets:
        chunk = val_data[offset : offset + L].unsqueeze(0).to(DEVICE)
        with torch.no_grad(), ctx:
            logits = model(chunk[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1),
            )
        losses.append(loss.item())
        del chunk
    return round(math.exp(sum(losses) / len(losses)), 3)


def pct_delta(a: float, b: float):
    if b == 0:
        return None
    return round((a / b - 1.0) * 100.0, 3)


def main():
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"missing val tensor: {VAL_PATH}")

    val = torch.load(VAL_PATH, map_location="cpu", weights_only=True).reshape(-1).long()
    raw_ppl = {}

    yarn_rows = {}
    for arm, tau in [("geo", 0.0), ("evq", 2.0)]:
        state = load_clean_state(WORK / f"{arm}_seed7" / "model.pt")
        base_inv = (
            geometric_inv_freq()
            if arm == "geo"
            else evq_cosh_inv_freq(head_dim=DIM, tau=tau, base=BASE)
        )
        model = build_model(base_inv, state)
        raw_rows = {}
        yarn_ppl = {}
        for L in EVAL_LENGTHS:
            raw_rows[str(L)] = eval_ppl_single(model, val, base_inv, L)
            scale = max(L / SEQ_LEN, 1.0)
            if scale <= 1.0:
                yarn_ppl[str(L)] = raw_rows[str(L)]
            else:
                yarn_inv = build_yarn_inv_freq(base_inv, DIM, scale)
                yarn_ppl[str(L)] = eval_ppl_single(model, val, yarn_inv, L)
            print(
                f"DONE {arm} raw L={L} ppl={raw_rows[str(L)]} | "
                f"yarn_auto scale={scale:.3f} ppl={yarn_ppl[str(L)]}",
                flush=True,
            )
        raw_ppl[arm] = raw_rows
        yarn_rows[arm] = yarn_ppl
        del model
        torch.cuda.empty_cache()

    yarn_vs_raw_pct = {
        arm: {
            str(L): pct_delta(yarn_rows[arm][str(L)], raw_ppl[arm][str(L)])
            for L in EVAL_LENGTHS
        }
        for arm in ["geo", "evq"]
    }
    evq_vs_geo_pct = {
        mode: {
            str(L): pct_delta(rows["evq"][str(L)], rows["geo"][str(L)])
            for L in EVAL_LENGTHS
        }
        for mode, rows in [("raw", raw_ppl), ("yarn_auto", yarn_rows)]
    }

    out = {
        "dataset": "fineweb-edu",
        "eval_lengths": EVAL_LENGTHS,
        "eval_chunks": EVAL_CHUNKS,
        "train_len": SEQ_LEN,
        "yarn_impl": "progressive_yarn_from_eval_pe_baselines",
        "yarn_scale_rule": "scale=L_eval/L_train",
        "raw": raw_ppl,
        "yarn_auto": yarn_rows,
        "yarn_vs_raw_pct": yarn_vs_raw_pct,
        "evq_vs_geo_pct": evq_vs_geo_pct,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
