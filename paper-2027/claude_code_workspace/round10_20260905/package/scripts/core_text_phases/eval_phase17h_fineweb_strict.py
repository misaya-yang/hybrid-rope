#!/usr/bin/env python3
"""
Strict in-domain re-eval for Phase17H checkpoints.

Purpose:
  Re-evaluate the finished 125M L=1024 fixed-length Geo/EVQ checkpoints on a
  caller-provided FineWeb-Edu validation tensor, with no dataset fallback.
"""

import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, eval_model, evq_cosh_inv_freq  # noqa: E402


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

WORK = Path(
    "/root/autodl-tmp/evq_phase17h_125m_L1024_fixed"
)
VAL_PATH = WORK / "data_cache" / "val_fineweb-edu_5000000.pt"
OUT_PATH = WORK / "compare_seed7_fineweb_eval.json"
EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
EVAL_CHUNKS = 32


def geometric_inv_freq(dim: int = DIM, base: float = BASE) -> torch.Tensor:
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(dim // 2)],
        dtype=torch.float32,
    )


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


def main():
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"missing val tensor: {VAL_PATH}")

    val = torch.load(VAL_PATH, map_location="cpu", weights_only=True).reshape(-1).long()
    rows = {}
    for arm, tau in [("geo", 0.0), ("evq", 2.0)]:
        state = load_clean_state(WORK / f"{arm}_seed7" / "model.pt")
        inv = geometric_inv_freq() if arm == "geo" else evq_cosh_inv_freq(head_dim=DIM, tau=tau, base=BASE)
        model = build_model(inv, state)
        ppl = eval_model(model, val, EVAL_LENGTHS, eval_chunks=EVAL_CHUNKS)
        rows[arm] = ppl
        del model
        torch.cuda.empty_cache()
        print(f"DONE {arm} {json.dumps(ppl)}", flush=True)

    out = {
        "dataset": "fineweb-edu",
        "eval_lengths": EVAL_LENGTHS,
        "eval_chunks": EVAL_CHUNKS,
        "final_ppl": rows,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
