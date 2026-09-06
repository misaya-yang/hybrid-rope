#!/usr/bin/env python3
"""
Phase 17E: 454M from-scratch frozen-tau probe at L=2048, 100M tokens.

Purpose:
  Test the user's intended experiment directly:
  train from scratch at L=2048 with mixed passkey data, but keep EVQ tau frozen
  at the stage-1 value tau=64/sqrt(512) ~= 2.8284 instead of using tau*(2048).

Defaults:
  - from-scratch EVQ only
  - train length: 2048
  - train tokens: 100M
  - frozen tau: 2.8284
  - single-passkey mix: 5%
  - probe eval every 25M tokens
  - raw PPL @ 1K/2K/4K/8K/16K
  - NIAH as supporting signal after final checkpoint

Required files on server:
  /root/autodl-tmp/evq_fresh_2048/train_2048_500M.pt
  /root/autodl-tmp/evq_multiseed_length/data/val_proof-pile-2_5M.pt

Environment overrides:
  PHASE17E_WORK
  PHASE17E_TRAIN_DATA
  PHASE17E_VAL_DATA
  PHASE17E_SEQ_LEN
  PHASE17E_TRAIN_TOKENS
  PHASE17E_FROZEN_TAU
  PHASE17E_SEED
  PHASE17E_MICRO_BATCH_SIZE
  PHASE17E_GRAD_ACCUM
  PHASE17E_PASSKEY_MIX_RATIO
  PHASE17E_EVAL_LENGTHS
  PHASE17E_EVAL_CHUNKS
  PHASE17E_EVAL_EVERY_TOKENS
  PHASE17E_NIAH_LENGTHS
  PHASE17E_NIAH_TRIALS
  PHASE17E_SAVE_MIXED_CACHE
  PHASE17E_COMPILE
"""

import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "supporting_eval"))

from eval_passkey_scratch import eval_passkey_nll_gap, make_passkey_training_sample  # noqa: E402
from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST, evq_cosh_inv_freq, set_seed  # noqa: E402


BASE = 500_000.0
DIM = 64
METHOD = os.environ.get("PHASE17E_METHOD", "evq").strip().lower()
if METHOD not in {"evq", "geo"}:
    raise ValueError(f"PHASE17E_METHOD must be 'evq' or 'geo', got {METHOD}")
SEQ_LEN = int(os.environ.get("PHASE17E_SEQ_LEN", "2048"))
TRAIN_TOKENS = int(os.environ.get("PHASE17E_TRAIN_TOKENS", "100000000"))
FROZEN_TAU = float(
    os.environ.get(
        "PHASE17E_FROZEN_TAU",
        f"{DIM / math.sqrt(512):.6f}",
    )
)
SEED = int(os.environ.get("PHASE17E_SEED", "7"))
LR = 2e-4
MIN_LR = LR * 0.1
MICRO_BS = int(os.environ.get("PHASE17E_MICRO_BATCH_SIZE", "5"))
GRAD_ACCUM = int(os.environ.get("PHASE17E_GRAD_ACCUM", "4"))
EFF_BS = MICRO_BS * GRAD_ACCUM
PASSKEY_MIX_RATIO = float(os.environ.get("PHASE17E_PASSKEY_MIX_RATIO", "0.05"))
SAVE_MIXED_CACHE = os.environ.get("PHASE17E_SAVE_MIXED_CACHE", "1") == "1"
USE_COMPILE = os.environ.get("PHASE17E_COMPILE", "1") == "1"

EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17E_EVAL_LENGTHS",
        "1024,2048,4096,8192,16384",
    ).split(",")
    if x.strip()
]
EVAL_CHUNKS = int(os.environ.get("PHASE17E_EVAL_CHUNKS", "10"))
EVAL_EVERY_TOKENS = int(os.environ.get("PHASE17E_EVAL_EVERY_TOKENS", "25000000"))

NIAH_LENGTHS = [
    int(x)
    for x in os.environ.get("PHASE17E_NIAH_LENGTHS", "2048,4096,8192").split(",")
    if x.strip()
]
NIAH_TRIALS = int(os.environ.get("PHASE17E_NIAH_TRIALS", "10"))

WORK = Path(
    os.environ.get(
        "PHASE17E_WORK",
        "/root/autodl-tmp/evq_phase17e_from_scratch_frozen_tau",
    )
)
TRAIN_DATA = Path(
    os.environ.get(
        "PHASE17E_TRAIN_DATA",
        "/root/autodl-tmp/evq_fresh_2048/train_2048_500M.pt",
    )
)
VAL_DATA = Path(
    os.environ.get(
        "PHASE17E_VAL_DATA",
        "/root/autodl-tmp/evq_multiseed_length/data/val_proof-pile-2_5M.pt",
    )
)

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


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def geometric_inv_freq():
    return torch.tensor(
        [1.0 / (BASE ** (2 * i / DIM)) for i in range(DIM // 2)],
        dtype=torch.float32,
    )


def _autocast_ctx():
    if USE_AUTOCAST:
        return torch.autocast("cuda", dtype=DTYPE)
    return nullcontext()


def build_inv_freq():
    if METHOD == "geo":
        return geometric_inv_freq()
    return evq_cosh_inv_freq(head_dim=DIM, tau=FROZEN_TAU, base=BASE)


def build_model(compile_model: bool = True):
    model = GPT(CFG_454M, build_inv_freq()).to(DEVICE)
    if compile_model and USE_COMPILE:
        model = torch.compile(model, mode="default")
    return model


def load_train_tensor():
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(f"train data missing: {TRAIN_DATA}")
    raw = torch.load(TRAIN_DATA, map_location="cpu", weights_only=True)
    if raw.ndim == 1:
        usable = (raw.numel() // SEQ_LEN) * SEQ_LEN
        raw = raw[:usable].view(-1, SEQ_LEN)
    elif raw.ndim != 2 or raw.shape[1] != SEQ_LEN:
        raise ValueError(f"unexpected train data shape: {tuple(raw.shape)}")
    need_rows = TRAIN_TOKENS // SEQ_LEN
    if raw.shape[0] < need_rows:
        raise ValueError(f"need {need_rows} rows but only have {raw.shape[0]}")
    return raw[:need_rows].to(torch.int32).clone()


def build_mixed_train_tensor(train_data: torch.Tensor):
    cache = WORK / f"mixed_seed{SEED}_tok{TRAIN_TOKENS}_pk{int(PASSKEY_MIX_RATIO * 100)}.pt"
    if SAVE_MIXED_CACHE and cache.exists():
        print(f"  [data] loading mixed cache: {cache}", flush=True)
        return torch.load(cache, map_location="cpu", weights_only=True)

    if PASSKEY_MIX_RATIO <= 0:
        return train_data

    print("  [data] building single-passkey mixed tensor", flush=True)
    from transformers import AutoTokenizer

    val = torch.load(VAL_DATA, map_location="cpu", weights_only=True)
    if val.dtype != torch.int32:
        val = val.to(torch.int32)
    filler = val[:50000]
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    mixed = train_data.clone()
    n_rows = mixed.shape[0]
    n_pk = int(n_rows * PASSKEY_MIX_RATIO)
    set_seed(SEED)
    indices = torch.randperm(n_rows)[:n_pk]
    print(f"  [data] injecting {n_pk} passkey rows", flush=True)
    for i, idx in enumerate(indices.tolist(), start=1):
        mixed[idx] = make_passkey_training_sample(filler, tok, seq_len=SEQ_LEN, seed=SEED + idx).to(torch.int32)
        if i % 1000 == 0 or i == n_pk:
            print(f"    passkey rows: {i}/{n_pk}", flush=True)

    if SAVE_MIXED_CACHE:
        torch.save(mixed, cache)
        save_json(
            cache.with_suffix(".json"),
            {
                "seed": SEED,
                "train_tokens": TRAIN_TOKENS,
                "seq_len": SEQ_LEN,
                "passkey_mix_ratio": PASSKEY_MIX_RATIO,
                "source_train_data": str(TRAIN_DATA),
            },
        )
    return mixed


def _iter_eval_chunks(flat_tokens: torch.Tensor, length: int, max_chunks: int):
    total = flat_tokens.numel() // length
    for idx in range(min(total, max_chunks)):
        start = idx * length
        yield flat_tokens[start:start + length]


@torch.no_grad()
def eval_ppl(model_path: Path, lengths):
    model = build_model(compile_model=False)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)
    model.eval()

    val = torch.load(VAL_DATA, map_location="cpu", weights_only=True).reshape(-1).long()
    out = {}
    for length in lengths:
        ppls = []
        for chunk in _iter_eval_chunks(val, length, EVAL_CHUNKS):
            batch = chunk.unsqueeze(0).to(DEVICE)
            with _autocast_ctx():
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            ppls.append(math.exp(loss.item()))
        if ppls:
            out[str(length)] = round(sum(ppls) / len(ppls), 4)
            print(f"    PPL@{length}: {out[str(length)]:.4f} (n={len(ppls)})", flush=True)

    del model, val
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def eval_niah(model_path: Path):
    from transformers import AutoTokenizer

    model = build_model(compile_model=False)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)
    model.eval()

    filler = torch.load(VAL_DATA, map_location="cpu", weights_only=True).reshape(-1)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    results = eval_passkey_nll_gap(
        model=model,
        tokenizer=tok,
        filler_tokens=filler,
        lengths=NIAH_LENGTHS,
        depths=[0.1, 0.3, 0.5, 0.7, 0.9],
        num_trials=NIAH_TRIALS,
    )
    del model, filler, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def checkpoint_eval(run_dir: Path, model_path: Path, step: int, tokens_done: int):
    print(f"\n  [eval] step={step} tokens={tokens_done/1e6:.1f}M", flush=True)
    ppl = eval_ppl(model_path, EVAL_LENGTHS)
    record = {
        "step": step,
        "tokens_done": tokens_done,
        "ppl": ppl,
    }
    save_json(run_dir / f"probe_eval_step_{step:05d}.json", record)
    return record


def train():
    WORK.mkdir(parents=True, exist_ok=True)
    run_tag = f"{METHOD}_seed{SEED}"
    if METHOD == "evq":
        run_tag += f"_tau{FROZEN_TAU:.2f}"
    run_dir = WORK / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Phase17E from-scratch frozen-tau probe", flush=True)
    print(f"  method: {METHOD}", flush=True)
    print(f"  train_data: {TRAIN_DATA}", flush=True)
    print(f"  val_data: {VAL_DATA}", flush=True)
    print(f"  seq_len: {SEQ_LEN}", flush=True)
    print(f"  train_tokens: {TRAIN_TOKENS/1e6:.1f}M", flush=True)
    if METHOD == "evq":
        print(f"  frozen_tau: {FROZEN_TAU:.6f}", flush=True)
    else:
        print("  rope: geometric (tau=0)", flush=True)
    print(f"  passkey_mix_ratio: {PASSKEY_MIX_RATIO:.2%}", flush=True)
    print("=" * 72, flush=True)

    train_data = build_mixed_train_tensor(load_train_tensor())
    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    seqs_per_step = EFF_BS
    tokens_per_step = seqs_per_step * SEQ_LEN
    total_steps = TRAIN_TOKENS // tokens_per_step
    warmup = max(1, int(total_steps * 0.02))
    eval_interval_steps = max(1, round(EVAL_EVERY_TOKENS / tokens_per_step))

    data_gpu = train_data.to(DEVICE)
    del train_data
    torch.cuda.empty_cache()

    t0 = time.time()
    seq_idx = 0
    running_loss = 0.0
    n_loss = 0
    probe_records = []

    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(GRAD_ACCUM):
            batch = data_gpu[seq_idx:seq_idx + MICRO_BS].long()
            seq_idx += MICRO_BS
            with _autocast_ctx():
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            (loss / GRAD_ACCUM).backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step <= warmup:
            lr = LR * step / warmup
        else:
            prog = (step - warmup) / max(1, total_steps - warmup)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        running_loss += step_loss / GRAD_ACCUM
        n_loss += 1
        tokens_done = step * tokens_per_step

        if step % 50 == 0 or step == total_steps:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, n_loss)
            eta_h = (elapsed / step) * (total_steps - step) / 3600 if step < total_steps else 0.0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"  step {step}/{total_steps}  loss={avg_loss:.4f}  lr={lr:.2e}  "
                f"tokens={tokens_done/1e6:.1f}M  GPU={gpu_mem:.1f}GB  ETA={eta_h:.2f}h",
                flush=True,
            )
            running_loss = 0.0
            n_loss = 0

        if step % eval_interval_steps == 0 or step == total_steps:
            ckpt = run_dir / f"step_{step:05d}.pt"
            torch.save({"model": model.state_dict()}, ckpt)
            probe_records.append(checkpoint_eval(run_dir, ckpt, step, tokens_done))

    final_model = run_dir / "model.pt"
    torch.save({"model": model.state_dict()}, final_model)
    niah = eval_niah(final_model)

    save_json(
        run_dir / "train_meta.json",
        {
            "seed": SEED,
            "train_data": str(TRAIN_DATA),
            "val_data": str(VAL_DATA),
            "train_tokens": TRAIN_TOKENS,
            "seq_len": SEQ_LEN,
            "method": METHOD,
            "frozen_tau": FROZEN_TAU if METHOD == "evq" else 0.0,
            "passkey_mix_ratio": PASSKEY_MIX_RATIO,
            "micro_batch_size": MICRO_BS,
            "grad_accum": GRAD_ACCUM,
            "eval_lengths": EVAL_LENGTHS,
            "eval_every_tokens": EVAL_EVERY_TOKENS,
            "niah_lengths": NIAH_LENGTHS,
            "niah_trials": NIAH_TRIALS,
            "probe_records": probe_records,
            "niah": niah,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    )
    print(f"\nDONE: {final_model}", flush=True)


if __name__ == "__main__":
    train()
