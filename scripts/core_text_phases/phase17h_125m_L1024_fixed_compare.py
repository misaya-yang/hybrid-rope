#!/usr/bin/env python3
"""
Phase 17H: 125M fixed-length L=1024 Geo vs EVQ comparison.

Purpose:
  Run the cleanest possible 125M from-scratch comparison after the staged probes:
  fixed training length, fixed data budget, no tau retargeting during training because
  the length never changes.

Default protocol:
  - 125M model
  - from-scratch, single stage @ L=1024
  - 399,998,976 tokens (nearest <= 400M divisible by 1024 * 96)
  - Geo: tau=0.0 exactly
  - EVQ: tau*=64/sqrt(1024)=2.0
  - pure LM by default (0% passkey mix)
  - final raw PPL eval only
  - eval lengths: 512,1024,2048,4096,8192,16384
"""

import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

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

from eval_passkey_scratch import make_passkey_training_sample  # noqa: E402
from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    set_seed,
)


BASE = 500_000.0
DIM = 64
CFG_125M = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=64,
    intermediate_size=3072,
)

ARMS = {"geo", "evq"}
ARM = os.environ.get("PHASE17H_ARM", "geo").strip().lower()
if ARM not in ARMS:
    raise ValueError(f"PHASE17H_ARM must be one of {sorted(ARMS)}, got {ARM}")

SEQ_LEN = int(os.environ.get("PHASE17H_SEQ_LEN", "1024"))
TRAIN_TOKENS = int(os.environ.get("PHASE17H_TRAIN_TOKENS", "399998976"))
SEED = int(os.environ.get("PHASE17H_SEED", "7"))
LR = float(os.environ.get("PHASE17H_LR", "6e-4"))
MIN_LR = LR * 0.1
WARMUP_FRAC = float(os.environ.get("PHASE17H_WARMUP_FRAC", "0.02"))
MICRO_BS = int(os.environ.get("PHASE17H_MICRO_BATCH_SIZE", "24"))
GRAD_ACCUM = int(os.environ.get("PHASE17H_GRAD_ACCUM", "4"))
EFF_BS = MICRO_BS * GRAD_ACCUM
PASSKEY_MIX_RATIO = float(os.environ.get("PHASE17H_PASSKEY_MIX_RATIO", "0.0"))
USE_COMPILE = os.environ.get("PHASE17H_COMPILE", "1") == "1"
FORCE_REDO = os.environ.get("PHASE17H_FORCE_REDO", "0") == "1"

EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17H_EVAL_LENGTHS",
        "512,1024,2048,4096,8192,16384",
    ).split(",")
    if x.strip()
]
EVAL_CHUNKS = int(os.environ.get("PHASE17H_EVAL_CHUNKS", "32"))

WORK = Path(
    os.environ.get(
        "PHASE17H_WORK",
        "/root/autodl-tmp/evq_phase17h_125m_L1024_fixed",
    )
)
TRAIN_DATA = Path(
    os.environ.get(
        "PHASE17H_TRAIN_DATA",
        "/root/autodl-tmp/evq_fresh_2048/train_2048_500M.pt",
    )
)
VAL_DATA = Path(
    os.environ.get(
        "PHASE17H_VAL_DATA",
        "/root/autodl-tmp/evq_multiseed_length/data/val_proof-pile-2_5M.pt",
    )
)

_TRAIN_FLAT = None
_VAL_FLAT = None
_TOKENIZER = None


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def geometric_inv_freq(dim: int = DIM, base: float = BASE) -> torch.Tensor:
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(dim // 2)],
        dtype=torch.float32,
    )


def tau_star(seq_len: int) -> float:
    return DIM / math.sqrt(seq_len)


def arm_tau() -> float:
    return 0.0 if ARM == "geo" else tau_star(SEQ_LEN)


def _autocast_ctx():
    if USE_AUTOCAST:
        return torch.autocast("cuda", dtype=DTYPE)
    return nullcontext()


def cfg_for_len(seq_len: int) -> Dict[str, int]:
    cfg = dict(CFG_125M)
    cfg["max_position_embeddings"] = seq_len
    cfg["seq_len"] = seq_len
    return cfg


def build_inv_freq() -> torch.Tensor:
    if ARM == "geo":
        return geometric_inv_freq()
    return evq_cosh_inv_freq(head_dim=DIM, tau=arm_tau(), base=BASE)


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        key = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        if ".rope." in key:
            continue
        cleaned[key] = value.detach().cpu()
    return cleaned


def load_clean_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"unsupported checkpoint at {path}")
    return clean_state_dict(ckpt)


def apply_state(model: GPT, state: Dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        raise RuntimeError(f"missing non-rope keys: {other_missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected}")


def build_model(state: Dict[str, torch.Tensor] | None = None, compile_model: bool = True):
    model = GPT(cfg_for_len(SEQ_LEN), build_inv_freq()).to(DEVICE)
    if state is not None:
        apply_state(model, state)
    if compile_model and USE_COMPILE:
        model = torch.compile(model, mode="default")
    return model


def load_train_flat() -> torch.Tensor:
    global _TRAIN_FLAT
    if _TRAIN_FLAT is None:
        if not TRAIN_DATA.exists():
            raise FileNotFoundError(f"train data missing: {TRAIN_DATA}")
        _TRAIN_FLAT = torch.load(TRAIN_DATA, map_location="cpu", weights_only=True).reshape(-1).to(torch.int32)
    return _TRAIN_FLAT


def load_val_flat() -> torch.Tensor:
    global _VAL_FLAT
    if _VAL_FLAT is None:
        if not VAL_DATA.exists():
            raise FileNotFoundError(f"val data missing: {VAL_DATA}")
        _VAL_FLAT = torch.load(VAL_DATA, map_location="cpu", weights_only=True).reshape(-1).long()
    return _VAL_FLAT


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return _TOKENIZER


def arm_dir() -> Path:
    return WORK / f"{ARM}_seed{SEED}"


def summary_path() -> Path:
    return arm_dir() / "summary.json"


def model_path() -> Path:
    return arm_dir() / "model.pt"


def result_path() -> Path:
    return arm_dir() / "result.json"


def build_train_tensor() -> tuple[torch.Tensor, Dict[str, int]]:
    cache_root = WORK / "data_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"train_L{SEQ_LEN}_tok{TRAIN_TOKENS}_pk{int(PASSKEY_MIX_RATIO * 100)}_seed{SEED}.pt"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and not FORCE_REDO:
        data = torch.load(cache, map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return data, meta

    flat = load_train_flat()
    if TRAIN_TOKENS > flat.numel():
        raise ValueError(f"need tokens up to {TRAIN_TOKENS}, source only has {flat.numel()}")
    if TRAIN_TOKENS % SEQ_LEN != 0:
        raise ValueError(f"train_tokens={TRAIN_TOKENS} not divisible by seq_len={SEQ_LEN}")
    if (TRAIN_TOKENS // SEQ_LEN) % EFF_BS != 0:
        raise ValueError(f"train rows must be divisible by effective batch size {EFF_BS}")

    data = flat[:TRAIN_TOKENS].clone().view(-1, SEQ_LEN)
    meta = {
        "seq_len": SEQ_LEN,
        "start_token": 0,
        "end_token": TRAIN_TOKENS,
        "usable_tokens": int(data.numel()),
        "n_rows": int(data.shape[0]),
        "passkey_mix_ratio": PASSKEY_MIX_RATIO,
        "source_train_data": str(TRAIN_DATA),
    }

    if PASSKEY_MIX_RATIO > 0:
        filler = load_val_flat().to(torch.int32)
        tok = get_tokenizer()
        mixed = data.clone()
        n_rows = mixed.shape[0]
        n_pk = int(round(n_rows * PASSKEY_MIX_RATIO))
        set_seed(SEED)
        indices = torch.randperm(n_rows)[:n_pk]
        for i, idx in enumerate(indices.tolist(), start=1):
            mixed[idx] = make_passkey_training_sample(
                filler,
                tok,
                seq_len=SEQ_LEN,
                seed=SEED + idx,
            ).to(torch.int32)
            if i % 2000 == 0 or i == n_pk:
                print(f"    [mix] passkey rows {i}/{n_pk}", flush=True)
        data = mixed
        meta["passkey_rows"] = n_pk

    torch.save(data, cache)
    save_json(meta_path, meta)
    return data, meta


def eval_ppl_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model = build_model(state=state, compile_model=False)
    val = load_val_flat()
    ppl = eval_model(model, val, EVAL_LENGTHS, eval_chunks=EVAL_CHUNKS)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ppl


def maybe_write_compare_summary() -> None:
    compare = {}
    ready = True
    for arm in sorted(ARMS):
        p = WORK / f"{arm}_seed{SEED}" / "summary.json"
        if not p.exists():
            ready = False
            continue
        compare[arm] = json.loads(p.read_text())
    if not ready:
        return

    rows = {arm: data.get("final_ppl", {}) for arm, data in compare.items()}
    geo = rows.get("geo", {})
    delta_vs_geo = {}
    if "evq" in rows:
        delta_vs_geo["evq"] = {
            k: round((rows["evq"][k] - geo[k]) / geo[k] * 100.0, 2)
            for k in geo
            if k in rows["evq"] and geo[k] > 0
        }

    save_json(
        WORK / f"compare_seed{SEED}.json",
        {
            "seed": SEED,
            "seq_len": SEQ_LEN,
            "train_tokens": TRAIN_TOKENS,
            "final_ppl": rows,
            "delta_vs_geo_pct": delta_vs_geo,
        },
    )


def run_arm() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    arm_dir().mkdir(parents=True, exist_ok=True)
    if result_path().exists() and model_path().exists() and not FORCE_REDO:
        print(f"[resume] {ARM} using existing result {result_path()}", flush=True)
        maybe_write_compare_summary()
        return

    print("=" * 72, flush=True)
    print("Phase17H 125M fixed-length L=1024 compare", flush=True)
    print(f"  arm: {ARM}", flush=True)
    print(f"  seed: {SEED}", flush=True)
    print(f"  seq_len: {SEQ_LEN}", flush=True)
    print(f"  train_tokens: {TRAIN_TOKENS}", flush=True)
    print(f"  micro_batch_size: {MICRO_BS}", flush=True)
    print(f"  grad_accum: {GRAD_ACCUM}", flush=True)
    print(f"  passkey_mix_ratio: {PASSKEY_MIX_RATIO:.2%}", flush=True)
    print(f"  tau: {arm_tau():.6f}", flush=True)
    print(f"  eval_lengths: {EVAL_LENGTHS}", flush=True)
    print(f"  eval_chunks: {EVAL_CHUNKS}", flush=True)
    print("=" * 72, flush=True)

    train_data, data_meta = build_train_tensor()
    model = build_model(compile_model=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    total_rows = train_data.shape[0]
    total_steps = total_rows // EFF_BS
    warmup = max(1, int(total_steps * WARMUP_FRAC))
    log_every = max(1, total_steps // 20)
    rows_gpu = train_data.to(DEVICE, non_blocking=True)
    del train_data
    torch.cuda.empty_cache()

    set_seed(SEED)
    perm = torch.randperm(total_rows, device=DEVICE)
    ptr = 0
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    last_loss = None

    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            idx = perm[ptr : ptr + MICRO_BS]
            ptr += MICRO_BS
            batch = rows_gpu[idx].long()
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

        last_loss = step_loss / GRAD_ACCUM
        running_loss += last_loss
        running_count += 1

        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, running_count)
            eta_h = (elapsed / step) * (total_steps - step) / 3600 if step < total_steps else 0.0
            print(
                f"  step {step}/{total_steps} loss={avg_loss:.4f} lr={lr:.2e} "
                f"L={SEQ_LEN} ETA={eta_h:.2f}h",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    clean_state = clean_state_dict(model.state_dict())
    torch.save(
        {
            "model": clean_state,
            "arm": ARM,
            "seq_len": SEQ_LEN,
            "tau": arm_tau(),
        },
        model_path(),
    )
    del model, optimizer, rows_gpu, perm
    gc.collect()
    torch.cuda.empty_cache()

    ppl = eval_ppl_from_state(clean_state)
    result = {
        "arm": ARM,
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "tau": arm_tau(),
        "usable_tokens": TRAIN_TOKENS,
        "micro_batch_size": MICRO_BS,
        "grad_accum": GRAD_ACCUM,
        "effective_batch_size": EFF_BS,
        "tokens_per_step": EFF_BS * SEQ_LEN,
        "steps": total_steps,
        "final_loss": round(float(last_loss), 6) if last_loss is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "ppl": ppl,
        "data": data_meta,
    }
    save_json(result_path(), result)
    summary = {
        "arm": ARM,
        "seed": SEED,
        "train_data": str(TRAIN_DATA),
        "val_data": str(VAL_DATA),
        "seq_len": SEQ_LEN,
        "train_tokens": TRAIN_TOKENS,
        "nominal_train_tokens": 400_000_000,
        "micro_batch_size": MICRO_BS,
        "grad_accum": GRAD_ACCUM,
        "passkey_mix_ratio": PASSKEY_MIX_RATIO,
        "tau": arm_tau(),
        "eval_lengths": EVAL_LENGTHS,
        "eval_chunks": EVAL_CHUNKS,
        "result": result,
        "final_ppl": ppl,
    }
    save_json(summary_path(), summary)
    maybe_write_compare_summary()
    print(f"\nDONE: {summary_path()}", flush=True)


if __name__ == "__main__":
    run_arm()
