#!/usr/bin/env python3
"""
Phase 17G: 125M progressive 256 -> 512 probe.

Purpose:
  Test the user's updated hypothesis under a less undertrained regime:
  125M model, 200M tokens at L=256, then 200M tokens at L=512.

Supported arms:
  - geo         : geometric RoPE
  - evq_dynamic : EVQ tau retargeted per stage
  - evq_frozen  : EVQ tau frozen at stage-0 tau (not run by default yet)

Defaults:
  - stage lens: 256 -> 512
  - stage tokens: 199,950,336 each (nearest <= 200M divisible by 98,304)
  - stage micro batch: 96 -> 48
  - grad accum: 4 -> 4
  - passkey mix: 5% single-passkey
  - final eval only
  - final PPL lengths: 512,1024,2048,4096,8192
"""

import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

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

ARMS = {"geo", "evq_dynamic", "evq_frozen"}
ARM = os.environ.get("PHASE17G_ARM", "geo").strip().lower()
if ARM not in ARMS:
    raise ValueError(f"PHASE17G_ARM must be one of {sorted(ARMS)}, got {ARM}")

STAGE_LENS = [
    int(x)
    for x in os.environ.get("PHASE17G_STAGE_LENS", "256,512").split(",")
    if x.strip()
]
STAGE_TOKENS = int(os.environ.get("PHASE17G_STAGE_TOKENS", "199950336"))
STAGE_MICRO_BS = [
    int(x)
    for x in os.environ.get("PHASE17G_STAGE_MICRO_BS", "96,48").split(",")
    if x.strip()
]
STAGE_GRAD_ACCUM = [
    int(x)
    for x in os.environ.get("PHASE17G_STAGE_GRAD_ACCUM", "4,4").split(",")
    if x.strip()
]
if len(STAGE_LENS) != len(STAGE_MICRO_BS) or len(STAGE_LENS) != len(STAGE_GRAD_ACCUM):
    raise ValueError("stage lens, micro batch, grad accum lengths must match")

SEED = int(os.environ.get("PHASE17G_SEED", "7"))
PASSKEY_MIX_RATIO = float(os.environ.get("PHASE17G_PASSKEY_MIX_RATIO", "0.05"))
LR = float(os.environ.get("PHASE17G_LR", "6e-4"))
MIN_LR = LR * 0.1
WARMUP_FRAC = float(os.environ.get("PHASE17G_WARMUP_FRAC", "0.02"))
USE_COMPILE = os.environ.get("PHASE17G_COMPILE", "1") == "1"
FORCE_REDO = os.environ.get("PHASE17G_FORCE_REDO", "0") == "1"
PER_STAGE_EVAL = os.environ.get("PHASE17G_PER_STAGE_EVAL", "0") == "1"

EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get("PHASE17G_EVAL_LENGTHS", "512,1024,2048,4096,8192").split(",")
    if x.strip()
]
EVAL_CHUNKS = int(os.environ.get("PHASE17G_EVAL_CHUNKS", "8"))

WORK = Path(
    os.environ.get(
        "PHASE17G_WORK",
        "/root/autodl-tmp/evq_phase17g_125m_256_to_512",
    )
)
TRAIN_DATA = Path(
    os.environ.get(
        "PHASE17G_TRAIN_DATA",
        "/root/autodl-tmp/evq_fresh_2048/train_2048_500M.pt",
    )
)
VAL_DATA = Path(
    os.environ.get(
        "PHASE17G_VAL_DATA",
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


def tau_for_arm(seq_len: int) -> float:
    if ARM == "geo":
        return 0.0
    if ARM == "evq_dynamic":
        return tau_star(seq_len)
    return tau_star(STAGE_LENS[0])


def stage_name(stage_idx: int, seq_len: int) -> str:
    return f"stage{stage_idx}_L{seq_len}"


def micro_bs_for_stage(stage_idx: int) -> int:
    return STAGE_MICRO_BS[stage_idx]


def grad_accum_for_stage(stage_idx: int) -> int:
    return STAGE_GRAD_ACCUM[stage_idx]


def eff_bs_for_stage(stage_idx: int) -> int:
    return micro_bs_for_stage(stage_idx) * grad_accum_for_stage(stage_idx)


def arm_dir() -> Path:
    return WORK / f"{ARM}_seed{SEED}"


def summary_path() -> Path:
    return arm_dir() / "summary.json"


def _autocast_ctx():
    if USE_AUTOCAST:
        return torch.autocast("cuda", dtype=DTYPE)
    return nullcontext()


def cfg_for_len(seq_len: int) -> Dict[str, int]:
    cfg = dict(CFG_125M)
    cfg["max_position_embeddings"] = seq_len
    cfg["seq_len"] = seq_len
    return cfg


def build_inv_freq(seq_len: int) -> torch.Tensor:
    if ARM == "geo":
        return geometric_inv_freq()
    return evq_cosh_inv_freq(head_dim=DIM, tau=tau_for_arm(seq_len), base=BASE)


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


def build_model(seq_len: int, state: Dict[str, torch.Tensor] | None = None, compile_model: bool = True):
    model = GPT(cfg_for_len(seq_len), build_inv_freq(seq_len)).to(DEVICE)
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


def build_stage_train_tensor(stage_idx: int, seq_len: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    cache_root = WORK / "data_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"stage{stage_idx}_L{seq_len}_tok{STAGE_TOKENS}_pk{int(PASSKEY_MIX_RATIO * 100)}_seed{SEED}.pt"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and not FORCE_REDO:
        data = torch.load(cache, map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return data, meta

    flat = load_train_flat()
    start = stage_idx * STAGE_TOKENS
    end = start + STAGE_TOKENS
    if end > flat.numel():
        raise ValueError(f"need tokens up to {end}, source only has {flat.numel()}")
    eff_bs = eff_bs_for_stage(stage_idx)
    if STAGE_TOKENS % seq_len != 0:
        raise ValueError(f"stage_tokens={STAGE_TOKENS} not divisible by seq_len={seq_len}")
    if (STAGE_TOKENS // seq_len) % eff_bs != 0:
        raise ValueError(f"stage rows for seq_len={seq_len} must be divisible by effective batch size {eff_bs}")

    data = flat[start:end].clone().view(-1, seq_len)
    meta = {
        "stage_idx": stage_idx,
        "seq_len": seq_len,
        "start_token": start,
        "end_token": end,
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
        stage_seed = SEED + stage_idx * 1000
        set_seed(stage_seed)
        indices = torch.randperm(n_rows)[:n_pk]
        for i, idx in enumerate(indices.tolist(), start=1):
            mixed[idx] = make_passkey_training_sample(
                filler,
                tok,
                seq_len=seq_len,
                seed=stage_seed + idx,
            ).to(torch.int32)
            if i % 2000 == 0 or i == n_pk:
                print(
                    f"    [mix] {stage_name(stage_idx, seq_len)} passkey rows {i}/{n_pk}",
                    flush=True,
                )
        data = mixed
        meta["passkey_rows"] = n_pk

    torch.save(data, cache)
    save_json(meta_path, meta)
    return data, meta


def eval_ppl_from_state(state: Dict[str, torch.Tensor], seq_len: int) -> Dict[str, float]:
    model = build_model(seq_len, state=state, compile_model=False)
    val = load_val_flat()
    ppl = eval_model(model, val, EVAL_LENGTHS, eval_chunks=EVAL_CHUNKS)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ppl


def train_stage(stage_idx: int, seq_len: int, input_state: Dict[str, torch.Tensor] | None):
    stage_dir = arm_dir() / stage_name(stage_idx, seq_len)
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "stage_result.json"
    ckpt_path = stage_dir / "model.pt"

    if result_path.exists() and ckpt_path.exists() and not FORCE_REDO:
        result = json.loads(result_path.read_text())
        state = load_clean_state(ckpt_path)
        print(f"[resume] {ARM} {stage_name(stage_idx, seq_len)}", flush=True)
        return state, result

    tau = tau_for_arm(seq_len)
    micro_bs = micro_bs_for_stage(stage_idx)
    grad_accum = grad_accum_for_stage(stage_idx)
    eff_bs = eff_bs_for_stage(stage_idx)
    tokens_per_step = eff_bs * seq_len
    print(
        f"\n=== {ARM} {stage_name(stage_idx, seq_len)} tau={tau:.6f} "
        f"tokens={STAGE_TOKENS/1e6:.3f}M micro={micro_bs} ga={grad_accum} ===",
        flush=True,
    )

    train_data, data_meta = build_stage_train_tensor(stage_idx, seq_len)
    model = build_model(seq_len, state=input_state, compile_model=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    total_rows = train_data.shape[0]
    total_steps = total_rows // eff_bs
    warmup = max(1, int(total_steps * WARMUP_FRAC))
    log_every = max(1, total_steps // 20)
    rows_gpu = train_data.to(DEVICE, non_blocking=True)
    del train_data
    torch.cuda.empty_cache()

    stage_seed = SEED + stage_idx * 1000
    set_seed(stage_seed)
    perm = torch.randperm(total_rows, device=DEVICE)
    ptr = 0
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    last_loss = None

    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            idx = perm[ptr : ptr + micro_bs]
            ptr += micro_bs
            batch = rows_gpu[idx].long()
            with _autocast_ctx():
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            (loss / grad_accum).backward()
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

        last_loss = step_loss / grad_accum
        running_loss += last_loss
        running_count += 1

        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, running_count)
            eta_h = (elapsed / step) * (total_steps - step) / 3600 if step < total_steps else 0.0
            print(
                f"  step {step}/{total_steps} loss={avg_loss:.4f} lr={lr:.2e} "
                f"L={seq_len} ETA={eta_h:.2f}h",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

    clean_state = clean_state_dict(model.state_dict())
    torch.save(
        {
            "model": clean_state,
            "arm": ARM,
            "stage_idx": stage_idx,
            "seq_len": seq_len,
            "tau": tau,
        },
        ckpt_path,
    )
    del model, optimizer, rows_gpu, perm
    gc.collect()
    torch.cuda.empty_cache()

    ppl = None
    if PER_STAGE_EVAL or stage_idx == len(STAGE_LENS) - 1:
        ppl = eval_ppl_from_state(clean_state, seq_len)

    result = {
        "arm": ARM,
        "seed": SEED,
        "stage_idx": stage_idx,
        "stage_name": stage_name(stage_idx, seq_len),
        "seq_len": seq_len,
        "tau": tau,
        "usable_tokens": STAGE_TOKENS,
        "micro_batch_size": micro_bs,
        "grad_accum": grad_accum,
        "effective_batch_size": eff_bs,
        "tokens_per_step": tokens_per_step,
        "steps": total_steps,
        "final_loss": round(float(last_loss), 6) if last_loss is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
        "ppl": ppl,
        "data": data_meta,
    }
    save_json(result_path, result)
    return clean_state, result


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
    for arm in ("evq_dynamic", "evq_frozen"):
        if arm not in rows:
            continue
        delta_vs_geo[arm] = {
            k: round((rows[arm][k] - geo[k]) / geo[k] * 100.0, 2)
            for k in geo
            if k in rows[arm] and geo[k] > 0
        }

    save_json(
        WORK / f"compare_seed{SEED}.json",
        {
            "seed": SEED,
            "stage_lens": STAGE_LENS,
            "stage_tokens": STAGE_TOKENS,
            "final_ppl": rows,
            "delta_vs_geo_pct": delta_vs_geo,
        },
    )


def run_arm():
    WORK.mkdir(parents=True, exist_ok=True)
    arm_dir().mkdir(parents=True, exist_ok=True)
    print("=" * 72, flush=True)
    print("Phase17G 125M progressive 256->512 probe", flush=True)
    print(f"  arm: {ARM}", flush=True)
    print(f"  seed: {SEED}", flush=True)
    print(f"  stage_lens: {STAGE_LENS}", flush=True)
    print(f"  stage_tokens: {STAGE_TOKENS}", flush=True)
    print(f"  stage_micro_bs: {STAGE_MICRO_BS}", flush=True)
    print(f"  stage_grad_accum: {STAGE_GRAD_ACCUM}", flush=True)
    print(f"  passkey_mix_ratio: {PASSKEY_MIX_RATIO:.2%}", flush=True)
    print(f"  per_stage_eval: {PER_STAGE_EVAL}", flush=True)
    print(f"  eval_lengths: {EVAL_LENGTHS}", flush=True)
    print("=" * 72, flush=True)

    state = None
    stage_results = []
    for stage_idx, seq_len in enumerate(STAGE_LENS):
        state, result = train_stage(stage_idx, seq_len, state)
        stage_results.append(result)

    final_ppl = stage_results[-1].get("ppl") or {}
    summary = {
        "arm": ARM,
        "seed": SEED,
        "train_data": str(TRAIN_DATA),
        "val_data": str(VAL_DATA),
        "stage_lens": STAGE_LENS,
        "stage_tokens": STAGE_TOKENS,
        "nominal_stage_tokens": 200_000_000,
        "stage_micro_bs": STAGE_MICRO_BS,
        "stage_grad_accum": STAGE_GRAD_ACCUM,
        "passkey_mix_ratio": PASSKEY_MIX_RATIO,
        "initial_tau": tau_for_arm(STAGE_LENS[0]),
        "final_tau": tau_for_arm(STAGE_LENS[-1]),
        "per_stage_eval": PER_STAGE_EVAL,
        "eval_lengths": EVAL_LENGTHS,
        "stage_results": stage_results,
        "final_ppl": final_ppl,
    }
    save_json(summary_path(), summary)
    maybe_write_compare_summary()
    print(f"\nDONE: {summary_path()}", flush=True)


if __name__ == "__main__":
    run_arm()
