#!/usr/bin/env python3
"""
Phase 17B: 454M staged continuation 512 -> 1024 with fresh data.

Goal:
  Continue existing 454M Geo and EVQ checkpoints trained at L=512, switch the
  training length to L=1024, and evaluate whether staged length extension helps.

Key design choices:
  - Geo fork keeps geometric frequencies.
  - EVQ fork retargets to tau*=d_head/sqrt(L_new)=64/sqrt(1024)=2.0.
  - Continuation data comes from an explicit token offset after the base stage,
    so stage-2 training uses unseen tokens rather than a re-chunked prefix.
  - Checkpoint eval tracks intra-stage dynamics at 25/50/75/100%.

Environment overrides:
  PHASE17B_WORK=/root/autodl-tmp/evq_phase17b_1024_continue
  PHASE17B_DATASET=proof-pile-2
  PHASE17B_SOURCE_DATA=/root/autodl-tmp/evq_phase17/train_proof-pile-2_2000000000_512.pt
  PHASE17B_DATA_CACHE_DIR=/root/autodl-tmp/evq_phase17b_1024_continue/data
  PHASE17B_DATA_TAG=after1b_stage2
  PHASE17B_BASE_TOKENS=1000000000
  PHASE17B_DATA_START_TOKEN=1000000000
  PHASE17B_TOKENS=1000000000
  PHASE17B_SEQ_LEN=1024
  PHASE17B_SEED=42
  PHASE17B_TAU=2.0
  PHASE17B_PASSKEY_MIX_RATIO=0.05
  PHASE17B_GEO_INIT_CKPT=/path/to/geo_ckpt.pt
  PHASE17B_EVQ_INIT_CKPT=/path/to/evq_ckpt.pt
  PHASE17B_RUN_ONLY=geo|evq
  PHASE17B_MICRO_BATCH_SIZE=10
  PHASE17B_GRAD_ACCUM=2
  PHASE17B_EVAL_LENGTHS=1024,2048,4096,8192,16384,32768
  PHASE17B_CKPT_EVAL_LENGTHS=1024,4096,8192,16384
"""

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

from eval_passkey_scratch import eval_passkey_nll_gap, make_passkey_training_sample  # noqa: E402
from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    evq_cosh_inv_freq,
    load_val,
    resolve_passkey_mix_ratio,
    set_seed,
)


BASE = 500_000.0
DIM = 64
SEQ_LEN = int(os.environ.get("PHASE17B_SEQ_LEN", "1024"))
TAU = float(os.environ.get("PHASE17B_TAU", f"{DIM / math.sqrt(SEQ_LEN):.6f}"))
SEED = int(os.environ.get("PHASE17B_SEED", "42"))
TOKENS = int(os.environ.get("PHASE17B_TOKENS", "1000000000"))
BASE_TOKENS = int(os.environ.get("PHASE17B_BASE_TOKENS", "1000000000"))
DATA_START_TOKEN = int(os.environ.get("PHASE17B_DATA_START_TOKEN", str(BASE_TOKENS)))
RUN_ONLY = os.environ.get("PHASE17B_RUN_ONLY", "").strip().lower()
PASSKEY_MIX_RATIO = float(
    os.environ.get(
        "PHASE17B_PASSKEY_MIX_RATIO",
        str(resolve_passkey_mix_ratio(default=0.05)),
    )
)
DATASET = os.environ.get("PHASE17B_DATASET", "proof-pile-2").strip() or "proof-pile-2"
EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17B_EVAL_LENGTHS",
        "512,1024,2048,4096,8192,16384",
    ).split(",")
    if x.strip()
]
CKPT_EVAL_LENGTHS = [
    int(x)
    for x in os.environ.get(
        "PHASE17B_CKPT_EVAL_LENGTHS",
        "512,1024,2048,4096,8192",
    ).split(",")
    if x.strip()
]
CKPT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
CKPT_EVAL_CHUNKS = 10
EVAL_CHUNKS = 10
PK_LENGTHS = [512, 1024, 2048, 4096, 8192]
CKPT_PK_TRIALS = 10
PK_TRIALS = 20
GPU_CHUNKS = int(os.environ.get("PHASE17B_GPU_CHUNKS", "8"))
SAVE_MIXED_CACHE = os.environ.get("PHASE17B_SAVE_MIXED_CACHE", "0") == "1"

WORK = Path(
    os.environ.get(
        "PHASE17B_WORK",
        "/root/autodl-tmp/evq_phase17b_1024_continue",
    )
)
SOURCE_DATA = Path(
    os.environ.get(
        "PHASE17B_SOURCE_DATA",
        "/root/autodl-tmp/evq_phase17/train_proof-pile-2_2000000000_512.pt",
    )
)
DATA_CACHE_DIR = Path(os.environ.get("PHASE17B_DATA_CACHE_DIR", str(WORK / "data")))
DATA_TAG = os.environ.get("PHASE17B_DATA_TAG", "after1b_stage2").strip() or "after1b_stage2"
GEO_INIT_CKPT = os.environ.get("PHASE17B_GEO_INIT_CKPT", "").strip()
EVQ_INIT_CKPT = os.environ.get("PHASE17B_EVQ_INIT_CKPT", "").strip()

CFG_454M_1024 = dict(
    vocab_size=50304,
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    head_dim=64,
    intermediate_size=4096,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=2e-4,
    batch_size=20,
    micro_batch_size=10,
    grad_accum=2,
)


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def parse_eval_lengths(values):
    out = sorted(set(int(v) for v in values if int(v) > 0))
    if not out:
        raise ValueError("evaluation lengths must contain at least one positive value")
    return out


def load_fresh_train_data():
    cache_name = f"train_{DATASET}_{DATA_TAG}_{DATA_START_TOKEN}_{TOKENS}_{SEQ_LEN}.pt"
    cache_path = DATA_CACHE_DIR / cache_name
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists():
        print(f"  [data] loading continuation cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [data] cached chunks={data.shape[0]} tokens={data.numel()/1e6:.1f}M")
        return data

    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"source data not found: {SOURCE_DATA}")

    print(f"  [data] loading source flat tokens from: {SOURCE_DATA}")
    source = torch.load(SOURCE_DATA, map_location="cpu", weights_only=True)
    flat = source.reshape(-1)
    total_needed = DATA_START_TOKEN + TOKENS
    if total_needed > flat.numel():
        raise ValueError(
            f"need {total_needed} tokens but source only has {flat.numel()}"
        )

    usable = (TOKENS // SEQ_LEN) * SEQ_LEN
    if usable <= 0:
        raise ValueError(f"TOKENS={TOKENS} is too small for seq_len={SEQ_LEN}")
    start = DATA_START_TOKEN
    end = start + usable
    data = flat[start:end].clone().view(-1, SEQ_LEN)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    save_json(
        meta_path,
        {
            "data_tag": DATA_TAG,
            "dataset": DATASET,
            "source_data": str(SOURCE_DATA),
            "data_start_token": start,
            "base_tokens": BASE_TOKENS,
            "requested_tokens": TOKENS,
            "usable_tokens": usable,
            "seq_len": SEQ_LEN,
            "n_chunks": int(data.shape[0]),
        },
    )
    print(
        f"  [data] built continuation cache: {cache_path} "
        f"({data.shape[0]} chunks, {data.numel()/1e6:.1f}M tokens)"
    )
    return data


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
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


def load_model_with_target_inv(cfg, ckpt_path: str, target_inv: torch.Tensor):
    model = GPT(cfg, target_inv).to(DEVICE)
    state = _load_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}")
    model.blocks[0].attn.rope.inv_freq.copy_(
        target_inv.to(model.blocks[0].attn.rope.inv_freq.device)
    )
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    return model


def build_mixed_train_tensor(train_data: torch.Tensor, filler: torch.Tensor, tokenizer):
    if PASSKEY_MIX_RATIO <= 0:
        print("  [passkey-train] mix disabled (pure LM)")
        return train_data

    mix_pct = int(round(PASSKEY_MIX_RATIO * 100))
    cache_path = DATA_CACHE_DIR / (
        f"mixed_{DATASET}_{DATA_TAG}_seed{SEED}_pk{mix_pct}_{TOKENS}_{SEQ_LEN}.pt"
    )
    if SAVE_MIXED_CACHE and cache_path.exists():
        print(f"  [passkey-train] loading cached mixed tensor: {cache_path}")
        mixed = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [passkey-train] cached shape={tuple(mixed.shape)}")
        return mixed

    mixed = train_data.clone()
    n_total = mixed.shape[0]
    n_pk = int(n_total * PASSKEY_MIX_RATIO)
    if n_pk <= 0:
        return mixed

    print(f"  [passkey-train] pre-generating {n_pk} passkey samples...")
    pk_indices = torch.randperm(n_total)[:n_pk]
    t0 = time.time()
    for i, idx in enumerate(pk_indices.tolist(), start=1):
        mixed[idx] = make_passkey_training_sample(
            filler_tokens=filler,
            tokenizer=tokenizer,
            seq_len=SEQ_LEN,
            seed=idx,
        )
        if i % 50_000 == 0 or i == n_pk:
            print(f"    generated {i}/{n_pk} passkey samples")

    if SAVE_MIXED_CACHE:
        torch.save(mixed, cache_path)
        print(f"  [passkey-train] built in {time.time() - t0:.1f}s, cached to {cache_path}")
    else:
        print(f"  [passkey-train] built in {time.time() - t0:.1f}s (cache disabled)")
    return mixed


def train_model_ga(model, data, cfg, seed=42, on_step_end=None):
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg["micro_batch_size"]
    grad_accum = cfg["grad_accum"]
    effective_bs = micro_bs * grad_accum

    adamw_kwargs = dict(lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if DEVICE == "cuda":
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    if not isinstance(data, torch.Tensor):
        raise TypeError("train_model_ga expects a tensor dataset for GPU chunk preloading")

    total_seqs = data.shape[0]
    steps = total_seqs // effective_bs
    warmup = max(1, int(steps * 0.02))
    if steps <= 0:
        raise ValueError(
            f"not enough chunks for one step: chunks={total_seqs}, effective_bs={effective_bs}"
        )

    print(
        f"  Train cfg: micro_bs={micro_bs}, grad_accum={grad_accum}, "
        f"effective_bs={effective_bs}, chunks={total_seqs}, steps={steps}, "
        f"gpu_chunks={GPU_CHUNKS}"
    )

    set_seed(seed)
    perm = torch.randperm(total_seqs)
    data = data[perm].contiguous()
    del perm
    t0 = time.time()

    chunk_size = total_seqs // GPU_CHUNKS
    global_step = 0

    for ci in range(GPU_CHUNKS):
        c_start = ci * chunk_size
        c_end = c_start + chunk_size if ci < GPU_CHUNKS - 1 else total_seqs
        chunk_len = c_end - c_start
        chunk_steps = chunk_len // effective_bs
        if chunk_steps <= 0:
            continue

        gpu_chunk = data[c_start:c_end].to(torch.int64).to(DEVICE).contiguous()
        print(
            f"  [gpu-chunk {ci+1}/{GPU_CHUNKS}] loaded {tuple(gpu_chunk.shape)} "
            f"({gpu_chunk.element_size() * gpu_chunk.numel() / 1e9:.1f}GB)"
        )

        for cs in range(chunk_steps):
            s = global_step
            if s >= steps:
                break

            if s < warmup:
                cur_lr = lr * (s + 1) / warmup
            else:
                cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
                )
            for g in opt.param_groups:
                g["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            if DEVICE == "cuda" and hasattr(torch, "compiler"):
                torch.compiler.cudagraph_mark_step_begin()

            for a in range(grad_accum):
                idx_start = cs * effective_bs + a * micro_bs
                idx_end = idx_start + micro_bs
                batch = gpu_chunk[idx_start:idx_end]
                inputs = batch[:, :-1].contiguous()
                targets = batch[:, 1:].contiguous()

                ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
                with ctx:
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                    loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item()

            if s < warmup:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1

            step_num = s + 1
            if s % 50 == 0 or step_num == steps:
                elapsed = time.time() - t0
                eta = elapsed / step_num * (steps - step_num) if step_num > 0 else 0
                gpu_mem = (
                    torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
                )
                print(
                    f"    step {step_num}/{steps}  loss={accum_loss:.4f}  "
                    f"lr={cur_lr:.2e}  GPU={gpu_mem:.1f}GB  ETA={eta/60:.0f}min"
                )

            if on_step_end is not None:
                on_step_end(step_num, steps)

        del gpu_chunk
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model, elapsed


def run_single_continue(
    tag,
    init_ckpt_path,
    target_inv,
    cfg,
    train_data,
    val_data,
    filler,
    tok,
    eval_lengths,
    ckpt_eval_lengths,
):
    run_dir = WORK / f"seed{SEED}" / tag
    ckpt_dir = run_dir / "checkpoints"
    result_file = run_dir / "result.json"
    ckpt_progress_file = run_dir / "checkpoint_eval_progress.json"

    if result_file.exists():
        print(f"\n[SKIP] {tag}: already done")
        with open(result_file) as f:
            return json.load(f)

    if not Path(init_ckpt_path).exists():
        raise FileNotFoundError(f"missing init checkpoint: {init_ckpt_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  {tag}  |  init={init_ckpt_path}")
    print(
        f"  Continue @L={SEQ_LEN}, tokens={TOKENS/1e6:.0f}M, seed={SEED}, "
        f"tau={TAU:.4f}, data_tag={DATA_TAG}, data_start={DATA_START_TOKEN/1e6:.0f}M"
    )
    print(f"{'=' * 72}")

    set_seed(SEED)
    model = load_model_with_target_inv(cfg, init_ckpt_path, target_inv)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")
        print("  torch.compile enabled (max-autotune)")
    print(
        f"  inv_freq set: max={target_inv.max().item():.8f} "
        f"min={target_inv.min().item():.8f}"
    )

    total_steps = len(train_data) // (cfg["micro_batch_size"] * cfg["grad_accum"])
    checkpoint_steps = sorted(
        set(max(1, min(total_steps, int(total_steps * frac))) for frac in CKPT_FRACTIONS)
    )
    print(f"  checkpoint_steps={checkpoint_steps}")

    ckpt_records = []
    ckpt_done = set()

    def on_step_end(step_num, steps_total):
        if step_num not in checkpoint_steps or step_num in ckpt_done:
            return

        ckpt_done.add(step_num)
        frac = step_num / steps_total
        ckpt_name = f"step_{step_num:05d}"
        ckpt_path = ckpt_dir / f"{ckpt_name}.pt"

        print(f"\n  [CKPT] {tag} {ckpt_name} ({frac:.1%})")
        torch.save(model.state_dict(), ckpt_path)

        model.eval()
        with torch.no_grad():
            ppl_ckpt = eval_model(model, val_data, ckpt_eval_lengths, CKPT_EVAL_CHUNKS)
            pk_ckpt = eval_passkey_nll_gap(
                model,
                tok,
                filler,
                lengths=PK_LENGTHS,
                depths=[0.5],
                num_trials=CKPT_PK_TRIALS,
            )

        g_ckpt = pk_ckpt.get("global", {})
        rec = dict(
            step=step_num,
            fraction=round(frac, 4),
            checkpoint=str(ckpt_path),
            ppl=ppl_ckpt,
            passkey_global=g_ckpt,
            passkey_trials=CKPT_PK_TRIALS,
            eval_chunks=CKPT_EVAL_CHUNKS,
        )
        ckpt_records.append(rec)
        save_json(ckpt_progress_file, ckpt_records)

        model.train()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    model, train_elapsed = train_model_ga(
        model=model,
        data=train_data,
        cfg=cfg,
        seed=SEED,
        on_step_end=on_step_end,
    )

    ppl = eval_model(model, val_data, eval_lengths, EVAL_CHUNKS)
    pk = eval_passkey_nll_gap(
        model,
        tok,
        filler,
        lengths=PK_LENGTHS,
        depths=[0.5],
        num_trials=PK_TRIALS,
    )

    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", target_inv.cpu().numpy())

    result = dict(
        method=tag,
        base=BASE,
        seed=SEED,
        init_ckpt=init_ckpt_path,
        continue_tokens=TOKENS,
        base_tokens=BASE_TOKENS,
        data_start_token=DATA_START_TOKEN,
        data_tag=DATA_TAG,
        model="454M",
        seq_len=SEQ_LEN,
        tau=TAU if "evq" in tag else 0.0,
        ppl=ppl,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        checkpoints=ckpt_records,
        checkpoint_fractions=CKPT_FRACTIONS,
        train_sec=round(train_elapsed, 1),
        config=dict(
            passkey_mix_ratio=PASSKEY_MIX_RATIO,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            lr=cfg["lr"],
            effective_bs=cfg["batch_size"],
            micro_bs=cfg["micro_batch_size"],
            grad_accum=cfg["grad_accum"],
            eval_lengths=eval_lengths,
            checkpoint_eval_lengths=ckpt_eval_lengths,
        ),
    )
    save_json(result_file, result)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    print("#" * 72)
    print("  Phase 17B: 454M staged continuation 512 -> 1024")
    print("#" * 72)

    if not GEO_INIT_CKPT or not EVQ_INIT_CKPT:
        raise ValueError(
            "PHASE17B_GEO_INIT_CKPT and PHASE17B_EVQ_INIT_CKPT must both be set"
        )
    if DATA_START_TOKEN < BASE_TOKENS:
        raise ValueError(
            f"data_start_token={DATA_START_TOKEN} must be >= base_tokens={BASE_TOKENS}"
        )

    cfg = CFG_454M_1024.copy()
    if os.environ.get("PHASE17B_MICRO_BATCH_SIZE"):
        cfg["micro_batch_size"] = int(os.environ["PHASE17B_MICRO_BATCH_SIZE"])
    if os.environ.get("PHASE17B_GRAD_ACCUM"):
        cfg["grad_accum"] = int(os.environ["PHASE17B_GRAD_ACCUM"])
    cfg["batch_size"] = cfg["micro_batch_size"] * cfg["grad_accum"]

    eval_lengths = parse_eval_lengths(EVAL_LENGTHS)
    ckpt_eval_lengths = parse_eval_lengths(CKPT_EVAL_LENGTHS)
    print(f"  device={DEVICE} dtype={DTYPE} autocast={USE_AUTOCAST}")
    print(
        f"  seq_len={SEQ_LEN} tau*={TAU:.4f} tokens={TOKENS/1e6:.0f}M "
        f"base_tokens={BASE_TOKENS/1e6:.0f}M data_start={DATA_START_TOKEN/1e6:.0f}M"
    )
    print(
        f"  micro_bs={cfg['micro_batch_size']} grad_accum={cfg['grad_accum']} "
        f"effective_bs={cfg['batch_size']} passkey_mix={PASSKEY_MIX_RATIO:.2%}"
    )
    print(f"  eval_lengths={eval_lengths}")
    print(f"  ckpt_eval_lengths={ckpt_eval_lengths}")
    print(f"  dataset={DATASET} data_tag={DATA_TAG} source={SOURCE_DATA}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading fresh continuation data...")
    train_data_raw = load_fresh_train_data()
    print("  Loading validation data...")
    val_data = load_val(tok, 5_000_000, DATASET, cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]
    train_data = build_mixed_train_tensor(train_data_raw, filler, tok)

    runs = [
        dict(
            tag="geo_454m_512_to_1024_continue",
            init_ckpt=GEO_INIT_CKPT,
            target_inv=geometric_inv_freq(),
        ),
        dict(
            tag=f"evq{TAU:g}_454m_512_to_1024_continue",
            init_ckpt=EVQ_INIT_CKPT,
            target_inv=evq_cosh_inv_freq(head_dim=DIM, tau=TAU, base=BASE),
        ),
    ]
    if RUN_ONLY in {"geo", "evq"}:
        if RUN_ONLY == "geo":
            runs = [r for r in runs if r["tag"].startswith("geo_")]
        else:
            runs = [r for r in runs if r["tag"].startswith("evq")]
        print(f"  run filter={RUN_ONLY} -> {[r['tag'] for r in runs]}")
    if not runs:
        raise RuntimeError("no runs selected after PHASE17B_RUN_ONLY filter")

    results = {}
    for run in runs:
        results[run["tag"]] = run_single_continue(
            tag=run["tag"],
            init_ckpt_path=run["init_ckpt"],
            target_inv=run["target_inv"],
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            filler=filler,
            tok=tok,
            eval_lengths=eval_lengths,
            ckpt_eval_lengths=ckpt_eval_lengths,
        )

    summary = dict(
        phase="17b_454m_512_to_1024_continue_ckpt_eval",
        model="454M",
        seq_len=SEQ_LEN,
        tokens=TOKENS,
        base_tokens=BASE_TOKENS,
        data_start_token=DATA_START_TOKEN,
        data_tag=DATA_TAG,
        dataset=DATASET,
        seed=SEED,
        tau=TAU,
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        runs=list(results.keys()),
        results=results,
    )
    summary_path = WORK / "phase17b_454m_512_to_1024_continue_summary.json"
    save_json(summary_path, summary)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
