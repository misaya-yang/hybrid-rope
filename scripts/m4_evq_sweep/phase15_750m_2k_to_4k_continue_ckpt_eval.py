#!/usr/bin/env python3
"""
Phase 15: 750M context continuation 2K -> 4K (Geo vs Hybrid).

Goal:
  Continue from existing 750M@2K checkpoints and evaluate whether 4K continuation
  improves long-context capability.

Runs:
  1) Geo 750M 2K ckpt -> continue at 4K
  2) Hybrid (tau=1.5, r=16) 750M 2K ckpt -> continue at 4K

Checkpoint eval:
  - 50%, 75%, 100%: PPL + passkey
  - 75%, 100%: RULER-style multi-needle

Example:
  python scripts/m4_evq_sweep/phase15_750m_2k_to_4k_continue_ckpt_eval.py

Env overrides:
  PHASE15_TOKENS=100000000
  PHASE15_PASSKEY_MIX_RATIO=0.03
  PHASE15_SEED=42
"""

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (  # noqa: E402
    GPT,
    DEVICE,
    DTYPE,
    USE_AUTOCAST,
    eval_model,
    get_batch_from_data,
    load_data,
    load_val,
    resolve_passkey_mix_ratio,
    set_seed,
)
from eval_multi_needle import eval_multi_needle_passkey  # noqa: E402
from eval_passkey_scratch import (  # noqa: E402
    eval_passkey_nll_gap,
    make_passkey_training_sample,
)
from eval_longbench_nll import (  # noqa: E402
    TASK_SETS as LB_TASK_SETS,
    NLL_FRIENDLY_TASKS as LB_TASKS,
    load_longbench_task as lb_load_longbench_task,
    format_prompt as lb_format_prompt,
    truncate_prompt_ids as lb_truncate_prompt_ids,
    compute_answer_nll as lb_compute_answer_nll,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = 500_000.0
DIM = 64
TAU = 1.5
R_KEEP = 16

SEED = int(os.environ.get("PHASE15_SEED", "42"))
TOKENS = int(os.environ.get("PHASE15_TOKENS", "100000000"))  # +100M by default
SEQ_LEN = 4096
PASSKEY_MIX_RATIO = float(
    os.environ.get(
        "PHASE15_PASSKEY_MIX_RATIO",
        str(resolve_passkey_mix_ratio(default=0.03)),
    )
)
_DOWNSTREAM_MIX_RAW = os.environ.get("PHASE15_DOWNSTREAM_MIX_RATIO", "0.03").strip()
DOWNSTREAM_MIX_RATIO = (
    float(_DOWNSTREAM_MIX_RAW[:-1]) / 100.0
    if _DOWNSTREAM_MIX_RAW.endswith("%")
    else float(_DOWNSTREAM_MIX_RAW)
)
DOWNSTREAM_TASKS = os.environ.get("PHASE15_DOWNSTREAM_TASKS", "qa4")
DOWNSTREAM_MAX_SAMPLES_PER_TASK = int(
    os.environ.get("PHASE15_DOWNSTREAM_MAX_SAMPLES_PER_TASK", "2000")
)
DOWNSTREAM_DATA_DIR = os.environ.get("PHASE15_DOWNSTREAM_DATA_DIR", "")
DOWNSTREAM_EVAL_SAMPLES = int(os.environ.get("PHASE15_DOWNSTREAM_EVAL_SAMPLES", "40"))
DOWNSTREAM_EVAL_ENABLED = os.environ.get("PHASE15_DOWNSTREAM_EVAL", "1") != "0"

if PASSKEY_MIX_RATIO + DOWNSTREAM_MIX_RATIO > 1.0:
    raise ValueError(
        "PHASE15_PASSKEY_MIX_RATIO + PHASE15_DOWNSTREAM_MIX_RATIO must be <= 1.0"
    )

EVAL_LENGTHS = [2048, 4096, 8192, 16384]
EVAL_CHUNKS = 4
PK_LENGTHS = [2048, 4096, 8192]
PK_TRIALS = 40

MN_LENGTHS = [4096, 8192]
MN_NEEDLES = 5
MN_TRIALS = 12

CKPT_FRACTIONS = [0.50, 0.75, 1.00]
CKPT_EVAL_CHUNKS = 4
CKPT_PK_TRIALS = 20
CKPT_MN_FRACTIONS = [0.75, 1.00]
CKPT_MN_TRIALS = 8

WORK = Path("/root/autodl-tmp/evq_phase15_750m_2k_to_4k")
DATA_CACHE_DIR = WORK / "data"

CFG_750M_4K = dict(
    vocab_size=50304,
    hidden_size=1536,
    num_layers=18,
    num_heads=24,
    head_dim=64,
    intermediate_size=6144,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=2e-4,
    batch_size=16,       # effective batch (micro * accum)
    micro_batch_size=8,  # tuned for 4K on 96GB
    grad_accum=2,
)

GEO_INIT_CKPT = (
    "/root/autodl-tmp/evq_phase9/seed42/"
    "geo_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt"
)
HYBRID_INIT_CKPT = (
    "/root/autodl-tmp/evq_phase9/seed42/"
    "hybrid1.5_r16_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt"
)


def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=TAU, r=R_KEEP):
    n = dim // 2
    geo = torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float64)
    n_evq = n - r
    if n_evq <= 0:
        return geo.float()
    theta_max = geo[r].item()
    theta_min = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / max(n_evq - 1, 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min ** phi) * (theta_max ** (1.0 - phi))
    return torch.cat([geo[:r], evq_part]).float()


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_state(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _resolve_downstream_tasks(spec: str):
    text = (spec or "").strip()
    if not text:
        return []
    if text in LB_TASK_SETS:
        return list(LB_TASK_SETS[text])
    return [t.strip() for t in text.split(",") if t.strip() in LB_TASKS]


def _pad_to_len(ids, seq_len, filler_tokens):
    if len(ids) >= seq_len:
        return ids[:seq_len]
    need = seq_len - len(ids)
    filler = filler_tokens.tolist()
    while len(filler) < need:
        filler.extend(filler_tokens.tolist())
    return ids + filler[:need]


def _build_downstream_bank(tokenizer, filler_tokens, seq_len, tasks):
    if not tasks:
        return None
    tag = "-".join(tasks)
    cache_path = DATA_CACHE_DIR / (
        f"downstream_bank_{tag}_{DOWNSTREAM_MAX_SAMPLES_PER_TASK}_{seq_len}.pt"
    )
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists():
        print(f"  [downstream-train] loading cached bank: {cache_path}")
        bank = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [downstream-train] cached samples={bank.shape[0]}")
        return bank

    rows = []
    task_stats = {}
    for task in tasks:
        try:
            samples = lb_load_longbench_task(
                task_name=task,
                max_samples=DOWNSTREAM_MAX_SAMPLES_PER_TASK,
                seed=SEED,
                data_dir=DOWNSTREAM_DATA_DIR,
            )
        except Exception as e:
            print(f"  [downstream-train] WARN load {task} failed: {e}")
            continue

        kept = 0
        for sample in samples:
            prompt_str, answer_str = lb_format_prompt(sample, task)
            if not prompt_str or not answer_str.strip():
                continue
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
            answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
            if not answer_ids:
                continue
            prompt_ids = lb_truncate_prompt_ids(
                tokenizer,
                prompt_ids,
                answer_ids,
                seq_len,
                strategy="middle",
            )
            ids = _pad_to_len(prompt_ids + answer_ids, seq_len, filler_tokens)
            rows.append(torch.tensor(ids, dtype=torch.long))
            kept += 1

        task_stats[task] = kept
        print(f"  [downstream-train] {task}: kept {kept}/{len(samples)}")

    if not rows:
        print("  [downstream-train] WARN: no valid downstream samples built")
        return None

    bank = torch.stack(rows, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, cache_path)
    save_json(
        meta_path,
        {
            "tasks": tasks,
            "seq_len": seq_len,
            "samples_per_task_limit": DOWNSTREAM_MAX_SAMPLES_PER_TASK,
            "built_samples": int(bank.shape[0]),
            "task_stats": task_stats,
        },
    )
    print(f"  [downstream-train] built {bank.shape[0]} samples, cached to {cache_path}")
    return bank


class MixedTrainingDataset(torch.utils.data.Dataset):
    """Three-way mixture: LM / passkey / downstream task."""

    def __init__(
        self,
        lm_data: torch.Tensor,
        filler_tokens: torch.Tensor,
        tokenizer,
        seq_len: int,
        passkey_ratio: float,
        downstream_ratio: float,
        downstream_bank: Optional[torch.Tensor],
    ):
        super().__init__()
        self.lm_data = lm_data
        self.filler_tokens = filler_tokens
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.passkey_ratio = max(0.0, float(passkey_ratio))
        self.downstream_ratio = max(0.0, float(downstream_ratio))
        self.downstream_bank = downstream_bank

    def __len__(self):
        return len(self.lm_data)

    def _pick_downstream(self, idx: int):
        if self.downstream_bank is None or len(self.downstream_bank) == 0:
            return self.lm_data[idx]
        j = (idx * 11400714819323198485 + 1) % len(self.downstream_bank)
        return self.downstream_bank[j]

    def __getitem__(self, idx: int):
        rng = random.Random(idx * 6364136223846793005 + 1)
        r = rng.random()
        if r < self.passkey_ratio:
            return make_passkey_training_sample(
                self.filler_tokens,
                self.tokenizer,
                seq_len=self.seq_len,
                seed=idx,
            )
        if r < self.passkey_ratio + self.downstream_ratio:
            return self._pick_downstream(idx)
        return self.lm_data[idx]


def maybe_wrap_with_training_mix(
    train_data: torch.Tensor,
    filler_tokens: torch.Tensor,
    tokenizer,
    seq_len: int,
    passkey_ratio: float,
    downstream_ratio: float,
    downstream_bank: Optional[torch.Tensor],
):
    if passkey_ratio <= 0 and downstream_ratio <= 0:
        print("  [train-mix] disabled (pure LM)")
        return train_data

    ds = MixedTrainingDataset(
        lm_data=train_data,
        filler_tokens=filler_tokens,
        tokenizer=tokenizer,
        seq_len=seq_len,
        passkey_ratio=passkey_ratio,
        downstream_ratio=downstream_ratio,
        downstream_bank=downstream_bank,
    )

    n = min(4000, len(ds))
    pk_n, ds_n = 0, 0
    for i in range(n):
        rr = random.Random(i * 6364136223846793005 + 1).random()
        if rr < passkey_ratio:
            pk_n += 1
        elif rr < passkey_ratio + downstream_ratio:
            ds_n += 1
    lm_n = n - pk_n - ds_n
    print(
        f"  [train-mix] target passkey={passkey_ratio:.2%}, downstream={downstream_ratio:.2%}; "
        f"sample_check passkey={pk_n/n:.2%}, downstream={ds_n/n:.2%}, lm={lm_n/n:.2%}"
    )
    return ds


@torch.no_grad()
def eval_downstream_nll_quick(model, tokenizer, max_context_len, tasks):
    device = next(model.parameters()).device
    out = {"tasks": {}, "_aggregate": {}}

    for task in tasks:
        try:
            samples = lb_load_longbench_task(
                task_name=task,
                max_samples=DOWNSTREAM_EVAL_SAMPLES,
                seed=SEED,
                data_dir=DOWNSTREAM_DATA_DIR,
            )
        except Exception as e:
            out["tasks"][task] = {"error": str(e)}
            continue

        nlls = []
        for sample in samples:
            prompt_str, answer_str = lb_format_prompt(sample, task)
            if not prompt_str or not answer_str.strip():
                continue
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
            answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
            if not answer_ids:
                continue
            prompt_ids = lb_truncate_prompt_ids(
                tokenizer,
                prompt_ids,
                answer_ids,
                max_context_len,
                strategy="middle",
            )
            nll, _ = lb_compute_answer_nll(
                model,
                prompt_ids,
                answer_ids,
                device=device,
                is_custom_gpt=True,
            )
            if not math.isnan(nll) and not math.isinf(nll):
                nlls.append(nll)

        if not nlls:
            out["tasks"][task] = {"n_samples": 0}
            continue

        mn = float(np.mean(nlls))
        out["tasks"][task] = {
            "mean_nll": mn,
            "ppl_from_nll": float(math.exp(mn)),
            "n_samples": len(nlls),
        }

    valid = [
        v["mean_nll"]
        for v in out["tasks"].values()
        if isinstance(v, dict) and "mean_nll" in v
    ]
    if valid:
        agg = float(np.mean(valid))
        out["_aggregate"] = {
            "mean_nll": agg,
            "ppl_from_nll": float(math.exp(agg)),
            "n_tasks": len(valid),
        }
    return out


def train_model_ga(model, data, cfg, seed=42, on_step_end=None):
    model.train()
    lr = cfg["lr"]
    min_lr = lr * 0.1
    micro_bs = cfg.get("micro_batch_size", cfg["batch_size"])
    grad_accum = cfg.get("grad_accum", 1)
    effective_bs = micro_bs * grad_accum

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    total_chunks = len(data)
    steps = total_chunks // effective_bs
    warmup = int(steps * 0.02)

    print(
        f"  Train cfg: micro_bs={micro_bs}, grad_accum={grad_accum}, "
        f"effective_bs={effective_bs}, chunks={total_chunks}, steps={steps}"
    )

    set_seed(seed)
    perm = torch.randperm(total_chunks)
    t0 = time.time()

    for s in range(steps):
        if s < warmup:
            cur_lr = lr * s / max(warmup, 1)
        else:
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))
            )
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for a in range(grad_accum):
            chunk_idx = s * effective_bs + a * micro_bs
            batch_cpu = get_batch_from_data(
                data, perm[chunk_idx : chunk_idx + micro_bs]
            )
            if batch_cpu.dtype != torch.long:
                batch_cpu = batch_cpu.to(dtype=torch.long)
            batch = batch_cpu.to(DEVICE)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous().to(dtype=torch.long)
            if s == 0 and a == 0:
                print(
                    f"    [dtype-check] batch_cpu={batch_cpu.dtype} "
                    f"batch={batch.dtype} targets={targets.dtype} device={targets.device}"
                )

            ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
            with ctx:
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step_num = s + 1
        if s % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1) if s > 0 else 0
            print(
                f"    step {s}/{steps}  loss={accum_loss:.4f}  lr={cur_lr:.2e}  ETA={eta/60:.0f}min"
            )

        if on_step_end is not None:
            on_step_end(step_num, steps)

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed/60:.1f} min")
    return model, elapsed


def run_single_continue(
    tag: str,
    init_ckpt_path: str,
    target_inv: torch.Tensor,
    cfg: dict,
    train_data,
    val_data,
    filler,
    tok,
    downstream_tasks,
    downstream_mix_ratio,
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
        raise FileNotFoundError(f"Missing init checkpoint: {init_ckpt_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  {tag}  |  init={init_ckpt_path}")
    print(
        f"  Continue @4K, tokens={TOKENS/1e6:.0f}M, seed={SEED}, "
        f"passkey_mix={PASSKEY_MIX_RATIO:.2%}, downstream_mix={downstream_mix_ratio:.2%}"
    )
    print(f"{'='*72}")

    set_seed(SEED)
    model = GPT(cfg, geometric_inv_freq()).to(DEVICE)
    state = _load_state(init_ckpt_path)
    model.load_state_dict(state, strict=False)

    # Overwrite RoPE frequencies to target method and rebuild caches at 4K.
    model.blocks[0].attn.rope.inv_freq.copy_(target_inv.to(model.blocks[0].attn.rope.inv_freq.device))
    model.blocks[0].attn.rope._build(cfg["max_position_embeddings"])
    print(
        f"  inv_freq set: max={target_inv.max().item():.8f} min={target_inv.min().item():.8f}"
    )

    total_steps = len(train_data) // (
        cfg.get("micro_batch_size", cfg["batch_size"]) * cfg.get("grad_accum", 1)
    )
    checkpoint_steps = sorted(
        set(max(1, min(total_steps, int(total_steps * f))) for f in CKPT_FRACTIONS)
    )
    ruler_checkpoint_steps = sorted(
        set(max(1, min(total_steps, int(total_steps * f))) for f in CKPT_MN_FRACTIONS)
    )
    print(f"  checkpoint_steps={checkpoint_steps}  ruler_steps={ruler_checkpoint_steps}")

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
            ppl_ckpt = eval_model(model, val_data, EVAL_LENGTHS, CKPT_EVAL_CHUNKS)
            pk_ckpt = eval_passkey_nll_gap(
                model,
                tok,
                filler,
                lengths=PK_LENGTHS,
                depths=[0.5],
                num_trials=CKPT_PK_TRIALS,
            )
            mn_ckpt = None
            if step_num in ruler_checkpoint_steps:
                print(f"    [CKPT-RULER] step={step_num} ({frac:.1%})")
                mn_ckpt = eval_multi_needle_passkey(
                    model,
                    tok,
                    filler,
                    lengths=MN_LENGTHS,
                    n_needles=MN_NEEDLES,
                    num_trials=CKPT_MN_TRIALS,
                    seed=SEED,
                )

        g_ckpt = pk_ckpt.get("global", {})
        rec = dict(
            step=step_num,
            fraction=round(frac, 4),
            checkpoint=str(ckpt_path),
            ppl=ppl_ckpt,
            ppl_16k=ppl_ckpt.get("16384", ppl_ckpt.get(16384, None)),
            passkey_global=g_ckpt,
            passkey_trials=CKPT_PK_TRIALS,
            eval_chunks=CKPT_EVAL_CHUNKS,
        )
        if mn_ckpt is not None:
            rec["multi_needle_global"] = mn_ckpt.get("global", {})
            rec["multi_needle_by_length"] = mn_ckpt.get("by_length", {})
            rec["multi_needle_trials"] = CKPT_MN_TRIALS

        ckpt_records.append(rec)
        save_json(ckpt_progress_file, ckpt_records)

        model.train()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    model, train_elapsed = train_model_ga(
        model, train_data, cfg, seed=SEED, on_step_end=on_step_end
    )

    ppl = eval_model(model, val_data, EVAL_LENGTHS, EVAL_CHUNKS)
    ppl_16k = ppl.get("16384", ppl.get(16384, None))

    print(f"  Final passkey eval ({PK_TRIALS} trials)")
    pk = eval_passkey_nll_gap(
        model,
        tok,
        filler,
        lengths=PK_LENGTHS,
        depths=[0.5],
        num_trials=PK_TRIALS,
    )
    g = pk.get("global", {})

    print(f"  Final RULER eval ({MN_NEEDLES} needles, {MN_TRIALS} trials)")
    mn = eval_multi_needle_passkey(
        model,
        tok,
        filler,
        lengths=MN_LENGTHS,
        n_needles=MN_NEEDLES,
        num_trials=MN_TRIALS,
        seed=SEED,
    )

    downstream_eval = None
    if DOWNSTREAM_EVAL_ENABLED and downstream_tasks:
        print(
            f"  Final downstream NLL eval ({','.join(downstream_tasks)}, "
            f"{DOWNSTREAM_EVAL_SAMPLES} samples/task)"
        )
        downstream_eval = eval_downstream_nll_quick(
            model=model,
            tokenizer=tok,
            max_context_len=SEQ_LEN,
            tasks=downstream_tasks,
        )

    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "inv_freq.npy", target_inv.cpu().numpy())

    result = dict(
        method=tag,
        base=BASE,
        seed=SEED,
        init_ckpt=init_ckpt_path,
        continue_tokens=TOKENS,
        model="750M",
        seq_len=SEQ_LEN,
        retrieval=g.get("retrieval_rate", 0),
        mean_nll_gap=g.get("mean_nll_gap", 0),
        ppl=ppl,
        ppl_16k=ppl_16k,
        passkey_global=pk.get("global", {}),
        passkey_summary=pk.get("summary", {}),
        multi_needle_global=mn.get("global", {}),
        multi_needle_by_length=mn.get("by_length", {}),
        downstream_nll=downstream_eval,
        checkpoints=ckpt_records,
        checkpoint_fractions=CKPT_FRACTIONS,
        train_sec=round(train_elapsed, 1),
        config=dict(
            passkey_mix_ratio=PASSKEY_MIX_RATIO,
            downstream_mix_ratio=downstream_mix_ratio,
            downstream_tasks=downstream_tasks,
            downstream_eval_samples=DOWNSTREAM_EVAL_SAMPLES,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            lr=cfg["lr"],
            effective_bs=cfg["batch_size"],
            micro_bs=cfg.get("micro_batch_size"),
            grad_accum=cfg.get("grad_accum"),
        ),
    )
    save_json(result_file, result)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    print("#" * 72)
    print("  Phase 15: 750M continuation 2K -> 4K (Geo vs Hybrid)")
    print("#" * 72)
    downstream_tasks = _resolve_downstream_tasks(DOWNSTREAM_TASKS)
    print(f"  device={DEVICE} dtype={DTYPE} autocast={USE_AUTOCAST}")
    print(
        f"  tokens={TOKENS/1e6:.0f}M seq_len={SEQ_LEN} "
        f"passkey_mix={PASSKEY_MIX_RATIO:.2%} downstream_mix={DOWNSTREAM_MIX_RATIO:.2%}"
    )
    print(f"  downstream tasks={downstream_tasks or '[]'}")

    cfg = CFG_750M_4K.copy()
    n_chunks = TOKENS // SEQ_LEN
    n_steps = n_chunks // cfg["batch_size"]
    print(
        f"  Config: micro_bs={cfg['micro_batch_size']}, accum={cfg['grad_accum']}, "
        f"effective_bs={cfg['batch_size']}, chunks={n_chunks}, steps={n_steps}"
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("  Loading continue-train data...")
    train_data = load_data(tok, TOKENS, SEQ_LEN, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    print("  Loading validation data...")
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(DATA_CACHE_DIR))
    filler = val_data[:50000]

    downstream_bank = _build_downstream_bank(
        tokenizer=tok,
        filler_tokens=filler,
        seq_len=SEQ_LEN,
        tasks=downstream_tasks,
    )
    effective_downstream_ratio = DOWNSTREAM_MIX_RATIO
    if DOWNSTREAM_MIX_RATIO > 0 and (downstream_bank is None or len(downstream_bank) == 0):
        print("  [downstream-train] disable downstream mix (no usable samples)")
        effective_downstream_ratio = 0.0

    train_data = maybe_wrap_with_training_mix(
        train_data=train_data,
        filler_tokens=filler,
        tokenizer=tok,
        seq_len=SEQ_LEN,
        passkey_ratio=PASSKEY_MIX_RATIO,
        downstream_ratio=effective_downstream_ratio,
        downstream_bank=downstream_bank,
    )

    runs = [
        dict(
            tag="geo_750m_2k_to_4k_continue",
            init_ckpt=GEO_INIT_CKPT,
            target_inv=geometric_inv_freq(),
        ),
        dict(
            tag="hybrid1.5_r16_750m_2k_to_4k_continue",
            init_ckpt=HYBRID_INIT_CKPT,
            target_inv=hybrid_evq_inv_freq(DIM, BASE, TAU, R_KEEP),
        ),
    ]

    results = {}
    for r in runs:
        results[r["tag"]] = run_single_continue(
            tag=r["tag"],
            init_ckpt_path=r["init_ckpt"],
            target_inv=r["target_inv"],
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            filler=filler,
            tok=tok,
            downstream_tasks=downstream_tasks,
            downstream_mix_ratio=effective_downstream_ratio,
        )

    geo = results["geo_750m_2k_to_4k_continue"]
    hyb = results["hybrid1.5_r16_750m_2k_to_4k_continue"]

    geo_ret = geo.get("passkey_global", {}).get("retrieval_rate", 0)
    hyb_ret = hyb.get("passkey_global", {}).get("retrieval_rate", 0)
    geo_ppl = geo.get("ppl_16k") or geo.get("ppl", {}).get("16384", 0)
    hyb_ppl = hyb.get("ppl_16k") or hyb.get("ppl", {}).get("16384", 0)

    print(f"\n{'='*72}")
    print("  SUMMARY (750M 2K->4K continuation)")
    print(f"{'='*72}")
    print(f"  Geo    : ret={geo_ret:.4f}, ppl16k={geo_ppl}")
    print(f"  Hybrid : ret={hyb_ret:.4f}, ppl16k={hyb_ppl}")
    geo_ds = (geo.get("downstream_nll") or {}).get("_aggregate", {})
    hyb_ds = (hyb.get("downstream_nll") or {}).get("_aggregate", {})
    if geo_ds:
        print(
            f"  Geo downstream: NLL={geo_ds.get('mean_nll', float('nan')):.4f} "
            f"(tasks={geo_ds.get('n_tasks', 0)})"
        )
    if hyb_ds:
        print(
            f"  Hybrid downstream: NLL={hyb_ds.get('mean_nll', float('nan')):.4f} "
            f"(tasks={hyb_ds.get('n_tasks', 0)})"
        )
    if geo_ret:
        print(f"  Hybrid vs Geo retrieval: {(hyb_ret / geo_ret - 1) * 100:+.2f}%")
    if geo_ppl:
        print(f"  Hybrid vs Geo PPL@16K: {(hyb_ppl / geo_ppl - 1) * 100:+.2f}%")
    if geo_ds and hyb_ds and geo_ds.get("mean_nll", 0):
        print(
            f"  Hybrid vs Geo downstream NLL: "
            f"{(hyb_ds['mean_nll'] / geo_ds['mean_nll'] - 1) * 100:+.2f}%"
        )

    summary = dict(
        phase="15_750m_2k_to_4k_continue_ckpt_eval",
        model="750M",
        seq_len=SEQ_LEN,
        tokens=TOKENS,
        runs=["geo", "hybrid1.5_r16"],
        seed=SEED,
        passkey_mix_ratio=PASSKEY_MIX_RATIO,
        downstream_mix_ratio=effective_downstream_ratio,
        downstream_tasks=downstream_tasks,
        downstream_eval_enabled=DOWNSTREAM_EVAL_ENABLED,
        checkpoint_fractions=CKPT_FRACTIONS,
        ruler_checkpoint_fractions=CKPT_MN_FRACTIONS,
        results=results,
    )
    save_json(WORK / "phase15_750m_2k_to_4k_continue_summary.json", summary)
    print(f"\nSaved: {WORK}/phase15_750m_2k_to_4k_continue_summary.json")


if __name__ == "__main__":
    main()
