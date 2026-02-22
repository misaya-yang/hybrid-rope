#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import run_phase4 as p4
from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, env_info, get_device, save_json, set_seed
from src.visualization import save_fig_both, set_plot_style

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def ensure_dependencies() -> None:
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("seaborn", "seaborn"),
    ]
    missing: List[str] = []
    for mod, pip_name in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    for pip_name in missing:
        print(f"[deps] installing {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


# Reuse tokenizer and base data loader from phase-4 script.
resolve_tokenizer = p4.resolve_tokenizer


@dataclass
class TrainConfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    head_dim: int = 64
    max_seq_len: int = 8192
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 3000
    eval_interval: int = 50
    save_interval: int = 500
    early_stop_patience: int = 500
    min_train_steps: int = 500
    grad_clip: float = 1.0


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def split_train_val(tokens: np.ndarray, val_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    n = int(tokens.size)
    n_val = max(4096, int(n * val_ratio))
    n_val = min(n_val, n // 3)
    return tokens[:-n_val], tokens[-n_val:]


def sample_batch(token_ids: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    max_start = int(token_ids.size - seq_len - 1)
    starts = rng.integers(low=0, high=max_start, size=batch_size, endpoint=False)
    arr = np.stack([token_ids[s: s + seq_len + 1] for s in starts], axis=0)
    return torch.tensor(arr, dtype=torch.long, device=device)


def build_fixed_eval_batches(token_ids: np.ndarray, seq_len: int, num_batches: int, batch_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    max_start = int(token_ids.size - seq_len - 1)
    starts_all = rng.integers(low=0, high=max_start, size=num_batches * batch_size, endpoint=False)
    out: List[np.ndarray] = []
    idx = 0
    for _ in range(num_batches):
        starts = starts_all[idx: idx + batch_size]
        idx += batch_size
        out.append(np.stack([token_ids[s: s + seq_len + 1] for s in starts], axis=0))
    return out


def evaluate_val_loss(model: p4.GPTSmall, val_batches: List[np.ndarray], device: torch.device, amp_dtype: torch.dtype, use_amp: bool) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for arr in val_batches:
            x = torch.tensor(arr, dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(x)
            losses.append(float(loss.detach().item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(model: p4.GPTSmall, optimizer: torch.optim.Optimizer, step: int, ckpt_dir: Path, tag: str) -> Path:
    out_dir = ckpt_dir / f"{tag}_step{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": int(step)}, out_dir / "checkpoint.pt")
    return out_dir


def save_best_checkpoint(model: p4.GPTSmall, optimizer: torch.optim.Optimizer, step: int, ckpt_dir: Path, tag: str, val_loss: float) -> Path:
    out_dir = ckpt_dir / f"{tag}_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Windows can intermittently lock same-name overwrite under heavy IO/scan.
    # Use unique filenames for each best update to avoid overwrite races.
    out_path = out_dir / f"checkpoint_step{int(step):06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_val_loss": float(val_loss),
        },
        out_path,
    )
    # Keep directory small: retain only newest best file.
    for old in sorted(out_dir.glob("checkpoint_step*.pt")):
        if old == out_path:
            continue
        try:
            old.unlink()
        except Exception:
            pass
    return out_path


def save_final_model(model: p4.GPTSmall, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")


def load_model_state(model: p4.GPTSmall, ckpt_or_model_path: Path, device: torch.device) -> None:
    payload = torch.load(ckpt_or_model_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        model.load_state_dict(payload["model"], strict=True)
    elif isinstance(payload, dict):
        model.load_state_dict(payload, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_or_model_path}")


def autotune_micro_batch(model: p4.GPTSmall, token_ids: np.ndarray, seq_len: int, candidates: Sequence[int], device: torch.device, amp_dtype: torch.dtype, use_amp: bool, seed: int) -> int:
    rng = np.random.default_rng(seed)
    model.train()
    for bsz in candidates:
        try:
            x = sample_batch(token_ids, seq_len=seq_len, batch_size=int(bsz), rng=rng, device=device)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            cleanup_cuda()
            print(f"[autotune] micro_batch={bsz} OK")
            return int(bsz)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                print(f"[autotune] micro_batch={bsz} OOM")
                cleanup_cuda()
                continue
            raise
    return 1

def train_single_model(
    root_dir: Path,
    model: p4.GPTSmall,
    tag: str,
    init_state: Dict[str, torch.Tensor],
    train_ids: np.ndarray,
    val_batches: List[np.ndarray],
    cfg: TrainConfig,
    data_seed: int,
    micro_batch: int,
    grad_accum: int,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    model.load_state_dict(init_state, strict=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    logs: List[Dict[str, float]] = []
    rng = np.random.default_rng(data_seed)
    ckpt_root = root_dir / "checkpoints"
    out_csv = root_dir / "data" / f"training_log_{tag}.csv"
    best_val = float("inf")
    best_step = 0
    best_path: Optional[Path] = None

    t0 = time.time()
    pbar = tqdm(range(1, cfg.max_steps + 1), desc=f"train-{tag}", dynamic_ncols=True)
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        lr = cosine_lr(step=step - 1, max_steps=cfg.max_steps, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        step_loss = 0.0
        for _ in range(grad_accum):
            x = sample_batch(train_ids, seq_len=cfg.max_seq_len, batch_size=micro_batch, rng=rng, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(x)
            step_loss += float(loss.detach().item())
            loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        train_loss = step_loss / grad_accum
        elapsed = time.time() - t0
        eta_h = (cfg.max_steps - step) * (elapsed / max(1, step)) / 3600.0

        if step % cfg.eval_interval == 0 or step == 1:
            val_loss = evaluate_val_loss(model, val_batches, device, amp_dtype, use_amp)
            logs.append({"step": int(step), "train_loss": float(train_loss), "val_loss": float(val_loss), "lr": float(lr), "elapsed_sec": float(elapsed)})
            pd.DataFrame(logs).to_csv(out_csv, index=False, encoding="utf-8")
            if val_loss < best_val - 1e-7:
                best_val = float(val_loss)
                best_step = int(step)
                best_path = save_best_checkpoint(model, opt, step, ckpt_root, tag, best_val)
            print(f"[train:{tag}] step={step}/{cfg.max_steps} lr={lr:.3e} train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}@{best_step} eta={eta_h:.2f}h")
            if step >= cfg.min_train_steps and (step - best_step) >= cfg.early_stop_patience:
                print(f"[train:{tag}] early stop at step={step}")
                break

        if step % cfg.save_interval == 0:
            save_checkpoint(model, opt, step, ckpt_root, tag)

        pbar.set_postfix(step=step, lr=f"{lr:.2e}", tr=f"{train_loss:.3f}", best=f"{best_val:.3f}")
    pbar.close()

    final_dir = root_dir / "checkpoints" / f"{tag}_final"
    save_final_model(model, final_dir)
    if best_path is None:
        best_path = save_best_checkpoint(model, opt, 0, ckpt_root, tag, float("nan"))

    return {
        "tag": tag,
        "log_csv": str(out_csv),
        "best_step": int(best_step),
        "best_val": float(best_val),
        "best_ckpt": str(best_path),
        "final_model": str(final_dir / "model.pt"),
    }


def build_completion_passkey_prompt(tokenizer: p4.TokenizerBase, context_length: int, passkey: str, ratio: float) -> Optional[List[int]]:
    filler = (
        "In the vast expanse of the universe, countless stars illuminate the darkness of space. "
        "Researchers carefully evaluate long-context reasoning under controlled settings. "
    )
    insertion = f"REMEMBER: The special number is {passkey}. REMEMBER: The special number is {passkey}. "
    extraction = "As stated above, the special number is "

    filler_ids = tokenizer.encode(filler)
    insertion_ids = tokenizer.encode(insertion)
    extraction_ids = tokenizer.encode(extraction)

    budget = context_length - len(insertion_ids) - len(extraction_ids) - 2
    if budget < 128:
        return None
    rep = budget // max(1, len(filler_ids)) + 3
    body = (filler_ids * rep)[:budget]
    pos = int(ratio * max(1, len(body) - 1))
    seq = body[:pos] + insertion_ids + body[pos:] + extraction_ids
    seq = seq[:context_length]
    if tokenizer.bos_token_id is not None:
        seq = [int(tokenizer.bos_token_id)] + seq
    return seq


def greedy_generate(model: p4.GPTSmall, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    x = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
    return x


def sample_false_passkey_with_same_token_len(
    rng: random.Random,
    tokenizer: p4.TokenizerBase,
    true_passkey: str,
    true_token_len: int,
    max_trials: int = 256,
) -> str:
    fallback = true_passkey
    for _ in range(max_trials):
        cand = f"{rng.randint(10000, 99999)}"
        if cand == true_passkey:
            continue
        fallback = cand
        if len(tokenizer.encode(cand)) == true_token_len:
            return cand
    if fallback == true_passkey:
        fallback = f"{(int(true_passkey) + 1) % 100000:05d}"
    return fallback


def key_only_ce_loss(
    model: p4.GPTSmall,
    prompt_ids: Sequence[int],
    key_ids: Sequence[int],
) -> float:
    if len(key_ids) <= 0:
        return float("nan")
    device = next(model.parameters()).device
    x_ids = list(prompt_ids) + list(key_ids)
    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    key_len = len(key_ids)
    with torch.no_grad():
        logits = model(x)
    pred = logits[:, -key_len - 1 : -1, :].contiguous()
    labels = x[:, -key_len:].contiguous()
    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1), reduction="mean")
    return float(loss.detach().item())


def evaluate_passkey_fixed_for_model(
    model: p4.GPTSmall,
    model_name: str,
    tokenizer: p4.TokenizerBase,
    lengths: List[int],
    ratios: List[float],
    repeats: int,
    max_new_tokens: int,
    seed: int,
    debug_examples: int = 3,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict] = []
    debug_printed = 0
    digit_re = re.compile(r"\d+")
    model.eval()
    pbar = tqdm(total=len(lengths) * len(ratios) * repeats, desc=f"passkey-{model_name}", dynamic_ncols=True)
    for lval in lengths:
        for rval in ratios:
            for rep_idx in range(repeats):
                key = f"{rng.randint(10000, 99999)}"
                prompt = build_completion_passkey_prompt(tokenizer, lval, key, rval)
                if prompt is None:
                    rows.append({"model": model_name, "context_length": lval, "position_ratio": rval, "repeat": rep_idx, "correct": 0, "status": "invalid"})
                    pbar.update(1)
                    continue
                x = torch.tensor([prompt], dtype=torch.long, device=next(model.parameters()).device)
                try:
                    out = greedy_generate(model, x, max_new_tokens=max_new_tokens)
                    gen_ids = out[0, x.size(1):].detach().cpu().tolist()
                    gen_txt = tokenizer.decode(gen_ids)
                    pred_digits = "".join(digit_re.findall(gen_txt))
                    ok = int((key in pred_digits) or (key in gen_txt))
                    if debug_printed < debug_examples and lval == min(lengths):
                        print(f"\n[passkey-debug] {model_name} L={lval} ratio={rval} rep={rep_idx}")
                        print("  prompt_tail:", tokenizer.decode(prompt[-200:]))
                        print("  generated_text:", gen_txt)
                        print("  generated_token_ids:", gen_ids)
                        print("  expected_passkey:", key)
                        print("  passkey_in_generated:", bool(ok))
                        debug_printed += 1
                    rows.append({"model": model_name, "context_length": lval, "position_ratio": rval, "repeat": rep_idx, "correct": ok, "status": "ok", "passkey": key, "pred_digits": pred_digits, "generated_text": gen_txt})
                except RuntimeError as ex:
                    rows.append({"model": model_name, "context_length": lval, "position_ratio": rval, "repeat": rep_idx, "correct": 0, "status": "oom" if "out of memory" in str(ex).lower() else "error"})
                    cleanup_cuda()
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def evaluate_passkey_teacher_forcing_for_model(
    model: p4.GPTSmall,
    model_name: str,
    tokenizer: p4.TokenizerBase,
    lengths: List[int],
    ratios: List[float],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict] = []
    model.eval()
    pbar = tqdm(total=len(lengths) * len(ratios) * repeats, desc=f"passkey-tf-{model_name}", dynamic_ncols=True)
    for lval in lengths:
        for rval in ratios:
            for rep_idx in range(repeats):
                key = f"{rng.randint(10000, 99999)}"
                prompt = build_completion_passkey_prompt(tokenizer, lval, key, rval)
                if prompt is None:
                    rows.append(
                        {
                            "model": model_name,
                            "context_length": lval,
                            "position_ratio": rval,
                            "repeat": rep_idx,
                            "correct": 0,
                            "status": "invalid",
                        }
                    )
                    pbar.update(1)
                    continue
                true_ids = tokenizer.encode(key)
                false_key = sample_false_passkey_with_same_token_len(rng, tokenizer, key, len(true_ids))
                false_ids = tokenizer.encode(false_key)
                try:
                    true_loss = key_only_ce_loss(model, prompt, true_ids)
                    false_loss = key_only_ce_loss(model, prompt, false_ids)
                    hit = int(true_loss < false_loss)
                    rows.append(
                        {
                            "model": model_name,
                            "context_length": lval,
                            "position_ratio": rval,
                            "repeat": rep_idx,
                            "correct": hit,
                            "status": "ok",
                            "passkey_true": key,
                            "passkey_false": false_key,
                            "true_loss": true_loss,
                            "false_loss": false_loss,
                            "margin_false_minus_true": float(false_loss - true_loss),
                        }
                    )
                except RuntimeError as ex:
                    rows.append(
                        {
                            "model": model_name,
                            "context_length": lval,
                            "position_ratio": rval,
                            "repeat": rep_idx,
                            "correct": 0,
                            "status": "oom" if "out of memory" in str(ex).lower() else "error",
                        }
                    )
                    cleanup_cuda()
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

def evaluate_ppl_vs_length_for_model(
    model: p4.GPTSmall,
    model_name: str,
    token_ids: np.ndarray,
    lengths: List[int],
    seed: int,
    base_samples: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    rows: List[Dict] = []
    model.eval()
    for L in lengths:
        if token_ids.size <= L + 2:
            rows.append({"model": model_name, "length": int(L), "loss": np.nan, "ppl": np.nan, "status": "short_data"})
            continue
        samples = base_samples if L <= 8192 else max(1, base_samples // 4)
        max_start = int(token_ids.size - L - 1)
        starts = rng.integers(low=0, high=max_start, size=samples, endpoint=False)
        losses: List[float] = []
        status = "ok"
        for st in starts:
            arr = token_ids[st: st + L + 1]
            x = torch.tensor(arr[None, :], dtype=torch.long, device=device)
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = model.compute_loss(x)
                losses.append(float(loss.detach().item()))
            except RuntimeError as ex:
                status = "oom" if "out of memory" in str(ex).lower() else "error"
                cleanup_cuda()
                break
        if losses:
            mean_loss = float(np.mean(losses))
            ppl = float(math.exp(min(mean_loss, 20.0)))
        else:
            mean_loss = float("nan")
            ppl = float("nan")
        rows.append({"model": model_name, "length": int(L), "loss": mean_loss, "ppl": ppl, "status": status})
    return pd.DataFrame(rows)


def compute_binned_positional_loss_for_model(
    model: p4.GPTSmall,
    model_name: str,
    token_ids: np.ndarray,
    seq_len: int,
    num_samples: int,
    bins: int,
    seed: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> pd.DataFrame:
    if token_ids.size <= seq_len + 2:
        return pd.DataFrame(columns=["model", "bin", "position_center", "loss"])
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    max_start = int(token_ids.size - seq_len - 1)
    starts = rng.integers(low=0, high=max_start, size=num_samples, endpoint=False)
    all_losses: List[np.ndarray] = []
    model.eval()
    for st in starts:
        arr = token_ids[st: st + seq_len + 1]
        x = torch.tensor(arr[None, :], dtype=torch.long, device=device)
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss_tok = model.compute_per_token_loss(x).detach().float().cpu().numpy()[0]
            all_losses.append(loss_tok)
        except RuntimeError:
            cleanup_cuda()
            continue
    if not all_losses:
        return pd.DataFrame(columns=["model", "bin", "position_center", "loss"])
    mean_loss = np.mean(np.stack(all_losses, axis=0), axis=0)
    chunks = np.array_split(np.arange(mean_loss.shape[0]), bins)
    rows: List[Dict] = []
    for bidx, idxs in enumerate(chunks, start=1):
        if idxs.size == 0:
            continue
        rows.append({"model": model_name, "bin": int(bidx), "position_center": float(idxs.mean() + 1), "loss": float(mean_loss[idxs].mean())})
    return pd.DataFrame(rows)


def plot_training_curves_all(root_dir: Path, train_meta: List[Dict], color_map: Dict[str, str]) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.4), sharex=True)
    for item in train_meta:
        tag = str(item["tag"])
        df = pd.read_csv(item["log_csv"])
        c = color_map.get(tag, None)
        axes[0].plot(df["step"], df["train_loss"], label=tag, color=c)
        axes[1].plot(df["step"], df["val_loss"], label=tag, color=c)
        best_step = int(item.get("best_step", 0) or 0)
        if best_step > 0:
            axes[1].axvline(best_step, linestyle="--", alpha=0.25, color=c)
    axes[0].set_ylabel("Training Loss")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_xlabel("Step")
    axes[0].legend(frameon=True)
    axes[1].legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "training_curves_all")
    plt.close(fig)


def plot_ppl_vs_length(root_dir: Path, df: pd.DataFrame, train_len_boundary: int) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for model_name in sorted(df["model"].unique().tolist()):
        sub = df[(df["model"] == model_name) & (df["status"] == "ok")].sort_values("length")
        if sub.empty:
            continue
        ax.plot(sub["length"], sub["ppl"], marker="o", label=model_name)
    ax.axvline(train_len_boundary, linestyle="--", color="black", alpha=0.7, label=f"train max len={train_len_boundary}")
    ax.set_xscale("log")
    ax.set_xlabel("Sequence Length L")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs Length")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "ppl_vs_length")
    plt.close(fig)


def plot_positional_loss(root_dir: Path, df: pd.DataFrame) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for model_name in sorted(df["model"].unique().tolist()):
        sub = df[df["model"] == model_name].sort_values("position_center")
        if sub.empty:
            continue
        ax.plot(sub["position_center"], sub["loss"], marker="o", label=model_name)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Positional Loss (L=8192, 32 bins)")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "positional_loss")
    plt.close(fig)


def plot_passkey_fixed(root_dir: Path, df: pd.DataFrame, model_order: List[str]) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    set_plot_style()
    n = len(model_order)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8), squeeze=False)
    axes = axes[0]
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        sub = ok_df[ok_df["model"] == model_name]
        piv = sub.groupby(["context_length", "position_ratio"], as_index=False)["correct"].mean().pivot(index="context_length", columns="position_ratio", values="correct")
        if piv.empty:
            ax.axis("off")
            ax.set_title(model_name)
            continue
        im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_title(model_name)
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([str(int(v)) for v in piv.index])
        ax.set_xlabel("Passkey Position")
        if idx == 0:
            ax.set_ylabel("Context Length")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Accuracy")
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "passkey_fixed")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.2, 4.8))
    mean_len = ok_df.groupby(["model", "context_length"], as_index=False)["correct"].mean()
    for model_name in model_order:
        sub = mean_len[mean_len["model"] == model_name].sort_values("context_length")
        if sub.empty:
            continue
        ax2.plot(sub["context_length"], sub["correct"], marker="o", label=model_name)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title("Passkey Accuracy by Length (Fixed Eval)")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, root_dir / "results" / "passkey_by_length_fixed")
    plt.close(fig2)


def plot_freq_comparison(root_dir: Path, freq_map: Dict[str, np.ndarray], max_seq_len: int) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    i = np.arange(len(next(iter(freq_map.values()))))
    ax = axes[0]
    for name, freq in freq_map.items():
        ax.plot(i, np.log10(np.clip(freq, 1e-16, None)), label=name)
    ax.set_xlabel("Frequency Pair Index i")
    ax.set_ylabel("log10(theta_i)")
    ax.set_title("Frequency Distribution")
    ax.legend(frameon=True)

    ax = axes[1]
    for name, freq in freq_map.items():
        ax.plot(i, 2.0 * math.pi / np.clip(freq, 1e-16, None), label=name)
    ax.axhline(max_seq_len, linestyle="--", color="black", alpha=0.6, label=f"L={max_seq_len}")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency Pair Index i")
    ax.set_ylabel("Wavelength lambda_i")
    ax.set_title("Effective Wavelength")
    ax.legend(frameon=True)

    ax = axes[2]
    for name, freq in freq_map.items():
        ratio = freq[1:] / np.clip(freq[:-1], 1e-16, None)
        ax.plot(i[1:], ratio, label=name)
    ax.set_xlabel("Frequency Pair Index i")
    ax.set_ylabel("theta_{i+1}/theta_i")
    ax.set_title("Adjacent Frequency Ratio")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, root_dir / "results" / "freq_comparison_trained")
    plt.close(fig)

def parse_int_list(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in str(s).split(",") if x.strip()})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase4 corrected: fixed passkey + anchored training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "hf", "byte"])
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument("--local_model_candidates", type=str, default="/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b,/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct,/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct")
    ap.add_argument("--dataset_mode", type=str, default="auto", choices=["auto", "hf", "local", "synthetic"])
    ap.add_argument("--target_tokens", type=int, default=24000000)
    ap.add_argument("--max_docs", type=int, default=20000)
    ap.add_argument("--synthetic_docs", type=int, default=2000)
    ap.add_argument(
        "--allow_byte_fallback",
        action="store_true",
        help="Allow silent fallback to byte tokenizer when HF/local tokenizer is unavailable.",
    )
    ap.add_argument(
        "--allow_synthetic_fallback",
        action="store_true",
        help="Allow auto/local/hf mode to fallback to synthetic data when target dataset is unavailable.",
    )
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=500)
    ap.add_argument("--min_train_steps", type=int, default=500)
    ap.add_argument("--effective_batch_target", type=int, default=8)
    ap.add_argument("--anchor_factor", type=float, default=20.0)
    ap.add_argument("--include_alpha_star", action="store_true")
    ap.add_argument("--passkey_repeats", type=int, default=20)
    ap.add_argument("--passkey_max_new_tokens", type=int, default=16)
    ap.add_argument("--passkey_lengths", type=str, default="1024,2048,4096,8192,16384")
    ap.add_argument(
        "--passkey_eval_mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "generation", "none"],
        help="Passkey evaluation protocol. teacher_forcing is the recommended robust mode.",
    )
    ap.add_argument("--ppl_lengths", type=str, default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--ppl_samples", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--rope_base", type=float, default=10000.0)
    ap.add_argument("--variants", type=str, default="standard,sigmoid,anchored20")
    ap.add_argument("--output_subdir", type=str, default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    set_seed(args.seed)

    base_root = Path(__file__).resolve().parent
    subdir = str(args.output_subdir).strip()
    root_dir = (base_root / subdir) if subdir else base_root
    (root_dir / "data").mkdir(parents=True, exist_ok=True)
    (root_dir / "results").mkdir(parents=True, exist_ok=True)
    (root_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[phase4-corrected] device:", device)

    tokenizer, tok_name = resolve_tokenizer(args.tokenizer_mode, args.tokenizer_path, args.local_model_candidates)
    print(f"[phase4-corrected] tokenizer: {tok_name}, vocab_size={tokenizer.vocab_size}")
    if tok_name == "byte" and args.tokenizer_mode != "byte" and not args.allow_byte_fallback:
        raise RuntimeError(
            "Tokenizer resolved to byte fallback in non-byte mode. "
            "This is usually protocol drift and can make PPL incomparable. "
            "Pass --allow_byte_fallback to override intentionally."
        )

    token_ids, dataset_name = p4.load_training_tokens(
        root_dir=root_dir,
        tokenizer=tokenizer,
        target_tokens=int(args.target_tokens),
        max_docs=int(args.max_docs),
        seed=int(args.seed),
        dataset_mode=str(args.dataset_mode),
        synthetic_docs=int(args.synthetic_docs),
    )
    if (
        dataset_name == "Synthetic-Passkey"
        and str(args.dataset_mode).lower() != "synthetic"
        and not args.allow_synthetic_fallback
    ):
        raise RuntimeError(
            "Dataset fell back to Synthetic-Passkey while dataset_mode is not synthetic. "
            "This can produce non-comparable PPL. Pass --allow_synthetic_fallback to override intentionally."
        )
    print(f"[phase4-corrected] dataset={dataset_name}, tokens={token_ids.size}")

    token_min = int(token_ids.min()) if token_ids.size else 0
    token_max = int(token_ids.max()) if token_ids.size else 0
    model_vocab_size = max(int(tokenizer.vocab_size), token_max + 1)
    print(f"[phase4-corrected] token_id_range=[{token_min}, {token_max}] tokenizer_vocab={tokenizer.vocab_size} model_vocab={model_vocab_size}")

    train_ids, val_ids = split_train_val(token_ids, val_ratio=0.05)
    print(f"[phase4-corrected] train_tokens={train_ids.size}, val_tokens={val_ids.size}")

    cfg = TrainConfig(
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        head_dim=int(args.head_dim),
        max_seq_len=int(args.max_seq_len),
        max_steps=int(args.max_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        eval_interval=int(args.eval_interval),
        save_interval=int(args.save_interval),
        early_stop_patience=int(args.early_stop_patience),
        min_train_steps=int(args.min_train_steps),
    )
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
    if cfg.d_model // cfg.n_heads != cfg.head_dim:
        raise ValueError(
            f"head_dim mismatch: d_model//n_heads={cfg.d_model // cfg.n_heads}, but --head_dim={cfg.head_dim}"
        )

    allocator = RoPEFrequencyAllocator(d=cfg.head_dim, base=float(args.rope_base))
    inv_std = allocator.standard()
    N = cfg.head_dim // 2
    k_sig = 16.05 / cfg.head_dim
    x0_sig = 0.47 * N
    j0_anchor = 0.47 * N
    inv_sig = allocator.sigmoid(k=k_sig, x0=x0_sig)
    inv_anchor20 = allocator.anchored_sigmoid(k=k_sig, j0=j0_anchor, anchor_factor=float(args.anchor_factor))
    theta_min_std = float(inv_std[-1].item())
    theta_target = float(2.0 * math.pi / float(cfg.max_seq_len))
    alpha_star = max(1.0, theta_target / max(theta_min_std, 1e-12))
    inv_anchor_star = allocator.anchored_sigmoid(k=k_sig, j0=j0_anchor, anchor_factor=alpha_star)

    print(f"[phase4-corrected] sigmoid params: k={k_sig:.6f}, x0={x0_sig:.4f}")
    print(f"[phase4-corrected] anchored params: k={k_sig:.6f}, j0={j0_anchor:.4f}, alpha={float(args.anchor_factor):.2f}")
    print(f"[phase4-corrected] alpha*={alpha_star:.4f}")

    torch.manual_seed(args.seed)
    template = p4.GPTSmall(vocab_size=model_vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_model=cfg.d_model, d_ff=cfg.d_ff, inv_freq=inv_std, gradient_checkpointing=True).to(device)
    param_count = p4.count_parameters(template)
    print(f"[phase4-corrected] model params: {param_count/1e6:.2f}M")

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    micro_batch = autotune_micro_batch(template, train_ids, cfg.max_seq_len, [16, 12, 8, 6, 4, 3, 2, 1], device, amp_dtype, use_amp, int(args.seed))
    if micro_batch > 8:
        print(f"[phase4-corrected] cap micro_batch {micro_batch} -> 8 for multi-model stability")
        micro_batch = 8
    grad_accum = max(1, int(math.ceil(float(args.effective_batch_target) / float(max(1, micro_batch)))))
    eff_batch = micro_batch * grad_accum
    print(f"[phase4-corrected] micro_batch={micro_batch}, grad_accum={grad_accum}, effective_batch={eff_batch}")

    init_state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    del template
    cleanup_cuda()
    val_batches = build_fixed_eval_batches(val_ids, cfg.max_seq_len, num_batches=8, batch_size=max(1, min(2, micro_batch)), seed=args.seed + 7)

    requested = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    freq_map: Dict[str, torch.Tensor] = {
        "standard": inv_std,
        "sigmoid": inv_sig,
        "anchored20": inv_anchor20,
        "anchored_alpha": inv_anchor_star,
    }
    if not requested:
        requested = ["standard", "sigmoid", "anchored20"]
    if args.include_alpha_star and "anchored_alpha" not in requested:
        requested.append("anchored_alpha")
    variant_specs: List[Tuple[str, torch.Tensor]] = []
    for name in requested:
        if name not in freq_map:
            raise ValueError(f"Unknown variant '{name}'. Supported: {', '.join(freq_map.keys())}")
        if name == "anchored_alpha" and (not args.include_alpha_star):
            continue
        variant_specs.append((name, freq_map[name]))
    color_map = {"standard": "#d62728", "sigmoid": "#1f77b4", "anchored20": "#2ca02c", "anchored_alpha": "#9467bd"}

    train_meta: List[Dict[str, object]] = []
    train_t0 = time.time()
    for tag, inv_freq in variant_specs:
        print(f"\n[phase4-corrected] ===== training {tag} =====")
        cur_mb = micro_batch
        while True:
            cur_accum = max(1, int(math.ceil(float(args.effective_batch_target) / float(max(1, cur_mb)))))
            torch.manual_seed(args.seed)
            model = p4.GPTSmall(
                vocab_size=model_vocab_size,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                inv_freq=inv_freq,
                gradient_checkpointing=True,
            ).to(device)
            try:
                meta = train_single_model(
                    root_dir,
                    model,
                    tag,
                    init_state,
                    train_ids,
                    val_batches,
                    cfg,
                    data_seed=args.seed + 123,
                    micro_batch=cur_mb,
                    grad_accum=cur_accum,
                )
                train_meta.append(meta)
                del model
                cleanup_cuda()
                break
            except RuntimeError as ex:
                del model
                cleanup_cuda()
                if "out of memory" in str(ex).lower() and cur_mb > 1:
                    next_mb = max(1, cur_mb // 2)
                    print(f"[phase4-corrected] OOM on {tag} with micro_batch={cur_mb}, retry -> {next_mb}")
                    cur_mb = next_mb
                    continue
                raise
    train_hours = (time.time() - train_t0) / 3600.0
    plot_training_curves_all(root_dir, train_meta, color_map)

    eval_models: List[Dict[str, object]] = []
    for tag, inv_freq in variant_specs:
        hit = [m for m in train_meta if str(m["tag"]) == tag]
        if not hit:
            continue
        eval_models.append({"tag": tag, "display": "Anchored-20" if tag == "anchored20" else ("Anchored-alpha*" if tag == "anchored_alpha" else tag.capitalize()), "inv_freq": inv_freq, "best_ckpt": Path(str(hit[0]["best_ckpt"])), "best_step": int(hit[0]["best_step"]), "best_val": float(hit[0]["best_val"])})

    passkey_lengths = parse_int_list(args.passkey_lengths)
    passkey_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    pass_frames: List[pd.DataFrame] = []
    passkey_df = pd.DataFrame()
    passkey_csv = root_dir / "data" / "passkey_fixed_results.csv"
    if str(args.passkey_eval_mode) != "none":
        for em in eval_models:
            model = p4.GPTSmall(
                vocab_size=model_vocab_size,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                inv_freq=em["inv_freq"],
                gradient_checkpointing=False,
            ).to(device)
            load_model_state(model, Path(em["best_ckpt"]), device)
            if str(args.passkey_eval_mode) == "teacher_forcing":
                frame = evaluate_passkey_teacher_forcing_for_model(
                    model,
                    str(em["display"]),
                    tokenizer,
                    passkey_lengths,
                    passkey_ratios,
                    int(args.passkey_repeats),
                    seed=args.seed + 900 + hash(str(em["tag"])) % 1000,
                )
            else:
                frame = evaluate_passkey_fixed_for_model(
                    model,
                    str(em["display"]),
                    tokenizer,
                    passkey_lengths,
                    passkey_ratios,
                    int(args.passkey_repeats),
                    int(args.passkey_max_new_tokens),
                    seed=args.seed + 900 + hash(str(em["tag"])) % 1000,
                    debug_examples=2,
                )
            pass_frames.append(frame)
            del model
            cleanup_cuda()
        passkey_df = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
        passkey_df.to_csv(passkey_csv, index=False, encoding="utf-8")
        plot_passkey_fixed(root_dir, passkey_df, [str(m["display"]) for m in eval_models])
    else:
        print("[phase4-corrected] passkey evaluation skipped by --passkey_eval_mode none")

    ppl_lengths = parse_int_list(args.ppl_lengths)
    ppl_frames: List[pd.DataFrame] = []
    for em in eval_models:
        model = p4.GPTSmall(vocab_size=model_vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_model=cfg.d_model, d_ff=cfg.d_ff, inv_freq=em["inv_freq"], gradient_checkpointing=False).to(device)
        load_model_state(model, Path(em["best_ckpt"]), device)
        ppl_frames.append(evaluate_ppl_vs_length_for_model(model, str(em["display"]), val_ids, ppl_lengths, seed=args.seed + 300 + hash(str(em["tag"])) % 1000, base_samples=int(args.ppl_samples), use_amp=use_amp, amp_dtype=amp_dtype))
        del model
        cleanup_cuda()
    ppl_df = pd.concat(ppl_frames, ignore_index=True) if ppl_frames else pd.DataFrame()
    ppl_csv = root_dir / "data" / "ppl_vs_length.csv"
    ppl_df.to_csv(ppl_csv, index=False, encoding="utf-8")
    plot_ppl_vs_length(root_dir, ppl_df, cfg.max_seq_len)

    pos_frames: List[pd.DataFrame] = []
    for em in eval_models:
        model = p4.GPTSmall(vocab_size=model_vocab_size, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_model=cfg.d_model, d_ff=cfg.d_ff, inv_freq=em["inv_freq"], gradient_checkpointing=False).to(device)
        load_model_state(model, Path(em["best_ckpt"]), device)
        pos_frames.append(compute_binned_positional_loss_for_model(model, str(em["display"]), val_ids, seq_len=cfg.max_seq_len, num_samples=8, bins=32, seed=args.seed + 500 + hash(str(em["tag"])) % 1000, use_amp=use_amp, amp_dtype=amp_dtype))
        del model
        cleanup_cuda()
    pos_df = pd.concat(pos_frames, ignore_index=True) if pos_frames else pd.DataFrame()
    pos_csv = root_dir / "data" / "positional_loss.csv"
    pos_df.to_csv(pos_csv, index=False, encoding="utf-8")
    plot_positional_loss(root_dir, pos_df)

    freq_map: Dict[str, np.ndarray] = {}
    for tag, invf in variant_specs:
        name = "Anchored-20" if tag == "anchored20" else ("Anchored-alpha*" if tag == "anchored_alpha" else tag.capitalize())
        freq_map[name] = invf.detach().cpu().double().numpy()
    plot_freq_comparison(root_dir, freq_map, cfg.max_seq_len)

    best_rows = []
    for em in eval_models:
        best_rows.append({"model": str(em["display"]), "best_step": int(em["best_step"]), "best_val": float(em["best_val"]), "best_ppl": float(math.exp(min(float(em["best_val"]), 20.0)))})
    best_df = pd.DataFrame(best_rows)

    passkey_ok = passkey_df[passkey_df["status"] == "ok"].copy() if not passkey_df.empty else pd.DataFrame()
    passkey_avg = passkey_ok.groupby(["model", "context_length"], as_index=False)["correct"].mean() if not passkey_ok.empty else pd.DataFrame(columns=["model", "context_length", "correct"])

    print("\n====================================================================")
    print("        Sigmoid-RoPE Phase 4 (Corrected) — Results")
    print("====================================================================")
    print(f"\n模型: {param_count/1e6:.1f}M params, head_dim={cfg.head_dim}, trained on {dataset_name}, {cfg.max_steps} steps")
    print("\nRoPE 参数:")
    print(f"  Sigmoid:    k={k_sig:.4f}, x0={x0_sig:.2f}")
    print(f"  Anchored:   k={k_sig:.4f}, j0={j0_anchor:.2f}, alpha={float(args.anchor_factor):.2f}")
    if args.include_alpha_star:
        print(f"  Anchored*:  alpha*={alpha_star:.4f}")
    print("\n训练结果 (best val):")
    for _, r in best_df.iterrows():
        print(f"  {r['model']:<15} best_step={int(r['best_step']):>5d}  val_loss={r['best_val']:.4f}  val_ppl={r['best_ppl']:.3f}")
    print("\n长度外推 PPL:")
    for model_name in sorted(ppl_df["model"].unique().tolist()):
        sub = ppl_df[ppl_df["model"] == model_name].sort_values("length")
        vals = []
        for L in [1024, 2048, 4096, 8192, 16384, 32768]:
            hit = sub[sub["length"] == L]
            if hit.empty or not np.isfinite(float(hit.iloc[0]["ppl"])):
                vals.append(f"L{L}:NA")
            else:
                vals.append(f"L{L}:{float(hit.iloc[0]['ppl']):.2f}")
        print(f"  {model_name:<15} " + "  ".join(vals))
    print("\nPasskey Retrieval (修复版):")
    for model_name in sorted(passkey_avg["model"].unique().tolist()) if not passkey_avg.empty else []:
        sub = passkey_avg[passkey_avg["model"] == model_name].sort_values("context_length")
        vals = [f"L{int(x)}:{float(y)*100:.1f}%" for x, y in zip(sub["context_length"], sub["correct"])]
        print(f"  {model_name:<15} " + "  ".join(vals))
    print("\n核心发现:")
    print("  1. 已修复 Passkey 评测协议（completion 模式 + 诊断打印 + 原始生成保存）。")
    print("  2. 已加入 Anchored-20（及可选 alpha*）并与 Standard/Sigmoid 共享初始化、数据顺序、超参。")
    print("  3. 已输出统一图：training_curves_all / ppl_vs_length / positional_loss / passkey_fixed / freq_comparison_trained。")
    print("====================================================================")

    save_json(root_dir / "data" / "phase4_corrected_summary.json", {
        "dataset_name": dataset_name,
        "tokenizer": tok_name,
        "total_tokens": int(token_ids.size),
        "param_count": int(param_count),
        "train_hours": float(train_hours),
        "train_cfg": {
            "max_steps": int(cfg.max_steps), "eval_interval": int(cfg.eval_interval), "save_interval": int(cfg.save_interval), "lr": float(cfg.lr), "warmup_steps": int(cfg.warmup_steps), "early_stop_patience": int(cfg.early_stop_patience), "micro_batch": int(micro_batch), "grad_accum": int(grad_accum), "effective_batch": int(eff_batch), "amp_dtype": str(amp_dtype)
        },
        "rope_params": {
            "sigmoid": {"k": float(k_sig), "x0": float(x0_sig)},
            "anchored20": {"k": float(k_sig), "j0": float(j0_anchor), "alpha": float(args.anchor_factor)},
            "anchored_alpha": {"k": float(k_sig), "j0": float(j0_anchor), "alpha": float(alpha_star)} if args.include_alpha_star else None,
        },
        "train_meta": train_meta,
        "best_table": best_rows,
        "artifacts": {
            "training_curves_all": str(root_dir / "results" / "training_curves_all.pdf"),
            "ppl_vs_length": str(root_dir / "results" / "ppl_vs_length.pdf"),
            "positional_loss": str(root_dir / "results" / "positional_loss.pdf"),
            "passkey_fixed": str(root_dir / "results" / "passkey_fixed.pdf"),
            "passkey_by_length_fixed": str(root_dir / "results" / "passkey_by_length_fixed.pdf"),
            "freq_comparison_trained": str(root_dir / "results" / "freq_comparison_trained.pdf"),
            "passkey_csv": str(passkey_csv), "ppl_csv": str(ppl_csv), "positional_csv": str(pos_csv)
        },
    })


if __name__ == "__main__":
    main()
