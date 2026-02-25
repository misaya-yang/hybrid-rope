#!/usr/bin/env python3
"""
Overnight 8-hour Automated Experiment Suite
============================================
Train 4 RoPE methods on LLaMA-3-8B-Instruct LoRA + NIAH evaluation.
Launch at night, find publication-ready results by morning.

Pipeline:
  Gate 0  Calibration check (4 methods)          ~5 min
  Gate 1  Full LoRA training (4×600 steps@16K)   ~5 h
  Gate 2  NIAH eval (base + 4 adapters)          ~1.5 h
  Gate 3  Comparison summary + heatmaps          ~1 min
  Total                                          ~7 h

Usage:
    cd /root/autodl-tmp/dfrope/hybrid-rope
    nohup /root/miniconda3/bin/python -u 2026-02-22/scripts/run_overnight_8h.py \
        > results/overnight_8h/console.log 2>&1 &
    echo $!
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ============================================================
# Constants
# ============================================================

PYTHON = "/root/miniconda3/bin/python"
REPO_ROOT = Path("/root/autodl-tmp/dfrope/hybrid-rope")
TRAIN_SCRIPT = str(REPO_ROOT / "2026-02-22" / "scripts" / "run_llama8b_fair_suite.py")
BASE_MODEL = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
OUTPUT_ROOT = REPO_ROOT / "results" / "overnight_8h"

METHODS = ["baseline", "pi", "yarn", "anchored_hybrid"]
TRAIN_STEPS = 300
MAX_SEQ_LEN = 16384
SEED = 42

NIAH_LENGTHS = [4096, 8192, 16384, 32768]
NIAH_DEPTHS = list(range(0, 101, 10))  # [0, 10, ..., 100]
NIAH_TRIALS = 3
NIAH_MAX_NEW_TOKENS = 24

FILLER_TEXT = (
    "This document section discusses long context memory, retrieval, information processing, "
    "and instruction fidelity across various domains including technology, science, and culture. "
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================
# Utilities
# ============================================================

_T0_GLOBAL = time.time()
_LOG_PATH: Optional[Path] = None


def log(stage: str, msg: str) -> None:
    elapsed = int(time.time() - _T0_GLOBAL)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    line = f"[{h:02d}:{m:02d}:{s:02d}] [{stage}] {msg}"
    print(line, flush=True)
    if _LOG_PATH:
        try:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def elapsed_str(t0: float) -> str:
    s = int(time.time() - t0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def run_subprocess(args: List[str], stage: str, timeout: int = 10800) -> Tuple[int, str]:
    """Run subprocess with output capture. 3h default timeout per call."""
    log(stage, f"CMD: {' '.join(args[:6])}...")
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
        )
        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            log(stage, f"FAILED rc={result.returncode}")
            # Print last 1500 chars of output for diagnosis
            log(stage, output[-1500:])
        return result.returncode, output
    except subprocess.TimeoutExpired:
        log(stage, "TIMEOUT")
        return -1, "TIMEOUT"
    except Exception as e:
        log(stage, f"SUBPROCESS ERROR: {e}")
        return -1, str(e)


# ============================================================
# Gate 0: Calibration
# ============================================================

def gate0_calibration() -> Dict[str, bool]:
    log("GATE0", "=" * 60)
    log("GATE0", "Calibration check for all methods")
    results = {}
    for method in METHODS:
        run_name = f"calib_{method}"
        args = [
            PYTHON, TRAIN_SCRIPT,
            "--method", method,
            "--run_name", run_name,
            "--output_root", str(OUTPUT_ROOT),
            "--max_seq_len", str(MAX_SEQ_LEN),
            "--calibration_only",
            "--seed", str(SEED),
        ]
        rc, _ = run_subprocess(args, f"GATE0/{method}", timeout=600)
        success = rc == 0
        results[method] = success
        report_path = OUTPUT_ROOT / run_name / "artifacts" / "calibration_report.json"
        if success and report_path.exists():
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
            n = rpt.get("inject_info", {}).get("patched_count", "?")
            log("GATE0", f"  {method}: OK, patched {n} layers")
        else:
            log("GATE0", f"  {method}: {'FAIL' if not success else 'OK (no report)'}")
    passed = sum(results.values())
    log("GATE0", f"Calibration: {passed}/{len(METHODS)} passed")
    return results


# ============================================================
# Gate 1: Full Training
# ============================================================

def gate1_training(calib_ok: Dict[str, bool]) -> Dict[str, Optional[Path]]:
    log("GATE1", "=" * 60)
    log("GATE1", f"Full training: {TRAIN_STEPS} steps × {len(METHODS)} methods @ {MAX_SEQ_LEN} ctx")
    adapters: Dict[str, Optional[Path]] = {}
    for method in METHODS:
        if not calib_ok.get(method, False):
            log("GATE1", f"SKIP {method} (calibration failed)")
            adapters[method] = None
            continue

        run_name = f"train_{method}"
        adapter_dir = OUTPUT_ROOT / run_name / "final_lora"
        t0 = time.time()
        args = [
            PYTHON, TRAIN_SCRIPT,
            "--method", method,
            "--run_name", run_name,
            "--output_root", str(OUTPUT_ROOT),
            "--max_steps", str(TRAIN_STEPS),
            "--max_seq_len", str(MAX_SEQ_LEN),
            "--seed", str(SEED),
        ]
        rc, output = run_subprocess(args, f"GATE1/{method}", timeout=36000)
        dt = elapsed_str(t0)

        if rc == 0 and adapter_dir.exists():
            log("GATE1", f"  {method}: OK ({dt})")
            adapters[method] = adapter_dir
            # Quick NaN check in train log
            log_file = OUTPUT_ROOT / run_name / "logs" / "train.log"
            if log_file.exists():
                txt = log_file.read_text("utf-8", errors="ignore")
                if "nan" in txt.lower():
                    log("GATE1", f"  WARNING: {method} log contains NaN!")
        else:
            log("GATE1", f"  {method}: FAILED ({dt})")
            adapters[method] = None

    ok = sum(1 for v in adapters.values() if v is not None)
    log("GATE1", f"Training complete: {ok}/{len(METHODS)} succeeded")
    return adapters


# ============================================================
# Gate 2: NIAH Evaluation
# ============================================================

def _find_rotary_modules(model: torch.nn.Module):
    out = []
    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and torch.is_tensor(getattr(mod, "inv_freq")):
            out.append((name, mod))
    return out


def _inject_inv_freq(model: torch.nn.Module, inv_freq: torch.Tensor) -> int:
    modules = _find_rotary_modules(model)
    for _, mod in modules:
        old = mod.inv_freq
        with torch.no_grad():
            old.copy_(inv_freq.to(device=old.device, dtype=old.dtype))
        for attr in ["_cos_cached", "_sin_cached", "cos_cached", "sin_cached",
                      "_cos_cache", "_sin_cache", "max_seq_len_cached"]:
            if hasattr(mod, attr):
                try:
                    v = getattr(mod, attr)
                    setattr(mod, attr, 0 if isinstance(v, (int, float)) else None)
                except Exception:
                    pass
    return len(modules)


def _build_niah_prompt(tokenizer, target_len: int, depth_pct: int, rng: random.Random):
    passkey = str(rng.randint(10000, 99999))
    needle = f"The special magic number is {passkey}. Keep this exact number in memory."
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER_TEXT, add_special_tokens=False)
    if not filler_ids:
        filler_ids = [tokenizer.eos_token_id or 0]

    budget = max(256, target_len - 500)
    filler_budget = max(0, budget - len(needle_ids))
    reps = filler_budget // len(filler_ids) + 1
    ctx_ids = (filler_ids * reps)[:filler_budget]
    pos = int(round(depth_pct / 100.0 * len(ctx_ids)))
    pos = max(0, min(pos, len(ctx_ids)))
    merged = ctx_ids[:pos] + needle_ids + ctx_ids[pos:]
    context_text = tokenizer.decode(merged, skip_special_tokens=True)

    messages = [{"role": "user", "content": (
        "Read the following document and answer with only the exact number.\n\n"
        f"{context_text}\n\n"
        "Question: What is the special magic number?"
    )}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(prompt_ids) > target_len:
        prompt_ids = [prompt_ids[0]] + prompt_ids[-(target_len - 1):]
    return prompt_ids, passkey


@torch.no_grad()
def _run_niah_trial(model, tokenizer, target_len, depth_pct, rng):
    prompt_ids, passkey = _build_niah_prompt(tokenizer, target_len, depth_pct, rng)
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids=x, max_new_tokens=NIAH_MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen_ids = out[0, x.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    codes = re.findall(r"\d{5,}", text)
    return passkey in codes, passkey, text


def _evaluate_niah_matrix(model, tokenizer, lengths, depths, trials, seed):
    matrix = np.full((len(depths), len(lengths)), np.nan, dtype=np.float32)
    raw: Dict = {}
    rng = random.Random(seed)
    skip_len = False

    for col, L in enumerate(lengths):
        raw[str(L)] = {}
        if skip_len:
            for d in depths:
                raw[str(L)][str(d)] = {"status": "skipped_oom"}
            continue

        for row, d in enumerate(depths):
            ok_count, total = 0, 0
            for t in range(trials):
                try:
                    ok, pk, gen = _run_niah_trial(model, tokenizer, L, d, rng)
                    total += 1
                    if ok:
                        ok_count += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log("NIAH", f"  OOM at L={L} d={d}%")
                        torch.cuda.empty_cache()
                        skip_len = True
                        break
                    log("NIAH", f"  Error at L={L} d={d}%: {e}")
                    break

            acc = ok_count / total if total > 0 else float("nan")
            matrix[row, col] = acc
            raw[str(L)][str(d)] = {"correct": ok_count, "total": total, "accuracy": acc}
            if skip_len:
                break

    return matrix, raw


def _save_heatmap(matrix, lengths, depths, path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("NIAH", "matplotlib unavailable, skipping heatmap")
        return
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    import pandas as pd
    df = pd.DataFrame(matrix, index=[f"{d}%" for d in depths], columns=[str(l) for l in lengths])
    plt.figure(figsize=(10, 7))
    if has_sns:
        sns.heatmap(df, cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f",
                    linewidths=.4, linecolor="white", cbar_kws={"label": "Accuracy"})
    else:
        ax = plt.gca()
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(lengths))); ax.set_xticklabels([str(l) for l in lengths])
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d}%" for d in depths])
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                v = matrix[r, c]
                ax.text(c, r, f"{v:.2f}" if not np.isnan(v) else "—", ha="center", va="center", fontsize=8)
        plt.colorbar(im, label="Accuracy")
    plt.title(title, fontsize=13)
    plt.xlabel("Context Length"); plt.ylabel("Needle Depth")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    log("NIAH", f"  Saved heatmap: {path.name}")


def _load_model_for_eval(inv_freq_path=None, adapter_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    model.config.use_cache = True
    model.eval()

    if inv_freq_path and Path(inv_freq_path).exists():
        inv = torch.load(str(inv_freq_path), map_location="cpu", weights_only=True)
        n = _inject_inv_freq(model, inv)
        log("NIAH", f"  Injected inv_freq ({n} layers) from {Path(inv_freq_path).name}")

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        log("NIAH", f"  Loaded LoRA adapter from {Path(adapter_path).name}")

    return model, tokenizer


def _unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


def gate2_niah_eval(adapters: Dict[str, Optional[Path]]) -> Dict[str, object]:
    log("GATE2", "=" * 60)
    log("GATE2", "NIAH recall evaluation")
    eval_root = OUTPUT_ROOT / "eval_niah"
    eval_root.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, object] = {}

    # Build eval configs: (label, inv_freq_path, adapter_path)
    configs: List[Tuple[str, Optional[str], Optional[str]]] = [
        ("base_no_lora", None, None),
    ]
    for method in METHODS:
        ap = adapters.get(method)
        if ap is None:
            continue
        inv_pt = ap.parent / "artifacts" / "custom_inv_freq.pt"
        configs.append((method, str(inv_pt) if inv_pt.exists() else None, str(ap)))

    for label, inv_path, adapter in configs:
        log("GATE2", f"Evaluating: {label}")
        out_dir = eval_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            model, tokenizer = _load_model_for_eval(inv_path, adapter)
            matrix, raw = _evaluate_niah_matrix(
                model, tokenizer, NIAH_LENGTHS, NIAH_DEPTHS, NIAH_TRIALS, SEED,
            )
            _save_heatmap(matrix, NIAH_LENGTHS, NIAH_DEPTHS,
                          out_dir / "niah_heatmap.png", f"NIAH Recall: {label}")

            # Save raw results
            (out_dir / "niah_results.json").write_text(
                json.dumps({"label": label, "raw": raw}, indent=2, ensure_ascii=False), "utf-8"
            )

            # Compute per-length mean accuracy
            means = {}
            for col, L in enumerate(NIAH_LENGTHS):
                col_vals = matrix[:, col]
                valid = col_vals[~np.isnan(col_vals)]
                means[str(L)] = float(np.mean(valid)) if len(valid) > 0 else None

            all_results[label] = {"means": means, "elapsed": elapsed_str(t0), "status": "ok"}
            log("GATE2", f"  {label}: means={means} ({elapsed_str(t0)})")
            _unload(model)

        except Exception as e:
            log("GATE2", f"  {label}: FAILED — {e}")
            all_results[label] = {"status": "failed", "error": str(e)}
            try:
                _unload(model)  # type: ignore
            except Exception:
                pass

    return all_results


# ============================================================
# Gate 3: Comparison Summary
# ============================================================

def gate3_summary(adapters: Dict[str, Optional[Path]], niah: Dict[str, object]) -> None:
    log("GATE3", "=" * 60)
    log("GATE3", "Generating comparison summary")
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Comparison table
    header = ["method"] + [str(L) for L in NIAH_LENGTHS]
    rows = []
    for label in ["base_no_lora"] + METHODS:
        entry = niah.get(label, {})
        if not isinstance(entry, dict):
            continue
        means = entry.get("means", {})
        row = [label] + [f"{means.get(str(L), 'N/A'):.3f}" if isinstance(means.get(str(L)), float) else "N/A"
               for L in NIAH_LENGTHS]
        rows.append(row)

    csv_lines = [",".join(header)]
    for r in rows:
        csv_lines.append(",".join(r))
    csv_text = "\n".join(csv_lines)
    (summary_dir / "comparison_table.csv").write_text(csv_text, "utf-8")
    log("GATE3", f"  Saved comparison_table.csv")

    # Markdown table
    md_lines = ["# Overnight Experiment Results", "", f"Date: {time.strftime('%Y-%m-%d %H:%M')}", "",
                f"Training: {TRAIN_STEPS} steps, {MAX_SEQ_LEN} ctx, LoRA r=64 α=128", "",
                "## NIAH Recall (mean accuracy across depths)", "",
                "| Method | " + " | ".join(str(L) for L in NIAH_LENGTHS) + " |",
                "|" + "|".join(["---"] * (len(NIAH_LENGTHS) + 1)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    md_lines.append("")

    # Per-method training summary
    md_lines.append("## Training Summary")
    md_lines.append("")
    for method in METHODS:
        ap = adapters.get(method)
        summary_path = OUTPUT_ROOT / f"train_{method}" / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text("utf-8"))
            loss = s.get("train_metrics", {}).get("train_loss", "?")
            secs = s.get("train_metrics", {}).get("train_seconds", "?")
            md_lines.append(f"- **{method}**: loss={loss}, time={secs}s, adapter={'✓' if ap else '✗'}")
        else:
            md_lines.append(f"- **{method}**: {'trained' if ap else 'FAILED'}")
    md_lines.append("")

    (summary_dir / "results.md").write_text("\n".join(md_lines), "utf-8")
    log("GATE3", f"  Saved results.md")

    # Comparison 4-panel heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("NIAH Recall Comparison (4 Methods)", fontsize=16, fontweight="bold")
        for idx, method in enumerate(METHODS):
            ax = axes[idx // 2][idx % 2]
            result_path = OUTPUT_ROOT / "eval_niah" / method / "niah_results.json"
            if result_path.exists():
                data = json.loads(result_path.read_text("utf-8"))
                raw = data.get("raw", {})
                matrix = np.full((len(NIAH_DEPTHS), len(NIAH_LENGTHS)), np.nan)
                for col, L in enumerate(NIAH_LENGTHS):
                    for row, d in enumerate(NIAH_DEPTHS):
                        cell = raw.get(str(L), {}).get(str(d), {})
                        acc = cell.get("accuracy")
                        if acc is not None:
                            matrix[row, col] = acc
                im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(NIAH_LENGTHS)))
                ax.set_xticklabels([str(l) for l in NIAH_LENGTHS], fontsize=9)
                ax.set_yticks(range(len(NIAH_DEPTHS)))
                ax.set_yticklabels([f"{d}%" for d in NIAH_DEPTHS], fontsize=8)
                for r in range(matrix.shape[0]):
                    for c in range(matrix.shape[1]):
                        v = matrix[r, c]
                        ax.text(c, r, f"{v:.0%}" if not np.isnan(v) else "—",
                                ha="center", va="center", fontsize=7,
                                color="white" if (not np.isnan(v) and v < 0.5) else "black")
            ax.set_title(method, fontsize=12, fontweight="bold")
            ax.set_xlabel("Context Length")
            ax.set_ylabel("Depth")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(summary_dir / "comparison_heatmap.png", dpi=200)
        plt.close(fig)
        log("GATE3", "  Saved comparison_heatmap.png")
    except Exception as e:
        log("GATE3", f"  Heatmap generation failed: {e}")

    # Final JSON dump
    final = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "methods": METHODS, "train_steps": TRAIN_STEPS,
            "max_seq_len": MAX_SEQ_LEN, "seed": SEED,
            "niah_lengths": NIAH_LENGTHS, "niah_depths": NIAH_DEPTHS,
            "niah_trials": NIAH_TRIALS,
        },
        "adapters": {m: str(p) if p else None for m, p in adapters.items()},
        "niah_results": niah,
        "total_elapsed": elapsed_str(_T0_GLOBAL),
    }
    (summary_dir / "final_report.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), "utf-8"
    )
    log("GATE3", f"  Saved final_report.json")
    log("GATE3", f"  Total experiment time: {elapsed_str(_T0_GLOBAL)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    global _LOG_PATH
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = OUTPUT_ROOT / "experiment.log"

    log("MAIN", "=" * 60)
    log("MAIN", "Overnight 8h Experiment Suite — Starting")
    log("MAIN", f"Methods: {METHODS}")
    log("MAIN", f"Train: {TRAIN_STEPS} steps @ {MAX_SEQ_LEN} ctx")
    log("MAIN", f"NIAH: lengths={NIAH_LENGTHS} depths={len(NIAH_DEPTHS)} trials={NIAH_TRIALS}")
    log("MAIN", f"Output: {OUTPUT_ROOT}")

    # Gate 0
    calib = gate0_calibration()
    if not any(calib.values()):
        log("MAIN", "ABORT: All calibrations failed!")
        return

    # Gate 1
    adapters = gate1_training(calib)
    if not any(v is not None for v in adapters.values()):
        log("MAIN", "ABORT: All trainings failed!")
        return

    # Gate 2
    niah = gate2_niah_eval(adapters)

    # Gate 3
    gate3_summary(adapters, niah)

    log("MAIN", "=" * 60)
    log("MAIN", "EXPERIMENT COMPLETE")
    log("MAIN", f"Check results at: {OUTPUT_ROOT / 'summary'}")
    log("MAIN", f"Total time: {elapsed_str(_T0_GLOBAL)}")


if __name__ == "__main__":
    main()
