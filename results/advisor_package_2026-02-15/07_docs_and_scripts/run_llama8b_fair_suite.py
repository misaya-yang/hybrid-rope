#!/usr/bin/env python3
"""
Run the fair 8B LoRA comparison suite in sequence:
- YaRN + 600 steps
- PI(linear) + 600 steps
- Hybrid + 600 steps
- PI-soft(dynamic) + 600 steps

It calls `scripts/train_llama8b_lora_variant.py` for each variant and writes:
- per-variant train/eval outputs under output_root/<variant>/
- global summary json with 16k/32k/64k PPL comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_csv_list(v: str) -> List[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: List[str], log_path: Path, cwd: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            f.write(line + "\n")
        rc = proc.wait()
        f.write(f"[{now()}] RC={rc}\n")
    return rc


def read_summary(summary_path: Path) -> Dict:
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fair 8B LoRA suite for yarn/pi/hybrid/pi_soft.")
    ap.add_argument("--repo_root", type=str, default="/root/autodl-tmp/dfrope/hybrid-rope")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument("--data_dir", type=str, default="/root/autodl-tmp/wikitext_data")
    ap.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_lora_suite",
    )
    ap.add_argument("--variants", type=str, default="yarn,pi,hybrid,pi_soft")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--eval_lengths", type=str, default="16384,32768,65536")
    ap.add_argument("--eval_chunks", type=int, default=5)
    ap.add_argument("--attn_implementation", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    suite_log = output_root / "suite.log"
    summary_json = output_root / "fair_ppl_compare.json"

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    variants = parse_csv_list(args.variants)
    all_results: Dict[str, Dict] = {}
    with suite_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== SUITE START {now()} variants={variants} ===\n")

    for variant in variants:
        variant_dir = output_root / variant
        summary_path = variant_dir / "summary.json"
        if summary_path.exists() and not args.force_rerun:
            all_results[variant] = read_summary(summary_path)
            print(f"[suite] skip {variant}, existing summary found", flush=True)
            continue

        cmd = [
            sys.executable,
            "scripts/train_llama8b_lora_variant.py",
            "--variant",
            variant,
            "--base_model_path",
            args.base_model_path,
            "--data_dir",
            args.data_dir,
            "--output_root",
            args.output_root,
            "--max_steps",
            str(args.max_steps),
            "--seq_len",
            str(args.seq_len),
            "--per_device_train_batch_size",
            str(args.per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--warmup_steps",
            str(args.warmup_steps),
            "--save_steps",
            str(args.save_steps),
            "--logging_steps",
            str(args.logging_steps),
            "--orig_ctx",
            str(args.orig_ctx),
            "--rope_factor",
            str(args.rope_factor),
            "--eval_lengths",
            args.eval_lengths,
            "--eval_chunks",
            str(args.eval_chunks),
            "--attn_implementation",
            args.attn_implementation,
            "--seed",
            str(args.seed),
        ]

        print(f"[suite] start variant={variant} at {now()}", flush=True)
        rc = run_cmd(cmd, log_path=variant_dir / "run.log", cwd=repo_root, env=env)
        print(f"[suite] done variant={variant} rc={rc} at {now()}", flush=True)

        # Best-effort CUDA cleanup between runs in a fresh process.
        _ = subprocess.run(
            [sys.executable, "-c", "import gc,torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )

        all_results[variant] = read_summary(summary_path)
        if rc != 0:
            all_results.setdefault(variant, {})
            all_results[variant]["_run_rc"] = rc

    compare: Dict[str, Dict] = {"meta": {}, "variants": {}}
    compare["meta"] = {
        "timestamp": now(),
        "variants": variants,
        "output_root": str(output_root),
        "eval_lengths": args.eval_lengths,
        "max_steps": args.max_steps,
        "seq_len": args.seq_len,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    for v in variants:
        s = all_results.get(v, {})
        eval_ppl = s.get("eval_ppl", {})
        compare["variants"][v] = {
            "train_loss": s.get("train", {}).get("train_loss"),
            "train_hours": s.get("train", {}).get("train_hours"),
            "ppl@16k": eval_ppl.get("16384", {}).get("ppl"),
            "ppl@32k": eval_ppl.get("32768", {}).get("ppl"),
            "ppl@64k": eval_ppl.get("65536", {}).get("ppl"),
            "raw_summary_path": str(output_root / v / "summary.json"),
            "run_rc": s.get("_run_rc", 0),
        }

    summary_json.write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding="utf-8")
    with suite_log.open("a", encoding="utf-8") as f:
        f.write(f"=== SUITE END {now()} summary={summary_json} ===\n")
    print(json.dumps(compare, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

