#!/usr/bin/env python3
"""
Post-training evaluation pipeline for fair 8B LoRA variants.

What it does:
1) (Optional) wait until fair training suite finishes.
2) Run NIAH recall (single + multi needle) for base and each available adapter.
3) Run LongBench comparison (base vs each adapter).
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


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(v: str) -> List[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
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
            print(line.rstrip("\n"), flush=True)
            f.write(line)
        rc = proc.wait()
        f.write(f"[{now()}] RC={rc}\n")
    return rc


def suite_running() -> bool:
    # Use `pgrep -af` and filter out the probe command itself.
    # The previous shell-based probe could match its own argv and loop forever.
    p = subprocess.run(
        ["pgrep", "-af", "run_llama8b_fair_suite.py"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return False
    for line in p.stdout.splitlines():
        if "pgrep -af run_llama8b_fair_suite.py" in line:
            continue
        if "run_llama8b_fair_suite.py" in line:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run post-training evals (NIAH + LongBench) for 8B fair suite.")
    ap.add_argument("--repo_root", type=str, default="/root/autodl-tmp/dfrope/hybrid-rope")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--suite_output_root",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_lora_suite_20260214",
    )
    ap.add_argument(
        "--post_eval_root",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_post_eval_20260214",
    )
    ap.add_argument("--variants", type=str, default="yarn,pi,hybrid,pi_soft")
    ap.add_argument("--wait_for_suite", action="store_true")
    ap.add_argument("--poll_seconds", type=int, default=120)
    ap.add_argument("--skip_niah", action="store_true", help="Skip NIAH evaluations.")
    ap.add_argument("--skip_longbench", action="store_true", help="Skip LongBench evaluations.")
    ap.add_argument("--niah_lengths", type=str, default="4096,8192,16384,32768")
    ap.add_argument("--niah_depths", type=str, default="0,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--niah_trials_per_cell", type=int, default=1)
    ap.add_argument("--niah_multi_needles", type=int, default=4)
    ap.add_argument("--longbench_tasks", type=str, default="qasper,hotpotqa,gov_report")
    ap.add_argument("--longbench_max_samples", type=int, default=100)
    ap.add_argument("--longbench_max_input_tokens", type=int, default=16384)
    ap.add_argument(
        "--longbench_local_data_dir",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data",
        help="Local LongBench jsonl directory for fully offline evaluation.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    suite_root = Path(args.suite_output_root)
    post_root = Path(args.post_eval_root)
    post_root.mkdir(parents=True, exist_ok=True)
    manifest_path = post_root / "post_eval_manifest.json"
    master_log = post_root / "post_eval.log"

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.wait_for_suite:
        while suite_running():
            msg = f"[{now()}] Waiting for fair suite to finish..."
            print(msg, flush=True)
            with master_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            time.sleep(max(5, args.poll_seconds))

    variants = parse_csv(args.variants)
    adapters: Dict[str, Path] = {}
    for v in variants:
        p = suite_root / v / "final_lora"
        if p.exists():
            adapters[v] = p

    manifest: Dict = {
        "meta": {
            "timestamp": now(),
            "suite_output_root": str(suite_root),
            "post_eval_root": str(post_root),
            "variants_requested": variants,
            "adapters_found": {k: str(v) for k, v in adapters.items()},
        },
        "niah": {},
        "longbench": {},
    }

    run_niah = not args.skip_niah
    run_longbench = not args.skip_longbench

    # NIAH evaluations
    if run_niah:
        # base single/multi
        for needles in [1, args.niah_multi_needles]:
            tag = f"base_needles{needles}"
            out_dir = post_root / "niah" / tag
            cmd = [
                sys.executable,
                "scripts/eval_niah_recall.py",
                "--base_model_path",
                args.base_model_path,
                "--base_only",
                "--output_dir",
                str(out_dir),
                "--lengths",
                args.niah_lengths,
                "--depths",
                args.niah_depths,
                "--trials_per_cell",
                str(args.niah_trials_per_cell),
                "--needles_per_prompt",
                str(needles),
                "--attn_implementation",
                "sdpa",
                "--seed",
                str(args.seed),
            ]
            rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=post_root / "logs" / f"niah_{tag}.log")
            manifest["niah"][tag] = {"rc": rc, "output_dir": str(out_dir)}

        # variant single/multi
        for v, apath in adapters.items():
            for needles in [1, args.niah_multi_needles]:
                tag = f"{v}_needles{needles}"
                out_dir = post_root / "niah" / tag
                cmd = [
                    sys.executable,
                    "scripts/eval_niah_recall.py",
                    "--base_model_path",
                    args.base_model_path,
                    "--adapter_path",
                    str(apath),
                    "--output_dir",
                    str(out_dir),
                    "--lengths",
                    args.niah_lengths,
                    "--depths",
                    args.niah_depths,
                    "--trials_per_cell",
                    str(args.niah_trials_per_cell),
                    "--needles_per_prompt",
                    str(needles),
                    "--attn_implementation",
                    "sdpa",
                    "--seed",
                    str(args.seed),
                ]
                rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=post_root / "logs" / f"niah_{tag}.log")
                manifest["niah"][tag] = {"rc": rc, "output_dir": str(out_dir), "adapter": str(apath)}

    # LongBench comparisons: base vs each adapter
    if run_longbench:
        for v, apath in adapters.items():
            out_json = post_root / "longbench" / f"longbench_base_vs_{v}.json"
            cmd = [
                sys.executable,
                "scripts/eval_longbench.py",
                "--base_model_path",
                args.base_model_path,
                "--hybrid_adapter_path",
                str(apath),
                "--tasks",
                args.longbench_tasks,
                "--max_samples_per_task",
                str(args.longbench_max_samples),
                "--max_input_tokens",
                str(args.longbench_max_input_tokens),
                "--longbench_local_data_dir",
                str(args.longbench_local_data_dir),
                "--attn_implementation",
                "sdpa",
                "--seed",
                str(args.seed),
                "--output_json",
                str(out_json),
            ]
            rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=post_root / "logs" / f"longbench_{v}.log")
            manifest["longbench"][v] = {"rc": rc, "output_json": str(out_json), "adapter": str(apath)}

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
