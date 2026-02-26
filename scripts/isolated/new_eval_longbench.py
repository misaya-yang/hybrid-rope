#!/usr/bin/env python3
"""
Isolated evaluator for the new LongAlpaca/LongQA experiment line.

Protocol lock:
- Base model fixed to Meta-Llama-3-8B-Instruct (8K-native).
- Full LongBench (lb21, all samples) + per-sample traces.
- NIAH + Passkey at multiple lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_MAX_POS = 8192
MODEL_LOCK_DEFAULT = Path("/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_seed_csv(text: str) -> List[int]:
    return [int(x) for x in parse_csv(text)]


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_model_lock(base_model_path: Path) -> Dict[str, object]:
    model_str = base_model_path.as_posix()
    low = model_str.lower()
    if MODEL_LOCK_NAME.lower() not in low:
        raise RuntimeError(
            "Model lock violation: expected Meta-Llama-3-8B-Instruct path, got "
            f"{model_str}"
        )
    if "3.1" in low or "128k" in low:
        raise RuntimeError("Model lock violation: Llama-3.1 / 128K model names are forbidden.")

    cfg_path = base_model_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    max_pos = int(cfg.get("max_position_embeddings", -1))
    if max_pos != MODEL_LOCK_MAX_POS:
        raise RuntimeError(
            f"Model lock violation: expected max_position_embeddings={MODEL_LOCK_MAX_POS}, got {max_pos}."
        )
    return {
        "base_model_path": model_str,
        "config_path": cfg_path.as_posix(),
        "config_sha256": sha256_file(cfg_path),
        "model_type": str(cfg.get("model_type", "")),
        "max_position_embeddings": max_pos,
    }


def find_run_dir(train_output_root: Path, method: str, seed: int, lora_rank: int, max_steps: int) -> Path:
    run_name = f"newds_{method}_seed{seed}_r{lora_rank}_s{max_steps}"
    run_dir = train_output_root / run_name
    if not run_dir.exists():
        raise RuntimeError(f"Missing training output dir: {run_dir}")
    return run_dir


def resolve_adapter_dir(run_dir: Path) -> Path:
    if (run_dir / "adapter_config.json").exists():
        return run_dir
    candidates = sorted(run_dir.glob("checkpoint-*"))
    for ckpt in reversed(candidates):
        if (ckpt / "adapter_config.json").exists():
            return ckpt
    raise RuntimeError(f"No adapter_config.json found under {run_dir}")


def resolve_custom_inv(run_dir: Path) -> Optional[Path]:
    p1 = run_dir / "artifacts" / "custom_inv_freq.pt"
    if p1.exists():
        return p1
    p2 = run_dir.parent / "artifacts" / "custom_inv_freq.pt"
    if p2.exists():
        return p2
    return None


def run_cmd(cmd: Sequence[str], cwd: Path, dry_run: bool) -> int:
    if dry_run:
        print("DRY RUN:", " ".join(cmd), flush=True)
        return 0
    p = subprocess.run(list(cmd), cwd=cwd.as_posix())
    return int(p.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate new dataset runs on LongBench/NIAH/Passkey.")
    parser.add_argument("--base_model_path", type=Path, default=MODEL_LOCK_DEFAULT)
    parser.add_argument("--train_output_root", type=Path, default=Path("artifacts/new_dataset_v1/runs"))
    parser.add_argument("--eval_output_root", type=Path, default=Path("artifacts/new_dataset_v1/eval"))
    parser.add_argument("--manifest_json", type=Path, default=Path("artifacts/new_dataset_v1/eval_manifest.json"))
    parser.add_argument("--methods", type=str, default="baseline,anchored_sigmoid")
    parser.add_argument("--seeds", type=str, default="42,1337")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--longbench_local_data_dir", type=str, default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    model_info = assert_model_lock(args.base_model_path)
    methods = parse_csv(args.methods)
    seeds = parse_seed_csv(args.seeds)
    if not methods or not seeds:
        raise RuntimeError("methods/seeds cannot be empty.")

    args.eval_output_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for method in methods:
        if method not in {"baseline", "anchored_sigmoid"}:
            raise RuntimeError(f"Unsupported method: {method}")
        for seed in seeds:
            run_dir = find_run_dir(
                train_output_root=args.train_output_root,
                method=method,
                seed=seed,
                lora_rank=args.lora_rank,
                max_steps=args.max_steps,
            )
            adapter_dir = resolve_adapter_dir(run_dir)
            custom_inv = resolve_custom_inv(run_dir)
            run_key = f"{method}_seed{seed}"
            out_dir = args.eval_output_root / run_key
            out_dir.mkdir(parents=True, exist_ok=True)

            manifest_dir = out_dir / "repro_manifest"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            lb_json = out_dir / "longbench_lb21.json"
            lb_manifest = out_dir / "longbench_manifest.json"

            lb_cmd = [
                sys.executable,
                "scripts/eval_longbench.py",
                "--base_model_path",
                str(args.base_model_path),
                "--adapter_path",
                str(adapter_dir),
                "--model_alias",
                run_key,
                "--skip_base_unfinetuned",
                "--task_set",
                "lb21",
                "--max_samples_per_task",
                "0",
                "--max_input_tokens",
                str(args.ctx),
                "--batch_size",
                str(args.batch_size),
                "--prompt_source",
                "official",
                "--chat_template",
                "auto",
                "--truncate_mode",
                "middle",
                "--max_new_tokens_policy",
                "official",
                "--score_scale",
                "pct",
                "--strict_parity_check",
                "--seed",
                str(seed),
                "--longbench_local_data_dir",
                str(args.longbench_local_data_dir),
                "--save_per_sample_traces",
                "1",
                "--trace_output_max_chars",
                "1024",
                "--manifest_json",
                str(lb_manifest),
                "--repro_manifest_dir",
                str(manifest_dir),
                "--attn_implementation",
                str(args.attn_implementation),
                "--output_json",
                str(lb_json),
            ]
            if custom_inv is not None:
                lb_cmd.extend(["--custom_inv_freq_path", str(custom_inv)])

            niah_dir = out_dir / "niah"
            niah_cmd = [
                sys.executable,
                "scripts/eval_niah_recall.py",
                "--base_model_path",
                str(args.base_model_path),
                "--adapter_path",
                str(adapter_dir),
                "--variant",
                "custom" if custom_inv else "base",
                "--output_dir",
                str(niah_dir),
                "--lengths",
                "4096,8192,16384,32768",
                "--depths",
                "10,25,50,75,90",
                "--trials_per_cell",
                "1",
                "--seed",
                str(seed),
                "--attn_implementation",
                str(args.attn_implementation),
            ]
            if custom_inv is not None:
                niah_cmd.extend(["--custom_inv_freq_path", str(custom_inv)])

            passkey_dir = out_dir / "passkey"
            passkey_cmd = [
                sys.executable,
                "scripts/eval_passkey_teacher_forcing.py",
                "--base_model_path",
                str(args.base_model_path),
                "--adapter_path",
                str(adapter_dir),
                "--variant",
                "custom" if custom_inv else "base",
                "--output_dir",
                str(passkey_dir),
                "--lengths",
                "4096,8192,16384,32768",
                "--depths",
                "10,50,90",
                "--trials_per_cell",
                "20",
                "--seed",
                str(seed),
                "--attn_implementation",
                str(args.attn_implementation),
            ]
            if custom_inv is not None:
                passkey_cmd.extend(["--custom_inv_freq_path", str(custom_inv)])

            rc_lb = run_cmd(lb_cmd, cwd=repo_root, dry_run=args.dry_run)
            rc_niah = run_cmd(niah_cmd, cwd=repo_root, dry_run=args.dry_run)
            rc_passkey = run_cmd(passkey_cmd, cwd=repo_root, dry_run=args.dry_run)

            records.append(
                {
                    "run_key": run_key,
                    "method": method,
                    "seed": seed,
                    "run_dir": run_dir.as_posix(),
                    "adapter_dir": adapter_dir.as_posix(),
                    "custom_inv_freq_path": custom_inv.as_posix() if custom_inv else "",
                    "return_codes": {
                        "longbench": rc_lb,
                        "niah": rc_niah,
                        "passkey": rc_passkey,
                    },
                    "outputs": {
                        "longbench_json": lb_json.as_posix(),
                        "niah_dir": niah_dir.as_posix(),
                        "passkey_dir": passkey_dir.as_posix(),
                    },
                }
            )

    manifest = {
        "timestamp": now(),
        "script": Path(__file__).name,
        "model_lock": model_info,
        "settings": {
            "ctx": int(args.ctx),
            "batch_size": int(args.batch_size),
            "attn_implementation": args.attn_implementation,
            "task_set": "lb21",
            "full_samples": True,
            "save_per_sample_traces": True,
        },
        "code_hashes": {
            "new_eval_longbench.py": sha256_file(Path(__file__)),
            "scripts/eval_longbench.py": sha256_file(repo_root / "scripts" / "eval_longbench.py"),
            "scripts/eval_niah_recall.py": sha256_file(repo_root / "scripts" / "eval_niah_recall.py"),
            "scripts/eval_passkey_teacher_forcing.py": sha256_file(repo_root / "scripts" / "eval_passkey_teacher_forcing.py"),
        },
        "runs": records,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {args.manifest_json}", flush=True)


if __name__ == "__main__":
    main()
