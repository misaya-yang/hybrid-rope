#!/usr/bin/env python3
"""
Plan B evaluation launcher for strict, protocol-locked comparison.

This script is isolated from currently running training jobs.
It orchestrates:
- LongBench full set (lb21, full samples) with official-parity controls
- NIAH multi-length recall
- Passkey teacher-forcing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_POS_EMB = 8192
FORBIDDEN_MODEL_TOKENS = ("llama-3.1", "128k")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_seed_csv(raw: str) -> List[int]:
    return [int(x) for x in parse_csv(raw)]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_model_integrity(base_model_path: Path) -> Dict[str, object]:
    p = base_model_path.as_posix()
    p_low = p.lower()
    if MODEL_LOCK_NAME.lower() not in p_low:
        raise RuntimeError(
            "Model integrity violation: expected base model path containing "
            f"'{MODEL_LOCK_NAME}', got '{p}'."
        )
    for token in FORBIDDEN_MODEL_TOKENS:
        if token in p_low:
            raise RuntimeError(
                "Model integrity violation: forbidden model token detected in path "
                f"('{token}') for '{p}'."
            )
    cfg_path = base_model_path / "config.json"
    cfg = read_json(cfg_path)
    max_pos = int(cfg.get("max_position_embeddings", -1))
    if max_pos != MODEL_LOCK_POS_EMB:
        raise RuntimeError(
            "Model integrity violation: expected max_position_embeddings=8192 "
            f"for 8K-native Llama-3-8B, got {max_pos}."
        )
    return {
        "base_model_path": p,
        "config_json": cfg_path.as_posix(),
        "model_type": str(cfg.get("model_type", "")),
        "max_position_embeddings": max_pos,
    }


def find_adapter_dir(runs_root: Path, method: str, seed: int) -> Path:
    run_name = f"planb_llama3_8b_{method}_{seed}"
    run_dir = runs_root / run_name
    if not run_dir.exists():
        raise RuntimeError(f"Missing run dir: {run_dir}")
    adapter_cfg = run_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        checkpoints = sorted(run_dir.glob("checkpoint-*"))
        if checkpoints:
            latest = checkpoints[-1]
            if (latest / "adapter_config.json").exists():
                return latest
    if not adapter_cfg.exists():
        raise RuntimeError(
            f"Missing adapter_config.json under {run_dir}. "
            "Run Plan B training first or point --runs_root to completed artifacts."
        )
    return run_dir


def maybe_custom_inv_freq(adapter_dir: Path, method: str) -> Optional[Path]:
    if method != "anchored_sigmoid":
        return None
    direct = adapter_dir / "artifacts" / "custom_inv_freq.pt"
    if direct.exists():
        return direct
    parent = adapter_dir.parent / "artifacts" / "custom_inv_freq.pt"
    if parent.exists():
        return parent
    return None


def run_cmd(cmd: Sequence[str], cwd: Path, dry_run: bool) -> int:
    if dry_run:
        print("DRY RUN:", " ".join(cmd), flush=True)
        return 0
    result = subprocess.run(list(cmd), cwd=cwd.as_posix())
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan B evaluator (LongBench/NIAH/Passkey).")
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=Path("/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument("--runs_root", type=Path, default=Path("artifacts/plan_b_runs"))
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/plan_b_eval"))
    parser.add_argument("--methods", type=str, default="baseline,anchored_sigmoid")
    parser.add_argument("--seeds", type=str, default="42,1337")
    parser.add_argument("--longbench_local_data_dir", type=str, default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_info = assert_model_integrity(args.base_model_path)
    methods = parse_csv(args.methods)
    seeds = parse_seed_csv(args.seeds)
    if not methods:
        raise RuntimeError("No methods selected for evaluation.")
    if not seeds:
        raise RuntimeError("No seeds selected for evaluation.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": now(),
        "script": Path(__file__).name,
        "base_model": model_info,
        "methods": methods,
        "seeds": seeds,
        "runs": [],
    }

    for method in methods:
        for seed in seeds:
            adapter_dir = find_adapter_dir(args.runs_root, method, seed)
            custom_inv = maybe_custom_inv_freq(adapter_dir=adapter_dir, method=method)
            run_id = f"{method}_seed{seed}"
            out_dir = args.output_root / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest_dir = out_dir / "repro_manifest"
            manifest_dir.mkdir(parents=True, exist_ok=True)

            longbench_json = out_dir / "longbench_lb21.json"
            longbench_manifest = out_dir / "longbench_manifest.json"
            longbench_cmd = [
                sys.executable,
                "scripts/eval_longbench.py",
                "--base_model_path",
                str(args.base_model_path),
                "--adapter_path",
                str(adapter_dir),
                "--model_alias",
                run_id,
                "--skip_base_unfinetuned",
                "--task_set",
                "lb21",
                "--max_samples_per_task",
                "-1",
                "--score_scale",
                "pct",
                "--prompt_source",
                "official",
                "--chat_template",
                "on",
                "--truncate_mode",
                "middle",
                "--max_new_tokens_policy",
                "official",
                "--strict_parity_check",
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(seed),
                "--longbench_local_data_dir",
                str(args.longbench_local_data_dir),
                "--save_per_sample_traces",
                "1",
                "--output_json",
                str(longbench_json),
                "--manifest_json",
                str(longbench_manifest),
                "--repro_manifest_dir",
                str(manifest_dir),
                "--attn_implementation",
                str(args.attn_implementation),
            ]
            if custom_inv is not None:
                longbench_cmd.extend(["--custom_inv_freq_path", str(custom_inv)])

            niah_dir = out_dir / "niah"
            niah_cmd = [
                sys.executable,
                "scripts/eval_niah_recall.py",
                "--base_model_path",
                str(args.base_model_path),
                "--adapter_path",
                str(adapter_dir),
                "--variant",
                "custom" if custom_inv is not None else "base",
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
                "custom" if custom_inv is not None else "base",
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

            rc_longbench = run_cmd(longbench_cmd, cwd=repo_root, dry_run=args.dry_run)
            rc_niah = run_cmd(niah_cmd, cwd=repo_root, dry_run=args.dry_run)
            rc_passkey = run_cmd(passkey_cmd, cwd=repo_root, dry_run=args.dry_run)

            manifest["runs"].append(
                {
                    "run_id": run_id,
                    "method": method,
                    "seed": seed,
                    "adapter_dir": adapter_dir.as_posix(),
                    "custom_inv_freq_path": custom_inv.as_posix() if custom_inv is not None else "",
                    "outputs": {
                        "longbench_json": longbench_json.as_posix(),
                        "niah_dir": niah_dir.as_posix(),
                        "passkey_dir": passkey_dir.as_posix(),
                    },
                    "return_codes": {
                        "longbench": rc_longbench,
                        "niah": rc_niah,
                        "passkey": rc_passkey,
                    },
                }
            )

    manifest_path = args.output_root / "plan_b_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
