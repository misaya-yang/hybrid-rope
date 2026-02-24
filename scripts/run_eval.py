#!/usr/bin/env python3
"""Unified runner for plan-v2 E1/E2 evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rope.schedules import (
    default_shape_params,
    infer_case,
    infer_rope_base_from_config,
    infer_shape_name,
)


METHOD_DIR_MAP = {
    "baseline": "baseline",
    "baseline_native": "baseline",
    "a_baseline": "baseline",
    "c_base_only": "baseline",
    "pi": "pi",
    "yarn": "yarn",
    "sigmoid": "sigmoid",
    "anchored_sigmoid": "anchored_sigmoid",
    "hybrid": "anchored_sigmoid",
    "b_shape_only": "anchored_sigmoid",
    "d_full_hybrid": "anchored_sigmoid",
}

METHOD_VARIANT_MAP = {
    "baseline": "base",
    "baseline_native": "base",
    "a_baseline": "base",
    "c_base_only": "base",
    "pi": "pi",
    "yarn": "yarn",
    "sigmoid": "custom",
    "anchored_sigmoid": "custom",
    "hybrid": "custom",
    "b_shape_only": "custom",
    "d_full_hybrid": "custom",
}


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_name(x: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in x).strip("_")


def resolve_adapter_and_inv(
    method: str,
    suite_output_root: Path,
    adapter_override: str,
    custom_inv_override: str,
) -> Dict[str, str]:
    if adapter_override:
        adapter_path = Path(adapter_override)
        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter_override not found: {adapter_path}")
        run_dir = adapter_path.resolve().parent
    else:
        key = method.strip().lower()
        if key not in METHOD_DIR_MAP:
            raise ValueError(
                f"Unknown method '{method}'. Provide --adapter_override for custom method names."
            )
        run_dir = suite_output_root / METHOD_DIR_MAP[key]
        adapter_path = run_dir / "final_lora"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path missing: {adapter_path}")

    custom_inv: Optional[Path] = None
    if custom_inv_override:
        candidate = Path(custom_inv_override)
        if not candidate.exists():
            raise FileNotFoundError(f"custom_inv_freq_path not found: {candidate}")
        if not candidate.is_file():
            raise IsADirectoryError(f"custom_inv_freq_path must be a file: {candidate}")
        custom_inv = candidate.resolve()
    else:
        candidate = run_dir / "artifacts" / "custom_inv_freq.pt"
        if candidate.exists() and candidate.is_file():
            custom_inv = candidate.resolve()

    return {
        "run_dir": str(run_dir),
        "adapter_path": str(adapter_path),
        "custom_inv_freq_path": str(custom_inv) if custom_inv else "",
    }


def run_cmd(cmd: List[str], cwd: Path, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
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


def maybe_load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def json_append(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def collect_git_state(repo_root: Path) -> str:
    cmds = [
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        ["git", "status", "--short"],
    ]
    out: List[str] = []
    for cmd in cmds:
        p = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        title = " ".join(cmd)
        out.append(f"$ {title}\n{p.stdout}{p.stderr}".rstrip())
    return "\n\n".join(out) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan-v2 experiment runner (E1/E2).")
    ap.add_argument("--exp", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, help="Base checkpoint path.")
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--suite", type=str, default="ppl,longbench_full,needle")
    ap.add_argument(
        "--method_base",
        type=float,
        default=0.0,
        help="Optional method rope base override for Case-B runs.",
    )
    ap.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument(
        "--suite_output_root",
        type=str,
        default="results/llama8b_fair_v2_longbench_stable_20260223_0150",
    )
    ap.add_argument("--adapter_override", type=str, default="")
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--run_root", type=str, default="runs")
    ap.add_argument("--registry", type=str, default="artifacts/registry.jsonl")
    ap.add_argument("--manifest_root", type=str, default="artifacts/manifests")
    ap.add_argument(
        "--longbench_tasks",
        type=str,
        default="qasper,hotpotqa,2wikimqa,multi_news,gov_report,narrativeqa",
    )
    ap.add_argument("--longbench_max_samples", type=int, default=80)
    ap.add_argument("--ppl_max_chunks", type=int, default=20)
    ap.add_argument("--needle_depths", type=str, default="10,50,90")
    ap.add_argument("--needle_trials_per_cell", type=int, default=24)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--text_path", type=str, default="/root/autodl-tmp/data/long_text.txt")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    suite_root = Path(args.suite_output_root)
    if not suite_root.is_absolute():
        suite_root = (repo_root / suite_root).resolve()

    suites = [s.strip().lower() for s in parse_csv(args.suite)]
    date_tag = time.strftime("%Y-%m-%d")
    model_tag = safe_name(Path(args.model).name or "model")
    method_tag = safe_name(args.method)
    ctx_tag = f"{int(args.ctx // 1024)}k" if args.ctx % 1024 == 0 else str(args.ctx)
    run_id = f"{date_tag}_{safe_name(args.exp)}_{model_tag}_{ctx_tag}_{method_tag}_{args.seed}"
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (repo_root / run_root).resolve()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_text(run_dir / "git_state.txt", collect_git_state(repo_root))

    resolved = resolve_adapter_and_inv(
        method=args.method,
        suite_output_root=suite_root,
        adapter_override=args.adapter_override,
        custom_inv_override=args.custom_inv_freq_path,
    )
    model_base = infer_rope_base_from_config(args.model, fallback=500000.0)
    method_base = float(args.method_base) if float(args.method_base) > 0 else float(model_base)
    shape = infer_shape_name(args.method)
    case_tag, _ = infer_case(args.method, base=method_base, model_base=model_base)
    inv_sha = sha256_file(Path(resolved["custom_inv_freq_path"])) if resolved["custom_inv_freq_path"] else None

    if METHOD_VARIANT_MAP.get(args.method.strip().lower(), "custom") == "custom" and not inv_sha:
        raise RuntimeError(
            f"Method '{args.method}' requires custom_inv_freq.pt, but none was found. "
            "Use --custom_inv_freq_path or --adapter_override."
        )

    config_yaml = f"""model:
  checkpoint: "{args.model}"
  dtype: "bf16"
  attn_impl: "{args.attn_implementation}"
rope:
  method: "{args.method}"
  base: {method_base}
  shape: "{shape}"
eval:
  ctx: {args.ctx}
  batch_size: 1
  seed: {args.seed}
  suite: {json.dumps(suites)}
logging:
  out_dir: "{run_dir}"
  registry: "{args.registry}"
"""
    write_text(run_dir / "config.yaml", config_yaml)

    rope_params = {
        "method": args.method,
        "base": float(method_base),
        "model_base": float(model_base),
        "shape": shape,
        "shape_params": default_shape_params(args.method, base=method_base, max_seq_len=args.ctx),
        "inv_freq_sha256": inv_sha,
        "adapter_path": resolved["adapter_path"],
        "custom_inv_freq_path": resolved["custom_inv_freq_path"],
        "case": case_tag,
        "notes": args.notes,
    }
    write_text(run_dir / "rope_params.json", json.dumps(rope_params, indent=2, ensure_ascii=False))

    manifest_root = Path(args.manifest_root)
    if not manifest_root.is_absolute():
        manifest_root = (repo_root / manifest_root).resolve()
    manifest_path = manifest_root / f"longbench_manifest_ctx{args.ctx}_seed{args.seed}.json"

    metrics_jsonl = run_dir / "metrics.jsonl"
    stdout_log = run_dir / "stdout.log"
    py = sys.executable
    stage_rc: Dict[str, int] = {}
    summary: Dict[str, object] = {
        "run_id": run_id,
        "timestamp": now(),
        "exp": args.exp,
        "ctx": args.ctx,
        "method": args.method,
        "seed": args.seed,
        "suites": suites,
        "paths": {
            "run_dir": str(run_dir),
            "adapter_path": resolved["adapter_path"],
            "custom_inv_freq_path": resolved["custom_inv_freq_path"],
            "manifest_path": str(manifest_path),
        },
        "metrics": {},
    }

    if args.dry_run:
        stage_rc = {s: 0 for s in suites}
    else:
        if "ppl" in suites:
            ppl_json = run_dir / "ppl.json"
            cmd = [
                py,
                str((repo_root / "eval/ppl/eval_ppl.py").resolve()),
                "--base_model_path",
                args.model,
                "--adapter_path",
                resolved["adapter_path"],
                "--variant",
                METHOD_VARIANT_MAP.get(args.method.strip().lower(), "custom"),
                "--ctx",
                str(args.ctx),
                "--max_chunks",
                str(args.ppl_max_chunks),
                "--seed",
                str(args.seed),
                "--attn_implementation",
                args.attn_implementation,
                "--text_path",
                args.text_path,
                "--output_json",
                str(ppl_json),
            ]
            if resolved["custom_inv_freq_path"]:
                cmd.extend(["--custom_inv_freq_path", resolved["custom_inv_freq_path"]])
            rc = run_cmd(cmd, cwd=repo_root, log_file=stdout_log)
            stage_rc["ppl"] = rc
            obj = maybe_load_json(ppl_json) or {}
            result = obj.get("result", {})
            ppl = result.get("ppl")
            summary["metrics"]["ppl"] = ppl
            json_append(
                metrics_jsonl,
                {
                    "stage": "ppl",
                    "rc": rc,
                    "output_json": str(ppl_json),
                    "ppl": ppl,
                },
            )

        if "longbench_full" in suites:
            lb_json = run_dir / "longbench.json"
            cmd = [
                py,
                str((repo_root / "scripts/eval_longbench.py").resolve()),
                "--base_model_path",
                args.model,
                "--hybrid_adapter_path",
                resolved["adapter_path"],
                "--output_json",
                str(lb_json),
                "--tasks",
                args.longbench_tasks,
                "--max_samples_per_task",
                str(args.longbench_max_samples),
                "--max_input_tokens",
                str(args.ctx),
                "--seed",
                str(args.seed),
                "--attn_implementation",
                args.attn_implementation,
                "--manifest_json",
                str(manifest_path),
            ]
            if resolved["custom_inv_freq_path"]:
                cmd.extend(["--variant", "custom", "--custom_inv_freq_path", resolved["custom_inv_freq_path"]])
            else:
                cmd.extend(["--variant", METHOD_VARIANT_MAP.get(args.method.strip().lower(), "base")])
            rc = run_cmd(cmd, cwd=repo_root, log_file=stdout_log)
            stage_rc["longbench_full"] = rc

            obj = maybe_load_json(lb_json) or {}
            tasks = parse_csv(args.longbench_tasks)
            vals = []
            for t in tasks:
                v = (
                    obj.get("models", {})
                    .get("hybrid_lora", {})
                    .get("tasks", {})
                    .get(t, {})
                    .get("score")
                )
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            lb_avg = float(sum(vals) / len(vals)) if vals else None
            summary["metrics"]["longbench_avg"] = lb_avg
            json_append(
                metrics_jsonl,
                {
                    "stage": "longbench_full",
                    "rc": rc,
                    "output_json": str(lb_json),
                    "longbench_avg": lb_avg,
                    "tasks": tasks,
                },
            )

        if "needle" in suites:
            needle_dir = run_dir / "needle"
            cmd = [
                py,
                str((repo_root / "scripts/eval_passkey_teacher_forcing.py").resolve()),
                "--base_model_path",
                args.model,
                "--adapter_path",
                resolved["adapter_path"],
                "--output_dir",
                str(needle_dir),
                "--lengths",
                str(args.ctx),
                "--depths",
                args.needle_depths,
                "--trials_per_cell",
                str(args.needle_trials_per_cell),
                "--attn_implementation",
                args.attn_implementation,
                "--seed",
                str(args.seed),
            ]
            if resolved["custom_inv_freq_path"]:
                cmd.extend(["--variant", "custom", "--custom_inv_freq_path", resolved["custom_inv_freq_path"]])
            else:
                cmd.extend(["--variant", METHOD_VARIANT_MAP.get(args.method.strip().lower(), "base")])
            rc = run_cmd(cmd, cwd=repo_root, log_file=stdout_log)
            stage_rc["needle"] = rc

            obj = maybe_load_json(needle_dir / "passkey_tf_summary.json") or {}
            m = (obj.get("by_length", {}) or {}).get(str(args.ctx), {})
            needle_acc = m.get("tf_accuracy")
            needle_margin = m.get("margin_mean")
            summary["metrics"]["needle_tf_accuracy"] = needle_acc
            summary["metrics"]["needle_margin"] = needle_margin
            json_append(
                metrics_jsonl,
                {
                    "stage": "needle",
                    "rc": rc,
                    "summary_json": str(needle_dir / "passkey_tf_summary.json"),
                    "tf_accuracy": needle_acc,
                    "margin_mean": needle_margin,
                },
            )

    status = "valid" if all(stage_rc.get(s, 1) == 0 for s in suites) else "invalid"
    summary["status"] = status
    summary["stage_rc"] = stage_rc
    write_text(run_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = (repo_root / registry_path).resolve()
    registry_row = {
        "run_id": run_id,
        "exp": args.exp,
        "model": args.model,
        "ctx": args.ctx,
        "method": args.method,
        "seed": args.seed,
        "status": status,
        "rope_base": method_base,
        "model_base": model_base,
        "rope_shape": shape,
        "inv_freq_sha256": inv_sha,
        "notes": args.notes,
        "summary_json": str(run_dir / "summary.json"),
        "manifest_json": str(manifest_path),
    }
    json_append(registry_path, registry_row)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
