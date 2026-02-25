#!/usr/bin/env python3
"""
Unified downstream evaluation runner for fair 8B SOTA comparison.

Runs:
1) NIAH recall heatmap
2) LongBench subset
3) Passkey teacher-forcing (true-vs-false NLL)

Then aggregates paper-ready tables/figures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def json_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        _ = json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def suite_running() -> bool:
    p = subprocess.run(["pgrep", "-af", "run_llama8b_fair_suite.py"], capture_output=True, text=True)
    if p.returncode != 0:
        return False
    for line in p.stdout.splitlines():
        if "pgrep -af run_llama8b_fair_suite.py" in line:
            continue
        if "run_llama8b_fair_suite.py" in line:
            return True
    return False


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


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mean_niah_accuracy(niah_json: Dict) -> Optional[float]:
    mat = niah_json.get("accuracy_matrix")
    if not isinstance(mat, dict):
        return None
    vals: List[float] = []
    for _, row in mat.items():
        if isinstance(row, dict):
            for _, v in row.items():
                try:
                    vals.append(float(v))
                except Exception:
                    pass
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def longbench_task_scores(longbench_json: Dict) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    tasks = (
        longbench_json.get("models", {})
        .get("hybrid_lora", {})
        .get("tasks", {})
    )
    if not isinstance(tasks, dict):
        return out
    for t, payload in tasks.items():
        if isinstance(payload, dict):
            try:
                out[str(t)] = float(payload.get("score"))
            except Exception:
                out[str(t)] = None
    return out


def mean_scores(d: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.6 + 1.2 * len(df)), 4.8))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(df[y].tolist()):
        if pd.notna(v):
            ax.text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SOTA downstream eval and build paper tables/figures.")
    ap.add_argument("--repo_root", type=str, default="/root/autodl-tmp/dfrope/hybrid-rope")
    ap.add_argument("--python_bin", type=str, default="/root/miniconda3/bin/python")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--suite_output_root",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_v2_longbench_stable_20260223_0150",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="baseline,pi,yarn,sigmoid,anchored_sigmoid",
    )
    ap.add_argument(
        "--eval_root",
        type=str,
        default="",
        help="Default: <suite_output_root>/downstream_eval_<timestamp>",
    )
    ap.add_argument("--wait_for_training", action="store_true")
    ap.add_argument("--poll_seconds", type=int, default=90)

    ap.add_argument("--skip_niah", action="store_true")
    ap.add_argument("--skip_longbench", action="store_true")
    ap.add_argument("--skip_passkey_tf", action="store_true")

    ap.add_argument("--niah_lengths", type=str, default="4096,8192,16384")
    ap.add_argument("--niah_depths", type=str, default="0,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--niah_trials_per_cell", type=int, default=2)
    ap.add_argument("--niah_needles_per_prompt", type=int, default=1)
    ap.add_argument("--niah_prompt_mode", type=str, default="qa", choices=["qa", "continuation"])

    ap.add_argument(
        "--longbench_tasks",
        type=str,
        default="",
        help="Explicit CSV task override. Empty string follows --longbench_task_set.",
    )
    ap.add_argument("--longbench_task_set", type=str, default="lb6", choices=["lb6", "lb21"])
    ap.add_argument("--longbench_max_samples", type=int, default=80)
    ap.add_argument("--longbench_max_input_tokens", type=int, default=16384)
    ap.add_argument(
        "--longbench_batch_size",
        type=int,
        default=1,
        help="Batch size for eval_longbench greedy generation. Increase to use more VRAM and reduce wall time.",
    )
    ap.add_argument(
        "--longbench_prompt_source",
        type=str,
        default="official",
        choices=["official", "legacy"],
    )
    ap.add_argument(
        "--longbench_chat_template",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
    )
    ap.add_argument(
        "--longbench_truncate_mode",
        type=str,
        default="middle",
        choices=["tail", "middle"],
    )
    ap.add_argument(
        "--longbench_max_new_tokens_policy",
        type=str,
        default="official",
        choices=["official", "manual"],
    )
    ap.add_argument("--strict_parity_check", action="store_true")
    ap.add_argument(
        "--longbench_repro_manifest_dir",
        type=str,
        default="",
        help="Optional root directory for per-method baseline_gold/env_freeze/code_hash outputs.",
    )
    ap.add_argument(
        "--manifest_root",
        type=str,
        default="artifacts/manifests",
        help="Shared manifest root for paired LongBench evaluation indices.",
    )
    ap.add_argument(
        "--longbench_local_data_dir",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data",
    )

    ap.add_argument("--passkey_lengths", type=str, default="1024,2048,4096,8192,16384")
    ap.add_argument("--passkey_depths", type=str, default="10,50,90")
    ap.add_argument("--passkey_trials_per_cell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable output-based resume; rerun all task stages even if outputs already exist.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    suite_root = Path(args.suite_output_root)
    if not suite_root.exists():
        raise FileNotFoundError(f"suite_output_root not found: {suite_root}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    eval_root = Path(args.eval_root) if args.eval_root else (suite_root / f"downstream_eval_{ts}")
    eval_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.wait_for_training:
        while suite_running():
            print(f"[{now()}] Waiting for training suite to finish...", flush=True)
            time.sleep(max(5, int(args.poll_seconds)))

    methods = parse_csv(args.methods)
    manifest_root = Path(args.manifest_root)
    if not manifest_root.is_absolute():
        manifest_root = (repo_root / manifest_root).resolve()
    manifest_path = manifest_root / f"longbench_manifest_ctx{int(args.longbench_max_input_tokens)}_seed{int(args.seed)}.json"
    manifest: Dict[str, object] = {
        "meta": {
            "timestamp": now(),
            "suite_output_root": str(suite_root),
            "eval_root": str(eval_root),
            "methods_requested": methods,
            "longbench_manifest_json": str(manifest_path),
            "resume_enabled": not bool(args.no_resume),
        },
        "methods": {},
    }

    for method in methods:
        run_dir = suite_root / method
        summary_path = run_dir / "summary.json"
        adapter_path = run_dir / "final_lora"
        custom_inv = run_dir / "artifacts" / "custom_inv_freq.pt"
        if not summary_path.exists() or not adapter_path.exists():
            manifest["methods"][method] = {
                "status": "skipped",
                "reason": "missing summary.json or final_lora",
                "run_dir": str(run_dir),
            }
            continue

        m_out = {
            "status": "started",
            "run_dir": str(run_dir),
            "summary_json": str(summary_path),
            "adapter_path": str(adapter_path),
            "custom_inv_freq_path": str(custom_inv) if custom_inv.exists() else "",
            "niah": {},
            "longbench": {},
            "passkey_tf": {},
        }
        manifest["methods"][method] = m_out

        common_custom_args: List[str] = []
        if custom_inv.exists():
            common_custom_args = ["--variant", "custom", "--custom_inv_freq_path", str(custom_inv)]

        rc_values: List[int] = []
        resume_enabled = not bool(args.no_resume)

        if not args.skip_niah:
            niah_out = eval_root / "niah" / method
            niah_json = niah_out / "niah_recall_results.json"
            skipped_existing = False
            cmd = [
                args.python_bin,
                "scripts/eval_niah_recall.py",
                "--base_model_path",
                args.base_model_path,
                "--adapter_path",
                str(adapter_path),
                "--output_dir",
                str(niah_out),
                "--lengths",
                args.niah_lengths,
                "--depths",
                args.niah_depths,
                "--trials_per_cell",
                str(args.niah_trials_per_cell),
                "--needles_per_prompt",
                str(args.niah_needles_per_prompt),
                "--prompt_mode",
                args.niah_prompt_mode,
                "--attn_implementation",
                "sdpa",
                "--seed",
                str(args.seed),
            ] + common_custom_args
            if resume_enabled and json_ready(niah_json):
                print(f"[{now()}] [resume] skip NIAH {method}: {niah_json}", flush=True)
                rc = 0
                skipped_existing = True
            else:
                rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=eval_root / "logs" / f"niah_{method}.log")
            rc_values.append(int(rc))
            m_out["niah"] = {
                "rc": rc,
                "output_dir": str(niah_out),
                "result_json": str(niah_json),
                "skipped_existing": skipped_existing,
            }

        if not args.skip_longbench:
            lb_out = eval_root / "longbench" / f"{method}.json"
            repro_manifest_dir = ""
            if args.longbench_repro_manifest_dir.strip():
                repro_manifest_dir = str((Path(args.longbench_repro_manifest_dir) / method).resolve())
            skipped_existing = False
            cmd = [
                args.python_bin,
                "scripts/eval_longbench.py",
                "--base_model_path",
                args.base_model_path,
                "--hybrid_adapter_path",
                str(adapter_path),
                "--output_json",
                str(lb_out),
                "--task_set",
                args.longbench_task_set,
                "--max_samples_per_task",
                str(args.longbench_max_samples),
                "--max_input_tokens",
                str(args.longbench_max_input_tokens),
                "--batch_size",
                str(int(args.longbench_batch_size)),
                "--longbench_local_data_dir",
                args.longbench_local_data_dir,
                "--attn_implementation",
                "sdpa",
                "--seed",
                str(args.seed),
                "--manifest_json",
                str(manifest_path),
                "--prompt_source",
                args.longbench_prompt_source,
                "--chat_template",
                args.longbench_chat_template,
                "--truncate_mode",
                args.longbench_truncate_mode,
                "--max_new_tokens_policy",
                args.longbench_max_new_tokens_policy,
            ] + common_custom_args
            if repro_manifest_dir:
                cmd.extend(["--repro_manifest_dir", repro_manifest_dir])
            if args.longbench_tasks.strip():
                cmd.extend(["--tasks", args.longbench_tasks.strip()])
            if args.strict_parity_check:
                cmd.append("--strict_parity_check")
            if resume_enabled and json_ready(lb_out):
                print(f"[{now()}] [resume] skip LongBench {method}: {lb_out}", flush=True)
                rc = 0
                skipped_existing = True
            else:
                rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=eval_root / "logs" / f"longbench_{method}.log")
            rc_values.append(int(rc))
            m_out["longbench"] = {
                "rc": rc,
                "output_json": str(lb_out),
                "manifest_json": str(manifest_path),
                "repro_manifest_dir": repro_manifest_dir,
                "skipped_existing": skipped_existing,
            }

        if not args.skip_passkey_tf:
            pk_out = eval_root / "passkey_tf" / method
            pk_summary = pk_out / "passkey_tf_summary.json"
            skipped_existing = False
            cmd = [
                args.python_bin,
                "scripts/eval_passkey_teacher_forcing.py",
                "--base_model_path",
                args.base_model_path,
                "--adapter_path",
                str(adapter_path),
                "--output_dir",
                str(pk_out),
                "--lengths",
                args.passkey_lengths,
                "--depths",
                args.passkey_depths,
                "--trials_per_cell",
                str(args.passkey_trials_per_cell),
                "--attn_implementation",
                "sdpa",
                "--seed",
                str(args.seed),
            ] + common_custom_args
            if resume_enabled and json_ready(pk_summary):
                print(f"[{now()}] [resume] skip Passkey-TF {method}: {pk_summary}", flush=True)
                rc = 0
                skipped_existing = True
            else:
                rc = run_cmd(cmd, cwd=repo_root, env=env, log_file=eval_root / "logs" / f"passkey_tf_{method}.log")
            rc_values.append(int(rc))
            m_out["passkey_tf"] = {
                "rc": rc,
                "output_dir": str(pk_out),
                "summary_json": str(pk_summary),
                "skipped_existing": skipped_existing,
            }

        m_out["status"] = "done" if all(x == 0 for x in rc_values) else "failed"

    # Build aggregate report.
    report_dir = eval_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    longbench_rows: List[Dict[str, object]] = []
    if args.longbench_tasks.strip():
        all_tasks = parse_csv(args.longbench_tasks)
    elif args.longbench_task_set == "lb21":
        all_tasks = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
    else:
        all_tasks = ["qasper", "hotpotqa", "2wikimqa", "multi_news", "gov_report", "narrativeqa"]

    for method, info in manifest["methods"].items():
        if not isinstance(info, dict):
            continue
        if info.get("status") == "skipped":
            continue
        run_dir = Path(str(info["run_dir"]))
        train_summary = load_json(run_dir / "summary.json") or {}
        niah_json = load_json(Path(str(info.get("niah", {}).get("result_json", "")))) if isinstance(info.get("niah"), dict) else None
        longbench_json = load_json(Path(str(info.get("longbench", {}).get("output_json", "")))) if isinstance(info.get("longbench"), dict) else None
        passkey_json = load_json(Path(str(info.get("passkey_tf", {}).get("summary_json", "")))) if isinstance(info.get("passkey_tf"), dict) else None

        lb_scores = longbench_task_scores(longbench_json or {})
        lb_avg = mean_scores(lb_scores)
        niah_avg = mean_niah_accuracy(niah_json or {})
        pk_16k = None
        pk_margin_16k = None
        if passkey_json:
            by_len = passkey_json.get("by_length", {})
            if isinstance(by_len, dict) and "16384" in by_len and isinstance(by_len["16384"], dict):
                try:
                    pk_16k = float(by_len["16384"].get("tf_accuracy"))
                except Exception:
                    pk_16k = None
                try:
                    pk_margin_16k = float(by_len["16384"].get("margin_mean"))
                except Exception:
                    pk_margin_16k = None

        row = {
            "method": method,
            "train_loss": train_summary.get("train_metrics", {}).get("train_loss"),
            "eval_loss": train_summary.get("eval_metrics", {}).get("eval_loss"),
            "tail_ppl_16k": train_summary.get("ppl_results", {}).get("16384", {}).get("ppl"),
            "passkey_gen_16k": train_summary.get("passkey_results", {}).get("16384", {}).get("accuracy"),
            "niah_mean_acc": niah_avg,
            "longbench_avg": lb_avg,
            "passkey_tf_16k": pk_16k,
            "passkey_tf_margin_16k": pk_margin_16k,
        }
        summary_rows.append(row)

        lb_row = {"method": method}
        for t in all_tasks:
            lb_row[t] = lb_scores.get(t)
        longbench_rows.append(lb_row)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values("method")
        df_summary.to_csv(report_dir / "summary_table.csv", index=False)

        numeric_cols = ["niah_mean_acc", "longbench_avg", "passkey_tf_16k", "tail_ppl_16k"]
        for col in numeric_cols:
            df_col = df_summary[["method", col]].dropna()
            if not df_col.empty:
                save_bar(
                    df=df_col,
                    x="method",
                    y=col,
                    title=f"SOTA Comparison: {col}",
                    out_png=report_dir / f"fig_{col}.png",
                    out_pdf=report_dir / f"fig_{col}.pdf",
                )

    if longbench_rows:
        df_lb = pd.DataFrame(longbench_rows).sort_values("method")
        df_lb.to_csv(report_dir / "longbench_task_table.csv", index=False)

    manifest_path = eval_root / "downstream_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] manifest={manifest_path}", flush=True)
    print(f"[done] report_dir={report_dir}", flush=True)


if __name__ == "__main__":
    main()
