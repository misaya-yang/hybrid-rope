#!/usr/bin/env python3
"""Two-stage 8-job runner for LLaMA-3-8B (8K) Anchored vs Geometric theory validation.

Plan contract:
- Stage A (4 jobs, seed=42, gate only):
  A1 geometric r32 s800
  A2 anchored r32 s800
  A3 anchored r48 s800
  A4 anchored r32 s600
- Stage B (4 jobs):
  B1 geometric_best seed1337 gate
  B2 anchored_best seed1337 gate
  B3 geometric_best seed42 full lb21
  B4 anchored_best seed42 full lb21

This script only orchestrates the canonical pipeline entry:
  scripts/isolated/longinst/new_lora_longinst_train_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


@dataclass
class Job:
    job_id: str
    phase: str
    method: str  # geometric | anchored
    seed: int
    lora_rank: int
    max_steps: int
    anchor_factor: float
    run_full_eval: bool
    notes: str = ""


def build_stage_a_jobs() -> List[Job]:
    return [
        Job("A1", "A", "geometric", 42, 32, 800, 1.0, False, "Geometric baseline"),
        Job("A2", "A", "anchored", 42, 32, 800, 4.0, False, "Anchored default"),
        Job("A3", "A", "anchored", 42, 48, 800, 4.0, False, "Anchored higher rank"),
        Job("A4", "A", "anchored", 42, 32, 600, 4.0, False, "Anchored fewer steps"),
    ]


def job_run_name(job: Job) -> str:
    return f"{job.job_id}_{job.method}_r{job.lora_rank}_s{job.max_steps}_seed{job.seed}"


def is_job_complete(job: Job, run_dir: Path) -> bool:
    run_cfg = run_dir / "run_config.json"
    gate_json = run_dir / "eval" / "qasper_musique_compare.json"
    if not run_cfg.exists() or not gate_json.exists():
        return False
    if job.run_full_eval:
        full_json = run_dir / "eval" / "longbench_full_compare.json"
        full_raw_json = run_dir / "eval" / "longbench_full_compare_raw.json"
        if not full_json.exists() or not full_raw_json.exists():
            return False
    return True


def run_job(
    job: Job,
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    execute: bool,
) -> Dict[str, object]:
    run_name = job_run_name(job)
    train_script = Path(args.train_script)
    run_dir = output_root / "train" / run_name
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    cmd = [
        args.python_exe,
        train_script.as_posix(),
        "--base_model_path",
        args.base_model_path,
        "--output_root",
        output_root.as_posix(),
        "--run_name",
        run_name,
        "--seed",
        str(job.seed),
        "--max_steps",
        str(job.max_steps),
        "--lora_rank",
        str(job.lora_rank),
        "--anchor_factor",
        str(job.anchor_factor),
        "--slope_raw",
        str(args.slope_raw),
        "--center_ratio",
        str(args.center_ratio),
        "--mix_long_ratio",
        str(args.mix_long_ratio),
        "--mix_wiki_ratio",
        str(args.mix_wiki_ratio),
        "--synthetic_ratio",
        str(args.synthetic_ratio),
        "--max_seq_len",
        str(args.max_seq_len),
        "--attn_implementation",
        args.attn_implementation,
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_steps",
        str(args.warmup_steps),
        "--longalpaca_path",
        args.longalpaca_path,
        "--longqa_path",
        args.longqa_path,
        "--wikitext_train_path",
        args.wikitext_train_path,
        "--longbench_local_data_dir",
        args.longbench_local_data_dir,
        "--qwen_seed42_json",
        args.qwen_seed42_json,
        "--qwen_seed1337_json",
        args.qwen_seed1337_json,
        "--morning_reference_json",
        args.morning_reference_json,
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--max_batch_input_tokens",
        str(args.max_batch_input_tokens),
        "--max_input_tokens_eval",
        str(args.max_input_tokens_eval),
    ]

    if job.run_full_eval:
        cmd.append("--run_full_eval")
    else:
        cmd.append("--no-run_full_eval")

    start = time.time()
    status = "PLANNED"
    return_code = None
    if execute:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=repo_root.as_posix(), stdout=logf, stderr=subprocess.STDOUT)
        return_code = int(proc.returncode)
        status = "DONE" if proc.returncode == 0 else "FAILED"
    elapsed = time.time() - start

    gate_json = run_dir / "eval" / "qasper_musique_compare.json"
    gate_scores = {}
    q_base = q_lora = m_base = m_lora = None
    gate_pass = None
    if gate_json.exists():
        gate_obj = load_json(gate_json)
        gate_scores = gate_obj.get("gate_scores", {}) if isinstance(gate_obj, dict) else {}
        if isinstance(gate_scores, dict):
            q = gate_scores.get("qasper")
            m = gate_scores.get("musique")
            if isinstance(q, dict) and isinstance(m, dict):
                qb = q.get("base")
                ql = q.get("lora")
                mb = m.get("base")
                ml = m.get("lora")
                if all(isinstance(x, (int, float)) for x in [qb, ql, mb, ml]):
                    q_base = float(qb)
                    q_lora = float(ql)
                    m_base = float(mb)
                    m_lora = float(ml)
                    gate_pass = bool((q_lora >= q_base) and (m_lora >= (m_base - 1.0)))

    full_json = run_dir / "eval" / "longbench_full_compare.json"
    full_raw_json = run_dir / "eval" / "longbench_full_compare_raw.json"
    macro_base = macro_lora = macro_delta = None
    if full_json.exists():
        full_obj = load_json(full_json)
        macro_base = float(full_obj.get("macro_base_pct", 0.0))
        macro_lora = float(full_obj.get("macro_lora_pct", 0.0))
        macro_delta = float(full_obj.get("macro_delta_pct", 0.0))

    run_config_path = run_dir / "run_config.json"
    code_hash = data_hash = inv_hash = ""
    if run_config_path.exists():
        rc = load_json(run_config_path)
        code_hash = str(rc.get("code_hash", ""))
        data_hash = str(rc.get("dataset", {}).get("data_hash_sha256", ""))
        inv_hash = str(rc.get("rope", {}).get("custom_inv_freq_sha256", ""))

    result = {
        "job_id": job.job_id,
        "phase": job.phase,
        "method": job.method,
        "seed": job.seed,
        "lora_rank": job.lora_rank,
        "max_steps": job.max_steps,
        "anchor_factor": job.anchor_factor,
        "run_full_eval": bool(job.run_full_eval),
        "run_name": run_name,
        "status": status,
        "return_code": return_code,
        "run_dir": run_dir.as_posix(),
        "log_path": log_path.as_posix(),
        "gate_json": gate_json.as_posix() if gate_json.exists() else "",
        "gate_pass": gate_pass,
        "gate_qasper_base": q_base,
        "gate_qasper_lora": q_lora,
        "gate_musique_base": m_base,
        "gate_musique_lora": m_lora,
        "full_json": full_json.as_posix() if full_json.exists() else "",
        "full_raw_json": full_raw_json.as_posix() if full_raw_json.exists() else "",
        "full_macro_base_pct": macro_base,
        "full_macro_lora_pct": macro_lora,
        "full_macro_delta_pct": macro_delta,
        "run_config_json": run_config_path.as_posix() if run_config_path.exists() else "",
        "code_hash": code_hash,
        "data_hash": data_hash,
        "inv_freq_hash": inv_hash,
        "started_at": now(),
        "elapsed_sec": round(float(elapsed), 2),
        "notes": job.notes,
        "cmd": cmd,
    }
    return result


def pick_best_anchored(stage_a_rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    cands = [r for r in stage_a_rows if r.get("method") == "anchored" and bool(r.get("gate_pass"))]
    if not cands:
        return None

    def score(row: Dict[str, object]) -> Tuple[float, float, float]:
        qb = float(row.get("gate_qasper_base") or 0.0)
        ql = float(row.get("gate_qasper_lora") or 0.0)
        mb = float(row.get("gate_musique_base") or 0.0)
        ml = float(row.get("gate_musique_lora") or 0.0)
        mean_delta = ((ql - qb) + (ml - mb)) / 2.0
        # tie-breaker: fewer steps then smaller rank
        return (mean_delta, -float(row.get("max_steps") or 0.0), -float(row.get("lora_rank") or 0.0))

    cands.sort(key=score, reverse=True)
    return cands[0]


def write_report(
    report_path: Path,
    rows: List[Dict[str, object]],
    stats_json: Optional[Path],
    anchored_best: Optional[Dict[str, object]],
    geometric_ref: Optional[Dict[str, object]],
) -> None:
    stage_a = [r for r in rows if str(r.get("phase")) == "A"]
    stage_b = [r for r in rows if str(r.get("phase")) == "B"]

    lines = [
        "# llama8k_theory_v1 Report",
        "",
        f"- generated_at: `{now()}`",
        f"- total_jobs_recorded: `{len(rows)}`",
        f"- stage_a_jobs: `{len(stage_a)}`",
        f"- stage_b_jobs: `{len(stage_b)}`",
        "",
        "## DoD Check",
        "",
        "1. Base model fixed to Meta-Llama-3-8B-Instruct: enforced in training entry script.",
        "2. Protocol lock with hashes: run_config + data_hash + code_hash + inv_freq_hash required.",
        "3. Gate rule: qasper>=base and musique>=base-1.0.",
        "4. Seed pair: 42 and 1337 covered by Stage B gate runs.",
        "5. Paired stats file: see stats section below.",
        "",
        "## Selected Config",
        "",
    ]

    if geometric_ref:
        lines.append(
            f"- geometric_ref: `{geometric_ref.get('run_name')}` (r={geometric_ref.get('lora_rank')}, steps={geometric_ref.get('max_steps')})"
        )
    if anchored_best:
        lines.append(
            f"- anchored_best: `{anchored_best.get('run_name')}` (r={anchored_best.get('lora_rank')}, steps={anchored_best.get('max_steps')})"
        )
    if not geometric_ref:
        lines.append("- geometric_ref: `NA`")
    if not anchored_best:
        lines.append("- anchored_best: `NA`")

    lines.extend(
        [
            "",
            "## Stage Jobs",
            "",
            "| job_id | phase | method | seed | rank | steps | gate_pass | macro_delta_pct | status | run_name |",
            "|---|---|---|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for r in rows:
        lines.append(
            "| {job_id} | {phase} | {method} | {seed} | {lora_rank} | {max_steps} | {gate_pass} | {macro} | {status} | {run_name} |".format(
                job_id=str(r.get("job_id", "")),
                phase=str(r.get("phase", "")),
                method=str(r.get("method", "")),
                seed=int(r.get("seed", 0) or 0),
                lora_rank=int(r.get("lora_rank", 0) or 0),
                max_steps=int(r.get("max_steps", 0) or 0),
                gate_pass=str(r.get("gate_pass", "NA")),
                macro=("NA" if r.get("full_macro_delta_pct") is None else f"{float(r.get('full_macro_delta_pct')):.4f}"),
                status=str(r.get("status", "")),
                run_name=str(r.get("run_name", "")),
            )
        )

    lines.extend(["", "## Stats", ""])
    if stats_json and stats_json.exists():
        st = load_json(stats_json)
        pooled = st.get("pooled", {}) if isinstance(st, dict) else {}
        lines.extend(
            [
                f"- stats_json: `{stats_json.as_posix()}`",
                f"- claim_grade: `{pooled.get('claim_grade', 'inconclusive')}`",
                f"- mean_diff_pct: `{float(pooled.get('mean_diff_pct', 0.0)):.4f}`",
                f"- ci95_pct: `[{float(pooled.get('ci95_low_pct', 0.0)):.4f}, {float(pooled.get('ci95_high_pct', 0.0)):.4f}]`",
                f"- p_permutation: `{float(pooled.get('p_permutation', float('nan'))):.6f}`",
                f"- effect_size_dz: `{float(pooled.get('effect_size_dz', 0.0)):.6f}`",
            ]
        )
    else:
        lines.append("- stats_json: `not_generated`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stats_if_possible(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    rows: List[Dict[str, object]],
    output_root: Path,
    execute: bool,
) -> Optional[Path]:
    geo_raws = [str(r.get("full_raw_json")) for r in rows if r.get("method") == "geometric" and str(r.get("full_raw_json", ""))]
    anc_raws = [str(r.get("full_raw_json")) for r in rows if r.get("method") == "anchored" and str(r.get("full_raw_json", ""))]

    if not geo_raws or not anc_raws:
        return None

    # Pair by seed when possible.
    seed_to_geo: Dict[int, str] = {int(r.get("seed", -1)): str(r.get("full_raw_json")) for r in rows if r.get("method") == "geometric" and str(r.get("full_raw_json", ""))}
    seed_to_anc: Dict[int, str] = {int(r.get("seed", -1)): str(r.get("full_raw_json")) for r in rows if r.get("method") == "anchored" and str(r.get("full_raw_json", ""))}
    common_seeds = sorted(set(seed_to_geo.keys()) & set(seed_to_anc.keys()))
    if not common_seeds:
        return None

    anchored_jsons = ",".join([seed_to_anc[s] for s in common_seeds])
    geometric_jsons = ",".join([seed_to_geo[s] for s in common_seeds])

    stats_dir = output_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_json = stats_dir / "paired_stats_anchored_vs_geometric.json"
    out_md = stats_dir / "paired_stats_anchored_vs_geometric.md"

    cmd = [
        args.python_exe,
        Path(args.stats_script).as_posix(),
        "--anchored_jsons",
        anchored_jsons,
        "--geometric_jsons",
        geometric_jsons,
        "--output_json",
        out_json.as_posix(),
        "--output_md",
        out_md.as_posix(),
    ]

    if execute:
        subprocess.run(cmd, cwd=repo_root.as_posix(), check=True)
    return out_json if out_json.exists() else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLaMA-3-8B (8K) theory validation orchestrator")
    repo_root = Path(__file__).resolve().parents[3]
    default_train = repo_root / "scripts/isolated/longinst/new_lora_longinst_train_v1.py"
    default_stats = repo_root / "scripts/isolated/longinst/paired_stats_llama8k_theory_v1.py"
    ap.add_argument("--python_exe", type=str, default=sys.executable)
    ap.add_argument("--train_script", type=str, default=default_train.as_posix())
    ap.add_argument("--stats_script", type=str, default=default_stats.as_posix())
    ap.add_argument("--output_root", type=str, default="artifacts/llama8k_theory_v1")

    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--longalpaca_path", type=str, required=True)
    ap.add_argument("--longqa_path", type=str, default="")
    ap.add_argument("--wikitext_train_path", type=str, required=True)
    ap.add_argument("--longbench_local_data_dir", type=str, required=True)

    ap.add_argument("--qwen_seed42_json", type=str, required=True)
    ap.add_argument("--qwen_seed1337_json", type=str, required=True)
    ap.add_argument("--morning_reference_json", type=str, required=True)

    ap.add_argument("--mix_long_ratio", type=float, default=0.7)
    ap.add_argument("--mix_wiki_ratio", type=float, default=0.1)
    ap.add_argument("--synthetic_ratio", type=float, default=0.2)

    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--slope_raw", type=float, default=20.0)
    ap.add_argument("--center_ratio", type=float, default=0.70)

    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_batch_input_tokens", type=int, default=98304)
    ap.add_argument("--max_input_tokens_eval", type=int, default=8192)

    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--write_docs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write/overwrite docs/exp registry+report. Automatically enabled when --execute is on.",
    )
    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry_rows: List[Dict[str, object]] = []

    stage_a_jobs = build_stage_a_jobs()
    for job in stage_a_jobs:
        run_name = job_run_name(job)
        run_dir = output_root / "train" / run_name
        if bool(args.skip_existing) and is_job_complete(job, run_dir):
            row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=False)
            row["status"] = "SKIPPED_EXISTING"
            registry_rows.append(row)
            continue
        row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=bool(args.execute))
        registry_rows.append(row)

    stage_a_rows = [r for r in registry_rows if str(r.get("phase")) == "A"]
    geometric_ref = next((r for r in stage_a_rows if str(r.get("job_id")) == "A1"), None)
    anchored_best = pick_best_anchored(stage_a_rows)

    stage_b_jobs: List[Job] = []
    if geometric_ref and anchored_best:
        g_rank = int(geometric_ref.get("lora_rank", 32) or 32)
        g_steps = int(geometric_ref.get("max_steps", 800) or 800)
        a_rank = int(anchored_best.get("lora_rank", 32) or 32)
        a_steps = int(anchored_best.get("max_steps", 800) or 800)

        stage_b_jobs = [
            Job("B1", "B", "geometric", 1337, g_rank, g_steps, 1.0, False, "Geometric best on seed1337 gate"),
            Job("B2", "B", "anchored", 1337, a_rank, a_steps, 4.0, False, "Anchored best on seed1337 gate"),
            Job("B3", "B", "geometric", 42, g_rank, g_steps, 1.0, True, "Geometric full lb21"),
            Job("B4", "B", "anchored", 42, a_rank, a_steps, 4.0, True, "Anchored full lb21"),
        ]

    for job in stage_b_jobs[:2]:
        run_name = job_run_name(job)
        run_dir = output_root / "train" / run_name
        if bool(args.skip_existing) and is_job_complete(job, run_dir):
            row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=False)
            row["status"] = "SKIPPED_EXISTING"
            registry_rows.append(row)
            continue
        row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=bool(args.execute))
        registry_rows.append(row)

    b1 = next((r for r in registry_rows if str(r.get("job_id")) == "B1"), None)
    b2 = next((r for r in registry_rows if str(r.get("job_id")) == "B2"), None)
    b_gate_ok = bool(b1 and b2 and bool(b1.get("gate_pass")) and bool(b2.get("gate_pass")))

    if b_gate_ok:
        for job in stage_b_jobs[2:]:
            run_name = job_run_name(job)
            run_dir = output_root / "train" / run_name
            if bool(args.skip_existing) and is_job_complete(job, run_dir):
                row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=False)
                row["status"] = "SKIPPED_EXISTING"
                registry_rows.append(row)
                continue
            row = run_job(job, args=args, repo_root=repo_root, output_root=output_root, execute=bool(args.execute))
            registry_rows.append(row)
    elif stage_b_jobs:
        registry_rows.append(
            {
                "job_id": "B3_B4",
                "phase": "B",
                "method": "NA",
                "seed": "NA",
                "lora_rank": "NA",
                "max_steps": "NA",
                "anchor_factor": "NA",
                "run_full_eval": True,
                "run_name": "NA",
                "status": "CANCELLED_BY_GATE",
                "return_code": "NA",
                "run_dir": "",
                "log_path": "",
                "gate_json": "",
                "gate_pass": False,
                "gate_qasper_base": "",
                "gate_qasper_lora": "",
                "gate_musique_base": "",
                "gate_musique_lora": "",
                "full_json": "",
                "full_raw_json": "",
                "full_macro_base_pct": "",
                "full_macro_lora_pct": "",
                "full_macro_delta_pct": "",
                "run_config_json": "",
                "code_hash": "",
                "data_hash": "",
                "inv_freq_hash": "",
                "started_at": now(),
                "elapsed_sec": 0.0,
                "notes": "B1/B2 gate failed; full lb21 stopped.",
                "cmd": [],
            }
        )

    stats_json = run_stats_if_possible(
        args=args,
        repo_root=repo_root,
        rows=registry_rows,
        output_root=output_root,
        execute=bool(args.execute),
    )

    write_docs = bool(args.execute) or bool(args.write_docs)
    if write_docs:
        registry_csv = repo_root / "docs/exp/llama8k_theory_v1_registry.csv"
        report_md = repo_root / "docs/exp/llama8k_theory_v1_report.md"
    else:
        registry_csv = output_root / "stats" / "llama8k_theory_v1_registry.preview.csv"
        report_md = output_root / "stats" / "llama8k_theory_v1_report.preview.md"

    fields = [
        "job_id",
        "phase",
        "method",
        "seed",
        "lora_rank",
        "max_steps",
        "anchor_factor",
        "run_full_eval",
        "run_name",
        "status",
        "return_code",
        "gate_pass",
        "gate_qasper_base",
        "gate_qasper_lora",
        "gate_musique_base",
        "gate_musique_lora",
        "full_macro_base_pct",
        "full_macro_lora_pct",
        "full_macro_delta_pct",
        "run_dir",
        "log_path",
        "gate_json",
        "full_json",
        "full_raw_json",
        "run_config_json",
        "code_hash",
        "data_hash",
        "inv_freq_hash",
        "started_at",
        "elapsed_sec",
        "notes",
    ]
    save_csv(registry_csv, registry_rows, fieldnames=fields)

    write_report(
        report_path=report_md,
        rows=registry_rows,
        stats_json=stats_json,
        anchored_best=anchored_best,
        geometric_ref=geometric_ref,
    )

    run_manifest = {
        "timestamp": now(),
        "execute": bool(args.execute),
        "output_root": output_root.as_posix(),
        "registry_csv": registry_csv.as_posix(),
        "report_md": report_md.as_posix(),
        "stats_json": stats_json.as_posix() if stats_json else None,
        "stage_a_jobs": [job_run_name(j) for j in stage_a_jobs],
        "stage_b_jobs": [job_run_name(j) for j in stage_b_jobs],
    }
    save_json(output_root / "stats" / "run_manifest.json", run_manifest)

    print("[DONE] llama8k_theory_v1 orchestration complete")
    print(f"- registry: {registry_csv}")
    print(f"- report:   {report_md}")
    if stats_json:
        print(f"- stats:    {stats_json}")


if __name__ == "__main__":
    main()
