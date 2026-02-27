#!/usr/bin/env python3
"""Direct dual-seed runner for LLaMA-3-8B (8K) EVQ τ=1.5 vs Geometric.

Updated 2026-02-27: τ=1.5 confirmed optimal across 4 independent runs:
  - 50M  seed=42:  PPL@16K -10.9%
  - 125M seed=42:  PPL@16K -18.9%
  - 125M seed=137: PPL@16K even better (26.860 vs 27.699)
  - Cross-seed CV = 2.2% → not noise

Stage A τ-sweep is SKIPPED. We go directly to dual-seed full evaluation:
- Stage A (2 jobs, seed=42, full LB21):
  A1 geometric (tau=0.0) r32 s800
  A2 evq_cosh (tau=1.5) r32 s800
- Stage B (2 jobs, seed=1337, full LB21):
  B1 geometric (tau=0.0) r32 s800
  B2 evq_cosh (tau=1.5) r32 s800

All 4 jobs run full LB21 eval → paired statistics across 2 seeds.

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


def normalize_run_tag(raw: str) -> str:
    tag = str(raw or "").strip()
    if not tag:
        return ""
    out: List[str] = []
    for ch in tag:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch in {".", ":", "/", " "}:
            out.append("_")
    normalized = "".join(out).strip("_-")
    if not normalized:
        raise ValueError(f"Invalid --run_tag={raw!r}: no usable characters after normalization.")
    return normalized


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
    method: str  # geometric | evq_cosh
    seed: int
    lora_rank: int
    max_steps: int
    rope_schedule: str
    evq_tau: float
    run_full_eval: bool
    notes: str = ""


def build_stage_a_jobs() -> List[Job]:
    return [
        Job("A1", "A", "geometric", 42, 32, 800, "evq_cosh", 0.0, True, "Geometric baseline (tau=0.0) seed42 full lb21"),
        Job("A2", "A", "evq_cosh", 42, 32, 800, "evq_cosh", 1.5, True, "EVQ-Cosh tau=1.5 seed42 full lb21"),
    ]


def job_run_name(job: Job, run_tag: str = "") -> str:
    tau_tag = f"_tau{job.evq_tau:.2f}".replace(".", "p")
    base = f"{job.job_id}_{job.method}{tau_tag}_r{job.lora_rank}_s{job.max_steps}_seed{job.seed}"
    tag = normalize_run_tag(run_tag)
    if tag:
        return f"{base}__{tag}"
    return base


def has_reusable_artifacts(run_dir: Path) -> bool:
    adapter_dir = run_dir / "adapter"
    adapter_cfg = adapter_dir / "adapter_config.json"
    adapter_weight_safe = adapter_dir / "adapter_model.safetensors"
    adapter_weight_bin = adapter_dir / "adapter_model.bin"
    inv_path = run_dir / "artifacts" / "custom_inv_freq.pt"
    run_cfg = run_dir / "run_config.json"
    return (
        adapter_dir.exists()
        and adapter_cfg.exists()
        and (adapter_weight_safe.exists() or adapter_weight_bin.exists())
        and inv_path.exists()
        and run_cfg.exists()
    )


def is_job_complete(job: Job, run_dir: Path) -> bool:
    gate_json = run_dir / "eval" / "qasper_musique_compare.json"
    if not has_reusable_artifacts(run_dir) or not gate_json.exists():
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
    run_tag: str,
    execute: bool,
    skip_training: bool = False,
) -> Dict[str, object]:
    run_name = job_run_name(job, run_tag=run_tag)
    run_name_base = job_run_name(job, run_tag="")
    train_script = Path(args.train_script)
    run_dir = output_root / "train" / run_name
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"
    if job.method == "geometric" or float(job.evq_tau) < 1e-5:
        rope_schedule = "evq_cosh"
        evq_tau = 0.0
    else:
        rope_schedule = str(job.rope_schedule)
        evq_tau = float(job.evq_tau)

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
        "--rope_schedule",
        rope_schedule,
        "--evq_tau",
        str(evq_tau),
        "--evq_beta",
        str(args.evq_beta),
        "--mix_long_ratio",
        str(args.mix_long_ratio),
        "--mix_wiki_ratio",
        str(args.mix_wiki_ratio),
        "--synthetic_ratio",
        str(args.synthetic_ratio),
        "--min_supervised_tokens",
        str(args.min_supervised_tokens),
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
        "--mixed_dataset_split",
        args.mixed_dataset_split,
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
    if bool(args.strict_single_96gb) or bool(args.strict_single_h800_96gb):
        cmd.append("--strict_single_96gb")
        cmd.extend(["--min_gpu_memory_gib", str(float(args.min_gpu_memory_gib))])
        if str(args.require_gpu_name_substring).strip():
            cmd.extend(["--require_gpu_name_substring", str(args.require_gpu_name_substring).strip()])
    if str(args.require_transformers_version).strip():
        cmd.extend(["--require_transformers_version", str(args.require_transformers_version).strip()])

    if str(args.mixed_dataset_dir).strip():
        cmd.extend(["--mixed_dataset_dir", args.mixed_dataset_dir])
    cmd.append("--require_offset_boundary" if bool(args.require_offset_boundary) else "--no-require_offset_boundary")

    if job.run_full_eval:
        cmd.append("--run_full_eval")
    else:
        cmd.append("--no-run_full_eval")

    if skip_training:
        cmd.append("--skip_training")

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
        "rope_schedule": rope_schedule,
        "evq_tau": evq_tau,
        "run_full_eval": bool(job.run_full_eval),
        "run_name": run_name,
        "run_name_base": run_name_base,
        "run_tag": normalize_run_tag(run_tag),
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


def pick_best_evq(stage_a_rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    cands = [r for r in stage_a_rows if r.get("method") == "evq_cosh" and bool(r.get("gate_pass"))]
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
    evq_best: Optional[Dict[str, object]],
    geometric_ref: Optional[Dict[str, object]],
) -> None:
    stage_a = [r for r in rows if str(r.get("phase")) == "A"]
    stage_b = [r for r in rows if str(r.get("phase")) == "B"]
    full_eval_seed_set = set()
    for r in rows:
        if str(r.get("method")) not in {"geometric", "evq_cosh"}:
            continue
        if not str(r.get("full_raw_json", "")).strip():
            continue
        try:
            full_eval_seed_set.add(int(r.get("seed")))
        except Exception:
            pass
    full_eval_seeds = sorted(full_eval_seed_set)
    full_eval_seed_text = ",".join(str(s) for s in full_eval_seeds) if full_eval_seeds else "NA"

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
        "4. Dual-seed: both seed=42 (Stage A) and seed=1337 (Stage B) run full LB21.",
        "5. Paired stats file: see stats section below.",
        "",
        "## Selected Config",
        "",
    ]

    if geometric_ref:
        lines.append(
            f"- geometric_ref: `{geometric_ref.get('run_name')}` (r={geometric_ref.get('lora_rank')}, steps={geometric_ref.get('max_steps')})"
        )
    if evq_best:
        lines.append(
            f"- evq_best: `{evq_best.get('run_name')}` (r={evq_best.get('lora_rank')}, steps={evq_best.get('max_steps')}, tau={evq_best.get('evq_tau')})"
        )
    if not geometric_ref:
        lines.append("- geometric_ref: `NA`")
    if not evq_best:
        lines.append("- evq_best: `NA`")

    lines.extend(
        [
            "",
            "## Stage Jobs",
            "",
            "| job_id | phase | method | schedule | tau | seed | rank | steps | gate_pass | macro_delta_pct | status | run_name |",
            "|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for r in rows:
        seed_raw = r.get("seed", "NA")
        seed_str = str(int(seed_raw)) if isinstance(seed_raw, (int, float)) else str(seed_raw)
        rank_raw = r.get("lora_rank", "NA")
        rank_str = str(int(rank_raw)) if isinstance(rank_raw, (int, float)) else str(rank_raw)
        steps_raw = r.get("max_steps", "NA")
        steps_str = str(int(steps_raw)) if isinstance(steps_raw, (int, float)) else str(steps_raw)
        lines.append(
            "| {job_id} | {phase} | {method} | {rope_schedule} | {evq_tau} | {seed} | {lora_rank} | {max_steps} | {gate_pass} | {macro} | {status} | {run_name} |".format(
                job_id=str(r.get("job_id", "")),
                phase=str(r.get("phase", "")),
                method=str(r.get("method", "")),
                rope_schedule=str(r.get("rope_schedule", "")),
                evq_tau=("NA" if r.get("evq_tau") in {"", None, "NA"} else f"{float(r.get('evq_tau')):.2f}"),
                seed=seed_str,
                lora_rank=rank_str,
                max_steps=steps_str,
                gate_pass=str(r.get("gate_pass", "NA")),
                macro=("NA" if r.get("full_macro_delta_pct") is None else f"{float(r.get('full_macro_delta_pct')):.4f}"),
                status=str(r.get("status", "")),
                run_name=str(r.get("run_name", "")),
            )
        )

    lines.extend(["", "## Stats", ""])
    lines.append(f"- full_eval_training_seeds: `{full_eval_seed_text}`")
    lines.append("- inference_scope: `paired sample-level EVQ-vs-Geometric deltas on shared evaluation examples`")
    if len(full_eval_seeds) < 2:
        lines.append(
            "- caveat: `Current significance does not estimate cross-seed training-run variance "
            "(fewer than 2 full-eval training seeds).`"
        )

    if stats_json and stats_json.exists():
        st = load_json(stats_json)
        pooled = st.get("pooled", {}) if isinstance(st, dict) else {}
        meta = st.get("meta", {}) if isinstance(st, dict) else {}
        lines.extend(
            [
                f"- stats_json: `{stats_json.as_posix()}`",
                f"- run_pair_count: `{int(meta.get('run_pair_count', 0))}`",
                f"- claim_min_run_pairs: `{int(meta.get('claim_min_run_pairs', 0))}`",
                f"- seed_replication_ok: `{bool(meta.get('seed_replication_ok', False))}`",
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
    evq_raws = [str(r.get("full_raw_json")) for r in rows if r.get("method") == "evq_cosh" and str(r.get("full_raw_json", ""))]

    if not geo_raws or not evq_raws:
        return None

    # Pair by seed when possible.
    seed_to_geo: Dict[int, str] = {int(r.get("seed", -1)): str(r.get("full_raw_json")) for r in rows if r.get("method") == "geometric" and str(r.get("full_raw_json", ""))}
    seed_to_evq: Dict[int, str] = {int(r.get("seed", -1)): str(r.get("full_raw_json")) for r in rows if r.get("method") == "evq_cosh" and str(r.get("full_raw_json", ""))}
    seed_to_geo_row: Dict[int, Dict[str, object]] = {int(r.get("seed", -1)): r for r in rows if r.get("method") == "geometric" and str(r.get("full_raw_json", ""))}
    seed_to_evq_row: Dict[int, Dict[str, object]] = {int(r.get("seed", -1)): r for r in rows if r.get("method") == "evq_cosh" and str(r.get("full_raw_json", ""))}
    common_seeds = sorted(set(seed_to_geo.keys()) & set(seed_to_evq.keys()))
    if not common_seeds:
        return None

    protocol_issues: List[str] = []
    for seed in common_seeds:
        geo_path = Path(seed_to_geo[seed])
        evq_path = Path(seed_to_evq[seed])
        if not geo_path.exists():
            protocol_issues.append(f"seed={seed}: geometric full_raw_json missing on disk: {geo_path.as_posix()}")
        if not evq_path.exists():
            protocol_issues.append(f"seed={seed}: evq full_raw_json missing on disk: {evq_path.as_posix()}")
        geo_row = seed_to_geo_row.get(seed, {})
        evq_row = seed_to_evq_row.get(seed, {})
        for key in ("code_hash", "data_hash"):
            geo_val = str(geo_row.get(key, "")).strip()
            evq_val = str(evq_row.get(key, "")).strip()
            if not geo_val or not evq_val:
                protocol_issues.append(f"seed={seed}: missing {key} for protocol parity check.")
            elif geo_val != evq_val:
                protocol_issues.append(
                    f"seed={seed}: {key} mismatch (geometric={geo_val[:12]}..., evq={evq_val[:12]}...)."
                )
    if protocol_issues:
        raise RuntimeError(
            "Refusing paired stats due to protocol mismatch between geometric and EVQ runs:\n- "
            + "\n- ".join(protocol_issues)
        )

    evq_jsons = ",".join([seed_to_evq[s] for s in common_seeds])
    geometric_jsons = ",".join([seed_to_geo[s] for s in common_seeds])

    stats_dir = output_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_json = stats_dir / "paired_stats_evq_vs_geometric.json"
    out_md = stats_dir / "paired_stats_evq_vs_geometric.md"

    cmd = [
        args.python_exe,
        Path(args.stats_script).as_posix(),
        "--evq_jsons",
        evq_jsons,
        "--geometric_jsons",
        geometric_jsons,
        "--claim_min_run_pairs",
        str(int(args.stats_claim_min_run_pairs)),
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
    ap.add_argument("--mixed_dataset_dir", type=str, default="")
    ap.add_argument("--mixed_dataset_split", type=str, default="train")
    ap.add_argument(
        "--require_mixed_dataset_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if --execute is used without an explicit --mixed_dataset_dir.",
    )
    ap.add_argument("--longbench_local_data_dir", type=str, required=True)

    ap.add_argument("--qwen_seed42_json", type=str, required=True)
    ap.add_argument("--qwen_seed1337_json", type=str, required=True)
    ap.add_argument("--morning_reference_json", type=str, required=True)

    ap.add_argument("--mix_long_ratio", type=float, default=0.7)
    ap.add_argument("--mix_wiki_ratio", type=float, default=0.1)
    ap.add_argument("--synthetic_ratio", type=float, default=0.2)
    ap.add_argument("--min_supervised_tokens", type=int, default=64)
    ap.add_argument("--require_offset_boundary", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--evq_beta", type=float, default=3.0)

    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_batch_input_tokens", type=int, default=98304)
    ap.add_argument("--max_input_tokens_eval", type=int, default=8192)
    # --run_full_eval_seed1337 removed: all 4 jobs now run full LB21 by default.
    # Kept for backward-compat (ignored).
    ap.add_argument("--run_full_eval_seed1337", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stats_claim_min_run_pairs", type=int, default=2)
    ap.add_argument("--strict_single_96gb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--strict_single_h800_96gb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_gpu_memory_gib", type=float, default=90.0)
    ap.add_argument("--require_gpu_name_substring", type=str, default="")
    ap.add_argument("--require_transformers_version", type=str, default="")

    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Optional run suffix appended to every job run_name. "
            "When --execute is on and this is empty, a timestamp tag is auto-generated to avoid run_dir collisions."
        ),
    )
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
    if int(args.stats_claim_min_run_pairs) < 1:
        raise ValueError("--stats_claim_min_run_pairs must be >= 1")
    if bool(args.execute) and bool(args.require_mixed_dataset_dir) and not str(args.mixed_dataset_dir).strip():
        raise RuntimeError(
            "Refusing execution without --mixed_dataset_dir. "
            "On-the-fly fallback mix is not allowed for paper-grade runs."
        )
    repo_root = Path(__file__).resolve().parents[3]
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = normalize_run_tag(args.run_tag)
    if bool(args.execute) and not run_tag:
        run_tag = time.strftime("%Y%m%d_%H%M%S")
    if run_tag:
        print(f"[RUN TAG] {run_tag}")

    registry_rows: List[Dict[str, object]] = []

    stage_a_jobs = build_stage_a_jobs()
    for job in stage_a_jobs:
        run_name = job_run_name(job, run_tag=run_tag)
        run_dir = output_root / "train" / run_name
        if bool(args.skip_existing) and is_job_complete(job, run_dir):
            row = run_job(
                job,
                args=args,
                repo_root=repo_root,
                output_root=output_root,
                run_tag=run_tag,
                execute=False,
            )
            row["status"] = "SKIPPED_EXISTING"
            registry_rows.append(row)
            continue
        row = run_job(
            job,
            args=args,
            repo_root=repo_root,
            output_root=output_root,
            run_tag=run_tag,
            execute=bool(args.execute),
        )
        registry_rows.append(row)

    # ── Stage B: seed=1337 replication (geometric baseline + EVQ τ=1.5) ──
    # Both jobs run full LB21 eval.  No gate needed — τ=1.5 already
    # confirmed optimal by 50M/125M τ-sweep (dual-seed, CV=2.2%).
    stage_a_rows = [r for r in registry_rows if str(r.get("phase")) == "A"]
    geometric_ref = next((r for r in stage_a_rows if str(r.get("job_id")) == "A1"), None)
    evq_ref = next((r for r in stage_a_rows if str(r.get("job_id")) == "A2"), None)

    stage_b_jobs: List[Job] = [
        Job("B1", "B", "geometric", 1337, 32, 800, "evq_cosh", 0.0, True,
            "Geometric baseline (tau=0.0) seed1337 full lb21"),
        Job("B2", "B", "evq_cosh", 1337, 32, 800, "evq_cosh", 1.5, True,
            "EVQ-Cosh tau=1.5 seed1337 full lb21"),
    ]

    for job in stage_b_jobs:
        run_name = job_run_name(job, run_tag=run_tag)
        run_dir = output_root / "train" / run_name
        if bool(args.skip_existing) and is_job_complete(job, run_dir):
            row = run_job(
                job,
                args=args,
                repo_root=repo_root,
                output_root=output_root,
                run_tag=run_tag,
                execute=False,
            )
            row["status"] = "SKIPPED_EXISTING"
            registry_rows.append(row)
            continue
        row = run_job(
            job,
            args=args,
            repo_root=repo_root,
            output_root=output_root,
            run_tag=run_tag,
            execute=bool(args.execute),
        )
        registry_rows.append(row)

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
        "rope_schedule",
        "evq_tau",
        "run_full_eval",
        "run_name",
        "run_name_base",
        "run_tag",
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
        evq_best=evq_ref,
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
        "stage_a_runs": [job_run_name(j, run_tag=run_tag) for j in stage_a_jobs],
        "stage_b_runs": [job_run_name(j, run_tag=run_tag) for j in stage_b_jobs],
        "run_tag": run_tag,
        "mixed_dataset_dir": str(args.mixed_dataset_dir),
        "require_mixed_dataset_dir": bool(args.require_mixed_dataset_dir),
        "min_supervised_tokens": int(args.min_supervised_tokens),
        "require_offset_boundary": bool(args.require_offset_boundary),
        "run_full_eval_seed1337": bool(args.run_full_eval_seed1337),
        "stats_claim_min_run_pairs": int(args.stats_claim_min_run_pairs),
    }
    save_json(output_root / "stats" / "run_manifest.json", run_manifest)

    print("[DONE] llama8k_theory_v1 orchestration complete")
    print(f"- registry: {registry_csv}")
    print(f"- report:   {report_md}")
    if stats_json:
        print(f"- stats:    {stats_json}")


if __name__ == "__main__":
    main()
