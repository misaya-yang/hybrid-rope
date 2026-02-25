#!/usr/bin/env python3
"""
H800/A800 parallel RoPE experiments runner.

Runs the user-specified queue:
- Group 1: geometric theta scaling baselines
- Group 2: hybrid (geo_10k + anchpoly_p3.9_omf0.3) with alpha sweep points
- Group 3: sigmoid allocation exploration

Uses the from-scratch TinyStories training/eval pipeline in:
  /opt/dfrope/from_scratch_dfrope_train_eval.py

This script can run jobs concurrently across multiple GPUs. On single-GPU
machines it will fall back to sequential execution.

Outputs:
  /opt/dfrope/results/h800_parallel/results_h800.json
  /opt/dfrope/results/h800_parallel/variants/<name>/result.json
  /opt/dfrope/results/h800_parallel/variants/<name>/run.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def now_ts() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def load_module(py_file: Path):
    spec = importlib.util.spec_from_file_location("dfrope_h800_fs", str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {py_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def geometric_freq(K: int, theta: float) -> np.ndarray:
    idx = np.arange(K, dtype=np.float64)
    return 1.0 / np.power(theta, idx / K)


def anchored_polynomial_freq(K: int, p: float, omega_max: float, omega_min: float) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    log_omega = np.log(omega_max) + np.power(t, p) * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def sigmoid_freq(
    K: int,
    omega_max: float,
    omega_min: float,
    steepness: float,
    midpoint: float,
) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    log_omega = np.log(omega_max) + s * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def hybrid_freq(omega_geo: np.ndarray, omega_alt: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * geo + (1 - alpha) * alt."""
    return alpha * omega_geo + (1.0 - alpha) * omega_alt


def hybrid_geo_poly(omega_geo: np.ndarray, omega_poly: np.ndarray, alpha_poly: float) -> np.ndarray:
    """
    Match user's naming: hybrid_alpha0.2 means 0.8 * geo + 0.2 * poly.
    """
    return (1.0 - alpha_poly) * omega_geo + alpha_poly * omega_poly


def _fmt_alpha(alpha: float) -> str:
    # Keep naming consistent with user: alpha0.15 / alpha0.25 etc.
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def build_job_queue(K: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    geo10k = geometric_freq(K, 10000.0)
    geo10k_min = float(geo10k[-1])
    omega_min = geo10k_min * 0.3
    anchpoly = anchored_polynomial_freq(K, p=3.9, omega_max=1.0, omega_min=omega_min)

    jobs: List[Dict[str, Any]] = []

    # Baseline (for normalization)
    jobs.append({"name": "geo_10k_baseline", "group": "baseline", "rope": {"kind": "standard", "theta": 10000.0}})

    # Group 1: theta scaling baselines
    for theta in [20000, 50000, 100000, 200000, 500000]:
        jobs.append(
            {
                "name": f"geo_{int(theta/1000)}k",
                "group": "theta_scaling",
                "rope": {"kind": "standard", "theta": float(theta)},
            }
        )

    # Group 2: hybrid fine-tuning
    for alpha in [0.15, 0.25]:
        omega = hybrid_freq(geo10k, anchpoly, alpha=float(alpha))
        jobs.append(
            {
                "name": f"hybrid_alpha{_fmt_alpha(alpha)}",
                "group": "hybrid",
                "rope": {
                    "kind": "custom",
                    "custom_omega": [float(x) for x in omega.tolist()],
                    "recipe": {
                        "geo_theta": 10000.0,
                        "anchpoly_p": 3.9,
                        "omega_min_factor": 0.3,
                        "alpha_geo": float(alpha),
                    },
                },
            }
        )

    # Group 3: sigmoid exploration
    for steep in [5.0, 8.0]:
        omega = sigmoid_freq(K, omega_max=1.0, omega_min=omega_min, steepness=float(steep), midpoint=0.5)
        jobs.append(
            {
                "name": f"sigmoid_steep{int(steep)}_mid0.5_omf0.3",
                "group": "sigmoid",
                "rope": {
                    "kind": "custom",
                    "custom_omega": [float(x) for x in omega.tolist()],
                    "recipe": {"steepness": float(steep), "midpoint": 0.5, "omega_min_factor": 0.3},
                },
            }
        )

    # Extra sweep (user request): theta-anchored sigmoid + high-theta hybrid(alpha0.2)
    for theta_base in [100000.0, 500000.0]:
        geo = geometric_freq(K, theta_base)
        omega_min_base = float(geo[-1]) * 0.3
        omega_sig = sigmoid_freq(K, omega_max=float(geo[0]), omega_min=omega_min_base, steepness=8.0, midpoint=0.5)
        jobs.append(
            {
                "name": f"sigmoid_th{int(theta_base/1000)}k_steep8_mid0.5_omf0.3",
                "group": "sigmoid_high_theta",
                "rope": {
                    "kind": "custom",
                    "custom_omega": [float(x) for x in omega_sig.tolist()],
                    "recipe": {
                        "theta_base": float(theta_base),
                        "steepness": 8.0,
                        "midpoint": 0.5,
                        "omega_min_factor": 0.3,
                    },
                },
            }
        )

        # Keep the same "poly recipe" but anchor omega_min off the chosen geo base.
        omega_poly = anchored_polynomial_freq(K, p=3.9, omega_max=float(geo[0]), omega_min=omega_min_base)
        omega_hyb = hybrid_geo_poly(geo, omega_poly, alpha_poly=0.2)
        jobs.append(
            {
                "name": f"hybrid_basegeo{int(theta_base/1000)}k_alpha0.2",
                "group": "hybrid_high_theta",
                "rope": {
                    "kind": "custom",
                    "custom_omega": [float(x) for x in omega_hyb.tolist()],
                    "recipe": {
                        "geo_theta": float(theta_base),
                        "anchpoly_p": 3.9,
                        "omega_min_factor": 0.3,
                        "alpha_poly": 0.2,
                    },
                },
            }
        )

    meta = {
        "K": K,
        "geo10k_min": geo10k_min,
        "omega_min_for_omf0.3": omega_min,
        "anchpoly_ref": {"p": 3.9, "omega_min_factor": 0.3},
    }
    return jobs, meta


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def prep_cache(mod, cache_dir: Path, dataset: str, tokenizer: str, train_tokens: int, val_tokens: int, seed: int) -> Dict[str, Any]:
    mod.set_seed(seed)
    train_t, val_t, meta = mod.build_or_load_token_cache(
        out_dir=cache_dir,
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        target_train_tokens=train_tokens,
        target_val_tokens=val_tokens,
        seed=seed,
    )
    n_seq = (train_t.numel() - 1) // 2048
    order = mod.build_train_order(n_seq, seed=seed, out_file=cache_dir / f"train_order_seed{seed}.pt")
    return {"data_meta": asdict(meta), "n_seq": int(n_seq), "order_file": str(cache_dir / f'train_order_seed{seed}.pt')}


def run_one_job(args: argparse.Namespace) -> None:
    spec_path = Path(args.job_spec)
    job = json.loads(spec_path.read_text())
    out_dir = Path(args.out_dir)
    var_dir = out_dir / "variants" / job["name"]
    var_dir.mkdir(parents=True, exist_ok=True)

    mod = load_module(Path(args.fs_script))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Load cached tokens + order
    cache_dir = Path(args.cache_dir)
    train_tokens, val_tokens, data_meta = mod.build_or_load_token_cache(
        out_dir=cache_dir,
        dataset_name=args.data_dataset,
        tokenizer_name=args.tokenizer,
        target_train_tokens=args.train_tokens,
        target_val_tokens=args.val_tokens,
        seed=args.seed,
    )
    n_seq = (train_tokens.numel() - 1) // 2048
    order = mod.build_train_order(n_seq, seed=args.seed, out_file=cache_dir / f"train_order_seed{args.seed}.pt")

    cfg = mod.TrainConfig(
        seq_len=2048,
        dim=512,
        n_layers=6,
        n_heads=8,
        intermediate=2048,
        vocab_size=50304,
        lr=6e-4,
        warmup_ratio=0.02,
        target_effective_batch=32,
    )

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_amp = not args.fp32

    rope = job["rope"]
    if rope["kind"] == "standard":
        rope_cfg = mod.RopeConfig(kind="standard", theta=float(rope["theta"]))
    elif rope["kind"] == "custom":
        rope_cfg = mod.RopeConfig(kind="custom", theta=1000.0, custom_omega=rope["custom_omega"])
    else:
        raise ValueError(f"unsupported rope kind: {rope['kind']}")

    spec = mod.VariantSpec(job["name"], rope_cfg)

    # Train -> checkpoint (temporary), then eval, then delete checkpoint.
    train_res = mod.train_one_variant(
        spec=spec,
        cfg=cfg,
        train_tokens=train_tokens,
        order=order,
        out_dir=var_dir,  # checkpoint lands here
        seed=args.seed,
        use_amp=use_amp,
        smoke_steps=0,
        amp_dtype=amp_dtype,
        save_checkpoint=True,
    )

    ckpt_path = Path(train_res["train"]["checkpoint"])

    device = torch.device("cuda")
    model = mod.TinyGPT(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        intermediate=cfg.intermediate,
        rope_cfg=rope_cfg,
    ).to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)

    eval_lengths = [2048, 16384]
    ppl = mod.eval_ppl_lengths(
        model=model,
        val_tokens=val_tokens,
        lengths=eval_lengths,
        n_chunks=args.n_eval_chunks,
        seed=args.seed + 1337,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    # delete weights (data-only intent)
    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    result = {
        "ts": now_ts(),
        "name": job["name"],
        "group": job["group"],
        "rope": job["rope"],
        "data": asdict(data_meta),
        "train": train_res["train"],
        "ppl": ppl,
    }
    _write_json(var_dir / "result.json", result)
    print(json.dumps({"name": job["name"], "ppl@16384": ppl["16384"]["mean"]}, indent=2))


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mod = load_module(Path(args.fs_script))

    # Cache prep (single-process) to avoid multi-process HF download races.
    print("[prep] building/loading token cache ...")
    prep = prep_cache(
        mod=mod,
        cache_dir=cache_dir,
        dataset=args.data_dataset,
        tokenizer=args.tokenizer,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        seed=args.seed,
    )

    # Build queue
    K = 32  # head_dim=64 -> K=32
    jobs, meta = build_job_queue(K)
    specs_dir = out_dir / "job_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        _write_json(specs_dir / f"{job['name']}.json", job)

    # Detect GPUs
    import torch

    n_gpu = torch.cuda.device_count()
    max_parallel = int(args.max_parallel)
    parallel = max(1, min(max_parallel, n_gpu if n_gpu > 0 else 1))
    print(f"[setup] cuda_device_count={n_gpu} max_parallel={max_parallel} -> parallel={parallel}")

    # Launch workers as subprocesses for isolation.
    pending = list(jobs)
    running: List[Tuple[subprocess.Popen, Dict[str, Any], int]] = []
    done: Dict[str, Any] = {}

    def read_result(name: str) -> Optional[Dict[str, Any]]:
        p = out_dir / "variants" / name / "result.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    # Resume support: skip already-finished jobs.
    new_pending = []
    for job in pending:
        if read_result(job["name"]) is not None:
            print(f"[resume] skipping already completed {job['name']}")
            continue
        new_pending.append(job)
    pending = new_pending

    gpu_ids = list(range(max(1, n_gpu)))
    gpu_rr = 0

    while pending or running:
        # Fill slots
        while pending and len(running) < parallel:
            job = pending.pop(0)
            gpu_id = gpu_ids[gpu_rr % len(gpu_ids)]
            gpu_rr += 1

            log_p = out_dir / "variants" / job["name"] / "run.log"
            log_p.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            # pin to a GPU if multiple; safe on single GPU too.
            if n_gpu > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--run_one",
                "--job_spec",
                str(specs_dir / f"{job['name']}.json"),
                "--fs_script",
                args.fs_script,
                "--out_dir",
                args.out_dir,
                "--cache_dir",
                args.cache_dir,
                "--data_dataset",
                args.data_dataset,
                "--tokenizer",
                args.tokenizer,
                "--train_tokens",
                str(args.train_tokens),
                "--val_tokens",
                str(args.val_tokens),
                "--n_eval_chunks",
                str(args.n_eval_chunks),
                "--seed",
                str(args.seed),
                "--amp_dtype",
                args.amp_dtype,
            ]
            if args.fp32:
                cmd.append("--fp32")

            with open(log_p, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            running.append((p, job, gpu_id))
            print(f"[launch] {job['name']} on gpu={gpu_id} pid={p.pid}")

        # Poll
        still: List[Tuple[subprocess.Popen, Dict[str, Any], int]] = []
        for p, job, gpu_id in running:
            rc = p.poll()
            if rc is None:
                still.append((p, job, gpu_id))
                continue
            if rc != 0:
                print(f"[error] {job['name']} exited rc={rc} (see run.log)")
                continue
            res = read_result(job["name"])
            if res is None:
                print(f"[warn] {job['name']} finished but result.json missing")
                continue
            done[job["name"]] = res
            print(f"[done] {job['name']} ppl@16384={res['ppl']['16384']['mean']:.3f}")
        running = still
        if pending or running:
            time.sleep(5)

    # Aggregate
    # Ensure we include resumed results too.
    for job in jobs:
        if job["name"] in done:
            continue
        res = read_result(job["name"])
        if res is not None:
            done[job["name"]] = res

    baseline = done.get("geo_10k_baseline", {}).get("ppl", {}).get("16384", {}).get("mean")
    if baseline is None:
        baseline = float("nan")

    summary = []
    for name, res in sorted(done.items()):
        ppl16384 = float(res["ppl"]["16384"]["mean"])
        delta = ppl16384 - float(baseline) if math.isfinite(float(baseline)) else float("nan")
        summary.append(
            {
                "name": name,
                "group": res["group"],
                "ppl_2048": float(res["ppl"]["2048"]["mean"]),
                "ppl_16384": ppl16384,
                "vs_geo_10k_delta": delta,
            }
        )

    payload = {
        "ts": now_ts(),
        "queue_meta": meta,
        "prep": prep,
        "baseline_geo_10k_ppl_16384": baseline,
        "summary": summary,
        "results": done,
    }
    out_json = out_dir / "results_h800.json"
    _write_json(out_json, payload)

    # Print quick table (sorted by ppl_16384)
    summary_sorted = sorted(summary, key=lambda x: x["ppl_16384"])
    print("\nPPL@16384 leaderboard (lower is better):")
    for row in summary_sorted:
        print(f"  {row['name']:<28} {row['ppl_16384']:.3f}  (delta vs geo_10k: {row['vs_geo_10k_delta']:+.3f})")
    print(f"\n[done] wrote {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_script", type=str, default="/opt/dfrope/from_scratch_dfrope_train_eval.py")
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/h800_parallel")
    ap.add_argument("--cache_dir", type=str, default="/opt/dfrope/results/h800_parallel/cache")
    ap.add_argument("--data_dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--train_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_500_000)
    ap.add_argument("--n_eval_chunks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--max_parallel", type=int, default=5)

    ap.add_argument("--run_one", action="store_true", help="Internal: run a single job spec.")
    ap.add_argument("--job_spec", type=str, default="")
    args = ap.parse_args()

    if args.run_one:
        if not args.job_spec:
            raise ValueError("--job_spec is required with --run_one")
        run_one_job(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
