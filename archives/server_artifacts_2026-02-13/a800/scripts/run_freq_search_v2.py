#!/usr/bin/env python3
"""Directed RoPE frequency search v2.

Stage 1 (CPU): generate candidates from four parameterizations, compute
surrogate risks, filter, and select top-5.

Stage 2 (GPU): train/eval top-5 custom frequencies with from-scratch setup,
plus reuse/evaluate geometric baselines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def now_ts() -> str:
    import time

    return time.strftime("%Y-%m-%d_%H%M%S")


def load_module(py_file: Path):
    spec = importlib.util.spec_from_file_location("fs_dfrope_freqv2", str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {py_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def geometric_freq(K: int, theta: float) -> np.ndarray:
    idx = np.arange(K, dtype=np.float64)
    return 1.0 / np.power(theta, idx / K)


def anchored_polynomial_freq(K: int, p: float, omega_max: float = 1.0, omega_min: float = 1e-4) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    log_omega = np.log(omega_max) + np.power(t, p) * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def piecewise_geometric_freq(K: int, theta_high: float, theta_low: float, split_frac: float) -> np.ndarray:
    split = int(round(K * split_frac))
    split = min(max(split, 1), K - 1)
    K_high = split
    K_low = K - split

    k_high = np.arange(K_high, dtype=np.float64)
    omega_high = 1.0 / np.power(theta_high, k_high / K_high)

    omega_boundary = omega_high[-1]
    k_low = np.arange(K_low, dtype=np.float64)
    omega_low = omega_boundary / np.power(theta_low, k_low / K_low)

    return np.concatenate([omega_high, omega_low], axis=0)


def sigmoid_freq(
    K: int,
    omega_max: float = 1.0,
    omega_min: float = 1e-4,
    steepness: float = 5.0,
    midpoint: float = 0.5,
) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    log_omega = np.log(omega_max) + s * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def anchored_poly_v2(K: int, p: float, a: float, b: float, omega_max: float = 1.0) -> np.ndarray:
    k = np.arange(K, dtype=np.float64)
    raw = np.power(a * k + b, -p)
    return raw * (omega_max / raw[0])


def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def compute_metrics(omega: np.ndarray, L_train: int, L_target: int) -> Dict[str, float]:
    delta_all = np.arange(1, L_target + 1, dtype=np.float64)
    delta_ood = np.arange(L_train + 1, L_target + 1, dtype=np.float64)

    s_all = np.cos(np.outer(delta_all, omega)).mean(axis=1)
    s_ood = np.cos(np.outer(delta_ood, omega)).mean(axis=1)

    r_collision_all = float(np.max(np.abs(s_all)))
    r_collision_ood = float(np.max(np.abs(s_ood)))

    x_train = omega * (0.5 * float(L_train))
    x_target = omega * (0.5 * float(L_target))
    V_train = 1.0 - np.square(sinc(x_train))
    V_target = 1.0 - np.square(sinc(x_target))
    dV = np.clip(V_target - V_train, a_min=0.0, a_max=None)
    r_ood = float(np.mean(dV))

    # Coupling proxy: joint penalty when both OOD increment and OOD-domain
    # collision risk are high.
    r_coupling = float(r_collision_ood * r_ood)
    r_total = float(r_collision_ood + r_ood + r_coupling)

    return {
        "r_collision_all": r_collision_all,
        "r_collision_ood": r_collision_ood,
        "r_ood": r_ood,
        "r_coupling": r_coupling,
        "r_total": r_total,
    }


def make_candidate(name: str, family: str, omega: np.ndarray, params: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "params": params,
        "omega": [float(x) for x in omega.tolist()],
        "omega_max": float(omega[0]),
        "omega_min": float(omega[-1]),
        "dynamic_range": float(omega[0] / omega[-1]),
        "metrics": metrics,
    }


def valid_omega(omega: np.ndarray) -> bool:
    if omega.ndim != 1:
        return False
    if not np.all(np.isfinite(omega)):
        return False
    if np.any(omega <= 0):
        return False
    if np.any(np.diff(omega) > 1e-12):
        return False
    return True


def print_top_table(rows: List[Dict[str, Any]], title: str, n: int) -> None:
    print(f"\n{title}")
    headers = ["rank", "name", "family", "w0", "w_last", "range", "Rcoll", "Rood", "Rcoupling", "Rtotal"]
    print(" | ".join(h.ljust(14) for h in headers))
    print("-" * 160)
    for i, c in enumerate(rows[:n], 1):
        m = c["metrics"]
        row = [
            str(i),
            c["name"],
            c["family"],
            f"{c['omega_max']:.3e}",
            f"{c['omega_min']:.3e}",
            f"{c['dynamic_range']:.3e}",
            f"{m['r_collision_ood']:.4f}",
            f"{m['r_ood']:.4f}",
            f"{m['r_coupling']:.4f}",
            f"{m['r_total']:.4f}",
        ]
        print(" | ".join(x.ljust(14) for x in row))


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    K = args.K
    L_train = args.L_train
    L_target = args.L_target

    geo10k = geometric_freq(K, 10000.0)
    geo10k_m = compute_metrics(geo10k, L_train, L_target)
    geo10k_min = float(geo10k[-1])

    candidates: List[Dict[str, Any]] = []

    # A: Anchored Polynomial
    p_vals = np.round(np.arange(0.3, 5.1, 0.1), 1)
    for p in p_vals:
        for omf in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            omega_min = geo10k_min * omf
            omega = anchored_polynomial_freq(K, p=float(p), omega_max=1.0, omega_min=float(omega_min))
            if not valid_omega(omega):
                continue
            m = compute_metrics(omega, L_train, L_target)
            name = f"anchpoly_p{p:.1f}_omf{omf:.1f}"
            candidates.append(make_candidate(name, "anchored_poly", omega, {"p": float(p), "omega_min_factor": omf}, m))

    # B: Piecewise Geometric
    for theta_high in [100, 500, 1000, 2000, 5000, 10000]:
        for theta_low in [1000, 5000, 10000, 50000, 100000]:
            if theta_low <= theta_high:
                continue
            for split in np.round(np.arange(0.2, 0.81, 0.1), 1):
                omega = piecewise_geometric_freq(K, float(theta_high), float(theta_low), float(split))
                if not valid_omega(omega):
                    continue
                m = compute_metrics(omega, L_train, L_target)
                name = f"pw_th{theta_high}_tl{theta_low}_s{split:.1f}"
                candidates.append(
                    make_candidate(
                        name,
                        "piecewise",
                        omega,
                        {"theta_high": int(theta_high), "theta_low": int(theta_low), "split_frac": float(split)},
                        m,
                    )
                )

    # C: Sigmoid
    for steepness in [2, 3, 5, 8, 12, 20]:
        for midpoint in np.round(np.arange(0.2, 0.81, 0.1), 1):
            for omf in [0.1, 0.5, 1.0, 3.0]:
                omega_min = geo10k_min * omf
                omega = sigmoid_freq(K, 1.0, float(omega_min), float(steepness), float(midpoint))
                if not valid_omega(omega):
                    continue
                m = compute_metrics(omega, L_train, L_target)
                name = f"sig_s{steepness}_m{midpoint:.1f}_omf{omf:.1f}"
                candidates.append(
                    make_candidate(
                        name,
                        "sigmoid",
                        omega,
                        {"steepness": int(steepness), "midpoint": float(midpoint), "omega_min_factor": float(omf)},
                        m,
                    )
                )

    # D: Anchored Poly V2
    p_vals = np.round(np.arange(1.0, 8.1, 0.5), 1)
    a_vals = np.round(np.arange(1.0, 15.1, 1.0), 1)
    b_vals = np.round(np.arange(0.5, 5.1, 0.5), 1)
    for p in p_vals:
        for a in a_vals:
            for b in b_vals:
                omega = anchored_poly_v2(K, float(p), float(a), float(b), 1.0)
                if not valid_omega(omega):
                    continue
                if float(omega[-1]) <= 1e-10:
                    continue
                m = compute_metrics(omega, L_train, L_target)
                name = f"anchv2_p{p:.1f}_a{a:.0f}_b{b:.1f}"
                candidates.append(make_candidate(name, "anchored_v2", omega, {"p": float(p), "a": float(a), "b": float(b)}, m))

    # Filtering constraints
    filtered: List[Dict[str, Any]] = []
    coupling_thr = float(geo10k_m["r_coupling"] * 2.0)
    omega_last_max = float(geo10k_min * 5.0)
    for c in candidates:
        if c["omega_max"] < 0.5:
            continue
        if c["omega_min"] > omega_last_max:
            continue
        if c["metrics"]["r_coupling"] > coupling_thr:
            continue
        filtered.append(c)

    filtered.sort(key=lambda x: x["metrics"]["r_total"])
    top5 = filtered[:5]
    top20 = filtered[:20]

    print(f"[search] total candidates={len(candidates)} filtered={len(filtered)}")
    print(f"[search] geo10k coupling threshold={coupling_thr:.6f}")
    print_top_table(top20, title="Top-20 (filtered, by R_total)", n=20)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_payload = {
        "ts": now_ts(),
        "config": {
            "K": K,
            "L_train": L_train,
            "L_target": L_target,
            "r_total_formula": "R_collision_ood + R_ood + R_coupling",
            "r_coupling_formula": "R_collision_ood * R_ood",
            "filters": {
                "omega_0_ge": 0.5,
                "omega_last_le": omega_last_max,
                "r_coupling_le": coupling_thr,
            },
        },
        "geo10k": {
            "omega": [float(x) for x in geo10k.tolist()],
            "omega_min": geo10k_min,
            "metrics": geo10k_m,
        },
        "counts": {"all": len(candidates), "filtered": len(filtered)},
        "top5": top5,
        "top20": top20,
        "all_filtered_sorted": filtered,
    }
    (out_dir / "search_results.json").write_text(json.dumps(search_payload, indent=2))

    # Plot frequency distributions: top5 + geo10k
    try:
        import matplotlib.pyplot as plt

        x = np.arange(K)
        plt.figure(figsize=(9, 5))
        plt.plot(x, geo10k, linewidth=2.0, label="geo_10k")
        for c in top5:
            omega = np.array(c["omega"], dtype=np.float64)
            plt.plot(x, omega, linewidth=1.2, label=c["name"])
        plt.yscale("log")
        plt.xlabel("Channel k")
        plt.ylabel("omega_k (log scale)")
        plt.title("Top-5 Frequency Distributions vs geo_10k")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "top5_vs_geo10k_freq.png", dpi=160)
        plt.close()
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    return search_payload


def maybe_reuse_baselines(baseline_json: Path, eval_lengths: List[int]) -> Dict[str, Dict[str, Any]]:
    if not baseline_json.exists():
        return {}
    raw = json.loads(baseline_json.read_text())
    variants = raw.get("variants", {})
    out: Dict[str, Dict[str, Any]] = {}
    mapping = {"geo_1k": "standard", "geo_10k": "high_theta"}
    for alias, src in mapping.items():
        if src not in variants:
            continue
        ppl = variants[src].get("ppl", {})
        if not all(str(L) in ppl for L in eval_lengths):
            continue
        out[alias] = {
            "variant": alias,
            "source": "reused_results",
            "reuse_from_variant": src,
            "reuse_from_file": str(baseline_json),
            "ppl": {str(L): ppl[str(L)] for L in eval_lengths},
        }
    return out


@torch.inference_mode()
def eval_checkpoint(mod, cfg, rope_cfg, ckpt_path: Path, val_tokens: torch.Tensor, eval_lengths: List[int], n_eval_chunks: int, seed: int):
    model = mod.TinyGPT(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        intermediate=cfg.intermediate,
        rope_cfg=rope_cfg,
    ).to(torch.device("cuda"))
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    ppl = mod.eval_ppl_lengths(model=model, val_tokens=val_tokens, lengths=eval_lengths, n_chunks=n_eval_chunks, seed=seed)
    del model
    torch.cuda.empty_cache()
    return ppl


def train_topk(args: argparse.Namespace, search_payload: Dict[str, Any]) -> Dict[str, Any]:
    mod = load_module(Path(args.script))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training stage")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    eval_lengths = [2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384]

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    cache_dir = Path(args.cache_dir)
    baseline_json = Path(args.baseline_results_json)
    baseline_ckpt_dir = Path(args.baseline_ckpt_dir)

    mod.set_seed(args.seed)
    print("[train] loading/reusing token cache ...")
    train_tokens, val_tokens, data_meta = mod.build_or_load_token_cache(
        out_dir=cache_dir,
        dataset_name=args.data_dataset,
        tokenizer_name=args.tokenizer,
        target_train_tokens=args.train_tokens,
        target_val_tokens=args.val_tokens,
        seed=args.seed,
    )
    n_seq = (train_tokens.numel() - 1) // cfg.seq_len
    order = mod.build_train_order(n_seq, seed=args.seed, out_file=cache_dir / f"train_order_seed{args.seed}.pt")

    results: Dict[str, Any] = {
        "ts": now_ts(),
        "data": {
            "dataset": data_meta.dataset,
            "tokenizer": data_meta.tokenizer,
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "seed": args.seed,
        },
        "train_config": asdict(cfg),
        "eval_lengths": eval_lengths,
        "n_eval_chunks": args.n_eval_chunks,
        "search_top5_names": [c["name"] for c in search_payload["top5"]],
        "variants": {},
    }

    reused = maybe_reuse_baselines(baseline_json, eval_lengths)
    results["variants"].update(reused)

    # Fallback to checkpoint eval if reused baseline not available.
    for alias, ck in [("geo_1k", "standard_model.pt"), ("geo_10k", "high_theta_model.pt")]:
        if alias in results["variants"]:
            continue
        ckpt = baseline_ckpt_dir / ck
        if not ckpt.exists():
            continue
        rope_cfg = mod.RopeConfig(kind="standard", theta=(1000.0 if alias == "geo_1k" else 10000.0))
        ppl = eval_checkpoint(mod, cfg, rope_cfg, ckpt, val_tokens, eval_lengths, args.n_eval_chunks, args.seed)
        results["variants"][alias] = {
            "variant": alias,
            "source": "reused_checkpoint_eval",
            "reuse_from_checkpoint": str(ckpt),
            "ppl": ppl,
        }

    for i, cand in enumerate(search_payload["top5"]):
        base_name = cand["name"]
        name = f"top{i+1}_{base_name}"
        omega = np.array(cand["omega"], dtype=np.float32)
        dyn = float(omega[0] / omega[-1])
        print(f"\n{'='*96}\n[train] {name}\n{'='*96}")
        print(f"[{name}] inv_freq={omega.tolist()}")
        print(f"[{name}] dynamic_range=max/min={dyn:.6e}")

        use_amp = not args.fp32
        if dyn > 1e8:
            print(f"[{name}] dynamic range > 1e8, forcing fp32")
            use_amp = False

        rope = mod.RopeConfig(kind="custom", theta=1000.0, custom_omega=[float(x) for x in omega.tolist()])
        spec = mod.VariantSpec(name, rope)
        train_res = mod.train_one_variant(
            spec=spec,
            cfg=cfg,
            train_tokens=train_tokens,
            order=order,
            out_dir=ckpt_dir,
            seed=args.seed,
            use_amp=use_amp,
            smoke_steps=0,
        )

        ppl = eval_checkpoint(
            mod=mod,
            cfg=cfg,
            rope_cfg=rope,
            ckpt_path=Path(train_res["train"]["checkpoint"]),
            val_tokens=val_tokens,
            eval_lengths=eval_lengths,
            n_eval_chunks=args.n_eval_chunks,
            seed=args.seed + 100 + i,
        )
        warn_2048 = float(ppl["2048"]["mean"]) > 15.0
        if warn_2048:
            print(f"[warning:{name}] PPL@2048={ppl['2048']['mean']:.3f} > 15.0")

        results["variants"][name] = {
            "variant": name,
            "source": "trained",
            "candidate_name": base_name,
            "candidate_family": cand["family"],
            "candidate_params": cand["params"],
            "candidate_metrics": cand["metrics"],
            "rope": {
                "kind": "custom",
                "inv_freq": [float(x) for x in omega.tolist()],
                "dynamic_range": dyn,
            },
            "train": train_res["train"],
            "ppl": ppl,
            "warning_2048_ppl_gt_15": warn_2048,
        }

    # Build requested table: geo_10k + top5
    ordered = ["geo_10k"] + [f"top{i+1}_{c['name']}" for i, c in enumerate(search_payload["top5"])]
    if "geo_1k" in results["variants"]:
        ordered = ["geo_1k"] + ordered

    print("\nLength | " + " | ".join(ordered))
    print("-" * (10 + 15 * len(ordered)))
    for L in eval_lengths:
        row = [str(L)]
        for n in ordered:
            d = results["variants"][n]["ppl"][str(L)]
            row.append(f"{d['mean']:.2f}±{d['std']:.2f}")
        print(" | ".join(row))

    # Success criteria
    success_flags: List[str] = []
    for n in ordered:
        if not n.startswith("top"):
            continue
        p = results["variants"][n]["ppl"]
        p2048 = float(p["2048"]["mean"])
        p8192 = float(p["8192"]["mean"])
        p16384 = float(p["16384"]["mean"])
        if p2048 <= 10.5 and p16384 < 61.60:
            success_flags.append(f"{n}: criterion1")
        if p2048 <= 11.0 and p8192 < 16.78:
            success_flags.append(f"{n}: criterion2")
    results["success_flags"] = success_flags

    out_json = out_dir / "results.json"
    payload = {
        "search": {
            "top20": search_payload["top20"],
            "top5": search_payload["top5"],
            "counts": search_payload["counts"],
            "geo10k_metrics": search_payload["geo10k"]["metrics"],
        },
        "train_eval": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {out_json}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="all", choices=["search", "train", "all"])
    ap.add_argument("--script", type=str, default="/opt/dfrope/from_scratch_dfrope_train_eval.py")
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/freq_search_v2")
    ap.add_argument("--cache_dir", type=str, default="/opt/dfrope/results/from_scratch/cache")
    ap.add_argument("--baseline_results_json", type=str, default="/opt/dfrope/results/from_scratch/results.json")
    ap.add_argument("--baseline_ckpt_dir", type=str, default="/opt/dfrope/results/from_scratch/checkpoints")
    ap.add_argument("--data_dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--train_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_500_000)
    ap.add_argument("--n_eval_chunks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--L_train", type=int, default=2048)
    ap.add_argument("--L_target", type=int, default=16384)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_payload = None
    search_json = out_dir / "search_results.json"
    if args.stage in ("search", "all"):
        search_payload = run_search(args)
    if args.stage in ("train", "all"):
        if search_payload is None:
            if not search_json.exists():
                raise FileNotFoundError(f"missing search results: {search_json}")
            search_payload = json.loads(search_json.read_text())
        train_topk(args, search_payload)


if __name__ == "__main__":
    main()

