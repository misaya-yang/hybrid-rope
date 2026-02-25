#!/usr/bin/env python3
"""F1 empirical validation: attention-distance priors on long-context samples.

Outputs:
  - figures/attention_distance_loglog.png
  - figures/rho_emp_vs_theory.png
  - results/alpha_summary.csv
  - results/prior_validation_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def longbench_cfg_candidates(task: str) -> List[str]:
    base = task.strip()
    cands: List[str] = []
    for cfg in (base, base.lower(), f"{base}_e", f"{base.lower()}_e"):
        if cfg not in cands:
            cands.append(cfg)
    return cands


def load_local_longbench_jsonl(local_dir: Path, task: str) -> Optional[List[Dict[str, object]]]:
    p = local_dir / f"{task}.jsonl"
    if not p.exists():
        return None
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows if rows else None


def load_hf_longbench(task: str, split: str) -> List[Dict[str, object]]:
    errors: List[str] = []
    for cfg in longbench_cfg_candidates(task):
        try:
            ds = load_dataset("THUDM/LongBench", cfg, split=split, trust_remote_code=True)
            return [dict(ds[i]) for i in range(len(ds))]
        except Exception as exc:
            errors.append(f"cfg={cfg}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Cannot load THUDM/LongBench task={task}\n" + "\n".join(errors))


def sample_to_text(sample: Dict[str, object]) -> str:
    primary_keys = [
        "context",
        "input",
        "article",
        "document",
        "passage",
        "text",
    ]
    secondary_keys = [
        "question",
        "query",
        "instruction",
        "prompt",
    ]
    parts: List[str] = []
    for k in primary_keys + secondary_keys:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if not parts:
        for _, v in sample.items():
            if isinstance(v, str) and len(v.strip()) > 64:
                parts.append(v.strip())
    return "\n\n".join(parts)


def maybe_random_window(
    ids: Sequence[int],
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> Optional[List[int]]:
    n = len(ids)
    if n < min_len:
        return None
    if n <= max_len:
        return list(ids)
    start = rng.randint(0, n - max_len)
    return list(ids[start : start + max_len])


def collect_longbench_samples(
    tokenizer,
    tasks: Sequence[str],
    local_data_dir: Optional[Path],
    split: str,
    min_samples: int,
    max_records_per_task: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    out: List[Dict[str, object]] = []

    for task in tasks:
        rows: Optional[List[Dict[str, object]]] = None
        if local_data_dir is not None:
            rows = load_local_longbench_jsonl(local_data_dir, task)
        if rows is None:
            rows = load_hf_longbench(task, split=split)
        rng.shuffle(rows)

        keep = 0
        for i, row in enumerate(rows):
            if i >= max_records_per_task:
                break
            text = sample_to_text(row)
            if len(text) < 128:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            win = maybe_random_window(
                ids=ids,
                min_len=min_tokens,
                max_len=max_tokens,
                rng=rng,
            )
            if win is None:
                continue
            out.append(
                {
                    "task": task,
                    "source_index": i,
                    "n_tokens": len(win),
                    "input_ids": win,
                }
            )
            keep += 1
        print(f"[data] task={task} collected={keep}", flush=True)

    rng.shuffle(out)
    if len(out) < min_samples:
        raise RuntimeError(
            f"Not enough valid long samples: got={len(out)}, need>={min_samples}. "
            f"Try reducing --min_tokens or increasing --max_records_per_task."
        )
    return out[:min_samples]


def model_object_candidates(model):
    cands = [model, getattr(model, "model", None), getattr(model, "base_model", None)]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        cands.append(getattr(base_model, "model", None))
        inner = getattr(base_model, "model", None)
        if inner is not None:
            cands.append(getattr(inner, "model", None))
    return cands


def locate_layers_module(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand.layers
        if hasattr(cand, "model") and hasattr(cand.model, "layers"):
            return cand.model.layers
    raise RuntimeError("Could not locate decoder layers.")


def locate_model_rotary(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "rotary_emb"):
            return cand.rotary_emb
        if hasattr(cand, "model") and hasattr(cand.model, "rotary_emb"):
            return cand.model.rotary_emb
    return None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def repeat_kv_fallback(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
except Exception:
    hf_apply_rotary_pos_emb = None

try:
    from transformers.models.llama.modeling_llama import repeat_kv as hf_repeat_kv
except Exception:
    hf_repeat_kv = None


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if hf_apply_rotary_pos_emb is not None:
        try:
            return hf_apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            pass
    return apply_rotary_pos_emb_fallback(q, k, cos, sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if hf_repeat_kv is not None:
        try:
            return hf_repeat_kv(x, n_rep)
        except Exception:
            pass
    return repeat_kv_fallback(x, n_rep)


def infer_rope_base(cfg: AutoConfig, fallback: float = 500000.0) -> float:
    val = getattr(cfg, "rope_theta", None)
    if val is None:
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict):
            val = rs.get("rope_theta")
    try:
        out = float(val)
    except Exception:
        out = float(fallback)
    if out <= 0:
        out = float(fallback)
    return out


def accumulate_hist_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    hist: torch.Tensor,
    max_distance: int,
    chunk_q: int = 128,
) -> None:
    # q/k: [heads, L, d]
    h, q_len, d = q.shape
    k_len = k.shape[1]
    if q_len != k_len:
        raise ValueError(f"q_len != k_len ({q_len} vs {k_len}) is unsupported in this script.")

    key_pos = torch.arange(k_len, device=q.device, dtype=torch.long)
    scale = 1.0 / math.sqrt(float(d))
    for s in range(0, q_len, chunk_q):
        e = min(q_len, s + chunk_q)
        q_blk = q[:, s:e, :]  # [H, B, d]
        q_pos = torch.arange(s, e, device=q.device, dtype=torch.long)  # [B]
        logits = torch.einsum("hbd,hkd->hbk", q_blk, k) * scale  # [H, B, K]
        causal = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [B, K]
        logits = logits.masked_fill(~causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1)  # [H, B, K]

        dists = q_pos.unsqueeze(1) - key_pos.unsqueeze(0)  # [B, K]
        valid = dists >= 1
        if not bool(valid.any()):
            continue
        dists = dists.clamp(min=0, max=max_distance).to(torch.long)
        d3 = dists.unsqueeze(0).expand(h, -1, -1)
        v3 = valid.unsqueeze(0).expand(h, -1, -1)
        hist.scatter_add_(0, d3[v3], probs[v3].to(hist.dtype))


def get_layer_histogram_for_one_sample(
    model,
    layers,
    model_rotary,
    hidden_states: Sequence[torch.Tensor],
    layer_idx: int,
    max_distance: int,
    chunk_q: int,
) -> torch.Tensor:
    layer = layers[layer_idx]
    attn = layer.self_attn
    x = hidden_states[layer_idx]
    x_ln = layer.input_layernorm(x)
    seq_len = int(x_ln.size(1))
    pos_ids = torch.arange(seq_len, device=x_ln.device, dtype=torch.long).unsqueeze(0)

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and hasattr(attn, "config"):
        num_heads = getattr(attn.config, "num_attention_heads", None)
    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and hasattr(attn, "config"):
        num_kv_heads = getattr(attn.config, "num_key_value_heads", None)
    if num_heads is None:
        num_heads = x_ln.size(-1) // attn.head_dim
    if num_kv_heads is None:
        num_kv_heads = num_heads

    q = attn.q_proj(x_ln).view(1, seq_len, int(num_heads), attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x_ln).view(1, seq_len, int(num_kv_heads), attn.head_dim).transpose(1, 2)

    if hasattr(attn, "rotary_emb"):
        try:
            cos, sin = attn.rotary_emb(k, pos_ids)
        except Exception:
            cos, sin = model_rotary(k, pos_ids)
    elif model_rotary is not None:
        cos, sin = model_rotary(k, pos_ids)
    else:
        raise RuntimeError("No rotary embedding provider found.")

    q, k = apply_rotary(q, k, cos, sin)
    n_rep = int(getattr(attn, "num_key_value_groups", int(int(num_heads) // int(num_kv_heads))))
    k = repeat_kv(k, n_rep)

    q = q[0].to(dtype=torch.float32)
    k = k[0].to(dtype=torch.float32)

    hist = torch.zeros(max_distance + 1, dtype=torch.float64, device=q.device)
    accumulate_hist_from_qk(q=q, k=k, hist=hist, max_distance=max_distance, chunk_q=chunk_q)
    return hist


def fit_power_and_exp(
    hist: np.ndarray,
    d_min: int,
    d_max: Optional[int] = None,
) -> Dict[str, float]:
    if d_max is None:
        d_max = len(hist) - 1
    d_max = min(int(d_max), len(hist) - 1)
    d = np.arange(max(1, d_min), d_max + 1, dtype=np.float64)
    p = hist[d.astype(np.int64)]
    mask = np.isfinite(p) & (p > 0)
    if int(mask.sum()) < 8:
        return {
            "alpha_powerlaw": float("nan"),
            "c_powerlaw": float("nan"),
            "r2_powerlaw": float("nan"),
            "lambda_exp": float("nan"),
            "c_exp": float("nan"),
            "r2_exp": float("nan"),
            "n_points": int(mask.sum()),
        }

    d_fit = d[mask]
    p_fit = p[mask]

    # Power-law: log p = log C - alpha log d
    x1 = np.log(d_fit)
    y = np.log(p_fit)
    slope1, intercept1, r1, _, _ = stats.linregress(x1, y)
    alpha = -float(slope1)
    c_pow = float(np.exp(intercept1))
    r2_pow = float(r1**2)

    # Exponential: log p = log C - lambda d
    x2 = d_fit
    slope2, intercept2, r2, _, _ = stats.linregress(x2, y)
    lam = -float(slope2)
    c_exp = float(np.exp(intercept2))
    r2_exp = float(r2**2)

    return {
        "alpha_powerlaw": alpha,
        "c_powerlaw": c_pow,
        "r2_powerlaw": r2_pow,
        "lambda_exp": lam,
        "c_exp": c_exp,
        "r2_exp": r2_exp,
        "n_points": int(mask.sum()),
    }


def normalize_hist(hist: np.ndarray) -> np.ndarray:
    out = hist.astype(np.float64).copy()
    out[0] = 0.0
    s = float(np.sum(out))
    if s <= 0:
        return out
    return out / s


def compute_E_diag_from_prior(
    prior: np.ndarray,
    phi: np.ndarray,
    base: float,
) -> np.ndarray:
    # prior[delta], delta starts at 0 (delta=0 not used).
    delta = np.arange(1, len(prior), dtype=np.float64)
    p = prior[1:].astype(np.float64)
    if p.sum() <= 0:
        raise RuntimeError("Prior mass is zero.")
    p = p / p.sum()
    omega = base ** (-phi)  # [P]
    angles = 2.0 * omega[:, None] * delta[None, :]
    return 0.5 * (1.0 + np.sum(np.cos(angles) * p[None, :], axis=1))


def rho_from_E(E: np.ndarray, phi: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    inv = 1.0 / np.clip(E, eps, None)
    norm = np.trapezoid(inv, phi) if hasattr(np, "trapezoid") else np.trapz(inv, phi)
    return inv / max(float(norm), eps)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Empirical grounding of attention distance priors.")
    ap.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument("--tasks", type=str, default="gov_report,multi_news,qasper")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument(
        "--longbench_local_data_dir",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data",
    )
    ap.add_argument("--min_samples", type=int, default=50)
    ap.add_argument("--max_records_per_task", type=int, default=300)
    ap.add_argument("--min_tokens", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--layers", type=str, default="12,20,28")
    ap.add_argument("--fit_min_delta", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile_mode", type=str, default="reduce-overhead")
    ap.add_argument("--chunk_q", type=int, default=128)
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--figures_dir", type=str, default="figures")
    ap.add_argument("--results_dir", type=str, default="results")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tasks = parse_csv(args.tasks)
    layers_req = [int(x) for x in parse_csv(args.layers)]
    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"[model] load {args.model_path} with attn={args.attn_implementation}, dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        device_map="auto" if args.device.startswith("cuda") else None,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)  # type: ignore[assignment]
            print("[model] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[model] torch.compile failed, continue without compile: {exc}", flush=True)

    cfg = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    rope_base = infer_rope_base(cfg, fallback=500000.0)

    local_dir = Path(args.longbench_local_data_dir)
    if not local_dir.exists():
        local_dir = None
    samples = collect_longbench_samples(
        tokenizer=tokenizer,
        tasks=tasks,
        local_data_dir=local_dir,
        split=args.split,
        min_samples=args.min_samples,
        max_records_per_task=args.max_records_per_task,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(f"[data] using samples={len(samples)}", flush=True)

    layers = locate_layers_module(model)
    n_layers = len(layers)
    layer_ids = sorted(set([x for x in layers_req if 0 <= x < n_layers]))
    if not layer_ids:
        raise RuntimeError(f"No valid layers in requested list {layers_req} for model layers={n_layers}")
    print(f"[probe] selected layers={layer_ids} / n_layers={n_layers}", flush=True)

    model_rotary = locate_model_rotary(model)
    max_distance = args.max_tokens
    hist_by_layer: Dict[int, np.ndarray] = {
        li: np.zeros(max_distance + 1, dtype=np.float64) for li in layer_ids
    }
    sample_meta: List[Dict[str, object]] = []

    progress = tqdm(samples, desc="prior-validation", dynamic_ncols=True)
    for s in progress:
        ids = s["input_ids"]
        x = torch.tensor(ids, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            out = model(
                x,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")

        one = {"task": s["task"], "source_index": s["source_index"], "n_tokens": int(len(ids))}
        for li in layer_ids:
            h = get_layer_histogram_for_one_sample(
                model=model,
                layers=layers,
                model_rotary=model_rotary,
                hidden_states=hidden_states,
                layer_idx=li,
                max_distance=max_distance,
                chunk_q=args.chunk_q,
            )
            hist_np = h.detach().cpu().numpy()
            hist_by_layer[li] += hist_np
            one[f"mass_layer_{li}"] = float(hist_np.sum())
            del h
        sample_meta.append(one)

        del out, hidden_states, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate histogram across selected layers.
    hist_global = np.mean([normalize_hist(hist_by_layer[li]) for li in layer_ids], axis=0)
    hist_global = normalize_hist(hist_global)

    rows: List[Dict[str, object]] = []
    for li in layer_ids:
        p = normalize_hist(hist_by_layer[li])
        fit = fit_power_and_exp(p, d_min=args.fit_min_delta, d_max=args.max_tokens)
        rows.append(
            {
                "scope": f"layer_{li}",
                "layer": li,
                "samples": len(samples),
                **fit,
            }
        )

    fit_global = fit_power_and_exp(hist_global, d_min=args.fit_min_delta, d_max=args.max_tokens)
    rows.append(
        {
            "scope": "global_mean",
            "layer": -1,
            "samples": len(samples),
            **fit_global,
        }
    )
    alpha_df = pd.DataFrame(rows)
    alpha_csv = results_dir / "alpha_summary.csv"
    alpha_df.to_csv(alpha_csv, index=False)

    # Figure 1: empirical vs power-law/exponential on log-log.
    d = np.arange(1, len(hist_global), dtype=np.float64)
    p = hist_global[1:]
    mask = np.isfinite(p) & (p > 0)
    d_plot = d[mask]
    p_plot = p[mask]

    alpha = float(fit_global["alpha_powerlaw"])
    c_pow = float(fit_global["c_powerlaw"])
    lam = float(fit_global["lambda_exp"])
    c_exp = float(fit_global["c_exp"])

    p_pow = c_pow * np.power(d_plot, -alpha) if np.isfinite(alpha) else np.full_like(d_plot, np.nan)
    p_exp = c_exp * np.exp(-lam * d_plot) if np.isfinite(lam) else np.full_like(d_plot, np.nan)

    plt.figure(figsize=(8.2, 5.4))
    plt.loglog(d_plot, p_plot, label="Empirical p(Δ)", linewidth=2.0)
    if np.isfinite(alpha):
        plt.loglog(
            d_plot,
            p_pow,
            "--",
            label=f"Power-law fit: α={alpha:.3f}, R²={fit_global['r2_powerlaw']:.3f}",
            linewidth=1.6,
        )
    if np.isfinite(lam):
        plt.loglog(
            d_plot,
            p_exp,
            "-.",
            label=f"Exponential fit: λ={lam:.4f}, R²={fit_global['r2_exp']:.3f}",
            linewidth=1.6,
        )
    plt.xlabel("Relative distance Δ")
    plt.ylabel("p(Δ)")
    plt.title("Attention Distance Prior: Empirical vs Fitted Laws")
    plt.grid(alpha=0.25, which="both")
    plt.legend()
    fig_loglog = figures_dir / "attention_distance_loglog.png"
    plt.tight_layout()
    plt.savefig(fig_loglog, dpi=220)
    plt.close()

    # Figure 2: rho_emp vs theory curves.
    phi = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    E_emp = compute_E_diag_from_prior(hist_global, phi=phi, base=rope_base)
    rho_emp = rho_from_E(E_emp, phi=phi)

    # Power-law-theory prior with fitted alpha (fallback alpha=1.0).
    alpha_for_theory = alpha if np.isfinite(alpha) else 1.0
    prior_theory = np.zeros_like(hist_global, dtype=np.float64)
    dd = np.arange(1, len(prior_theory), dtype=np.float64)
    prior_theory[1:] = dd ** (-alpha_for_theory)
    prior_theory = normalize_hist(prior_theory)
    E_theory_diag = compute_E_diag_from_prior(prior_theory, phi=phi, base=rope_base)
    rho_theory_diag = rho_from_E(E_theory_diag, phi=phi)

    rho_cosh = np.cosh(1.0 - phi)
    rho_cosh = rho_cosh / (np.trapezoid(rho_cosh, phi) if hasattr(np, "trapezoid") else np.trapz(rho_cosh, phi))

    plt.figure(figsize=(8.2, 5.4))
    plt.plot(phi, rho_emp, label="ρ_emp(ϕ) from empirical D_emp", linewidth=2.2)
    plt.plot(phi, rho_theory_diag, "--", label="ρ_theory_diag(ϕ) ∝ 1/E_diag", linewidth=1.8)
    plt.plot(phi, rho_cosh, ":", label="ρ_theory_full(ϕ) ∝ cosh(1-ϕ)", linewidth=1.8)
    plt.xlabel("ϕ")
    plt.ylabel("Normalized density ρ(ϕ)")
    plt.title("Empirical Prior-Induced Density vs Theory")
    plt.grid(alpha=0.25)
    plt.legend()
    fig_rho = figures_dir / "rho_emp_vs_theory.png"
    plt.tight_layout()
    plt.savefig(fig_rho, dpi=220)
    plt.close()

    # Store fine-grained outputs.
    np.save(results_dir / "distance_prior_hist_global.npy", hist_global)
    np.save(results_dir / "rho_emp.npy", rho_emp)
    np.save(results_dir / "rho_theory_diag.npy", rho_theory_diag)
    np.save(results_dir / "rho_theory_cosh.npy", rho_cosh)
    pd.DataFrame(sample_meta).to_csv(results_dir / "sample_metadata.csv", index=False)

    results = {
        "config": {
            "model_path": args.model_path,
            "tasks": tasks,
            "split": args.split,
            "min_samples": int(args.min_samples),
            "min_tokens": int(args.min_tokens),
            "max_tokens": int(args.max_tokens),
            "layer_ids": layer_ids,
            "fit_min_delta": int(args.fit_min_delta),
            "attn_implementation": args.attn_implementation,
            "bf16": bool(args.bf16),
            "compile_model": bool(args.compile_model),
            "rope_base": float(rope_base),
            "chunk_q": int(args.chunk_q),
        },
        "summary": {
            "n_samples": int(len(samples)),
            "global_alpha_powerlaw": float(fit_global["alpha_powerlaw"]),
            "global_r2_powerlaw": float(fit_global["r2_powerlaw"]),
            "global_lambda_exp": float(fit_global["lambda_exp"]),
            "global_r2_exp": float(fit_global["r2_exp"]),
            "alpha_supports_heavy_tail": bool(
                np.isfinite(fit_global["alpha_powerlaw"])
                and float(fit_global["r2_powerlaw"]) >= 0.80
                and 0.5 <= float(fit_global["alpha_powerlaw"]) <= 2.5
            ),
        },
        "paths": {
            "attention_distance_loglog": str(fig_loglog),
            "rho_emp_vs_theory": str(fig_rho),
            "alpha_summary_csv": str(alpha_csv),
            "sample_metadata_csv": str(results_dir / "sample_metadata.csv"),
        },
    }
    out_json = results_dir / "prior_validation_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done] generated files:")
    print(f"  - {fig_loglog}")
    print(f"  - {fig_rho}")
    print(f"  - {alpha_csv}")
    print(f"  - {out_json}")
    print(
        f"[summary] alpha={fit_global['alpha_powerlaw']:.4f}, "
        f"R2_power={fit_global['r2_powerlaw']:.4f}, "
        f"R2_exp={fit_global['r2_exp']:.4f}"
    )


if __name__ == "__main__":
    main()
