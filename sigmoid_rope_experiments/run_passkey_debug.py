#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, env_info, get_device, load_json, save_json, set_seed
from src.visualization import save_fig_both, set_plot_style

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class RotaryModuleInfo:
    name: str
    module: torch.nn.Module
    orig_inv: torch.Tensor
    orig_forward: object


def ensure_dependencies() -> None:
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("transformers", "transformers"),
        ("seaborn", "seaborn"),
    ]
    missing: List[str] = []
    for mod, pip_name in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    for pip_name in missing:
        print(f"[deps] installing {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Passkey debug for Sigmoid-RoPE injection.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Local model path. If empty, use --local_model_candidates first existing.",
    )
    ap.add_argument(
        "--local_model_candidates",
        type=str,
        default=(
            "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct,"
            "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
        ),
    )
    ap.add_argument("--lengths", type=str, default="4096,8192,16384,32768")
    ap.add_argument("--ratios", type=str, default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument(
        "--base_override",
        type=float,
        default=0.0,
        help="If >0, force this base instead of model rope_theta.",
    )
    ap.add_argument("--force_monkey_patch", action="store_true")
    ap.add_argument("--quick_probe_only", action="store_true")
    return ap.parse_args()


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def infer_rope_theta(config) -> float:
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        try:
            return float(rope_theta)
        except Exception:
            pass

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rt = rope_scaling.get("rope_theta", None)
        if rt is not None:
            try:
                return float(rt)
            except Exception:
                pass

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        try:
            return float(rope_parameters["rope_theta"])
        except Exception:
            pass
    return 10000.0


def tensor_stats(t: torch.Tensor) -> Dict:
    x = t.detach().float().cpu()
    n = int(x.numel())
    first_k = min(5, n)
    last_k = min(5, n)
    diff = x[1:] - x[:-1] if n > 1 else torch.tensor([], dtype=x.dtype)
    descending = bool(torch.all(diff <= 1e-12).item()) if diff.numel() > 0 else True
    ascending = bool(torch.all(diff >= -1e-12).item()) if diff.numel() > 0 else True
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "first_5": [float(v) for v in x[:first_k].tolist()],
        "last_5": [float(v) for v in x[-last_k:].tolist()],
        "descending": descending,
        "ascending": ascending,
    }


def print_stats(prefix: str, t: torch.Tensor) -> None:
    s = tensor_stats(t)
    print(
        f"{prefix}: shape={tuple(s['shape'])} dtype={s['dtype']} "
        f"min={s['min']:.8g} max={s['max']:.8g} descending={s['descending']}"
    )
    print(f"  first_5={s['first_5']}")
    print(f"  last_5={s['last_5']}")


def compare_stats(name: str, ref: torch.Tensor, cand: torch.Tensor) -> None:
    rd = ref.detach().float().cpu()
    cd = cand.detach().float().cpu()
    shape_ok = tuple(rd.shape) == tuple(cd.shape)
    max_abs = float(torch.max(torch.abs(rd - cd)).item()) if shape_ok else float("nan")
    print(
        f"[compare:{name}] shape_match={shape_ok} "
        f"ref_range=[{rd.min().item():.8g},{rd.max().item():.8g}] "
        f"cand_range=[{cd.min().item():.8g},{cd.max().item():.8g}] "
        f"max_abs_diff={max_abs:.8g}"
    )


def clear_rotary_cache(module: torch.nn.Module, name: str, verbose: bool = True) -> List[str]:
    cleared: List[str] = []
    known = [
        "_cos_cached",
        "_sin_cached",
        "cos_cached",
        "sin_cached",
        "_cos_cache",
        "_sin_cache",
    ]
    for attr in known:
        if hasattr(module, attr):
            try:
                delattr(module, attr)
                cleared.append(attr)
            except Exception:
                pass

    for attr in list(vars(module).keys()):
        al = attr.lower()
        if ("cos" in al or "sin" in al) and ("cache" in al or "cached" in al):
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                    if attr not in cleared:
                        cleared.append(attr)
                except Exception:
                    pass
    if verbose and cleared:
        print(f"  clearing cache attrs in {name}: {sorted(set(cleared))}")
    return sorted(set(cleared))


def assign_inv_freq(module: torch.nn.Module, new_inv: torch.Tensor) -> None:
    dst = new_inv.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype).contiguous()
    try:
        module.register_buffer("inv_freq", dst, persistent=False)
    except Exception:
        module.inv_freq = dst
    if hasattr(module, "original_inv_freq"):
        try:
            module.register_buffer("original_inv_freq", dst.clone(), persistent=False)
        except Exception:
            module.original_inv_freq = dst.clone()
    if hasattr(module, "max_seq_len_cached"):
        try:
            module.max_seq_len_cached = 0
        except Exception:
            pass


def maybe_monkey_patch_forward(module: torch.nn.Module, inv_freq: torch.Tensor) -> None:
    attention_scaling = getattr(module, "attention_scaling", 1.0)
    inv = inv_freq.detach().float().cpu().clone()

    @torch.no_grad()
    def patched_forward(self, x, position_ids):
        inv_local = inv.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_local[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    module.forward = patched_forward.__get__(module, module.__class__)


def collect_rotary_modules(model: torch.nn.Module) -> List[RotaryModuleInfo]:
    out: List[RotaryModuleInfo] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq"):
            out.append(
                RotaryModuleInfo(
                    name=name,
                    module=module,
                    orig_inv=module.inv_freq.detach().clone(),
                    orig_forward=module.forward,
                )
            )
    return out


def apply_freq_to_all(
    infos: List[RotaryModuleInfo],
    inv_freq: torch.Tensor,
    use_monkey_patch: bool,
    verbose: bool = False,
) -> None:
    for info in infos:
        assign_inv_freq(info.module, inv_freq)
        clear_rotary_cache(info.module, info.name, verbose=verbose)
        if use_monkey_patch:
            maybe_monkey_patch_forward(info.module, inv_freq=inv_freq)
        else:
            info.module.forward = info.orig_forward


def restore_original(infos: List[RotaryModuleInfo], use_monkey_patch: bool, verbose: bool = False) -> None:
    for info in infos:
        assign_inv_freq(info.module, info.orig_inv)
        clear_rotary_cache(info.module, info.name, verbose=verbose)
        if use_monkey_patch:
            maybe_monkey_patch_forward(info.module, inv_freq=info.orig_inv)
        else:
            info.module.forward = info.orig_forward


def detect_model_path(arg_path: str, candidates: str) -> Optional[str]:
    if arg_path and os.path.exists(arg_path):
        return arg_path
    for p in [x.strip() for x in candidates.split(",") if x.strip()]:
        if os.path.exists(p):
            return p
    return None


def load_formula_params(root_dir: Path, head_dim: int, l_target: int) -> Dict[str, float]:
    phase3_summary = root_dir / "data" / "phase3" / "summary.json"
    if phase3_summary.exists():
        payload = load_json(phase3_summary, default={})
        sens = payload.get("sensitivity", {})
        k_formula = sens.get("k_formula", None)
        x0_formula = sens.get("x0_formula", None)
        k_opt = sens.get("k_opt", None)
        x0_opt = sens.get("x0_opt", None)
        if all(v is not None for v in [k_formula, x0_formula, k_opt, x0_opt]):
            return {
                "k_formula": float(k_formula),
                "x0_formula": float(x0_formula),
                "k_opt": float(k_opt),
                "x0_opt": float(x0_opt),
                "source": str(phase3_summary),
            }

    fine_csv = root_dir / "data" / "fine_search_results.csv"
    if fine_csv.exists():
        df = pd.read_csv(fine_csv)
        row = df[(df["d"] == head_dim) & (df["L"] == l_target)]
        if not row.empty:
            r = row.iloc[0]
            return {
                "k_formula": float(r["k_optimal"]),
                "x0_formula": float(r["x0_optimal"]),
                "k_opt": float(r["k_optimal"]),
                "x0_opt": float(r["x0_optimal"]),
                "source": str(fine_csv),
            }
    return {
        "k_formula": 16.0 / float(head_dim),
        "x0_formula": 0.47 * float(head_dim),
        "k_opt": 16.0 / float(head_dim),
        "x0_opt": 0.49 * float(head_dim),
        "source": "fallback",
    }

def probe_rotary_outputs(
    infos: List[RotaryModuleInfo],
    device: torch.device,
    seq_len: int = 16,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for info in infos:
        mod = info.module
        inv_n = int(mod.inv_freq.numel())
        head_dim = inv_n * 2
        x = torch.zeros((1, 1, seq_len, head_dim), dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, device=device)
        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        try:
            with torch.no_grad():
                y = mod(x, pos)
            if isinstance(y, tuple) and len(y) >= 2 and hasattr(y[0], "shape") and hasattr(y[1], "shape"):
                cos, sin = y[0], y[1]
                out[info.name] = {
                    "cos_shape": list(cos.shape),
                    "sin_shape": list(sin.shape),
                    "cos_sample": [float(v) for v in cos[0, 0, :4].detach().float().cpu().tolist()],
                    "sin_sample": [float(v) for v in sin[0, 0, :4].detach().float().cpu().tolist()],
                }
            else:
                out[info.name] = {"output_type": str(type(y))}
        except Exception as ex:
            out[info.name] = {"error": f"{type(ex).__name__}: {ex}"}
    return out


def capture_hook_shapes(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_input_tokens: int = 16,
) -> Dict[str, Dict]:
    captured: Dict[str, Dict] = {}
    hooks = []

    def make_hook(name: str):
        def fn(module, inputs, output):
            in_shapes = []
            for item in inputs:
                if hasattr(item, "shape"):
                    in_shapes.append(list(item.shape))
                else:
                    in_shapes.append(str(type(item)))
            if isinstance(output, tuple):
                out_shapes = [list(x.shape) if hasattr(x, "shape") else str(type(x)) for x in output]
            else:
                out_shapes = [list(output.shape)] if hasattr(output, "shape") else [str(type(output))]
            captured[name] = {"input_shapes": in_shapes, "output_shapes": out_shapes}

        return fn

    for name, module in model.named_modules():
        nl = name.lower()
        if "rope" in nl or "rotary" in nl:
            hooks.append(module.register_forward_hook(make_hook(name)))

    text = "Hello world. This is a rope hook probe."
    ids = tokenizer(text, return_tensors="pt")
    ids = {k: v[:, :max_input_tokens].to(device) for k, v in ids.items()}
    try:
        with torch.no_grad():
            model(**ids)
    finally:
        for h in hooks:
            h.remove()
    return captured


def build_passkey_instance(
    tokenizer,
    context_length: int,
    passkey: str,
    position_ratio: float,
    filler_ids: List[int],
) -> Optional[List[int]]:
    ask_ids = tokenizer.encode("\nWhat is the pass key? The pass key is ", add_special_tokens=False)
    needle_ids = tokenizer.encode(
        f" The pass key is {passkey}. Remember it. {passkey} is the pass key. ",
        add_special_tokens=False,
    )
    budget = context_length - len(ask_ids) - len(needle_ids) - 4
    if budget < 32:
        return None
    rep = budget // len(filler_ids) + 2
    body = (filler_ids * rep)[:budget]
    pos = int(position_ratio * max(1, len(body) - 1))
    seq = body[:pos] + needle_ids + body[pos:] + ask_ids
    seq = seq[:context_length]
    if tokenizer.bos_token_id is not None:
        seq = [tokenizer.bos_token_id] + seq
    return seq


def evaluate_passkey(
    model,
    tokenizer,
    methods: Sequence[Tuple[str, Optional[torch.Tensor]]],
    infos: List[RotaryModuleInfo],
    lengths: List[int],
    ratios: List[float],
    repeats: int,
    max_new_tokens: int,
    use_monkey_patch: bool,
) -> pd.DataFrame:
    filler = "The grass is green. The sky is blue. The sun is yellow. "
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    device = next(model.parameters()).device
    rows: List[Dict] = []
    digit_pattern = re.compile(r"\d+")

    for method_name, inv in methods:
        if inv is None:
            restore_original(infos, use_monkey_patch=use_monkey_patch)
        else:
            apply_freq_to_all(infos, inv_freq=inv, use_monkey_patch=use_monkey_patch)

        pbar = tqdm(total=len(lengths) * len(ratios) * repeats, desc=f"passkey-{method_name}", dynamic_ncols=True)
        for lval in lengths:
            for rval in ratios:
                for rep_idx in range(repeats):
                    passkey = f"{random.randint(10000, 99999)}"
                    seq = build_passkey_instance(
                        tokenizer=tokenizer,
                        context_length=lval,
                        passkey=passkey,
                        position_ratio=rval,
                        filler_ids=filler_ids,
                    )
                    if seq is None:
                        rows.append(
                            {
                                "method": method_name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": 0,
                                "status": "invalid",
                                "pred_digits": "",
                                "passkey": passkey,
                            }
                        )
                        pbar.update(1)
                        continue

                    x = torch.tensor([seq], dtype=torch.long, device=device)
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                input_ids=x,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                            )
                        gen = tokenizer.decode(out[0, x.shape[1] :], skip_special_tokens=True)
                        digits = "".join(digit_pattern.findall(gen))
                        ok = int(passkey in digits)
                        rows.append(
                            {
                                "method": method_name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": ok,
                                "status": "ok",
                                "pred_digits": digits,
                                "passkey": passkey,
                                "generated_text": gen,
                            }
                        )
                    except RuntimeError as ex:
                        rows.append(
                            {
                                "method": method_name,
                                "context_length": lval,
                                "position_ratio": rval,
                                "repeat": rep_idx,
                                "correct": 0,
                                "status": "oom" if "out of memory" in str(ex).lower() else "error",
                                "pred_digits": "",
                                "passkey": passkey,
                                "error": str(ex),
                            }
                        )
                        cleanup_cuda()
                    pbar.update(1)
        pbar.close()
    return pd.DataFrame(rows)


def print_accuracy_tables(df: pd.DataFrame, method_order: Sequence[str]) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        print("[result] no successful passkey generations.")
        return

    g1 = ok_df.groupby("method", as_index=False)["correct"].mean()
    g2 = ok_df.groupby(["method", "context_length"], as_index=False)["correct"].mean()

    print("\n+------------------------+----------+")
    print("| Method                 | Accuracy |")
    print("+------------------------+----------+")
    for m in method_order:
        row = g1[g1["method"] == m]
        acc = float(row["correct"].iloc[0]) if not row.empty else float("nan")
        print(f"| {m:<22} | {acc:>8.3f} |")
    print("+------------------------+----------+")

    print("\n+------------------------+---------+----------+")
    print("| Method                 | Length  | Accuracy |")
    print("+------------------------+---------+----------+")
    for m in method_order:
        sub = g2[g2["method"] == m].sort_values("context_length")
        for _, r in sub.iterrows():
            print(f"| {m:<22} | {int(r['context_length']):>7d} | {float(r['correct']):>8.3f} |")
    print("+------------------------+---------+----------+")


def plot_passkey(df: pd.DataFrame, out_heatmap_stem: Path, out_line_stem: Path, method_order: Sequence[str]) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    set_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 7.8), sharey=True)
    axes_flat = axes.flatten()
    im = None
    for ax, m in zip(axes_flat, method_order):
        sub = ok_df[ok_df["method"] == m]
        pivot = sub.groupby(["context_length", "position_ratio"], as_index=False)["correct"].mean().pivot(
            index="context_length",
            columns="position_ratio",
            values="correct",
        )
        ax.set_title(m)
        if pivot.empty:
            ax.axis("off")
            continue
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index])
        ax.set_xlabel("Passkey Position Ratio")
        ax.set_ylabel("Context Length")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.92)
        cbar.set_label("Accuracy")
    fig.tight_layout()
    save_fig_both(fig, out_heatmap_stem)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.6))
    mean_len = ok_df.groupby(["method", "context_length"], as_index=False)["correct"].mean()
    for m in method_order:
        sub = mean_len[mean_len["method"] == m].sort_values("context_length")
        if sub.empty:
            continue
        ax2.plot(sub["context_length"], sub["correct"], marker="o", label=m)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title("Passkey Retrieval by Length (v3)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, out_line_stem)
    plt.close(fig2)

def select_supported_lengths(
    model,
    tokenizer,
    lengths: List[int],
    ratio: float,
    max_new_tokens: int,
) -> Tuple[List[int], Dict[int, str]]:
    filler_ids = tokenizer.encode("The grass is green. The sky is blue. The sun is yellow. ", add_special_tokens=False)
    device = next(model.parameters()).device
    ok_lengths: List[int] = []
    reasons: Dict[int, str] = {}
    for lval in lengths:
        seq = build_passkey_instance(
            tokenizer=tokenizer,
            context_length=lval,
            passkey="12345",
            position_ratio=ratio,
            filler_ids=filler_ids,
        )
        if seq is None:
            reasons[lval] = "invalid_context_budget"
            continue
        x = torch.tensor([seq], dtype=torch.long, device=device)
        try:
            with torch.no_grad():
                model.generate(
                    input_ids=x,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            ok_lengths.append(lval)
        except RuntimeError as ex:
            reasons[lval] = f"runtime_error:{type(ex).__name__}"
            cleanup_cuda()
        except Exception as ex:
            reasons[lval] = f"error:{type(ex).__name__}"
    return ok_lengths, reasons


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    set_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    result_dir = root_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[debug] device:", device)

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as ex:
        raise RuntimeError(f"transformers import failed: {ex}") from ex

    model_path = detect_model_path(args.model_path, args.local_model_candidates)
    if model_path is None:
        raise FileNotFoundError("No local model path found. Provide --model_path.")
    print(f"[debug] model_path: {model_path}")
    print(f"[debug] transformers version: {transformers.__version__}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    load_kwargs["device_map"] = "auto" if device.type == "cuda" else "cpu"

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    print(f"[debug] model loaded in {time.time() - t_load:.2f}s")

    print("\n=== Step 4: RoPE Architecture Diagnostics ===")
    print(f"Model config rope_theta: {getattr(model.config, 'rope_theta', 'N/A')}")
    print(f"Model config rope_scaling: {getattr(model.config, 'rope_scaling', 'N/A')}")
    print(f"Model config max_position_embeddings: {getattr(model.config, 'max_position_embeddings', 'N/A')}")
    for name, module in model.named_modules():
        nl = name.lower()
        if "rope" in nl or "rotary" in nl:
            attrs = [a for a in dir(module) if not a.startswith("_")]
            print(f"\n=== {name} ===")
            print(f"  Type: {type(module)}")
            print(f"  Attributes(sample): {attrs[:30]}")
            if hasattr(module, "inv_freq"):
                print(f"  inv_freq shape={tuple(module.inv_freq.shape)} dtype={module.inv_freq.dtype}")
            if hasattr(module, "config"):
                print(f"  module.config.rope_theta={getattr(module.config, 'rope_theta', 'N/A')}")
            try:
                print(f"  forward signature: {inspect.signature(module.forward)}")
            except Exception:
                pass

    infos = collect_rotary_modules(model)
    if not infos:
        raise RuntimeError("No module with inv_freq found.")
    print(f"\n[debug] rotary modules with inv_freq: {len(infos)}")
    for info in infos:
        print(f"  - {info.name}: shape={tuple(info.orig_inv.shape)} dtype={info.orig_inv.dtype}")

    head_dim = int(infos[0].orig_inv.numel() * 2)
    rope_theta_model = infer_rope_theta(model.config)
    rope_base = float(args.base_override) if args.base_override > 0 else float(rope_theta_model)
    print(f"[debug] inferred rope_theta(model)={rope_theta_model}")
    print(f"[debug] using base for injected freqs={rope_base}")

    params = load_formula_params(root_dir=root_dir, head_dim=head_dim, l_target=131072)
    print(f"[debug] k/x0 source: {params['source']}")
    print(
        "[debug] k_formula={:.8f}, x0_formula={:.4f}, k_opt={:.8f}, x0_opt={:.4f}".format(
            params["k_formula"],
            params["x0_formula"],
            params["k_opt"],
            params["x0_opt"],
        )
    )

    print("\n=== Step 1: inv_freq Meaning / Direction Check ===")
    for info in infos:
        print_stats(f"[orig] {info.name}.inv_freq", info.orig_inv)

    alloc_model_base = RoPEFrequencyAllocator(d=head_dim, base=rope_base)
    alloc_10k = RoPEFrequencyAllocator(d=head_dim, base=10000.0)
    standard_reinject = alloc_model_base.standard()
    sig_formula = alloc_model_base.sigmoid(k=float(params["k_formula"]), x0=float(params["x0_formula"]))
    sig_grid = alloc_model_base.sigmoid(k=float(params["k_opt"]), x0=float(params["x0_opt"]))
    sig_formula_10k = alloc_10k.sigmoid(k=float(params["k_formula"]), x0=float(params["x0_formula"]))

    print_stats("[cand] standard_reinject", standard_reinject)
    print_stats("[cand] sigmoid_formula(base=model)", sig_formula)
    print_stats("[cand] sigmoid_grid(base=model)", sig_grid)
    print_stats("[cand] sigmoid_formula(base=10000)", sig_formula_10k)
    compare_stats("orig_vs_standard_reinject", infos[0].orig_inv, standard_reinject)
    compare_stats("orig_vs_sigmoid_formula(base=model)", infos[0].orig_inv, sig_formula)
    compare_stats("orig_vs_sigmoid_formula(base=10000)", infos[0].orig_inv, sig_formula_10k)

    print("\n=== Step 2: Apply inv_freq + Clear Cache ===")
    use_monkey_patch = bool(args.force_monkey_patch)
    print(f"[debug] force_monkey_patch={use_monkey_patch}")
    apply_freq_to_all(infos, inv_freq=standard_reinject, use_monkey_patch=use_monkey_patch, verbose=True)
    for info in infos:
        print_stats(f"[after_apply_standard] {info.name}.inv_freq", info.module.inv_freq)
    restore_original(infos, use_monkey_patch=use_monkey_patch, verbose=True)
    for info in infos:
        print_stats(f"[after_restore_orig] {info.name}.inv_freq", info.module.inv_freq)

    print("\n=== Step 5: Probe Actual cos/sin and Hook Shapes ===")
    probe_original = probe_rotary_outputs(infos, device=device, seq_len=16)
    apply_freq_to_all(infos, inv_freq=sig_formula, use_monkey_patch=use_monkey_patch, verbose=False)
    probe_sigmoid = probe_rotary_outputs(infos, device=device, seq_len=16)
    restore_original(infos, use_monkey_patch=use_monkey_patch, verbose=False)
    hooks_original = capture_hook_shapes(model=model, tokenizer=tokenizer, device=next(model.parameters()).device, max_input_tokens=16)
    print("[probe] original rotary outputs:", json.dumps(probe_original, indent=2, ensure_ascii=False))
    print("[probe] sigmoid rotary outputs:", json.dumps(probe_sigmoid, indent=2, ensure_ascii=False))
    print("[probe] hook captured modules:", list(hooks_original.keys()))

    print("\n=== Step 3: 4-way Control Test + Full Passkey ===")
    req_lengths = parse_int_list(args.lengths)
    ratios = parse_float_list(args.ratios)
    supported_lengths, skip_reasons = select_supported_lengths(
        model=model,
        tokenizer=tokenizer,
        lengths=req_lengths,
        ratio=0.5,
        max_new_tokens=args.max_new_tokens,
    )
    if not supported_lengths:
        raise RuntimeError(f"No supported lengths from requested {req_lengths}. Reasons={skip_reasons}")
    if skip_reasons:
        print(f"[warn] some lengths skipped in probe: {skip_reasons}")
    print(f"[debug] passkey supported lengths: {supported_lengths}")

    method_order = [
        "Original",
        "Standard(re-inject)",
        "Sigmoid(Formula)",
        "Sigmoid(Grid-Opt)",
    ]
    methods: List[Tuple[str, Optional[torch.Tensor]]] = [
        ("Original", None),
        ("Standard(re-inject)", standard_reinject),
        ("Sigmoid(Formula)", sig_formula),
        ("Sigmoid(Grid-Opt)", sig_grid),
    ]

    run_lengths = supported_lengths
    run_repeats = int(args.repeats)
    if args.quick_probe_only:
        run_lengths = supported_lengths[:1]
        run_repeats = min(2, args.repeats)
        print(f"[debug] quick_probe_only enabled: lengths={run_lengths}, repeats={run_repeats}")

    pass_df = evaluate_passkey(
        model=model,
        tokenizer=tokenizer,
        methods=methods,
        infos=infos,
        lengths=run_lengths,
        ratios=ratios,
        repeats=run_repeats,
        max_new_tokens=int(args.max_new_tokens),
        use_monkey_patch=use_monkey_patch,
    )
    out_csv = data_dir / "passkey_results_v3.csv"
    pass_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[result] saved: {out_csv}")

    print_accuracy_tables(pass_df, method_order=method_order)

    heat_stem = result_dir / "passkey_retrieval_v3"
    line_stem = result_dir / "passkey_retrieval_by_length_v3"
    plot_passkey(pass_df, out_heatmap_stem=heat_stem, out_line_stem=line_stem, method_order=method_order)
    print(f"[result] saved: {heat_stem.with_suffix('.pdf')}")
    print(f"[result] saved: {line_stem.with_suffix('.pdf')}")

    diagnostics = {
        "env": env_info(),
        "transformers_version": transformers.__version__,
        "model_path": model_path,
        "rope_theta_model": rope_theta_model,
        "rope_base_used": rope_base,
        "head_dim": head_dim,
        "params_source": params["source"],
        "k_formula": params["k_formula"],
        "x0_formula": params["x0_formula"],
        "k_opt": params["k_opt"],
        "x0_opt": params["x0_opt"],
        "requested_lengths": req_lengths,
        "supported_lengths": supported_lengths,
        "skipped_lengths_reasons": skip_reasons,
        "ratios": ratios,
        "repeats": run_repeats,
        "force_monkey_patch": use_monkey_patch,
        "probe_original": probe_original,
        "probe_sigmoid": probe_sigmoid,
        "hook_shapes_original": hooks_original,
        "passkey_csv": str(out_csv),
        "elapsed_sec": time.time() - t_load,
    }
    diag_path = data_dir / "passkey_debug_diagnostics.json"
    save_json(diag_path, diagnostics)
    print(f"[result] saved diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
