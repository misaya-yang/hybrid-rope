#!/usr/bin/env python3
"""E0 probe: RoPE injection sanity + 64K/32K VRAM feasibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rope.inject import apply_inv_freq_inplace, hash_tensor_sha256
from rope.schedules import build_inv_freq, canonical_method, infer_rope_base_from_config, infer_shape_name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run E0 probe and write main_ctx decision.")
    ap.add_argument("--model", type=str, required=True, help="Base checkpoint path.")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--method_probe", type=str, default="hybrid")
    ap.add_argument("--max_headroom_threshold_gb", type=float, default=15.0)
    ap.add_argument("--output_dir", type=str, default="artifacts/results")
    ap.add_argument("--run_tag", type=str, default="")
    return ap.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


@torch.no_grad()
def forward_last_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    return out.logits[:, -1, :].float().cpu()


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_tag = args.run_tag.strip() or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"probe_{run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    probe_path = out_root / "vram_probe.json"
    main_ctx_path = out_root / "main_ctx.txt"

    payload: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "ctx": int(args.ctx),
        "batch_size": int(args.bs),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "method_probe": args.method_probe,
        "unit_tests": {},
    }

    main_ctx = "32768"

    try:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=cfg,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype(args.dtype),
            attn_implementation=args.attn_implementation,
        ).cuda().eval()
        modules = [(n, m) for n, m in model.named_modules() if hasattr(m, "inv_freq")]
        if not modules:
            raise RuntimeError("No rotary modules with inv_freq found.")
        head_dim = int(modules[0][1].inv_freq.numel() * 2)
        base = infer_rope_base_from_config(args.model, fallback=500000.0)

        vocab_size = int(getattr(cfg, "vocab_size", 128256))
        probe_ids = torch.randint(
            low=10,
            high=max(11, vocab_size - 1),
            size=(1, 512),
            dtype=torch.long,
            device="cuda",
        )

        baseline_inv = build_inv_freq("baseline", head_dim=head_dim, base=base, max_seq_len=args.ctx)
        info0 = apply_inv_freq_inplace(model, baseline_inv)
        logits_a = forward_last_logits(model, probe_ids)
        info1 = apply_inv_freq_inplace(model, baseline_inv)
        logits_b = forward_last_logits(model, probe_ids)
        baseline_max_diff = float(torch.max(torch.abs(logits_a - logits_b)).item())
        payload["unit_tests"]["baseline_reinject"] = {
            "patched_count_first": int(info0["patched_count"]),
            "patched_count_second": int(info1["patched_count"]),
            "max_abs_logit_diff": baseline_max_diff,
            "pass": bool(baseline_max_diff <= 5e-4),
        }

        probe_method = canonical_method(args.method_probe)
        method_inv = build_inv_freq(probe_method, head_dim=head_dim, base=base, max_seq_len=args.ctx)
        info2 = apply_inv_freq_inplace(model, method_inv)
        logits_m = forward_last_logits(model, probe_ids)
        method_logit_diff = float(torch.max(torch.abs(logits_m - logits_b)).item())
        payload["unit_tests"]["method_swap"] = {
            "canonical_method": probe_method,
            "shape": infer_shape_name(probe_method),
            "changed_modules": info2["changed_modules"],
            "patched_count": int(info2["patched_count"]),
            "logit_diff_vs_baseline": method_logit_diff,
            "inv_freq_sha256": hash_tensor_sha256(method_inv),
            "pass": bool(method_logit_diff > 1e-6 and len(info2["changed_modules"]) > 0),
        }

        ctx_ids = torch.randint(
            low=10,
            high=max(11, vocab_size - 1),
            size=(int(args.bs), int(args.ctx)),
            dtype=torch.long,
            device="cuda",
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(input_ids=ctx_ids, use_cache=False)
        torch.cuda.synchronize()
        dt = time.time() - t0

        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        headroom_gb = total_gb - peak_gb

        payload.update(
            {
                "success": True,
                "elapsed_sec": round(dt, 3),
                "peak_vram_gb": round(peak_gb, 3),
                "reserved_vram_gb": round(reserved_gb, 3),
                "total_vram_gb": round(total_gb, 3),
                "headroom_gb": round(headroom_gb, 3),
                "tokens_per_sec": round((args.bs * args.ctx) / max(dt, 1e-6), 3),
            }
        )
        main_ctx = "65536" if headroom_gb >= float(args.max_headroom_threshold_gb) else "32768"
    except Exception as e:
        payload.update(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        main_ctx = "32768"

    main_ctx_path.write_text(main_ctx + "\n", encoding="utf-8")
    write_json(probe_path, payload)

    # Compatibility copies for the exact plan paths.
    compat_root = Path(args.output_dir)
    write_json(compat_root / "vram_probe.json", payload)
    (compat_root / "main_ctx.txt").write_text(main_ctx + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"main_ctx={main_ctx}", flush=True)


if __name__ == "__main__":
    main()
