#!/usr/bin/env python3
"""P2 capability probe — staged VRAM/memory boundary test for the 4080.

Fail-fast, never forces anything, records the exact boundary. Stages:
  0  environment report (GPU, VRAM, torch, disk)
  1  1B bf16 baseline: load + forwards at 2K/8K/16K (must pass; sanity)
  2  7B bf16 CPU load (records CPU RAM), then .to(cuda) — expected to be the
     first failure point on a 16GB card (~15.2GB weights alone)
  3  (only if stage 2 passed) 7B forwards at 1K/2K/4K/8K/16K, OOM-guarded
  Writes probe_result.json after EVERY stage (crash-safe). A failed stage is
  recorded as the boundary and the probe stops — that result is the answer.

Usage:
  python probe_capability.py --model-1b <1B path> --model-7b <7B path> \
      --out $B12/probe4080 [--skip-1b]
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch


def env_report() -> dict:
    rep = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rep["gpu"] = p.name
        rep["vram_total_gb"] = round(p.total_memory / 1e9, 2)
        rep["compute_capability"] = f"{p.major}.{p.minor}"
    try:
        df = shutil.disk_usage("/root/autodl-tmp")
        rep["disk_autodl_tmp_free_gb"] = round(df.free / 1e9, 2)
    except Exception:
        pass
    try:
        rep["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv"], capture_output=True, text=True,
            timeout=15).stdout.strip()
    except Exception as e:
        rep["nvidia_smi_error"] = str(e)
    return rep


@torch.inference_mode()
def timed_forward(model, ids, device):
    t = time.time()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model(x, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    return time.time() - t


def stage_1b(model_path, device, result):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).eval().to(device)
    ids = (list(range(1000, 2000)) * 32)[:16384]
    st = {}
    for L in (2048, 8192, 16384):
        torch.cuda.reset_peak_memory_stats()
        dt = timed_forward(model, ids[:L], device)
        st[f"forward_{L}"] = {
            "seconds": round(dt, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    result["stage1_1b_baseline"] = {"status": "PASS", **st}
    del model
    gc.collect()
    torch.cuda.empty_cache()


def stage_7b(model_path, device, result, skip_stage3=False):
    from transformers import AutoModelForCausalLM
    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True)
        model.eval()
        load_cpu = {"status": "PASS", "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        result["stage2_7b_load"] = {"status": "FAIL_CPU_LOAD", "error": repr(e)[:500]}
        return
    try:
        torch.cuda.reset_peak_memory_stats()
        model.to(device)
        torch.cuda.synchronize()
        load_gpu = {"status": "PASS",
                    "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3)}
    except torch.cuda.OutOfMemoryError as e:
        result["stage2_7b_load"] = {**load_cpu, "to_cuda": "FAIL_OOM",
                                    "error": repr(e)[:500],
                                    "boundary": "7B bf16 weights do not fit in VRAM"}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return
    except Exception as e:
        result["stage2_7b_load"] = {**load_cpu, "to_cuda": "FAIL",
                                    "error": repr(e)[:500]}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return
    result["stage2_7b_load"] = {**load_cpu, "to_cuda": load_gpu}
    if skip_stage3:
        del model
        return
    ids = (list(range(1000, 2000)) * 32)[:16384]
    st = {}
    for L in (1024, 2048, 4096, 8192, 16384):
        try:
            torch.cuda.reset_peak_memory_stats()
            dt = timed_forward(model, ids[:L], device)
            st[f"forward_{L}"] = {
                "status": "PASS", "seconds": round(dt, 3),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
        except torch.cuda.OutOfMemoryError as e:
            st[f"forward_{L}"] = {"status": "FAIL_OOM", "error": repr(e)[:300]}
            st["boundary"] = f"7B bf16 fits loaded but OOM at forward length {L}"
            break
    result["stage3_7b_forward"] = st
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-1b", required=True)
    ap.add_argument("--model-7b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-1b", action="store_true")
    ap.add_argument("--skip-stage3", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "probe_result.json"
    result = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "purpose": "capability boundary probe; a recorded OOM IS the result"}
    result["stage0_env"] = env_report()
    result_path.write_text(json.dumps(result, indent=2))

    if not args.skip_1b:
        try:
            stage_1b(args.model_1b, args.device, result)
        except Exception as e:
            result["stage1_1b_baseline"] = {"status": "FAIL", "error": repr(e)[:500]}
        result_path.write_text(json.dumps(result, indent=2))

    stage_7b(args.model_7b, args.device, result, skip_stage3=args.skip_stage3)
    result_path.write_text(json.dumps(result, indent=2))
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
