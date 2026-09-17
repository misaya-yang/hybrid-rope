#!/usr/bin/env python3
"""Measure CA-NCP plane, prefill, decode, and cache behavior on one frozen row."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from .io_utils import atomic_json


def cuda_ms(torch, callback) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    callback()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "model_loaded": False, "gpu_execution": False,
            "measurements": ["rank2_plane", "prefill", "decode", "cache_length"],
        }))
        return
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip()
        if active:
            raise RuntimeError("a GPU process is active; runtime benchmark does not co-run or stop it")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("runtime benchmark requires CUDA")
        from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
        from scripts.experiments.cross_audit.tables import install_static
        from .runtime import install_alignment

        model, _, _ = load_model(args.model, "Native", checkpoint=None, training=False)
        table = json.loads((args.root / "construction/carrier_ncp.json").read_text())
        install_static(model, np.asarray(table["values_float32"], dtype=np.float32), 1.0)
        handles, alignment = install_alignment(model, args.root / "alignment/alignment.npz")
        pilot = json.loads((args.root / "assets/pilot/manifest.json").read_text())
        panel = Path(pilot["panel"]["source_inputs"])
        row = json.loads(next(line for line in panel.read_text().splitlines() if line.strip()))
        ids = torch.as_tensor(row["prompt_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
        layer = model.model.layers[0].self_attn
        planes = layer.ca_ncp_alignment
        heads = int(model.config.num_attention_heads)
        head_dim = int(getattr(model.config, "head_dim", None) or model.config.hidden_size // heads)
        q_shape = (1, ids.shape[1], heads, head_dim)
        q = torch.randn(q_shape, dtype=torch.bfloat16, device="cuda")
        q_map = np.repeat(
            np.arange(int(model.config.num_key_value_heads)),
            heads // int(model.config.num_key_value_heads),
        )
        with torch.inference_mode():
            model(input_ids=ids[:, :64], use_cache=False)
            plane_ms = [cuda_ms(torch, lambda: planes.apply(
                q, heads=heads, head_to_group=q_map,
            )) for _ in range(args.repeats)]
            prefill_ms, decode_ms, cache_before, cache_after = [], [], [], []
            for _ in range(args.repeats):
                holder = {}
                def prefill():
                    holder["output"] = model(input_ids=ids, use_cache=True, return_dict=True)
                prefill_ms.append(cuda_ms(torch, prefill))
                output = holder["output"]
                past = output.past_key_values
                before = int(past.get_seq_length())
                next_token = output.logits[:, -1].argmax(-1, keepdim=True)
                position = torch.arange(before, before + 1, device="cuda")
                decode_ms.append(cuda_ms(torch, lambda: model(
                    input_ids=next_token, past_key_values=past, use_cache=True,
                    cache_position=position, return_dict=True,
                )))
                cache_before.append(before)
                cache_after.append(int(past.get_seq_length()))
        for handle in handles:
            handle.remove()
    report = {
        "status": "CA_NCP_RUNTIME_COST_COMPLETE_V1_1",
        "repeats": args.repeats,
        "prompt_tokens": int(ids.shape[1]),
        "rank2_plane_q_all_layers_equivalent_note": "reported time is one layer Q path at the measured shape",
        "rank2_plane_ms": plane_ms,
        "rank2_plane_ms_median": statistics.median(plane_ms),
        "prefill_ms": prefill_ms,
        "prefill_ms_median": statistics.median(prefill_ms),
        "decode_one_token_ms": decode_ms,
        "decode_one_token_ms_median": statistics.median(decode_ms),
        "cache_length_before_decode": cache_before,
        "cache_length_after_decode": cache_after,
        "cache_increased_by_one": all(after == before + 1 for before, after in zip(cache_before, cache_after)),
        "alignment": alignment,
        "scope": "Runtime cost and cache canary only; no accuracy result.",
    }
    if not report["cache_increased_by_one"]:
        raise RuntimeError("decode cache length did not increase by one")
    atomic_json(args.root / "reports/COST_REPORT.json", report)
    print(json.dumps({"status": report["status"], "prefill_ms_median": report["prefill_ms_median"]}))


if __name__ == "__main__":
    main()
