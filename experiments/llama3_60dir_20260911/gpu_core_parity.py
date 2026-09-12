"""Real-checkpoint parity probe for the Plan B Llama-3 core operator path.

This is an implementation test, not a task score.  It runs stock plus the
named core operators on the same fixed valid token sequence, requires Native to
match stock, and requires every intervention to produce finite, non-identical
logits.  Results are written after each arm so an interruption is inspectable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import llama_runner as R
import operators as O


DEFAULT_ARMS = "Native,MR,OfficialYaRN,BM,UNI,D01a,D02a,D03a,D06a"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--native-npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--authorized-scopes", default="frequency,frequency_assignment")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.llama import modeling_llama as M

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: torch reports no CUDA device")
    if args.length < 8 or args.length > 32768:
        raise SystemExit("REFUSING: length must be in [8, 32768]")

    g = O.Geometry.from_native(np.load(args.native_npy))
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    scopes = tuple(x.strip() for x in args.authorized_scopes.split(",") if x.strip())
    allowed = set(R.CONTROLS) | set(O.config_ids())
    if not arms or any(x not in allowed for x in arms):
        raise SystemExit(f"REFUSING: unknown arm in {arms}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa").eval()
    # Valid in-vocabulary Llama-3 tokens.  Content is deliberately fixed and
    # carries no task claim; this probe tests only the real forward path.
    unit = [9906, 1917, 13, 198]
    ids = ([128000] + unit * ((args.length - 1 + len(unit) - 1) // len(unit)))[:args.length]
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)

    with torch.inference_mode():
        stock = model(input_ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    report = {
        "status": "RUNNING", "scope": "GPU operator parity; not task evidence",
        "model": args.model, "length": args.length, "arms": [],
    }
    out = Path(args.out)
    atomic_json(out, report)

    for name in arms:
        op = R.build_arm(name, g, scopes)
        cache = R.build_cache(op, g, torch, torch.float32, "cuda", args.length + 1)
        old_apply = M.apply_rotary_pos_emb
        old_rotary = M.LlamaRotaryEmbedding.forward
        patched_apply, patched_rotary = R.make_rotary_patch(torch, M, cache)
        M.apply_rotary_pos_emb = patched_apply
        M.LlamaRotaryEmbedding.forward = patched_rotary
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            with torch.inference_mode():
                logits = model(input_ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        finally:
            M.apply_rotary_pos_emb = old_apply
            M.LlamaRotaryEmbedding.forward = old_rotary
        delta = (logits - stock).abs()
        row = {
            "arm": name, "seconds": time.monotonic() - started,
            "finite": bool(torch.isfinite(logits).all()),
            "max_abs_vs_stock": float(delta.max()),
            "mean_abs_vs_stock": float(delta.mean()),
            "argmax": int(logits.argmax()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "nu_sha256": O._sha(op.nu()),
        }
        report["arms"].append(row)
        atomic_json(out, report)
        print(json.dumps(row), flush=True)
        if not row["finite"]:
            raise SystemExit(f"REFUSING: {name} produced non-finite logits")
        if name == "Native" and row["max_abs_vs_stock"] != 0.0:
            raise SystemExit(f"REFUSING: Native parity drift {row['max_abs_vs_stock']}")
        if name != "Native" and row["max_abs_vs_stock"] == 0.0:
            raise SystemExit(f"REFUSING: {name} did not trigger the intervention")

    report["status"] = "COMPLETE"
    atomic_json(out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
