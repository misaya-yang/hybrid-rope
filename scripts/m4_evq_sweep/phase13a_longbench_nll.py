#!/usr/bin/env python3
"""
Phase 13A: LongBench NLL Evaluation — 750M Geo vs Hybrid
Runs qa4 tasks at ctx=2048,4096,8192 for both models.
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Checkpoint paths (Phase 9F, 100% = final) ──────────────────────────────
GEO_CKPT = "/root/autodl-tmp/evq_phase9/seed42/geo_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt"
HYBRID_CKPT = "/root/autodl-tmp/evq_phase9/seed42/hybrid1.5_r16_750m_2k_1bdata_ckpt/checkpoints/step_15258.pt"

OUTPUT_DIR = "/root/autodl-tmp/results/phase13a_longbench_nll"
SCRIPT = Path(__file__).resolve().parent / "eval_longbench_nll.py"

RUNS = [
    # (method_name, model_path, rope_type, tau, ctx_len)
    ("geo_750m", GEO_CKPT, "geo", 0.0, 2048),
    ("geo_750m", GEO_CKPT, "geo", 0.0, 4096),
    ("geo_750m", GEO_CKPT, "geo", 0.0, 8192),
    ("hybrid_750m", HYBRID_CKPT, "hybrid", 1.5, 2048),
    ("hybrid_750m", HYBRID_CKPT, "hybrid", 1.5, 4096),
    ("hybrid_750m", HYBRID_CKPT, "hybrid", 1.5, 8192),
]


def main():
    print(f"Phase 13A: LongBench NLL Evaluation")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Runs: {len(RUNS)}")
    print()

    # Verify checkpoints exist
    for ckpt in [GEO_CKPT, HYBRID_CKPT]:
        if not Path(ckpt).exists():
            print(f"  ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)
    print("  Checkpoints verified.")

    results_files = []
    for i, (method, ckpt, rope_type, tau, ctx) in enumerate(RUNS):
        print(f"\n{'#'*70}")
        print(f"# Run {i+1}/{len(RUNS)}: {method} ctx={ctx}")
        print(f"{'#'*70}\n")

        cmd = [
            sys.executable, str(SCRIPT),
            "--model_path", ckpt,
            "--tier", "750m",
            "--rope_type", rope_type,
            "--tau", str(tau),
            "--hybrid_r", "16",
            "--base", "500000.0",
            "--tasks", "qa4",
            "--max_context_len", str(ctx),
            "--max_samples", "100",
            "--method_name", method,
            "--output_dir", OUTPUT_DIR,
            "--dtype", "bfloat16",
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0

        out_file = Path(OUTPUT_DIR) / f"longbench_nll_{method}_ctx{ctx}.json"
        results_files.append(str(out_file))

        if proc.returncode != 0:
            print(f"  FAILED (return code {proc.returncode}) after {elapsed:.0f}s")
        else:
            print(f"  Done in {elapsed:.0f}s")

    # Final summary
    print(f"\n{'='*70}")
    print(f"Phase 13A COMPLETE — All results:")
    print(f"{'='*70}")

    import json
    for rf in results_files:
        if Path(rf).exists():
            with open(rf) as f:
                data = json.load(f)
            method = data.get("method", "?")
            ctx = data.get("max_context_len", "?")
            agg = data.get("results", {}).get("_aggregate", {})
            nll = agg.get("mean_nll", float('nan'))
            print(f"  {method:20s} ctx={ctx:>5}  NLL={nll:.4f}")
        else:
            print(f"  MISSING: {rf}")


if __name__ == "__main__":
    main()
