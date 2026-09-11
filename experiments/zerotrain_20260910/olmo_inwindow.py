#!/usr/bin/env python3
"""The in-window constraint curve ON OLMo -- closing the analysis's weakest link.

WHY THIS EXISTS.  Every claim about the constraint side of the KKT currently rests
on a CROSS-CHECKPOINT substitution: the in-window cost curve was measured on
Qwen2.5-3B (window 32768, 16 FineWeb-Edu documents, tail-512 NLL) and the scores
it was compared against are OLMo's 16K RULER numbers.  The knee was found at
mean(m) ~ 0.63-0.65 on Qwen and the OLMo winners sit at 0.656-0.662.  That
coincidence is the single most load-bearing number in the campaign and it has
never been checked on the checkpoint it is used for.

WHAT IT MEASURES.  Tail-512 next-token NLL at prefix lengths inside OLMo's own
4096 native window, on the held-out natural documents the KKT measurement
already uses (`prepared_nll_02/doc_*.npy`), for a set of tables spanning the
budget axis.  Same instrument shape as the Qwen screen so the two curves are
read the same way -- the point is to compare CURVES, not absolutes.

THE BASELINE IS MEASURED IN-RUN, NOT READ FROM AN ARCHIVE.  The Qwen screen
pairs against archived per-row MrRoPE cells and therefore carries a length-
dependent batching offset (about 4e-4 at 32K, the same size as the table effect
it is trying to resolve), which has to be subtracted afterwards.  Here every arm
including the native reference is measured in one process with one batch policy,
so the deltas are internally consistent by construction and no offset correction
is needed.  The cost of that choice is that these numbers are NOT comparable to
any archived OLMo NLL; they are comparable to each other, which is all the curve
needs.

WHAT IT CANNOT SAY.  In-window NLL at 2048/4096 prices the SUPPORT of the
compression, not the long-range capability.  On Qwen the entire three-band family
sat within 0.005 nats of each other on this axis; if OLMo reproduces that, the
axis is slack there too and the knee must be found another way.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
GAIN = 1.138629436111989          # the gain BOTH archived OLMo tables carry


def build_tables():
    """The budget sweep, in OLMo's own geometry, with sum(m) as the axis.

    Chosen so the family walks the budget from well below to well past the
    deployed BM's 40.5, because the knee is somewhere above it: on Qwen the cost
    was still <= 0 at mean(m) 0.572 and turned positive by 0.644.
    """
    from experiments.curvature_20260910.tables import (m_incr_beta, m_leak,
                                                       m_mrpro, m_step, m_taper)
    lo, n = OLMO["low"], OLMO["n"]
    out = [("native", np.zeros(64))]
    for b in (0.0, 0.5, 1.0, 2.0, 4.0):
        out.append((f"beta_b{b:g}".replace(".", "p"),
                    np.asarray(m_incr_beta(b, n=n, low=lo), dtype=float)))
    for a in (0.005, 0.01, 0.02):
        out.append((f"leak_a{a:g}".replace(".", "p"),
                    np.asarray(m_leak(a, base="bm", n=n, low=lo), dtype=float)))
    for d in (0.002, 0.0063):
        out.append((f"taper_d{d:g}".replace(".", "p"),
                    np.asarray(m_taper(d, hi=32, lo=lo), dtype=float)))
    for hi in (22, 28, 32):
        out.append((f"step_hi{hi}", np.asarray(m_step(hi, lo=lo), dtype=float)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True, help="dir of doc_NN.npy")
    ap.add_argument("--lengths", default="2048,4096")
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    for L in lengths:
        if L > OLMO["window"]:
            print(f"REFUSING: length {L} is outside the {OLMO['window']} native "
                  "window; this is the IN-WINDOW instrument",
                  file=sys.stderr)
            return 2

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    docs = sorted(Path(args.nll_dir).glob("doc_*.npy"))[: args.docs]
    if not docs:
        print(f"REFUSING: no doc_*.npy under {args.nll_dir}", file=sys.stderr)
        return 2
    data = {L: [np.load(f)[:L + 1] for f in docs] for L in lengths}

    def measure(values, L):
        """One table, one length: mean tail-512 NLL over the documents."""
        rot = model.model.rotary_emb
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(
            device=next(model.parameters()).device)
        rot.attention_scaling = float(GAIN)
        losses = []
        with torch.inference_mode():
            for arr in data[L]:
                ids = torch.tensor(arr[:L].astype(np.int64), device="cuda")[None]
                tgt = torch.tensor(arr[L - 511:].astype(np.int64), device="cuda")
                out = model(ids, use_cache=False, logits_to_keep=512).logits[0].float()
                losses.append(float(torch.nn.functional.cross_entropy(out, tgt)))
                del ids, tgt
        return float(np.mean(losses)), losses

    native_np = (OLMO["theta"] ** (-np.arange(64) / 64)).astype(np.float32)
    # the native reference is measured HERE, in this process, at this batch policy
    base = {}
    for L in lengths:
        base[L] = measure(native_np, L)[0]
    print(json.dumps({"phase": "BASELINE", "native_nll": base,
                      "lengths": lengths, "docs": len(docs)}), flush=True)

    tables = build_tables()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        tables = [(n_, m) for n_, m in tables if n_ in want]

    rows = []
    for name, m in tables:
        values = (OLMO["theta"] ** (-np.arange(64) / 64)) * np.power(4.0, -m)
        rec = dict(name=name, sum_m=float(np.asarray(m).sum()),
                   mean_m=float(np.asarray(m).sum() / 64),
                   per_length={})
        for L in lengths:
            nll, per = measure(values.astype(np.float32), L)
            rec["per_length"][str(L)] = dict(
                nll=nll, native=base[L], delta=nll - base[L], per_doc=per)
            print(json.dumps({"arm": name, "length": L, "nll": round(nll, 6),
                              "delta": round(nll - base[L], 6)}), flush=True)
        rec["mean_delta"] = float(np.mean([rec["per_length"][str(L)]["delta"]
                                           for L in lengths]))
        rows.append(rec)
        with (root / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    (root / "manifest.json").write_text(json.dumps(dict(
        status="COMPLETE", model=args.model, lengths=lengths, docs=len(docs),
        baseline_measured_in_run=base, arms=[r["name"] for r in rows],
        scope=("Tail-512 next-token NLL inside OLMo's OWN 4096 window. Prices the "
               "support of the compression, not long-range capability. The "
               "baseline is measured in this process so deltas are internally "
               "consistent; they are NOT comparable to archived OLMo NLL.")),
        indent=1))

    print("\n=== in-window cost vs budget ON OLMO ===")
    print(f"{'arm':16s} {'mean(m)':>8s} {'delta':>11s}")
    for r in sorted(rows, key=lambda z: z["mean_m"]):
        print(f"{r['name']:16s} {r['mean_m']:8.4f} {r['mean_delta']:+11.6f}")
    d = np.array([r["mean_delta"] for r in rows])
    mm = np.array([r["mean_m"] for r in rows])
    print(f"\n  spread across every arm: {d.max() - d.min():.6f} nats")
    print("  On Qwen the same kind of family spanned 0.019 nats once the four "
          "extreme arms were excluded. If OLMo reproduces that, the in-window "
          "axis is slack here too and the knee is NOT visible on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
