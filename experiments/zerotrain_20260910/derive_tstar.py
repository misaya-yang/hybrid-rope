#!/usr/bin/env python3
"""DERIVE the optimal band edge instead of scanning for it.

THE ANALYTIC PROGRAM THIS IMPLEMENTS.  The mechanism analysis gives a corner-point
theorem: with per-slot utility

    G_j(m) = 0.25 * 4^m  -  lambda * 4 * t_j * (1 - 4^(-m))

where the first term is the slot's coverage contribution on the test range
[0, 4W] (saturating at m=1, because nu_j = omega_j/4 is exactly the native
frequency of a 4x-longer context) and the second is its phase drift, the function
is STRICTLY CONVEX in m (G'' = (ln4)^2 [0.25*4^m + 4*lambda*t_j*4^(-m)] > 0), so
every slot's optimum is at an ENDPOINT.  The corner test is

    G_j(1) - G_j(0) = 0.75 - 3*lambda*t_j  > 0   <=>   t_j < t* := 0.25/lambda

so the optimal table is a STEP IN TURN COORDINATES, m_j = 1[t_j < t*], and the
only free quantity is t*.  lambda is fixed by the in-window constraint

    sum_{j : t_j < t*} (1/2) F_jj (ln4)^2  =  D_N(budget)

with F_jj the diagonal output-Fisher of the in-window loss w.r.t. slot j.  F_jj
is MEASURABLE with one forward per slot, which is what this file does.

WHY THIS IS NOT A SCAN.  Every band-edge experiment this campaign has run picked
its edge from the turn rule's hand-tuned constants (beta = 16, 32, 64, 128) or
from a round number (hi = 22, 25, 28, 32).  Here the edge is the SOLUTION of an
equation whose every input is measured on the checkpoint.  The output is a
prediction, and if it is wrong that is a fact about the mechanism.

THE ONE DECLARED CHOICE, AND IT IS NOT FITTED TO ANY SCORE.  The budget
D_N(budget) has to come from somewhere, and the honest source is the incumbent:
the deployed BM's own in-window damage, measured on the same documents with the
same harness.  So the prediction reads "the step whose in-window damage equals
the deployed table's, by the measured Fisher metric, has its edge at t*".  If the
step at that t* beats the deployed table, the analytic program has produced a
better table from measurements alone; if it loses, the metric or the convexity
model is wrong -- also a result, and the more informative one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
GAIN = 1.138629436111989
LN4 = math.log(4.0)


def native_nu(theta=OLMO["theta"], k=OLMO["K"]):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def turns(nu, W=OLMO["window"]):
    return np.asarray(nu, dtype=np.float64) * W / (2.0 * math.pi)


def measure_fisher(model, docs, L, base_values, delta, slots, tail=512):
    """F_jj by one forward per slot: D_N(delta e_j) = 0.5 F_jj delta^2 + O(d^3).

    D_N(0) = 0 exactly, so there is no first-order term and a one-sided forward
    per slot suffices -- no backward, no Hessian, no Fisher-vector product.

    The KL is taken on the tail `tail` positions only.  The constraint in the plan
    is an output-KL against the frozen native table, and the in-window positions
    are the ones that define it; keeping the full distribution for a 4096-token
    sequence would cost 4096 x vocab floats per document for a quantity that is
    read on 512 of them.
    """
    import torch

    rot = model.model.rotary_emb
    dev = next(model.parameters()).device

    def logits_for(values):
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        outs = []
        with torch.inference_mode():
            for arr in docs:
                ids = torch.tensor(arr[:L].astype(np.int64), device=dev)[None]
                o = model(ids, use_cache=False,
                          logits_to_keep=tail).logits[0].float()
                outs.append(torch.log_softmax(o, dim=-1))
                del ids, o
        return outs

    base_logp = logits_for(base_values)
    out = {}
    t0 = time.monotonic()
    for j in slots:
        v = np.asarray(base_values, dtype=np.float64).copy()
        v[j] = v[j] * math.exp(delta)
        pert = logits_for(v)
        kl = 0.0
        for a, b in zip(base_logp, pert):
            kl += float((a.exp() * (a - b)).sum())
            del b
        out[j] = 2.0 * (kl / len(base_logp)) / (delta * delta)
        if (j + 1) % 8 == 0:
            print(json.dumps({"fisher": j, "F_jj": out[j],
                              "seconds": round(time.monotonic() - t0, 1)}),
                  flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--slots", default=",".join(str(i) for i in range(64)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

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

    files = sorted(Path(args.nll_dir).glob("doc_*.npy"))[: args.docs]
    docs = [np.load(f) for f in files]
    if not docs:
        print(f"REFUSING: no doc_*.npy under {args.nll_dir}", file=sys.stderr)
        return 2

    nu0 = native_nu()
    slots = [int(s) for s in args.slots.split(",") if s.strip()]
    F = measure_fisher(model, docs, args.length, nu0, args.delta, slots)

    # ---- the derivation ---------------------------------------------------
    t = turns(nu0)
    Fv = np.array([F[j] for j in range(64)], dtype=np.float64)
    # damage of a full step at every candidate edge, in the measured metric
    dam = np.cumsum(Fv * LN4 * LN4 / 2.0)          # dam[j] = sum_{k<=j} ...
    # the constraint set is {t_j < t*}, i.e. j > j*, so the step's damage at edge
    # hi is the SUFFIX sum from hi to 63
    suffix = np.cumsum((Fv * LN4 * LN4 / 2.0)[::-1])[::-1]
    # the deployed BM's own damage, in the SAME measured metric
    from experiments.curvature_20260910.tables import m_incr_beta
    m_bm = np.asarray(m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]), dtype=np.float64)
    dam_bm = float(np.sum(0.5 * Fv * (LN4 ** 2) * m_bm ** 2))

    rows = []
    for hi in range(64):
        rows.append(dict(hi=int(hi), t_hi=float(t[hi]),
                         step_damage=float(suffix[hi]),
                         at_or_below_bm=bool(suffix[hi] <= dam_bm)))
    feasible = [r for r in rows if r["at_or_below_bm"]]
    derived_hi = min(r["hi"] for r in feasible) if feasible else None
    out = dict(length=args.length, docs=len(docs), delta=args.delta,
               F_diag={str(j): F[j] for j in range(64)},
               turns=t.tolist(), step_damage=suffix.tolist(), rows=rows,
               damage_bm_in_measured_metric=dam_bm,
               derived_hi=derived_hi,
               derived_tstar=(float(t[derived_hi]) if derived_hi is not None else None),
               scope=("t* is the SOLUTION of  sum_{t_j<t*} 0.5 F_jj (ln4)^2 = "
                      "D_N(deployed BM), with F_jj measured on this checkpoint and "
                      "D_N(BM) computed in the same measured metric. Nothing is "
                      "fitted to any RULER score. A step at the derived edge is a "
                      "PREDICTION; if it loses, the metric or the convexity model "
                      "is wrong."))
    (root / "tstar.json").write_text(json.dumps(out, indent=1))

    print("\n=== measured F_jj (in-window output-Fisher diagonal) ===")
    for j in range(0, 64, 8):
        print("  slot %2d..%2d: " % (j, j + 7)
              + " ".join("%9.4g" % Fv[k] for k in range(j, min(j + 8, 64))))
    print(f"\n  damage of the deployed BM in this metric : {dam_bm:.6g}")
    print(f"  feasible step edges (damage <= BM's)     : "
          f"{[r['hi'] for r in feasible][:12]}{' ...' if len(feasible) > 12 else ''}")
    if derived_hi is not None:
        print(f"\n  DERIVED EDGE  hi* = {derived_hi}   (t* = {t[derived_hi]:.3f} turns)")
        print(f"  -> m_j = 1 for j >= {derived_hi}")
        print("  Compare: the deployed BM's plateau starts at 33; the hand-tuned")
        print("  turn arms used edges at beta = 16/32/64 and hi = 20/22/25/32.")
    else:
        print("\n  no feasible edge: even the last slot's full compression exceeds")
        print("  the deployed table's damage in the measured metric. That itself is")
        print("  a statement about the metric and is worth recording.")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
