#!/usr/bin/env python3
"""Read the b sweep against the archived endpoints, and say which theory survives.

WHAT THE SWEEP IS FOR.  The two deployed tables are the ends of one family,
`eps_k ~ k(n+1-k)^b` with b = 0 MrRoPE and b = 1 the deployed BM -- same band,
same gain, same span, bitwise-identical plateaus.  On OLMo-2-0425-1B they differ
by 44.6 RULER points; on Qwen2.5-3B by -7.3.  The interior is where nothing has
been measured, and the shape of the interior curve decides between the three
theories that survive the derivations:

    A  smoothness is the damage      -> peaks at b = 1 (BM is that objective's own
                                        minimizer, provably, within the family)
    B  flat-Fisher damage            -> peaks at b = 0 (closest to the extreme
                                        step, which minimizes flat damage)
    C  measured-metric KKT           -> peak is model-dependent, no shared b*

A and B are ALREADY dead as universal statements -- each is refuted by one of the
two measured endpoints, in opposite directions (Qwen puts b=0 above b=1, OLMo the
reverse).  What the interior decides is whether EITHER survives per model, and
whether the curve is a ramp or a cliff.

THE TWO SHAPES THAT MATTER, declared before the interior is read:

    RAMP   monotone between the endpoints.  Then b is a real continuous dial, the
           optimum is at or outside an endpoint, and the right next move is to
           extend the family past b=2 (or into the mixture family) rather than
           to refine inside.
    CLIFF  an interior jump: adjacent arms land on opposite sides of a large gap.
           Then the axis is a threshold, no local expansion describes it, and the
           object to locate is the EDGE -- which is exactly what the KKT
           programme could not see when its gradient came out zero on the band.

Both are results.  A flat interior with both endpoints better is a third outcome
and would say the family is not the right coordinate at all.

USAGE.  Endpoints come from the archive and are NOT re-run; the interior arms are
the ones this campaign generated.  Point --olmo at the run dir and the script
locates both automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The archived endpoints, entered from the run receipts, not recomputed here.
ENDPOINTS = {
    "olmo": {"b0": dict(arm="MrPro", acc=0.0709, src="run_ruler_newtasks_01/MrPro.json"),
             "b1": dict(arm="MrProBM", acc=0.4167, src="run_ruler_newtasks_01/MrProBM.json")},
    "qwen": {"b0": dict(arm="MrPro", acc=0.7813, src="run_qwen3_01/MrPro.jsonl"),
             "b1": dict(arm="MrProBM", acc=0.7083, src="run_qwen3_01/MrProBM.jsonl")},
}


def load_arm(path, model):
    d = json.loads(Path(path).read_text())
    acc = d.get("accuracy")
    if acc is None and "score_sum" in d and d.get("n"):
        acc = d["score_sum"] / d["n"]
    return dict(arm=d.get("arm", Path(path).stem), acc=float(acc),
                n=int(d.get("n", 0)), sum_m=float(d.get("sum_m", float("nan"))),
                vs_mrpro=(d.get("vs_MrPro") or {}).get("mine"),
                vs_bm=(d.get("vs_MrProBM") or {}).get("mine"))


def b_of(name):
    """`beta_b0p25` -> 0.25.  Returns None for anything that is not a b arm."""
    if not name.startswith("beta_b"):
        return None
    tok = name[len("beta_b"):].replace("p", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def classify(curve):
    """RAMP, CLIFF or FLAT, from the sorted b curve, with the rule stated.

    TWO INTERIOR POINTS ARE REQUIRED BEFORE A SHAPE IS NAMED.  With a single
    interior point the curve has two steps, and the larger one is >= 50% of the
    range by construction -- so a three-point curve would be labelled CLIFF
    essentially always.  That is not a finding, it is the classifier reading its
    own arithmetic, and it is exactly the way a thin result gets promoted into a
    structural claim (ds_workspace/LESSONS L1).  With fewer points the honest
    answer is UNDERDETERMINED.
    """
    if len(curve) < 4:
        return ("UNDERDETERMINED",
                f"{len(curve)} points is not enough to name a shape; "
                "two interior arms are required")
    bs = [b for b, _ in curve]
    accs = [a for _, a in curve]
    lo, hi = min(accs), max(accs)
    span = hi - lo
    if span <= 0:
        return "FLAT", "every point identical"
    # a cliff is a single adjacent step carrying most of the total range
    steps = [abs(accs[i + 1] - accs[i]) for i in range(len(accs) - 1)]
    biggest = max(steps)
    if biggest >= 0.6 * span:
        i = steps.index(biggest)
        return "CLIFF", (f"one step b={bs[i]:g}->{bs[i+1]:g} carries "
                         f"{biggest / span:.0%} of the total range")
    return "RAMP", (f"largest single step is {biggest / span:.0%} of the range")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ENDPOINTS))
    ap.add_argument("--dir", required=True, help="run dir with beta_b*_summary.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    ep = ENDPOINTS[args.model]
    curve = [(0.0, ep["b0"]["acc"]), (1.0, ep["b1"]["acc"])]
    for f in sorted(Path(args.dir).glob("beta_b*_summary.json")):
        b = b_of(f.stem.replace("_summary", ""))
        if b is None or b in (0.0, 1.0):
            continue
        curve.append((b, load_arm(f, args.model)["acc"]))
    curve.sort()

    print(f"=== {args.model}: b sweep, endpoints from the archive ===")
    print(f"  b = 0.00  {ep['b0']['acc']:.4f}   {ep['b0']['arm']} "
          f"(archived, {ep['b0']['src']})")
    for b, a in curve:
        if b in (0.0, 1.0):
            continue
        print(f"  b = {b:<5g} {a:.4f}   (this run)")
    print(f"  b = 1.00  {ep['b1']['acc']:.4f}   {ep['b1']['arm']} "
          f"(archived, {ep['b1']['src']})")

    shape, why = classify(curve)
    print(f"\n  shape: {shape}  ({why})")
    interior = [(b, a) for b, a in curve if 0.0 < b < 1.0]
    if interior:
        best_b, best_a = max(interior, key=lambda z: z[1])
        both = max(ep["b0"]["acc"], ep["b1"]["acc"])
        print(f"  best interior b = {best_b:g} at {best_a:.4f}")
        if best_a > both:
            print("  INTERIOR PEAK: an interior arm beats BOTH archived endpoints.")
            print("  -> the family's optimum is not at an endpoint; refine in the")
            print("     MIXTURE family eps_k ~ k(C-k), which is the interpolation")
            print("     the two endpoints imply (see tables.m_mixC).")
        else:
            top = "b=0 (MrRoPE)" if ep["b0"]["acc"] > ep["b1"]["acc"] else "b=1 (BM)"
            print(f"  no interior peak; best endpoint is {top} at {both:.4f}")
    print("\n  A flat-Fisher theory predicts a peak at b=0; a smoothness theory at\n"
          "  b=1. A and B are already refuted as UNIVERSAL statements by the two\n"
          "  endpoints disagreeing across models. An interior peak refutes both\n"
          "  as per-model statements too, and names the mixture family instead.")

    out = dict(model=args.model, dir=args.dir, curve=curve, shape=shape,
               why=why, endpoints=ep,
               scope=("endpoints are the archived runs, not regenerated; interior "
                      "arms are this campaign's. The shape call is a rule applied "
                      "to the curve, not a significance test -- n is 350 rows per "
                      "OLMo arm and 36 per Qwen arm, so the Qwen curve is coarse."))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
