#!/usr/bin/env python3
"""Read round 1: the in-window fidelity axis, referenced to NATIVE and to our own MrPro.

THREE REFERENCE POINTS, AND ONLY ONE OF THEM IS VALID FOR RANKING.

  1. `native` -- the fidelity FLOOR.  m = 0 everywhere, no compression, and by
     construction the table the checkpoint was trained with.  MrRoPE's claim is
     that it stays close to this floor while buying long-range ability; how close
     it stays is the thing to measure, and it is what the user's correction says
     round 1's first number was showing.
  2. `mrpro_n17` AS RE-MEASURED HERE -- the reference for arm ranking.  The same
     table re-scored differs from the archived baseline by +2.5e-4 nats, which is
     the size of the effects under study.  That offset is the batching: the
     archived harness ran one document per forward and this one batches 8-32 of
     them, and bf16 reductions are not batch-invariant.  So the archive gives the
     ABSOLUTE level and this run gives the COMPARISONS; mixing them would credit
     every arm with a harness difference.
  3. `incr_r1` -- bit-identical to `mrpro_n17` in the table, so it MUST score
     identically.  It is a free check that the harness is deterministic and that
     the reference above is a reference.

THE THEORY PREDICTION THIS FILE TESTS.  The price structure computed from the
Phase-0 full-model gradients (`analysis/p0_gradients/`) is NOT flat over the
transition band: the minimum price sits at gap 24 -- the first increment -- and
the mean profile survives leave-one-out with the argmin at gap 24 in every one of
the six splits.  Since  dL/dDelta_j = ln 4 * price_j,  a low price at gap 24 means
the long-range objective wants MORE budget there, i.e. a LARGER first increment,
i.e. a LARGER `a` in the (a, r) dial family.  That is the opposite of what
MrRoPE does.  The `front_a*` arms vary exactly that increment with the tail shape
held proportional, so round 1 measures the NATIVE COST of the move the long-range
gradient asks for -- and round 2 measures whether the long-range benefit is real.
Both halves are needed; neither alone decides it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRONT_ARMS = ["front_a0p001", "mrpro_n17", "front_a0p03", "front_a0p1"]


def load(rows_path):
    rows = [json.loads(l) for l in open(rows_path)]
    by = {r["name"]: r for r in rows}
    return rows, by


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows, by = load(Path(args.out) / "rows.jsonl")
    lengths = sorted(next(iter(by.values()))["per_length"], key=int)

    report = {}

    # ---- 1. the free consistency check ----------------------------------
    if "incr_r1" in by and "mrpro_n17" in by:
        a, b = by["incr_r1"], by["mrpro_n17"]
        d = {L: a["per_length"][L]["mean_nll"] - b["per_length"][L]["mean_nll"]
             for L in lengths}
        report["anchor_check"] = dict(
            bit_identical_tables=True, max_abs_nll_diff=float(max(abs(x) for x in d.values())),
            by_length={L: float(x) for L, x in d.items()},
            ok=bool(max(abs(x) for x in d.values()) < 1e-9),
            note="incr_r1 and mrpro_n17 are the same table; any difference is "
                 "harness nondeterminism and would invalidate the reference")
        if not report["anchor_check"]["ok"]:
            print(json.dumps(report["anchor_check"], indent=1))
            print("REFUSING to rank: the two identical tables scored differently",
                  file=sys.stderr)
            return 2

    # ---- 2. harness vs archive offset -----------------------------------
    mr = by.get("mrpro_n17")
    if mr:
        report["harness_offset_vs_archive"] = dict(
            by_length={L: float(mr["per_length"][L]["mean_nll"]
                                - mr["per_length"][L]["mrpro_mean"]) for L in lengths},
            note="this run re-measured MrPro against the archived MrPro rows; the "
                 "residual is a batching difference, not a table difference, and "
                 "it is why every ranking below uses OUR MrPro as zero")

    # ---- 3. the fidelity axis, referenced to native ---------------------
    nat = by.get("native")
    ref = mr or nat
    fid = []
    for r in rows:
        rec = dict(name=r["name"], sum_m=r["sum_m"])
        for L in lengths:
            rec[L] = r["per_length"][L]["mean_nll"] - nat["per_length"][L]["mean_nll"]
        rec["mean_vs_native"] = float(np.mean([rec[L] for L in lengths]))
        rec["mean_vs_mrpro"] = float(np.mean(
            [r["per_length"][L]["mean_nll"] - ref["per_length"][L]["mean_nll"]
             for L in lengths]))
        fid.append(rec)
    fid.sort(key=lambda x: x["mean_vs_native"])
    report["fidelity_vs_native"] = fid

    # ---- 4. the front dial: does more compression at gap 24 cost in-window?
    front = [f for f in fid if f["name"] in FRONT_ARMS]
    order = {n: i for i, n in enumerate(FRONT_ARMS)}
    front.sort(key=lambda x: order.get(x["name"], 99))
    report["front_dial"] = dict(
        arms=[dict(name=f["name"], sum_m=f["sum_m"],
                   vs_native=f["mean_vs_native"], vs_mrpro=f["mean_vs_mrpro"])
              for f in front],
        a_values=[0.001, 2.0 / 306.0, 0.03, 0.10][:len(front)],
        prediction=("the Phase-0 price structure puts the minimum price at gap 24, "
                    "so the long-range objective wants a LARGER first increment. "
                    "If the native cost is monotone in a, this dial prices that "
                    "move; if it is flat, the move is free in-window and only "
                    "round 2 can decide it"))

    # ---- 5. the two dials, side by side ---------------------------------
    by_name = {f["name"]: f for f in fid}
    dials = {}
    for prefix, names in (("front_a", ["front_a0p001", "mrpro_n17", "front_a0p03",
                                       "front_a0p1"]),
                          ("back_r", ["back_r0", "mrpro_n17", "back_r2", "back_r4"])):
        dials[prefix] = [dict(name=n, vs_native=by_name[n]["mean_vs_native"],
                              vs_mrpro=by_name[n]["mean_vs_mrpro"])
                         for n in names if n in by_name]
    report["dials"] = dials

    Path(args.out, "analysis.json").write_text(json.dumps(report, indent=1, default=str))

    # ---- print ----------------------------------------------------------
    print("=" * 92)
    print("ROUND 1 -- in-window tail NLL, 16 FineWeb-Edu docs x 3 lengths")
    print("=" * 92)
    if "harness_offset_vs_archive" in report:
        off = report["harness_offset_vs_archive"]["by_length"]
        print("harness offset vs the archived MrPro rows (NOT a table effect): "
              + "  ".join(f"{L}:{v:+.2e}" for L, v in off.items()))
    print(f"\n{'arm':18s} {'sum_m':>8s} " + " ".join(f"{'d'+L:>11s}" for L in lengths)
          + f" {'vs native':>11s} {'vs MrPro':>10s}")
    for f in fid:
        print(f"{f['name']:18s} {f['sum_m']:8.3f} "
              + " ".join(f"{f[L]:+11.6f}" for L in lengths)
              + f" {f['mean_vs_native']:+11.6f} {f['mean_vs_mrpro']:+10.6f}")
    print("\nfront dial (native cost of moving budget into the first increment):")
    for a in report["front_dial"]["arms"]:
        print(f"  {a['name']:14s} vs native {a['vs_native']:+.6f}   "
              f"vs MrPro {a['vs_mrpro']:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
