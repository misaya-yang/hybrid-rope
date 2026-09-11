#!/usr/bin/env python3
"""Read the natural-QA generalisation panel and apply the pre-registered rule.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python natural_read.py

Governed by NATURAL_PREREG_20260911.md, committed before any row was read.

THE QUESTION.  The 180-row RULER panel put the plateau members at +6.54pp at
16384 on six SYNTHETIC retrieval tasks.  Is that a general long-context
capability, or a property of needle-style benchmarks?  These 391 rows are five
NATURAL reading-comprehension families that this campaign has never run, all at
16384, scored with the same ruler_bench.score.

WHY IT DECIDES SOMETHING.  verdict() already says the members are TRADEOFF --
long range up, in-window down.  Whether that trade is worth anything depends on
whether the long-range half is real capability.  If the gain is confined to
synthetic retrieval it is a benchmark artefact; if it survives on natural QA it
is a real capability bought at a measured price.

The gate is absolute accuracy: on a 1B model these tasks may sit near the floor,
where a delta is governed by noise rather than by ability.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
NAT = f"{ROOT}/natural_out"
BASE = "nat_bm"
MEMBERS = [("b3_lo14", "nat_b3"), ("a1_b64", "nat_a1b64"), ("b4wide", "nat_b4w")]


def load(name):
    path = os.path.join(NAT, name + ".jsonl")
    out = {}
    if not os.path.exists(path):
        return None
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "row_id" in d:
            out[d["row_id"]] = d
    return out


def paired(a, b, ids):
    va = np.array([a[i]["correct"] for i in ids], float)
    vb = np.array([b[i]["correct"] for i in ids], float)
    d = va - vb
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return d, se


def main():
    base = load(BASE)
    if base is None:
        print(f"REFUSING: {BASE}.jsonl not found under {NAT}")
        return 2
    present = [(n, k) for n, k in MEMBERS if load(k) is not None]
    missing = [n for n, k in MEMBERS if load(k) is None]

    print("=" * 80)
    print("NATURAL-QA GENERALISATION PANEL  (5 families never run in this campaign)")
    print("=" * 80)
    print(f"  deployed table {BASE}: {len(base)} rows")
    for n, k in present:
        print(f"  {n:<16} {len(load(k))} rows")
    if missing:
        print(f"  STILL RUNNING: {missing}")
    if not present:
        print("\nnothing to compare yet")
        return 0

    ids = sorted(base)
    for n, k in present:
        ids = [i for i in ids if i in load(k)]
    print(f"\n  common rows: {len(ids)}")

    bm_acc = np.mean([base[i]["correct"] for i in ids])
    print(f"\n--- GATE: BM absolute accuracy = {bm_acc:.4f} "
          f"(must be >= 0.10) ---")
    if bm_acc < 0.10:
        print("  GATE FAILED: these tasks are at the floor for this model and a")
        print("  delta would be governed by noise, not ability.  UNDECIDABLE.")
        return 0
    print("  gate passed")

    print("\n--- per member vs the deployed table ---")
    print(f"  {'arm':<16} {'acc':>7} {'delta':>9} {'se':>7} {'t':>7}  W/L/T")
    for n, k in present:
        d, se = paired(load(k), base, ids)
        acc = np.mean([load(k)[i]["correct"] for i in ids])
        w = int((d > 0).sum()); l = int((d < 0).sum()); t = int((d == 0).sum())
        print(f"  {n:<16} {acc:7.4f} {d.mean()*100:+9.2f} {se*100:7.2f} "
              f"{(d.mean()/se if se > 0 else 0):+7.2f}  {w}/{l}/{t}")

    deltas = np.mean([paired(load(k), base, ids)[0] for _, k in present], axis=0)
    se = deltas.std(ddof=1) / math.sqrt(len(deltas))
    tv = deltas.mean() / se if se > 0 else 0.0
    print("\n--- PRIMARY (pre-registered): pooled delta over the members ---")
    print(f"  pooled delta = {deltas.mean()*100:+.2f}pp   SE = {se*100:.2f}pp   "
          f"t = {tv:+.2f}")

    print("\n--- per task family (5 families; the pool can hide one moving) ---")
    print(f"  {'family':<20} {'n':>4} {'BM':>7} " +
          " ".join(f"{n:>9}" for n, _ in present))
    for fam in sorted({base[i].get("task") for i in ids}):
        sub = [i for i in ids if base[i].get("task") == fam]
        bm = np.mean([base[i]["correct"] for i in sub])
        cells = " ".join(f"{np.mean([load(k)[i]['correct']-base[i]['correct'] for i in sub])*100:>+9.1f}"
                         for _, k in present)
        print(f"  {fam:<20} {len(sub):>4} {bm:7.3f} {cells}")

    if len(present) > 1:
        print("\n--- members against each other ---")
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ni, ki = present[i]
                nj, kj = present[j]
                d, s = paired(load(ki), load(kj), ids)
                print(f"  {ni:<16} - {nj:<16} {d.mean()*100:+7.2f}pp  t="
                      f"{(d.mean()/s if s > 0 else 0):+5.2f}")

    print("\n--- VERDICT (rule fixed in NATURAL_PREREG_20260911.md, "
          "before any row was read) ---")
    if deltas.mean() * 100 >= 3.0 and tv >= 2.0:
        print("  GENERALISES.  The long-range gain is not a synthetic-retrieval")
        print("  artefact: it survives on five natural reading-comprehension")
        print("  families.  The TRADEOFF is a real capability bought at a price.")
    elif abs(deltas.mean() * 100) <= 1.5 and se * 100 <= 2.0:
        print("  DOES NOT GENERALISE.  The gain is specific to the synthetic")
        print("  RULER families, so the TRADEOFF holds only there.")
    else:
        print("  UNRESOLVED.  Report the interval and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
