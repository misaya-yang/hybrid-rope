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
# Arm names come from the FLAGS the runner was given, not from stage labels:
# --betas 1.0 -> beta_b1p0.jsonl, --betas 3.0 -> beta_b3p0.jsonl,
# --wide-betas 1.0 -> wide_b1p0.jsonl, --wide-betas 4.0 -> wide_b4p0.jsonl.
# Hardcoding stage labels here silently skipped the base arm once already.
BASE = "beta_b1p0"
MEMBERS = [("b3_lo14", "beta_b3p0"), ("a1_b64", "wide_b1p0"), ("b4wide", "wide_b4p0")]


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

    # COMPLETENESS GUARD.  The arms are written in panel order, so a partial run
    # yields the FIRST families only -- and those are the ones with the highest
    # BM accuracy (2wikimqa 0.25, hotpotqa 0.32 vs 0.016 for narrativeqa).  A
    # verdict on that subset would be a verdict on the easiest two families.
    # Refuse rather than report a biased number.
    complete = len(base) >= 391 and all(len(load(k)) >= 391 for _, k in present)
    if not complete:
        print("  INCOMPLETE: base has "
              f"{len(base)} rows, members "
              f"{[len(load(k)) for _, k in present]}; need 391 each.")
        print("  The intersection covers only the families that finished first,")
        print("  which are the highest-accuracy ones.  NO VERDICT IS VALID HERE.")
        print("  Re-run once every arm has 391 rows.")
        return 4

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

    # ---- reference-count split (the reason the RULER gain was partial credit)
    print("\n--- split by reference count (1 ref = binary metric) ---")
    for lo, hi, label in ((1, 1, "1 reference (binary)"),
                          (2, 99, "2+ references (item recall)")):
        sub = [i for i in ids if lo <= len(base[i].get("references") or [1]) <= hi]
        if not sub:
            continue
        dd = np.mean([paired(load(k), base, sub)[0] for _, k in present], axis=0)
        ss = dd.std(ddof=1) / math.sqrt(len(dd)) if len(dd) > 1 else float("nan")
        bm = np.mean([base[i]["correct"] for i in sub])
        print(f"  {label:<28} n={len(sub):<4} BM={bm:.3f}  pooled delta="
              f"{dd.mean()*100:+.2f}pp  se={ss*100:.2f}pp")

    print("\n--- BINARIZED (correct >= 0.999): whole-row right, supplementary ---")
    dd = np.mean([[ (1.0 if load(k)[i]["correct"] >= 0.999 else 0.0)
                    - (1.0 if base[i]["correct"] >= 0.999 else 0.0)
                    for i in ids] for _, k in present], axis=0)
    ss = dd.std(ddof=1) / math.sqrt(len(dd))
    print(f"  pooled binarized delta = {dd.mean()*100:+.2f}pp  SE = {ss*100:.2f}pp  "
          f"t = {(dd.mean()/ss if ss > 0 else 0):+.2f}")

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
