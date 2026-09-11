#!/usr/bin/env python3
"""Read the out-of-sample confirmation of the walk's interior point.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python walkconf_read.py

Governed by WALK_RESULT_AND_CONFIRM_PREREG_20260911.md, committed before the
walk_a0p5 arm ran.

THE CLAIM UNDER TEST.  On the 180-row panel, m(0.5) = (m_BM + m_a1b64)/2 looked
like a good interior point: it had the BEST long-range gain of the whole segment
(+10.87pp at 16384, t=+3.28) AND the smallest in-window loss (-2.69pp, t=-1.51,
not significant).  But the linearity null was rejected there, and the point was
located on rows already used to choose the plateau members -- the same shape as
the +12..14pp illusion the held-out panel destroyed.

    pooled delta >= +5pp and t >= 2   -> generalises across task families
    |delta| <= 1.5pp with SE <= 2.0pp -> does not generalise; a selection effect
    otherwise                         -> unresolved, report the interval and stop

CONTEXT THAT MUST BE READ ALONGSIDE.  chain_natural measures the three plateau
members on this same panel.  If the interior point does not beat THOSE, it has no
new content even if it clears the bar.  And the panel is 89% single-reference, so
its natural reading is the binary one -- where the interior point's long-range
gain was only +3.33pp (t=1.07) on the RULER panel.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
NAT = f"{ROOT}/natural_out"
BASE = "beta_b1p0"
TARGET = "walk_a0p5"
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
    tgt = load(TARGET)
    if base is None:
        print(f"REFUSING: {BASE}.jsonl not found under {NAT}")
        return 2
    if tgt is None:
        print(f"REFUSING: {TARGET}.jsonl not found under {NAT}")
        return 2

    ids = sorted(set(base) & set(tgt))
    print("=" * 78)
    print("OUT-OF-SAMPLE CONFIRMATION OF THE WALK'S INTERIOR POINT (a = 0.50)")
    print("391 natural-QA rows, five families, never used in this campaign")
    print("=" * 78)
    print(f"  common rows: {len(ids)}  (expect 391)")

    bm_acc = np.mean([base[i]["correct"] for i in ids])
    print(f"\n  gate: BM accuracy = {bm_acc:.4f} "
          f"({'pass' if bm_acc >= 0.10 else 'FAIL -- tasks at floor'})")
    if bm_acc < 0.10:
        return 0

    for binar in (False, True):
        f = (lambda x: 1.0 if x >= 0.999 else 0.0) if binar else (lambda x: x)
        d = np.array([f(tgt[i]["correct"]) - f(base[i]["correct"]) for i in ids], float)
        se = d.std(ddof=1) / math.sqrt(len(d))
        lbl = "WHOLE ROW  " if binar else "fractional "
        print(f"  {lbl} acc={np.mean([f(tgt[i]['correct']) for i in ids]):.4f}  "
              f"delta={d.mean()*100:+.2f}pp  SE={se*100:.2f}  "
              f"t={(d.mean()/se if se > 0 else 0):+.2f}  "
              f"W/L/T={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")

    print("\n--- per family ---")
    for fam in sorted({base[i].get("task") for i in ids}):
        sub = [i for i in ids if base[i].get("task") == fam]
        bm = np.mean([base[i]["correct"] for i in sub])
        mm = np.mean([tgt[i]["correct"] for i in sub])
        print(f"  {fam:<20} n={len(sub):>4} BM={bm:.3f} a0p5={mm:.3f} "
              f"{(mm-bm)*100:>+7.1f}pp")

    # against the plateau members on the same panel: an interior point that does
    # not beat them adds nothing even if it clears the absolute bar
    print("\n--- the interior point vs the plateau members (same panel) ---")
    for n, k in MEMBERS:
        o = load(k)
        if o is None:
            print(f"  {n:<16} (not run yet)")
            continue
        common = [i for i in ids if i in o]
        d1, s1 = paired(tgt, base, common)
        d2, s2 = paired(o, base, common)
        dd, ss = paired(tgt, o, common)
        print(f"  a0p5 {d1.mean()*100:+7.2f}pp | {n:<10} {d2.mean()*100:+7.2f}pp "
              f"| a0p5 - {n:<10} {dd.mean()*100:+6.2f}pp "
              f"t={(dd.mean()/ss if ss > 0 else 0):+5.2f}")

    d, se = paired(tgt, base, ids)
    t = d.mean() / se if se > 0 else 0.0
    print("\n--- VERDICT (rule fixed before this arm ran) ---")
    if d.mean() * 100 >= 5.0 and t >= 2.0:
        print("  GENERALISES.  The interior point's long-range gain crosses task")
        print("  families; worth a final test.")
    elif abs(d.mean() * 100) <= 1.5 and se * 100 <= 2.0:
        print("  DOES NOT GENERALISE.  The interior point is a selection effect on")
        print("  the 180 rows, the same class as the plateau members' +12..14pp.")
    else:
        print("  UNRESOLVED: report the interval and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
