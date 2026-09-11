#!/usr/bin/env python3
"""Read the LongBridge signed pair ported to OLMo, on both panels.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python lb_read.py

Governed by LONGBRIDGE_PREREG_20260911.md, committed before any row was read.

THE DESIGN.  Two arms differing ONLY in the sign of a nu shift on the four slots
whose effective period spans [W, 4W]:

    nu_m6p104em05   nu_j - delta   (the historical "Slower")
    nu_p6p104em05   nu_j + delta   (the historical "Faster")

THE STATISTIC.  The pre-registered primary is the SIGNED PAIR difference
(slower - faster) at 16384, on the 350-row panel, with the 180-row held-out as
confirmation.  Pairing the two arms against EACH OTHER is what makes this design
work at low SNR: they share every panel-level property -- task mix, difficulty,
selection bias -- which cancels in the difference.  Comparing either arm to a
baseline (MrRoPE, BM) does not have that property.

    delta >= +4pp and t >= 2 (350 rows), same sign on held-out -> the direction
        is real and cross-model; the first powered winner in this campaign
    |delta| <= 2pp (350 rows) -> the direction does not hold; the historical
        t=+0.75 was noise.  Closes the "slow-end nu tweak" direction.
    otherwise -> unresolved; report the interval and stop.

Arm names are globbed rather than hardcoded so a naming change cannot silently
produce an empty read.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
PANELS = [("350-row selection", f"{ROOT}/olmo_lb", 350),
          ("180-row held-out", f"{ROOT}/olmo_lb_h", 180)]


def load(p):
    out = {}
    if not os.path.exists(p):
        return None
    for ln in open(p):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "row_id" in d:
            out[d["row_id"]] = d
    return out


def find(d):
    """Return (minus_arm, plus_arm) by glob so naming cannot silently break it."""
    minus = glob.glob(f"{d}/nu_m*.jsonl")
    plus = glob.glob(f"{d}/nu_p*.jsonl")
    return (load(minus[0]) if minus else None,
            load(plus[0]) if plus else None)


def main():
    shown = False
    for label, d, want in PANELS:
        m, p = find(d)
        if m is None or p is None:
            print(f"\n### {label}: not run yet ({d})")
            continue
        ids = sorted(set(m) & set(p))
        if len(ids) < min(want, 50):
            print(f"\n### {label}: only {len(ids)} common rows -- still running")
            continue
        shown = True
        print("\n" + "=" * 76)
        print(f"{label}   n = {len(ids)}")
        print("=" * 76)
        for cap in (4096, 16384):
            sub = [i for i in ids if m[i]["length_cap"] == cap]
            if not sub:
                continue
            # the signed pair difference: slower - faster
            dd = np.array([m[i]["correct"] - p[i]["correct"] for i in sub], float)
            se = dd.std(ddof=1) / math.sqrt(len(dd))
            t = dd.mean() / se if se > 0 else float("nan")
            bm = np.mean([m[i]["correct"] for i in sub])
            bp = np.mean([p[i]["correct"] for i in sub])
            print(f"\n  cap={cap:<6} n={len(sub):<4} slower={bm:.4f} faster={bp:.4f}")
            print(f"    SIGNED PAIR  delta(slower-faster) = {dd.mean()*100:+.2f}pp  "
                  f"SE={se*100:.2f}  t={t:+.2f}  "
                  f"W/L/T={int((dd>0).sum())}/{int((dd<0).sum())}/{int((dd==0).sum())}")

        sub = [i for i in ids if m[i]["length_cap"] == 16384]
        if sub:
            dd = np.array([m[i]["correct"] - p[i]["correct"] for i in sub], float)
            se = dd.std(ddof=1) / math.sqrt(len(dd))
            t = dd.mean() / se if se > 0 else 0.0
            print(f"\n  --- per task @16384 (slower - faster) ---")
            for tk in sorted({m[i].get("task") for i in sub}):
                s = [i for i in sub if m[i].get("task") == tk]
                d2 = np.array([m[i]["correct"] - p[i]["correct"] for i in s], float)
                print(f"    {tk:<20} n={len(s):<4} {d2.mean()*100:+7.1f}pp")

    if not shown:
        print("\nnothing to read yet")
        return 0

    print("\n" + "=" * 76)
    print("VERDICT (rule fixed in LONGBRIDGE_PREREG_20260911.md before reading)")
    print("=" * 76)
    m, p = find(PANELS[0][1])
    if m and p:
        ids = sorted(set(m) & set(p))
        sub = [i for i in ids if m[i]["length_cap"] == 16384]
        if sub:
            dd = np.array([m[i]["correct"] - p[i]["correct"] for i in sub], float)
            se = dd.std(ddof=1) / math.sqrt(len(dd))
            t = dd.mean() / se if se > 0 else 0.0
            if dd.mean() * 100 >= 4.0 and t >= 2.0:
                print("  DIRECTION IS REAL on the 350-row panel.  Check the held-out")
                print("  panel for the same sign before claiming anything.")
            elif abs(dd.mean() * 100) <= 2.0:
                print("  DIRECTION DOES NOT HOLD.  The historical t=+0.75 was noise.")
                print("  The slow-end nu-tweak direction is closed.")
            else:
                print("  UNRESOLVED: report the interval and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
