#!/usr/bin/env python3
"""Read pro_step42 on the 180 held-out rows and apply the pre-registered rule.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python step42_read.py

Governed by STEP42_PREREG_20260911.md, committed before any row was read.

step42 is the first table here to beat the deployed BM significantly in BOTH
readings on the 350-row selection panel (+7.26pp / +6.00pp), but +32 of those
points come from niah_single_3 -- the exact task the earlier held-out test
identified as the source of the plateau members' illusory +12..14pp.

The comparison lives on tasks the selection panel never used, so this is the
same test that killed the previous claim:
    delta >= +5pp and t >= 2   -> survives out of sample; first real win
    delta <= +2pp or t < 2     -> selection effect, same class as before
    otherwise                  -> unresolved, report the interval and stop
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
ARM = f"{ROOT}/s42_out/pro_step42.jsonl"
REF = f"{ROOT}/holdout180/beta_b1p0.jsonl"


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


def main():
    a = load(ARM)
    if a is None:
        print(f"REFUSING: {ARM} not found")
        return 2
    b = load(REF)
    if b is None:
        print(f"REFUSING: reference {REF} not found")
        return 2
    ids = sorted(set(a) & set(b))
    print("=" * 76)
    print("pro_step42 OUT-OF-SAMPLE  (180 held-out rows, tasks the selection")
    print("panel never used -- niah_single_3 is NOT among them)")
    print("=" * 76)
    print(f"  common rows: {len(ids)}")
    if len(ids) < 100:
        print("  STILL RUNNING (fewer rows than the 180-row reference)")

    for binar in (False, True):
        f = (lambda x: 1.0 if x >= 0.999 else 0.0) if binar else (lambda x: x)
        d = np.array([f(a[i]["correct"]) - f(b[i]["correct"]) for i in ids], float)
        se = d.std(ddof=1) / math.sqrt(len(d))
        acc = np.mean([f(a[i]["correct"]) for i in ids])
        bacc = np.mean([f(b[i]["correct"]) for i in ids])
        lbl = "WHOLE ROW  " if binar else "fractional "
        print(f"\n  {lbl} BM acc={bacc:.4f}  step42 acc={acc:.4f}  "
              f"delta={d.mean()*100:+.2f}pp  SE={se*100:.2f}  "
              f"t={(d.mean()/se if se>0 else 0):+.2f}  "
              f"W/L/T={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")

    print("\n--- per task (a gain concentrated in ONE task is the warning sign) ---")
    print(f"  {'task':<20} {'n':>4} {'BM':>7} {'step42':>8} {'delta':>9}")
    for tk in sorted({b[i].get("task") for i in ids}):
        sub = [i for i in ids if b[i].get("task") == tk]
        bm = np.mean([b[i]["correct"] for i in sub])
        mm = np.mean([a[i]["correct"] for i in sub])
        print(f"  {tk:<20} {len(sub):>4} {bm:>7.3f} {mm:>8.3f} "
              f"{(mm-bm)*100:>+8.1f}pp")

    d = np.array([a[i]["correct"] - b[i]["correct"] for i in ids], float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0.0
    print("\n--- VERDICT (rule fixed in STEP42_PREREG_20260911.md before reading) ---")
    if d.mean() * 100 >= 5.0 and t >= 2.0:
        print("  SURVIVES OUT OF SAMPLE.  First table in this campaign to beat the")
        print("  deployed BM on tasks never used for selection.  Worth scaling up.")
    elif d.mean() * 100 <= 2.0 or t < 2.0:
        print("  DOES NOT SURVIVE.  The selection-panel +7.26pp is a selection")
        print("  effect, the same class as the plateau members' +12..14pp.")
    else:
        print("  UNRESOLVED: report the interval and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
