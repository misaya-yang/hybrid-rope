#!/usr/bin/env python3
"""Read the gain sweep: does the measured optimum sit at YaRN's derived value?

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python gsweep_read.py

THE ANALYTIC PREDICTION UNDER TEST.  YaRN sets the rotary amplitude to
mscale = 0.1*ln(s) + 1 with s = L/L_train.  At L=16384 on a 4096-window model
that is 1.138629436111989, and this campaign has inherited it for every arm.
The sweep brackets that value (1.00, 1.05, 1.10, 1.1386, 1.20) on one fixed
table, 350 rows each, so the response curve can be read directly.

  peak inside [1.10, 1.20] containing 1.1386 -> the derivation is validated for
       this model as well as this data
  peak strictly outside that band                -> the derived value is off
  monotone over the whole range                  -> the optimum is not bracketed

No pre-registered decision rule: this is a response curve, and the shape is the
result.  The existing 1.0 and 1.1386 arms live in olmo_gain/ and are read from
there so all points come from the same panel and scorer.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
DIRS = [f"{ROOT}/olmo_gain", f"{ROOT}/olmo_gsweep"]
YARN = 1.138629436111989


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
    arms = {}
    for d in DIRS:
        for f in glob.glob(f"{d}/gain_bm_g*.jsonl"):
            g = os.path.basename(f)[len("gain_bm_g"):-len(".jsonl")].replace("p", ".")
            try:
                arms[float(g)] = load(f)
            except ValueError:
                continue
    if not arms:
        print("REFUSING: no gain_bm_g* arms found")
        return 2

    base = arms.get(1.0)
    if base is None:
        print("REFUSING: gain 1.0 arm missing -- it anchors the curve")
        return 2

    print("=" * 76)
    print("GAIN SWEEP on the deployed table (350-row panel, all @16384)")
    print(f"YaRN's analytic value: 0.1*ln(4)+1 = {YARN:.12f}")
    print("=" * 76)
    print(f"\n  {'gain':>8} {'n':>5} {'acc':>8} {'vs gain1.0':>12} {'t':>7}")
    pts = []
    for g in sorted(arms):
        a = arms[g]
        ids = sorted(set(a) & set(base))
        if len(ids) < 300:
            print(f"  {g:>8.4f} {len(ids):>5}  (still running)")
            continue
        d = np.array([a[i]["correct"] - base[i]["correct"] for i in ids], float)
        se = d.std(ddof=1) / math.sqrt(len(d))
        acc = np.mean([a[i]["correct"] for i in ids])
        t = d.mean() / se if se > 0 else float("nan")
        pts.append((g, acc, d.mean(), se, t))
        mark = "  <- YaRN" if abs(g - YARN) < 1e-9 else ""
        print(f"  {g:>8.4f} {len(ids):>5} {acc:>8.4f} {d.mean()*100:>+11.2f}pp "
              f"{t:>+7.2f}{mark}")

    if len(pts) >= 3:
        best = max(pts, key=lambda p: p[1])
        lo = min(pts, key=lambda p: abs(p[0] - 1.10))
        hi = min(pts, key=lambda p: abs(p[0] - 1.20))
        print(f"\n  measured best: gain={best[0]:.4f}  acc={best[1]:.4f}")
        print(f"  YaRN value   : gain={YARN:.4f}  "
              f"acc={[p[1] for p in pts if abs(p[0]-YARN) < 1e-9][0]:.4f}"
              if any(abs(p[0] - YARN) < 1e-9 for p in pts) else "")
        print("\n--- reading ---")
        # NOTE: "interior" here means the peak is bracketed on BOTH sides by
        # measured points.  If the upper bracket is still running, say so rather
        # than implying the derived value is off.
        upper_measured = max(p[0] for p in pts)
        if abs(best[0] - YARN) < 1e-9:
            print(f"  YaRN's derived value {YARN:.6f} IS the measured best of the "
                  f"{len(pts)} points done.")
            if upper_measured <= YARN:
                print(f"  (upper bracket not measured yet: highest point is "
                      f"{upper_measured:.4f}. Re-read once it lands before")
                print("   claiming the optimum is at or below the derived value.)")
            else:
                print("  Both sides are measured and the derived value sits at the")
                print("  top -- the analytic derivation is validated here.")
        elif best[1] > [p[1] for p in pts if abs(p[0]-YARN) < 1e-9][0]:
            print(f"  A measured point ({best[0]:.4f}) beats YaRN's value "
                  f"({YARN:.6f}) on this panel.  Report the curve.")
        else:
            print("  Report the curve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
