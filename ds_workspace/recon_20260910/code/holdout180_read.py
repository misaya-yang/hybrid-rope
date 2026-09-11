#!/usr/bin/env python3
"""Read the 180-row held-out panel and apply the PRE-REGISTERED decision rule.

Run on the server:
    cd /root/autodl-tmp/phase1_20260910 && python holdout180_read.py

The rule was committed (see HOLDOUT180_PREREG_20260911.md, git 62cef88) BEFORE
any row of this panel was read, and this script implements exactly that rule --
it does not choose the statistic after seeing the numbers.

    pooled delta >= +5pp and t >= 2      -> effect real; 72 rows was underpowered
    pooled delta <= +1.5pp and SE <= 3pp -> effect inside noise; established
    otherwise                            -> unresolved; report the interval, stop

Pooling is legitimate because the hypothesis ("the plateau members beat the
deployed table") was stated before the pool existed.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910/holdout180"
BASE = "beta_b1p0"
MEMBERS = [("b3_lo14", "beta_b3p0"), ("turns_a1_b64", "wide_b1p0"),
           ("wide_b4", "wide_b4p0")]


def load(name):
    path = os.path.join(ROOT, name + ".jsonl")
    out = {}
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for ln in fh:
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
        print(f"REFUSING: {BASE}.jsonl not found under {ROOT}")
        return 2
    present = [(n, k) for n, k in MEMBERS if load(k) is not None]
    missing = [n for n, k in MEMBERS if load(k) is None]

    print("=" * 78)
    print("180-ROW HELD-OUT PANEL  (6 tasks the selection panel never used)")
    print("=" * 78)
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

    # ---- per member ---------------------------------------------------------
    print("\n--- per member vs the deployed table ---")
    print(f"  {'arm':<16} {'acc':>7} {'delta':>9} {'se':>7} {'t':>7}  W/L/T")
    for n, k in present:
        d, se = paired(load(k), base, ids)
        acc = np.mean([load(k)[i]["correct"] for i in ids])
        w = int((d > 0).sum()); l = int((d < 0).sum()); t = int((d == 0).sum())
        print(f"  {n:<16} {acc:7.4f} {d.mean()*100:+9.2f} {se*100:7.2f} "
              f"{(d.mean()/se if se > 0 else 0):+7.2f}  {w}/{l}/{t}")

    # ---- the pre-registered primary statistic -------------------------------
    deltas = np.mean([paired(load(k), base, ids)[0] for _, k in present], axis=0)
    se = deltas.std(ddof=1) / math.sqrt(len(deltas))
    tv = deltas.mean() / se if se > 0 else 0.0
    w = int((deltas > 0).sum()); l = int((deltas < 0).sum()); t = int((deltas == 0).sum())

    print("\n--- PRIMARY (pre-registered): pooled delta over the members ---")
    print(f"  pooled delta = {deltas.mean()*100:+.2f}pp   SE = {se*100:.2f}pp   "
          f"t = {tv:+.2f}   W/L/T = {w}/{l}/{t}")

    print("\n--- VERDICT (rule fixed at git 62cef88, before any row was read) ---")
    if deltas.mean() * 100 >= 5.0 and tv >= 2.0:
        print("  EFFECT REAL.  The 72-row reading was underpowered, not negative.")
    elif deltas.mean() * 100 <= 1.5 and se * 100 <= 3.0:
        print("  EFFECT INSIDE NOISE.  'The tuning does not generalise' is now an")
        print("  established result rather than a failure to measure.")
    else:
        print("  UNRESOLVED.  Report the interval and stop.")
        print("  Per the pre-registration: do NOT add rows to move this.")

    # ---- secondary: stratify by length_cap ----------------------------------
    print("\n--- secondary: stratified by length_cap (not a decision criterion) ---")
    for cap in sorted({base[i].get("length_cap") for i in ids}):
        sub = [i for i in ids if base[i].get("length_cap") == cap]
        if not sub:
            continue
        dd = np.mean([paired(load(k), base, sub)[0] for _, k in present], axis=0)
        ss = dd.std(ddof=1) / math.sqrt(len(dd)) if len(dd) > 1 else float("nan")
        bm = np.mean([base[i]["correct"] for i in sub])
        print(f"  cap={cap:<6} n={len(sub):<4} BM acc={bm:.3f}  pooled delta="
              f"{dd.mean()*100:+.2f}pp  se={ss*100:.2f}pp")

    print("\n--- secondary: per task x cap ---")
    keys = sorted({(base[i].get("task"), base[i].get("length_cap")) for i in ids})
    print(f"  {'task':<18} {'cap':>6} {'n':>4} {'BM':>7} " +
          " ".join(f"{n:>10}" for n, _ in present))
    for tk, cap in keys:
        sub = [i for i in ids if base[i].get("task") == tk
               and base[i].get("length_cap") == cap]
        bm = np.mean([base[i]["correct"] for i in sub])
        cells = []
        for n, k in present:
            acc = np.mean([load(k)[i]["correct"] for i in sub])
            cells.append(f"{(acc - bm)*100:+10.1f}")
        print(f"  {tk:<18} {cap:>6} {len(sub):>4} {bm:7.3f} " + " ".join(cells))
    print("\n  (member columns are delta vs BM in pp)")

    # ---- members against each other ----------------------------------------
    if len(present) > 1:
        print("\n--- secondary: members against each other ---")
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ni, ki = present[i]
                nj, kj = present[j]
                d, s = paired(load(ki), load(kj), ids)
                print(f"  {ni:<16} - {nj:<16} {d.mean()*100:+7.2f}pp  t="
                      f"{(d.mean()/s if s > 0 else 0):+5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
