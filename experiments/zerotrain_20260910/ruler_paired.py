#!/usr/bin/env python3
"""Paired RULER comparison across every arm that has per-row data on this panel.

Run ON THE SERVER, where the row files live.

The statistic is the mean of the per-row difference (correct_a - correct_b) over
the rows both arms share, with its paired SE -- the same test for every pair, so
the numbers are comparable to each other.  The sign-test column (W/L/T over the
non-tied rows) is reported alongside because with 255/350 ties the two answer
slightly different questions, and this campaign has already been burned once by
comparing a paired number against an unpaired one.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
ARCH = "/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01"

FILES = {
    "ctl_C42V24": f"{ROOT}/olmo_c42/ctl_C42V24.jsonl",
    "ctl_C42": f"{ROOT}/olmo_c42/ctl_C42.jsonl",
    "turns_a1_b64": f"{ROOT}/olmo_turns/turns_a1_b64.jsonl",
    "turns_a1_b64_r2": f"{ROOT}/r2turns/turns_a1_b64.jsonl",
    "turns_a1_b64_olmo": f"{ROOT}/olmo/turns_a1_b64.jsonl",
    "turns_a1_b16": f"{ROOT}/olmo_turns/turns_a1_b16.jsonl",
    "turns_a0p5_b32": f"{ROOT}/olmo/turns_a0p5_b32.jsonl",
    "step_hi25": f"{ROOT}/olmo_tstar/step_hi25.jsonl",
    "b3_lo14": f"{ROOT}/olmo_b3/beta_b3p0.jsonl",
    "wide_b4": f"{ROOT}/olmo_wide/wide_b4p0.jsonl",
    "MrProBM": f"{ARCH}/MrProBM.jsonl",
    "MrPro": f"{ARCH}/MrPro.jsonl",
}

PAIRS = [
    ("ctl_C42V24", "turns_a1_b64"),
    ("ctl_C42V24", "turns_a1_b64_r2"),
    ("ctl_C42V24", "turns_a1_b64_olmo"),
    ("ctl_C42V24", "turns_a1_b16"),
    ("ctl_C42V24", "MrProBM"),
    ("ctl_C42V24", "ctl_C42"),
    ("ctl_C42V24", "b3_lo14"),
    ("ctl_C42V24", "wide_b4"),
    ("turns_a1_b64", "MrProBM"),
    ("turns_a1_b64_olmo", "MrProBM"),
    ("b3_lo14", "MrProBM"),
    ("ctl_C42", "MrProBM"),
]


def rows(path):
    out = {}
    try:
        with open(path) as fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                if "row_id" in d:
                    out[d["row_id"]] = d.get("correct")
    except Exception:
        pass
    return out


def main():
    R = {}
    for k, p in FILES.items():
        if os.path.exists(p):
            R[k] = rows(p)
    print("可用逐行数据:")
    for k in sorted(R):
        print("   %-20s n=%d" % (k, len(R[k])))

    def paired(a, b):
        if a not in R or b not in R:
            return None
        ka, kb = R[a], R[b]
        common = sorted(set(ka) & set(kb))
        if not common:
            return None
        va = np.array([ka[c] for c in common], float)
        vb = np.array([kb[c] for c in common], float)
        d = va - vb
        w = int((d > 0).sum())
        lo = int((d < 0).sum())
        ti = int((d == 0).sum())
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        return dict(n=len(common), w=w, l=lo, tie=ti,
                    dp=float(d.mean()) * 100, se=float(se) * 100,
                    tv=float(d.mean() / se) if se > 0 else 0.0)

    print("\n=== 配对检验（同 row_id，Δ = a − b，单位 pp）===")
    print("  %-20s %-20s %4s %11s %9s %8s" % ("a", "b", "n", "W/L/T", "Δpp", "t"))
    for a, b in PAIRS:
        r = paired(a, b)
        if r is None:
            continue
        print("  %-20s %-20s %4d %4d/%3d/%3d %+9.2f %+8.2f %s"
              % (a, b, r["n"], r["w"], r["l"], r["tie"], r["dp"], r["tv"],
                 "**显著**" if abs(r["tv"]) > 2.1 else "不显著"))

    print("\n=== 绝对分 ===")
    for k in sorted(R, key=lambda z: -np.mean([x for x in R[z].values()
                                               if x is not None])):
        vals = [x for x in R[k].values() if x is not None]
        if vals:
            print("   %-20s n=%-4d acc=%.4f" % (k, len(vals), float(np.mean(vals))))


if __name__ == "__main__":
    main()
