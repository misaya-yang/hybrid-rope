"""Reader for the amplitude sweep (AMP8X_SWEEP_PREREG_20260911).

Reports the two lengths SEPARATELY, because the prereg's primary output is the
32768 row count and its secondary output is 4096 -- collapsing them would hide
the trade the whole design is about.

Framework: acc is higher-is-better, so a POSITIVE delta favours the second arm.
This script prints a "who is better" column rather than a signed delta alone --
an earlier sign-reading error in this campaign came from exactly that ambiguity.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

ROOT = os.environ.get("AMP_ROOT", "s8_amp8x")
ARMS = ["wb1", "s1p25", "s1p5", "s1p75", "s2p0"]   # wb1 = the s=1.00 anchor
LENS = (4096, 32768)


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    path = os.path.join(ROOT, "rows.jsonl")
    if not os.path.exists(path):
        print(f"no rows yet at {path}")
        return 2
    rows = load(path)

    # key the panel rows by (row_id, length) so arms pair exactly
    by_arm = defaultdict(dict)
    for r in rows:
        arm = r.get("arm") or r.get("beta") or r.get("label")
        rid = r.get("row_id")
        L = r.get("length") or r.get("len_cap") or r.get("cap")
        if arm is None or rid is None or L is None:
            continue
        by_arm[arm][(rid, int(L))] = r

    print(f"{'arm':8s} " + " ".join(f"{('@%d' % L):>18s}" for L in LENS))
    got = {}
    for arm in ARMS:
        d = by_arm.get(arm, {})
        cells = []
        for L in LENS:
            k = [(rid, l) for (rid, l) in d if l == L]
            if not k:
                cells.append(f"{'-':>18s}")
                continue
            ok = sum(1 for key in k if d[key].get("score", d[key].get("acc", 0)) > 0.5)
            cells.append(f"{ok:>6d}/{len(k):<4d}{ok/len(k):>7.3f}")
            got[(arm, L)] = (ok, len(k))
        print(f"{arm:8s} " + " ".join(cells))

    # paired comparison against the anchor, per length -- the prereg's结构
    print()
    for L in LENS:
        base = got.get(("wb1", L))
        if not base:
            continue
        print(f"-- @{L}: anchor wb1 (s=1.00) = {base[0]}/{base[1]} --")
        for arm in ARMS[1:]:
            cur = got.get((arm, L))
            if not cur:
                continue
            d = cur[0] - base[0]
            better = arm if d > 0 else ("wb1" if d < 0 else "tie")
            print(f"     {arm:6s} {cur[0]:>3d}/{cur[1]:<3d}  delta {d:+d} rows   -> {better}")

    print("\n判据见 prereg_protocols/AMP8X_SWEEP_PREREG_20260911.md §二")
    print("  key readings: is s=1.25 zero (sharp threshold at u=1.5)?")
    print("                does the peak sit at 1.50 (axiom) or 1.75 (its own metric)?")
    print("                is s=2.00 >= s=1.50 (dose-generalisation)?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
