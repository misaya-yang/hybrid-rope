#!/usr/bin/env python3
"""Read the frontier walk on the CONTINUOUS instrument, at both lengths.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python contwalk_read.py

Governed by WALK_PREREG_20260911.md sections 7b/7c, written before these numbers
were read.

WHAT IT DECIDES.  CONSTRAINT_IS_SLACK_20260911.md holds, at three stars, that
in-band reallocation is free -- measured as the per-slot in-window gradient of
this instrument (sign-alternating, 0.002-0.010, mean +0.0015, mostly inside 1-3
paired SE).  But that is a statement about the MEAN.  The RULER panel loses 4-6pp
in-window while only 12 of 60 rows ever break, which is a statement about the
TAIL.  A mean and a tail can disagree, and these two do unless the damage is
sparse.

  contwalk/    --length 16385  -> the LONG-RANGE mean
  contwalk4k/  --length 4097   -> the IN-WINDOW mean

  in-window mean flat       + RULER 4096 clearly negative -> sparse tail.
        "in-band free on average" survives; in-band is NOT thereby harmless,
        and section 3 of the index needs that qualifier.
  in-window mean rises with a + RULER 4096 clearly negative -> the mean moves.
        "in-band free" was a low-power artefact and section 3 must be demoted.

Paired across the same 16 documents, so the SE is the paired one, not the
across-arm one -- the per-document spread is large (1.38 to 3.89 nats) and
pairing removes it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
RUNS = [("contwalk", 16384, "LONG-RANGE mean (16384)"),
        ("contwalk4k", 4096, "IN-WINDOW mean (4096)")]
DOSES = ["walk_a0p0", "walk_a0p25", "walk_a0p5", "walk_a0p75", "walk_a1p0"]


def load(d):
    path = os.path.join(ROOT, d, "rows.jsonl")
    out = {}
    if not os.path.exists(path):
        return None
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "arm" in r:
            out[r["arm"]] = r
    return out


def main():
    any_shown = False
    for d, L, label in RUNS:
        rows = load(d)
        if not rows:
            print(f"\n### {label}: not run yet ({d}/rows.jsonl missing)")
            continue
        have = [a for a in DOSES if a in rows]
        if "walk_a0p0" not in rows or len(have) < 2:
            print(f"\n### {label}: only {have} -- need a=0 plus at least one more")
            continue
        any_shown = True
        print("\n" + "=" * 78)
        print(f"{label}   ({len(have)}/{len(DOSES)} doses)")
        print("=" * 78)
        base = np.array(rows["walk_a0p0"]["per_doc"], float)
        print(f"  a=0 (the deployed table): NLL = {base.mean():.6f}")

        print(f"\n  {'a':>5} {'NLL':>10} {'dNLL':>10} {'se':>8} {'t':>7}   better?")
        dd = {}
        for a in have:
            r = rows[a]
            v = np.array(r["per_doc"], float)
            if len(v) != len(base):
                print(f"  {a}: doc count {len(v)} != {len(base)}; skipped")
                continue
            delta = v - base
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            t = delta.mean() / se if se > 0 else float("nan")
            dd[a] = (delta.mean(), se, t)
            print(f"  {a[-4:]:>5} {v.mean():>10.6f} {delta.mean():>+10.6f} "
                  f"{se:>8.6f} {t:>+7.2f}   {'yes' if delta.mean() < 0 else 'no'}")

        # linearity in a, anchored at a=0 where dNLL == 0 by construction
        xs, ys, ws = [], [], []
        for a in have:
            av = float(a[len("walk_a"):].replace("p", "."))
            if av == 0.0 or a not in dd:
                continue
            xs.append(av)
            ys.append(dd[a][0])
            ws.append(1.0 / max(dd[a][1], 1e-12) ** 2)
        if xs:
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)
            slope = float((xs * ys * ws).sum() / (xs * xs * ws).sum())
            resid = ys - xs * slope
            chi2 = float((resid ** 2 * ws).sum() / max(len(xs) - 1, 1))
            print(f"\n  linear fit anchored at a=0:  slope = {slope:+.6f} nats/unit a"
                  f"   chi2/dof = {chi2:.2f}")
            print(f"    observed dNLL: " + "  ".join(f"{y:+.6f}" for y in ys))
            print(f"    linear       : " + "  ".join(f"{x*slope:+.6f}" for x in xs))

        # the pre-registered adjudication
        nz = [(a, dd[a]) for a in have if a != "walk_a0p0" and a in dd]
        if nz:
            worst = max(v[0] for _, v in nz)
            best = min(v[0] for _, v in nz)
            sig_bad = [a for a, v in nz if v[0] > 0 and abs(v[2]) >= 2.0]
            print(f"\n  range of dNLL across doses: {best:+.6f} .. {worst:+.6f}")
            if sig_bad:
                print(f"  MEAN MOVES: {sig_bad} are significantly WORSE than a=0 "
                      f"(t>=2).")
                if L == 4096:
                    print("  -> 'in-band free on average' is a low-power artefact "
                          "at this length; index section 3 must be DEMOTED.")
            else:
                print("  MEAN FLAT: no dose is significantly worse than a=0.")
                if L == 4096:
                    print("  -> sparse tail: the in-window cost is a distributional")
                    print("     effect the mean cannot see.  'In-band free on")
                    print("     average' survives, but in-band is NOT harmless and")
                    print("     index section 3 needs that qualifier.")
    if not any_shown:
        print("\nnothing to read yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
