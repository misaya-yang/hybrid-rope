#!/usr/bin/env python3
"""Read the frontier walk and apply the PRE-REGISTERED rule (WALK_PREREG_20260911.md).

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python walk_read.py

WHAT THIS MEASURES.  m(a) = (1-a)*m_BM + a*m_a1b64, a in {0,.25,.5,.75,1}.  The
two endpoints are already-measured arms, so the whole curve is bracketed by known
points and the in-window/out-of-window EXCHANGE RATE is measured rather than
argued.  a=0 and a=1 are re-run inside this one process as built-in controls.

THE STRONG NULL IS LINEARITY.  Because a=0 is BM by construction, Delta(0) == 0
exactly, so a linear response is Delta(a) = a*delta -- a straight line through the
origin.  If that holds, NO interior point of the segment can gain long range
without losing in-window, and the answer to "can we get DEVELOPMENT_WIN here" is
a measured no.  A kink is what a DEVELOPMENT_WIN would look like.

GUARD AGAINST THE SELECTION EFFECT.  The pre-registration is explicit that finding
a good a here only LOCATES a candidate.  It cannot be claimed as a result until it
reproduces on data this segment was not fitted on.  The 180 rows were already used
to choose the plateau members, so a win found here is exactly the shape of the
+12..14pp illusion that the held-out panel destroyed.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
WALK = f"{ROOT}/walk_out"
REF = f"{ROOT}/holdout180"
CAPS = (4096, 16384)


def load(path):
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
    arms = {}
    for f in sorted(glob.glob(f"{WALK}/walk_a*.jsonl")):
        nm = os.path.basename(f)[:-6]
        arms[nm] = load(f)
    if not arms:
        print(f"REFUSING: no walk arms under {WALK}")
        return 2

    bm = load(f"{REF}/beta_b1p0.jsonl")
    if bm is None:
        print(f"REFUSING: reference BM {REF}/beta_b1p0.jsonl missing")
        return 2

    def aval(nm):
        return float(nm[len("walk_a"):].replace("p", "."))

    order = sorted(arms, key=aval)
    ids = sorted(bm)
    for nm in order:
        ids = [i for i in ids if i in arms[nm]]
    print("=" * 84)
    print("FRONTIER WALK  m(a) = (1-a)*m_BM + a*m_a1b64   on the 180-row panel")
    print("=" * 84)
    print(f"  arms present: {', '.join(order)}")
    print(f"  common rows : {len(ids)}   "
          f"(4096: {sum(1 for i in ids if bm[i]['length_cap']==4096)}, "
          f"16384: {sum(1 for i in ids if bm[i]['length_cap']==16384)})")

    # ---- built-in endpoint controls ----------------------------------------
    print("\n--- CONTROLS: a=0 must BE the deployed BM, a=1 must BE a1_b64 ---")
    for nm, ref, label in (("walk_a0p0", f"{REF}/beta_b1p0.jsonl", "deployed BM"),
                           ("walk_a1p0", f"{REF}/wide_b1p0.jsonl", "turns_a1_b64")):
        if nm not in arms:
            print(f"  {nm:<11} MISSING")
            continue
        r = load(ref)
        if r is None:
            print(f"  {nm:<11} reference {ref} missing")
            continue
        common = sorted(set(arms[nm]) & set(r))
        ident = sum(1 for i in common
                    if arms[nm][i]["output_text"] == r[i]["output_text"])
        corr = sum(1 for i in common if arms[nm][i]["correct"] == r[i]["correct"])
        flag = "OK" if ident == len(common) else "*** MISMATCH ***"
        print(f"  {nm:<11} vs {label:<14} rows={len(common):<4} "
              f"identical_text={ident} identical_correct={corr}  {flag}")
        if ident != len(common):
            print("     ^^^ the instrument moved between runs: the curve is NOT")
            print("         comparable to the stored endpoints. STOP.")
            return 3

    # ---- the curve ----------------------------------------------------------
    print("\n--- THE CURVE (paired vs the deployed table, measured in-run) ---")
    print(f"  {'a':>5} {'sum_m':>8} | {'d4096':>8} {'se':>6} {'t':>6} | "
          f"{'d16384':>8} {'se':>6} {'t':>6} | {'rate':>7}")
    curve = {}
    for nm in order:
        a = aval(nm)
        row = {}
        for cap in CAPS:
            sub = [i for i in ids if bm[i]["length_cap"] == cap]
            d, se = paired(arms[nm], bm, sub)
            row[cap] = (d.mean(), se, d.mean() / se if se > 0 else float("nan"))
        rate = (row[16384][0] / -row[4096][0]) if row[4096][0] < 0 else float("inf")
        curve[a] = row
        print(f"  {a:>5.2f} {'':>8} | {row[4096][0]*100:>+8.2f} {row[4096][1]*100:>6.2f} "
              f"{row[4096][2]:>+6.2f} | {row[16384][0]*100:>+8.2f} "
              f"{row[16384][1]*100:>6.2f} {row[16384][2]:>+6.2f} | {rate:>7.2f}")

    # ---- absolute accuracies (a big delta on a floor is not a capability) ---
    print("\n--- ABSOLUTE accuracy (BM is the a=0 row) ---")
    for cap in CAPS:
        sub = [i for i in ids if bm[i]["length_cap"] == cap]
        cells = "  ".join(f"a={a:.2f}:{np.mean([arms[nm][i]['correct'] for i in sub]):.3f}"
                          for a, nm in ((aval(n), n) for n in order))
        print(f"  cap={cap:<6} n={len(sub):<4} {cells}")

    # ---- the pre-registered null: linearity through the origin -------------
    print("\n--- LINEARITY (the pre-registered strong null) ---")
    for cap in CAPS:
        xs = np.array([a for a in sorted(curve)], float)
        ys = np.array([curve[a][cap][0] for a in sorted(curve)], float)
        w = np.array([1.0 / max(curve[a][cap][1], 1e-9) ** 2 for a in sorted(curve)])
        # Delta(0) == 0 by construction, so fit the single slope delta
        delta = float((xs * ys * w).sum() / (xs * xs * w).sum())
        pred = xs * delta
        res = ys - pred
        chi2 = float((res ** 2 * w).sum() / max(len(xs) - 1, 1))
        print(f"  cap={cap:<6} delta={delta*100:+.3f}pp per unit a   "
              f"chi2/dof={chi2:.2f}   max|resid|={np.abs(res).max()*100:.2f}pp")
        print(f"         observed: " + "  ".join(f"{y*100:+.2f}" for y in ys))
        print(f"         linear  : " + "  ".join(f"{p*100:+.2f}" for p in pred))

    # ---- per task -----------------------------------------------------------
    print("\n--- per task x cap (delta vs BM, pp) ---")
    keys = sorted({(bm[i].get("task"), bm[i].get("length_cap")) for i in ids})
    hdr = "  ".join(f"a={aval(n):.2f}" for n in order)
    print(f"  {'task':<18} {'cap':>6} {'n':>4} {'BM':>7}  {hdr}")
    for tk, cap in keys:
        sub = [i for i in ids if bm[i].get("task") == tk and bm[i].get("length_cap") == cap]
        bmv = np.mean([bm[i]["correct"] for i in sub])
        cells = "  ".join(
            f"{np.mean([arms[n][i]['correct']-bm[i]['correct'] for i in sub])*100:>+6.1f}"
            for n in order)
        print(f"  {tk:<18} {cap:>6} {len(sub):>4} {bmv:>7.3f}  {cells}")

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 84)
    print("VERDICT (rule fixed in WALK_PREREG_20260911.md, git 82dc5e9, "
          "before any row was read)")
    print("=" * 84)
    interior_win = [(a, curve[a]) for a in sorted(curve)
                    if a > 0 and curve[a][4096][0] >= 0.0 and curve[a][16384][0] >= 0.04]
    all_neg = all(curve[a][4096][0] < 0 for a in sorted(curve) if a > 0)
    if interior_win:
        a, row = interior_win[0]
        print(f"  INTERIOR CANDIDATE at a={a:.2f}: d4096={row[4096][0]*100:+.2f}pp, "
              f"d16384={row[16384][0]*100:+.2f}pp")
        print("  Per the pre-registration this LOCATES a candidate only.  It was")
        print("  found on rows already used to choose the plateau members, so it")
        print("  must reproduce on data this segment was not fitted on before it")
        print("  can be called a result.  Do NOT report it as a win yet.")
    elif all_neg:
        print("  PURE TRADE on this segment: every measured a>0 loses in-window")
        print("  AND gains long range.  Exchange rate per a above.  Solution space")
        print("  contracts: no DEVELOPMENT_WIN point exists on this segment.")
    else:
        print("  UNRESOLVED: non-monotone or mixed signs.  Report the interval and")
        print("  stop; the sign structure itself is the finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
