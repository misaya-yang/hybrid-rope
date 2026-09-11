#!/usr/bin/env python3
"""Read the powered Qwen 4x verdict.  Governed by QWEN4X_POWER_PREREG_20260911.md.

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python qwen4x_power_read.py

THE QUESTION.  THE_ANSWER's leverage table has one underpowered row: on Qwen at
4x, BM minus MrRoPE is -0.0074 with t=-3.34, but only 4 documents -- sign test
4/4, p=0.125.  That is "direction right, power absent", and it is the one number
holding the sign-flip claim open.

THE FIX.  A fresh PG19 corpus: 30 books-worth of non-overlapping 131073-token
segments from PG19 test, tokenised with Qwen's own tokenizer.  The instrument
concatenates and re-slices at exactly this length, so pieces align with source
documents one-for-one and the book id carries over.

THE STATISTIC.  Per-piece paired (BM - mrpro).  Two SEs are reported:
  * by piece      (n = 29)
  * clustered by book (~18 clusters)   <- the pre-registered criterion
Chunks of one book are not independent, so the clustered SE is the honest one.

    |delta| >= 0.004 and |t| >= 3 (clustered) -> the sign is settled
                                                 negative => MrRoPE genuinely
                                                 wins on Qwen at 4x
    |delta| <  0.002                          -> indistinguishable on Qwen 4x
    otherwise                                 -> unresolved; report and stop
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
RUN = f"{ROOT}/qwen4x_power"
MANIFEST = "/root/autodl-tmp/longtext/prepared_pg19_4x/rows.json"


def load_rows(p):
    out = {}
    if not os.path.exists(p):
        return None
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "arm" in r:
            out[r["arm"]] = r
    return out


def clustered_se(d, groups):
    """SE of the mean with one cluster per group (book): use cluster means."""
    g = {}
    for v, k in zip(d, groups):
        g.setdefault(k, []).append(v)
    m = np.array([np.mean(v) for v in g.values()], float)
    if len(m) < 2:
        return float("nan")
    return m.std(ddof=1) / math.sqrt(len(m))


def main():
    rows = load_rows(f"{RUN}/rows.jsonl")
    if not rows or "native" not in rows:
        print(f"REFUSING: {RUN}/rows.jsonl missing or lacks native")
        return 2
    need = ("beta_b1_BM", "mrpro")
    if not all(k in rows for k in need):
        print(f"still running: have {sorted(rows)}")
        return 0

    bm = np.array(rows["beta_b1_BM"]["per_doc"], float)
    mr = np.array(rows["mrpro"]["per_doc"], float)
    nat = np.array(rows["native"]["per_doc"], float)
    n = min(len(bm), len(mr))
    bm, mr, nat = bm[:n], mr[:n], nat[:n]

    books = None
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
        books = [man[i]["book"] for i in range(n)] if len(man) >= n else None

    print("=" * 76)
    print("QWEN 4x, POWERED  (PG19 test, 131073-token segments)")
    print("=" * 76)
    print(f"  pieces: {n}   books: {len(set(books)) if books else 'n/a'}")

    dd = bm - mr
    se_p = dd.std(ddof=1) / math.sqrt(len(dd))
    print(f"\n  BM NLL   = {np.mean(rows['beta_b1_BM']['per_doc']) + 0:+.6f} (delta form)")
    print(f"  mrpro    = {np.mean(rows['mrpro']['per_doc']):+.6f} (delta form)")
    print(f"\n  BM - MrRoPE  = {dd.mean():+.6f} nats")
    print(f"    by piece       : SE={se_p:.6f}  t={dd.mean()/se_p:+.2f}  (n={len(dd)})")
    if books:
        se_c = clustered_se(dd, books)
        print(f"    by book cluster: SE={se_c:.6f}  t={dd.mean()/se_c:+.2f}  "
              f"({len(set(books))} clusters)")
        se_use, t_use = se_c, dd.mean() / se_c
    else:
        se_use, t_use = se_p, dd.mean() / se_p
    print(f"    sign agreement : {int((dd<0).sum())} negative / "
          f"{int((dd>0).sum())} positive of {len(dd)}")

    res = bm - nat
    se_r = res.std(ddof=1) / math.sqrt(len(res))
    print(f"\n  rescue (native - BM) = {-res.mean():+.6f} nats  SE={se_r:.6f}  "
          f"t={-res.mean()/se_r:+.2f}")

    if books:
        print("\n  --- per book (BM - MrRoPE) ---")
        for b in sorted(set(books)):
            idx = [i for i, x in enumerate(books) if x == b]
            print(f"    book {b:>3}  n={len(idx):<3} {dd[idx].mean():+.6f}")

    print("\n--- VERDICT (rule fixed in QWEN4X_POWER_PREREG_20260911.md) ---")
    if abs(dd.mean()) >= 0.004 and abs(t_use) >= 3.0:
        print(f"  SIGN SETTLED.  BM - MrRoPE = {dd.mean():+.6f} nats, t={t_use:+.2f}.")
        if dd.mean() < 0:
            print("  NEGATIVE: MrRoPE genuinely beats BM on Qwen at 4x.  Combined with")
            print("  OLMo 4x (+0.826 the other way), the sign flip is established with")
            print("  power and the 'rescue-dependent optimum' answer closes.")
        else:
            print("  POSITIVE: BM beats MrRoPE on Qwen too.  The n=4 reading was noise")
            print("  and the leverage table's Qwen row must be rewritten.")
    elif abs(dd.mean()) < 0.002:
        print(f"  INDISTINGUISHABLE on Qwen 4x (|delta|={abs(dd.mean()):.6f} < 0.002).")
        print("  The leverage table's Qwen row should read 'no difference'.")
    else:
        print(f"  UNRESOLVED: delta={dd.mean():+.6f} t={t_use:+.2f}. Report and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
