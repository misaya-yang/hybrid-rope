#!/usr/bin/env python3
"""Read the ordering causal experiment, and separate ORDER from sum(m).

THE DESIGN, AND THE ONE THING THAT MAKES IT A CAUSAL TEST.  Six tables share a
band, endpoints, total span AND the multiset of increments {2q/(n(n+1))}.  Only
the order in which that mass is spent differs, so any difference between them is
attributable to sequence.  `progressive` is MrRoPE -- identity permutation, float
equal to `m_mrpro(17)` -- and its score is archived, so it enters as a free
anchor rather than as a re-run.

THE CONFOUND IS BUILT IN AND IS REPORTED FIRST.  Holding the multiset fixed does
NOT hold sum(m) fixed: spending the large increments last (progressive) gives
sum(m) = 29.33, spending them first (reversed) gives 34.67.  sum(m) is the
average compression actually applied, so a score that tracks sum(m) is evidence
for "how much" while a score that tracks the permutation is evidence for "in what
order".  The reader reports the correlation of the score against BOTH, because
the two explanations make different predictions and this experiment can tell them
apart -- which is the whole point of running it.

WHAT EACH OUTCOME MEANS (declared before the numbers are read):

  progressive wins, others ordered by how far they are from it
      -> ordering is the causal variable. MrRoPE's arithmetic progression is a
         mechanism, not an accident.
  score tracks sum(m) and not the permutation
      -> everything collapses to "how much total compression", the ordering story
         is decoration, and the family's whole shape literature is about a
         quantity that was never the operative one.
  some random permutation ties or beats progressive
      -> arithmetic progression is a LUCKY POINT. The Pro analysis names this
         outcome explicitly, and tonight's KKT measurement already leans toward
         it: <e, BM - MrRoPE> = -0.178 against |e| = 92.4 is near-orthogonal, and
         a genuine order effect should not be invisible to the first-order term.
  progressive wins on exact match but margins are flat
      -> threshold amplification across an argmax rather than a capability gain.

No branch is a failure.  The first one is the first mechanistic explanation this
project would have; the second retires an entire literature line; the third
retires MrRoPE's specialness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The archived progressive endpoint, entered from the run receipt.
N_INCR = 18   # OLMo band width; this reader is for the OLMo ordering run
ARCHIVED = {"progressive": dict(acc=0.0709, src="run_ruler_newtasks_01/MrPro.json")}


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_distance(p, identity):
    """Kendall tau distance: how far a permutation is from progressive."""
    p = list(p)
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1
    return inv


def check_perms_are_distinct(n=17, seed=20260910):
    """There must be six DIFFERENT orderings, and that is not automatic.

    A random permutation can coincide with the identity or with the reversal, and
    a duplicated arm would look like a replication while actually being the same
    table twice -- which would then be read as a stable result.  This is checked
    rather than assumed because the seed is fixed and the draw is not re-examined
    at run time.
    """
    import numpy as _np
    rng = _np.random.default_rng(int(seed))
    perms = {"progressive": _np.arange(n), "reversed": _np.arange(n)[::-1]}
    for i in range(4):
        perms[f"rand{i + 1}"] = rng.permutation(n)
    seen = {}
    dupes = []
    for name, p in perms.items():
        key = tuple(int(x) for x in p)
        if key in seen:
            dupes.append((seen[key], name))
        seen[key] = name
    return perms, dupes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="olmo_ord run dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    root = Path(args.dir)
    rows = []
    for f in sorted(root.glob("ord_*_summary.json")):
        d = json.loads(f.read_text())
        name = d.get("arm", f.stem).replace("ord_", "")
        rows.append(dict(name=name, acc=float(d["accuracy"]), n=int(d["n"]),
                         sum_m=float(d["sum_m"]),
                         wins=int((d.get("vs_MrPro") or {}).get("wins", 0)),
                         losses=int((d.get("vs_MrPro") or {}).get("losses", 0))))
    if not rows:
        print(f"no ord_*_summary.json under {root}", file=sys.stderr)
        return 1

    arch = ARCHIVED["progressive"]
    rows.append(dict(name="progressive", acc=arch["acc"], n=350,
                     sum_m=float("nan"), wins=None, losses=None, archived=True))

    # recover the permutations so the distance axis is available
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase1_screen import order_perms
    # n must match the checkpoint whose band this run used; OLMo's band is 18
    # slots wide, and drawing an n=17 permutation there crashes m_order's guard.
    perms = order_perms(n=N_INCR)
    _, dupes = check_perms_are_distinct(n=N_INCR)
    if dupes:
        print(f"REFUSING: duplicate orderings {dupes}; the arm set is not six "
              "distinct tables", file=sys.stderr)
        return 2
    for r in rows:
        p = perms.get(r["name"])
        r["kendall_from_progressive"] = (perm_distance(p, None)
                                         if p is not None else float("nan"))

    rows.sort(key=lambda r: r["acc"], reverse=True)
    print("=== ordering causal experiment ===")
    print(f"{'order':14s} {'acc':>8s} {'sum_m':>8s} {'Kendall':>8s} "
          f"{'W':>4s} {'L':>4s}  note")
    for r in rows:
        note = "ARCHIVED (MrRoPE, not re-run)" if r.get("archived") else ""
        print(f"{r['name']:14s} {r['acc']:8.4f} {r['sum_m']:8.3f} "
              f"{r['kendall_from_progressive']:8.1f} "
              f"{(r['wins'] if r['wins'] is not None else ''):>4} "
              f"{(r['losses'] if r['losses'] is not None else ''):>4}  {note}")

    measured = [r for r in rows if not r.get("archived")]
    if len(measured) >= 3:
        acc = [r["acc"] for r in measured]
        sm = [r["sum_m"] for r in measured]
        kd = [r["kendall_from_progressive"] for r in measured]
        r_sm = spearman(sm, acc)
        r_kd = spearman(kd, acc)
        print(f"\n  Spearman(score, sum_m)                  = {r_sm:+.3f}")
        print(f"  Spearman(score, Kendall-from-progressive) = {r_kd:+.3f}")
        print()
        if abs(r_sm) > abs(r_kd) + 0.3:
            print("  -> the score tracks SUM(m), not the permutation.")
            print("     'How much' beats 'in what order': the ordering story is")
            print("     decoration and the operative variable was never the shape.")
        elif abs(r_kd) > abs(r_sm) + 0.3:
            print("  -> the score tracks the PERMUTATION, not sum(m).")
            print("     Ordering is a real causal variable.")
        else:
            print("  -> neither axis dominates; the six points do not separate")
            print("     them. More permutations, or a range of sum(m) at fixed")
            print("     order, would be needed to tell the two apart.")
        top = max(measured, key=lambda r: r["acc"])
        if top["name"] != "progressive" and top["acc"] > arch["acc"]:
            print(f"\n  LUCKY-POINT OUTCOME: {top['name']} ({top['acc']:.4f}) beats the "
                  f"archived progressive ({arch['acc']:.4f}).")
            print("  Arithmetic progression is then not a law, and the table that")
            print("  beats MrRoPE here is a concrete, non-archival candidate.")

    out = dict(rows=rows, archived=arch,
               spearman_score_vs_sum_m=(r_sm if len(measured) >= 3 else None),
               spearman_score_vs_kendall=(r_kd if len(measured) >= 3 else None),
               scope=("six tables differing ONLY in the order of a fixed increment "
                      "multiset; the progressive member is the archived MrRoPE row "
                      "and was not regenerated. sum(m) is free and is reported "
                      "because it is the competing explanation."))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
