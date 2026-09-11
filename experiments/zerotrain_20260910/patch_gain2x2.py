#!/usr/bin/env python3
"""Extend --gains so the Pro model's 2x2 gain-interaction design can actually run.

THE PRO MODEL'S REQUEST (RESEARCH_PLAN.md §5, §9.2).  Our campaign measured that
the rotary amplitude `gain` is worth 0.0502 nats in-window while the whole
frequency table is worth 0.0007 -- a 70x ratio -- and every arm in the campaign
carries the gain MrRoPE inherited from YaRN (1.138629436111989).  The plan makes
two corrections and one prescription:

  * CORRECTION: the 70x depends on which denominator you pick, and it is a
    NATIVE-vs-native NLL ratio, not a RULER effect ratio.  Gain is a potential
    EFFECT MODIFIER, not a confounder that invalidates the paired comparisons we
    already have.
  * CORRECTION: it is too strong to say the literature never handled gain
    jointly -- AdaRoPE studies exactly that.
  * PRESCRIPTION: measure the interaction.  Tables {a1_b64, MrRoPE} x gain
    {1.0, 1.138629436111989}, and report

        I = [R(new,g1) - R(Mr,g1)] - [R(new,gY) - R(Mr,gY)]

    I ~ 0 supports a stable table effect over this range; I clearly nonzero means
    the two must be optimised jointly.

WHY THE EXISTING FLAG CANNOT DO IT.  `--gains` hard-codes
`base = m_incr_beta(1.0, ...)`, i.e. the deployed BM only, so the only 2x2 it can
reach is {BM} x {g1, gY} -- one table, which cannot show an interaction.

WHAT THIS PATCH ADDS.  `--gain-tables` takes a comma list of table names built in
THIS checkpoint's geometry:

    bm      deployed BM             m_incr_beta(1.0, n=18, low=14)
    a1_b64  the RULER-confirmed winner  m_incr_beta(1.0, n=21, low=11)
    b3      the campaign champion   m_incr_beta(3.0, n=18, low=14)
    mrpro   the incumbent           m_mrpro(n=18, low=14)

Each is run at each value of `--gains`, so `--gains 1.0,1.138629436111989
--gain-tables bm,a1_b64,mrpro` produces the full 2x2 (plus BM, which is the
archived endpoint and a free consistency check).
"""
from __future__ import annotations

import pathlib
import sys

P = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")

OLD = '''        from experiments.curvature_20260910.tables import m_incr_beta as _mb
        base = np.asarray(_mb(1.0, n=OLMO["n"], low=OLMO["low"]), dtype=np.float64)
        for gs in [s.strip() for s in args.gains.split(",") if s.strip()]:
            g = float(gs)
            print(json.dumps({"gain_arm": g, "table": "deployed BM",
                              "sum_m": float(base.sum())}), flush=True)
            out.append(run_arm(f"gain_{gs}".replace(".", "p"), base, gain=g))'''

NEW = '''        from experiments.curvature_20260910.tables import (m_incr_beta as _mb,
                                                           m_mrpro as _mm)
        # THE GAIN x TABLE DESIGN.  Without a table list this flag can only reach
        # {deployed BM} x {gains}, and one table cannot show an interaction.  The
        # names are built in THIS checkpoint's geometry so the same flag means
        # the same experiment on any checkpoint.
        _tables = {
            "bm":     lambda: np.asarray(_mb(1.0, n=OLMO["n"], low=OLMO["low"]), float),
            "a1_b64": lambda: np.asarray(_mb(1.0, n=21, low=11), float),
            "b3":     lambda: np.asarray(_mb(3.0, n=OLMO["n"], low=OLMO["low"]), float),
            "mrpro":  lambda: np.asarray(_mm(n=OLMO["n"], low=OLMO["low"]), float),
        }
        want = [s.strip() for s in (args.gain_tables or "bm").split(",") if s.strip()]
        bad = [w for w in want if w not in _tables]
        if bad:
            print(f"REFUSING: unknown --gain-tables {bad}; known {sorted(_tables)}",
                  file=sys.stderr)
            return 2
        for tname in want:
            base = _tables[tname]()
            for gs in [s.strip() for s in args.gains.split(",") if s.strip()]:
                g = float(gs)
                print(json.dumps({"gain_arm": g, "table": tname,
                                  "sum_m": float(base.sum())}), flush=True)
                out.append(run_arm(f"gain_{tname}_g{gs}".replace(".", "p"),
                                   base, gain=g))'''


def main():
    s = P.read_text()
    if "--gain-tables" in s:
        print("already patched")
        return 0
    if OLD not in s:
        print("REFUSING: anchor not found", file=sys.stderr)
        return 2
    s = s.replace(OLD, NEW, 1)
    flag_anchor = '    ap.add_argument("--gains", default="",'
    if flag_anchor not in s:
        print("REFUSING: flag anchor not found", file=sys.stderr)
        return 2
    s = s.replace(flag_anchor,
                  '    ap.add_argument("--gain-tables", dest="gain_tables", default="bm",\n'
                  '                    help="comma list of bm,a1_b64,b3,mrpro built in THIS "\n'
                  '                         "geometry; each is run at every --gains value, so "\n'
                  '                         "the gain x table interaction is measurable")\n'
                  + flag_anchor, 1)
    P.write_text(s)
    print("patched olmo_beta.py with --gain-tables (gain x table 2x2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
