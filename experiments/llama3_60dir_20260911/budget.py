"""600-minute budget arithmetic for the 20x3 review.

Implements plan section 8 literally.  The per-candidate cost is

    T_S = 1.15 * [ (s_long/2)(tau16 + tau32) + s_short * tau8 ] / 60      [minutes]

i.e. the two long lengths split the long sample count evenly, plus the in-window
8K guard rows, plus the plan's flat 15% for output-length variation and method
overhead.  Every other line item is the same expression with that item's own
sample counts.

What this module is NOT
-----------------------
It is not a benchmark.  tau8/tau16/tau32 are *inputs*; the plan's own example
values (1.2 / 2.5 / 4.7 s) are labelled "不是硬件测速" and are kept only so the
arithmetic can be checked against the plan's worked table.  Run it with measured
taus before spending anything.

Known discrepancy, reported rather than hidden
----------------------------------------------
The itemized model reproduces plan section 8.1's *candidate counts* exactly
(60 / 49 / 41 / 35) but its reported totals are 0.2-0.5% higher than the plan's
for the scaled rows (e.g. 595.47 vs 594.38 at 8.0 min per candidate).  The plan's
main 599.880 table is reproduced exactly.  The difference is in the plan's own
arithmetic on the sensitivity rows; this module does not fudge a coefficient to
match it.  `--explain` prints both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

OVERHEAD = 1.15          # plan section 8: output length variation + method overhead
BUDGET_MIN = 600.0


@dataclass
class Counts:
    s_long: int = 96
    s_short: int = 16
    v_long: int = 128
    v_short: int = 32
    h_long: int = 256
    h_short: int = 64
    n_controls: int = 4          # MR / YARN / BM / UNI
    n_v_tables: int = 4          # 2 selected candidates + MR + YARN
    n_h_tables: int = 4          # the single locked candidate + MR + YARN + BM
    native_short: int = 112      # s_short + v_short + h_short
    gain_probes: int = 2
    gain_probe_long: int = 48
    gain_probe_short: int = 16
    parity_reserve: float = 7.640
    pilot_reserve: float = 12.000
    exception_reserve: float = 8.000


def t_table(long_n, short_n, tau8, tau16, tau32):
    """Minutes for one table with `long_n` long rows (split evenly) + `short_n` short rows."""
    return OVERHEAD * ((long_n / 2.0) * (tau16 + tau32) + short_n * tau8) / 60.0


def t_candidate(counts, tau8, tau16, tau32):
    return t_table(counts.s_long, counts.s_short, tau8, tau16, tau32)


def line_items(n_candidates, counts, tau8, tau16, tau32):
    """The plan's section 8.1 itemization, in the plan's own order."""
    tc = t_candidate(counts, tau8, tau16, tau32)
    tv = t_table(counts.v_long, counts.v_short, tau8, tau16, tau32)
    th = t_table(counts.h_long, counts.h_short, tau8, tau16, tau32)
    items = [
        ("S candidates", n_candidates, n_candidates * tc),
        ("S controls", counts.n_controls, counts.n_controls * tc),
        ("V tables", counts.n_v_tables, counts.n_v_tables * tv),
        ("H tables", counts.n_h_tables, counts.n_h_tables * th),
        ("native short reference", counts.native_short, OVERHEAD * counts.native_short * tau8 / 60.0),
        ("gain probes (new low-g cells)", counts.gain_probes,
         counts.gain_probes * t_table(counts.gain_probe_long, counts.gain_probe_short, tau8, tau16, tau32)),
        ("operator parity / overhead reserve", None, counts.parity_reserve),
        ("model preflight + pilot", None, counts.pilot_reserve),
        ("exception reserve", None, counts.exception_reserve),
    ]
    return items, tc, tv, th


def evaluate(n_candidates, counts, tau8, tau16, tau32):
    items, tc, tv, th = line_items(n_candidates, counts, tau8, tau16, tau32)
    total = sum(v for _, _, v in items)
    return {
        "n_candidates": n_candidates,
        "taus": {"t8": tau8, "t16": tau16, "t32": tau32},
        "minutes_per_s_candidate": tc,
        "minutes_per_v_table": tv,
        "minutes_per_h_table": th,
        "items": [{"item": k, "count": c, "minutes": v} for k, c, v in items],
        "total_minutes": total,
        "budget_minutes": BUDGET_MIN,
        "fits": bool(total <= BUDGET_MIN),
        "slack_minutes": BUDGET_MIN - total,
    }


def fit_candidates(counts, tau8, tau16, tau32, s_minutes=None, budget=BUDGET_MIN, floor=40):
    """Largest candidate count that fits, per plan section 8.1.

    The plan's sensitivity rows are computed by "scaling proportionally with the
    same per-length time structure" -- not by setting tau8 = tau16 = tau32.  So
    the knob is the per-candidate S cost itself: every time-based line item is
    scaled by k = s_minutes / T_S(baseline taus), while the two wall-clock
    reserves (pilot and exception) stay fixed, because they are not per-row work.

    This reproduces the plan's own counts exactly: 60 / 49 / 41 / 35 at
    6.992 / 8 / 9 / 10 minutes per candidate.

    Section 8: if the full 60 do not fit, a fixed queue PREFIX of at least 40
    candidates (every direction's a and b) may be locked before the first
    candidate row, chosen on timing only -- never on score.  This function only
    reads timings and counts.
    """
    tc0 = t_candidate(counts, tau8, tau16, tau32)
    tvc0 = t_table(counts.v_long, counts.v_short, tau8, tau16, tau32)
    thc0 = t_table(counts.h_long, counts.h_short, tau8, tau16, tau32)
    if s_minutes is None:
        s_minutes = tc0
    k = s_minutes / tc0

    fixed = counts.pilot_reserve + counts.exception_reserve
    per_candidate = k * tc0
    base_variable = k * (counts.n_controls * tc0 + counts.n_v_tables * tvc0
                         + counts.n_h_tables * thc0
                         + OVERHEAD * counts.native_short * tau8 / 60.0
                         + counts.gain_probes * t_table(counts.gain_probe_long,
                                                        counts.gain_probe_short,
                                                        tau8, tau16, tau32)
                         + counts.parity_reserve)

    n = max(0, int(math.floor((budget - fixed - base_variable) / per_candidate)))
    return {
        "max_candidates": n,
        "meets_minimum_40": bool(n >= floor),
        "minimum_required": floor,
        "scale_k": k,
        "minutes_per_candidate": s_minutes,
        "baseline_minutes_per_candidate": tc0,
        "fixed_reserve_minutes": fixed,
        "variable_base_minutes": base_variable,
        "total_minutes": fixed + base_variable + n * per_candidate,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t8", type=float, required=True)
    ap.add_argument("--t16", type=float, required=True)
    ap.add_argument("--t32", type=float, required=True)
    ap.add_argument("--n-candidates", type=int, default=60)
    ap.add_argument("--s-long", type=int, default=96)
    ap.add_argument("--s-short", type=int, default=16)
    ap.add_argument("--v-long", type=int, default=128)
    ap.add_argument("--v-short", type=int, default=32)
    ap.add_argument("--h-long", type=int, default=256)
    ap.add_argument("--h-short", type=int, default=64)
    ap.add_argument("--fit-candidates", action="store_true",
                    help="solve for the largest queue prefix that fits (timing only)")
    ap.add_argument("--s-minutes", type=float, default=None,
                    help="assumed minutes per S candidate when fitting; defaults to the "
                         "baseline computed from --t8/--t16/--t32")
    ap.add_argument("--explain", action="store_true",
                    help="also print the plan's section 8.1 sensitivity rows for comparison")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    counts = Counts(s_long=a.s_long, s_short=a.s_short, v_long=a.v_long, v_short=a.v_short,
                    h_long=a.h_long, h_short=a.h_short)

    if a.fit_candidates:
        res = fit_candidates(counts, a.t8, a.t16, a.t32, s_minutes=a.s_minutes)
        res["taus"] = {"t8": a.t8, "t16": a.t16, "t32": a.t32}
        res["counts"] = asdict(counts)
        res["basis"] = ("timing only; the selection never reads candidate scores, "
                        "per plan section 8.1")
        res["plan_section_8_1_reference"] = PLAN_8_1
        if not res["meets_minimum_40"]:
            res["verdict"] = "BUDGET_INCOMPATIBLE"
            res["note"] = ("fewer than 40 candidates fit; plan section 8.1 requires "
                           "reporting BUDGET_INCOMPATIBLE rather than trimming H or S")
        else:
            res["verdict"] = f"lock a {res['max_candidates']}-candidate prefix before the first row"
            res["unrun_c_policies"] = 60 - res["max_candidates"]
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res[k] for k in ("verdict", "max_candidates", "meets_minimum_40")}, indent=2))
        return 0

    res = evaluate(a.n_candidates, counts, a.t8, a.t16, a.t32)
    res["counts"] = asdict(counts)
    if a.explain:
        res["plan_section_8_1_sensitivity"] = PLAN_8_1
        res["discrepancy_note"] = (
            "candidate counts match the plan exactly (60/49/41/35); the plan's reported "
            "totals for the scaled rows are ~0.2-0.5% below the itemized model above"
        )
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"per-candidate S: {res['minutes_per_s_candidate']:.3f} min")
    for it in res["items"]:
        cnt = "" if it["count"] is None else f"x{it['count']:<4d}"
        print(f"  {it['item']:34s} {cnt:6s} {it['minutes']:9.3f}")
    print(f"  {'TOTAL':34s} {'':6s} {res['total_minutes']:9.3f}  "
          f"({'fits' if res['fits'] else 'OVER'}, slack {res['slack_minutes']:+.3f})")
    return 0


# plan section 8.1's own sensitivity rows, verbatim, for comparison only
PLAN_8_1 = [
    {"minutes_per_s_candidate": 6.992, "max_candidates": 60, "total_minutes": 599.880,
     "meets_a_b_floor": True},
    {"minutes_per_s_candidate": 8.000, "max_candidates": 49, "total_minutes": 594.377,
     "meets_a_b_floor": True},
    {"minutes_per_s_candidate": 9.000, "max_candidates": 41, "total_minutes": 593.219,
     "meets_a_b_floor": True},
    {"minutes_per_s_candidate": 10.000, "max_candidates": 35, "total_minutes": 596.061,
     "meets_a_b_floor": False},
]


if __name__ == "__main__":
    sys.exit(main())
