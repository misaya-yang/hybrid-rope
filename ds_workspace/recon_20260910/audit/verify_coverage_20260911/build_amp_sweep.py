"""Amplitude sweep at 8x: the arm that discriminates two theories that disagree.

WHY THIS EXISTS.  Two lines of work make OPPOSITE predictions about where the
optimal compression amplitude sits, on the same family, at the same length:

  (A) coverage/ceiling theory  -- amplitude should EQUAL log4(target/W).  For the
      s8 panel (32768) that is 1.5 exactly: below it you leave depth uncovered,
      above it you over-compress the window-interior and pay for nothing.
      => predicts an INTERIOR optimum at s=1.5.

  (B) this campaign's dose result -- the MrRoPE<->BM dose curve is monotone with
      its optimum pinned at the maximum-compression END at every length tested
      (a*(L) == 1, |t| >= 5.6 at 1x/2x/4x).  Generalised, that says: push to the
      end of the tested range, the optimum is not interior.
      => predicts flat-or-rising in s, i.e. s=2.0 no worse than s=1.5.

Both cannot be right.  A single sweep separates them, and it is cheap: same panel,
same rows, only the scalar changes.

NOTE ON ARM COUNT.  s=1.0 is `turns_a1_b64` -- ALREADY measured on this panel at
32768 (0/48).  It is a free anchor, not a new arm.  So the sweep costs 4 new arms,
not 5.

WHAT THIS SCRIPT DOES NOT DO.  It computes the theory-side predictions only (pure
numpy).  It does not run the model; it writes the table JSONs the server runner
consumes, so the run is ready the moment a card is available.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
sys.path.insert(0, os.path.dirname(__file__))

import coverage_theory_20260911 as T  # noqa: E402
from verify_coverage import my_coverage, my_kappa, my_n_int, my_turns  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "code", "tables")
LN4 = math.log(4.0)
SCALES = (1.00, 1.25, 1.50, 1.75, 2.00)


def spectrum_gap_max(m: np.ndarray, theta: float, window: int) -> float:
    """Max ratio (in native-slot units) between consecutive nu, the theory's
    second-order penalty.  1.0 = perfectly smooth; the theory calls 4.49x bad."""
    turns = my_turns(theta, window)
    g = np.log2(turns)
    lam = math.log(theta) / (64 * math.log(2.0))
    dlog_nu = np.diff(g - 2.0 * m)
    return float(np.max(-dlog_nu) / lam)


def write_arm(name: str, m: np.ndarray, theta: float, window: int) -> str:
    path = os.path.abspath(os.path.join(OUT, f"{name}.json"))
    with open(path, "w") as fh:
        json.dump({"arm": name, "config": {"theta": theta, "window": window, "K": 64},
                   "sum_m": float(m.sum()), "m": [float(x) for x in m]}, fh, indent=1)
    return path


def main() -> int:
    theta, window = T.THETA, T.W
    base = np.asarray(T.ARMS["turns_a1_b64"][0], float)   # wide BM, b=1, low=11
    turns = my_turns(theta, window)
    kappa = my_kappa(turns, 0.25)
    alive = turns >= 0.25

    print(f"{'s':>5s} {'sum_m':>7s} {'n(1.5)':>7s} {'zone[1,1.5]':>12s} "
          f"{'n_int':>7s} {'uw(-2)':>7s} {'gap':>6s}  {'theory says':s}")
    rows = []
    for s in SCALES:
        m = s * base
        n15 = my_coverage(m, 1.5, kappa, alive, 2.0)
        us = np.linspace(1.0, 1.5, 81)
        zone = float(np.mean([my_coverage(m, u, kappa, alive, 2.0) for u in us]))
        nint = my_n_int(m, theta, window)
        uw = my_coverage(m, -2.0, kappa, alive, 2.0)          # window-interior
        gap = spectrum_gap_max(m, theta, window)
        rows.append((s, n15, zone, nint, uw, gap))
        tag = ""
        if abs(s - 1.0) < 1e-9:
            tag = "ANCHOR (already run: 0/48)"
        elif abs(s - 1.5) < 1e-9:
            tag = "** theory's predicted optimum **"
        print(f"{s:5.2f} {m.sum():7.2f} {n15:7d} {zone:12.2f} "
              f"{nint:7.2f} {uw:7d} {gap:6.2f}  {tag}")
        if abs(s - 1.0) > 1e-9:
            write_arm(f"amp8x_s{str(s).replace('.', 'p')}", m, theta, window)

    print("\n-- what each position predicts for THIS family (read the table above) --")
    print("  (A)  theory AXIOM      : amplitude == log4(target/W) == 1.50 exactly.")
    print("  (A') theory's OWN METRIC: zone coverage peaks at s=1.75 (10.21), not 1.50 (9.64);")
    print("       n(1.5) peaks at 1.75/2.00 (9).  So the axiom and the metric DISAGREE.")
    print("  (B)  dose-generalisation: s=2.00 >= s=1.50 (optimum pinned at the end).")
    print()
    print("  !! my first draft of this printout claimed the theory predicts an interior")
    print("     optimum BECAUSE uw(-2) degrades with scale.  The table shows uw(-2) is")
    print("     CONSTANT at 12 for every scale -- that mechanism is not in this metric.")
    print("     The interior penalty, if any, lives outside n(u); it is NOT shown here.")
    print()
    print("  sharp sub-test: s=1.25 has zone coverage 4.62 but n(1.5)=0 -- coverage is")
    print("  non-monotone right at the threshold, so 1.25 vs 1.50 is a step test, not a slope.")
    print("\n  separating signals:")
    print("    score(1.50) vs score(1.75)  ->  axiom (A) vs its own metric (A')")
    print("    score(2.00) vs score(1.50)  ->  (A)/(A') vs dose-generalisation (B)")

    print("\n-- written --")
    for s in SCALES:
        if abs(s - 1.0) > 1e-9:
            print(f"  amp8x_s{str(s).replace('.', 'p')}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
