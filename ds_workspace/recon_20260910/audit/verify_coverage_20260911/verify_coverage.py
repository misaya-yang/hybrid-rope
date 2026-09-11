"""Independent verification of the coverage/ceiling theory.

WHY.  `theory/COVERAGE_CEILING_THEORY_20260911.md` now claims to explain the whole
campaign with two constants, and its prescription (`scale8x_wide`) is what the next
GPU run would test.  Before spending card time on a prescription, reproduce the
numbers the theory quotes for itself.

METHOD (independence).  Do NOT import their coverage function.  Re-derive it from
the two axioms as stated in the theory §1 and implement it here; then compare
against the n_int values the theory publishes.  The m arrays themselves are imported
from their script on purpose -- the tables are the objects under test, not the
theory, and re-typing 15 constructors would only invite transcription error.

AXIOMS (theory §1, verbatim):
  A: slot j only knows phases with turns <= t_j(W);   t_j(W) = (W/2pi) * theta^(-j/K)
  B: a slot has resolution at distance d only if turns >= delta
  =>  slot j is usable at depth u = log4(d/W)  <=>  m_j - min(kappa_j, L) <= u <= m_j
      kappa_j = log4(t_j(W)/delta)
  n(u) = number of usable alive slots at u;  n_int = mean of n(u) over u in [0.02,0.98]

VERDICTS this script can return:
  REPRODUCED   -- my n_int matches theirs to <0.01 on every quoted arm
  MISMATCH     -- some quoted n_int differs; the theory's numbers are not reproducible
  INSENSITIVE  -- reproduces AND the ranking is stable across delta in [.125,1], L in [1,10]
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "code")
import coverage_theory_20260911 as T  # noqa: E402  (tables + quoted arms only)

K = 64
LN4 = math.log(4.0)


def my_turns(theta: float, window: int, k: int = K) -> np.ndarray:
    """Turns completed by slot j within the training window."""
    return (window / (2.0 * math.pi)) * theta ** (-np.arange(k) / float(k))


def my_kappa(turns: np.ndarray, delta: float) -> np.ndarray:
    return np.log(turns / delta) / LN4


def my_coverage(m: np.ndarray, u: float, kappa: np.ndarray, alive: np.ndarray,
                reach_cap: float) -> int:
    """# alive slots whose window [m_j - min(kappa_j, L), m_j] contains u."""
    reach = np.minimum(kappa, reach_cap)
    ceiling_ok = m >= u - 1e-12          # the slot's ceiling reaches depth u
    floor_ok = (m - reach) <= u + 1e-12  # not over-compressed past resolution
    return int(np.count_nonzero(alive & ceiling_ok & floor_ok))


def my_n_int(m: np.ndarray, theta: float, window: int, delta: float = 0.25,
             reach_cap: float = 2.0, n_grid: int = 161) -> float:
    turns = my_turns(theta, window)
    kappa = my_kappa(turns, delta)
    alive = turns >= delta
    us = np.linspace(0.0, 1.6, n_grid)
    zone = (us >= 0.02) & (us <= 0.98)
    return float(np.mean([my_coverage(m, u, kappa, alive, reach_cap) for u in us[zone]]))


def main() -> int:
    theta, window = T.THETA, T.W

    # --- step 1: reproduce the n_int values the theory publishes -------------
    # The theory's §3 row labelled "wide BM" is the wide arm at the SAME b as BM
    # (b=1, low=11) -- i.e. turns_a1_b64, not b4_wide.  Getting this mapping wrong
    # once already produced a spurious 4.78 mismatch; the label is right, my key was not.
    quoted = {                       # arm name -> n_int as printed in the theory
        "MrRoPE": 9.08,              # convex  eps ~ k
        "BM": 11.99,                 # deployed (parabolic eps)
        "turns_a1_b64": 13.48,       # theory's "wide BM" row (also quotes acc 0.5384)
    }
    # The theory's §3 table also quotes YaRN(0,0)=11.99, but there is no YaRN arm
    # in ARMS (it says "未测" -- never run), so it cannot be checked here.
    print("== step 1: reproduce quoted n_int (theta=5e5, W=4096, delta=0.25, L=2) ==")
    print(f"{'arm':12s} {'mine':>8s} {'quoted':>8s} {'diff':>8s}")
    ok = True
    for name, q in quoted.items():
        m = np.asarray(T.ARMS[name][0], float)
        mine = my_n_int(m, theta, window)
        d = abs(mine - q)
        ok &= d < 0.01
        print(f"{name:12s} {mine:8.2f} {q:8.2f} {d:8.3f}")
    print("  ->", "REPRODUCED" if ok else "**MISMATCH**")

    # --- step 2: does the theory's own rho reproduce? ------------------------
    print("\n== step 2: rank correlation n_int vs measured RULER, on their 15 arms ==")
    rows = []
    for name, (m, ruler, _cont) in T.ARMS.items():
        if ruler is None:
            continue
        rows.append((name, my_n_int(np.asarray(m, float), theta, window), float(ruler)))
    rows.sort(key=lambda r: r[1])
    if len(rows) >= 4:
        x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
        rho = float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
        print(f"  n arms with RULER = {len(rows)},  spearman(n_int, acc) = {rho:+.3f}")
        print(f"  theory quotes rho = +0.926 on this family")
        for n_, ni, ac in rows:
            print(f"    {n_:16s} n_int {ni:6.2f}   acc {ac:.4f}")

    # --- step 3: sensitivity -- is the RANKING robust to (delta, L)? ---------
    print("\n== step 3: sensitivity of the ranking to delta and L ==")
    worst = 0.0
    for dlt in (0.125, 0.25, 0.5, 1.0):
        for L in (1.0, 2.0, 5.0, 10.0):
            vals = [(my_n_int(np.asarray(m, float), theta, window, delta=dlt,
                              reach_cap=L), float(r)) for _n, (m, r, _c) in T.ARMS.items()
                    if r is not None]
            x = np.array([v for v, _ in vals]); y = np.array([r for _, r in vals])
            rho = float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
            worst = max(worst, abs(rho - 0.926))
    print(f"  max |rho - 0.926| over delta in [.125,1] x L in [1,10] = {worst:.3f}")
    print(f"  theory claims rho varies < 0.02 over this grid")

    verdict = ("REPRODUCED" if ok else "MISMATCH")
    print(f"\nVERDICT: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
