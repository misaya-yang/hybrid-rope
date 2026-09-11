"""Offline derivation of the amplitude prescription (no GPU).

Companion to prereg_protocols/AMP4X_PRESCRIPTION_PREREG_20260911.md.

QUESTION.  The coverage theory says a schedule's plateau height m_p sets how deep
its slots reach.  The deployed tables all use m_p = 1, which is what "one full
compression by 4" means, and the campaign's dose curve found its optimum pinned
at that end of the family with interior points rejected at |t| >= 5.6.  If the
coverage criterion instead puts the optimum ABOVE 1, then the family cap was the
binding constraint and the optimum lay outside the family -- which would explain
the pinning without any claim that m_p = 1 is best.

METHOD.  Use the theory's own two axioms and nothing else:
  slot j usable at depth u  <=>  m_j - min(kappa_j, L) <= u <= m_j
  kappa_j = log4(t_j(W)/delta),  t_j(W) = W theta^{-j/K} / (2 pi)
  alive  <=>  t_j(W) >= delta
For a ramp-and-plateau table of height m_p, average n(u) over u in [0, log4 S]
and report the m_p that maximises it, for each target S.

READ THE OUTPUT AS A HYPOTHESIS, NOT A RESULT.  This is pure derivation against
a theory whose cross-model transfer has already been falsified (see REPORT.md in
this directory), so the prescription is claimed for the OLMo family only.
"""
from __future__ import annotations

import math

import numpy as np

K = 64


def solve(S, delta=0.25, reach_cap=2.0, lo=14, hi=32, theta=5e5, W=4096,
          grid=0.025, span=(0.5, 2.01)):
    """Best plateau height for target ratio S, and the coverage it attains."""
    ln4 = math.log(4.0)
    tW = (W / (2 * math.pi)) * theta ** (-np.arange(K) / K)
    kappa = np.log(tW / delta) / ln4
    alive = tW >= delta
    reach = np.minimum(kappa, reach_cap)

    def n_of_u(m, u):
        return int(np.count_nonzero(alive & (m >= u - 1e-12)
                                    & ((m - reach) <= u + 1e-12)))

    def plateau(mp):
        m = np.zeros(K)
        for j in range(K):
            m[j] = 0.0 if j < lo else (mp * (j - lo) / (hi - lo) if j <= hi else mp)
        return m

    umax = math.log(S, 4)
    us = np.linspace(0.0, umax, 161)
    best, bc = None, -1.0
    for mp in np.arange(span[0], span[1], grid):
        c = float(np.mean([n_of_u(plateau(mp), u) for u in us]))
        if c > bc:
            bc, best = c, mp
    c_at = float(np.mean([n_of_u(plateau(umax), u) for u in us]))
    return best, bc, c_at


def main() -> int:
    print("prescription vs target (OLMo: theta=5e5 W=4096 band [14,32] delta=0.25 L=2)")
    print(f"{'S':>4s} {'log4S':>7s} {'best m_p':>9s} {'cov@best':>9s} {'cov@log4S':>10s}")
    for S in (2, 4, 6, 8, 12, 16):
        best, bc, c_at = solve(S)
        print(f"{S:>4d} {math.log(S,4):>7.2f} {best:>9.2f} {bc:>9.2f} {c_at:>10.2f}")

    print("\nsensitivity at S=4 (the deployed regime)")
    print(f"{'delta':>7s} {'L':>5s} {'band':>9s} {'best m_p':>9s}")
    for d in (0.125, 0.25, 0.5):
        for L in (1.0, 2.0, 5.0):
            print(f"{d:>7.3f} {L:>5.1f} {'[14,32]':>9s} {solve(4, delta=d, reach_cap=L)[0]:>9.2f}")
    for lo, hi, th, Wd in ((11, 32, 5e5, 4096), (14, 40, 1e6, 32768), (23, 40, 1e6, 32768)):
        print(f"{0.25:>7.3f} {2.0:>5.1f} {f'[{lo},{hi}]':>9s} "
              f"{solve(4, lo=lo, hi=hi, theta=th, W=Wd)[0]:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
