#!/usr/bin/env python3
"""WHY do the winning tables win?  Test candidate explanations against the scores.

THE SITUATION THIS FILE EXISTS TO FIX.  Nine OLMo measurements now exist, three
of them winners by very different manipulations, and the campaign has been
accumulating them without forcing any theory to explain them.  `sum(m)` happens
to correlate at r=0.96, but it is not sufficient (two arms at sum(m) 38.47 and
38.50 differ by 12 points), and `sum(m)` is not a mechanism -- it is a number
about the table that says nothing about what the model does.  This file replaces
it with candidates that DO name a mechanism, computed from the table alone so
they cost no GPU, and scores them against the measurements.

THE CANDIDATES, each with the mechanism it names.  All are computed at the TEST
distance D, since the score is a 16K score on a checkpoint trained at 4K.

  A  sum(m)                        budget spent. No mechanism; the incumbent.
  B  n_ramp                        number of slots strictly between the
                                   plateaus -- "how wide the transition is".
  C  n_held                        slots still at m=0. The three-band family's
                                   defining choice; EVQ sets it to 0.
  D  turn(D) profile: how many slots complete between a and b turns at distance
     D.  A slot with turn count >> 1 aliases (its phase wraps many times, so the
     relative-distance signal is destroyed); a slot with turn count << 1 barely
     varies over the context.  The count in the useful band is the most direct
     "how much usable positional signal exists at this distance" quantity, and
     it is what a fixed turn window [1,32] is implicitly trying to control.
  E  reach coverage: number of slots whose half-period 2*pi/nu exceeds D, i.e.
     slots that can still tell positions apart at distance D (Nyquist).
  F  gap regularity: max/min of the compressed log-frequency gaps inside the
     band -- the "no spectral hole / no pile-up" reading, which is what the
     deployed BM was built to optimize (the Dirichlet/parity objective).
  G  spectral hole: the LARGEST compressed log-gap anywhere in the spectrum,
     relative to the native gap. A big hole means a band of relative distances
     with no slot resolving it.

HOW TO READ THE RESULT.  These nine points are not independent (four of them are
the same family), so a high correlation is weak evidence and a low one is strong
evidence AGAINST a candidate.  The honest output is a ranking with the count of
candidates tried stated up front, and a refusal to crown anything that does not
separate the matched pairs -- the pairs where two arms agree on the candidate and
disagree on the score are the ones that kill it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
# The test distance the RULER panel uses on this checkpoint.
D_TEST = 16384.0

# Every OLMo measurement to date. Entered from the run receipts.
MEASURED = [
    ("MrPro b=0 (arch)", 0.0709),
    ("beta_b0p25", 0.1447),
    ("turns_a1_b16", 0.2651),
    ("turns_a0p5_b32", 0.1604),
    ("beta_b0p5", 0.2320),
    ("MrProBM b=1 (arch)", 0.4167),
    ("turns_a2_b32", 0.5100),
    ("turns_a1_b64", 0.5384),
    ("beta_b2", 0.5001),
]


def table_of(name):
    """Rebuild each measured arm's m-vector in OLMo's own geometry."""
    from experiments.curvature_20260910.tables import (m_incr_beta, m_mrpro,
                                                       m_turns)
    lo, n = OLMO["low"], OLMO["n"]
    if name.startswith("turns_"):
        spec = name[len("turns_"):]
        a_s, b_s = spec.split("_b")
        a = float(a_s.replace("a", "").replace("p", "."))
        b = float(b_s.replace("p", "."))
        return np.asarray(m_turns(a, b, OLMO["theta"], OLMO["window"],
                                  OLMO["head_dim"], ramp="beta1"), dtype=float)
    if name.startswith("beta_b"):
        return np.asarray(m_incr_beta(float(name[len("beta_b"):].replace("p", ".")),
                                      n=n, low=lo), dtype=float)
    if name.startswith("MrPro b=0"):
        return np.asarray(m_mrpro(n=n, low=lo), dtype=float)
    if name.startswith("MrProBM"):
        return np.asarray(m_incr_beta(1.0, n=n, low=lo), dtype=float)
    raise KeyError(name)


def features(m, D=D_TEST):
    """The candidate explanations, all computed from the table alone."""
    th = OLMO["theta"]
    K = OLMO["K"]
    nu = th ** (-np.arange(K, dtype=float) / K) * np.power(4.0, -m)
    turns = nu * D / (2.0 * math.pi)          # turns completed at the test distance
    lg = np.log(nu)
    gap = -np.diff(lg)                        # compressed log-frequency gaps
    native_gap = math.log(th) / K

    held = int(np.sum(m < 1e-12))
    plateau = int(np.sum(m > 1 - 1e-12))
    ramp = K - held - plateau

    def count_turns(a, b):
        return int(np.sum((turns >= a) & (turns <= b)))

    # ---- the aliasing family.  THE PHYSICAL STORY THESE TEST: at 4x the trained
    # window every slot in the transition region completes far more than one turn
    # at the test distance, so its phase there is effectively arbitrary.  If that
    # is what costs the score, every winner should lower the total phase wrap at
    # D, and three different manipulations should all help for that one reason.
    # sum(nu) is the total wrap per unit distance and is deliberately NOT a
    # function of sum(m) alone: it is a sum of exponentials, dominated by the
    # LEAST compressed slots, so it weights the fast end far more heavily.
    wrap = float(nu.sum())
    turns_band = turns[np.asarray(m) > 1e-12]
    return dict(
        A_sum_m=float(m.sum()),
        H_wrap=wrap,
        H_log_wrap=float(np.log(wrap)),
        I_band_mean_turns=float(turns_band.mean()),
        J_band_median_turns=float(np.median(turns_band)),
        K_band_max_turns=float(turns_band.max()),
        L_ramp_integral=float(np.asarray(m)[np.asarray(m) > 1e-12].sum()
                              - np.sum(np.asarray(m) > 1 - 1e-12)),
        M_mid_crossing=float(np.flatnonzero(np.asarray(m) >= 0.5)[0]
                             if (np.asarray(m) >= 0.5).any() else -1),
        B_n_ramp=ramp,
        C_n_held=held,
        D_turns_0p25_4=count_turns(0.25, 4.0),
        D_turns_0p5_2=count_turns(0.5, 2.0),
        D_turns_0p125_8=count_turns(0.125, 8.0),
        E_reach=int(np.sum(nu < math.pi / D)),
        F_gap_ratio=float(gap.max() / gap.min()),
        G_max_hole=float(gap.max() / native_gap),
    )


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    names = [n for n, _ in MEASURED]
    acc = np.array([a for _, a in MEASURED])
    F = []
    for n in names:
        try:
            m = table_of(n)
        except Exception as exc:                       # pragma: no cover
            print(f"skip {n}: {exc}", file=sys.stderr)
            continue
        F.append(features(m, D_TEST))
    if not F:
        return 1
    keys = sorted(F[0])

    print(f"=== {len(F)} OLMo measurements, {len(keys)} candidate explanations ===")
    print(f"test distance D = {D_TEST:g}  (16K RULER panel)\n")
    print(f"{'feature':18s} {'Spearman':>9s} {'Pearson':>9s}")
    rows = []
    for k in keys:
        v = np.array([f[k] for f in F], dtype=float)
        sp = spearman(v, acc)
        pe = (float(np.corrcoef(v, acc)[0, 1]) if v.std() > 0 else float("nan"))
        rows.append((k, sp, pe, v))
    for k, sp, pe, _ in sorted(rows, key=lambda r: -abs(r[1]) if np.isfinite(r[1]) else 1):
        print(f"{k:18s} {sp:+9.3f} {pe:+9.3f}")

    print("\n=== the KILLER TEST: matched pairs ===")
    print("Two arms that agree on a feature but differ on the score kill it.")
    print("Two arms that differ on a feature but agree on the score also kill it.\n")
    by = {n: (f, a) for n, f, a in zip(names, F, acc)}
    pairs = [("turns_a2_b32", "turns_a1_b64"), ("turns_a1_b64", "beta_b2"),
             ("beta_b0p25", "turns_a1_b16"), ("beta_b0p25", "turns_a0p5_b32"),
             ("beta_b0p5", "turns_a1_b16")]
    votes = {k: 0 for k in keys}
    total = {k: 0 for k in keys}
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        fa, aa = by[a]
        fb, ab = by[b]
        dacc = abs(aa - ab)
        for k in keys:
            va, vb = fa[k], fb[k]
            scale = max(abs(va), abs(vb), 1e-12)
            same_feat = abs(va - vb) / scale < 0.05
            same_acc = dacc < 0.05
            total[k] += 1
            if same_feat and not same_acc:
                votes[k] -= 1          # feature blind to a real difference
            elif not same_feat and same_acc:
                votes[k] -= 1          # feature over-predicts a difference
            else:
                votes[k] += 1          # consistent
    for a, b in pairs:
        if a in by and b in by:
            print(f"  {a:16s} {by[a][1]:.4f}   vs   {b:16s} {by[b][1]:.4f}"
                  f"   d={abs(by[a][1]-by[b][1]):.4f}")
    print(f"\n{'feature':18s} {'net votes':>10s}  (of {max(total.values())} pairs)")
    for k, _, _, _ in sorted(rows, key=lambda r: -votes[r[0]]):
        print(f"{k:18s} {votes[k]:10d}")

    out = dict(n_points=len(F), D=D_TEST,
               correlations=[dict(feature=k, spearman=sp, pearson=pe)
                             for k, sp, pe, _ in rows],
               pair_votes={k: votes[k] for k in keys},
               caveat=("nine points, four of them the same family; a high "
                       "correlation is weak evidence, a low one is strong "
                       "evidence against. The matched pairs are the test."))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
