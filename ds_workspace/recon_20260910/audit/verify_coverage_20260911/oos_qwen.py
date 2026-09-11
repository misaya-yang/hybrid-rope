"""Out-of-sample test: does the coverage theory order the QWEN methods?

WHY THIS IS A REAL TEST.  The theory was calibrated on OLMo (theta=5e5, W=4096)
-- its two constants (delta, L) and its whole n_int / rho=+0.926 story come from
that family.  `ground_truth_tables.json` holds 28 QWEN2.5-3B methods with
measured RULER accuracy, a different model, a different tokenizer, a different
theta (1e6) and a different window (32768).  Nothing about them was used to
build the theory.

So: recompute the theory's own metric with QWEN's constants and ask whether it
still orders the methods.  This is the cheapest available check on whether
"the binding constraint is coverage" generalises across models -- which is
exactly the question the campaign has been unable to answer (goal item 3).

WHAT WOULD COUNT AS WHAT
  rho >= 0.7   -> the coverage order transfers across models.  Strengthens P1.
  rho ~ 0      -> it does NOT transfer; the theory is an OLMo-family account and
                  should not be cited as a cross-model law.
  rho < 0      -> the metric is actively wrong on Qwen; P1's prescription loses
                  its main support.

UNITS: panel_scores.score_128K_pct is in PERCENT (the theory's own numbers are
fractions; only the rank matters for rho, so units cancel).  128K is 4x W for
Qwen, i.e. u in [0,1] -- the zone the theory is about.  The 32K scores are
window-interior and are reported separately, not pooled.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# Self-contained on purpose: this test must not import the theory's module, so
# that a bug there cannot be inherited here.  These are the axioms of §1.
LN4 = math.log(4.0)


def my_turns(theta, window, k=64):
    """Turns completed by slot j within the training window."""
    return (window / (2.0 * math.pi)) * theta ** (-np.arange(k) / float(k))


def my_kappa(turns, delta):
    return np.log(turns / delta) / LN4


def my_n_int(m, theta, window, delta=0.25, reach_cap=2.0, n_grid=161):
    turns = my_turns(theta, window)
    kappa = my_kappa(turns, delta)
    alive = turns >= delta
    reach = np.minimum(kappa, reach_cap)
    us = np.linspace(0.0, 1.6, n_grid)
    zone = (us >= 0.02) & (us <= 0.98)
    vals = []
    for u in us[zone]:
        vals.append(int(np.count_nonzero(alive & (m >= u - 1e-12)
                                         & ((m - reach) <= u + 1e-12))))
    return float(np.mean(vals))

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", ".."))
GT = os.path.join(REPO, "analysis", "unify_20260910", "tables",
                  "ground_truth_tables.json")
QWEN = dict(theta=1.0e6, window=32768)      # Qwen2.5-3B-Instruct, per the file's meta


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    d = json.load(open(GT))
    M = d["methods"]
    theta, W = QWEN["theta"], QWEN["window"]
    turns = my_turns(theta, W)
    kappa = my_kappa(turns, 0.25)
    alive = turns >= 0.25
    print(f"Qwen constants: theta={theta:.0e} W={W}  alive slots="
          f"{int(alive.sum())} of 64  (OLMo had {int((my_turns(5e5,4096)>=0.25).sum())})")

    rows = []
    for name, v in M.items():
        ps = v.get("panel_scores")
        if not ps or ps.get("score_128K_pct") is None:
            continue
        m = np.asarray(v["m_j"], float)
        if m.size != 64:
            continue
        rows.append((name, my_n_int(m, theta, W), float(ps["score_128K_pct"]),
                     ps.get("rows"), ps.get("wins"), ps.get("losses")))

    if len(rows) < 5:
        print("not enough scored methods:", len(rows))
        return 2
    rows.sort(key=lambda r: r[1])
    print(f"\n{len(rows)} Qwen methods with a measured 128K score")
    print(f"{'method':26s} {'n_int':>7s} {'128K%':>8s} {'rows':>5s}  W/L")
    for n_, ni, sc, nr, w, l in rows:
        print(f"{n_:26s} {ni:7.2f} {sc:8.2f} {str(nr):>5s}  {w}/{l}")

    x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    rho = spearman(x, y)
    print(f"\n  spearman(n_int_qwen, 128K accuracy) = {rho:+.3f}   (n={len(rows)})")
    print(f"  the theory's own OLMo figure was   = +0.926")

    # control: the same statistic on 32K (window-interior) -- the theory does NOT
    # claim to explain the interior, so a weaker rho here is expected, not a failure
    rows32 = [r for r in rows if r[0] in {q[0] for q in rows}]
    y32 = np.array([float(M[r[0]]["panel_scores"].get("score_32K_pct", np.nan))
                    for r in rows32])
    ok = ~np.isnan(y32)
    if ok.sum() >= 5:
        rho32 = spearman(np.array([r[1] for r in rows32])[ok], y32[ok])
        print(f"  control, same methods at 32K (interior)  = {rho32:+.3f}")

    print("\n  VERDICT:", end=" ")
    if rho >= 0.7:
        print("the coverage order TRANSFERS to Qwen. P1's basis is cross-model.")
    elif rho >= 0.3:
        print("weak transfer.  Treat the theory as an OLMo-family account that")
        print("  partially extends, not a cross-model law.")
    elif rho > -0.3:
        print("NO transfer.  n_int does not order Qwen.  Do NOT cite the coverage")
        print("  story as the cross-model explanation; P1 remains an OLMo test.")
    else:
        print("ANTI-correlated on Qwen -- the metric is wrong there, and P1's")
        print("  prescription loses its main support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
