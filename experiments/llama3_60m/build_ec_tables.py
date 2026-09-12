"""Build the six tables of Plan B section 5 / E-C (scale x gain 2x2 + 2 controls).

Plan B states the cell definitions and the identification of the old arm:

    C00  omega*4^-r    g4     old / mismatched anchor
    C10  omega*8^-r    g4     scale changed alone
    C01  omega*4^-r    g8     gain changed alone
    C11  omega*8^-r    g8     the correctly matched 8x deployment
    CY8  official YaRN, s=8, g8
    CM8  MrRoPE-Pro,     s=8, g8

    "原 scale8x_wide=1.5×wideBM 在 log4 坐标下应对应 C10 或 C11，取决于真实
     gain。先核对，不给同一数组换两个名字。"

CHECKED, not assumed: `scale8x_wide`'s stored m equals 1.5 * r_wideBM exactly, and
`omega*8^-r` reproduces it to 1.1e-16 while `omega*4^-1.5r` reproduces it exactly
(the two are the same by section 2.2's identity `4^-1.5r = 8^-r`).  It has always
been run at g4, so **`scale8x_wide` IS C10**, the mismatched-anchor cell -- not
C11.  That is a candidate explanation for the 8x arms scoring 0/48: the correct
8x deployment (C11, gain g8) was never among them.

Only FOUR distinct m-arrays exist; the six cells are (array, gain) pairs.

OLMo geometry: theta=5e5, W=4096, band (14,32), n=18; the profile r is the
normalised wideBM shape (band [11,32], n=21), held FIXED across all cells.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

THETA, W, K = 500000.0, 4096, 64
WIDE_LOW, WIDE_N = 11, 21
MR_LOW, MR_N = 14, 18


def omega():
    return THETA ** (-np.arange(K) / K)


def r_widebm():
    """The normalised wideBM profile: eps ~ i(n+1-i) over band [11, 32]."""
    k = np.arange(1, WIDE_N + 1, dtype=np.float64)
    w = k * (WIDE_N + 1 - k)
    eps = w / w.sum()
    r = np.zeros(K)
    r[WIDE_LOW:WIDE_LOW + WIDE_N + 1] = np.concatenate([[0.0], np.cumsum(eps)])
    r[WIDE_LOW + WIDE_N + 1:] = 1.0
    return r


def r_mrpro(scale):
    """MrRoPE-Pro on OLMo's band.  The profile is s-INDEPENDENT; only nu moves."""
    q = np.clip(np.arange(K) - MR_LOW, 0, MR_N).astype(np.float64)
    return q * (q + 1.0) / (MR_N * (MR_N + 1.0))


def nu_yarn(scale):
    """Official index YaRN.  Section 2.3: its profile is s-DEPENDENT, so r_YARN(4)
    must not be reused at s=8."""
    q = np.clip(np.arange(K) - MR_LOW, 0, MR_N).astype(np.float64)
    u = q / MR_N
    return omega() * ((1.0 - u) + u / scale)


def gain_for(scale):
    return 1.0 + 0.1 * math.log(scale) if scale > 1 else 1.0


def build():
    om, r = omega(), r_widebm()
    tables = {
        "ec_C00_C01": {"m": r, "nu": om * 4.0 ** (-r),
                       "note": "profile r at s=4; used at g4 (C00) and g8 (C01)"},
        "ec_C10_C11": {"m": 1.5 * r, "nu": om * 8.0 ** (-r),
                       "note": "same profile at s=8; used at g4 (C10) and g8 (C11). "
                               "This array IS the historical scale8x_wide."},
        "ec_CY8": {"m": -np.log(nu_yarn(8) / om) / math.log(8), "nu": nu_yarn(8),
                   "note": "official index YaRN at s=8"},
        "ec_CM8": {"m": r_mrpro(8), "nu": om * 8.0 ** (-r_mrpro(8)),
                   "note": "MrRoPE-Pro at s=8"},
    }
    cells = [
        ("C00", "ec_C00_C01", 4.0), ("C10", "ec_C10_C11", 4.0),
        ("C01", "ec_C00_C01", 8.0), ("C11", "ec_C10_C11", 8.0),
        ("CY8", "ec_CY8", 8.0), ("CM8", "ec_CM8", 8.0),
    ]
    for name, tbl, s in cells:
        tables[tbl].setdefault("cells", []).append({"cell": name, "gain": gain_for(s)})
    return tables, cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, cells = build()
    om = omega()

    for name, t in tables.items():
        m = np.asarray(t["m"], dtype=np.float64)
        # the runner requires nu strictly decreasing
        if not np.all(np.diff(t["nu"]) < 0):
            raise SystemExit(f"REFUSING: {name} is not strictly decreasing in nu")
        (out / f"{name}.json").write_text(json.dumps(
            {"arm": name, "sum_m": float(m.sum()), "max_m": float(m.max()),
             "m": m.tolist(), "note": t["note"]}, indent=1))

    # sanity: the identity section 2.2 is built on
    r = r_widebm()
    ident = float(np.max(np.abs(4.0 ** (-1.5 * r) - 8.0 ** (-r))))
    plan = {"cells": [{"cell": c, "table": t, "gain": g,
                       "nu_sha256": __import__("hashlib").sha256(
                           np.ascontiguousarray(tables[t]["nu"]).tobytes()).hexdigest()[:16]}
                      for c, t, g in cells],
            "identity_4_pow_neg1p5r_eq_8_pow_negr": ident,
            "scale8x_wide_identified_as": "C10 (array omega*8^-r at g4)",
            "note": ("four distinct m-arrays; the six cells are (array, gain) pairs. "
                     "C00/C01 and C10/C11 share arrays and differ only in gain."),
            "strictly_decreasing": True}
    (out / "ec_plan.json").write_text(json.dumps(plan, indent=2))

    print(f"wrote {len(tables)} arrays + 6 cells to {out}")
    print(f"  section 2.2 identity |4^-1.5r - 8^-r| = {ident:.2e}")
    for c, t, s in cells:
        m = np.asarray(tables[t]["m"])
        print(f"  {c:4s} table={t:12s} gain={gain_for(s):.9f}  sum_m={m.sum():7.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
