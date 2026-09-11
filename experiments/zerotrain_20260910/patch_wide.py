#!/usr/bin/env python3
"""Add --wide-betas to olmo_beta.py so the WIDE band can go to the RULER panel.

WHY.  The continuous instrument's leaderboard now reads

    1. b4_wide   S=46.68  2.8216
    2. b3_wide   S=45.64  2.8244
    3. b3_lo14   S=43.64  2.8267
    4. ctl_C42V24 S=42.00 2.8309
    5. turns_a1_b64 S=42.00 2.8402
    8. beta_b1_BM S=40.50 2.8627

-- a plateau of five statistically tied arms, all better than the deployed
table.  `turns_a1_b64` is the wide band with the b=1 ramp and is already
RULER-confirmed (+12.17pp, t=+5.7).  `b4_wide` is the nominal continuous leader
and its nominal per-step gain is inside the noise, so it may well be a
continuation of the SAME win rather than a new one -- which is exactly the
question the RULER panel has to answer, and it cannot be answered by the
instrument that produced the tie.

`m_incr_beta(1.0, n=21, low=11)` is bit-for-bit `m_turns(1.0, 64.0, ramp=beta1)`
(verified maxdiff 0.0e+00), so this flag builds the wide band in the same
geometry the RULER run already accepts.
"""
from __future__ import annotations

import pathlib
import sys

P = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")

FLAG_OLD = '    ap.add_argument("--betas", default="")'
FLAG_NEW = (
    '    ap.add_argument("--betas", default="")\n'
    '    ap.add_argument("--wide-betas", dest="wide_betas", default="",\n'
    '                    help="eps ~ k(n+1-k)^b on the WIDE band [11,32] (n=21). "\n'
    '                         "b=1 is bit-for-bit turns_a1_b64.")'
)

BLOCK_OLD = "    if args.arms.strip():"
BLOCK_NEW = '''    if args.wide_betas.strip():
        # THE WIDE BAND.  m_incr_beta(1.0, n=21, low=11) equals
        # m_turns(1.0, 64.0, ramp="beta1") bit-for-bit (maxdiff 0.0e+00), i.e.
        # the RULER-confirmed turns_a1_b64 -- so this axis is that winner's band
        # with the ramp shape varied, which is the walk the continuous
        # instrument scored 2.8449 / 2.8244 / 2.8216 for b = 2 / 3 / 4.
        for bs in args.wide_betas.split(","):
            b = float(bs)
            out.append(run_arm(f"wide_b{bs}".replace(".", "p"),
                               m_incr_beta(b, n=21, low=11)))
    if args.arms.strip():'''


def main():
    s = P.read_text()
    if "--wide-betas" in s:
        print("already patched")
        return 0
    for old in (FLAG_OLD, BLOCK_OLD):
        if old not in s:
            print(f"REFUSING: anchor not found: {old!r}", file=sys.stderr)
            return 2
    s = s.replace(FLAG_OLD, FLAG_NEW, 1).replace(BLOCK_OLD, BLOCK_NEW, 1)
    P.write_text(s)
    print("patched", P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
