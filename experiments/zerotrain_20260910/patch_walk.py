#!/usr/bin/env python3
"""Add `--walk` to olmo_beta.py: the dose-response between BM and a1_b64.

RUN ON THE SERVER (edits a server file).

WHY.  The 180-row panel says every plateau member is a TRADEOFF in the
project's own vocabulary (ruler_bench.verdict):

    member      d(4096)   d(16384)     status
    b3_lo14      -5.58      +2.18       TRADEOFF
    turns_a1_b64 -4.11      +7.83       TRADEOFF
    wide_b4      -6.42      +9.61       TRADEOFF

and the two arms `turns_a1_b64` and the deployed BM are THE TWO ENDPOINTS OF ONE
FAMILY:

    BM         = m_incr_beta(1.0, n=18, low=14)   increments on slots [15,32]
    a1_b64     = m_incr_beta(1.0, n=21, low=11)   increments on slots [12,32]

They share the ramp exponent and the m=1 plateau from slot 32; the wide arm
simply reaches three slots further toward the fast end.  So the whole
difference is a one-parameter family and the in-window/out-of-window exchange
rate can be MEASURED instead of argued about:

    m(a) = (1-a) * m_BM + a * m_a1b64,   a = 0 exactly BM, a = 1 exactly a1_b64

a = 0 and a = 1 are already measured on the 180-row panel, so three new arms fix
the interior of the frontier.  This is the KKT trade-off itself -- in-window
loss against extrapolation gain -- as a measured curve rather than a derivation.

The interpolation is linear in m, hence also in the forcing eps = Lm, which the
campaign's frame says is the design variable.  Linearity buys a strong null: if
the response is linear in a then the frontier is a straight line through the
origin and there is NO interior point that gains long range without losing
in-window.  A kink is what a DEVELOPMENT_WIN would look like.

Idempotent: refuses to patch twice.  Verifies the flag is live before returning.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")

FLAG = '''    ap.add_argument("--walk", default="",
                    help="comma-separated a in [0,1]: the dose-response "
                         "m(a) = (1-a)*m_BM + a*m_a1b64. a=0 IS the deployed "
                         "table and a=1 IS turns_a1_b64, both already measured, "
                         "so the interior fixes the in-window/out-of-window "
                         "exchange rate. See S8/WALK prereg")
'''

BODY = '''    if args.walk.strip():
        # THE FRONTIER WALK.  Both endpoints are measured arms (--betas 1.0 and
        # --wide-betas 1.0); this interpolates the increments between them, so
        # every arm here is bracketed by two known points rather than being a
        # fresh guess.  Linear in m = linear in eps, the frame's design variable.
        _bm = m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"])
        _wid = m_incr_beta(1.0, n=21, low=11)
        for _a in args.walk.split(","):
            av = float(_a)
            if not 0.0 <= av <= 1.0:
                print(f"REFUSING: walk a={av} is outside [0,1]", file=sys.stderr)
                return 2
            out.append(run_arm(f"walk_a{_a}".replace(".", "p"),
                               (1.0 - av) * _bm + av * _wid))
'''

ANCHOR_FLAG = '    ap.add_argument("--wide-betas", dest="wide_betas", default="",'
ANCHOR_BODY = '    if args.wide_betas.strip():'


def main():
    s = SRC.read_text()
    if "--walk" in s:
        print("already patched; nothing to do")
    else:
        assert s.count(ANCHOR_FLAG) == 1, "flag anchor not unique"
        assert s.count(ANCHOR_BODY) == 1, "body anchor not unique"
        s = s.replace(ANCHOR_FLAG, FLAG + ANCHOR_FLAG, 1)
        s = s.replace(ANCHOR_BODY, BODY + ANCHOR_BODY, 1)
        SRC.write_text(s)
        print("patched")
    # verify it is live: the help text must appear, and a syntax check must pass
    r = subprocess.run(["/root/miniconda3/bin/python", "-c",
                        "import ast,sys; ast.parse(open(sys.argv[1]).read())",
                        str(SRC)], capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    r = subprocess.run(["/root/miniconda3/bin/python", str(SRC), "--help"],
                       capture_output=True, text=True)
    live = "--walk" in r.stdout
    print("flag live:", live)
    if not live:
        print(r.stdout[-500:], r.stderr[-500:])
    return 0 if (live and r.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
