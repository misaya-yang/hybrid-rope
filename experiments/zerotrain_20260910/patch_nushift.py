#!/usr/bin/env python3
"""Add `--nu-shift` to olmo_beta.py: the long-bridge signed pair, ported to OLMo.

RUN ON THE SERVER (edits a server file).

WHY.  Re-reading the history turned up several zero-training tables that scored
above MrRoPE on the Qwen panel, but every one of them is |t| <= 1.23 -- noise at
n=36.  The one design that CAN detect a directed effect at that power is a SIGNED
PAIR: same slots, same magnitude, opposite sign.  The library has exactly one
real such pair, LongBridge:

    LongBridgeSlower   slots 36-39,  nu_j -= 1/131072   -> 128K +1.94pp
    LongBridgeFaster   slots 36-39,  nu_j += 1/131072   -> 128K -4.17pp
    difference +6.11pp, SE 8.14, t = +0.75

and the ground truth states its semantics precisely: "公共相位 -1 rad @128K",
i.e. over the evaluation length 4W the phase shifts by one radian.  The other two
"mirror controls" in the library are byte-identical to their tested arms (SE 0.00
on all 36 rows), so they are not controls at all and provide zero information.

The slots are not arbitrary.  They are exactly the slots whose EFFECTIVE period
(2*pi*4^m/omega) lies in [W, 4W] -- the band that spans precisely the
extrapolation range.  Recomputed independently here:

    Qwen  MrRoPE -> [36,37,38,39]   <- reproduces the historical choice exactly
    OLMo  MrRoPE -> [28,29,30,31]

So the same experiment can be run where there is real power: OLMo has a 350-row
panel and a 180-row held-out panel, against Qwen's 36 rows.

THE FLAG.  `--nu-shift "<delta>:<j1,j2,...>"`, repeatable with ";".  delta is in
the same absolute nu units as the historical arms, so 1/(4W) reproduces their
semantics: OLMo 4W = 16384 -> 1/16384 = 6.103515625e-5.  Both signs are run, and
the pre-registered statistic is the SIGNED PAIR difference.

Idempotent; verified by dry run before returning.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")
ROOT = "/root/autodl-tmp/phase1_20260910"

FLAG = '''    ap.add_argument("--nu-shift", dest="nu_shift", default="",
                    help="semicolon list of <delta>:<j1,j2,...>; shifts nu on the "
                         "named slots by delta (absolute, same units as the "
                         "historical LongBridge arms: delta=1/(4W) reproduces "
                         "'phase -1 rad at the eval length'). Both signs are "
                         "meant to be run as a SIGNED PAIR")
'''

BODY = '''    if args.nu_shift.strip():
        # THE LONG-BRIDGE SIGNED PAIR, ported to this geometry.  Slot choice is
        # not free: they are the slots whose effective period lies in [W, 4W].
        from experiments.curvature_20260910.tables import inv_freq_to_m as _i2m
        for spec in [s for s in args.nu_shift.split(";") if s.strip()]:
            _d, _js = spec.split(":")
            delta = float(_d)
            slots = [int(x) for x in _js.split(",") if x.strip()]
            base_m = np.asarray(m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]), float)
            nu = m_to_nu(base_m, OLMO["theta"]).copy()
            before = nu[slots].copy()
            nu[slots] += delta
            m2 = _i2m(nu, OLMO["theta"])
            print(json.dumps({"nu_shift": delta, "slots": slots,
                              "rel_change": (delta / before).tolist(),
                              "sum_m": float(m2.sum())}), flush=True)
            out.append(run_arm(f"nu_{'p' if delta > 0 else 'm'}"
                               f"{abs(delta):.3e}".replace(".", "p").replace("-", "m"),
                               m2))
'''

ANCHOR = '    if args.walk.strip():\n'


def main():
    s = SRC.read_text()
    if 'ap.add_argument("--nu-shift"' in s:
        print("flag already present")
    else:
        assert s.count(ANCHOR) == 1, "anchor not unique"
        # flag goes with the other arg definitions; body before the walk block
        arg_anchor = '    ap.add_argument("--walk", default="",\n'
        assert s.count(arg_anchor) == 1, "arg anchor not unique"
        s = s.replace(arg_anchor, FLAG + arg_anchor, 1)
        s = s.replace(ANCHOR, BODY + ANCHOR, 1)
        SRC.write_text(s)
        print("patched")
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    # dry run both signs so the arm names and the relative change are visible
    d = subprocess.run(["bash", "-lc",
        f"cd {ROOT} && PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:"
        f"{ROOT}/repoharness:{ROOT} /root/miniconda3/bin/python olmo_beta.py "
        f"--root /tmp/nuchk "
        f"--model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct "
        f"--panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl "
        f"--archive {ROOT}/empty_archive --betas '' --turns '' "
        f"--nu-shift '6.103515625e-05:28,29,30,31;-6.103515625e-05:28,29,30,31' "
        f"--dry-run"], capture_output=True, text=True)
    print(d.stdout.strip()[-500:])
    if d.stderr.strip():
        print("STDERR:", d.stderr[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
