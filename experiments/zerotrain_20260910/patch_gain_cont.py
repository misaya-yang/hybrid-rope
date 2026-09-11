#!/usr/bin/env python3
"""Add `--gain` to olmo_longnll.py so the gain axis can be measured continuously.

RUN ON THE SERVER (edits a server file).

WHY.  The 350-row RULER panel now has a gain measurement on it:

    BM @ gain 1.000             acc = 0.0371
    BM @ gain 1.138629436111989 acc = 0.4190     (the inherited YaRN mscale)
    -> the gain alone is worth +38.19pp, t = +15.05

That dwarfs the entire frequency-allocation programme, whose best arms move
+-5pp.  And the campaign's own note on the gain says the opposite ordering was
measured in NLL: "native-at-gain-1 is 0.0502 nats BETTER than native-at-YaRN-gain,
while the entire frequency table contributes ~0.0007".

If that NLL ordering also holds on OLMo, then the gain axis shows the SAME
inversion the frequency tables show -- NLL improving while task accuracy
collapses -- on a completely independent design face and with a task effect 7x
larger than anything the frequency work has produced.  That would make the
inversion a property of the objective, not of this table family.

`olmo_longnll.py` hardcodes GAIN = 1.138629436111989 and sets
`rot.attention_scaling` from it.  This adds an optional `--gain` that defaults to
that constant, so every existing invocation is bit-identical.  Both runners were
checked to use the same constant, so the two instruments are comparable.

Idempotent; verifies the flag is live before returning.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_longnll.py")

FLAG = '''    ap.add_argument("--gain", type=float, default=GAIN,
                    help=f"rotary amplitude (attention_scaling). Defaults to the "
                         f"inherited YaRN mscale {GAIN!r}, so every existing "
                         f"invocation is unchanged. The gain is a separate design "
                         f"face from the frequency allocation and is worth ~38pp "
                         f"on the 350-row RULER panel")
'''

ANCHOR = '    ap.add_argument("--only", default=None)\n'
USE = "        rot.attention_scaling = float(GAIN)\n"
USE_NEW = "        rot.attention_scaling = float(args.gain)\n"


def main():
    s = SRC.read_text()
    if "--gain" in s:
        print("already patched")
    else:
        assert s.count(ANCHOR) == 1, "arg anchor not unique"
        assert s.count(USE) == 1, "use anchor not unique"
        s = s.replace(ANCHOR, ANCHOR + FLAG, 1)
        s = s.replace(USE, USE_NEW, 1)
        SRC.write_text(s)
        print("patched")
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    # the default must be exactly the old constant, or every past NLL number moves
    r2 = subprocess.run(["/root/miniconda3/bin/python", "-c", f"""
import sys
sys.path.insert(0,"/root/autodl-tmp/nongeometric_screen_20260909/code")
sys.path.insert(0,"/root/autodl-tmp/phase1_20260910/repoharness")
sys.path.insert(0,"/root/autodl-tmp/phase1_20260910")
import olmo_longnll as L, argparse, inspect
src = inspect.getsource(L.main)
print("uses args.gain:", "float(args.gain)" in src)
print("default is the old constant:", L.GAIN == 1.138629436111989)
"""], capture_output=True, text=True)
    print(r2.stdout.strip())
    if r2.stderr.strip():
        print("STDERR:", r2.stderr[-300:])
    return 0 if ("syntax: OK" in r.stdout or r.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
