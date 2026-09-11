#!/usr/bin/env python3
"""patch_mfile.py — add `--m-file path.json:armname` to olmo_beta.py.

For the coverage-theory tables (COVERAGE_PREREG_20260911 P1): they are literal
m-arrays (1.5 x wide-BM, +0.5 shift), not members of any existing family flag,
so they need one generic entry point.  The patch is idempotent and REFUSES to
run an array that is not a legal table:

  * length must be 64
  * nu_j = theta^(-j/64) * 4^(-m_j) must strictly decrease with the runner's
    own theta (the OLMo config constant inside olmo_beta.py)
  * sum_m is recorded into the row/summary the same as family arms

Usage on the server (write access required — the GPU session, not this repo):

    cd /root/autodl-tmp/phase1_20260910
    /root/miniconda3/bin/python patch_mfile.py            # applies + self-checks
    # then, for the 8x panel:
    export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
    setsid nohup /root/miniconda3/bin/python olmo_beta.py \
      --root /root/autodl-tmp/phase1_20260910/s8_scale8x \
      --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
      --panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_s8_01/screen.jsonl \
      --archive /root/autodl-tmp/phase1_20260910/empty_archive \
      --betas "" --turns "" \
      --m-file /root/autodl-tmp/phase1_20260910/tables/scale8x_wide.json:scale8x_wide \
      > s8_scale8x.log 2>&1 < /dev/null & disown

(the two JSON files live in this repo at code/tables/ and must be copied to
/root/autodl-tmp/phase1_20260910/tables/ first).
"""
from __future__ import annotations

import json
import re
import sys

TARGET = "olmo_beta.py"


def main() -> int:
    src = open(TARGET).read()
    if "--m-file" in src:
        print("--m-file already present; nothing to do")
        return 0

    arg = '''    ap.add_argument("--m-file", default="",
                    help="comma list of path.json:armname with a literal m "
                         "array (length 64, nu strictly decreasing); the "
                         "coverage-theory tables enter through this door")
'''
    m = re.search(r"(    ap\.add_argument\(\"--pro-tables\".*?\n)", src, re.S)
    assert m, "anchor --pro-tables not found"
    src = src[:m.end()] + arg + src[m.end():]

    body = '''    if args.m_file.strip():
        # literal m-array arms (coverage-theory tables); same run_arm path as
        # every family arm, with legality checks that refuse bad arrays
        import math as _math
        _lam = _math.log(500_000.0) / (64 * _math.log(2.0))
        for spec in args.m_file.split(","):
            path, arm = spec.split(":")
            d = json.load(open(path))
            arr = np.asarray(d["m"], dtype=np.float64)
            if arr.shape != (64,):
                raise SystemExit(f"REFUSING {arm}: m array must have length 64")
            if not np.all(np.diff(arr) > -_lam / 2 - 1e-12):
                raise SystemExit(f"REFUSING {arm}: nu not strictly decreasing")
            if abs(float(arr.sum()) - float(d.get("sum_m", arr.sum()))) > 1e-9:
                raise SystemExit(f"REFUSING {arm}: sum_m mismatch vs file")
            out.append(run_arm(arm, arr))
'''
    anchor = "    out = []\n"
    assert src.count(anchor) == 1, "anchor 'out = []' not unique"
    src = src.replace(anchor, anchor + body)

    open(TARGET, "w").write(src)
    print("patched olmo_beta.py with --m-file; verify with --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
