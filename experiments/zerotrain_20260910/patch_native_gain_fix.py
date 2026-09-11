#!/usr/bin/env python3
"""Fix the native-gain patch: the first attempt guarded on a false positive.

RUN ON THE SERVER.

WHAT WENT WRONG.  The guard was `'"native"' in s and "np.zeros(64)" in s`, which
is true of the UNPATCHED file: `"native"` occurs in the GEOMETRY_FREE tuple
("native", "native_gain1", "interp") and np.zeros(64) occurs elsewhere.  So the
script printed "native already in _tables", skipped the edit, and armed a chain
that the runner will refuse with `unknown --gain-tables ['native']`.

This is the same failure class as the other lessons on this campaign -- a check
that reports success without testing the thing it claims to test.  The guard here
looks for the exact registry key `"native":` inside the `_tables` block.

Idempotent and verified by reading the file back.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")
ROOT = "/root/autodl-tmp/phase1_20260910"

ANCHOR = "        _tables = {\n"
NEW = ('        _tables = {\n'
       '            # the untouched grid: every slot at its trained frequency, so\n'
       '            # only the gain moves between arms.  Needed because the gain\'s\n'
       '            # NLL effect changes sign between this table and bm.\n'
       '            "native": lambda: np.zeros(64),\n')


def registry_has_native() -> bool:
    """True only if `_tables` itself contains the key -- read back from disk."""
    s = SRC.read_text()
    blk = s.split("_tables = {", 1)
    if len(blk) < 2:
        return False
    return '"native":' in blk[1].split("}", 1)[0]


def main():
    if registry_has_native():
        print("registry already has native")
    else:
        s = SRC.read_text()
        assert s.count(ANCHOR) == 1, f"anchor count {s.count(ANCHOR)}"
        SRC.write_text(s.replace(ANCHOR, NEW, 1))
        print("patched")
    ok = registry_has_native()
    print("registry has native (read back):", ok)
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    # and prove it end to end: a dry run must list native as a known table
    d = subprocess.run(["bash", "-lc",
        f"cd {ROOT} && PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:"
        f"{ROOT}/repoharness:{ROOT} /root/miniconda3/bin/python olmo_beta.py "
        f"--root /tmp/ngainchk "
        f"--model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct "
        f"--panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl "
        f"--archive /root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01 "
        f'--betas "" --turns "" --gain-tables native --gains 1.0 --dry-run'],
        capture_output=True, text=True)
    ok_run = "DRY_RUN_OK" in d.stdout and "REFUSING" not in d.stdout
    print("dry run with --gain-tables native:", "OK" if ok_run else "FAILED")
    if not ok_run:
        print(d.stdout[-400:], d.stderr[-300:])
    return 0 if (ok and ok_run) else 1


if __name__ == "__main__":
    sys.exit(main())
