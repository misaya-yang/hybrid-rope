#!/usr/bin/env python3
"""Add `--m-file path.json:armname` to olmo_beta.py — v2, structure-robust.

WHY A v2.  The v1 patch (ds_workspace/.../code/patch_mfile.py) was written
against an older olmo_beta.py and anchored on a line that no longer exists
(`("pro_step42", _pro_table("step42")),`).  It did not detect this, inserted
into the middle of a multi-line add_argument call, and CORRUPTED the runner —
a syntax error at line 121.  The runner was restored from the backup this
script makes.  v1 is therefore unsafe on the current file and must not be used.

WHAT v2 DOES DIFFERENTLY
  1. Anchors on text verified present in the CURRENT file, and refuses loudly
     (before writing) if any anchor is missing or non-unique.
  2. Backs up first, and AUTO-RESTORES on any failure — a broken runner is the
     one outcome that must not be possible.
  3. Verifies everything that is verifiable without a GPU: ast.parse (the exact
     failure mode of v1), the flag and entry block each present exactly once, and
     the entry sitting immediately after the arms accumulator.  FULL functional
     verification needs one real run --- the arms are built by a closure inside
     main(), so they cannot be constructed by importing the module.  Success is
     reported only if the structural checks pass, and that limit is stated.

RUN ON THE INSTANCE, in the directory holding olmo_beta.py:
    /root/miniconda3/bin/python patch_mfile_v2.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")
BAK = SRC.with_suffix(".py.bak_premfile_v2")
PY = "/root/miniconda3/bin/python"

# Anchor 1: end of the --pro-tables argument (two lines, verified present).
ANCHOR1 = ('    ap.add_argument("--pro-tables", dest="pro_tables", default="",\n'
           '                    help="condEVQ and/or step42 from the Pro model plan")\n')
FLAG = ('    ap.add_argument("--m-file", dest="m_file", default="",\n'
        '                    help="path.json:armname[;...] literal 64-slot m arrays")\n')

# Anchor 2: the arms accumulator, after which every family appends.
ANCHOR2 = "    out = []\n"
ENTRY = '''
    if args.m_file.strip():
        # Literal m-arrays supplied as path.json:armname, for tables that are not
        # members of any family flag (the amplitude-scaling sweeps).  The patch
        # refuses at apply time if this anchor is not unique, so landing here is
        # not in question; the array itself is validated by run_arm.
        import pathlib as _pl
        for _spec in [x for x in args.m_file.split(";") if x.strip()]:
            _path, _, _nm = _spec.partition(":")
            _m = np.asarray(json.loads(_pl.Path(_path).read_text())["m"], float)
            assert _m.size == 64, f"{_nm}: expected 64 slots, got {_m.size}"
            out.append(run_arm(_nm.strip(), _m))
'''

VERIFY = r'''
import ast, sys, pathlib
src = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py").read_text()
tree = ast.parse(src)                      # syntax is the v1 failure mode
nadd = sum(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
           for n in ast.walk(tree))
ok_flag = src.count("--m-file") == 1
ok_entry = src.count("if args.m_file.strip():") == 1
ok_loop = "out.append(run_arm(_nm.strip(), _m))" in src
# the insertion must sit AFTER the accumulator and INSIDE main
i_out = src.index("    out = []")
i_ent = src.index("    out = []") + len("    out = []")
ok_order = src[i_ent:i_ent+40].lstrip().startswith("if args.m_file")
print("add_argument calls :", nadd)
print("flag present once  :", ok_flag)
print("entry present once :", ok_entry)
print("arm append present :", ok_loop)
print("entry right after accumulator:", ok_order)
sys.exit(0 if (ok_flag and ok_entry and ok_loop and ok_order) else 1)
'''


def main() -> int:
    src = SRC.read_text()
    if "--m-file" in src:
        print("already patched; nothing to do")
        return 0

    for name, anchor in (("ANCHOR1", ANCHOR1), ("ANCHOR2", ANCHOR2)):
        c = src.count(anchor)
        if c != 1:
            print(f"REFUSING: {name} occurs {c} times, expected exactly 1")
            return 2

    shutil.copy2(SRC, BAK)
    print(f"backup -> {BAK.name}")
    try:
        patched = src.replace(ANCHOR1, ANCHOR1 + FLAG, 1)
        patched = patched.replace(ANCHOR2, ANCHOR2 + ENTRY, 1)
        SRC.write_text(patched)

        r = subprocess.run([PY, "-c", "import ast,sys;ast.parse(open(sys.argv[1]).read())",
                            str(SRC)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"syntax: {r.stderr[-400:]}")

        r = subprocess.run([PY, "-c", VERIFY], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"verify failed:\n{r.stdout[-600:]}\n{r.stderr[-400:]}")
        print(r.stdout.strip())
    except Exception as exc:                      # noqa: BLE001 - restore on anything
        shutil.copy2(BAK, SRC)
        print(f"FAILED and RESTORED the original runner:\n{exc}")
        return 1

    print("patched; structural checks passed. Full functional check happens at\n          first run: the arms are built inside main() and cannot be constructed\n          by import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
