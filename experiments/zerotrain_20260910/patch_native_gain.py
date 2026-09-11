#!/usr/bin/env python3
"""Add `native` to the gain-table registry and arm the native x gain 2x2.

RUN ON THE SERVER (edits a server file, then arms a chain).

WHY.  The continuous instrument says the gain's NLL effect FLIPS SIGN with the
table at 4096:

    native    : gain 1.0 = 2.834894   gain YaRN = 2.984121   -> gain 1.0 better
    beta_b1_BM: gain 1.0 = 3.245447   gain YaRN = 2.955406   -> gain YaRN better

That reproduces the campaign's own Qwen note (which was about native, and says
gain-1 is 0.0502 nats better) with the same sign and a larger magnitude, and it
establishes a gain x table interaction.

On the TASK side only the BM column exists: BM@1.0 = 0.0371, BM@YaRN = 0.4190.
The native row is missing, and it is the interesting one -- it is the only
configuration where the NLL ordering points AWAY from the deployed setting.

  * native@1.0 ALSO better on tasks -> the deployed setting is dominated in both
    senses, and that is an actionable zero-training result
  * native@1.0 worse on tasks     -> a second, cleaner instance of the NLL/task
    inversion, because not even the frequency table changes

BM at both gains is already measured, so this needs two arms -- roughly 25 min.

`native` is added to the gain-table dict rather than run through `--arms` so the
gains loop applies to it; `--arms` has no gains loop.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")
ROOT = "/root/autodl-tmp/phase1_20260910"

ANCHOR = '        _tables = {\n'
NEW = ('        _tables = {\n'
       '            # the untouched grid: every slot at its trained frequency, so\n'
       '            # only the gain moves between arms. Needed because the gain\'s\n'
       '            # NLL effect changes sign between this table and bm.\n'
       '            "native": lambda: np.zeros(64),\n')

CHAIN = r'''#!/bin/bash
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
OUT=/root/autodl-tmp/phase1_20260910/olmo_ngain
mkdir -p "$OUT"

while pgrep -f "chain_natur[a]l" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "chain_wal[k]" >/dev/null 2>&1; do sleep 30; done
sleep 20

echo "=== STAGE native_gain_2x2 START $(date -Is) ==="
"$PY" olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$ARCH" \
    --betas "" --turns "" --gain-tables native \
    --gains 1.0,1.138629436111989 > olmo_ngain.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== STAGE native_gain_2x2 FAILED rc=$rc $(date -Is) ==="; tail -25 olmo_ngain.log
else
  echo "=== STAGE native_gain_2x2 OK $(date -Is) ==="
fi
echo "=== NGAIN COMPLETE $(date -Is) ==="
'''


def main():
    s = SRC.read_text()
    if '"native"' in s and "np.zeros(64)" in s and "_tables" in s:
        print("native already in _tables")
    else:
        assert s.count(ANCHOR) == 1, "tables anchor not unique"
        s = s.replace(ANCHOR, NEW, 1)
        SRC.write_text(s)
        print("patched _tables")
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    # verify the registry now resolves native and that it is the untouched grid
    c = subprocess.run(["/root/miniconda3/bin/python", "-c", """
import re
src = open("/root/autodl-tmp/phase1_20260910/olmo_beta.py").read()
print("native registered:", '"native":' in src)
print("is zeros:", "np.zeros(64)" in src.split('"native"')[1][:80])
"""], capture_output=True, text=True)
    print(c.stdout.strip())

    p = pathlib.Path(ROOT) / "chain_ngain.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    chk = subprocess.run(["bash", "-lc", f"test -s {p} && wc -l < {p}"],
                         capture_output=True, text=True)
    print("chain landed, lines:", chk.stdout.strip() or "MISSING")
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_ngain.sh > chain_ngain.log "
                    "2>&1 < /dev/null & disown ; sleep 5 ; "
                    "ps -eo pid,args | grep 'chain_ngai[n]' | head -2"], check=False)
    print("armed chain_ngain.sh: native at gain 1.0 and gain YaRN, 350 rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
