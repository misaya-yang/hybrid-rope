#!/usr/bin/env python3
"""Smoke test 32768 on OLMo before arming the length-axis chain.

RUN THIS ON THE SERVER (it writes server paths):
    scp s8_smoke.py root@...:/root/autodl-tmp/phase1_20260910/
    ssh ... 'cd /root/autodl-tmp/phase1_20260910 && /root/miniconda3/bin/python s8_smoke.py'

WHY A SMOKE TEST.  32768 is 8x the native window and has never been run on this
model in this campaign.  Two failure modes are invisible to `--dry-run` (which
deliberately does not load the model):

  * SDPA falls back to a non-flash kernel -> the 16-head x 32768^2 score tensor
    is tens of GB and the process dies on the first row, mid-chain
  * prefill is fine but generation against a 32640-token KV cache is not

One row costs about a minute and removes both.  The panel is one row at each
length pulled out of prepared_s8_01.  Running two rows cannot constitute
"reading the panel": the pre-registration (S8_PREREG_20260911.md, git a909d1b)
governs the four-arm comparison, not this feasibility probe.  Results discarded.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-tmp/phase1_20260910")
SRC = pathlib.Path("/root/autodl-tmp/olmo_fast_screen_20260908/prepared_s8_01/screen.jsonl")
PANEL = ROOT / "s8_smoke_panel"
OUT = ROOT / "s8_smoke_out"

# one row at each length: 32768 is the feasibility question, 4096 the baseline
rows = [json.loads(l) for l in SRC.open()]
far = [r for r in rows if r["length_cap"] == 32768]
near = [r for r in rows if r["length_cap"] == 4096]
assert far and near, "panel shape changed"
PANEL.mkdir(parents=True, exist_ok=True)
with (PANEL / "screen.jsonl").open("w") as fh:
    for r in [far[0], near[0]]:
        fh.write(json.dumps(r) + "\n")
print("smoke panel:", [r["row_id"] for r in [far[0], near[0]]], flush=True)

(ROOT / "empty_archive").mkdir(exist_ok=True)
cmd = [
    "/root/miniconda3/bin/python", str(ROOT / "olmo_beta.py"),
    "--root", str(OUT),
    "--model", "/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct",
    "--panel", str(PANEL / "screen.jsonl"),
    "--archive", str(ROOT / "empty_archive"),
    "--betas", "1.0", "--turns", "",
]
env = dict(PATH="/root/miniconda3/bin:/usr/bin:/bin",
           PYTHONPATH=":".join([
               "/root/autodl-tmp/nongeometric_screen_20260909/code",
               str(ROOT / "repoharness"), str(ROOT)]))
import os
env = {**os.environ, **env}

r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
print("\n".join(tail), flush=True)
print("rc =", r.returncode, flush=True)

f = OUT / "beta_b1p0.jsonl"
if f.exists():
    print("\n--- rows written ---")
    for l in f.open():
        d = json.loads(l)
        print(f"  {d['row_id']:<28} cap={d['length_cap']:<6} "
              f"correct={d['correct']} eos={d['ended_eos']}")
else:
    print("NO OUTPUT WRITTEN")
