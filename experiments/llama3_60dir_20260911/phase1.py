"""Phase 1: the arms that close the existing theory.  Registry + preflight + launch.

Now governed by Plan B (RoPE_Integrated_Experiment_Guide_Codex_20260911.md) §5.
NOTE the prefix trap: Plan B's `E-` labels are SECTION numbers, not arm ids.
The mapping is E-A<->A, E-B<->B, E-C<->C/C2, E-E<->D, E-G<->E.

    A   qwen4x_power + BM arm      E-A  cross-model sign of the ramp ordering
    B   amp4x m_p = 1.13 / 1.30    E-B  is the m<=1 box artificial?
    C   scale x gain 2x2           E-C  C00/C10/C01/C11 -- was the 8x failure the
                                   scale multiple, the gain, or a real limit?
    C2  CY8 / CM8 controls         E-C  do the published methods recover at s=8?
    E   g2x2 native/MR/BM/b3       E-G  is the gain x table interaction general?
    D   official YaRN same panel   E-E  is BM a checkpoint-mismatch artefact?

Why this file exists at all
---------------------------
Three defects were found while wiring these up, each of the "reads right, fails
on launch" kind that has already cost this campaign whole cycles:

1. `--m-file` is a plain `store` argparse action, so `--m-file A --m-file B`
   keeps only B.  The documented B command used that form and would have run
   only the 1.30 arm, silently dropping the 1.13 arm -- the entire point.
   The runner's own parser expects ONE `--m-file` with `path:name;path:name`.

2. `--dry-run` returns at olmo_beta.py:243, but the `--m-file` block is at
   line 314.  So a passing dry-run is NOT evidence that the injection works,
   and the preregistration's "dry-run passed" claim did not cover it.
   `preflight` therefore validates the argv itself, not just the dry-run.

3. `--m-file` dropped the gain entirely and `--gains` built its own tables, so
   the two were mutually exclusive and Plan B's scale x gain 2x2 (E-C) was
   INEXPRESSIBLE.  `patch_mgain.py` adds `--m-gain`; it validates its anchors and
   refuses to write unless the patched source compiles.

4. The documented PYTHONPATH (`.../nongeometric_screen_20260909/code:$D/repoharness:$D`)
   does not import: `$D/repoharness` does not exist and that root has no
   `scripts/` package.  The working root is `.../olmo_fast_screen_20260908/code`
   for `scripts.*`, plus the nongeometric root for `experiments.*`.

Run:  python phase1.py commands          # print the exact shell for every arm
      python phase1.py preflight         # check prerequisites over ssh (read-only)
      python phase1.py launch <arm-id>   # launch one arm (needs GPU)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

SSH = ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
       "-p", "53405", "root@connect.westc.seetacloud.com"]

D = "/root/autodl-tmp"
P = f"{D}/phase1_20260910"
PY = "/root/miniconda3/bin/python"

# the two roots that actually import (see the module docstring, defect 3)
PYTHONPATH = f"{D}/olmo_fast_screen_20260908/code:{D}/nongeometric_screen_20260909/code"

OLMO = f"{D}/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct"
QWEN15 = f"{D}/qwen25_1p5b_32k"
PANEL350 = f"{D}/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl"
ARCHIVE350 = f"{D}/olmo_fast_screen_20260908/run_ruler_newtasks_01"
PANEL_S8 = f"{D}/olmo_fast_screen_20260908/prepared_s8_01"
PREPARED_YARN = f"{D}/olmo_fast_screen_20260908/prepared_natural_yarn_01"
HARNESS_RUN = f"{D}/olmo_fast_screen_20260908/code/scripts/experiments/olmo_fast_screen/run.py"
LONGTEXT = f"{D}/longtext/prepared_pg19_4x"

# the exact byte size the mirror must deliver for the 1.5B checkpoint
QWEN15_BYTES = 3087467144


@dataclass
class Arm:
    id: str
    name: str
    why: str
    runner: str                      # the python file to execute
    args: list
    root: str                        # --root (output dir)
    needs_model: str = OLMO
    closes: str = ""
    reader: str = ""
    requires: list = field(default_factory=list)

    def argv(self):
        """One line: the command must be copy-pasteable without re-wrapping."""
        return " ".join([f"$PY {self.runner}"] + list(self.args))


def _amp4x_args():
    # ONE --m-file, semicolon separated.  Two flags would silently keep only the
    # second (defect 1).
    spec = (f"{P}/tables/amp4x_1p13.json:amp4x_1p13;"
            f"{P}/tables/amp4x_1p30.json:amp4x_1p30")
    return ["--root", f"{P}/olmo_amp4x", "--model", OLMO,
            "--panel", PANEL350, "--archive", ARCHIVE350,
            "--betas", '""', "--turns", '""',
            "--m-file", f'"{spec}"']


ARMS = {
    "A": Arm(
        id="A", name="qwen4x_power + BM arm",
        why=("the only powered cross-model sign test for the ramp-shape ordering. "
             "BM>MR => the effect exists across two checkpoint families; "
             "~0 or reversed => the frozen optimum is checkpoint-specific, not a "
             "universal geometric law."),
        runner=f"{P}/qwen_longnll.py",
        args=["--root", f"{P}/qwen4x_power", "--model", QWEN15,
              "--nll-dir", LONGTEXT, "--far", "131073", "--tail", "512",
              "--only", "native,beta_b1_BM"],
        root=f"{P}/qwen4x_power",
        needs_model=QWEN15,
        closes="Is the OLMo ramp-shape ordering model-specific?",
        reader=f"$PY {P}/qwen4x_power_read.py",
        requires=[("file", f"{QWEN15}/model.safetensors", QWEN15_BYTES),
                  ("dir", LONGTEXT),
                  ("file", f"{P}/qwen4x_power/rows.jsonl", None),
                  ("arm_in_runner", f"{P}/qwen_longnll.py", "beta_b1_BM")],
    ),
    "B": Arm(
        id="B", name="amp4x  m_p = 1.13 / 1.30",
        why=("was every past ramp search sealed inside the m<=1 box?  If 1.13/1.30 "
             "improve, the method factors as m_j = a * r_j (amplitude x profile). "
             "If not, reach-only coverage is insufficient and T3 must be demoted to "
             "a reach-resolution tradeoff."),
        runner=f"{P}/olmo_beta.py", args=_amp4x_args(), root=f"{P}/olmo_amp4x",
        closes="Is the m<=1 box artificial?",
        reader=f"$PY {P}/amp4x_read.py --panel 350",
        requires=[("file", f"{P}/tables/amp4x_1p13.json", None),
                  ("file", f"{P}/tables/amp4x_1p30.json", None),
                  ("file", PANEL350, None), ("dir", ARCHIVE350),
                  ("m_file_support", f"{P}/olmo_beta.py", None)],
    ),
    "C": Arm(
        id="C", name="E-C: scale x gain 2x2 + two existing-method controls",
        why=("Plan B section 5 / E-C.  The historical 8x arms all used g4 on the "
             "s=8 array, i.e. they were the MISMATCHED-anchor cell C10, never the "
             "correctly deployed C11.  This 2x2 separates 'the actual scale "
             "multiple' from 'the gain' and decides whether the 0/48 was a "
             "deployment error or a real limit.  Verified, not assumed: the old "
             "scale8x_wide IS ec_C10_C11 bit-for-bit, always run at g4."),
        runner=f"{P}/olmo_beta.py",
        # two invocations: the 2x2 (both gains applied to both arrays) and the
        # two existing-method controls at g8.  --m-gain is comma-separated and
        # now applies to --m-file arms (patch_mgain.py), which is what makes the
        # factorial expressible at all.
        args=["--root", f"{P}/olmo_ec", "--model", OLMO,
              "--panel", f"{PANEL_S8}/screen.jsonl", "--archive", ARCHIVE350,
              "--betas", '""', "--turns", '""',
              "--m-file", f'"{P}/tables/ec/ec_C00_C01.json:ec_C00_C01;'
                           f'{P}/tables/ec/ec_C10_C11.json:ec_C10_C11"',
              "--m-gain", '"1.138629436111989,1.207944154089986"'],
        root=f"{P}/olmo_ec",
        closes=("Was the 8x failure the amplitude box, the gain, or a real limit?"),
        reader=f"$PY {P}/amp4x_read.py --panel 180",
        requires=[("file", f"{P}/tables/ec/ec_C00_C01.json", None),
                  ("file", f"{P}/tables/ec/ec_C10_C11.json", None),
                  ("file", f"{P}/tables/ec/ec_CY8.json", None),
                  ("file", f"{P}/tables/ec/ec_CM8.json", None),
                  ("file", f"{PANEL_S8}/screen.jsonl", None),
                  ("m_file_support", f"{P}/olmo_beta.py", None),
                  ("m_gain_support", f"{P}/olmo_beta.py", None)],
    ),
    "C2": Arm(
        id="C2", name="E-C controls: CY8 (official YaRN s=8) and CM8 (MrRoPE s=8)",
        why=("the two strong existing-method controls.  If they also recover, the "
             "old '8x unreachable' claim was at least partly an artefact of the old "
             "deployment conditions; if only C11 recovers, the effect is a genuine "
             "scale x gain interaction and cannot be attributed to coverage alone."),
        runner=f"{P}/olmo_beta.py",
        args=["--root", f"{P}/olmo_ec_controls", "--model", OLMO,
              "--panel", f"{PANEL_S8}/screen.jsonl", "--archive", ARCHIVE350,
              "--betas", '""', "--turns", '""',
              "--m-file", f'"{P}/tables/ec/ec_CY8.json:ec_CY8;'
                           f'{P}/tables/ec/ec_CM8.json:ec_CM8"',
              "--m-gain", '"1.207944154089986"'],
        root=f"{P}/olmo_ec_controls",
        closes="Do the published methods recover at 8x under a matched gain?",
        reader=f"$PY {P}/amp4x_read.py --panel 180",
        requires=[("file", f"{P}/tables/ec/ec_CY8.json", None),
                  ("file", f"{P}/tables/ec/ec_CM8.json", None),
                  ("m_gain_support", f"{P}/olmo_beta.py", None)],
    ),
    "D": Arm(
        id="D", name="official YaRN on the same panel",
        why=("cheap and mandatory.  fresh_72 already shows YaRN ~ MrPro << BM; "
             "reproducing it on the frozen panel forecloses the reviewer's most "
             "natural objection, that BM only fixes a checkpoint mismatch."),
        runner=HARNESS_RUN,
        args=["--prepared", PREPARED_YARN, "--out", f"{P}/olmo_yarn_baseline"],
        root=f"{P}/olmo_yarn_baseline",
        closes="Is BM an artefact of comparing against a mis-built baseline?",
        reader="$PY scripts/experiments/olmo_fast_screen/run.py --read "
               f"{P}/olmo_yarn_baseline",
        requires=[("file", f"{PREPARED_YARN}/tables.json", None),
                  ("file", f"{PREPARED_YARN}/spec.json", None),
                  ("file", f"{PREPARED_YARN}/screen.jsonl", None),
                  ("file", HARNESS_RUN, None)],
    ),
    "E": Arm(
        id="E", name="g2x2 columns mrpro / b3",
        why=("the table x gain interaction is currently carried by native/BM alone.  "
             "Same direction on mrpro and b3 => attention scaling is a general "
             "interacting axis; only BM => gain is not a universal rescue and "
             "depends on the frequency shape.  Both outcomes change the writing."),
        runner=f"{P}/olmo_beta.py",
        args=["--root", f"{P}/olmo_gain2x2", "--model", OLMO,
              "--panel", PANEL350, "--archive", ARCHIVE350,
              "--betas", '""', "--turns", '""',
              "--gain-tables", "mrpro,b3",
              "--gains", "1.0,1.138629436111989"],
        root=f"{P}/olmo_gain2x2",
        closes="Is the gain x table interaction general or BM-specific?",
        reader=f"$PY {P}/amp4x_read.py --panel 350",
        requires=[("file", PANEL350, None), ("dir", ARCHIVE350)],
    ),
}

ORDER = ["B", "A", "C", "C2", "E", "D"]   # author order: amp4x was flagged highest in master


def _run(cmd, timeout=180):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _remote_checks(arm):
    """Build one remote shell probe for this arm's prerequisites."""
    lines = []
    for req in arm.requires:
        kind, path = req[0], req[1]
        if kind == "file":
            want = req[2]
            if want:
                lines.append(
                    f'echo "file|{path}|$(stat -c %s "{path}" 2>/dev/null || echo MISSING)|{want}"')
            else:
                lines.append(f'echo "file|{path}|$([ -f "{path}" ] && echo present || echo MISSING)|-"')
        elif kind == "dir":
            lines.append(f'echo "dir|{path}|$([ -d "{path}" ] && echo present || echo MISSING)|-"')
        elif kind == "arm_in_runner":
            name = req[2]
            lines.append(
                f'echo "arm|{path}:{name}|$(grep -c \'"{name}"\' "{path}" 2>/dev/null || echo 0)|>=1"')
        elif kind == "m_file_support":
            lines.append(f'echo "mfile|{path}|$(grep -c -- "--m-file" "{path}" 2>/dev/null || echo 0)|>=1"')
        elif kind == "m_gain_support":
            # E-C is inexpressible without this: the --m-file branch used to drop the
            # gain entirely and the --gains branch built its own tables.
            lines.append(f'echo "mgain|{path}|$(grep -c -- "--m-gain" "{path}" 2>/dev/null || echo 0)|>=1"')
    return "\n".join(lines)


def preflight(arm_ids=None):
    ids = arm_ids or ORDER
    script = [
        f"export PYTHONPATH={PYTHONPATH}",
        f"echo 'pyimport|scripts|'$({PY} -c 'import scripts.experiments.olmo_fast_screen.runtime' 2>/dev/null && echo OK || echo FAIL)'|OK'",
        f"echo 'pyimport|experiments|'$({PY} -c 'import experiments.curvature_20260910.tables' 2>/dev/null && echo OK || echo FAIL)'|OK'",
        # Two traps here, both verified on this instance.  `nvidia-smi -L | wc -l`
        # and `nvidia-smi --query-gpu=... | grep -c .` BOTH return 1 on a no-card
        # instance, because "No devices were found" goes to stdout (rc 6).  Ask
        # torch instead: that is the answer the runner itself will get.
        f"echo 'gpu|cuda devices (torch)|'$({PY} -c "
        f"'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)'|>=1'",
    ]
    for aid in ids:
        arm = ARMS[aid]
        # argv-level check: `--m-file` must appear exactly once (defect 1)
        n_mfile = sum(1 for a in arm.args if a == "--m-file")
        script.append(f"echo 'argv|{aid} --m-file count|{n_mfile}|{{0,1}}'")
        if n_mfile == 1:
            joined = " ".join(arm.args)
            n_spec = joined.count(";") + 1
            script.append(f"echo 'argv|{aid} m-file specs|{n_spec}|>=1'")
        script.append(_remote_checks(arm))
    rc, out, err = _run("\n".join(script))
    if rc != 0 and not out:
        print(f"ssh failed: {err.strip()[:300]}")
        return 2

    results, failures = [], []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        kind, what, got, want = parts
        if want == "-":
            ok = got != "MISSING"
        elif want == "{0,1}":
            ok = got in ("0", "1")
        elif want == ">=1":
            try:
                ok = int(got) >= 1
            except ValueError:
                ok = False
        elif want.isdigit():
            ok = got.isdigit() and int(got) == int(want)
        else:
            ok = got == want
        results.append((kind, what, got, want, ok))
        if not ok:
            failures.append((kind, what, got, want))

    width = max(len(w) for _, w, _, _, _ in results) if results else 10
    for kind, what, got, want, ok in results:
        print(f"  [{'ok ' if ok else 'FAIL'}] {what:<{width}}  got={got:<14} want={want}")

    print()
    if failures:
        print(f"REFUSING: {len(failures)} prerequisite(s) unmet")
        for kind, what, got, want in failures:
            print(f"  - {what}: got {got}, want {want}")
        if any(k == "gpu" for k, *_ in failures):
            print("  note: GPU is the platform's no-card mode; arms can be prepared but not run.")
        return 2
    print(f"all {len(results)} preflight checks pass")
    return 0


def commands(arm_ids=None):
    ids = arm_ids or ORDER
    print(f"export PYTHONPATH={PYTHONPATH}")
    print(f"PY={PY}")
    for aid in ids:
        a = ARMS[aid]
        print(f"\n# ---- {aid}: {a.name}  ({a.closes}) ----")
        print(f"# why: {a.why}")
        print(f"mkdir -p {a.root}")
        print(f"setsid nohup {a.argv()} >> {P}/{aid.lower()}.log 2>&1 &")
        if a.reader:
            print(f"# read: {a.reader}")


def launch(arm_id, dry):
    if arm_id not in ARMS:
        raise SystemExit(f"unknown arm {arm_id}; known {sorted(ARMS)}")
    a = ARMS[arm_id]
    if dry:
        print(f"[dry] would launch {arm_id}")
        print(a.argv())
        return 0
    rc = preflight([arm_id])
    if rc != 0:
        print("preflight failed; not launching")
        return rc
    cmd = (f"export PYTHONPATH={PYTHONPATH}\nPY={PY}\nmkdir -p {a.root}\n"
           f"setsid nohup {a.argv()} >> {P}/{arm_id.lower()}.log 2>&1 &\necho launched $!")
    rc, out, err = _run(cmd)
    print(out.strip() or err.strip()[:400])
    return 0 if rc == 0 else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands"); c.add_argument("arms", nargs="*")
    pf = sub.add_parser("preflight"); pf.add_argument("arms", nargs="*")
    lz = sub.add_parser("launch"); lz.add_argument("arm"); lz.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "commands":
        commands(a.arms or None); return 0
    if a.cmd == "preflight":
        return preflight(a.arms or None)
    return launch(a.arm, a.dry)


if __name__ == "__main__":
    sys.exit(main())
