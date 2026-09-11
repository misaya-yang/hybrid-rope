#!/usr/bin/env python3
"""Complete (and idempotently repair) the Pro-table wiring in both runners.

The first attempt left `olmo_beta.py` with a dangling `--pro-tables` flag and
`olmo_longnll.py` with arms that referenced a `_pro_table` helper that was never
inserted.  This finishes the job and then PROVES it by actually instantiating the
arms and running the RULER runner's --dry-run.

Run on the server:  cd /root/autodl-tmp/phase1_20260910 && python repair_pro_arms.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-tmp/phase1_20260910")

LOADER = '''

def _pro_table(name, k=64):
    """Tables from the Pro model's RESEARCH_PLAN.md, via pro_tables_20260911.py.

    Kept as a lazy import so neither runner gains a hard dependency on the file
    at import time; the failure mode is then a clear NameError naming the table
    rather than an import error at module load.
    """
    import importlib.util as _iu
    _p = Path(__file__).resolve().parent / "pro_tables_20260911.py"
    if not _p.exists():
        raise FileNotFoundError(f"{_p} is missing; copy it next to this runner")
    _spec = _iu.spec_from_file_location("_pro_tables", _p)
    _m = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    if name == "condEVQ":
        return _m.cond_evq()["m"]
    if name == "step42":
        return _m.step42()
    raise KeyError(f"unknown pro table {name!r}; known: condEVQ, step42")

'''

BETA_BLOCK = '''    if getattr(args, "pro_tables", "").strip():
        # THE PRO MODEL'S TABLES (RESEARCH_PLAN.md §8 and §9.3).
        for nm in [s.strip() for s in args.pro_tables.split(",") if s.strip()]:
            m = np.asarray(_pro_table(nm), dtype=np.float64)
            print(json.dumps({"pro_table": nm, "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(f"pro_{nm}", m))
'''


def insert_loader(path: pathlib.Path) -> bool:
    s = path.read_text()
    if "def _pro_table(" in s:
        print(f"  {path.name}: loader already present")
        return True
    anchor = "def build_arms("
    if anchor not in s:
        print(f"  {path.name}: NO build_arms anchor", file=sys.stderr)
        return False
    path.write_text(s.replace(anchor, LOADER.lstrip("\n") + "\n" + anchor, 1))
    print(f"  {path.name}: loader inserted")
    return True


def insert_beta_block(path: pathlib.Path) -> bool:
    s = path.read_text()
    if "pro_tables" in s and "_pro_table(nm)" in s:
        print(f"  {path.name}: block already present")
        return True
    for anchor in ('    if args.c42:', '    if args.betas.strip():'):
        if anchor in s:
            path.write_text(s.replace(anchor, BETA_BLOCK + "\n" + anchor, 1))
            print(f"  {path.name}: block inserted before {anchor.strip()!r}")
            return True
    print(f"  {path.name}: no insertion anchor found", file=sys.stderr)
    return False


def main():
    beta = ROOT / "olmo_beta.py"
    long = ROOT / "olmo_longnll.py"
    ok = True
    ok &= insert_loader(beta)
    ok &= insert_loader(long)
    ok &= insert_beta_block(beta)

    print("\n--- syntax ---")
    for f in (beta, long):
        r = subprocess.run([sys.executable, "-c",
                            f"import ast;ast.parse(open({str(f)!r}).read());print('OK')"],
                           capture_output=True, text=True)
        print(f"  {f.name}: {(r.stdout or r.stderr).strip()}")
        ok &= r.returncode == 0

    print("\n--- instantiate the pro arms in olmo_longnll ---")
    code = (
        "import sys; sys.path.insert(0,'/root/autodl-tmp/nongeometric_screen_20260909/code');"
        "sys.path.insert(0,'/root/autodl-tmp/phase1_20260910');"
        "import importlib.util, numpy as np;"
        "spec=importlib.util.spec_from_file_location('l','/root/autodl-tmp/phase1_20260910/olmo_longnll.py');"
        "l=importlib.util.module_from_spec(spec); spec.loader.exec_module(l);"
        "[print('  ',nm,np.asarray(m,float).sum()) for nm,m in l.build_arms() if nm.startswith('pro_')]"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    ok &= r.returncode == 0

    print("\n--- olmo_beta --dry-run with --pro-tables ---")
    cmd = ("cd /root/autodl-tmp/phase1_20260910 && "
           "PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910 "
           f"{sys.executable} olmo_beta.py "
           "--root /tmp/prodry --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct "
           "--panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl "
           "--archive /root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01 "
           "--betas '' --turns '' --pro-tables step42 --dry-run")
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join("  " + x for x in tail[-6:]))
    ok &= r.returncode == 0

    print("\n" + ("ALL WIRED AND VERIFIED" if ok else "**STILL BROKEN**"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
