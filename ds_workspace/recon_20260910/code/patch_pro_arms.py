#!/usr/bin/env python3
"""Wire the Pro model's tables into the runners so they can be executed directly.

Run ON THE SERVER (the paths below are server paths):

    cd /root/autodl-tmp/phase1_20260910
    python patch_pro_arms.py

WHAT IT ADDS

  olmo_beta.py --pro-tables condEVQ          one arm on the 350-row RULER panel
  olmo_beta.py --pro-tables step42
  olmo_longnll.py  (arms condEVQ, step42)     the cheap continuous screen, ~25 s

The tables themselves come from `pro_tables_20260911.py`, which must sit in the
same directory; it builds them from the plan's closed forms and self-checks tau,
S, the plateau endpoints and nu-monotonicity.

WHY THESE TWO TABLES AND NOT A SWEEP

  step42   §9.3 of the plan.  m_j = 1[j >= 22], the UNIQUE step with S = 42 --
           the same budget as all four plateau members.  No previous step arm
           was budget-matched (step_hi25 had S = 39), so this is the first clean
           test of the step family at the winners' budget.  The plan is explicit
           that a loss here kills only this table, not all steps at all budgets.

  condEVQ  §8.  The conditional-EVQ reference: fix the winner's band [11,32] and
           both ends, solve the EVQ functional subject to unit mass AND mean
           1/2, and pin the scale by one geometric boundary condition.  The plan
           does NOT claim it is performance-optimal; it is offered as a
           zero-budget residual DIRECTION inside an already-validated table, to
           be explored with m(a) = m_win + a (m_condEVQ - m_win) once the B/D
           results are in.  Testing the endpoint is the cheap first look.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path("/root/autodl-tmp/phase1_20260910")
BETA = ROOT / "olmo_beta.py"
LONG = ROOT / "olmo_longnll.py"

LOADER = '''
def _pro_table(name, k=64):
    """Tables from the Pro model's RESEARCH_PLAN.md, via pro_tables_20260911."""
    import importlib.util as _iu
    _p = Path(__file__).resolve().parent / "pro_tables_20260911.py"
    _spec = _iu.spec_from_file_location("_pro_tables", _p)
    _m = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    if name == "condEVQ":
        return _m.cond_evq()["m"]
    if name == "step42":
        return _m.step42()
    raise KeyError(name)

'''

BETA_BLOCK = '''    if getattr(args, "pro_tables", "").strip():
        for nm in [s.strip() for s in args.pro_tables.split(",") if s.strip()]:
            m = np.asarray(_pro_table(nm), dtype=np.float64)
            print(json.dumps({"pro_table": nm, "sum_m": float(m.sum())}), flush=True)
            out.append(run_arm(f"pro_{nm}", m))
'''

LONG_BLOCK = '''        ("pro_condEVQ", _pro_table("condEVQ")),
        ("pro_step42", _pro_table("step42")),
'''


def _patch(path: pathlib.Path, edits, needs_loader=True):
    s = path.read_text()
    if any(new.split("\n")[0] in s for _, new in edits):
        print(f"  {path.name}: already patched")
        return True
    if needs_loader and "_pro_table(" not in s:
        anchor = "def build_arms("
        if anchor not in s:
            print(f"  REFUSING {path.name}: no build_arms anchor", file=sys.stderr)
            return False
        s = s.replace(anchor, LOADER + anchor, 1)
    for old, new in edits:
        if old not in s:
            print(f"  REFUSING {path.name}: anchor missing: {old[:60]!r}",
                  file=sys.stderr)
            return False
        s = s.replace(old, new, 1)
    path.write_text(s)
    print(f"  {path.name}: patched")
    return True


def main():
    ok = True

    # ---- the RULER runner ---------------------------------------------------
    s = BETA.read_text()
    if "--pro-tables" not in s:
        flag_anchor = '    ap.add_argument("--betas", default="")'
        if flag_anchor not in s:
            print("REFUSING: olmo_beta.py flag anchor missing", file=sys.stderr)
            return 2
        s = s.replace(flag_anchor,
                      '    ap.add_argument("--pro-tables", dest="pro_tables", default="",\n'
                      '                    help="condEVQ and/or step42 from the Pro model\'s\n'
                      '                         RESEARCH_PLAN.md, built by pro_tables_20260911.py")\n'
                      + flag_anchor, 1)
        BETA.write_text(s)
    ok &= _patch(BETA, [("    if args.c42:", BETA_BLOCK + "\n    if args.c42:")],
                 needs_loader=True)

    # ---- the cheap continuous screen ---------------------------------------
    s = LONG.read_text()
    anchor = '        ("rel_b4w", release(np.asarray(m_incr_beta(4.0, n=21, low=11), float))),'
    if "pro_condEVQ" not in s:
        if anchor not in s:
            print("REFUSING: olmo_longnll.py anchor missing", file=sys.stderr)
            return 2
        s = s.replace(anchor, anchor + "\n" + LONG_BLOCK, 1)
        LONG.write_text(s)
        print("  olmo_longnll.py: patched")
    ok &= _patch(LONG, [], needs_loader=True)

    for f in (BETA, LONG):
        try:
            compile(f.read_text(), str(f), "exec")
            print(f"  {f.name}: syntax OK")
        except SyntaxError as e:
            print(f"  {f.name}: SYNTAX ERROR {e}", file=sys.stderr)
            ok = False
    print("\npro arms wired:" if ok else "\nPROBLEMS ABOVE")
    if ok:
        print("  RULER :  python olmo_beta.py --root <dir> --model <M> --panel <P> "
              "--archive <A> --turns '' --pro-tables condEVQ,step42")
        print("  cheap :  python olmo_longnll.py --root <dir> --model <M> --nll-dir <D> "
              "--only pro_condEVQ,pro_step42,beta_b1_BM")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
