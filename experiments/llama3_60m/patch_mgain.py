"""Add `--m-gain` to olmo_beta.py so a table AND a gain can be given together.

WHY THIS IS NEEDED
------------------
Plan B section 5 / E-C needs a scale x gain 2x2 on a FIXED normalised profile:

    C00  omega*4^-r   g4      C10  omega*8^-r   g4
    C01  omega*4^-r   g8      C11  omega*8^-r   g8

plus two existing-method controls (CY8 = official YaRN at s=8, CM8 = MrRoPE-Pro
at s=8), both at g8.

The runner cannot express those.  `run_arm(name, m, gain=...)` does take a gain,
but the `--m-file` branch calls it WITHOUT one, and the `--gains` branch builds
its own tables from a hardcoded name list, so `--m-file` and `--gains` are
mutually exclusive.  Without this patch the four cells collapse to two and the
scale x gain interaction -- the thing E-C exists to measure -- is unmeasurable.

The patch is idempotent, validates its anchors before writing, and restores the
original on any failure.  A previous attempt at an m-file patch silently broke
the runner; this one refuses to write unless it finds exactly what it expects.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

# The --m-file call SPANS TWO LINES (the help string is a continuation).  An
# anchor on the first line alone inserts between the call and its continuation
# and produces a syntax error -- which the compile self-check below caught and
# refused to write.  Match the whole call.
ANCHOR_ARG = ('    ap.add_argument("--m-file", dest="m_file", default="",\n'
              '                    help="path.json:armname[;...] literal 64-slot m arrays")')
ANCHOR_BLOCK = "    if args.m_file.strip():"
ANCHOR_RUN = "            out.append(run_arm(_nm.strip(), _m))"


def _validate_patched_source(src: str):
    """Require the gain branch to live inside the --m-file branch.

    A compile-only check is insufficient: the first patch version compiled but
    inserted the new code in the unrelated ``--turns`` loop, where ``_nm`` and
    ``_m`` are stale or undefined.  Validate the relevant AST subtree instead.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return False, f"source does not compile ({exc})"

    main = next((n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == "main"), None)
    if main is None:
        return False, "main() not found"

    target = None
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        test = ast.get_source_segment(src, node.test) or ""
        if test == "args.m_file.strip()":
            if target is not None:
                return False, "multiple args.m_file branches found"
            target = node
    if target is None:
        return False, "args.m_file branch not found"

    segment = ast.get_source_segment(src, target) or ""
    if "args.m_gain" not in segment:
        return False, "args.m_gain is not used inside the args.m_file branch"
    gained_calls = [
        node for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_arm"
        and any(keyword.arg == "gain" for keyword in node.keywords)
    ]
    if not gained_calls:
        return False, "no run_arm(..., gain=...) call inside the args.m_file branch"
    return True, "semantic placement validated"


def patch(path: Path, dry=False):
    src = path.read_text(encoding="utf-8")
    if "--m-gain" in src:
        ok, detail = _validate_patched_source(src)
        return (("already patched and validated" if ok else
                 f"REFUSING: existing --m-gain patch is invalid: {detail}"),
                0 if ok else 2)

    if src.count(ANCHOR_ARG) != 1:
        return (f"REFUSING: expected exactly 1 occurrence of the --m-file argument line, "
                f"found {src.count(ANCHOR_ARG)}", 2)
    if src.count(ANCHOR_BLOCK) != 1:
        return (f"REFUSING: expected exactly 1 '{ANCHOR_BLOCK.strip()}' block, "
                f"found {src.count(ANCHOR_BLOCK)}", 2)
    if src.count(ANCHOR_RUN) != 1:
        return (f"REFUSING: expected exactly 1 m-file run_arm call, "
                f"found {src.count(ANCHOR_RUN)}", 2)

    # 1. the new option, placed right after --m-file
    new_arg = (ANCHOR_ARG + "\n"
               '    ap.add_argument("--m-gain", dest="m_gain", default="",\n'
               '                    help="gain applied to every --m-file arm; E-C needs a '
               'table AND a gain together")')
    src2 = src.replace(ANCHOR_ARG, new_arg, 1)

    # 2. carry the gain into the exact run_arm call inside the m-file loop.
    # Do not regex across arbitrary blocks: an earlier non-greedy regex still
    # matched the next run_arm call, in the --turns loop, because the m-file
    # call was split differently than expected.
    indent = ANCHOR_RUN[:len(ANCHOR_RUN) - len(ANCHOR_RUN.lstrip())]
    new_call = (
        f"{indent}_mg = [s.strip() for s in (args.m_gain or '').split(',') if s.strip()]\n"
        f"{indent}if _mg:\n"
        f"{indent}    for _g in _mg:\n"
        f"{indent}        _gg = float(_g)\n"
        f"{indent}        print(json.dumps({{\"m_gain\": _gg, \"arm\": _nm.strip()}}), flush=True)\n"
        f"{indent}        out.append(run_arm(f\"{{_nm.strip()}}_g{{_g}}\".replace('.', 'p'), "
        f"_m, gain=_gg))\n"
        f"{indent}else:\n"
        f"{indent}    {ANCHOR_RUN.strip()}")
    src3 = src2.replace(ANCHOR_RUN, new_call, 1)

    # 3. structure self-check before writing
    for needle in ("--m-gain", "_mg = [s.strip()", "run_arm(f\"{_nm.strip()}"):
        if needle not in src3:
            return f"REFUSING: post-check failed, {needle!r} missing", 2
    ok, detail = _validate_patched_source(src3)
    if not ok:
        return f"REFUSING: patched source failed semantic validation: {detail}", 2

    if dry:
        return "dry run: patch would apply cleanly", 0
    shutil.copy2(path, path.with_suffix(path.suffix + ".pre_mgain.bak"))
    path.write_text(src3, encoding="utf-8")
    return f"patched {path} (backup: {path.name}.pre_mgain.bak)", 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    msg, rc = patch(Path(a.runner), dry=a.dry_run)
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
