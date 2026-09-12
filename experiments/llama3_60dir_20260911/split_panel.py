"""Freeze the P/S/V/H split (plan sections 5.2, 5.4, 6.1).

    P  1 per task per length, 16 long + 8 short guard   MR/native, operator parity
    S  6 per task per length (96 long), 16 short guard  60 approved + 4 controls
    V  8 per task per length (128 long), 32 short guard  top-2 from S + MR + YaRN
    H  16+ per task per length (256+ long), 64 guard    the one locked method + MR + YaRN + BM

Three rules are enforced here rather than left to the operator, because each one
is a mistake this campaign has already made:

1. **Split by SOURCE, never by row.**  A source document may appear in exactly
   one of P/S/V/H.  The archive's "180-row held-out" turned out to hold only 120
   unique prompts (60 duplicated pairs scored identically), and every
   significance calculation had to be redone.  Rows sharing a source are one
   observation, so splitting rows instead of sources leaks.

2. **Length is not a second sample.**  The same source at 16K and at 32K is not
   two independent observations (plan section 5.2).  So a source is assigned to
   one split AND one length, and the two lengths draw from disjoint sources.

3. **QA sources are grouped before filling.**  Natural QA items from the same
   passage behave like one draw; they are kept together.

The split is deterministic given `--seed`, and the manifest records a SHA256 per
stage so a later run can prove it used the frozen split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# plan section 5.2
STAGES = {
    "P": {"per_task_per_length": 1, "guard": 8},
    "S": {"per_task_per_length": 6, "guard": 16},
    "V": {"per_task_per_length": 8, "guard": 32},
    "H": {"per_task_per_length": 16, "guard": 64},
}
STAGE_ORDER = ("P", "S", "V", "H")


def _rows(path):
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def _src(r):
    for k in ("source_id", "group_id", "doc_id"):
        if r.get(k) is not None:
            return str(r[k])
    return str(r.get("row_id"))


def _task(r):
    return str(r.get("task", "?"))


def _length(r):
    for k in ("length_cap", "cap", "length"):
        if r.get(k) is not None:
            return int(r[k])
    return -1


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, help="candidate rows jsonl (all tasks/lengths)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--guard-length", type=int, default=4096,
                    help="the in-window guard length treated as the retention leg")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--tasks", default="", help="comma list; default = all present")
    ap.add_argument("--long-lengths", default="16384,32768")
    a = ap.parse_args(argv)

    rows = _rows(a.rows)
    if not rows:
        raise SystemExit("no rows")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = sorted({_task(r) for r in rows}) if not a.tasks else \
        [t.strip() for t in a.tasks.split(",") if t.strip()]
    long_lengths = [int(x) for x in a.long_lengths.split(",") if x.strip()]

    # ---- the grouping unit is the source; the length is an attribute of it ----
    by_task_len = defaultdict(lambda: defaultdict(set))
    for r in rows:
        by_task_len[_task(r)][_length(r)].add(_src(r))

    # a source must not appear under two lengths within a task, or rule 2 breaks
    dup = []
    for t, d in by_task_len.items():
        seen = {}
        for ln, srcs in d.items():
            for s in srcs:
                if s in seen and seen[s] != ln:
                    dup.append({"task": t, "source": s, "lengths": [seen[s], ln]})
                seen[s] = ln
    if dup:
        print(f"REFUSING: {len(dup)} source(s) appear at more than one length; "
              "the same passage at two lengths is not two independent samples "
              "(plan section 5.2)", file=sys.stderr)
        for d in dup[:5]:
            print(f"  {d}", file=sys.stderr)
        return 2

    rng = random.Random(a.seed)
    assigned = {}          # (task, length) -> stage -> [sources]
    counts = {}
    for t in tasks:
        for ln in long_lengths:
            pool = sorted(by_task_len[t].get(ln, set()))
            rng.shuffle(pool)
            take = {}
            i = 0
            for stage in STAGE_ORDER:
                need = STAGES[stage]["per_task_per_length"]
                take[stage] = pool[i:i + need]
                i += need
            counts[(t, ln)] = {"pool": len(pool), "taken": i,
                               **{s: len(v) for s, v in take.items()}}
            assigned[(t, ln)] = take

    short = []
    for t in tasks:
        pool = sorted(by_task_len[t].get(a.guard_length, set()))
        rng.shuffle(pool)
        i = 0
        take = {}
        for stage in STAGE_ORDER:
            need = STAGES[stage]["guard"]
            take[stage] = pool[i:i + need]
            i += need
        short.append({"task": t, "pool": len(pool), "taken": i,
                      **{s: len(v) for s, v in take.items()}})
        assigned[(t, a.guard_length)] = take

    # ---- verify the invariant that matters: a source is used exactly once ----
    # This is stronger than "no source in two stages": it also catches a source
    # reused across tasks or across lengths, which is the same independence
    # violation wearing different clothes.
    seen_at = {}
    violations = []
    for (t, ln), take in assigned.items():
        for stage, srcs in take.items():
            for s in srcs:
                if s in seen_at and seen_at[s] != (t, ln, stage):
                    violations.append({"source": s, "first": list(seen_at[s]),
                                       "again": [t, ln, stage]})
                seen_at[s] = (t, ln, stage)
    if violations:
        print(f"REFUSING: {len(violations)} source(s) used more than once.  A source is "
              "one observation; reusing it across a task, a length or a stage leaks "
              "information between them (plan sections 5.2, 6.1).", file=sys.stderr)
        for v in violations[:5]:
            print(f"  {v}", file=sys.stderr)
        return 2

    shortfalls = [c for c in counts.values() if c["S"] < STAGES["S"]["per_task_per_length"]]
    if shortfalls:
        print(f"REFUSING: {len(shortfalls)} task x length cell(s) cannot fill S "
              f"(need {STAGES['S']['per_task_per_length']} sources each)", file=sys.stderr)
        return 2

    # ---- write the stages ----
    index = {}
    for r in rows:
        index.setdefault((_task(r), _length(r), _src(r)), []).append(r)

    manifest = {"seed": a.seed, "tasks": tasks, "long_lengths": long_lengths,
                "guard_length": a.guard_length, "stages": {}, "cells": {},
                "rule": ("a source document appears in exactly one stage; splitting is by "
                         "source, not by row, because rows sharing a source are one "
                         "observation"),
                "short_guard": short}
    for stage in STAGE_ORDER:
        got = []
        for (t, ln), take in sorted(assigned.items()):
            for s in take[stage]:
                got.extend(index[(t, ln, s)])
        p = out / f"{stage}.jsonl"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in got), encoding="utf-8")
        manifest["stages"][stage] = {"rows": len(got), "path": str(p), "sha256": _sha(p)}
    manifest["cells"] = {f"{t}|{ln}": v for (t, ln), v in sorted(counts.items())}
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for stage in STAGE_ORDER:
        m = manifest["stages"][stage]
        print(f"  {stage}: {m['rows']:4d} rows  {m['sha256'][:12]}")
    print(f"guards (length {a.guard_length}): " +
          ", ".join(f"{s}:{sum(x[s] for x in short)}" for s in STAGE_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
