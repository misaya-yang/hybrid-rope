"""Parse the review plan's section 4 into machine-readable form.

Hand-transcribing 20 direction cards would introduce exactly the kind of
silent divergence the review is trying to eliminate.  So the cards are parsed
from the source document and the result is committed; `--check` re-parses and
fails if the committed JSON has drifted from the plan.

Source of truth:
    LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md  (section 4)

Outputs:
    directions.json   20 cards: scope, evidence grade, priority, construction,
                      the three policies, fail criteria, transfer rule, refs
    candidates.csv    60 rows, execution_authorized=false (plan section 0.1)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_PLAN = Path(
    "/Users/yang/Downloads/LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md"
)

HEAD_RE = re.compile(r"^###\s+(D\d\d)\s*[　\s]+(.+?)\s*$")
SCOPE_RE = re.compile(r"\*\*范围：\*\*\s*`([^`]+)`")
GRADE_RE = re.compile(r"\*\*依据等级：\*\*\s*([^；;]+)")
PRIO_RE = re.compile(r"\*\*执行优先级：\*\*\s*(\d+)")
# The middle cell may itself contain a pipe (D19b's policy is "phi=Theta-|(z mod
# 2Theta)-Theta|"), so the delimiter is taken to be the LAST pipe on the line
# rather than the first -- hence the greedy middle group.
CONFIG_ROW_RE = re.compile(r"^\|\s*(D\d\d[a-c])\s*\|(.*)\|([^|]*)\|\s*$")
FIELD_RE = re.compile(r"^\*\*(.+?)。?\*\*\s*(.*)$")


def _fence_after(lines, start):
    """Return the first ```text fence at or after `start`."""
    i = start
    while i < len(lines):
        if lines[i].strip().startswith("```text"):
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                body.append(lines[j])
                j += 1
            return "\n".join(body).strip(), j
        i += 1
    return None, start


def parse(plan_path: Path):
    lines = plan_path.read_text(encoding="utf-8").splitlines()
    heads = [(i, m) for i, m in ((i, HEAD_RE.match(l)) for i, l in enumerate(lines)) if m]

    cards = []
    for n, (i, m) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        block = lines[i:end]
        text = "\n".join(block)

        did, title = m.group(1), m.group(2)
        scope = SCOPE_RE.search(text)
        grade = GRADE_RE.search(text)
        prio = PRIO_RE.search(text)
        if not (scope and grade and prio):
            raise ValueError(f"{did}: could not parse scope/grade/priority")

        construction, fence_end = _fence_after(block, 0)
        if construction is None:
            raise ValueError(f"{did}: no ```text construction block")

        configs = []
        for ln in block[fence_end:]:
            cm = CONFIG_ROW_RE.match(ln)
            if cm:
                configs.append({
                    "id": cm.group(1),
                    "policy": cm.group(2).strip(),
                    "why": cm.group(3).strip(),
                })
        if len(configs) != 3:
            raise ValueError(f"{did}: expected 3 configs, got {len(configs)}")

        # free-text fields, keyed by their bold lead-in
        fields = {}
        for ln in block:
            fm = FIELD_RE.match(ln.strip())
            if fm:
                fields.setdefault(fm.group(1).strip(), fm.group(2).strip())

        cards.append({
            "id": did,
            "title": title,
            "scope": scope.group(1).strip(),
            "evidence_grade": grade.group(1).strip(),
            "execution_priority": int(prio.group(1)),
            "question": fields.get("从RoPE出发的问题", ""),
            "construction": construction,
            "guarantee": fields.get("能严格保证什么", ""),
            "gap": fields.get("从性质到任务还缺什么", ""),
            "configs": configs,
            "must_compare": fields.get("必须比较", ""),
            "fail_scope": fields.get("失败的可判范围", ""),
            "key_risk": fields.get("关键风险", ""),
            "transfer_rule": fields.get("未来无调参迁移规则", ""),
            "refs": fields.get("依据", ""),
            "history_dedup": fields.get("历史去重", ""),
        })

    if len(cards) != 20:
        raise ValueError(f"expected 20 directions, parsed {len(cards)}")
    for c in cards:
        for cfg in c["configs"]:
            if not cfg["id"].startswith(c["id"]):
                raise ValueError(f"{cfg['id']} does not belong to {c['id']}")
    return cards


def write_outputs(cards, out_dir: Path):
    doc = {
        "source": "LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md section 4",
        "note": ("parsed mechanically from the review plan; prose fields are verbatim "
                 "fragments, they are metadata for the executor, not executable spec"),
        "n_directions": len(cards),
        "n_configs": sum(len(c["configs"]) for c in cards),
        "execution_authorized_default": False,
        "directions": cards,
    }
    (out_dir / "directions.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "direction", "scope", "evidence_grade",
                    "execution_priority", "policy", "execution_authorized"])
        for c in cards:
            for cfg in c["configs"]:
                w.writerow([cfg["id"], c["id"], c["scope"], c["evidence_grade"],
                            c["execution_priority"], cfg["policy"], "false"])
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--check", action="store_true",
                    help="re-parse only; fail if the committed JSON has drifted")
    a = ap.parse_args(argv)

    if not a.plan.exists():
        raise SystemExit(f"plan not found: {a.plan}")
    cards = parse(a.plan)

    if a.check:
        cur = json.loads((a.out / "directions.json").read_text(encoding="utf-8"))
        if cur["directions"] != cards:
            print("DRIFT: directions.json no longer matches the plan")
            return 1
        print(f"ok: {len(cards)} directions, {sum(len(c['configs']) for c in cards)} configs, no drift")
        return 0

    doc = write_outputs(cards, a.out)
    print(f"wrote {doc['n_directions']} directions / {doc['n_configs']} configs")
    by_scope = {}
    for c in cards:
        by_scope.setdefault(c["scope"], 0)
        by_scope[c["scope"]] += len(c["configs"])
    for s, n in sorted(by_scope.items(), key=lambda kv: -kv[1]):
        print(f"  {s:24s} {n:2d} configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
