"""Apply the 2026-09-11 author correction to the 60-config review.

Reads directions.json (parsed from the plan) and priority_correction_20260911.json
(the author's verdict on all 20 directions, plus the execution order), and emits
the corrected arm sets:

    candidates_corrected.csv   60 rows, each with its tier and a run/hold/drop decision
    arms_llama3.csv            the 5 controls + 16 candidates that survive the cut
    execution_order.json       phase 1 (close existing theory) -> phase 2 (Llama) -> phase 3

The correction is a first-class input, not an edit to directions.json: the plan
stays exactly as written so the diff between what was proposed and what is being
run stays visible and auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# tier -> what happens to a policy
RUN_TIERS = {"core_mechanism", "mechanism_probe", "baseline_or_orthogonal"}


def load(corr_path, dirs_path):
    corr = json.loads(Path(corr_path).read_text(encoding="utf-8"))
    dirs = json.loads(Path(dirs_path).read_text(encoding="utf-8"))
    return corr, dirs


def validate(corr, dirs):
    problems = []
    ids = {d["id"] for d in dirs["directions"]}
    for did, v in corr["direction_verdicts"].items():
        if did not in ids:
            problems.append(f"{did}: verdict for an unknown direction")
            continue
        if v["tier"] not in RUN_TIERS | {"medium", "removed", "dont_run", "separate_line"}:
            problems.append(f"{did}: unknown tier {v['tier']}")
        for p in v.get("policies", []):
            if p not in ("a", "b", "c"):
                problems.append(f"{did}: bad policy {p}")
    missing = ids - set(corr["direction_verdicts"])
    if missing:
        problems.append(f"no verdict for {sorted(missing)}")
    # the Llama set must be a subset of what the correction says to run
    allowed = set()
    for did, v in corr["direction_verdicts"].items():
        if v["tier"] in RUN_TIERS:
            allowed |= {f"{did}{p}" for p in v.get("policies", ["a", "b", "c"])}
    stage = corr["llama3_stage"]
    stray = [c for c in stage["candidates"] if c not in allowed]
    if stray:
        problems.append(f"Llama arm set contains policies the correction does not run: {stray}")
    if len(stage["candidates"]) != stage["n_candidate_arms"]:
        problems.append("n_candidate_arms does not match the candidate list length")
    return problems, allowed


def decision_for(did, policy, verdict):
    tier = verdict["tier"]
    if tier in RUN_TIERS:
        pols = verdict.get("policies", ["a", "b", "c"])
        return ("RUN" if policy in pols else "HOLD"), tier
    return "DROP", tier


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--correction", type=Path, default=HERE / "priority_correction_20260911.json")
    ap.add_argument("--directions", type=Path, default=HERE / "directions.json")
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args(argv)

    corr, dirs = load(a.correction, a.directions)
    problems, allowed = validate(corr, dirs)
    if problems:
        print("REFUSING: the correction does not validate against the plan")
        for p in problems:
            print(f"  - {p}")
        return 2

    rows = []
    for d in dirs["directions"]:
        v = corr["direction_verdicts"][d["id"]]
        for cfg in d["configs"]:
            pol = cfg["id"][3]
            decision, tier = decision_for(d["id"], pol, v)
            rows.append({
                "config": cfg["id"],
                "direction": d["id"],
                "policy": cfg["policy"],
                "scope": d["scope"],
                "plan_priority": d["execution_priority"],
                "tier": tier,
                "decision": decision,
                "rationale": v.get("note", ""),
                "execution_authorized": "false",
            })
    with (a.out / "candidates_corrected.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    stage = corr["llama3_stage"]
    arms = []
    for c in stage["controls"]:
        arms.append({"arm": c, "role": "control", "config": "", "scope": ""})
    by_cfg = {r["config"]: r for r in rows}
    for c in stage["candidates"]:
        r = by_cfg[c]
        arms.append({"arm": c, "role": "candidate", "config": c, "scope": r["scope"],
                     "policy": r["policy"], "tier": r["tier"]})
    with (a.out / "arms_llama3.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "role", "config", "scope", "policy", "tier"])
        w.writeheader()
        for r in arms:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    order = {
        "correction": corr["headline"],
        "required_fixes": corr["required_fixes"],
        "phase_1_close_existing_theory": corr["execution_order"]["phase_1_close_existing_theory"],
        "phase_2_llama3": {
            "note": corr["execution_order"]["phase_2_llama16"],
            "controls": stage["controls"],
            "candidates": stage["candidates"],
            "n_arms": len(stage["controls"]) + len(stage["candidates"]),
            "questions_answered": stage["questions_answered"],
        },
        "phase_3": corr["execution_order"]["phase_3"],
        "failure_narrative_if_all_negative": corr["failure_narrative_if_all_negative"],
    }
    (a.out / "execution_order.json").write_text(
        json.dumps(order, indent=2, ensure_ascii=False), encoding="utf-8")

    n_run = sum(1 for r in rows if r["decision"] == "RUN")
    n_hold = sum(1 for r in rows if r["decision"] == "HOLD")
    n_drop = sum(1 for r in rows if r["decision"] == "DROP")
    print(f"60 configs -> RUN {n_run} / HOLD {n_hold} / DROP {n_drop}")
    print(f"Llama-3-8B stage: {len(stage['controls'])} controls + "
          f"{len(stage['candidates'])} candidates = "
          f"{len(stage['controls']) + len(stage['candidates'])} arms")
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], 0)
        by_tier[r["tier"]] += 1
    for t, n in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        print(f"  {t:26s} {n:2d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
