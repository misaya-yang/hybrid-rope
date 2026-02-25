#!/usr/bin/env python3
"""
Audit LongBench pipeline parity between local and reference outputs.

Supports two common input formats:
1) eval_longbench.py output (`models -> <model_alias> -> tasks -> task -> score_pct/score_raw/score`)
2) official LongBench result.json (`task -> score`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_from_task_obj(task_obj: Dict, prefer_pct: bool) -> Optional[float]:
    if not isinstance(task_obj, dict):
        return None
    if prefer_pct and isinstance(task_obj.get("score_pct"), (int, float)):
        return float(task_obj["score_pct"])
    if isinstance(task_obj.get("score"), (int, float)):
        return float(task_obj["score"])
    if isinstance(task_obj.get("score_pct"), (int, float)):
        return float(task_obj["score_pct"])
    if isinstance(task_obj.get("score_raw"), (int, float)):
        return float(task_obj["score_raw"]) * (100.0 if prefer_pct else 1.0)
    return None


def extract_scores(obj: Dict, model_alias: str, prefer_pct: bool) -> Dict[str, float]:
    # Format A: eval_longbench output
    if isinstance(obj.get("models"), dict):
        models = obj["models"]
        key = model_alias if model_alias in models else None
        if key is None:
            for fallback in ("hybrid_lora", "base_unfinetuned"):
                if fallback in models:
                    key = fallback
                    break
        if key is None:
            key = next(iter(models.keys()), None)
        if key:
            tasks = models.get(key, {}).get("tasks", {})
            out: Dict[str, float] = {}
            if isinstance(tasks, dict):
                for task, task_obj in tasks.items():
                    score = _score_from_task_obj(task_obj, prefer_pct=prefer_pct)
                    if isinstance(score, float):
                        out[str(task)] = float(score)
            return out

    # Format B: official result.json style task->score or task->dict
    out: Dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
        elif isinstance(v, dict):
            score = _score_from_task_obj(v, prefer_pct=prefer_pct)
            if isinstance(score, float):
                out[str(k)] = float(score)
    return out


def rank_order(scores: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def overlap_tasks(a: Dict[str, float], b: Dict[str, float], task_filter: List[str]) -> List[str]:
    both = sorted(set(a.keys()) & set(b.keys()))
    if task_filter:
        both = [t for t in both if t in set(task_filter)]
    return both


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit LongBench parity between local and reference outputs.")
    ap.add_argument("--candidate_json", type=str, required=True)
    ap.add_argument("--reference_json", type=str, required=True)
    ap.add_argument("--candidate_model_alias", type=str, default="hybrid_lora")
    ap.add_argument("--reference_model_alias", type=str, default="")
    ap.add_argument("--tasks", type=str, default="")
    ap.add_argument("--prefer_pct", action="store_true", help="Interpret scores in pct(0-100) when possible.")
    ap.add_argument("--tolerance_abs", type=float, default=1.0, help="Absolute tolerance on pct scale.")
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, required=True)
    args = ap.parse_args()

    candidate = load_json(Path(args.candidate_json))
    reference = load_json(Path(args.reference_json))

    cand_scores = extract_scores(candidate, model_alias=args.candidate_model_alias, prefer_pct=bool(args.prefer_pct))
    ref_alias = args.reference_model_alias or args.candidate_model_alias
    ref_scores = extract_scores(reference, model_alias=ref_alias, prefer_pct=bool(args.prefer_pct))

    tasks = overlap_tasks(cand_scores, ref_scores, parse_csv(args.tasks))
    if not tasks:
        raise RuntimeError("No overlapping tasks found between candidate and reference outputs.")

    rows = []
    within_tol = 0
    for t in tasks:
        c = float(cand_scores[t])
        r = float(ref_scores[t])
        diff = c - r
        ok = abs(diff) <= float(args.tolerance_abs)
        if ok:
            within_tol += 1
        rows.append(
            {
                "task": t,
                "candidate": c,
                "reference": r,
                "delta": diff,
                "abs_delta": abs(diff),
                "within_tolerance": bool(ok),
            }
        )

    cand_rank = rank_order({t: cand_scores[t] for t in tasks})
    ref_rank = rank_order({t: ref_scores[t] for t in tasks})
    rank_consistent = cand_rank == ref_rank

    result = {
        "meta": {
            "candidate_json": str(Path(args.candidate_json).resolve()),
            "reference_json": str(Path(args.reference_json).resolve()),
            "candidate_model_alias": args.candidate_model_alias,
            "reference_model_alias": ref_alias,
            "prefer_pct": bool(args.prefer_pct),
            "tolerance_abs": float(args.tolerance_abs),
            "n_tasks": len(tasks),
        },
        "summary": {
            "within_tolerance_ratio": float(within_tol / max(1, len(tasks))),
            "within_tolerance_count": int(within_tol),
            "rank_consistent": bool(rank_consistent),
            "max_abs_delta": float(max(x["abs_delta"] for x in rows)),
        },
        "task_rows": rows,
        "candidate_rank": cand_rank,
        "reference_rank": ref_rank,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# LongBench Pipeline Parity Audit",
        "",
        f"- Candidate: `{result['meta']['candidate_json']}`",
        f"- Reference: `{result['meta']['reference_json']}`",
        f"- Tasks compared: **{len(tasks)}**",
        f"- Tolerance (abs): **{float(args.tolerance_abs):.3f}**",
        "",
        "## Summary",
        "",
        f"- Within tolerance: **{within_tol}/{len(tasks)}**",
        f"- Max abs delta: **{result['summary']['max_abs_delta']:.4f}**",
        f"- Ranking consistent: **{rank_consistent}**",
        "",
        "## Per-task Diff",
        "",
        "| task | candidate | reference | delta | abs_delta | within_tolerance |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            "| {task} | {candidate:.4f} | {reference:.4f} | {delta:.4f} | {abs_delta:.4f} | {within_tolerance} |".format(
                **row
            )
        )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
