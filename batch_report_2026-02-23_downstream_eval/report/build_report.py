#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


METHODS = ["baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid"]
PROFILES = ["downstream_eval_autorun", "downstream_eval_parallel_seed42_m2"]


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_floats(values: Iterable) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def niah_mean(niah_obj: Optional[dict]) -> Optional[float]:
    if not niah_obj:
        return None
    matrix = niah_obj.get("accuracy_matrix", {})
    vals: List[float] = []
    if isinstance(matrix, dict):
        for row in matrix.values():
            if isinstance(row, dict):
                vals.extend(flatten_floats(row.values()))
    if not vals:
        return None
    return sum(vals) / len(vals)


def longbench_scores(longbench_obj: Optional[dict]) -> Dict[str, float]:
    if not longbench_obj:
        return {}
    tasks = (
        longbench_obj.get("models", {})
        .get("hybrid_lora", {})
        .get("tasks", {})
    )
    out: Dict[str, float] = {}
    if isinstance(tasks, dict):
        for task, payload in tasks.items():
            if isinstance(payload, dict) and payload.get("score") is not None:
                try:
                    out[str(task)] = float(payload["score"])
                except Exception:
                    pass
    return out


def longbench_mean(longbench_obj: Optional[dict]) -> Optional[float]:
    scores = longbench_scores(longbench_obj)
    if not scores:
        return None
    vals = list(scores.values())
    return sum(vals) / len(vals)


def passkey_16k(passkey_obj: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not passkey_obj:
        return None, None
    by_length = passkey_obj.get("by_length", {})
    if not isinstance(by_length, dict):
        return None, None
    payload = by_length.get("16384", {})
    if not isinstance(payload, dict):
        return None, None
    acc: Optional[float] = None
    margin: Optional[float] = None
    try:
        if payload.get("tf_accuracy") is not None:
            acc = float(payload["tf_accuracy"])
    except Exception:
        pass
    try:
        if payload.get("margin_mean") is not None:
            margin = float(payload["margin_mean"])
    except Exception:
        pass
    return acc, margin


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(num_bytes)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(v)} {u}"
            return f"{v:.2f} {u}"
        v /= 1024
    return f"{num_bytes} B"


@dataclass
class Completion:
    niah: bool
    longbench: bool
    passkey: bool

    @property
    def coverage(self) -> int:
        return int(self.niah) + int(self.longbench) + int(self.passkey)


def main() -> None:
    report_dir = Path(__file__).resolve().parent
    bundle_root = report_dir.parent
    raw_results_root = bundle_root / "data" / "raw" / "results"
    run_dirs = sorted([p for p in raw_results_root.glob("llama8b_fair_v2_longbench_stable_*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit("No run directory found under data/raw/results")
    run_root = run_dirs[-1]

    completion_rows: List[dict] = []
    downstream_rows: List[dict] = []
    longbench_rows: List[dict] = []
    method_summary_rows: List[dict] = []

    # Method training summaries.
    for method in METHODS:
        summary_path = run_root / method / "summary.json"
        summary_obj = read_json(summary_path)
        if not summary_obj:
            continue
        method_summary_rows.append(
            {
                "method": method,
                "summary_json": str(summary_path.relative_to(bundle_root)),
                "train_loss": summary_obj.get("train_metrics", {}).get("train_loss"),
                "eval_loss": summary_obj.get("eval_metrics", {}).get("eval_loss"),
                "ppl_16k": summary_obj.get("ppl_results", {}).get("16384", {}).get("ppl"),
                "passkey_gen_16k": summary_obj.get("passkey_results", {}).get("16384", {}).get("accuracy"),
            }
        )

    # Downstream by profile and method.
    completion: Dict[Tuple[str, str], Completion] = {}
    for profile in PROFILES:
        profile_root = run_root / profile
        if not profile_root.exists():
            continue
        for method in METHODS:
            niah_path = profile_root / "niah" / method / "niah_recall_results.json"
            longbench_path = profile_root / "longbench" / f"{method}.json"
            passkey_path = profile_root / "passkey_tf" / method / "passkey_tf_summary.json"

            has_niah = niah_path.exists()
            has_longbench = longbench_path.exists()
            has_passkey = passkey_path.exists()
            comp = Completion(has_niah, has_longbench, has_passkey)
            completion[(profile, method)] = comp

            completion_rows.append(
                {
                    "profile": profile,
                    "method": method,
                    "niah": "Y" if has_niah else "N",
                    "longbench": "Y" if has_longbench else "N",
                    "passkey_tf": "Y" if has_passkey else "N",
                    "coverage": comp.coverage,
                }
            )

            niah_obj = read_json(niah_path) if has_niah else None
            longbench_obj = read_json(longbench_path) if has_longbench else None
            passkey_obj = read_json(passkey_path) if has_passkey else None
            pk_acc_16k, pk_margin_16k = passkey_16k(passkey_obj)
            lb_scores = longbench_scores(longbench_obj)

            downstream_rows.append(
                {
                    "profile": profile,
                    "method": method,
                    "niah_mean": niah_mean(niah_obj),
                    "longbench_avg": longbench_mean(longbench_obj),
                    "passkey_tf_16k": pk_acc_16k,
                    "passkey_margin_16k": pk_margin_16k,
                    "longbench_tasks": len(lb_scores),
                }
            )
            for task, score in sorted(lb_scores.items()):
                longbench_rows.append(
                    {
                        "profile": profile,
                        "method": method,
                        "task": task,
                        "score": score,
                    }
                )

    # Best-available consolidated method metrics.
    consolidated_rows: List[dict] = []
    profile_priority = {name: i for i, name in enumerate(PROFILES)}
    downstream_by_key = {(r["profile"], r["method"]): r for r in downstream_rows}
    for method in METHODS:
        candidates = []
        for profile in PROFILES:
            comp = completion.get((profile, method), Completion(False, False, False))
            candidates.append((comp.coverage, -profile_priority.get(profile, 99), profile))
        candidates.sort(reverse=True)
        best_profile = candidates[0][2]
        comp = completion.get((best_profile, method), Completion(False, False, False))
        row = downstream_by_key.get((best_profile, method), {})
        consolidated_rows.append(
            {
                "method": method,
                "source_profile": best_profile,
                "coverage": comp.coverage,
                "niah_mean": row.get("niah_mean"),
                "longbench_avg": row.get("longbench_avg"),
                "passkey_tf_16k": row.get("passkey_tf_16k"),
                "passkey_margin_16k": row.get("passkey_margin_16k"),
            }
        )

    # Inventory.
    all_files = list((bundle_root / "data" / "raw").rglob("*"))
    all_files = [p for p in all_files if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in all_files)

    # Write CSV outputs.
    def write_csv(path: Path, rows: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(report_dir / "completion_matrix.csv", completion_rows)
    write_csv(report_dir / "method_summary_metrics.csv", method_summary_rows)
    write_csv(report_dir / "downstream_metrics_by_profile.csv", downstream_rows)
    write_csv(report_dir / "longbench_task_scores.csv", longbench_rows)
    write_csv(report_dir / "method_metrics_best_available.csv", consolidated_rows)

    # Build markdown report.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# 批次实验整理汇报（2026-02-23）")
    lines.append("")
    lines.append(f"- 生成时间: `{ts}`")
    lines.append(f"- 本地归档目录: `{bundle_root}`")
    lines.append(f"- 运行根目录: `{run_root.relative_to(bundle_root)}`")
    lines.append(f"- 已归档文件数: `{len(all_files)}`")
    lines.append(f"- 已归档体积: `{human_size(total_bytes)}`")
    lines.append("")
    lines.append("## 1) 归档内容")
    lines.append("")
    lines.append("- 数据压缩包: `data/llama8b_batch_20260223_dataonly.tgz`")
    lines.append("- 解压数据: `data/raw/results/llama8b_fair_v2_longbench_stable_20260223_0150/`")
    lines.append("- 实时状态快照: `logs/remote_status_snapshot_clean.txt`")
    lines.append("- 自动汇总表: `report/*.csv`")
    lines.append("")
    lines.append("## 2) 完成矩阵（按 profile）")
    lines.append("")
    lines.append("| profile | method | NIAH | LongBench | Passkey-TF | coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in completion_rows:
        lines.append(
            f"| {r['profile']} | {r['method']} | {r['niah']} | {r['longbench']} | {r['passkey_tf']} | {r['coverage']} |"
        )
    lines.append("")
    lines.append("## 3) 方法级汇总（best available source）")
    lines.append("")
    lines.append("| method | source_profile | coverage | niah_mean | longbench_avg | passkey_tf@16k | passkey_margin@16k |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in consolidated_rows:
        lines.append(
            "| {method} | {source_profile} | {coverage} | {niah} | {lb} | {pk} | {pm} |".format(
                method=r["method"],
                source_profile=r["source_profile"],
                coverage=r["coverage"],
                niah=f"{r['niah_mean']:.4f}" if isinstance(r["niah_mean"], float) else "NA",
                lb=f"{r['longbench_avg']:.4f}" if isinstance(r["longbench_avg"], float) else "NA",
                pk=f"{r['passkey_tf_16k']:.4f}" if isinstance(r["passkey_tf_16k"], float) else "NA",
                pm=f"{r['passkey_margin_16k']:.4f}" if isinstance(r["passkey_margin_16k"], float) else "NA",
            )
        )
    lines.append("")
    lines.append("## 4) 说明")
    lines.append("")
    lines.append("- 本批次已按“数据优先、权重剔除”方式回收，便于后续论文统计与复核。")
    lines.append("- 若需要完整可复现实验镜像（含 checkpoint/adapter 权重），建议额外单独归档。")
    lines.append("- 当前远端仍可能存在 autopilot 补跑任务；可继续增量同步 `orchestrator.log` 与新增 `*.json`。")
    lines.append("")
    lines.append("## 5) 关键文件")
    lines.append("")
    lines.append("- `report/completion_matrix.csv`")
    lines.append("- `report/method_summary_metrics.csv`")
    lines.append("- `report/downstream_metrics_by_profile.csv`")
    lines.append("- `report/method_metrics_best_available.csv`")
    lines.append("- `report/longbench_task_scores.csv`")
    lines.append("")

    (report_dir / "BATCH_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
