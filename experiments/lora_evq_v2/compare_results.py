#!/usr/bin/env python3
"""
对比 Base Instruct vs EVQ-Cosh LoRA 评测结果
=============================================
读取两个 eval JSON，输出对比表格和论文可用的 LaTeX 片段。

Usage:
    python compare_results.py \
        --base_result results/eval_base_instruct.json \
        --evq_result results/eval_evq_lora.json \
        --output results/comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_pct(base_val: float, evq_val: float) -> str:
    """Format percentage change."""
    if base_val == 0:
        return "N/A"
    pct = (evq_val - base_val) / abs(base_val) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def compare(base: dict, evq: dict) -> dict:
    """Generate comparison report."""
    report = {"comparisons": [], "summary": {}}

    print("\n" + "=" * 70)
    print("BASE INSTRUCT vs EVQ-COSH LORA  COMPARISON")
    print("=" * 70)

    # --- PPL ---
    if "ppl" in base and "ppl" in evq:
        print("\n📊 Perplexity (lower is better):")
        print(f"  {'Metric':<15s} {'Base':>10s} {'EVQ':>10s} {'Δ':>10s} {'Status':>8s}")
        print("  " + "-" * 55)
        for key in sorted(base["ppl"].keys()):
            if not key.startswith("ppl@"):
                continue
            b = base["ppl"].get(key, 0)
            e = evq["ppl"].get(key, 0)
            # For PPL, lower is better → negative change is good
            delta = fmt_pct(b, e)
            # At short context: EVQ should not be much worse (<20%)
            # At long context: EVQ should be MUCH better
            ctx = int(key.split("@")[1].replace("K", ""))
            if ctx <= 8:
                status = "✅" if e < b * 1.2 else "⚠️"
            else:
                status = "✅" if e < b * 0.8 else "⚠️" if e < b else "❌"
            print(f"  {key:<15s} {b:>10.2f} {e:>10.2f} {delta:>10s} {status:>8s}")
            report["comparisons"].append({
                "metric": key, "base": b, "evq": e,
                "delta_pct": delta, "category": "ppl"
            })

    # --- Passkey ---
    if "passkey" in base and "passkey" in evq:
        print("\n🔑 Passkey Retrieval (higher is better):")
        print(f"  {'Metric':<15s} {'Base':>10s} {'EVQ':>10s} {'Δ':>10s} {'Status':>8s}")
        print("  " + "-" * 55)
        for key in sorted(base["passkey"].keys()):
            b = base["passkey"].get(key, 0)
            e = evq["passkey"].get(key, 0)
            delta = fmt_pct(b, e)
            status = "✅" if e >= 0.95 else "⚠️" if e >= 0.7 else "❌"
            print(f"  {key:<15s} {b:>10.1%} {e:>10.1%} {delta:>10s} {status:>8s}")
            report["comparisons"].append({
                "metric": key, "base": b, "evq": e,
                "delta_pct": delta, "category": "passkey"
            })

    # --- LongBench ---
    if "longbench" in base and "longbench" in evq:
        print("\n📚 LongBench (higher is better):")
        print(f"  {'Task':<20s} {'Base':>10s} {'EVQ':>10s} {'Δ':>10s} {'Status':>8s}")
        print("  " + "-" * 60)

        lb_base = base["longbench"]
        lb_evq = evq["longbench"]
        task_deltas = []

        for task in sorted(lb_base.keys()):
            if task.startswith("_"):
                continue
            b_score = lb_base[task]["score"] if isinstance(lb_base[task], dict) else lb_base[task]
            e_score = lb_evq.get(task, {})
            e_score = e_score["score"] if isinstance(e_score, dict) else (e_score or 0)
            delta = fmt_pct(b_score, e_score)
            pct_change = (e_score - b_score) / abs(b_score) * 100 if b_score > 0 else 0
            task_deltas.append(pct_change)
            status = "✅" if pct_change >= 15 else "⚠️" if pct_change >= 0 else "❌"
            print(f"  {task:<20s} {b_score:>10.4f} {e_score:>10.4f} {delta:>10s} {status:>8s}")
            report["comparisons"].append({
                "metric": f"LB/{task}", "base": b_score, "evq": e_score,
                "delta_pct": delta, "category": "longbench"
            })

        # Overall
        b_overall = lb_base.get("_overall", 0)
        e_overall = lb_evq.get("_overall", 0)
        overall_delta = fmt_pct(b_overall, e_overall)
        overall_pct = (e_overall - b_overall) / abs(b_overall) * 100 if b_overall > 0 else 0
        print("  " + "-" * 60)
        status = "✅✅" if overall_pct >= 15 else "✅" if overall_pct >= 0 else "❌"
        print(f"  {'OVERALL':<20s} {b_overall:>10.4f} {e_overall:>10.4f} {overall_delta:>10s} {status:>8s}")

        report["summary"]["longbench_overall_delta"] = f"{overall_pct:.1f}%"
        report["summary"]["longbench_meets_15pct"] = overall_pct >= 15
        report["summary"]["avg_task_delta"] = f"{sum(task_deltas)/len(task_deltas):.1f}%" if task_deltas else "N/A"

    # --- 总结 ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    meets = report["summary"].get("longbench_meets_15pct", False)
    if meets:
        print("  🎉 EVQ-Cosh LoRA 在 LongBench 上超过 Base Instruct ≥15% — 目标达成!")
    else:
        overall = report["summary"].get("longbench_overall_delta", "?")
        print(f"  ⚠️  LongBench 提升 {overall} — 未达 15% 目标")
        print("  可能的下一步:")
        print("    1. 增加训练步数 (当前 600 → 试 1200)")
        print("    2. 提高 seq_len (当前 8K → 试 16K)")
        print("    3. 换 Qwen2.5-7B-Instruct (更长原生上下文)")

    # --- LaTeX snippet ---
    latex = generate_latex_table(report)
    report["latex_table"] = latex

    return report


def generate_latex_table(report: dict) -> str:
    """Generate a LaTeX table snippet for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{EVQ-Cosh LoRA ($r{=}64$, $\tau{=}1.414$) vs.\ base LLaMA-3-8B-Instruct.}",
        r"\label{tab:lora-evq-v2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Base Instruct} & \textbf{EVQ LoRA} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]

    for item in report["comparisons"]:
        m = item["metric"]
        b = item["base"]
        e = item["evq"]
        d = item["delta_pct"]
        cat = item["category"]

        if cat == "ppl":
            lines.append(f"  {m} & {b:.2f} & {e:.2f} & {d} \\\\")
        elif cat == "passkey":
            lines.append(f"  {m} & {b:.0%} & {e:.0%} & {d} \\\\")
        elif cat == "longbench":
            lines.append(f"  {m} & {b:.3f} & {e:.3f} & {d} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_result", required=True)
    p.add_argument("--evq_result", required=True)
    p.add_argument("--output", default="comparison.json")
    args = p.parse_args()

    base = load_json(args.base_result)
    evq = load_json(args.evq_result)

    report = compare(base, evq)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 对比报告已保存: {args.output}")

    # Also print LaTeX table
    print("\n--- LaTeX Table (可直接粘贴到论文) ---")
    print(report.get("latex_table", ""))


if __name__ == "__main__":
    main()
