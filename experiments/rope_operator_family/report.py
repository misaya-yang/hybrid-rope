"""Readable evidence for one method configuration; no winner/arm selection."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def make_report(factors, output, diagnostic=None, evaluation=None, generation=None, profile=None):
    root = Path(factors)
    manifest = json.loads((root / "manifest.json").read_text())
    spec = manifest["specification"]
    shape = spec["shape"]
    lines = ["# Operator-family 方法定位结果", "",
             f"状态：{manifest['status']}。本报告只汇总这一份方法配置。", "",
             f"内容维度 {shape['content_rank']}，rotary 实数维度 {shape['rotary_dim']}；"
             f"每层每 token 共 {shape['content_rank'] + shape['rotary_dim']} 个缓存实数。", "",
             "## 拟合过程", "",
             "| 层 | updates | 首步 score NMSE | 末步 score NMSE | 拟合秒数 |",
             "|---|---:|---:|---:|---:|"]
    fit_seconds = 0.0
    for index in range(manifest["layers"]):
        logfile = root / f"layer_{index:03d}" / "fit.jsonl"
        if not logfile.exists():
            continue
        rows = [json.loads(line) for line in logfile.read_text().splitlines()]
        if not rows:
            continue
        fit_seconds += rows[-1]["seconds"]
        lines.append(f"| {index} | {rows[-1]['step']} | {rows[0]['relative_score_mse']:.6g} | {rows[-1]['relative_score_mse']:.6g} | {rows[-1]['seconds']:.3f} |")
    lines += ["", "各步可能使用不同文档/距离；上表用于定位优化过程，留出诊断用于判断是否改善。",
              f"累计拟合时间：{fit_seconds:.3f} 秒（不含捕获、初始化、模型加载和评测）。"]
    if diagnostic:
        data = json.loads(Path(diagnostic).read_text())
        cal, held = data["calibration_prediction"], data["held_out_observation"]
        lines += ["", "## 留出内容与完整 attention 响应", "",
                  f"层 {data['layer']}；position scale {held['position_scale']}。", "",
                  "| 指标 | 校准预测 | 留出实测 |", "|---|---:|---:|"]
        for key in ("static_score_mse", "position_response_mse", "relative_position_response_mse", "relative_score_mse", "attention_kl", "relative_output_mse", "teacher_top2_margin_mae"):
            if key in cal["means"] and key in held["means"]:
                lines.append(f"| {key} | {cal['means'][key]:.6g} | {held['means'][key]:.6g} |")
        geometry = data.get("generator_diagnostics", {})
        lines += ["", f"几何诊断：{json.dumps(geometry, ensure_ascii=False)}。几何量与模型行为分开解释。"]
    if evaluation:
        data = json.loads(Path(evaluation).read_text())
        lines += ["", "## 完整模型 NLL", "",
                  f"{len(data['rows'])} 个独立文档，文档均值 NLL：{data['mean_document_nll']:.6g}。",
                  "这是当前方法的模型结果；单个 NLL 数字本身不证明优于其他方法。"]
    if generation:
        data = json.loads(Path(generation).read_text())
        lines += ["", "## 单个实际生成样例", "",
                  f"输入 {data['input_tokens']} tokens；输出 {len(data['generated_ids'])} tokens；{data['seconds']:.3f} 秒。", "",
                  "> " + data["output"].replace("\n", "\n> ")]
        if "answer_nll" in data:
            lines += ["", f"已知答案：{data['expected_answer']}；exact match：{data['answer_exact']}；"
                      f"条件答案 NLL：{data['answer_nll']:.6g}。"]
    if profile:
        data = json.loads(Path(profile).read_text())
        samples = data["samples"]
        lines += ["", "## 当前实现的成本", "",
                  f"输入长度 {data['input_tokens']}，固定 decode {data['decode_tokens']} tokens。"]
        for key in ("prefill_seconds", "decode_seconds", "cached_bytes_at_prefill", "cached_bytes_after_decode"):
            lines.append(f"- {key} 中位数：{statistics.median(row[key] for row in samples):.6g}")
        lines += ["", "成本 profile 使用随机 token，不作为任务质量证据。这里只测当前配置，不外推成多模型、多任务的总工期。"]
    lines += ["", "## 证据", "", f"- 配置与完成状态：{(root / 'manifest.json').resolve()}"]
    for path in (diagnostic, evaluation, generation, profile):
        if path:
            lines.append(f"- {Path(path).resolve()}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n")
