"""Readable evidence for one method configuration; no winner/arm selection."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def compare_objectives(method, control, method_diagnostic=None, control_diagnostic=None,
                       method_generation=None, control_generation=None):
    """Compare one score-fit/output-KD pair; do not launch or select candidates."""
    from .study import digest
    roots = [Path(method), Path(control)]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    specs = [item["specification"] for item in manifests]
    for item in manifests:
        if item["status"] != "complete" or item.get("checkpoint_role") == "unoptimized_initialization":
            raise ValueError("objective attribution needs two completed fitted checkpoints")
    for field in ("base_model", "layers", "runtime"):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f"unmatched {field}")
    for field in ("capture_sha256", "shape", "fold", "initialization_sha256"):
        if field not in specs[0] or specs[0][field] != specs[1].get(field):
            raise ValueError(f"unmatched or missing {field}; fit from the same --init-from")
    fits = [dict(spec["fit"]) for spec in specs]
    for fit, expected in zip(fits, ((1.0, 0.0), (0.0, 1.0))):
        weights = (fit.pop("score_weight"), fit.pop("output_weight"))
        if weights != expected:
            raise ValueError("expected score/value method versus output/value distillation")
        if fit["steps"] < 1 or not all(fit[field] for field in ("learn_projections", "learn_frequency", "learn_content")):
            raise ValueError("both objectives must train the same full parameter set")
    if fits[0] != fits[1]:
        raise ValueError("fit settings differ beyond score/output objective weights")
    layers = []
    for layer in range(manifests[0]["layers"]):
        receipts = [json.loads((root / f"layer_{layer:03d}" / "result.json").read_text()) for root in roots]
        for field in ("initial_state_sha256", "data_position_schedule_sha256", "optimizer_updates"):
            if field not in receipts[0] or receipts[0][field] != receipts[1].get(field):
                raise ValueError(f"layer {layer}: unmatched {field}")
        layers.append({"layer": layer, **{field: receipts[0][field] for field in
                       ("initial_state_sha256", "data_position_schedule_sha256", "optimizer_updates")},
                       "method_fit_seconds": receipts[0]["seconds"], "control_fit_seconds": receipts[1]["seconds"]})
    result = {"contrast": "operator-score fitting versus attention-output distillation; shared parameterization",
              "delta_convention": "method minus control; lower errors/NLL are better", "matched_layers": layers,
              "method_manifest_sha256": digest(roots[0] / "manifest.json"),
              "control_manifest_sha256": digest(roots[1] / "manifest.json")}
    if bool(method_diagnostic) != bool(control_diagnostic) or bool(method_generation) != bool(control_generation):
        raise ValueError("provide both sides of each observation")
    if method_diagnostic:
        data = [json.loads(Path(path).read_text()) for path in (method_diagnostic, control_diagnostic)]
        layer = data[0]["layer"]
        for item, root in zip(data, roots):
            if item["layer"] != layer or item["factor_sha256"] != digest(root / f"layer_{layer:03d}.pt"):
                raise ValueError("diagnostic does not match the fitted layer")
            if item["capture_sha256"] != specs[0]["capture_sha256"]:
                raise ValueError("diagnostic capture differs from the shared capture")
        held = [item["held_out_observation"] for item in data]
        signatures = [[(row["id"], row["source_id"], row["record_sha256"]) for row in item["rows"]] for item in held]
        if not signatures[0] or signatures[0] != signatures[1] or held[0]["position_scale"] != held[1]["position_scale"]:
            raise ValueError("held-out documents or distances differ")
        metrics = ("position_response_mse", "static_score_mse", "relative_output_mse", "relative_value_mse")
        result["held_out"] = {"layer": layer, "position_scale": held[0]["position_scale"],
                              "documents": len(signatures[0]), "deltas": {
                                  key: held[0]["means"][key] - held[1]["means"][key] for key in metrics},
                              "paired_rows": [{"id": a["id"], "source_id": a["source_id"],
                                               **{key: a[key] - b[key] for key in metrics}}
                                              for a, b in zip(held[0]["rows"], held[1]["rows"])]}
    if method_generation:
        data = [json.loads(Path(path).read_text()) for path in (method_generation, control_generation)]
        for item, root in zip(data, roots):
            if item["factors_sha256"] != digest(root / "manifest.json"):
                raise ValueError("generation does not match the fitted checkpoint")
        for field in ("base_model", "runtime", "prompt_sha256", "input_ids_sha256", "generation_settings", "expected_answer_ids"):
            if field not in data[0] or data[0][field] != data[1].get(field):
                raise ValueError(f"generation has unmatched {field}")
        result["answer"] = {"answer_nll_delta": data[0]["answer_nll"] - data[1]["answer_nll"],
                            "method_exact": data[0]["answer_exact"], "control_exact": data[1]["answer_exact"],
                            "method_output": data[0]["output"], "control_output": data[1]["output"]}
    return result


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
