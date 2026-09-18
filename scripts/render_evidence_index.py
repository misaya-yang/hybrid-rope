#!/usr/bin/env python3
"""Render portable evidence navigation from the canonical asset registry."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "paper-2027/research/evidence"


def cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render() -> str:
    registry = json.loads((OWNER / "asset_registry.json").read_text())
    assets = registry["assets"]
    ids = [asset["id"] for asset in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate asset IDs")
    text = """# 论文证据索引

本页由 `python3 scripts/render_evidence_index.py` 从
[asset_registry.json](asset_registry.json) 生成。修改资产身份、当前论文位置或来源时，
先修改注册表再重建本页；分数、协议和完成状态继续由各结果owner维护。

## 按问题进入

- 当前论证和章节位置：[主张映射](../EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md)。
- 核心证据形成过程：[证据时间线](../../../docs/research/next_stage_20260912/KEY_EXPERIMENT_COMPASS_20260914.md)。
- 最新稿件及封板：[当前handoff](../../HANDOFF.md)。
- 实验结果与复用代码：[实验索引](../../../experiments/index.md)。

固定支持与学习见A01–A13；TailSpline构造与主结果见A37、A39–A53；
直接YaRN、跨模型、70B及Kanana见A59–A61、A65；原生研究见A54、A62–A64。
Figure 1已是配置／距离响应概念图；固定支持和crossing结果仍在§3与附录C。

## 资产与当前安放

下表包含当前稿证据与历史／探索资产；`Extended records`或历史状态不表示结果无效，
只说明它在当前稿中的职责。科学解释与使用边界见注册表的`interpretation`。
来源链接按原位置保留；ignored材料只列路径，不作为跨机器必需链接。

| ID | 资产／问题 | 当前论文位置或研究状态 | 证据身份 | 来源 |
|---|---|---|---|---|
"""
    for asset in assets:
        sources = []
        for source in asset["sources"]:
            availability = source["availability"]
            if availability in {"local-ignored", "missing"}:
                sources.append(f'`{source["path"]}` ({availability})')
            else:
                path = ROOT / source["path"]
                relative = Path(os.path.relpath(path, OWNER)).as_posix()
                sources.append(f'[{path.name}]({relative})')
        values = [asset["id"], asset["name"] + "：" + asset["question"],
                  asset["manuscript_location"], asset["evidence_grade"],
                  "<br>".join(sources)]
        text += "| " + " | ".join(cell(value) for value in values) + " |\n"
    text += """
## 使用边界

报告聚合、逐行存储分数复算和完整输出重新评分是不同核验层级。CPU证明／代理诊断
不自动变成模型性能结论；部分任务、旧开发面板与新确认面板不得混算。
NCP的原生LM、RULER诊断和自然QA分别陈述。完整旋转对有效秩不是模型损失预测器；
minimum-bending最优性对应声明的目标，不升级为Nyquist或通用任务最优性。

原人工索引的旧图号／旧版本解释保存在[导航历史](../../../docs/archive/navigation_20260918/index.md)。
"""
    return text


if __name__ == "__main__":
    (OWNER / "index.md").write_text(render())
    print("Rendered evidence index from asset_registry.json")
