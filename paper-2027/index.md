# Beyond the Base：当前论文索引

标题：**Beyond the Base: Exponent Allocation in RoPE**。2026-09-13构建记录为科学正文9页、总52页；下列写作指导已于2026-09-14更新，不代表PDF同步重建。

| 需要什么 | 文件 |
|---|---|
| 读稿 / 源码 / 源码包 | [main.pdf](main.pdf)、[main.tex](main.tex)、[source.zip](exponent-allocation-source.zip) |
| 当前修订目标与交接 | [修订目标](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md) |
| 已有构建与验收记录 | [2026-09-13区间修订回执](research/PAPER_INTERVAL_REORIENTATION_20260913.md) |
| 证据 / 公式 / 图表 | [证据index](research/evidence/index.md)、[主张映射](research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md) |
| 按需研究与来源查询 | [research/index.md](research/index.md) |
| 下一阶段安排 | [研究计划](../docs/research/next_stage_20260912/index.md) |
| 写作约定 | [NARRATIVE_GUIDE](NARRATIVE_GUIDE.md)、[REVISION_BRIEF](REVISION_BRIEF.md) |

## 构建与数字复核

从仓库根执行：

```bash
bash paper-2027/compile.sh
python3 paper-2027/package_source.py
```

按改动需要复核；`figs/make_exponent_revision_figures.py`和`figs/make_story_figures.py`重建图表；`verify_explicit_geometry.py`、`verify_profile_diagnostics.py`、`verify_recovered_assets.py`、`verify_routing_schedule.py`检查数学/摘要/历史调度。它们不运行模型。历史独立源码包重建记录见[2026-09-12重构说明](research/STORY_RESTRUCTURE_20260912.md)，不代表后续版本已独立重建。

源包按`package_source.py`的实际依赖收集，不包含内部research、Pro指导或全仓库原始实验流。2026-09-13已更新TeX/PDF与论文校验脚本；未运行模型实验。
