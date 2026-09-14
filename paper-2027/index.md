# Beyond the Base：当前论文索引

标题：**Beyond the Base: Exponent Allocation in RoPE**。当前以z价值、解析构造及学习/零训练收益为主线，纳入TailSpline的Llama和OLMo结果。见[本轮修订与记录](REVISION_BRIEF.md)。

| 需要什么 | 文件 |
|---|---|
| 读稿 / 源码 / 源码包 | [main.pdf](main.pdf)、[main.tex](main.tex)、[source.zip](exponent-allocation-source.zip) |
| 当前修订目标与交接 | [修订目标](REVISION_BRIEF.md)、[HANDOFF](HANDOFF.md) |
| 当前交付与复现说明 | [本轮回执](research/THREE_DISCOVERIES_REVISION_20260914.md)、[补充材料说明](SUPPLEMENT_README.md)、[冻结评测运行说明](runtime/README.md) |
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

按改动需要复核；`figs/make_exponent_revision_figures.py`和`figs/make_story_figures.py`重建历史图表，最后运行`figs/make_allocation_value.py`生成主图及实证图；`verify_explicit_geometry.py`、`verify_profile_diagnostics.py`、`verify_recovered_assets.py`、`verify_routing_schedule.py`检查数学/摘要/历史调度。它们不运行模型。历史独立源码包重建记录见[2026-09-12重构说明](research/STORY_RESTRUCTURE_20260912.md)，不代表后续版本已独立重建。

源包按`package_source.py`的实际依赖收集，不包含内部research、Pro指导或全仓库原始实验流。2026-09-14已更新TeX/PDF与源码包，并在解包目录运行五项CPU核验并独立编译；未运行模型实验。

当前写作锚点修订：151.9M保留固定支持下的三seed外推收益，几何对照配置仅注明FMRoPE来源；Fig3几何与Fig4学习分开，Fig5简化为边界与两模型Full-13，代价留正文。正文9页、五图两表；最新处理与验收见[修订目标](REVISION_BRIEF.md)。
