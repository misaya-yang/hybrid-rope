# Repo Reorganization & Documentation Overhaul (2026-02-27)

## What was done

1. **AI_HANDOFF.md 全面重写**: 从旧的 anchored-sigmoid ops card 改写为 V5/EVQ 总纲 (Master Guide)
   - 新增 AI 工作守则（强制整理 + 写文档）
   - 新增理论框架速览表
   - 新增代码核心入口表
   - 新增实验全景（论文引用 / 进行中 / 历史）
   - 新增文件夹结构约定
   - 新增新 AI 快速上手路径（8 步阅读顺序）
   - 新增术语映射（V4 旧名 ↔ V5 新名）
   - 新增紧急任务优先级

2. **docs/PAPER_DRAFT_STATUS.md 创建**: 论文进度追踪文档
   - 版本演进表
   - 页面预算
   - 各 Section 详细状态（含 ⚠️ 数据过时标记）
   - 3 个关键缺口的详细描述和解决方案
   - 编译指令

3. **docs/EXPERIMENT_REGISTRY.md 更新**:
   - 新增 Tier 0: EVQ τ-sweep 实验（EXP_EVQ_50M_SWEEP, EXP_EVQ_125M_SWEEP, EXP_EVQ_8B_LONGINST）
   - 添加 V5 框架不匹配警告

4. **knowledge_base/README.md 更新**:
   - 添加旧术语警告

## Files Modified

- `AI_HANDOFF.md` — 全面重写
- `docs/PAPER_DRAFT_STATUS.md` — 新建
- `docs/EXPERIMENT_REGISTRY.md` — 新增 Tier 0
- `docs/exp/2026-02-27_repo_reorganization.md` — 本报告
- `knowledge_base/README.md` — 添加注意事项

## Operator

Claude (Cowork mode)
