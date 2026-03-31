# Advisor Brief — EVQ-Cosh NeurIPS 2026

> 最后更新: 2026-03-13

## 论文三锚点

1. **Closed-form theory**: RoPE 频率分配是变分逆问题的闭式解，geometric RoPE 是 τ→0 退化极限
2. **Extreme extrapolation**: EVQ 在 DAPE-style 128→8K 极端外推中匹敌/超越可学习 PE (Phase 11b, 3-seed)
3. **Systems result**: EVQ + Progressive YaRN = 100% passkey @8× vs Geo + YaRN = 61-65% (Phase 14c, 3+3 seed)

## 最新进展 (2026-03-12/13)

### 已完成
- ✅ **99-run τ* scaling law 验证** (Phase 16): τ*=d_head/√L 在 50M/125M 多配置下 R²>0.95
- ✅ **Phase 17c 454M Stage 2-3**: seeds 42-44 确认 1024→2048 续训仍保持 EVQ 优势
- ✅ **750M continued-pretrain**: 16K PPL -45.9%, 8K AR exact 77.5% vs 0% (单 seed, supporting)
- ✅ **QuALITY 下游评估** (n=2086): Gold NLL -30.1% @8K, -21.4% @16K (EVQ vs Geo)
- ✅ **LaTeX 初稿完成**: 10页正文 + 8页附录, 7 figures, 6 tables
- ✅ **Attention distance visualization**: 750M EVQ vs Geo head-level 对比

### 单点风险
| 风险 | 严重程度 | 缓解策略 |
|------|---------|---------|
| Phase 17c 仅 seeds 42-44, single-config | ⚠️ HIGH | 补充 multi-config 或标注 limitations |
| 750M 仅 single-seed | ⚠️ HIGH | 标注为 supporting evidence |
| 454M 下游准确率无差异 | MEDIUM | 已改用 Gold NLL (信号清晰) |

## 最强当前证据

| 证据 | Phase | 规模 | Seeds | 关键数字 |
|------|-------|------|-------|---------|
| EVQ+YaRN synergy | 14c | 350M | 3+3 | 100% vs 61-65% passkey @8K |
| PE-dominant regime | 11 | 125M-454M | 3 | Raw PPL -52% @8× extrapolation |
| τ* scaling law | 16 | 50M-125M | 99 runs | Predicted vs actual R²>0.95 |
| Downstream NLL | 21b | 750M | n=2086 | Gold NLL -30.1% @8K |
| 750M scale-up | 15 | 750M | 1 | 16K PPL -45.9% |

## 仍然缺失

1. **下游准确率差异**: 454M 容量地板，750M 仅 single-seed，无法在准确率上展示 PE 差异
2. **理论 polish**: Broadband surrogate approximation 的 perturbation bound 需收紧
3. **Theorem/Proposition 边界**: Scaling law 目前是 empirical，需要标注清楚
4. **Cross-modal**: Video temporal 数据已有但仅作 appendix-level supporting

## 需要老师帮助

1. **理论收紧**: Surrogate approximation argument, theorem vs conjecture 边界
2. **Capacity compensation 假说**: 是否值得单独做 scale sweep (见 `plans/capacity_compensation_hypothesis.md`)
3. **下一步资源分配**: 1.5B text anchor vs 强化 video package vs 补 multi-seed
4. **论文叙事**: 当前 three-anchor 结构是否最优，是否需要调整重心

## 快速入口

| 内容 | 路径 |
|------|------|
| 论文 LaTeX | `paper/main.tex` |
| 叙事线 | `internal/mainstory.md` |
| P0-P3 优先级 + 完整缺口 | `team/status/WORKFLOW_AND_PAPER_GAPS.md` |
| 论文↔实验映射 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| 完整实验报告索引 | `docs/exp/README.md` |
