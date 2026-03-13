# Senior Collaborator Brief — EVQ-Cosh NeurIPS 2026

> 最后更新: 2026-03-13
> 本文件帮助合作者在 10 分钟内理解项目全貌并进入写作/实验状态。

---

## 一、论文一句话

EVQ-Cosh 是 RoPE 频率分配的闭式变分解（单参数 τ），在 50M–750M 规模上系统性验证了三个核心 claim：geometric RoPE 是退化极限、τ* 可预测最优频率分配、EVQ+YaRN 在 8× 外推下实现 100% passkey（vs Geo+YaRN 61-65%）。

---

## 二、仓库五层结构

| 层 | 路径 | 内容 |
|----|------|------|
| **论文** | `paper/` | LaTeX 源码、图表、表格。入口: `paper/main.tex` |
| **实验** | `scripts/` | Phase 8–21 核心实验链 + 出图脚本 + RoPE 库 |
| **文档** | `docs/` | 理论推导 (`theory/`)、实验报告 (`exp/`)、概览 (`overview/`) |
| **结果** | `results/` | 实验输出 (JSON/checkpoint)。按 `core_text/`, `theory/` 等分类 |
| **协作** | `team/` | 本文件所在目录。briefs/status/plans 三子目录 |

> 开发者文档和 AI handoff 在 `internal/`，合作者无需阅读。

---

## 三、当前进展快照

### 已完成的关键里程碑

| 里程碑 | Phase | 规模 | Seeds | 状态 |
|--------|-------|------|-------|------|
| 99-run τ* scaling law 验证 | 16 | 50M/125M | 3+ seeds × 27 configs | ✅ R²>0.95 |
| EVQ+YaRN 100% passkey @8× | 14c | 350M | 3+3 seeds | ✅ vs Geo 61-65% |
| PE-dominant extreme extrap (128→8K) | 11b | 125M | 3 seeds | ✅ EVQ ≥ DAPE |
| 454M 续训 2K→48K | 17c | 454M | seeds 42-44 | ✅ PPL<3.3 @48K |
| 750M continued-pretrain | 15 | 750M | single-seed | ✅ 16K PPL -45.9% |
| QuALITY 下游评估 (n=2086) | 21b | 750M | — | ✅ Gold NLL -30.1% @8K |
| LaTeX 初稿 | — | — | — | ✅ 10页正文 + 8页附录 |

### 单点风险

| 风险项 | 程度 | 影响 |
|--------|------|------|
| Phase 17c 仅 seeds 42-44, single config | ⚠️ HIGH | C3/C4 claim 从 A 降 B+ |
| 750M 仅 single-seed | ⚠️ HIGH | Table 6 仅为 supporting |
| 454M 下游准确率无差异 (容量地板 ~25%) | MEDIUM | 已改用 Gold NLL |

---

## 四、论文叙事线

**三锚点结构** (§3–§5):

1. **Theory anchor (§3)**: 频率分配是变分逆问题 → EVQ-Cosh 闭式解 → Geometric 是 τ→0 退化极限
2. **Extrapolation anchor (§5.2)**: EVQ 在 PE-dominant regime (L=256→8K) 匹敌/超越可学习 PE
3. **Systems anchor (§5.1)**: EVQ + Progressive YaRN = qualitative capability gap (100% vs 61-65%)

**叙事定位**: Theory + mechanism 论文，非 benchmark SOTA 论文。所有理论预测 (τ*, waterbed, synergy) 均有实验验证。

---

## 五、需要老师/合作者参与的决策

| # | 决策点 | 上下文 | 参考文件 |
|---|--------|--------|---------|
| 1 | τ* 措辞: "scaling law" vs "empirical conjecture" | 99-run 验证强，但无 first-principles 推导 | `status/WORKFLOW_AND_PAPER_GAPS.md` T7 |
| 2 | Broadband surrogate 形式化 | 核心假设 R²>0.99，需要写成 Lemma 还是 Remark | 同上 T2 |
| 3 | 资源分配: 1.5B 验证 vs 补 multi-seed vs 强化 video | GPU 时间有限 | `plans/README.md` |
| 4 | 下游证据叙事: NLL reversal 进正文 vs 只放 appendix | 详见策略分析 | `status/downstream_strategy_2026-03-12.md` |

---

## 六、快速入口

| 想要做什么 | 去哪里 |
|-----------|--------|
| 编辑论文 | `paper/main.tex` → `paper/sections/` |
| 看论文叙事 | `internal/mainstory.md` |
| 看所有图表 | `paper/figs/` |
| 查 Figure/Table 来源 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| 看完整实验报告 | `docs/exp/README.md` |
| 看 P0-P3 优先级 | `team/status/WORKFLOW_AND_PAPER_GAPS.md` |
| 看缺口快速清单 | `team/status/open_gaps.md` |
| 看下一步实验计划 | `team/plans/README.md` |
| 复现实验 | `docs/overview/REPRODUCE.md` |

---

## 七、合作守则

- 论文相关改动走 `paper/` 目录，不要修改 `scripts/` 或 `results/` 中的实验代码
- 单 seed 证据不放主论文锚点 → 标注为 supporting evidence (appendix)
- 新实验报告放 `docs/exp/YYYY-MM-DD_slug.md`
- 需要讨论的问题写在 `team/status/open_gaps.md`
