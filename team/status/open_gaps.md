# Open Gaps — EVQ-Cosh NeurIPS 2026

> 最后更新: 2026-03-13
> 本文件是当前所有未关闭缺口的快速清单。详细分析见 `WORKFLOW_AND_PAPER_GAPS.md`。

---

## P0: 提交阻塞项

| # | 缺口 | 当前状态 | 所需行动 | 预估时间 |
|---|------|---------|---------|---------|
| 1 | Phase 17c 多 seed 确认 (Stage 2-3) | seeds 42-44 Stage 1 ✅, Stage 2-3 进行中 | 完成 Stage 2-3, 确认方向一致 | 1-2 周 GPU |
| 2 | Collision-block 图 | Phase 18 数据已有 | 画 base vs collision fraction 图，解释 base=10K 失败 | 2-3 小时 |
| 3 | τ* 措辞修正 | 论文中称 "scaling law" | 改为 "empirical scaling conjecture"，标注非定理 | 1 小时写作 |
| 4 | Broadband 投影形式化 | §3 中是核心假设但无 Lemma 格式 | 写 Lemma + 显式边界条件 (D∝1/Δ, base∈[8K,100K], L≥4096) | 2 小时写作 |

---

## P1: 强 accept → spotlight

| # | 缺口 | 当前状态 | 所需行动 | 预估时间 |
|---|------|---------|---------|---------|
| 5 | NLL 任务分解可视化 | Phase 21a 数据已有 | 画 QA vs non-QA 的 NLL gap 图 | 1 小时 |
| 6 | Inference latency 实测 | 论文声称 "zero overhead" 无实测 | 写 benchmark 脚本，报告均值±std | 1-2 小时 |
| 7 | Waterbed 不等式证明 | 定性描述 | Appendix Jensen 证明 | 2-3 小时写作 |
| 8 | 置信区间补全 | Phase 17c 主结果无不确定性 | 补充 seed 间/chunk 间方差 | 1 小时 |

---

## P2: 锦上添花

| # | 缺口 | 当前状态 | 所需行动 |
|---|------|---------|---------|
| 9 | τ sweep heatmap (PPL gap vs τ/τ*) | Phase 16 数据已有 | 画热力图 |
| 10 | Training loss 曲线 (Geo vs EVQ) | 训练日志已有 | 画曲线，展示训练中 waterbed |
| 11 | Reproducibility appendix (超参数卡) | `internal/AIHANDOFF.md` Part 1 有完整配置 | 转为 appendix 表格 |
| 12 | Phase 18 多 seed (base 泛化) | 部分数据已有 | 补 seed 确认 base 泛化 |

---

## P3: 有余力再做

| # | 缺口 | 备注 |
|---|------|------|
| 13 | 1.5B 验证 (Phase 20) | 需大量 GPU，camera-ready 或 follow-up |
| 14 | SCROLLS 其他子集 | 已有 GovReport，可补 narrative_qa 等 |
| 15 | 代码开源准备 | 待论文接收后进行 |

---

## 已关闭的缺口 (近期)

| 缺口 | 关闭时间 | 结果 |
|------|---------|------|
| ~~Phase 17c 454M Stage 1~~ | 2026-03-10 | ✅ seeds 42-44 全部确认 |
| ~~QuALITY 下游评估~~ | 2026-03-11 | ✅ Gold NLL -30.1%, n=2086 |
| ~~99-run τ* validation~~ | 2026-03-08 | ✅ R²>0.95 across configs |
| ~~Attention visualization~~ | 2026-03-12 | ✅ 750M EVQ vs Geo head-level |
| ~~LaTeX 初稿~~ | 2026-03-12 | ✅ 10页正文 + 8页附录 |

---

## 参考

- 详细分析: `team/status/WORKFLOW_AND_PAPER_GAPS.md` (Theory/Experiment/Figure 矩阵 + GPU 防空转方案)
- 下游策略: `team/status/downstream_strategy_2026-03-12.md`
- 实验计划: `team/plans/README.md`
