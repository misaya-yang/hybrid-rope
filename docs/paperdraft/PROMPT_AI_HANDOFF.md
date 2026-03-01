# AI Handoff: EVQ-Cosh NeurIPS 2026

> **最后更新**: 2026-03-02
> **语言**: 全程中文
> **完整实验报告**: `docs/paperdraft/FULL_EXPERIMENT_REPORT.md`（Phase 0-8 全量数据）
> **理论参考**: `docs/paperdraft/CORE_THEORY.md`（写论文时必读）

---

## 项目

NeurIPS 2026。方法 EVQ-Cosh：单参数 τ 控制 RoPE 频率分配，变分逆问题闭式最优解。

## 进度

| Phase | 状态 | 结论 |
|-------|------|------|
| 1-3 | ✅ | 128-tok: EVQ τ=1.5 → -18.3% PPL@8K vs Geo; EVQ(1 param) > DAPE(32 params) |
| 6 | ✅ | τ=0→5.0 单调无peak; YaRN全崩(+121%); 1024-tok τ*≈2.0; passkey EVQ 55% > Geo 48.5% |
| 7 | ✅ | multi-seed CV<2.3%; τ lr不敏感(20x范围→1.14±0.02); 无waterbed; 350M ext Geo≈EVQ |
| 8A | ✅ | 8x ext: Hybrid +1.7% vs Geo @16K; PI +205%, YaRN +94% 崩溃 |
| 8B | ✅ | 续训消融: PPL gap 4.7%→3.1%; passkey gap结构性不闭合 |
| 8C | ✅ | From-scratch 4K τ=2.0: PPL -6.3% 赢Geo, Passkey -3pp |
| 8D | ✅ | Scaling law: C=67.84, R²=0.76, 适用L≥1024 |
| **8E** | **✅** | **🎯 Headline: EVQ τ=1.0 PK 72% > Geo 69%; Hybrid τ=1.0 PPL+PK 双赢** |
| **8F** | **待做** | Multi-seed 验证 8E 结果（3pp margin 需要统计确认） |
| **9** | **待做** | 1B Pro 6000: 4K→32K ext, τ=1.0, 待 8F 确认后启动 |

## Headline 结果（8C/8E From-scratch 4K, 350M）

| Method | τ | PPL@16K | PK@1K | PK Global | vs Geo |
|--------|---|---------|-------|-----------|--------|
| Geometric | — | 175.4 | 87% | 69.0% | baseline |
| EVQ | 2.0 | **164.4** (-6.3%) | 82% | 66.0% | PPL赢PK输 |
| **EVQ** | **1.0** | 180.1 | **88%** | **72.0%** (+3pp) | **PK赢** |
| **Hybrid** | **1.0** | **172.6** (-1.6%) | **93%** | **70.5%** (+1.5pp) | **双赢** |

⚠️ 3pp / 1.5pp margin 在 400 trials 下 ~1.4σ / 0.7σ，**8F multi-seed 必须确认**

## 核心发现总结

1. **τ 控制 PPL ↔ Passkey tradeoff**: τ=2.0 偏 PPL(-6.3%), τ=1.0 偏 Passkey(+3pp), Hybrid 兼得
2. **Scaling law**: τ*(L) ≈ 68/√L, L≥1024; L<1024 单调无peak
3. **From-scratch EVQ 赢, Extension EVQ 输 passkey**: alignment cost 是结构性的，非数据量可解决
4. **PI/YaRN 在 8x 扩展比下崩溃**: +205% / +94% PPL@16K
5. **EVQ(1 param) > DAPE(32 params)**: -7.8% PPL@8K

## 文件路径

**核心文档**:
- `docs/paperdraft/FULL_EXPERIMENT_REPORT.md` — **Phase 0-8 完整数据**（最权威）
- `docs/paperdraft/CORE_THEORY.md` — 理论精简版
- `docs/paperdraft/phase8_analysis.md` — Phase 8 分析 + Passkey 策略
- `docs/paperdraft/PAPER_ERROR_CORRECTIONS.md` — 论文 7 个已知错误

**实验计划**（在 `docs/prompts/`）:
- `PROMPT_PHASE8_EXTENDED_RATIO.md`（8A-8E）
- `PROMPT_PHASE9_1B_PRO6000.md`（1B 终极验证）

**数据 JSON**:
- `data/evq_128tok_results/results_final.json` / `results_phase6.json` / `results_phase8.json`

**代码**:
- `rope/learnable_evq.py` — EVQ 核心实现
- `scripts/m4_evq_sweep/run_evq_sweep.py` — 训练 pipeline

## 红线

1. 全程中文
2. 不写 LLaMA-3 实验（协议问题）
3. τ* 是 regime-dependent，不是 universal default
4. Algorithm 1 不可用（残差 35-49%），降级 Appendix
5. Qwen 用的 anchored_sigmoid 不是 EVQ
6. LongBench 是 21 tasks 不是 6
7. 不擅自更新论文 LaTeX
8. **8E 结果未经 multi-seed 确认前，不可作为论文 claim 的唯一支撑**
