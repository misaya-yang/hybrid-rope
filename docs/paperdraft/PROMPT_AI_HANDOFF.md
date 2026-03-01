# AI Handoff: EVQ-Cosh NeurIPS 2026

> **最后更新**: 2026-03-01
> **语言**: 全程中文
> **理论参考**: `docs/CORE_THEORY.md`（写论文时必读）

---

## 项目

NeurIPS 2026。方法 EVQ-Cosh：单参数 τ 控制 RoPE 频率分配，变分逆问题闭式最优解。

## 进度

| Phase | 状态 | 结论 |
|-------|------|------|
| 1-3 | ✅ | 128-tok: EVQ τ=1.5 → -18.3% PPL@8K vs Geo |
| 6 | ✅ | τ=0→5.0 单调下降无peak; YaRN全崩; 1024-tok τ*≈2.0; passkey EVQ>Geo |
| 7 | ✅ | YaRN确认方法级不兼容; multi-seed CV<2.3%; τ lr不敏感; 无waterbed; 350M ext Geo≈EVQ |
| **8A** | **✅** | 8x ext: Hybrid +1.7%, EVQ1.5 +2.9% vs Geo @16K; PI +205%, YaRN +94% 崩溃; Passkey @8K 全趋同 52-54% |
| **8B** | **✅** | 续训消融: EVQ passkey @1K 63→66→72%(上升中), Geo饱和80%; PPL gap 4.7%→3.1% |
| **8C** | **✅** | From-scratch 4K: EVQ τ=2.0 PPL@16K 164.4(-6.3%) 赢Geo, Passkey 66%(-3pp) |
| **8D** | **✅** | Scaling law: C=67.84(预测64), R²=0.76, 适用域L≥1024(短L单调无peak) |
| **8E** | **✅** | **🎯 EVQ τ=1.0 passkey 72% > Geo 69%! Hybrid τ=1.0 PPL+PK 双赢!** |
| **9** | **待做** | 1B Pro 6000: 4K pretrain → 32K ext, Geo/EVQ/Hybrid, τ=1.0 主力 |

## Phase 8 关键待看

1. **Hybrid EVQ (A7)** passkey 是否追上 Geometric
2. **EVQ τ=2.5 (A6)** PPL 是否比 τ=1.5 好
3. **8B 续训量消融** passkey 是否随续训量恢复
4. **8D Scaling law**: τ*(256)=4.0? τ*(512)=2.83? → 验证 τ*=64/√L

## 核心数据一览

**From-scratch 128-tok（PE-dominant）**:
- τ=5.0: -35% PPL@8K (FW), -57% (TS), 无waterbed
- Passkey: EVQ 55% vs Geo 48.5%
- Learnable τ → 1.14±0.02, PPL@128 平坦度 1.3%

**Context extension 350M 512→2K（4x）**:
- PPL: EVQ ≈ Geo >> YaRN >> PI
- Passkey: EVQ < Geo（Q/K alignment cost）

**Context extension 350M 512→4K（8x, Phase 8A 完整结果）**:
- PPL@16K: Geo 83.3, Hybrid 84.7(+1.7%), EVQ1.5 85.7(+2.9%), PI 254.4(+205%), YaRN 161.9(+94%)
- Passkey @8K: 全趋同 50-54%（350M 能力天花板，非 PE 问题）
- Passkey @1K: Geo 82%, Hybrid 74%, EVQ1.5 70%（短距 EVQ 落后 = Q/K alignment cost）

**Passkey 核心证据链**:
- From-scratch 128-tok: **EVQ 55% > Geo 48.5%**（EVQ 赢）
- Extension: EVQ < Geo（alignment cost）
- 8C from-scratch 4K τ=2.0: EVQ PPL **赢 6.3%**, Passkey 输 3pp
- **8E from-scratch 4K τ=1.0: EVQ passkey 72% > Geo 69% (+3pp)!** 用对τ就赢
- **Hybrid τ=1.0: PPL 172.6 < Geo 175.4 且 Passkey 70.5% > Geo 69% — 双赢!**
- τ 控制 PPL-Passkey tradeoff: τ=2.0 偏 PPL, τ=1.0 偏 Passkey, Hybrid 兼得

**Passkey 问题已解决**: from-scratch + τ=1.0 → EVQ 赢; Phase 9 用 τ=1.0 + Hybrid 即可

## 文件路径

**核心文档**:
- `docs/CORE_THEORY.md` — 理论精简版（写论文必读）
- `docs/paperdraft/PAPER_ERROR_CORRECTIONS.md` — 论文 7 个已知错误

**实验报告**:
- `docs/paperdraft/phase6_report.md` / `phase7_report.md`

**实验计划**（在 `docs/prompts/`）:
- `docs/prompts/PROMPT_PHASE8_EXTENDED_RATIO.md`（8A-8D，含 scaling law 验证）
- `docs/prompts/PROMPT_PHASE9_1B_PRO6000.md`（1B 终极验证，待 Phase 8 完成后启动）

**数据 JSON**:
- `data/evq_128tok_results/results_phase6.json`
- `data/evq_128tok_results/results_phase7.json`
- Phase 8: `/root/autodl-tmp/evq_phase8/results_phase8.json`（待生成）
- Phase 9: `/root/autodl-tmp/evq_phase9/results_phase9.json`（待生成）

**代码**:
- `rope/learnable_evq.py` — EVQ 核心实现
- `scripts/m4_evq_sweep/run_evq_sweep.py` — 训练 pipeline
- `scripts/m4_evq_sweep/phase7f_context_ext.py` — context extension

**论文**: `submission/paper/hybrid_rope_neurips.tex`（有 7 个已知错误，暂不更新）

## 红线

1. 全程中文
2. 不写 LLaMA-3 实验（协议问题）
3. τ*=1.5 不是 universal default（regime-dependent）
4. Algorithm 1 不可用（残差 35-49%）
5. Qwen 用的 anchored_sigmoid 不是 EVQ
6. LongBench 是 21 tasks 不是 6
7. 不擅自更新论文 LaTeX
