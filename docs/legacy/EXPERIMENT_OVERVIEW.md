# 实验总览

> 最后更新：2026-02-22

本文档是项目实验的**唯一权威索引**，按可信度分级，明确标注每个实验的参考价值。

---

## 📊 实验可信度分级

| 等级 | 含义 | 是否可引用 |
|------|------|-----------|
| ✅ **Tier 1** | 从零训练、多规模验证、统计显著 | **论文主线** |
| ✅ **Tier 2** | 机理实验、推理侧验证、公式拟合良好 | 论文支撑 |
| ⏳ **Tier 3** | 进行中、待验证 | 暂不引用 |
| ⚠️ **Tier 4** | 已知问题、需重跑 | **禁止引用** |
| ❌ **废弃** | 严重缺陷、仅作存档 | **禁止引用** |

---

## ✅ Tier 1：从零训练证据（论文主线）

### 1.1 多规模 Scaling 实验

| 规模 | 配置 | Geo 500k PPL@16K | Hybrid PPL@16K | 改善 | 状态 |
|------|------|----------------:|---------------:|-----:|------|
| **50M** | 3cfg × 3seed | 18.207±0.768 | **17.324±0.360** | ~5% | ✅ |
| **100M** | 单次 | 10.888 | **9.417** | ~13.5% | ✅ |
| **350M** | 多seed | 14.653±3.851 | **12.646±3.093** | ~13.7% | ✅ |

**关键文件**：
- `results/advisor_package_2026-02-15/01_scaling_from_scratch/`
- `artifacts/a100_2026-02-13/data/`

**结论**：跨规模一致正向改善，是论文最可信主线。

---

### 1.2 700M 训练频率对比

| 实验 | 结果位置 | 状态 |
|------|----------|------|
| Standard vs Sigmoid 频率训练 | `results/eval_700m/` | ✅ 完成 |

**关键文件**：
- `results/advisor_package_2026-02-15/06_700m_trainfreq/`

---

## ✅ Tier 2：机理实验（论文支撑）

### 2.1 Phase Collision 机理验证

| 指标 | Geo(10k) | Sigmoid(100k) | 改善 |
|------|---------|--------------|------|
| 崩溃比 | 22x | **1.08x** | ~95% |
| Phase Collision 平均改善 | - | - | ~55.67% |

**公式拟合**：R² > 0.99

**关键文件**：
- `knowledge_base/06_phase_collision_D_analysis.md`
- `results/anchored_sigmoid_v3_followup/`

---

### 2.2 频率范围分析

| 实验 | 结论 | 状态 |
|------|------|------|
| 最优 base 搜索 | base=100k 时 phase collision 最优 | ✅ |
| 频谱形状 vs base 选择 | 统一 base 后 geo 在 phase collision 上最优 | ✅ |

**关键文件**：
- `knowledge_base/07_frequency_design_theory.md`

---

### 2.3 Llama 长程对照

| 实验 | 内容 | 状态 |
|------|------|------|
| Shape/Theta 控制 | 频谱形状保持 + theta 边界消融 | ✅ |
| Triangle 边界 | 频率边界探测 | ✅ |

**关键文件**：
- `results/llama_shape_theta_min/`
- `results/llama13b_triangle_boundary/`

---

## ⏳ Tier 3：进行中的实验

### 3.1 Sigmoid-RoPE Phase 4（训练时验证）

| 内容 | 状态 |
|------|------|
| 124M 标准频率训练 | ✅ 完成 |
| 124M Sigmoid 频率训练 | ✅ 完成 |
| 50M 多 base 对照 | ✅ 完成 |
| 长程 PPL 评测 | 进行中 |

**关键文件**：
- `sigmoid_rope_experiments/phase4_geo100k_124m_2026-02-22/`
- `sigmoid_rope_experiments/phase4_local_50m_shape_base200k_2026-02-22/`
- `sigmoid_rope_experiments/phase4_local_50m_shape_base300k_2026-02-22/`

---

### 3.2 公平 8B LoRA 实验（新设计）

**问题修复**：旧实验 YaRN/PI 用 `rope_scaling`，Hybrid 用 monkey patch → 不公平。

**新设计**：
- 统一 `inv_freq.copy_()` buffer 覆写
- 4 方法完全同条件（baseline / PI / YaRN / anchored_hybrid）
- NIAH 热力图评测 [4K, 8K, 16K, 32K]

**脚本**：
- `scripts/run_llama8b_fair_suite.py`

**状态**：待运行

---

## ⚠️ Tier 4：已知问题实验（禁止引用）

### 4.1 旧 8B LoRA 实验

| 方法 | train_loss | PPL@16K | 问题 |
|------|-----------|---------|------|
| YaRN (旧) | 1.7248 | 6.0566 | ⚠️ 不公平协议 |
| PI (旧) | 1.9493 | 6.1369 | ⚠️ 不公平协议 |
| Hybrid (旧) | 2.0565 | 11.8753 | ❌ 超参未调优 + 不公平比较 |

**禁止原因**：
1. YaRN/PI 用 HF `rope_scaling`，Hybrid 用自定义 monkey patch
2. Hybrid 超参未调优
3. 无 rigid core 高频保护

**详情**：`knowledge_base/08_8b_experiment_analysis.md`

---

### 4.2 Qwen Hybrid-LoRA

| 实验 | 状态 | 问题 |
|------|------|------|
| Qwen Hybrid-LoRA 训练 | ⚠️ | 同样存在公平性问题 |

**存档文件**：
- `results/qwen_hybrid_lora/`
- `results/qwen_plugandplay_wikitext_v1/`

---

## ❌ 废弃实验

### Zero-shot 直接替换频率

**结论**：大概率崩溃，需要训练适应。

**教训**：频谱形状改造必须配合训练。

---

## 📁 关键文件索引

### 论文可引用

| 内容 | 路径 |
|------|------|
| 从零训练证据 | `results/advisor_package_2026-02-15/01_scaling_from_scratch/` |
| Llama 长程对照 | `results/advisor_package_2026-02-15/02_llama_long_context/` |
| Phase Collision 机理 | `knowledge_base/06_phase_collision_D_analysis.md` |
| 频率设计理论 | `knowledge_base/07_frequency_design_theory.md` |

### 项目知识库

| 内容 | 路径 |
|------|------|
| 项目总览 | `knowledge_base/00_项目与结论总览.md` |
| 核心数据 | `knowledge_base/01_已完成实验核心数据.md` |
| 论文故事线 | `knowledge_base/02_论文故事线与主张.md` |
| 负结果复盘 | `knowledge_base/03_负结果与风险复盘.md` |
| 实验环境 | `knowledge_base/04_实验环境与约束.md` |
| 下一步计划 | `knowledge_base/05_下一步执行计划.md` |

### 脚本位置

| 内容 | 路径 |
|------|------|
| 所有运行脚本 | `scripts/` |
| A100 实验脚本 | `artifacts/a100_2026-02-13/scripts/` |
| Sigmoid 实验 | `sigmoid_rope_experiments/` |

---

## 🔬 理论验证实验（2026-02-22）

### 7.1 理论公式验证 [✅ Tier 2]

| 实验 | 结论 | 状态 |
|------|------|------|
| EDIAG 精确 vs 仿射近似 | R²=0.995, MAE=0.006 | ✅ 理论正确 |
| 相变点扫描 (L/b=1.6~1000) | L/b>100 时出现交叉点 | ✅ 理论预测正确 |

**关键文件**：`results/theory_2026-02-22/`

### 7.2 Geo100k 124M 训练 [⏳ Tier 3]

| 指标 | 值 |
|------|-----|
| 训练时间 | 0.87 小时 |
| 最佳 step | 1900 |
| PPL@16K | **1.052** |
| PPL@32K | **1.233** |

**问题**：标准频率表现过好（PPL接近1），可能是数据或配置问题，需要进一步检查。

**关键文件**：`sigmoid_rope_experiments/phase4_geo100k_124m_2026-02-22_rerun1/`

### 7.3 50M Base 消融 [⚠️ Tier 4 / 禁止引用]

| 配置 | Standard PPL@16K | Sigmoid PPL@16K | 改善 |
|------|----------------:|---------------:|-----:|
| base=200k | 1.214 | **1.177** | 3.1% |
| base=300k | **1.035** | 1.129 | ❌ -9.1% |

**结论状态**：**禁止引用（负结果）**
**问题**：base=300k 时 Standard 更好，说明 base 选择对结果有重大影响。这是典型的局部失效案例，证明 Sigmoid 频率在所有 base 下并非总是优于 Standard。需要系统性消融。

**关键文件**：`sigmoid_rope_experiments/phase4_local_50m_shape_base*/`

### 7.4 注意力距离分布分析 [✅ Tier 2]

| 层 | γ (幂律指数) | R² |
|----|------------|-----|
| L0 | 0.81 | 0.975 |
| L2 | **1.31** | 0.958 |
| L5 | 0.53 | 0.961 |
| L11 | 0.61 | 0.936 |
| **平均** | **0.72** | **0.950** |

**结论**：浅层 γ 较高（偏向局部），深层 γ 较低（偏向全局），验证了理论预测。

**关键文件**：`results/attention_distribution/`

### 7.5 探针与除错验证 [✅ Tier 2]

| 验证项 | 现象 | 教训与修复 |
|--------|------|------------|
| **Sigmoid 刻度溢出** | `positions * 100` 导致激活值全部饱和归零，所有高频均退化为 1e-6 | 纯工程算术溢出导致模型彻底崩溃，必须加入数值边界预检单元测试 |
| **隐性注入失效** | `.copy_` 覆盖缓冲层未在 Attention 前向动态生效，损失看似正常实则随机 | 必须使用动态 `LOGIT_DIFF > 1e-4` 探针校验注入是否在计算图内真实生效 |
| **中频信息饿死** | 极端死守边界（Anchored-20）导致中频稀释，NIAH 评测完全失败 | 揭示了信息水床效应（Fisher Info. Waterbed）必须保持跨频段相对平滑 |

**关键文件**：
- `scripts/debug_sigmoid_rope.py`
- `archives/2026-02-22/scripts/_test_inject.py`
- `scripts/run_passkey_sanity_check.py`

---

## 🗂️ 其他实验（需审核）[⏳ Tier 3]

| 目录 | 状态 | 问题 |
|------|------|------|
| `phase_collision_comparison/` | ⏳ | 待审核 |
| `phase_collision_comparison_v2/` | ⏳ | 待审核 |
| `phase_transition/` | ⏳ | 待审核 |
| `phase4_passkey_sanity/` | ⏳ | 待审核 |
| `frequency_range_analysis/` | ⏳ | 待审核 |
| `optimal_base_search/` | ⏳ | 待审核 |
| `comprehensive_theta/` | ⏳ | 待审核 |
| `night_run_*/` | ⏳ | 待审核 |
| `theoretical_validation/` | ⏳ | 待审核 |
| `theory_validation/` | ⏳ | 待审核 |

---

## 🎯 当前可信结论

1. ✅ 频谱形状设计在长程外推上有效（从零训练已证）
2. ✅ 频谱形状是独立于 theta 放大的关键变量
3. ✅ 高频保护是维持指令能力的必要条件
4. ✅ 注意力距离分布近似幂律 D(Δ) ∝ Δ^(-γ)，γ ≈ 0.72（R²=0.95）
5. ✅ 理论相变点预测正确（L/b>100 时 hybrid 优于 geo）
6. ⚠️ Base 选择对结果有重大影响，需系统性消融

## 🚫 暂不宜宣传

1. "所有协议下都优于 YaRN/PI"
2. "无需训练可直接替换频率"
3. "单个 PPL 指标改善即代表方法有效"
4. "Sigmoid 频率在所有 base 下都优于 Standard"（反例：base=300k）

---

*文档整理：2026-02-22*
