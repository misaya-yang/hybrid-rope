# Claude Code 任务提示词：审核 Learnable τ 设计并准备 100M 实验

---

## 你的角色

你是一个严格的 AI 工程审核员 + 实验设计师。你需要：
1. 审核已有代码和设计文档的正确性
2. 发现问题就修，没问题就设计 100M 模型的训练实验

## 项目背景

这是一个 NeurIPS 2026 论文项目：**"RoPE Scaling as a Variational Inverse Problem"**。

核心创新：用变分优化推导出 RoPE 频率的最优参数化形式 EVQ-Cosh，由单参数 τ 控制。**τ 现在被设计为可学习参数**，在训练中自动收敛到数据最优值。

## 关键文件位置

### 必读文件（审核优先级从高到低）

1. **`rope/learnable_evq.py`** — Learnable EVQ-Cosh 的 PyTorch 实现
   - `LearnableEVQRoPE` 类：核心 nn.Module
   - `EVQRoPEWrapper`：transformer drop-in wrapper
   - `estimate_tau_from_distance_prior()`：Algorithm 1 (D(Δ) → τ*)
   - `measure_distance_distribution()`：经验距离分布测量
   - `TauLogger`：τ 轨迹记录
   - `setup_optimizer_with_tau()`：双学习率优化器设置
   - 底部有 `__main__` 验证代码，请运行

2. **`docs/paperdraft/LEARNABLE_TAU_DESIGN.md`** — 完整设计方案
   - §1 数学基础（梯度解析式、边界锚定）
   - §2 参数化选择（softplus, 初始化策略）
   - §3 训练配方（LR multiplier, warmup）
   - §4 实验设计（三阶段：校准、对照、跨数据集）
   - §5 图表设计
   - §6 论文叙事

3. **`docs/paperdraft/THEORY_IRONCLAD.md`** — 理论参考（不要修改）
   - 三大定理的完整陈述和证明
   - 推导链条
   - 审稿防御要点

4. **`rope/schedules.py`** — 现有固定 EVQ 实现（第 180-190 行）
   - 注意：代码用 `u = idx / float(n)` 但论文用 `u_k = (k+0.5)/N`
   - learnable_evq.py 已经对齐为中点量化

5. **`docs/paperdraft/FINAL_ACTION_PLAN.md`** — 之前的行动方案
   - 注意：learnable τ 的策略已从 "Appendix only" 升级为 "核心贡献"

### 项目根目录结构

```
hybrid-rope/
├── rope/
│   ├── schedules.py         # 固定频率调度
│   ├── learnable_evq.py     # 【新】可学习 τ 实现
│   └── ...
├── docs/paperdraft/
│   ├── LEARNABLE_TAU_DESIGN.md    # 【新】设计文档
│   ├── THEORY_IRONCLAD.md          # 理论参考（只读）
│   ├── FINAL_ACTION_PLAN.md        # 行动方案（需更新）
│   └── ...
├── experiments/              # 实验脚本（需要新增）
├── submission/               # 投稿代码（需要更新）
└── ...
```

## 审核清单

请逐项检查以下内容：

### A. 数学正确性
- [ ] φ_k(τ) 公式是否与论文一致（注意 arcsinh 而非 arcsin）
- [ ] 梯度 ∂φ_k/∂τ 的解析式是否正确（可以用 `torch.autograd.gradcheck` 验证）
- [ ] τ → 0 极限是否真的恢复 geometric RoPE
- [ ] 边界锚定：φ_0 ≈ 0, φ_{N-1} ≈ 1 是否成立
- [ ] softplus 参数化的 inverse 是否正确

### B. 代码质量
- [ ] `LearnableEVQRoPE` 能否正确前向传播并反向传播
- [ ] `estimate_tau_from_distance_prior` 的核矩阵构造是否正确
- [ ] 两步拟合法（非对角→β, 对角→α）是否稳健
- [ ] 数值稳定性：τ 非常小（<1e-6）和非常大（>10）时是否安全
- [ ] dtype：频率计算是否使用 float64（精度关键）

### C. 训练配方合理性
- [ ] τ_lr_multiplier = 10× 是否合理（可能需要实验调整）
- [ ] 不做 warmup 是否安全（边界锚定是否足够保证稳定性）
- [ ] 不做 weight decay 是否正确（τ 只有 1 个参数）

### D. Algorithm 1 (D→τ) 验证
- [ ] 在合成数据上测试：均匀分布 → τ ≈ 0, power-law → τ > 0
- [ ] 拟合残差是否能正确反映 broadband 近似质量

## 审核完成后：设计 100M 实验

### 模型规格（100M GPT-2 style）
- Layers: 12
- Hidden: 768
- Heads: 12
- Head dim: 64 → N = 32 frequencies
- Vocab: 50257 (GPT-2 tokenizer) 或 FineWeb-Edu 对应的 tokenizer
- Context: 训练 2K/4K, 测试外推到 8K/16K

### 数据集
- **FineWeb-Edu** (优先, 数据质量高, 距离分布更复杂)
- TinyStories 作为对照（如果时间够）

### 实验矩阵（100M, FineWeb-Edu）

| Run | Method | 配置 | GPU 时间估算 |
|-----|--------|------|-------------|
| 1 | Geometric baseline | 标准 RoPE | ~1.5 hr |
| 2 | Fixed EVQ τ=0.5 | 固定 | ~1.5 hr |
| 3 | Fixed EVQ τ=1.0 | 固定 | ~1.5 hr |
| 4 | Fixed EVQ τ=1.5 | 固定 | ~1.5 hr |
| 5 | Fixed EVQ τ=2.0 | 固定 | ~1.5 hr |
| 6 | **Learnable EVQ** τ_init=1.0, lr_mult=10 | 学习 | ~1.5 hr |
| 7 | **Learnable EVQ** τ_init=0.01, lr_mult=10 | 从 geometric 开始学习 | ~1.5 hr |

总计约 10.5 小时。

### 训练超参数（参考已有 50M/125M 实验）

请检查项目中已有的训练脚本和配置文件，确保 100M 实验与已有实验保持一致的：
- Optimizer (AdamW)
- LR schedule (cosine with warmup)
- Batch size
- Training steps/tokens
- Evaluation protocol (PPL @ 2K/4K/8K/16K)

### 产出文件

1. `experiments/train_100m_learnable.py` — 训练脚本
2. `experiments/configs/100m_fineweb.yaml` — 配置文件
3. `experiments/plot_tau_trajectory.py` — τ 轨迹可视化
4. `experiments/eval_context_extrapolation.py` — 外推评估脚本

### 成功标准

1. τ 在训练结束前收敛（last 20% std < 0.05）
2. Learnable EVQ 的 PPL 在最优 fixed EVQ 的 ±1% 以内
3. Learnable EVQ PPL < Geometric PPL（至少 -3% @16K）
4. τ_learned 值在合理范围 (0.3 - 3.0)

## 重要提醒

1. **理论不要动**: `THEORY_IRONCLAD.md` 是只读参考，理论已经被验证正确
2. **u_k 对齐**: 新代码用 `(k+0.5)/N`，旧代码 `schedules.py` 用 `k/N`。100M 实验统一用中点量化
3. **论文叙事已变**: τ 从 "future work" 升级为核心贡献。learnable τ 不是稀释理论，而是理论的实践化
4. **先证 100M 有用**: 100M 成功后再考虑扩展到更大规模
5. **τ 轨迹图是关键**: 每次实验都要记录 τ 的训练轨迹，这会是论文的核心 Figure

---

*生成日期: 2026-03-01*
*由 Cowork Claude Opus 4.6 生成*
