# 500M 从零训练实验计划

## 目标

在 500M 规模上验证 EVQ τ=1.5 vs geometric (τ=0) 的从零训练优势，
为 NeurIPS 2026 论文提供 scaling evidence（50M → 125M → 500M）。

## 对标论文：DAPE (NeurIPS 2024)

最接近的中稿先例。以下是我们的实验设计与 DAPE 的逐项对比。

| 维度 | DAPE (NeurIPS 2024) | 我们 | 差距/状态 |
|------|-------|------|----------|
| 模型规模 | 125M, 350M (GPT-NeoX) | 50M, 125M, **521M** | **OK** — 我们覆盖更宽的规模范围 |
| 架构 | GPT-NeoX (decoder-only) | GPT (decoder-only, RoPE, SwiGLU) | **OK** — 标准架构 |
| 训练步数 | 50K steps | ~12K steps (500M tier) | **注意** — 步数少但 tokens/step 不同 |
| 训练 tokens | ~6.5B (batch=256 x seq=512 x 50K) | 200M (500M tier) | **GAP** — 我们少 32x (见下方分析) |
| 训练 seq_len | 128, 512, 1024 | 2048 | **OK** — 我们更长 |
| 训练数据 | Pile-ArXiv, Pile-Books3 | TinyStories | **GAP** — 数据质量和多样性不足 |
| 评估长度 | 512-8192 | 2048-16384 | **OK** — 我们外推更远 |
| PPL 计算 | **最后 256 tokens** | 全序列 | **需改进** — 应增加 last-N 模式 |
| Seeds | **3 seeds + std + p-value** | 1 seed (计划 2) | **GAP** — 至少需要 2, 推荐 3 |
| Passkey | CHE (合成形式语言) | Passkey retrieval | **OK** — 互补的合成任务 |
| Baselines | 9 个 PE 方法 | 仅 geometric vs EVQ | **可接受** — 论文定位为理论验证而非 SOTA |
| Init | 标准 | **depth-scaled** (已修复) | **OK** |
| LR schedule | Adam, 固定 lr | Cosine + warmup + **min_lr floor** (已修复) | **OK** |
| 可复现性 | 3 seeds | **全 seed 控制** (已修复) | **OK** |

### 关键差距分析

**1. 训练数据 (最大风险)**

DAPE 使用 Pile-ArXiv 和 Pile-Books3 — 真实学术/文学文本。
我们使用 TinyStories — 合成儿童故事，词汇和结构单一。

**影响**: 审稿人可能质疑：「在简单数据上的优势能否泛化到真实文本？」

**应对**:
- 论文中明确声明 TinyStories 是控制变量实验的 testbed
- 强调 EVQ 优势是位置编码层面的，不依赖语料分布
- 如时间允许，在 Pile-Books3 子集上重跑 125M 作为 robustness check

**2. 训练 tokens 不足**

DAPE 用 ~6.5B tokens 训练 125M 模型（52 tokens/param）。
我们用 200M tokens 训练 521M 模型（0.4 tokens/param）。

**影响**: 模型严重欠训练，可能低估 EVQ 优势（或放大噪声）。

**应对**:
- 所有 τ 值使用完全相同的数据和步数 → 相对比较仍然公平
- 论文中报告 tokens/param 比率
- 如 PPL 结果噪声大，考虑增加到 500M tokens（多 ~6h）

**3. 种子数不足**

DAPE 用 3 seeds + std + p-value。我们计划 1-2 seeds。

**应对**:
- **最低要求**: seed=42 + seed=137（2 seeds）
- 报告均值和范围而非单点
- 如跨 seed 方向一致（125M 已验证），可辩护

## 实验设计

| 参数 | 值 |
|------|-----|
| 模型 | GPT (自定义, ~521M params) |
| hidden_size | 1024 |
| num_layers | 28 |
| num_heads | 16, head_dim=64 |
| intermediate_size | 4096 |
| init | std=0.02, residual proj scaled by 1/√(2×layers) |
| 训练数据 | TinyStories (streaming via hf-mirror) |
| train_tokens | 200M |
| seq_len | 2048 |
| lr | 1.5e-4, cosine schedule, min_lr=1.5e-5 |
| warmup | 2% of total steps |
| batch_size | 16 (96GB GPU auto-adjust) |
| optimizer | AdamW (β₁=0.9, β₂=0.95, wd=0.1) |
| grad_clip | 1.0 |
| τ 值 | 0.0 (geometric baseline), 1.5 (EVQ) |
| seed | 42 (P0), 137 (P1) |

## 评估

### PPL (内置在 sweep 脚本中)
- 长度: 2048, 4096, 8192, 16384
- 10 chunks per length, **随机偏移** (已修复)
- 固定 eval_seed=9999 保证跨 run 可比

### Passkey Retrieval (eval_passkey.py)
- 上下文长度: 1024, 2048, 4096, 8192, 16384
- 深度比: 0.1, 0.3, 0.5, 0.7, 0.9
- **每组 50 trials** (95% CI ≈ ±14% at 50% acc)
- 测量: accuracy, SE, 95% CI, mean_rank, mean_prob
- **Filler 文本已过滤** number-word 污染

## 代码修复记录

| 问题 | 严重性 | 修复 |
|------|--------|------|
| 无 depth-scaled init | CRITICAL | residual proj 按 1/√(2n) 缩放 |
| LR 衰减到 0 | CRITICAL | 增加 min_lr = lr × 0.1 |
| 只设 torch.manual_seed | HIGH | 增加 cuda/numpy/random 全 seed |
| PPL eval 固定取首段 | HIGH | 改为随机偏移采样 |
| Filler 含 number words | HIGH | 增加 sanitization 过滤 |
| 仅 25 trials | HIGH | 增加到 50, 增加 SE 和 95% CI |
| shell 管道静默失败 | HIGH | 增加 set -o pipefail |
| cos/sin 缓存不刷新 | MEDIUM | load_state_dict 后 rebuild |
| 无置信区间 | MEDIUM | 增加 Wilson SE + CI |
| u_k = k/K 与论文不一致 | MEDIUM | 对齐为 (k+0.5)/K 中点量化 |

### 已知限制 (需在论文中声明)

1. **u_k = (k+0.5)/K 中点量化**: ✅ 已对齐论文公式 (9)。τ=0 时为半步偏移的
   geometric RoPE (base^{-(k+0.5)/K})，偏差可忽略 (K=32)，理论上更优 (Prop G.5)。
2. **训练 tokens 远低于 Chinchilla 最优**: 200M/521M = 0.38 tokens/param，
   论文需注明这是 controlled comparison 设定。
3. **TinyStories 非标准语料**: 结果是否泛化需要额外实验验证。
4. **Tier 名称 vs 实际参数**: "500m" 实际 521M, "350m" 实际 454M。
   论文中使用实际参数计数。

## 代码文件

| 文件 | 说明 |
|------|------|
| `scripts/m4_evq_sweep/run_evq_sweep.py` | 训练+PPL eval (已添加 500m tier + 全部修复) |
| `scripts/m4_evq_sweep/eval_passkey.py` | Passkey 评测 (新建, 含 filler sanitization) |
| `scripts/m4_evq_sweep/run_500m_server.sh` | 服务器一键执行脚本 |

## 服务器执行

```bash
# 0. 确认依赖 (首次需要)
pip install datasets transformers torch

# 1. 开机后 sync 代码
cd /root/autodl-tmp/dfrope/hybrid-rope
git pull  # 或手动 rsync

# 2. 在 screen 中运行
screen -S exp500m
bash scripts/m4_evq_sweep/run_500m_server.sh 2>&1 | tee ~/500m_experiment.log

# 3. Ctrl+A D 后台运行, 关机前 screen -r exp500m 查看状态
```

### 依赖
- `torch` (已安装, 2.8.0)
- `transformers` (已安装, 5.1.0)
- `datasets` (需确认, 用于 streaming TinyStories)
- `numpy`
- tokenizer: `EleutherAI/gpt-neox-20b` (首次运行自动下载, 通过 hf-mirror.com)

## 预期结果

基于 50M/125M 的 scaling trend:

| 规模 | Δ PPL@16K (seed=42) | 预期 500M |
|------|:-------------------:|:---------:|
| 50M  | -10.9%              | — |
| 125M | -18.9%              | — |
| 500M | ?                   | **-25% ~ -30%** (外推) |

Passkey 预期: EVQ 在 >2048 的外推长度上保持更高检索准确率。

## 成本估算

| 阶段 | 耗时 | 费用 (6元/h) |
|------|------|-------------|
| 训练 τ=0.0, seed=42 | ~6h | 36 元 |
| 训练 τ=1.5, seed=42 | ~6h | 36 元 |
| Passkey eval | ~2h | 12 元 |
| **P0 总计** | **~14h** | **~84 元** |
| [P1] seed=137 x 2 runs | ~12h | 72 元 |
| [P1] 50M+125M 重跑 | ~3h | 18 元 |
| **含 P1 总计** | **~29h** | **~174 元** |

## 后续优先级

1. **[P0]** 500M seed=42: τ=0.0 + τ=1.5 + passkey
2. **[P1]** 500M seed=137: 双种子验证
3. **[P1]** 同一服务器重跑 50M+125M (环境一致性)
4. **[P2]** τ 网格: 500M 上测 τ=0.3,0.5,0.8,1.0
5. **[P2]** 在 Pile-Books3 子集上重跑 125M (数据泛化性)
